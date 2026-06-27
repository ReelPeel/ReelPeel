"""Video transcript statement extraction plus topic guideline classification."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional
import os
from pipeline.RAG_vdb.query_guideline_vdb import DEFAULT_EMBED_MODEL
from pipeline.core.models import PipelineState
from pipeline.core.orchestrator import PipelineOrchestrator, config_uses_pubmed
from pipeline.test_configs.preprompts import PROMPT_TMPL_S2

from evaluation.topic_guideline import (
    TOPIC_PROFILES,
    TopicProfile,
    _serialize_evidence,
    _guideline_columns_for_item,
    _serialize_guideline_documents,
    atomic_write_json,
    label_to_code,
    file_sha256,
    validate_vdb,
)

DEFAULT_TRANSCRIPTS_ROOT = Path("/data/home/jak38842/disk/fact_checker/videos")
MAX_EXTRACTED_STATEMENTS = 20

VIDEO_TOPIC_TRANSCRIPTS: Mapping[str, str] = MappingProxyType(
    {
        "beikost": "downloads_beikost.json",
        "vitamin-d": "downloads_vitamind.json",
    }
)

VIDEO_EXTRACTION_PROMPT = PROMPT_TMPL_S2.replace(
    "up to a maximum of **5**. If more than 8 are present, return the **8 most clinically important and/or potentially harmful**",
    "up to a maximum of **20**. If more than 20 are present, return the **20 most clinically important and/or potentially harmful**",
).replace(
    "A valid JSON array of 1–5 strings.",
    "A valid JSON array of 0-20 strings.",
)


@dataclass(frozen=True)
class VideoTranscriptRecord:
    topic: str
    video_file: str
    language: str
    transcript: str
    source_index: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def transcript_path_for_topic(topic: str, transcripts_root: Path = DEFAULT_TRANSCRIPTS_ROOT) -> Path:
    try:
        filename = VIDEO_TOPIC_TRANSCRIPTS[topic]
    except KeyError as exc:
        expected = ", ".join(sorted(VIDEO_TOPIC_TRANSCRIPTS))
        raise ValueError(f"Unsupported video topic {topic!r}; expected one of: {expected}") from exc
    return Path(transcripts_root).expanduser().resolve() / filename


def load_video_transcripts(topic: str, transcripts_root: Path = DEFAULT_TRANSCRIPTS_ROOT) -> List[VideoTranscriptRecord]:
    path = transcript_path_for_topic(topic, transcripts_root)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Transcript file must contain an 'items' list: {path}")

    records: List[VideoTranscriptRecord] = []
    seen_files: set[str] = set()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Transcript item {index} in {path} must be an object.")
        video_file = str(item.get("file") or "").strip()
        if not video_file:
            raise ValueError(f"Transcript item {index} in {path} is missing 'file'.")
        if video_file in seen_files:
            raise ValueError(f"Duplicate transcript video file {video_file!r} in {path}.")
        seen_files.add(video_file)
        records.append(
            VideoTranscriptRecord(
                topic=topic,
                video_file=video_file,
                language=str(item.get("language") or "").strip(),
                transcript=str(item.get("text") or "").strip(),
                source_index=index,
            )
        )
    return records


def build_video_pipeline_config(
    profile: TopicProfile,
    *,
    model: str,
    top_k: int,
    min_score: float,
    llm_settings: Optional[Dict[str, str]] = None,
    max_statements: int = MAX_EXTRACTED_STATEMENTS,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "name": f"Video_Topic_Guideline_{profile.key}",
        "debug": False,
        "steps": [
            {
                "type": "mock_transcript",
                "settings": {"transcript_text": "PLACEHOLDER"},
            },
            {
                "type": "extraction",
                "settings": {
                    "model": model,
                    "prompt_template": VIDEO_EXTRACTION_PROMPT,
                    "temperature": 0.0,
                    "max_retries": 1,
                    "max_statements": int(max_statements),
                },
            },
            {
                "type": "translate_claim",
                "settings": {
                    "model": model,
                    "temperature": 0.0,
                    "max_retries": 1,
                },
            },
            {
                "type": "normalize_topic_claim",
                "settings": {"topic": profile.key},
            },
            {
                "type": "retrieve_guideline_facts",
                "settings": {
                    "db_path": str(profile.vdb_path.resolve()),
                    "top_k": int(top_k),
                    "min_score": float(min_score),
                    "dense_k": max(int(top_k) * 5, int(top_k)),
                    "bm25_k": max(int(top_k) * 5, int(top_k)),
                    "rrf_k": 60,
                    "claim_mode": True,
                    "include_parent_context": True,
                    "embed_model": DEFAULT_EMBED_MODEL,
                },
            },
            {
                "type": "classify_guideline",
                "settings": {
                    "topic": profile.key,
                    "allowed_labels": list(profile.labels),
                    "label_definitions": dict(profile.label_definitions),
                    "model": model,
                    "temperature": 0.0,
                    "max_retries": 1,
                },
            },
        ],
    }
    if llm_settings:
        config["llm_settings"] = dict(llm_settings)
        for step in config["steps"]:
            step.setdefault("settings", {})["llm_settings"] = dict(llm_settings)
    if config_uses_pubmed(config):
        raise ValueError("Video topic guideline configuration must not contain PubMed steps.")
    return config


def _set_transcript(config: Dict[str, Any], record: VideoTranscriptRecord) -> None:
    config["steps"][0]["settings"]["transcript_text"] = record.transcript


BASE_VIDEO_CSV_COLUMNS = (
    "topic",
    "video_file",
    "video_statement_id",
    "language",
    "statement",
    "predicted_label",
    "predicted_label_text",
    "retrieved_chunk_count",
    "classification_status",
    "retrieval_status",
)


def video_csv_columns(max_guidelines: int) -> List[str]:
    columns = list(BASE_VIDEO_CSV_COLUMNS)
    for index in range(1, max_guidelines + 1):
        columns.extend(
            [
                f"guideline_{index}_title",
                f"guideline_{index}_label",
                f"guideline_{index}_label_text",
                f"guideline_{index}_cited_chunks",
                f"guideline_{index}_pages",
            ]
        )
    columns.extend(["status", "error"])
    return columns


def write_video_results_csv(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    item_list = list(items)
    max_guidelines = max((len(item.get("guideline_documents", []) or []) for item in item_list), default=0)
    columns = video_csv_columns(max_guidelines)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        rows = sorted(
            item_list,
            key=lambda item: (str(item.get("video_file", "")), int(item.get("video_statement_id") or 0)),
        )
        for item in rows:
            label_text = item.get("predicted_label")
            row = {column: item.get(column, "") for column in BASE_VIDEO_CSV_COLUMNS}
            row.update(
                {
                    "predicted_label": label_to_code(label_text),
                    "predicted_label_text": label_text or "",
                    "retrieved_chunk_count": item.get(
                        "usable_retrieved_chunk_count", item.get("retrieved_chunk_count", 0)
                    ),
                    "status": item.get("status", ""),
                    "error": item.get("error", ""),
                }
            )
            row.update(_guideline_columns_for_item(item, max_guidelines))
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


CSV_COLUMNS = tuple(video_csv_columns(0))


def _empty_video_item(record: VideoTranscriptRecord, error: str) -> Dict[str, Any]:
    return {
        "topic": record.topic,
        "video_file": record.video_file,
        "video_statement_id": 0,
        "language": record.language,
        "statement": "",
        "predicted_label": None,
        "translated_text_de": None,
        "translated_text_en": None,
        "normalized_text": None,
        "claim_type": None,
        "routing_reason": None,
        "retrieval_status": None,
        "classification_status": None,
        "failure_stage": "input",
        "fallback_label_used": False,
        "retrieval_queries": [],
        "cited_chunk_ids": [],
        "evidence": [],
        "guideline_documents": [],
        "retrieved_chunk_count": 0,
        "usable_retrieved_chunk_count": 0,
        "raw_retrieved_chunk_count": 0,
        "status": "error",
        "error": error,
        "pipeline_seconds": 0.0,
    }


def _statement_item(record: VideoTranscriptRecord, statement, *, pipeline_seconds: float, vdb: Dict[str, Any]) -> Dict[str, Any]:
    evidence = _serialize_evidence(statement)
    usable_count = int(statement.usable_retrieved_chunk_count or len(evidence))
    raw_count = int(statement.raw_retrieved_chunk_count or usable_count)
    return {
        "topic": record.topic,
        "video_file": record.video_file,
        "video_statement_id": int(statement.id),
        "language": record.language,
        "statement": statement.text,
        "predicted_label": statement.guideline_label,
        "translated_text_de": statement.translated_text_de,
        "translated_text_en": statement.translated_text_en,
        "normalized_text": statement.normalized_text,
        "claim_type": statement.claim_type,
        "routing_reason": statement.routing_reason,
        "retrieval_status": statement.retrieval_status,
        "classification_status": statement.classification_status,
        "failure_stage": statement.failure_stage,
        "fallback_label_used": bool(statement.fallback_label_used),
        "retrieval_queries": list(statement.retrieval_queries),
        "cited_chunk_ids": list(statement.cited_chunk_ids),
        "evidence": evidence,
        "guideline_documents": _serialize_guideline_documents(statement),
        "usable_retrieved_chunk_count": usable_count,
        "raw_retrieved_chunk_count": raw_count,
        "retrieved_chunk_count": usable_count,
        "vdb_path": vdb["path"],
        "vdb_sha256": vdb["sha256"],
        "status": "ok" if statement.guideline_label else "error",
        "error": None if statement.guideline_label else "Missing predicted label after pipeline completion.",
        "pipeline_seconds": pipeline_seconds,
    }


def _transcript_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "generated_at": payload.get("generated_at"),
        "model": payload.get("model"),
        "device": payload.get("device"),
        "root_folder": payload.get("root_folder"),
        "count_ok": payload.get("count_ok"),
        "count_failed": payload.get("count_failed"),
    }


def run_video_topic_evaluation(
    profile: TopicProfile,
    *,
    output_root: Path,
    model: str,
    top_k: int = 20,
    min_score: float = 0.25,
    limit_videos: Optional[int] = None,
    transcripts_root: Path = DEFAULT_TRANSCRIPTS_ROOT,
    llm_settings: Optional[Dict[str, str]] = None,
    max_statements: int = MAX_EXTRACTED_STATEMENTS,
) -> Dict[str, Path]:
    records = load_video_transcripts(profile.key, transcripts_root)
    if limit_videos is not None:
        records = records[: int(limit_videos)]
    vdb = validate_vdb(profile)
    transcript_path = transcript_path_for_topic(profile.key, transcripts_root)
    transcript_manifest = _transcript_manifest(transcript_path)
    config = build_video_pipeline_config(
        profile,
        model=model,
        top_k=top_k,
        min_score=min_score,
        llm_settings=llm_settings,
        max_statements=max_statements,
    )

    run_dir = output_root.resolve() / profile.key
    checkpoint_path = run_dir / "checkpoint.json"
    csv_path = run_dir / "predictions.csv"
    manifest_path = run_dir / "manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    fingerprint_payload = {
        "topic": profile.key,
        "transcript_path": transcript_manifest["path"],
        "transcript_sha256": transcript_manifest["sha256"],
        "vdb_sha256": vdb["sha256"],
        "model": model,
        "top_k": top_k,
        "min_score": min_score,
        "max_statements": int(max_statements),
        "labels": list(profile.labels),
    }
    config_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("config_fingerprint") != config_fingerprint:
            raise ValueError(
                f"Existing checkpoint configuration differs for {profile.key}: {checkpoint_path}"
            )
    else:
        checkpoint = {
            "topic": profile.key,
            "config_fingerprint": config_fingerprint,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "completed": False,
            "items": {},
        }

    manifest = {
        **fingerprint_payload,
        "config_fingerprint": config_fingerprint,
        "transcript": transcript_manifest,
        "vdb": vdb,
        "allowed_labels": list(profile.labels),
        "pipeline_steps": [step["type"] for step in config["steps"]],
        "pubmed_enabled": config_uses_pubmed(config),
        "created_at": checkpoint["created_at"],
        "updated_at": utc_now(),
    }
    atomic_write_json(manifest_path, manifest)

    for index, record in enumerate(records, start=1):
        existing_keys = [key for key in checkpoint["items"] if key.startswith(f"{record.video_file}::")]
        if existing_keys and all(checkpoint["items"][key].get("status") == "ok" for key in existing_keys):
            continue
        for key in existing_keys:
            checkpoint["items"].pop(key, None)

        if not record.transcript:
            key = f"{record.video_file}::0"
            checkpoint["items"][key] = _empty_video_item(record, "Empty transcript text.")
            checkpoint["updated_at"] = utc_now()
            atomic_write_json(checkpoint_path, checkpoint)
            write_video_results_csv(csv_path, checkpoint["items"].values())
            print(f"[{profile.key}] {index}/{len(records)} {record.video_file}: error empty transcript")
            continue

        started = time.perf_counter()
        try:
            video_config = copy.deepcopy(config)
            _set_transcript(video_config, record)
            final_state = PipelineOrchestrator(video_config).run(PipelineState())
            pipeline_seconds = time.perf_counter() - started
            if not final_state.statements:
                key = f"{record.video_file}::0"
                checkpoint["items"][key] = _empty_video_item(record, "No statements extracted from transcript.")
            else:
                for statement in final_state.statements[: int(max_statements)]:
                    key = f"{record.video_file}::{int(statement.id)}"
                    checkpoint["items"][key] = _statement_item(
                        record, statement, pipeline_seconds=pipeline_seconds, vdb=vdb
                    )
        except Exception as exc:
            key = f"{record.video_file}::0"
            checkpoint["items"][key] = _empty_video_item(record, f"{type(exc).__name__}: {exc}")

        checkpoint["updated_at"] = utc_now()
        atomic_write_json(checkpoint_path, checkpoint)
        write_video_results_csv(csv_path, checkpoint["items"].values())
        count = len([key for key in checkpoint["items"] if key.startswith(f"{record.video_file}::")])
        print(f"[{profile.key}] {index}/{len(records)} {record.video_file}: {count} row(s)")

    expected_video_files = {record.video_file for record in records}
    completed_files = {
        key.split("::", 1)[0]
        for key, item in checkpoint["items"].items()
        if key.split("::", 1)[0] in expected_video_files and item.get("status") == "ok"
    }
    checkpoint["completed"] = len(completed_files) == len(expected_video_files)
    checkpoint["updated_at"] = utc_now()
    atomic_write_json(checkpoint_path, checkpoint)
    write_video_results_csv(csv_path, checkpoint["items"].values())

    return {
        "run_dir": run_dir,
        "checkpoint": checkpoint_path,
        "csv": csv_path,
        "manifest": manifest_path,
    }


def profiles_for_topics(topic: str) -> List[TopicProfile]:
    selected = ("beikost", "vitamin-d") if topic == "all" else (topic,)
    return [TOPIC_PROFILES[key] for key in selected]
