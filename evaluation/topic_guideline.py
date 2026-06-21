"""Filename-routed, VDB-only guideline classification evaluations."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from pipeline.RAG_vdb.query_guideline_vdb import DEFAULT_EMBED_MODEL
from pipeline.core.models import PipelineState
from pipeline.core.orchestrator import PipelineOrchestrator, config_uses_pubmed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOT_MENTIONED = "wird nicht in Leitlinien genannt"
COMMON_LABELS = (
    "entspricht Leitlinie",
    "widerspricht Leitlinien",
    NOT_MENTIONED,
    "entspricht teilweise Leitlinien",
)


@dataclass(frozen=True)
class TopicProfile:
    key: str
    claims_filename: str
    claims_path: Path
    source_csv_path: Path
    vdb_path: Path
    labels: tuple[str, ...]
    label_definitions: Mapping[str, str]


_COMMON_DEFINITIONS = {
    "entspricht Leitlinie": (
        "The claim fully matches the guideline in meaning. Central aspects and relevant qualifications are present."
    ),
    "widerspricht Leitlinien": (
        "The claim fully or in central aspects directly contradicts the guideline content."
    ),
    NOT_MENTIONED: (
        "The guidelines provide no information on the relevant content aspect, so no link to guideline content can be established."
    ),
    "entspricht teilweise Leitlinien": (
        "Central aspects match the guideline, but the claim is too general, imprecise, omits important details, or combines guideline-consistent parts with unsupported parts."
    ),
}

TOPIC_PROFILES: Mapping[str, TopicProfile] = MappingProxyType(
    {
        "beikost": TopicProfile(
            key="beikost",
            claims_filename="beikost_claims.txt",
            claims_path=PROJECT_ROOT / "evaluation/data_set/Masterarbeiten/beikost_claims.txt",
            source_csv_path=PROJECT_ROOT / "evaluation/data_set/Masterarbeiten/beikost.CSV",
            vdb_path=PROJECT_ROOT / "pipeline/RAG_vdb/beikost_vdb.sqlite",
            labels=COMMON_LABELS + ("keine Beikostempfehlung", "Leitlinien uneinig"),
            label_definitions=MappingProxyType(
                {
                    **_COMMON_DEFINITIONS,
                    "keine Beikostempfehlung": (
                        "The claim is not a complementary-feeding recommendation, but explanatory, descriptive, or outside recommendation scope."
                    ),
                    "Leitlinien uneinig": (
                        "Retrieved guideline sources give directly conflicting recommendations about the claim."
                    ),
                }
            ),
        ),
        "vitamin-d": TopicProfile(
            key="vitamin-d",
            claims_filename="vitadminD_claims.txt",
            claims_path=PROJECT_ROOT / "evaluation/data_set/Masterarbeiten/vitadminD_claims.txt",
            source_csv_path=PROJECT_ROOT / "evaluation/data_set/Masterarbeiten/vitaminD.CSV",
            vdb_path=PROJECT_ROOT / "pipeline/RAG_vdb/vitaminD_vdb.sqlite",
            labels=COMMON_LABELS,
            label_definitions=MappingProxyType(dict(_COMMON_DEFINITIONS)),
        ),
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})")}


def profile_for_claims_file(path: Path) -> TopicProfile:
    filename = Path(path).name
    matches = [profile for profile in TOPIC_PROFILES.values() if profile.claims_filename == filename]
    if len(matches) != 1:
        expected = ", ".join(sorted(p.claims_filename for p in TOPIC_PROFILES.values()))
        raise ValueError(f"Unsupported claims filename {filename!r}; expected one of: {expected}")
    return matches[0]


def _validate_profile_files(profile: TopicProfile) -> None:
    routed = profile_for_claims_file(profile.claims_path)
    if routed.key != profile.key:
        raise ValueError(f"Claims filename/profile mismatch for {profile.key}.")
    canonical = TOPIC_PROFILES[profile.key]
    if profile.claims_path.resolve() != canonical.claims_path.resolve():
        raise ValueError(f"Claims path mismatch for {profile.key}.")
    if profile.vdb_path.resolve() != canonical.vdb_path.resolve():
        raise ValueError(f"VDB path mismatch for {profile.key}.")


def _load_claims(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Claims file must contain a JSON list: {path}")

    claims = []
    seen = set()
    for item in payload:
        claim_id = int(item["id"])
        statement = str(item["statement"]).strip()
        if claim_id in seen:
            raise ValueError(f"Duplicate claim ID {claim_id} in {path}")
        if not statement:
            raise ValueError(f"Empty statement for claim ID {claim_id} in {path}")
        seen.add(claim_id)
        claims.append(
            {
                "claim_id": claim_id,
                "claim": statement,
                "gold_label": item.get("gold_label", item.get("label")),
            }
        )
    return claims


def _load_source_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="cp1252", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))

    enriched = []
    current_video_id = ""
    current_url = ""
    for index, row in enumerate(rows):
        raw_video_id = (row.get("ID") or "").strip()
        raw_url = (row.get("Link") or "").strip()
        if raw_video_id:
            current_video_id = raw_video_id
        if raw_url and raw_url != "#NV":
            current_url = raw_url
        enriched.append(
            {
                "source_row": str(index + 2),
                "video_id": current_video_id,
                "url": current_url,
                "source_claim_number": (row.get("Aussagennummer") or str(index + 1)).strip(),
                "source_claim": (row.get("Aussagen") or "").strip(),
            }
        )
    return enriched


def load_topic_records(profile: TopicProfile) -> List[Dict[str, Any]]:
    _validate_profile_files(profile)
    claims = _load_claims(profile.claims_path)
    source_rows = _load_source_rows(profile.source_csv_path)
    if len(claims) != len(source_rows):
        raise ValueError(
            f"{profile.key} claim/source row mismatch: {len(claims)} != {len(source_rows)}"
        )

    records = []
    for claim, source in zip(claims, source_rows):
        if claim["claim"] != source["source_claim"]:
            raise ValueError(
                f"{profile.key} claim text mismatch at claim ID {claim['claim_id']} "
                f"(source row {source['source_row']})."
            )
        records.append({**claim, **source, "topic": profile.key})
    return records


def validate_vdb(profile: TopicProfile) -> Dict[str, Any]:
    _validate_profile_files(profile)
    actual = profile.vdb_path.resolve()
    if not actual.is_file():
        raise FileNotFoundError(f"Topic VDB not found: {actual}")

    con = sqlite3.connect(str(actual))
    try:
        document_count = int(con.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        chunk_count = int(con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        document_columns = _table_columns(con, "documents")
        if "embed_model" in document_columns and "dim" in document_columns:
            models = [row[0] for row in con.execute("SELECT DISTINCT embed_model FROM documents")]
            dimensions = [int(row[0]) for row in con.execute("SELECT DISTINCT dim FROM documents")]
        else:
            models = [row[0] for row in con.execute("SELECT DISTINCT embed_model FROM embeddings")]
            dimensions = [int(row[0]) for row in con.execute("SELECT DISTINCT dim FROM embeddings")]
    finally:
        con.close()
    if not document_count or not chunk_count:
        raise ValueError(f"Topic VDB is empty: {actual}")
    if len(models) != 1 or len(dimensions) != 1:
        raise ValueError(f"Topic VDB has inconsistent embedding metadata: {actual}")
    return {
        "path": str(actual),
        "sha256": file_sha256(actual),
        "documents": document_count,
        "chunks": chunk_count,
        "embed_model": models[0],
        "dimension": dimensions[0],
    }


def build_pipeline_config(
    profile: TopicProfile,
    *,
    model: str,
    top_k: int,
    min_score: float,
    llm_settings: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    _validate_profile_files(profile)
    config = {
        "name": f"Topic_Guideline_{profile.key}",
        "debug": False,
        "steps": [
            {
                "type": "mock_statements",
                "settings": {"statements": [{"id": 0, "text": "PLACEHOLDER"}]},
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
                "settings": {
                    "topic": profile.key,
                },
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
        config["steps"][-1]["settings"]["llm_settings"] = dict(llm_settings)
    if config_uses_pubmed(config):
        raise ValueError("Topic guideline configuration must not contain PubMed steps.")
    return config


def _set_claim(config: Dict[str, Any], record: Dict[str, Any]) -> None:
    mock = config["steps"][0]["settings"]["statements"]
    mock[:] = [{"id": int(record["claim_id"]), "text": record["claim"]}]


def _serialize_evidence(statement) -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": evidence.chunk_id,
            "score": evidence.score,
            "source_path": evidence.source_path,
            "pages": list(evidence.pages),
            "text": evidence.abstract,
        }
        for evidence in statement.evidence
    ]


CSV_COLUMNS = (
    "topic", "claim_id", "video_id", "url",
    "claim", "gold_label", "predicted_label", "claim_type", "routing_reason",
    "cited_sources", "cited_pages", "retrieved_chunk_count",
    "status", "error",
)


def write_results_csv(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for item in sorted(items, key=lambda value: int(value["claim_id"])):
            evidence = item.get("evidence", [])
            evidence_by_id = {ev["chunk_id"]: ev for ev in evidence}
            cited = item.get("cited_chunk_ids", [])
            cited_evidence = [
                evidence_by_id[chunk_id]
                for chunk_id in cited
                if chunk_id in evidence_by_id
            ]
            writer.writerow(
                {
                    **{column: item.get(column, "") for column in CSV_COLUMNS},
                    "cited_sources": "|".join(
                        dict.fromkeys(evidence_by_id[c]["source_path"] for c in cited if c in evidence_by_id)
                    ),
                    "cited_pages": "|".join(
                        f"{c}:{','.join(map(str, evidence_by_id[c]['pages']))}"
                        for c in cited if c in evidence_by_id
                    ),
                    "retrieved_chunk_count": item.get("usable_retrieved_chunk_count", item.get("retrieved_chunk_count", 0)),
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def compute_gold_metrics(items: Iterable[Dict[str, Any]], labels: Sequence[str]) -> Optional[Dict[str, Any]]:
    pairs = [
        (item.get("gold_label"), item.get("predicted_label"))
        for item in items
        if item.get("gold_label") not in (None, "") and item.get("status") == "ok"
    ]
    if not pairs:
        return None
    unknown = sorted({value for pair in pairs for value in pair if value not in labels})
    if unknown:
        raise ValueError(f"Unknown gold/predicted labels: {unknown}")

    per_label = {}
    for label in labels:
        tp = sum(gold == label and pred == label for gold, pred in pairs)
        fp = sum(gold != label and pred == label for gold, pred in pairs)
        fn = sum(gold == label and pred != label for gold, pred in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "count": len(pairs),
        "accuracy": sum(gold == pred for gold, pred in pairs) / len(pairs),
        "macro_f1": sum(value["f1"] for value in per_label.values()) / len(labels),
        "per_label": per_label,
    }


def run_topic_evaluation(
    profile: TopicProfile,
    *,
    output_root: Path,
    model: str,
    top_k: int = 10,
    min_score: float = 0.25,
    limit: Optional[int] = None,
    llm_settings: Optional[Dict[str, str]] = None,
) -> Dict[str, Optional[Path]]:
    records = load_topic_records(profile)
    if limit is not None:
        records = records[: int(limit)]
    vdb = validate_vdb(profile)
    config = build_pipeline_config(
        profile, model=model, top_k=top_k, min_score=min_score, llm_settings=llm_settings,
    )

    run_dir = output_root.resolve() / profile.key
    checkpoint_path = run_dir / "checkpoint.json"
    csv_path = run_dir / "predictions.csv"
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    fingerprint_payload = {
        "topic": profile.key,
        "claims_path": str(profile.claims_path.resolve()),
        "claims_sha256": file_sha256(profile.claims_path),
        "vdb_sha256": vdb["sha256"],
        "model": model,
        "top_k": top_k,
        "min_score": min_score,
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
        "source_csv_path": str(profile.source_csv_path.resolve()),
        "vdb": vdb,
        "allowed_labels": list(profile.labels),
        "pipeline_steps": [step["type"] for step in config["steps"]],
        "pubmed_enabled": config_uses_pubmed(config),
        "created_at": checkpoint["created_at"],
        "updated_at": utc_now(),
    }
    atomic_write_json(manifest_path, manifest)

    for index, record in enumerate(records, start=1):
        key = str(record["claim_id"])
        existing_item = checkpoint["items"].get(key)
        if existing_item and existing_item.get("status") == "ok":
            continue

        item = {
            **record,
            "predicted_label": None,
            "translated_text_de": None,
            "translated_text_en": None,
            "normalized_text": None,
            "claim_type": None,
            "routing_reason": None,
            "retrieval_status": None,
            "classification_status": None,
            "failure_stage": None,
            "fallback_label_used": False,
            "retrieval_queries": [],
            "cited_chunk_ids": [],
            "evidence": [],
            "retrieved_chunk_count": 0,
            "usable_retrieved_chunk_count": 0,
            "raw_retrieved_chunk_count": 0,
            "vdb_path": vdb["path"],
            "vdb_sha256": vdb["sha256"],
            "status": "error",
            "error": None,
            "pipeline_seconds": None,
        }
        started = time.perf_counter()
        try:
            claim_config = copy.deepcopy(config)
            _set_claim(claim_config, record)
            final_state = PipelineOrchestrator(claim_config).run(PipelineState())
            if len(final_state.statements) != 1:
                raise ValueError("Topic evaluation must return exactly one statement.")
            statement = final_state.statements[0]
            item["predicted_label"] = statement.guideline_label
            item["translated_text_de"] = statement.translated_text_de
            item["translated_text_en"] = statement.translated_text_en
            item["normalized_text"] = statement.normalized_text
            item["claim_type"] = statement.claim_type
            item["routing_reason"] = statement.routing_reason
            item["retrieval_status"] = statement.retrieval_status
            item["classification_status"] = statement.classification_status
            item["failure_stage"] = statement.failure_stage
            item["fallback_label_used"] = bool(statement.fallback_label_used)
            item["retrieval_queries"] = list(statement.retrieval_queries)
            item["cited_chunk_ids"] = list(statement.cited_chunk_ids)
            item["evidence"] = _serialize_evidence(statement)
            item["usable_retrieved_chunk_count"] = int(statement.usable_retrieved_chunk_count or len(item["evidence"]))
            item["raw_retrieved_chunk_count"] = int(statement.raw_retrieved_chunk_count or item["usable_retrieved_chunk_count"])
            item["retrieved_chunk_count"] = item["usable_retrieved_chunk_count"]
            item["status"] = "ok" if item["predicted_label"] else "error"
            item["error"] = None if item["predicted_label"] else "Missing predicted label after pipeline completion."
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        item["pipeline_seconds"] = time.perf_counter() - started

        checkpoint["items"][key] = item
        checkpoint["updated_at"] = utc_now()
        atomic_write_json(checkpoint_path, checkpoint)
        write_results_csv(csv_path, checkpoint["items"].values())
        print(f"[{profile.key}] {index}/{len(records)} claim {record['claim_id']}: {item['status']}")

    checkpoint["completed"] = (
        len(checkpoint["items"]) == len(records)
        and all(item.get("status") == "ok" for item in checkpoint["items"].values())
    )
    checkpoint["updated_at"] = utc_now()
    atomic_write_json(checkpoint_path, checkpoint)
    write_results_csv(csv_path, checkpoint["items"].values())

    metrics = compute_gold_metrics(checkpoint["items"].values(), profile.labels)
    if metrics is not None:
        atomic_write_json(metrics_path, metrics)

    return {
        "run_dir": run_dir,
        "checkpoint": checkpoint_path,
        "csv": csv_path,
        "manifest": manifest_path,
        "metrics": metrics_path if metrics is not None else None,
    }
