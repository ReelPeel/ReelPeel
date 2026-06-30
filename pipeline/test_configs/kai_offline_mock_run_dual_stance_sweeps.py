from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.core.models import PipelineState
from pipeline.core.orchestrator import PipelineOrchestrator
from pipeline.test_configs.video_transcription_config import VIDEO_URL_PIPELINE_CONFIG


OFFLINE_ROOT = REPO_ROOT / "offline_mock"
OLD_SWEEP_ROOT = OFFLINE_ROOT / "sweep_20260626_072546"
COMPARISON_PATH = OFFLINE_ROOT / "dual_stance_sweep_comparison.json"
SUMMARY_ENDPOINT = "http://127.0.0.1:6006/evidence_summary"

VIDEOS = [
    {
        "reel_id": "DT0UIgzDZ79",
        "reel_url": "https://www.instagram.com/reels/DT0UIgzDZ79/",
        "audio_path": REPO_ROOT / "audios" / "DT0UIgzDZ79.wav",
    },
    {
        "reel_id": "DT0UbkjDZZj",
        "reel_url": "https://www.instagram.com/reels/DT0UbkjDZZj/",
        "audio_path": REPO_ROOT / "audios" / "DT0UbkjDZZj.wav",
    },
]

SWEEPS = [
    {
        "name": "Section-aware",
        "stance_mode": "section_aware",
        "root": OFFLINE_ROOT / "sweep_section_aware_stance",
        "section_chunking_enabled": True,
    },
    {
        "name": "Fixed-token",
        "stance_mode": "fixed_token",
        "root": OFFLINE_ROOT / "sweep_fixed_token_stance",
        "section_chunking_enabled": False,
    },
]


def _post_json(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _iter_step_defs(steps: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for step in steps:
        yield step
        settings = step.get("settings") or {}
        nested = settings.get("steps")
        if isinstance(nested, list):
            yield from _iter_step_defs(nested)


def _build_audio_pipeline_config(
    audio_path: Path,
    stance_mode: str,
    section_chunking_enabled: bool,
    whisper_model: Optional[str],
    run_id: str,
) -> Dict[str, Any]:
    config = copy.deepcopy(VIDEO_URL_PIPELINE_CONFIG)
    config["name"] = f"Offline_{stance_mode}_Audio_Sweep"
    config["run_id"] = run_id

    config["steps"] = [
        step
        for step in config.get("steps", [])
        if step.get("type") not in {"download_reel", "video_to_audio"}
    ]

    found_audio = False
    found_stance = False
    for step in _iter_step_defs(config.get("steps", [])):
        step_type = step.get("type")
        settings = step.setdefault("settings", {})
        if step_type == "audio_to_transcript":
            settings["audio_path"] = str(audio_path)
            if whisper_model:
                settings["whisper_model"] = whisper_model
            found_audio = True
        elif step_type == "stance_evidence":
            settings["section_chunking_enabled"] = bool(section_chunking_enabled)
            found_stance = True

    if not found_audio:
        raise RuntimeError("Configured pipeline is missing audio_to_transcript")
    if not found_stance:
        raise RuntimeError("Configured pipeline is missing stance_evidence")
    return config


def _summary_filename(statement_index: int, evidence_index: int, pubmed_id: Optional[str]) -> str:
    source = str(evidence_index + 1).zfill(2)
    statement = str(statement_index + 1).zfill(2)
    suffix = str(pubmed_id or f"source-{source}").replace("/", "_")
    return f"statement-{statement}-source-{source}-{suffix}.json"


def _write_evidence_summaries(
    process_data: Dict[str, Any],
    process_dir: Path,
    endpoint: str,
    timeout: int,
) -> Tuple[List[Dict[str, Any]], int]:
    output_dir = process_dir / "evidence_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_items: List[Dict[str, Any]] = []
    failed = 0

    for statement_index, statement in enumerate(process_data.get("statements", []) or []):
        statement_text = statement.get("text") or ""
        for evidence_index, evidence in enumerate(statement.get("evidence", []) or []):
            abstract = evidence.get("abstract") or ""
            if not abstract:
                continue

            filename = _summary_filename(
                statement_index,
                evidence_index,
                evidence.get("pubmed_id"),
            )
            response_file = output_dir / filename
            payload = {
                "statement": statement_text,
                "evidence": {
                    "abstract": abstract,
                    "title": (
                        evidence.get("title")
                        or evidence.get("article_title")
                        or evidence.get("paper_title")
                        or ""
                    ),
                    "pubmed_id": evidence.get("pubmed_id"),
                    "url": evidence.get("url"),
                    "stance": evidence.get("stance") or None,
                },
            }

            try:
                response_data = _post_json(endpoint, payload, timeout)
            except Exception as exc:
                failed += 1
                response_data = {
                    "error": str(exc),
                    "payload_includes_stance": True,
                }

            response_file.write_text(
                json.dumps(response_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            manifest_items.append(
                {
                    "statement_index": statement_index,
                    "statement_id": statement.get("id"),
                    "evidence_index": evidence_index,
                    "pubmed_id": evidence.get("pubmed_id"),
                    "url": evidence.get("url"),
                    "response_file": f"evidence_summaries/{filename}",
                    "payload_includes_stance": True,
                }
            )

    return manifest_items, failed


def _run_one_process(
    sweep: Dict[str, Any],
    run_index: int,
    video: Dict[str, Any],
    endpoint: str,
    timeout: int,
    whisper_model: Optional[str],
) -> Tuple[Dict[str, Any], int]:
    run_id = f"run_{run_index:02d}"
    reel_id = video["reel_id"]
    process_dir = sweep["root"] / run_id / reel_id
    process_dir.mkdir(parents=True, exist_ok=True)

    pipeline_run_id = f"{sweep['stance_mode']}_{run_id}_{reel_id}"
    config = _build_audio_pipeline_config(
        audio_path=video["audio_path"],
        stance_mode=sweep["stance_mode"],
        section_chunking_enabled=sweep["section_chunking_enabled"],
        whisper_model=whisper_model,
        run_id=pipeline_run_id,
    )

    print(
        f"[{sweep['stance_mode']}] {run_id} {reel_id}: pipeline start",
        flush=True,
    )
    state = PipelineState()
    final_state = PipelineOrchestrator(config).run(state)
    process_data = final_state.model_dump(mode="json")
    (process_dir / "process.json").write_text(
        json.dumps(process_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"[{sweep['stance_mode']}] {run_id} {reel_id}: evidence summaries start",
        flush=True,
    )
    summary_items, failures = _write_evidence_summaries(
        process_data=process_data,
        process_dir=process_dir,
        endpoint=endpoint,
        timeout=timeout,
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reel_url": video["reel_url"],
        "reel_id": reel_id,
        "stance_mode": sweep["stance_mode"],
        "section_chunking_enabled": sweep["section_chunking_enabled"],
        "whisper_model": whisper_model,
        "summary_payload_includes_stance": True,
        "process_response": "process.json",
        "evidence_summaries": summary_items,
    }
    (process_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"[{sweep['stance_mode']}] {run_id} {reel_id}: done "
        f"({len(summary_items)} summaries, {failures} failures)",
        flush=True,
    )
    return {
        "reel_url": video["reel_url"],
        "reel_id": reel_id,
        "process_response": f"{run_id}/{reel_id}/process.json",
        "manifest": f"{run_id}/{reel_id}/manifest.json",
        "summary_count": len(summary_items),
        "summary_failures": failures,
    }, failures


def _summary_label(text: str) -> str:
    lowered = (text or "").lower()
    neutral_hits = [
        "unclear",
        "not definitively",
        "does not directly",
        "doesn't directly",
        "not directly support or refute",
    ]
    refute_hits = [
        "contradict",
        "incorrect",
        "refute",
        "not through skin",
        "not directly support",
    ]
    support_hits = [
        "supports the statement",
        "support the statement",
        "supports",
        "may reduce",
        "help prevent",
        "reduced",
    ]
    if any(hit in lowered for hit in neutral_hits):
        return "Neutral"
    if any(hit in lowered for hit in refute_hits):
        return "Refutes/Neutral"
    if any(hit in lowered for hit in support_hits):
        return "Supports"
    return "Unknown"


def _is_mismatch(summary_label: str, stance_label: str) -> bool:
    if summary_label == "Unknown":
        return False
    if summary_label == "Supports":
        return stance_label != "Supports"
    if summary_label == "Neutral":
        return stance_label != "Neutral"
    if summary_label == "Refutes/Neutral":
        return stance_label == "Supports"
    return False


def _collect_sweep_stats(sweep: Dict[str, Any]) -> Dict[str, Any]:
    process_files = sorted(sweep["root"].glob("run_*/DT*/process.json"))
    manifest_files = sorted(sweep["root"].glob("run_*/DT*/manifest.json"))
    stance_counts: Counter[str] = Counter()
    summary_counts: Counter[str] = Counter()
    evidence_items = 0
    summary_items = 0
    summary_failures = 0
    mismatches = 0
    labels_by_key: Dict[str, str] = {}

    for process_file in process_files:
        process_data = json.loads(process_file.read_text(encoding="utf-8"))
        rel_process = process_file.relative_to(sweep["root"])
        run_id = rel_process.parts[0]
        reel_id = rel_process.parts[1]
        for statement_index, statement in enumerate(process_data.get("statements", []) or []):
            statement_text = statement.get("text") or ""
            for evidence_index, evidence in enumerate(statement.get("evidence", []) or []):
                evidence_items += 1
                stance = evidence.get("stance") or {}
                stance_label = stance.get("abstract_label") or "Unknown"
                stance_counts[stance_label] += 1
                key = "|".join(
                    [
                        run_id,
                        reel_id,
                        str(statement_index),
                        statement_text,
                        str(evidence.get("pubmed_id") or evidence_index),
                    ]
                )
                labels_by_key[key] = stance_label

    missing_refs = []
    for manifest_file in manifest_files:
        process_data = json.loads((manifest_file.parent / "process.json").read_text(encoding="utf-8"))
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        for item in manifest.get("evidence_summaries", []) or []:
            summary_items += 1
            response_file = manifest_file.parent / item.get("response_file", "")
            if not response_file.exists():
                missing_refs.append(str(response_file))
                continue
            response_data = json.loads(response_file.read_text(encoding="utf-8"))
            if "error" in response_data:
                summary_failures += 1
                continue
            summary_direction = _summary_label(response_data.get("summary", ""))
            summary_counts[summary_direction] += 1
            statement = process_data["statements"][item["statement_index"]]
            evidence = statement["evidence"][item["evidence_index"]]
            stance_label = (evidence.get("stance") or {}).get("abstract_label") or "Unknown"
            if _is_mismatch(summary_direction, stance_label):
                mismatches += 1

    return {
        "root": str(sweep["root"].relative_to(REPO_ROOT)),
        "stance_mode": sweep["stance_mode"],
        "process_files": len(process_files),
        "evidence_items": evidence_items,
        "evidence_summaries": summary_items,
        "summary_failures": summary_failures,
        "missing_manifest_references": missing_refs,
        "stance_counts": dict(stance_counts),
        "summary_direction_counts": dict(summary_counts),
        "summary_stance_mismatches": mismatches,
        "summary_stance_mismatch_rate": round(mismatches / summary_items, 4) if summary_items else 0.0,
        "labels_by_key": labels_by_key,
    }


def _compare_label_transitions(
    section_stats: Dict[str, Any],
    fixed_stats: Dict[str, Any],
) -> Dict[str, Any]:
    section_labels = section_stats.get("labels_by_key", {})
    fixed_labels = fixed_stats.get("labels_by_key", {})
    shared_keys = sorted(set(section_labels) & set(fixed_labels))
    transitions: Counter[str] = Counter()
    changed = 0
    for key in shared_keys:
        transition = f"{fixed_labels[key]}->{section_labels[key]}"
        transitions[transition] += 1
        if fixed_labels[key] != section_labels[key]:
            changed += 1
    return {
        "shared_evidence_keys": len(shared_keys),
        "changed_labels": changed,
        "fixed_to_section_transitions": dict(transitions),
    }


def _print_markdown_table(report: Dict[str, Any]) -> None:
    print()
    print("| Sweep | Supports | Refutes | Neutral | Evidence Summaries | Mismatch mit Evidence Summary |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name in ["Section-aware", "Fixed-token"]:
        stats = report["sweeps"][name]
        counts = stats.get("stance_counts", {})
        print(
            f"| {name} | "
            f"{counts.get('Supports', 0)} | "
            f"{counts.get('Refutes', 0)} | "
            f"{counts.get('Neutral', 0)} | "
            f"{stats.get('evidence_summaries', 0)} | "
            f"{stats.get('summary_stance_mismatches', 0)} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run dual clean sweeps comparing section-aware and fixed-token stance."
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--endpoint", default=SUMMARY_ENDPOINT)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--whisper-model", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-old-sweep", action="store_true")
    args = parser.parse_args()

    missing_audio = [str(video["audio_path"]) for video in VIDEOS if not video["audio_path"].exists()]
    if missing_audio:
        raise RuntimeError(f"Missing audio files: {missing_audio}")

    start = time.time()
    OFFLINE_ROOT.mkdir(parents=True, exist_ok=True)

    if args.force and not args.keep_old_sweep and OLD_SWEEP_ROOT.exists():
        print(f"Deleting old sweep: {OLD_SWEEP_ROOT}", flush=True)
        shutil.rmtree(OLD_SWEEP_ROOT)

    for sweep in SWEEPS:
        if sweep["root"].exists():
            if not args.force:
                raise RuntimeError(f"Sweep output already exists: {sweep['root']}. Use --force.")
            print(f"Deleting existing output: {sweep['root']}", flush=True)
            shutil.rmtree(sweep["root"])
        sweep["root"].mkdir(parents=True, exist_ok=True)

    total_summary_failures = 0
    for sweep in SWEEPS:
        root_manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stance_mode": sweep["stance_mode"],
            "section_chunking_enabled": sweep["section_chunking_enabled"],
            "whisper_model": args.whisper_model,
            "summary_payload_includes_stance": True,
            "summary_endpoint": args.endpoint,
            "runs": [],
        }
        for run_index in range(1, args.runs + 1):
            run_manifest = {
                "run_id": f"run_{run_index:02d}",
                "stance_mode": sweep["stance_mode"],
                "videos": [],
            }
            for video in VIDEOS:
                video_manifest, failures = _run_one_process(
                    sweep=sweep,
                    run_index=run_index,
                    video=video,
                    endpoint=args.endpoint,
                    timeout=args.timeout,
                    whisper_model=args.whisper_model,
                )
                total_summary_failures += failures
                run_manifest["videos"].append(video_manifest)
            run_dir = sweep["root"] / f"run_{run_index:02d}"
            (run_dir / "manifest.json").write_text(
                json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            root_manifest["runs"].append(run_manifest)

        (sweep["root"] / "manifest.json").write_text(
            json.dumps(root_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    section_stats = _collect_sweep_stats(SWEEPS[0])
    fixed_stats = _collect_sweep_stats(SWEEPS[1])
    transitions = _compare_label_transitions(section_stats, fixed_stats)

    section_stats.pop("labels_by_key", None)
    fixed_stats.pop("labels_by_key", None)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs_per_sweep": args.runs,
        "videos": [
            {
                "reel_id": video["reel_id"],
                "reel_url": video["reel_url"],
                "audio_path": str(video["audio_path"].relative_to(REPO_ROOT)),
            }
            for video in VIDEOS
        ],
        "sweeps": {
            "Section-aware": section_stats,
            "Fixed-token": fixed_stats,
        },
        "label_comparison": transitions,
        "elapsed_seconds": round(time.time() - start, 1),
        "total_summary_failures": total_summary_failures,
    }

    COMPARISON_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print_markdown_table(report)
    print()
    print(f"Report written: {COMPARISON_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
