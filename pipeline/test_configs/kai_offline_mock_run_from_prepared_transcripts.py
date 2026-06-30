from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.core.models import PipelineState
from pipeline.core.orchestrator import PipelineOrchestrator
from pipeline.test_configs.video_transcription_config import VIDEO_URL_PIPELINE_CONFIG


OFFLINE_ROOT = REPO_ROOT / "offline_mock"
VIDEO_IDS = ["DT0UIgzDZ79", "DT0UbkjDZZj"]


def _build_config(video_id: str, retmax: int | None = None) -> Dict[str, Any]:
    config = copy.deepcopy(VIDEO_URL_PIPELINE_CONFIG)
    config["name"] = "Offline_From_Prepared_Transcript"
    config["run_id"] = f"prepared_transcript_{video_id}"
    config["steps"] = [
        step
        for step in config.get("steps", [])
        if step.get("type") not in {"download_reel", "video_to_audio", "audio_to_transcript"}
    ]
    if retmax is not None:
        for step in config.get("steps", []):
            step_type = step.get("type")
            settings = step.setdefault("settings", {})
            if step_type == "generate_query":
                prefetch_links = settings.setdefault("prefetch_links", {})
                prefetch_links["retmax"] = int(retmax)
            elif step_type == "fetch_links":
                settings["retmax"] = int(retmax)
    return config


def _run_one(video_id: str, force: bool, retmax: int | None = None) -> Dict[str, Any]:
    video_dir = OFFLINE_ROOT / video_id
    manifest_path = video_dir / "manifest.json"
    transcript_path = video_dir / "transcript.txt"
    process_path = video_dir / "process.json"

    if not manifest_path.exists():
        raise RuntimeError(f"Missing manifest: {manifest_path}")
    if not transcript_path.exists():
        raise RuntimeError(f"Missing transcript: {transcript_path}")
    if process_path.exists() and not force:
        raise RuntimeError(f"Output already exists: {process_path}. Use --force.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    transcript = transcript_path.read_text(encoding="utf-8").strip()
    if not transcript:
        raise RuntimeError(f"Transcript is empty: {transcript_path}")

    state = PipelineState(
        transcript=transcript,
        audio_path=str(REPO_ROOT / manifest["audio"]),
        video_path=str(REPO_ROOT / manifest["video"]),
    )
    config = _build_config(video_id, retmax=retmax)
    final_state = PipelineOrchestrator(config).run(state)

    process_path.write_text(
        json.dumps(final_state.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest["pipeline_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["pipeline_mode"] = "from_prepared_transcript"
    manifest["pipeline_config_basis"] = "VIDEO_URL_PIPELINE_CONFIG_without_media_steps"
    manifest["offline_retmax_override"] = retmax
    manifest["process_response"] = str(process_path.relative_to(REPO_ROOT))
    manifest["pipeline_skipped_steps"] = [
        "download_reel",
        "video_to_audio",
        "audio_to_transcript",
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "reel_id": video_id,
        "process_response": str(process_path.relative_to(REPO_ROOT)),
        "retmax": retmax,
        "statement_count": len(final_state.statements),
        "evidence_count": sum(len(stmt.evidence) for stmt in final_state.statements),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the rest of the current video pipeline from prepared transcripts."
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retmax", type=int, default=None)
    args = parser.parse_args()

    results: List[Dict[str, Any]] = []
    for video_id in VIDEO_IDS:
        print(f"[pipeline] {video_id}: start", flush=True)
        result = _run_one(video_id, force=args.force, retmax=args.retmax)
        results.append(result)
        print(
            f"[pipeline] {video_id}: wrote {result['process_response']} "
            f"({result['statement_count']} statements, {result['evidence_count']} evidence)",
            flush=True,
        )

    root_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "from_prepared_transcript",
        "config_basis": "VIDEO_URL_PIPELINE_CONFIG_without_media_steps",
        "offline_retmax_override": args.retmax,
        "videos": results,
    }
    root_manifest_path = OFFLINE_ROOT / "prepared_transcript_pipeline_manifest.json"
    root_manifest_path.write_text(
        json.dumps(root_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest written: {root_manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
