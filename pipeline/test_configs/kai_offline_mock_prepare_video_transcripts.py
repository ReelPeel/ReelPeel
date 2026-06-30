from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.core.models import PipelineState
from pipeline.steps.audio_to_transcript import AudioToTranscriptStep
from pipeline.steps.reel_utils import DownloadReelStep
from pipeline.steps.video_to_audio import VideoToAudioStep
from pipeline.test_configs.video_transcription_config import VIDEO_URL_PIPELINE_CONFIG


OFFLINE_ROOT = REPO_ROOT / "offline_mock"
WHISPER_MODEL = "large-v3"

VIDEOS: List[Dict[str, str]] = [
    {
        "reel_id": "DT0UIgzDZ79",
        "reel_url": "https://www.instagram.com/reels/DT0UIgzDZ79/",
        "local_video": "browser-extension_vault/offline-demo/assets/reel-DT0UIgzDZ79.mp4",
    },
    {
        "reel_id": "DT0UbkjDZZj",
        "reel_url": "https://www.instagram.com/reels/DT0UbkjDZZj/",
        "local_video": "browser-extension_vault/offline-demo/assets/reel-DT0UbkjDZZj.mp4",
    },
]


def _settings_for(step_type: str) -> Dict[str, Any]:
    for step in VIDEO_URL_PIPELINE_CONFIG.get("steps", []):
        if step.get("type") == step_type:
            return dict(step.get("settings") or {})
    return {}


def _copy_to_stable_path(source: str, destination: Path) -> Path:
    source_path = Path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != destination.resolve():
        shutil.copy2(source_path, destination)
    return destination


def _prepare_one(video: Dict[str, str], force: bool) -> Dict[str, Any]:
    reel_id = video["reel_id"]
    reel_url = video["reel_url"]
    output_dir = OFFLINE_ROOT / reel_id

    if output_dir.exists():
        if not force:
            raise RuntimeError(f"Output already exists: {output_dir}. Use --force.")
        shutil.rmtree(output_dir)

    video_dir = output_dir / "video"
    audio_dir = output_dir / "audio"
    work_dir = output_dir / "work"
    video_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    state = PipelineState()

    download_settings = _settings_for("download_reel")
    download_settings["video_url"] = reel_url
    download_settings["output_dir"] = str(work_dir)
    download_source = "instagram"
    try:
        state = DownloadReelStep(download_settings).run(state)
    except Exception as exc:
        local_video = REPO_ROOT / video["local_video"]
        if not local_video.exists():
            raise
        print(
            f"[prepare] {reel_id}: download failed ({exc}); using local video {local_video}",
            flush=True,
        )
        state.video_path = str(local_video)
        download_source = "local_fallback"

    downloaded_video = Path(state.video_path)
    stable_video = _copy_to_stable_path(
        str(downloaded_video),
        video_dir / f"{reel_id}{downloaded_video.suffix or '.mp4'}",
    )
    state.video_path = str(stable_video)

    video_to_audio_settings = _settings_for("video_to_audio")
    video_to_audio_settings["output_path"] = str(audio_dir / f"{reel_id}.wav")
    state = VideoToAudioStep(video_to_audio_settings).run(state)

    audio_to_transcript_settings = _settings_for("audio_to_transcript")
    audio_to_transcript_settings["audio_path"] = state.audio_path
    audio_to_transcript_settings["whisper_model"] = WHISPER_MODEL
    state = AudioToTranscriptStep(audio_to_transcript_settings).run(state)

    transcript = state.transcript or ""
    (output_dir / "transcript.txt").write_text(transcript + "\n", encoding="utf-8")

    state_path = output_dir / "state_transcript_only.json"
    state_path.write_text(
        json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reel_id": reel_id,
        "reel_url": reel_url,
        "mode": "transcript_only",
        "config_basis": "VIDEO_URL_PIPELINE_CONFIG",
        "pipeline_stopped_after": "audio_to_transcript",
        "whisper_model": WHISPER_MODEL,
        "video_source": download_source,
        "video": str(stable_video.relative_to(REPO_ROOT)),
        "audio": str(Path(state.audio_path).relative_to(REPO_ROOT)),
        "transcript": str((output_dir / "transcript.txt").relative_to(REPO_ROOT)),
        "state": str(state_path.relative_to(REPO_ROOT)),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the two test reels, extract audio, and transcribe only."
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OFFLINE_ROOT.mkdir(parents=True, exist_ok=True)
    manifests = []
    for video in VIDEOS:
        print(f"[prepare] {video['reel_id']}: start", flush=True)
        manifest = _prepare_one(video, force=args.force)
        manifests.append(manifest)
        print(
            f"[prepare] {video['reel_id']}: transcript written to {manifest['transcript']}",
            flush=True,
        )

    root_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "transcript_only",
        "whisper_model": WHISPER_MODEL,
        "videos": manifests,
    }
    manifest_path = OFFLINE_ROOT / "video_transcript_prepare_manifest.json"
    manifest_path.write_text(
        json.dumps(root_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest written: {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
