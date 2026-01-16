from __future__ import annotations

from copy import deepcopy
import os
from typing import Any, Dict, Optional

from pipeline.core.models import PipelineState
from pipeline.core.orchestrator import PipelineOrchestrator
from pipeline.test_configs.audio_transcription_config import AUDIO_PIPELINE_CONFIG
from pipeline.test_configs.video_transcription_config import VIDEO_URL_PIPELINE_CONFIG


def _build_audio_config(audio_path: str) -> Dict[str, Any]:
    cfg = deepcopy(AUDIO_PIPELINE_CONFIG)
    for step in cfg.get("steps", []):
        if step.get("type") == "audio_to_transcript":
            settings = step.setdefault("settings", {})
            settings["audio_path"] = audio_path
            break
    else:
        raise ValueError("AUDIO_PIPELINE_CONFIG is missing audio_to_transcript step")
    return cfg


def _build_video_url_config(video_url: str) -> Dict[str, Any]:
    cfg = deepcopy(VIDEO_URL_PIPELINE_CONFIG)
    for step in cfg.get("steps", []):
        if step.get("type") == "download_reel":
            settings = step.setdefault("settings", {})
            settings["video_url"] = video_url
            break
    else:
        raise ValueError("VIDEO_URL_PIPELINE_CONFIG is missing download_reel step")
    return cfg


def run_pipeline(
    audio_path: Optional[str] = None,
    video_url: Optional[str] = None,
) -> Dict[str, Any]:
    if bool(audio_path) == bool(video_url):
        raise ValueError("Provide exactly one of audio_path or video_url")

    if audio_path:
        audio_path = os.path.abspath(audio_path)
        config = _build_audio_config(audio_path)
    else:
        config = _build_video_url_config(video_url)
    state = PipelineState()
    orchestrator = PipelineOrchestrator(config)
    final_state = orchestrator.run(state)
    return final_state.model_dump(mode="json")
