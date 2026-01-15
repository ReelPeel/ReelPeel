from __future__ import annotations

from copy import deepcopy
import os
from typing import Any, Dict

from pipeline.core.models import PipelineState
from pipeline.core.orchestrator import PipelineOrchestrator
from pipeline.test_configs.video_transcription_config import VIDEO_PIPELINE_CONFIG


def _build_audio_config(audio_path: str) -> Dict[str, Any]:
    cfg = deepcopy(VIDEO_PIPELINE_CONFIG)
    for step in cfg.get("steps", []):
        if step.get("type") == "audio_to_transcript":
            settings = step.setdefault("settings", {})
            settings["audio_path"] = audio_path
            break
    else:
        raise ValueError("AUDIO_PIPELINE_CONFIG is missing audio_to_transcript step")
    return cfg


def run_pipeline(audio_path: str) -> Dict[str, Any]:
    if not audio_path:
        raise ValueError("audio_path is required")
    audio_path = os.path.abspath(audio_path)

    config = _build_audio_config(audio_path)
    state = PipelineState()
    orchestrator = PipelineOrchestrator(config)
    final_state = orchestrator.run(state)
    return final_state.model_dump(mode="json")
