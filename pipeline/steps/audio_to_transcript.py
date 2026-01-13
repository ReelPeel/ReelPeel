from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..core.base import PipelineStep
from ..core.models import PipelineState


DEFAULT_WHISPER_MODEL = "turbo"


class AudioToTranscriptStep(PipelineStep):
    """
    Transcribes a local audio file using OpenAI Whisper and stores the result
    in PipelineState.transcript.

    Config keys:
      - audio_path: str (optional if state.audio_path is set)
      - whisper_model: str (default: "turbo")
      - fp16: bool (default: True)
      - translate_non_english: bool (default: True)
    """

    def execute(self, state: PipelineState) -> PipelineState:
        audio_path = self.config.get("audio_path") or state.audio_path
        if not audio_path:
            raise ValueError(
                "[AudioToTranscriptStep] Missing audio_path (config or state.audio_path)"
            )

        path = Path(audio_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"[AudioToTranscriptStep] Audio file not found: {path}")

        try:
            import whisper
        except Exception as exc:
            raise RuntimeError(
                "[AudioToTranscriptStep] openai-whisper is required for audio transcription."
            ) from exc

        model_name = self.config.get("whisper_model", DEFAULT_WHISPER_MODEL)
        fp16 = bool(self.config.get("fp16", True))
        translate_non_english = bool(self.config.get("translate_non_english", True))

        print(f"[{self.__class__.__name__}] Transcribing audio: {path}")
        model = whisper.load_model(model_name)
        result = model.transcribe(str(path), fp16=fp16)

        transcript = str(result.get("text", "")).strip()
        if translate_non_english and result.get("language", "en") != "en":
            result = model.transcribe(str(path), task="translate", fp16=fp16)
            transcript = str(result.get("text", "")).strip()

        state.audio_path = str(path)
        state.transcript = transcript
        state.generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

        if self.debug:
            self.log_artifact("Transcript", transcript)

        return state
