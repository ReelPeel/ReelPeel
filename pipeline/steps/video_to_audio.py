from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..core.base import PipelineStep
from ..core.models import PipelineState


DEFAULT_AUDIO_EXT = ".wav"


class VideoToAudioStep(PipelineStep):
    """
    Converts a local video file to an audio file and stores the output path
    in PipelineState.audio_path.

    Config keys:
      - video_path: str (required)
      - output_path: str (optional, file or directory)
      - fps: int (optional, default: moviepy default)
      - codec: str (optional)
      - bitrate: str (optional)
    """

    def execute(self, state: PipelineState) -> PipelineState:
        video_path = self.config.get("video_path")
        if not video_path:
            raise ValueError("[VideoToAudioStep] Missing required config: video_path")

        path = Path(video_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"[VideoToAudioStep] Video file not found: {path}")

        output_path = self._resolve_output_path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import moviepy.editor as mp
        except Exception as exc:
            raise RuntimeError(
                "[VideoToAudioStep] moviepy is required for video-to-audio conversion."
            ) from exc

        fps = self.config.get("fps")
        codec = self.config.get("codec")
        bitrate = self.config.get("bitrate")

        print(f"[{self.__class__.__name__}] Extracting audio: {path} -> {output_path}")

        clip = mp.VideoFileClip(str(path))
        try:
            if clip.audio is None:
                raise ValueError(f"[VideoToAudioStep] No audio track found in: {path}")

            clip.audio.write_audiofile(
                str(output_path),
                fps=fps,
                codec=codec,
                bitrate=bitrate,
            )
        finally:
            if clip.audio is not None:
                clip.audio.close()
            clip.close()

        state.audio_path = str(output_path)
        state.generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

        if self.debug:
            self.log_artifact("Audio Path", state.audio_path)

        return state

    def _resolve_output_path(self, video_path: Path) -> Path:
        output_path = self.config.get("output_path")
        if not output_path:
            return Path("temp") / f"{video_path.stem}{DEFAULT_AUDIO_EXT}"

        output = Path(output_path).expanduser()
        if output.exists() and output.is_dir():
            return output / f"{video_path.stem}{DEFAULT_AUDIO_EXT}"

        if output.suffix == "":
            return output / f"{video_path.stem}{DEFAULT_AUDIO_EXT}"

        return output
