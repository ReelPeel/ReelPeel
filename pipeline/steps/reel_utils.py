from __future__ import annotations

import os
import shutil
import tempfile
from typing import Optional
from urllib.parse import urlparse

from ..core.base import PipelineStep
from ..core.models import PipelineState


def _extract_shortcode(video_url: str) -> str:
    path = urlparse(video_url).path.rstrip("/")
    parts = path.split("/")
    for marker in ("reel", "reels", "p"):
        try:
            return parts[parts.index(marker) + 1]
        except (ValueError, IndexError):
            continue
    raise ValueError("Could not extract reel ID from URL")


def _find_mp4(target_dir: str) -> Optional[str]:
    for name in os.listdir(target_dir):
        if name.endswith(".mp4"):
            return os.path.join(target_dir, name)
    return None


def _download_with_yt_dlp(video_url: str, target_dir: str) -> Optional[str]:
    try:
        import yt_dlp
    except Exception:
        return None

    ydl_opts = {
        "outtmpl": os.path.join(target_dir, "%(id)s.%(ext)s"),
        "format": "mp4/best",
        "quiet": True,
        "noplaylist": True,
        "restrictfilenames": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            return ydl.prepare_filename(info)
    except Exception as exc:
        print(f"yt-dlp download failed: {exc}")
        return None


def _download_with_instaloader(video_url: str, target_dir: str) -> Optional[str]:
    try:
        import instaloader
    except Exception as exc:
        print(f"instaloader not available: {exc}")
        return None

    try:
        shortcode = _extract_shortcode(video_url)
    except Exception as exc:
        print(f"Error parsing reel URL: {exc}")
        return None

    try:
        os.makedirs(target_dir, exist_ok=True)
        loader = instaloader.Instaloader()
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        loader.download_post(post, target=target_dir)
        return _find_mp4(target_dir)
    except Exception as exc:
        print(f"Error downloading reel via instaloader: {exc}")
        return None


class DownloadReelStep(PipelineStep):
    """
    Downloads a reel video URL to a local mp4 file and stores its path in state.video_path.

    Config keys:
      - video_url | url | reel_url: str (required)
      - output_dir: str (optional, base directory for temp downloads)
    """

    def execute(self, state: PipelineState) -> PipelineState:
        video_url = (
            self.config.get("video_url")
            or self.config.get("url")
            or self.config.get("reel_url")
        )
        if not video_url:
            raise ValueError("[DownloadReelStep] Missing video_url (config)")

        output_dir = self.config.get("output_dir")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            download_dir = tempfile.mkdtemp(prefix="reel_", dir=output_dir)
        else:
            download_dir = tempfile.mkdtemp(prefix="reel_")

        video_path = _download_with_yt_dlp(video_url, download_dir)
        if not video_path:
            video_path = _download_with_instaloader(video_url, download_dir)

        if not video_path:
            shutil.rmtree(download_dir, ignore_errors=True)
            raise RuntimeError(
                "[DownloadReelStep] Failed to download reel without login. "
                "Install yt-dlp or check URL availability."
            )

        state.video_path = os.path.abspath(video_path)
        if self.debug:
            self.log_artifact("Video Path", video_path)
        return state
