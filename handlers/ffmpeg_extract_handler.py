from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from handlers.base_handler import BaseHandler


class FFmpegExtractHandler(BaseHandler):
    def __init__(self, temp_dir: str = "temp", overwrite: bool = True) -> None:
        self.temp_dir = temp_dir
        self.overwrite = bool(overwrite)

    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        video_path = context.get("video_path")
        if not video_path:
            raise ValueError("'video_path' not provided in context")

        Path(self.temp_dir).mkdir(exist_ok=True)

        video_name = Path(video_path).stem
        audio_path = str(Path(self.temp_dir) / f"{video_name}.wav")

        # Если файл уже есть и overwrite=False — используем кеш
        if (not self.overwrite) and os.path.exists(audio_path):
            context["audio_path"] = audio_path
            return context

        cmd = [
            "ffmpeg",
            "-y" if self.overwrite else "-n",
            "-i", video_path,
            "-vn",                 # без видео
            "-ac", "1",            # mono
            "-ar", "16000",        # 16kHz удобно для STT
            "-acodec", "pcm_s16le",
            audio_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise RuntimeError("FFmpeg not found. Install: brew install ffmpeg") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg extraction failed: {e.stderr}") from e

        context["audio_path"] = audio_path
        return context
