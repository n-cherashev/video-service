"""
FFmpegExtractHandler - извлекает аудио из видео через ffmpeg.

Extractor: создаёт WAV/FLAC файл для последующей обработки.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any, ClassVar, FrozenSet, Literal, Optional

from core.base_handler import ExtractorHandler
from models.keys import Key


AudioFormat = Literal["wav", "flac"]


class FFmpegExtractHandler(ExtractorHandler):
    """Handler для извлечения аудио через ffmpeg.

    Поддерживает:
    - WAV (pcm_s16le, 16kHz, mono) - стандарт для STT
    - FLAC (lossless, меньше места)

    Кэширование по fingerprint: если аудио уже извлечено, повторно не извлекает.

    Provides:
    - AUDIO_PATH: путь к извлечённому аудио
    """

    requires: ClassVar[FrozenSet[Key]] = frozenset({Key.VIDEO_PATH})
    provides: ClassVar[FrozenSet[Key]] = frozenset({Key.AUDIO_PATH})

    def __init__(
        self,
        temp_dir: str = "temp",
        format: AudioFormat = "wav",
        sample_rate: int = 16000,
        channels: int = 1,
        overwrite: bool = False,
        use_fingerprint_cache: bool = True,
    ) -> None:
        self.temp_dir = Path(temp_dir)
        self.format = format
        self.sample_rate = sample_rate
        self.channels = channels
        self.overwrite = overwrite
        self.use_fingerprint_cache = use_fingerprint_cache

        self._ffmpeg_available: Optional[bool] = None

    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        print("[3] FFmpegExtractHandler")

        video_path = context.get("video_path")
        if not video_path:
            raise ValueError("'video_path' not provided in context")

        # Проверяем ffmpeg
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg not found. Install: https://ffmpeg.org/download.html")

        # Создаём temp директорию
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Определяем имя выходного файла
        if self.use_fingerprint_cache:
            fingerprint = context.get("video_fingerprint", "")
            if fingerprint:
                audio_filename = f"audio_{fingerprint}.{self.format}"
            else:
                audio_filename = f"{Path(video_path).stem}.{self.format}"
        else:
            audio_filename = f"{Path(video_path).stem}.{self.format}"

        audio_path = self.temp_dir / audio_filename

        # Проверяем кэш
        if not self.overwrite and audio_path.exists():
            context["audio_path"] = str(audio_path)
            print(f"✓ Audio cached: {audio_path.name}")
            return context

        # Извлекаем аудио
        self._extract_audio(video_path, str(audio_path))

        context["audio_path"] = str(audio_path)
        context["audio_sample_rate"] = self.sample_rate
        context["audio_channels"] = self.channels

        file_size_mb = audio_path.stat().st_size / 1024 / 1024
        print(f"✓ Audio extracted: {audio_path.name} ({file_size_mb:.1f} MB)")

        return context

    def _check_ffmpeg(self) -> bool:
        """Проверяет доступность ffmpeg."""
        if self._ffmpeg_available is not None:
            return self._ffmpeg_available

        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
            )
            self._ffmpeg_available = True
        except (FileNotFoundError, subprocess.CalledProcessError):
            self._ffmpeg_available = False

        return self._ffmpeg_available

    def _extract_audio(self, video_path: str, audio_path: str) -> None:
        """Извлекает аудио через ffmpeg."""
        # Определяем кодек
        if self.format == "wav":
            codec_args = ["-acodec", "pcm_s16le"]
        elif self.format == "flac":
            codec_args = ["-acodec", "flac"]
        else:
            codec_args = []

        cmd = [
            "ffmpeg",
            "-y",  # overwrite
            "-i", video_path,
            "-vn",  # no video
            "-ac", str(self.channels),
            "-ar", str(self.sample_rate),
            *codec_args,
            audio_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg extraction failed: {e.stderr}") from e

    def cleanup(self, context: dict[str, Any]) -> None:
        """Удаляет временный аудио файл."""
        audio_path = context.get("audio_path")
        if audio_path and Path(audio_path).exists():
            Path(audio_path).unlink()
