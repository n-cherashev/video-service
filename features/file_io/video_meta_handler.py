"""
VideoMetaHandler - извлекает метаданные видео.

Extractor: использует ffprobe (предпочтительно) или OpenCV для получения
duration, fps, frame_count, width, height, has_audio.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, ClassVar, FrozenSet, Optional

import cv2
import numpy as np

from core.base_handler import ExtractorHandler
from models.keys import Key


class VideoMetaHandler(ExtractorHandler):
    """Handler для извлечения метаданных видео.

    Использует ffprobe если доступен (более точный), иначе OpenCV.

    Provides:
    - FPS: частота кадров
    - DURATION_SECONDS: длительность видео
    - FRAME_COUNT: количество кадров
    - video_width, video_height: разрешение
    - video_has_audio: наличие аудио дорожки
    """

    requires: ClassVar[FrozenSet[Key]] = frozenset({Key.VIDEO_PATH})
    provides: ClassVar[FrozenSet[Key]] = frozenset({
        Key.FPS,
        Key.DURATION_SECONDS,
        Key.FRAME_COUNT,
    })

    def __init__(self, prefer_ffprobe: bool = True) -> None:
        self.prefer_ffprobe = prefer_ffprobe
        self._ffprobe_available: Optional[bool] = None

    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        print("[2] VideoMetaHandler")

        video_path = context.get("video_path")
        if not video_path:
            raise ValueError("'video_path' not provided in context")

        # Пробуем ffprobe сначала
        if self.prefer_ffprobe and self._check_ffprobe():
            meta = self._extract_with_ffprobe(video_path)
        else:
            meta = self._extract_with_opencv(video_path)

        # Записываем в контекст
        context["fps"] = meta["fps"]
        context["duration_seconds"] = meta["duration_seconds"]
        context["frame_count"] = meta["frame_count"]
        context["video_width"] = meta.get("width")
        context["video_height"] = meta.get("height")
        context["video_has_audio"] = meta.get("has_audio", True)
        context["video_codec"] = meta.get("codec")
        context["video_bitrate"] = meta.get("bitrate")

        print(
            f"✓ Video meta: {meta['fps']:.1f} fps, "
            f"{meta['duration_seconds']:.1f}s, "
            f"{meta.get('width', '?')}x{meta.get('height', '?')}"
        )

        return context

    def _check_ffprobe(self) -> bool:
        """Проверяет доступность ffprobe."""
        if self._ffprobe_available is not None:
            return self._ffprobe_available

        try:
            subprocess.run(
                ["ffprobe", "-version"],
                capture_output=True,
                check=True,
            )
            self._ffprobe_available = True
        except (FileNotFoundError, subprocess.CalledProcessError):
            self._ffprobe_available = False

        return self._ffprobe_available

    def _extract_with_ffprobe(self, video_path: str) -> dict[str, Any]:
        """Извлекает метаданные через ffprobe (более точный способ)."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        # Ищем видео стрим
        video_stream = None
        audio_stream = None

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and video_stream is None:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and audio_stream is None:
                audio_stream = stream

        if not video_stream:
            raise ValueError(f"No video stream found in {video_path}")

        # Извлекаем данные
        format_info = data.get("format", {})

        # Duration
        duration = float(format_info.get("duration", 0))
        if duration == 0:
            duration = float(video_stream.get("duration", 0))

        # FPS
        fps = 25.0
        r_frame_rate = video_stream.get("r_frame_rate", "25/1")
        if "/" in r_frame_rate:
            num, den = map(float, r_frame_rate.split("/"))
            if den > 0:
                fps = num / den

        # Frame count
        frame_count = int(video_stream.get("nb_frames", 0))
        if frame_count == 0 and duration > 0 and fps > 0:
            frame_count = int(duration * fps)

        # Resolution
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        # Codec
        codec = video_stream.get("codec_name")

        # Bitrate
        bitrate = int(format_info.get("bit_rate", 0))

        return {
            "fps": fps,
            "duration_seconds": duration,
            "frame_count": frame_count,
            "width": width if width > 0 else None,
            "height": height if height > 0 else None,
            "has_audio": audio_stream is not None,
            "codec": codec,
            "bitrate": bitrate if bitrate > 0 else None,
        }

    def _extract_with_opencv(self, video_path: str) -> dict[str, Any]:
        """Извлекает метаданные через OpenCV (fallback)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 0:
                fps = 25.0

            frame_count_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_count = int(frame_count_raw) if np.isfinite(frame_count_raw) and frame_count_raw > 0 else None

            duration_seconds = None
            if frame_count and fps > 0:
                duration_seconds = float(frame_count) / fps

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            return {
                "fps": fps,
                "duration_seconds": duration_seconds,
                "frame_count": frame_count,
                "width": width if width > 0 else None,
                "height": height if height > 0 else None,
                "has_audio": True,  # OpenCV не может определить
                "codec": None,
                "bitrate": None,
            }
        finally:
            cap.release()
