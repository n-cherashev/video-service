"""
Artifact Models - типизированные модели для артефактов пайплайна.

Артефакт = файл на диске с метаданными.
ArtifactRef = ссылка на артефакт (путь + fingerprint).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ArtifactKind(str, Enum):
    """Типы артефактов."""
    VIDEO = "video"
    AUDIO = "audio"
    VIDEO_META = "video_meta"
    SCENES = "scenes"
    MOTION_SERIES = "motion_series"
    AUDIO_SERIES = "audio_series"
    TRANSCRIPT = "transcript"
    SENTIMENT_SERIES = "sentiment_series"
    LAUGHTER_SERIES = "laughter_series"
    CLARITY_SERIES = "clarity_series"
    HUMOR_SERIES = "humor_series"
    TIMELINE = "timeline"
    ANALYSIS_RESULT = "analysis_result"


@dataclass(frozen=True)
class ArtifactRef:
    """Ссылка на артефакт с fingerprint для кэширования."""
    kind: ArtifactKind
    path: str
    fingerprint: str  # sha256 от (input + settings_hash)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def exists(self) -> bool:
        return Path(self.path).exists()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "path": self.path,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactRef:
        return cls(
            kind=ArtifactKind(data["kind"]),
            path=data["path"],
            fingerprint=data["fingerprint"],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
        )


@dataclass(frozen=True)
class VideoArtifact:
    """Входное видео как артефакт."""
    ref: ArtifactRef
    size_bytes: int
    original_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref.to_dict(),
            "size_bytes": self.size_bytes,
            "original_name": self.original_name,
        }


@dataclass(frozen=True)
class AudioArtifact:
    """Извлечённое аудио как артефакт."""
    ref: ArtifactRef
    sample_rate: int
    channels: int
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref.to_dict(),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class VideoMeta:
    """Метаданные видео."""
    duration_seconds: float
    fps: float
    frame_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    has_audio: bool = True
    codec: Optional[str] = None
    bitrate: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "has_audio": self.has_audio,
            "codec": self.codec,
            "bitrate": self.bitrate,
        }


@dataclass
class SeriesArtifact:
    """Артефакт для time series данных (motion, audio features, etc.)."""
    ref: ArtifactRef
    step_seconds: float
    num_points: int
    columns: list[str]  # например ["loudness", "energy", "speech_probability"]
    summary: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref.to_dict(),
            "step_seconds": self.step_seconds,
            "num_points": self.num_points,
            "columns": self.columns,
            "summary": self.summary,
        }


@dataclass
class TranscriptArtifact:
    """Артефакт транскрипции."""
    ref: ArtifactRef
    language: str
    num_segments: int
    total_speech_duration: float
    total_characters: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref.to_dict(),
            "language": self.language,
            "num_segments": self.num_segments,
            "total_speech_duration": self.total_speech_duration,
            "total_characters": self.total_characters,
        }
