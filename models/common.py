from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VideoInfo:
    path: str
    size_bytes: int
    fps: float
    duration_seconds: float
    frame_count: int


@dataclass(frozen=True, slots=True)
class Scene:
    index: int
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class TopicSegment:
    start: float
    end: float
    topic: str


@dataclass(frozen=True, slots=True)
class AudioEvent:
    start: float
    end: float
    time: float
    type: str
    confidence: float
    peak_value: float