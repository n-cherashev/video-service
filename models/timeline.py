from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimelinePoint:
    time: float
    interest: float
    motion: float
    audio_loudness: float
    sentiment: float
    humor: float
    has_laughter: bool
    has_loud_sound: bool
    is_scene_boundary: bool
    is_dialogue: bool
