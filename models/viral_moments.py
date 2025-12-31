from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Candidate:
    """Кандидат-клип для анализа."""
    start: float
    end: float
    anchor_time: float
    anchor_type: str
    duration: float

    def __post_init__(self):
        self.duration = self.end - self.start


@dataclass
class ScoredClip:
    """Клип с полным скорингом и метаданными."""
    start: float
    end: float
    score: float
    score_breakdown: Dict[str, float]
    anchor_type: str
    reasons: List[str]
    overlap_group_id: Optional[int] = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для JSON."""
        return {
            "start": self.start,
            "end": self.end,
            "score": round(self.score, 4),
            "score_breakdown": {k: round(v, 2) for k, v in self.score_breakdown.items()},
            "anchor_type": self.anchor_type,
            "reasons": self.reasons,
            "overlap_group_id": self.overlap_group_id
        }


@dataclass
class ViralMomentsConfig:
    """Конфигурация для поиска виральных моментов."""

    # Anchor detection
    interest_percentile: float = 0.85
    motion_percentile: float = 0.80
    audio_percentile: float = 0.80

    # Candidate generation
    candidate_durations: List[float] = None
    anchor_positions: List[str] = None  # ["center", "start", "end"]
    min_clip_duration: float = 30.0  # Минимальная длина клипа (сек)
    max_clip_duration: float = 60.0  # Максимальная длина клипа (1 минута)

    # Scoring weights
    hook_weight: float = 0.25
    pace_weight: float = 0.20
    clarity_weight: float = 0.15
    intensity_weight: float = 0.20
    emotion_weight: float = 0.10
    boundary_weight: float = 0.10

    # Diversification
    max_clips: int = 8
    min_gap_seconds: float = 30.0  # Минимальный gap между клипами
    strong_overlap_threshold: float = 0.50  # Снижено с 0.65
    medium_overlap_threshold: float = 0.25  # Снижено с 0.35
    strong_penalty: float = 0.2  # Усилен штраф
    medium_penalty: float = 0.6

    # Фильтрация дубликатов
    duplicate_overlap_threshold: float = 0.70  # Клипы с overlap > 70% считаются дубликатами

    # Local re-fit
    refit_shifts: List[int] = None

    def __post_init__(self):
        if self.candidate_durations is None:
            self.candidate_durations = [15.0, 25.0, 35.0, 45.0, 60.0]

        if self.anchor_positions is None:
            self.anchor_positions = ["center", "start", "end"]

        if self.refit_shifts is None:
            self.refit_shifts = [-8, -5, -3, -2, 2, 3, 5, 8]
