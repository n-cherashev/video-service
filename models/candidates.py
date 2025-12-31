"""
Candidate Models - модели для кандидатов и якорей.

Anchor - точка интереса для генерации кандидатов.
CandidateWindow - окно-кандидат для оценки.
ScoredClipV2 - оценённый клип с полным breakdown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


class AnchorType(str, Enum):
    """Типы якорей для генерации кандидатов."""
    INTEREST_PEAK = "interest_peak"
    MOTION_PEAK = "motion_peak"
    AUDIO_PEAK = "audio_peak"
    SCENE_BOUNDARY = "scene_boundary"
    DIALOGUE_TRANSITION = "dialogue_transition"
    SENTIMENT_CHANGE = "sentiment_change"


@dataclass(frozen=True)
class Anchor:
    """Якорь - точка интереса для генерации кандидата."""
    time: float
    type: AnchorType
    value: float
    subtype: Optional[str] = None
    index: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": round(self.time, 2),
            "type": self.type.value,
            "value": round(self.value, 3),
            "subtype": self.subtype,
            "index": self.index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Anchor:
        return cls(
            time=float(data["time"]),
            type=AnchorType(data["type"]),
            value=float(data["value"]),
            subtype=data.get("subtype"),
            index=data.get("index"),
        )


@dataclass
class CandidateWindow:
    """Окно-кандидат для оценки."""
    id: str
    start: float
    end: float
    anchor_time: float
    anchor_type: AnchorType

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "anchor_time": round(self.anchor_time, 2),
            "anchor_type": self.anchor_type.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateWindow:
        return cls(
            id=data["id"],
            start=float(data["start"]),
            end=float(data["end"]),
            anchor_time=float(data["anchor_time"]),
            anchor_type=AnchorType(data["anchor_type"]),
        )


@dataclass
class ScoreComponents:
    """Компоненты скоринга клипа."""
    hook: float = 0.0
    pace: float = 0.0
    clarity: float = 0.0
    intensity: float = 0.0
    emotion: float = 0.0
    boundary: float = 0.0
    momentum: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "hook": round(self.hook, 3),
            "pace": round(self.pace, 3),
            "clarity": round(self.clarity, 3),
            "intensity": round(self.intensity, 3),
            "emotion": round(self.emotion, 3),
            "boundary": round(self.boundary, 3),
            "momentum": round(self.momentum, 3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> ScoreComponents:
        return cls(
            hook=data.get("hook", 0.0),
            pace=data.get("pace", 0.0),
            clarity=data.get("clarity", 0.0),
            intensity=data.get("intensity", 0.0),
            emotion=data.get("emotion", 0.0),
            boundary=data.get("boundary", 0.0),
            momentum=data.get("momentum", 0.0),
        )

    def weighted_sum(self, weights: dict[str, float]) -> float:
        """Вычисляет взвешенную сумму компонентов."""
        total = 0.0
        total_weight = 0.0
        for name, weight in weights.items():
            value = getattr(self, name, 0.0)
            total += weight * value
            total_weight += weight
        return total / total_weight if total_weight > 0 else 0.0


@dataclass
class ScoredClipV2:
    """Оценённый клип V2 с полным breakdown и метаданными."""
    id: str
    start: float
    end: float
    score: float
    components: ScoreComponents
    anchor_type: AnchorType
    reasons: List[str] = field(default_factory=list)
    overlap_group: Optional[int] = None

    # LLM refinement (опционально)
    llm_score: Optional[float] = None
    llm_reasoning: Optional[str] = None
    final_score: Optional[float] = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def get_effective_score(self) -> float:
        """Возвращает финальный скор (с учётом LLM если есть)."""
        return self.final_score if self.final_score is not None else self.score

    def to_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "score": round(self.score, 4),
            "score_breakdown": self.components.to_dict(),
            "anchor_type": self.anchor_type.value,
            "reasons": self.reasons,
            "overlap_group": self.overlap_group,
        }
        if self.llm_score is not None:
            result["llm_score"] = round(self.llm_score, 4)
            result["llm_reasoning"] = self.llm_reasoning
        if self.final_score is not None:
            result["final_score"] = round(self.final_score, 4)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoredClipV2:
        return cls(
            id=data["id"],
            start=float(data["start"]),
            end=float(data["end"]),
            score=float(data["score"]),
            components=ScoreComponents.from_dict(data.get("score_breakdown", {})),
            anchor_type=AnchorType(data["anchor_type"]),
            reasons=data.get("reasons", []),
            overlap_group=data.get("overlap_group"),
            llm_score=data.get("llm_score"),
            llm_reasoning=data.get("llm_reasoning"),
            final_score=data.get("final_score"),
        )


@dataclass
class AnchorSummary:
    """Сводка по якорям."""
    total: int = 0
    by_type: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "by_type": self.by_type,
        }
