"""
Analysis Result Models - публичные и полные результаты анализа.

AnalysisResultPublic - для API ответа (компактный).
AnalysisResultFull - для сохранения на диск (полный + артефакты).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .artifacts import ArtifactRef


@dataclass
class TranscriptSegment:
    """Сегмент транскрипции."""
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text}


@dataclass
class ScoreBreakdown:
    """Breakdown скоринга клипа."""
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
    def from_dict(cls, data: dict[str, float]) -> ScoreBreakdown:
        return cls(
            hook=data.get("hook", 0.0),
            pace=data.get("pace", 0.0),
            clarity=data.get("clarity", 0.0),
            intensity=data.get("intensity", 0.0),
            emotion=data.get("emotion", 0.0),
            boundary=data.get("boundary", 0.0),
            momentum=data.get("momentum", 0.0),
        )


@dataclass
class ViralClipResult:
    """Viral клип в результате анализа."""
    id: str
    start: float
    end: float
    score: float
    score_breakdown: ScoreBreakdown
    anchor_type: str
    reasons: List[str]
    duration: float = 0.0

    def __post_init__(self) -> None:
        if self.duration == 0.0:
            self.duration = self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "score": round(self.score, 4),
            "score_breakdown": self.score_breakdown.to_dict(),
            "anchor_type": self.anchor_type,
            "reasons": self.reasons,
        }


@dataclass
class ChapterResult:
    """Глава в результате анализа."""
    id: str
    start: float
    end: float
    title: str
    description: str
    duration: float = 0.0

    def __post_init__(self) -> None:
        if self.duration == 0.0:
            self.duration = self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "title": self.title,
            "description": self.description,
        }


@dataclass
class AnalysisSummary:
    """Сводка анализа."""
    total_scenes: int = 0
    total_speech_duration: float = 0.0
    speech_ratio: float = 0.0
    mean_motion: float = 0.0
    mean_loudness: float = 0.0
    mean_interest: float = 0.0
    detected_language: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_scenes": self.total_scenes,
            "total_speech_duration": round(self.total_speech_duration, 2),
            "speech_ratio": round(self.speech_ratio, 3),
            "mean_motion": round(self.mean_motion, 3),
            "mean_loudness": round(self.mean_loudness, 3),
            "mean_interest": round(self.mean_interest, 3),
            "detected_language": self.detected_language,
        }


@dataclass
class AnalysisResultPublic:
    """Публичный результат анализа (для API)."""
    task_id: str
    video_name: str
    duration_seconds: float
    processing_time_seconds: float

    viral_clips: List[ViralClipResult]
    chapters: List[ChapterResult]
    summary: AnalysisSummary

    created_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "video_name": self.video_name,
            "duration_seconds": round(self.duration_seconds, 2),
            "processing_time_seconds": round(self.processing_time_seconds, 2),
            "viral_clips": [c.to_dict() for c in self.viral_clips],
            "chapters": [c.to_dict() for c in self.chapters],
            "summary": self.summary.to_dict(),
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }


@dataclass
class TimelinePointResult:
    """Точка timeline для полного результата."""
    time: float
    interest: float
    motion: float
    audio_loudness: float
    clarity: float
    sentiment: float
    has_laughter: bool
    is_scene_boundary: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": round(self.time, 2),
            "interest": round(self.interest, 3),
            "motion": round(self.motion, 3),
            "audio_loudness": round(self.audio_loudness, 3),
            "clarity": round(self.clarity, 3),
            "sentiment": round(self.sentiment, 3),
            "has_laughter": self.has_laughter,
            "is_scene_boundary": self.is_scene_boundary,
        }


@dataclass
class NodeTimingResult:
    """Тайминг выполнения узла."""
    name: str
    execution_time_seconds: float
    layer: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "execution_time_seconds": round(self.execution_time_seconds, 3),
            "layer": self.layer,
        }


@dataclass
class AnalysisResultFull:
    """Полный результат анализа (для сохранения на диск)."""

    # Базовая информация
    public: AnalysisResultPublic

    # Артефакты
    artifacts: Dict[str, ArtifactRef] = field(default_factory=dict)

    # Детальные данные (опционально - могут быть большими)
    timeline_preview: List[TimelinePointResult] = field(default_factory=list)  # sampled
    transcript_segments: List[TranscriptSegment] = field(default_factory=list)
    scene_boundaries: List[float] = field(default_factory=list)

    # Диагностика
    node_timings: List[NodeTimingResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    settings_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Метаданные
    pipeline_version: str = "2.0"
    run_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "public": self.public.to_dict(),
            "artifacts": {k: v.to_dict() for k, v in self.artifacts.items()},
            "timeline_preview": [t.to_dict() for t in self.timeline_preview],
            "transcript_segments": [s.to_dict() for s in self.transcript_segments],
            "scene_boundaries": [round(b, 2) for b in self.scene_boundaries],
            "node_timings": [t.to_dict() for t in self.node_timings],
            "warnings": self.warnings,
            "settings_snapshot": self.settings_snapshot,
            "pipeline_version": self.pipeline_version,
            "run_id": self.run_id,
        }

    def to_public(self) -> AnalysisResultPublic:
        """Возвращает публичную часть."""
        return self.public
