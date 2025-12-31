from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Set
import time

from .common import VideoInfo, Scene, TopicSegment, AudioEvent
from .timeline import TimelinePoint
from .highlights import Highlight
from .chapters import Chapter


PipelineStatus = Literal["pending", "in_progress", "completed", "failed"]


@dataclass
class PipelineState:
    settings: Any                      # VideoServiceSettings или более строгий тип
    video_info: Optional[VideoInfo] = None

    # Сырые данные
    input_path: Optional[str] = None
    audio_path: Optional[str] = None

    motion_heatmap: list[dict] = field(default_factory=list)
    audio_features: dict[str, list[dict]] = field(default_factory=dict)
    audio_events: list[AudioEvent] = field(default_factory=list)

    scenes: list[Scene] = field(default_factory=list)
    scene_boundaries: list[float] = field(default_factory=list)

    transcript_segments: list[dict] = field(default_factory=list)
    full_transcript: Optional[str] = None

    sentiment_timeline: list[dict] = field(default_factory=list)
    humor_scores: list[dict] = field(default_factory=list)
    topic_segments: list[TopicSegment] = field(default_factory=list)

    # Speech quality metrics (NEW)
    speech_quality: Optional[dict] = None
    clarity_timeline: list[dict] = field(default_factory=list)

    timeline_points: list[TimelinePoint] = field(default_factory=list)
    highlights: list[Highlight] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)

    # summary/diagnostics
    summaries: dict[str, Any] = field(default_factory=dict)

    # DAG execution tracking (NEW)
    completed_stages: Set[str] = field(default_factory=set)
    current_stage: Optional[str] = None
    status: PipelineStatus = "pending"
    error: Optional[str] = None

    # Timing information (NEW)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    layer_timings: Dict[int, float] = field(default_factory=dict)
    node_timings: Dict[str, float] = field(default_factory=dict)

    # Pipeline mode (NEW)
    pipeline_mode: Literal["minimal", "full"] = "full"

    def mark_stage_completed(self, stage_name: str) -> None:
        self.completed_stages.add(stage_name)
        self.current_stage = None

    def mark_stage_started(self, stage_name: str) -> None:
        self.current_stage = stage_name

    def is_stage_completed(self, stage_name: str) -> bool:
        return stage_name in self.completed_stages

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.error = error
        self.end_time = time.time()

    def mark_completed(self) -> None:
        self.status = "completed"
        self.end_time = time.time()

    @property
    def total_processing_time(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
