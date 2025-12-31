"""
Models package for video service.

Экспортирует все типизированные модели для пайплайна:
- Keys: перечень ключей для контрактов
- Artifacts: модели артефактов (ArtifactRef, VideoArtifact, etc.)
- Contracts: NodePatch, StateView, HandlerContract
- Pipeline State: PipelineStateV2, PipelineMetrics
- Results: AnalysisResultPublic, AnalysisResultFull
- Candidates: Anchor, CandidateWindow, ScoredClipV2
- Common: VideoInfo, Scene, Chapter, TimelinePoint, etc.
"""

# Keys
from .keys import Key, MERGEABLE_KEYS, ARTIFACT_KEYS

# Artifacts
from .artifacts import (
    ArtifactKind,
    ArtifactRef,
    VideoArtifact,
    AudioArtifact,
    VideoMeta,
    SeriesArtifact,
    TranscriptArtifact,
)

# Contracts
from .contracts import (
    HandlerContract,
    NodePatch,
    StateView,
)

# Pipeline State
from .pipeline_state import (
    PipelineStateV2,
    PipelineMetrics,
    PipelineStatusV2,
)

# Results
from .results import (
    TranscriptSegment,
    ScoreBreakdown,
    ViralClipResult,
    ChapterResult,
    AnalysisSummary,
    AnalysisResultPublic,
    AnalysisResultFull,
    TimelinePointResult,
    NodeTimingResult,
)

# Candidates
from .candidates import (
    AnchorType,
    Anchor,
    CandidateWindow,
    ScoreComponents,
    ScoredClipV2,
    AnchorSummary,
)

# Legacy/Common models (обратная совместимость)
from .common import VideoInfo, Scene, TopicSegment, AudioEvent
from .timeline import TimelinePoint
from .highlights import Highlight, HighlightType
from .chapters import Chapter
from .state import PipelineState, PipelineStatus
from .viral_moments import Candidate, ScoredClip, ViralMomentsConfig
from .serde import to_jsonable

__all__ = [
    # Keys
    "Key",
    "MERGEABLE_KEYS",
    "ARTIFACT_KEYS",
    # Artifacts
    "ArtifactKind",
    "ArtifactRef",
    "VideoArtifact",
    "AudioArtifact",
    "VideoMeta",
    "SeriesArtifact",
    "TranscriptArtifact",
    # Contracts
    "HandlerContract",
    "NodePatch",
    "StateView",
    # Pipeline State V2
    "PipelineStateV2",
    "PipelineMetrics",
    "PipelineStatusV2",
    # Results
    "TranscriptSegment",
    "ScoreBreakdown",
    "ViralClipResult",
    "ChapterResult",
    "AnalysisSummary",
    "AnalysisResultPublic",
    "AnalysisResultFull",
    "TimelinePointResult",
    "NodeTimingResult",
    # Candidates
    "AnchorType",
    "Anchor",
    "CandidateWindow",
    "ScoreComponents",
    "ScoredClipV2",
    "AnchorSummary",
    # Legacy/Common
    "VideoInfo",
    "Scene",
    "TopicSegment",
    "AudioEvent",
    "TimelinePoint",
    "Highlight",
    "HighlightType",
    "Chapter",
    "PipelineState",
    "PipelineStatus",
    "Candidate",
    "ScoredClip",
    "ViralMomentsConfig",
    "to_jsonable",
]
