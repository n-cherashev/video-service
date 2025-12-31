"""
Pipeline Keys - полный перечень ключей для requires/provides контрактов.

Каждый ключ имеет уникальное имя и ассоциированный тип.
Это позволяет DAGExecutor валидировать контракты и мерджить результаты безопасно.
"""
from __future__ import annotations

from enum import Enum
from typing import Final


class Key(str, Enum):
    """Все ключи состояния пайплайна."""

    # === Inputs (Extractors) ===
    INPUT_PATH = "input_path"
    VIDEO_PATH = "video_path"
    VIDEO_ARTIFACT = "video_artifact"
    VIDEO_META = "video_meta"
    AUDIO_PATH = "audio_path"
    AUDIO_ARTIFACT = "audio_artifact"

    # === Video Analysis Artifacts ===
    MOTION_SERIES = "motion_series"
    MOTION_SUMMARY = "motion_summary"
    SCENES = "scenes"
    SCENE_BOUNDARIES = "scene_boundaries"
    SCENE_SUMMARY = "scene_summary"

    # === Audio Analysis Artifacts ===
    AUDIO_SERIES = "audio_series"
    AUDIO_FEATURES_META = "audio_features_meta"
    AUDIO_EVENTS = "audio_events"
    LAUGHTER_SERIES = "laughter_series"
    LAUGHTER_SUMMARY = "laughter_summary"

    # === Speech Quality ===
    SPEECH_QUALITY = "speech_quality"
    CLARITY_SERIES = "clarity_series"

    # === NLP Artifacts ===
    TRANSCRIPT = "transcript"
    TRANSCRIPT_SEGMENTS = "transcript_segments"
    FULL_TRANSCRIPT = "full_transcript"
    LANGUAGE = "language"

    # === Sentiment/Emotion ===
    SENTIMENT_SERIES = "sentiment_series"
    SENTIMENT_SUMMARY = "sentiment_summary"
    HUMOR_SERIES = "humor_series"
    HUMOR_SUMMARY = "humor_summary"

    # === Topics/Blocks ===
    TOPIC_SEGMENTS = "topic_segments"
    BLOCK_ANALYSIS = "block_analysis"

    # === Fusion ===
    TIMELINE = "timeline"
    TIMELINE_SUMMARY = "timeline_summary"

    # === Candidates/Highlights ===
    ANCHORS = "anchors"
    ANCHOR_SUMMARY = "anchor_summary"
    VIRAL_CANDIDATES = "viral_candidates"
    VIRAL_CLIPS = "viral_clips"
    HIGHLIGHTS = "highlights"
    REFINED_CANDIDATES = "refined_candidates"

    # === Chapters ===
    CHAPTERS = "chapters"

    # === Final Results ===
    ANALYSIS_RESULT = "analysis_result"

    # === Metadata/Metrics (mergeable) ===
    DURATION_SECONDS = "duration_seconds"
    FPS = "fps"
    FRAME_COUNT = "frame_count"
    PROCESSING_TIME = "processing_time_seconds"
    COMPLETED_STAGES = "completed_stages"
    LAYER_TIMINGS = "layer_timings"
    NODE_TIMINGS = "node_timings"
    WARNINGS = "warnings"


# Ключи, которые можно мерджить (не конфликтуют)
MERGEABLE_KEYS: Final[frozenset[Key]] = frozenset({
    Key.COMPLETED_STAGES,  # union set
    Key.LAYER_TIMINGS,     # merge dict
    Key.NODE_TIMINGS,      # merge dict
    Key.WARNINGS,          # concat list
})


# Ключи с большими данными - должны храниться как артефакты
ARTIFACT_KEYS: Final[frozenset[Key]] = frozenset({
    Key.MOTION_SERIES,
    Key.AUDIO_SERIES,
    Key.CLARITY_SERIES,
    Key.LAUGHTER_SERIES,
    Key.SENTIMENT_SERIES,
    Key.HUMOR_SERIES,
    Key.TIMELINE,
})
