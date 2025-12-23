from __future__ import annotations

from typing import Any, TypedDict

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class VideoServiceSettings(BaseSettings):
    input_video_path: str = Field(..., description="Default input video path.")

    motion_resize_width: int = Field(320)
    motion_frame_step: int = Field(1)

    min_scene_duration_sec: float = Field(10.0)
    min_highlight_duration_sec: float = Field(30.0)

    audio_target_sr: int = Field(16000)
    audio_window_size_ms: int = Field(200)
    audio_hop_size_ms: int = Field(100)

    enable_stt: bool = Field(True)
    enable_sentiment: bool = Field(True)
    enable_humor: bool = Field(True)

    whisper_model_name: str = Field("base")
    sentiment_model_name: str = Field("distilbert-base-uncased-finetuned-sst-2-english")
    humor_model_name: str = Field("humor-classifier-mini")

    weight_motion: float = Field(0.4)
    weight_audio: float = Field(0.3)
    weight_sentiment: float = Field(0.2)
    weight_humor: float = Field(0.1)

    model_config = SettingsConfigDict(
        env_prefix="VIDEO_SERVICE_",
        env_file="sample.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class TimelinePoint(TypedDict):
    time: float
    interest: float
    motion: float | None
    audio_loudness: float | None
    sentiment: float | None
    humor: float | None
    has_laughter: bool | None
    has_loud_sound: bool | None
    is_scene_boundary: bool | None
    is_dialogue: bool | None


class Highlight(TypedDict):
    start: float
    end: float
    type: str
    score: float


class Chapter(TypedDict):
    start: float
    end: float
    title: str
    description: str


class Context(TypedDict, total=False):
    input_path: str
    video_path: str
    audio_path: str

    fps: float
    frame_count: int | None
    duration_seconds: float | None

    motion_heatmap: list[dict[str, float]]
    motion_summary: dict[str, Any]
    motion_detection_method: str
    motion_processing_time_seconds: float

    scenes: list[dict[str, Any]]
    scene_boundaries: list[float]
    scene_summary: dict[str, Any]

    audio_features: dict[str, Any]
    audio_features_meta: dict[str, Any]
    audio_events: list[dict[str, Any]]

    transcript_segments: list[dict[str, Any]]
    full_transcript: str

    sentiment_timeline: list[dict[str, float]]
    sentiment_summary: dict[str, Any]

    humor_scores: list[dict[str, float]]
    humor_summary: dict[str, Any]

    topic_segments: list[dict[str, Any]]

    timeline: list[TimelinePoint]
    highlights: list[Highlight]
    chapters: list[Chapter]

    processing_time_seconds: float


def build_initial_context(settings: VideoServiceSettings, input_path: str) -> Context:
    video_path = input_path or settings.input_video_path
    return Context(input_path=video_path, video_path=video_path)
