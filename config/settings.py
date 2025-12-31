from __future__ import annotations

from typing import Any, TypedDict

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

from models.state import PipelineState


class VideoServiceSettings(BaseSettings):
    # Важно: путь к видео НЕ должен быть обязательным полем Settings,
    # иначе API/UI/Celery падают при создании settings из параметров запроса.
    # Источник пути: аргумент input_path в build_initial_* или env VIDEO_SERVICE_INPUT_VIDEO_PATH.
    input_video_path: str = Field("", description="Default input video path (optional).")

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

    # LLM settings
    llm_enabled: bool = Field(True)
    llm_base_url: str = Field("http://localhost:11434")
    llm_model: str = Field("qwen2.5-coder:14b")
    llm_timeout_seconds: float = Field(60.0)
    llm_retries: int = Field(2)
    llm_temperature: float = Field(0.0)
    llm_max_input_chars: int = Field(2000)

    # Block analysis settings
    block_analysis_enabled: bool = Field(True)
    block_analysis_max_blocks_per_request: int = Field(6)
    block_analysis_max_text_per_block: int = Field(500)

    # New feature flags (ALL ENABLED)
    enable_yamnet: bool = Field(True)
    enable_candidates: bool = Field(True)
    enable_llm_refine: bool = Field(True)

    # YAMNet / Laughter settings (снижено для русского контента)
    yamnet_threshold: float = Field(0.3)
    laughter_threshold: float = Field(0.3)

    # GPU / CUDA settings
    device: str = Field("cpu", description="Default device: cuda, cpu, auto")
    torch_device: str = Field("cpu", description="PyTorch device: cuda, cpu, mps")
    use_fp16: bool = Field(True, description="Use FP16 for faster inference on GPU")

    # STT improvements
    whisper_device: str = Field("cpu")  # auto, cpu, cuda

    # Candidate selection settings
    candidates_max_count: int = Field(50)
    candidates_min_duration: float = Field(10.0)
    candidates_max_duration: float = Field(120.0)
    candidates_interest_threshold: float = Field(0.6)
    candidates_laughter_threshold: float = Field(0.5)

    # LLM refine settings
    llm_refine_max_per_request: int = Field(5)
    llm_refine_max_retries: int = Field(3)
    llm_refine_backoff_seconds: float = Field(2.0)
    llm_refine_context_window_seconds: float = Field(15.0)

    # DAG Pipeline settings (NEW)
    enable_parallel: bool = Field(True, description="Enable parallel execution of independent handlers")
    num_workers: int = Field(8, description="Max parallel workers for DAG execution")
    pipeline_mode: str = Field("full", description="Pipeline mode: 'minimal' or 'full'")

    # Batch sizes for ML models (optimized for GPU)
    batch_size_stt: int = Field(64, description="Batch size for speech-to-text")
    batch_size_sentiment: int = Field(128, description="Batch size for sentiment analysis")
    batch_size_llm: int = Field(10, description="Batch size for LLM requests")

    # Enable viral moments detection (NEW)
    enable_viral_moments: bool = Field(True, description="Use viral moments detection instead of traditional highlights")

    # Speech quality settings (NEW)
    enable_speech_quality: bool = Field(True, description="Enable speech quality analysis")
    speech_quality_vad_threshold: float = Field(0.5, description="VAD threshold for speech quality")

    # Interest formula weights (NEW - согласно ТЗ)
    interest_weight_motion: float = Field(0.20)
    interest_weight_loudness: float = Field(0.18)
    interest_weight_energy: float = Field(0.15)
    interest_weight_sentiment: float = Field(0.15)
    interest_weight_clarity: float = Field(0.12)
    interest_weight_speech_prob: float = Field(0.10)
    interest_weight_loud_sound: float = Field(0.10)

    # Viral clips settings (NEW)
    min_clip_duration: float = Field(30.0, description="Minimum clip duration in seconds")
    max_clip_duration: float = Field(60.0, description="Maximum clip duration (1 minute)")
    min_gap_seconds: float = Field(15.0, description="Minimum gap between clips")
    max_viral_clips: int = Field(8, description="Maximum number of viral clips to select")

    # Clip scoring weights (NEW - согласно ТЗ)
    clip_weight_hook: float = Field(0.20)
    clip_weight_pace: float = Field(0.15)
    clip_weight_intensity: float = Field(0.15)
    clip_weight_clarity: float = Field(0.15)
    clip_weight_emotion: float = Field(0.15)
    clip_weight_boundary: float = Field(0.10)
    clip_weight_momentum: float = Field(0.10)

    model_config = SettingsConfigDict(
        env_prefix="VIDEO_SERVICE_",
        env_file="sample.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Legacy TypedDict для совместимости
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

    # Speech quality (NEW)
    speech_quality: dict[str, Any]
    clarity_timeline: list[dict[str, float]]

    timeline: list[TimelinePoint]
    highlights: list[Highlight]
    chapters: list[Chapter]

    processing_time_seconds: float

    # DAG execution tracking (NEW)
    completed_stages: set[str]
    layer_timings: dict[int, float]
    node_timings: dict[str, float]
    pipeline_mode: str


def build_initial_context(settings: VideoServiceSettings, input_path: str) -> Context:
    video_path = input_path or settings.input_video_path
    if not video_path:
        raise ValueError("input_path is required (or set VIDEO_SERVICE_INPUT_VIDEO_PATH)")
    return Context(input_path=video_path, video_path=video_path)


def build_initial_state(settings: VideoServiceSettings, input_path: str) -> PipelineState:
    """Новая функция для создания PipelineState."""
    video_path = input_path or settings.input_video_path
    if not video_path:
        raise ValueError("input_path is required (or set VIDEO_SERVICE_INPUT_VIDEO_PATH)")
    return PipelineState(
        settings=settings,
        input_path=video_path
    )
