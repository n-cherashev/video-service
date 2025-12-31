import argparse
import json
import time
from typing import Any

from config.settings import Context, VideoServiceSettings, build_initial_context
from core.pipeline import run_pipeline
from core.base_handler import BaseHandler

from models.serde import to_jsonable
from utils.truncation import truncate_large_lists

# Импорты хэндлеров из новой структуры
from features.file_io.read_file_handler import ReadFileHandler
from features.file_io.video_meta_handler import VideoMetaHandler
from features.file_io.ffmpeg_extract_handler import FFmpegExtractHandler
from features.motion.motion_analysis_frame_diff_handler import MotionAnalysisFrameDiffHandler
from features.audio.audio_features_handler import AudioFeaturesHandler
from features.audio.audio_events_handler import AudioEventsHandler
from features.scenes.scene_detection_handler import SceneDetectionHandler
from features.nlp.speech_to_text_handler import SpeechToTextHandler
from features.nlp.sentiment_analysis_handler import SentimentAnalysisHandler
from features.nlp.humor_detection_handler import HumorDetectionHandler
from features.nlp.topic_segmentation_handler import TopicSegmentationHandler
from features.fusion.fusion_timeline_handler import FusionTimelineHandler
from features.highlights.highlight_detection_handler import HighlightDetectionHandler
from features.chapters.chapter_builder_handler import ChapterBuilderHandler
from features.finalize.finalize_analysis_handler import FinalizeAnalysisHandler


def print_summary(context: Context) -> None:
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    motion = context.get("motion_summary")
    if motion:
        print("Motion Analysis:")
        print(f"   Mean motion:  {motion.get('mean', 0):.3f}")
        print(f"   Max motion:   {motion.get('max', 0):.3f}")
        print(f"   Seconds:      {motion.get('seconds', 0)}")

    fps = context.get("fps")
    if fps is not None:
        print(f"Video FPS:       {fps:.1f}")

    duration = context.get("duration_seconds")
    if duration:
        print(f"Duration:        {duration:.1f}s")

    total_time = context.get("processing_time_seconds")
    if total_time is not None:
        print(f"Total time:      {total_time:.2f}s")

    highlights = context.get("highlights") or []
    chapters = context.get("chapters") or []
    print(f"Highlights:      {len(highlights)}")
    print(f"Chapters:        {len(chapters)}")


def build_handlers(settings: VideoServiceSettings) -> list[BaseHandler]:
    handlers: list[BaseHandler] = [
        ReadFileHandler(),
        VideoMetaHandler(),
        FFmpegExtractHandler(),

        MotionAnalysisFrameDiffHandler(
            resize_width=settings.motion_resize_width,
            frame_step=settings.motion_frame_step,
        ),

        AudioFeaturesHandler(
            target_sr=settings.audio_target_sr,
            window_size_ms=settings.audio_window_size_ms,
            hop_size_ms=settings.audio_hop_size_ms,
        ),
        AudioEventsHandler(),

        SceneDetectionHandler(
            min_scene_duration=settings.min_scene_duration_sec,
            detector="adaptive",
            downscale=1,
            frame_skip=0,
            auto_downscale=False,
        )
    ]

    if settings.enable_stt:
        handlers.append(SpeechToTextHandler(model_name=settings.whisper_model_name))
    if settings.enable_sentiment:
        handlers.append(
            SentimentAnalysisHandler(strategy="model", model_name=settings.sentiment_model_name)
        )
    if settings.enable_humor:
        handlers.append(HumorDetectionHandler(model_name=settings.humor_model_name))

    handlers.extend([
        TopicSegmentationHandler(),
        FusionTimelineHandler(
            weight_motion=settings.weight_motion,
            weight_audio=settings.weight_audio,
            weight_sentiment=settings.weight_sentiment,
            weight_humor=settings.weight_humor,
        ),
        HighlightDetectionHandler(
            min_duration_seconds=settings.min_highlight_duration_sec,
            mode="top_interest",
            top_k=6,
            peak_quantile=0.90,
            snap_to_scenes=True,
            snap_window_seconds=5.0,
        ),
        ChapterBuilderHandler(),
        FinalizeAnalysisHandler(),
    ])
    return handlers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video analysis pipeline")
    parser.add_argument(
        "video_path",
        nargs="?",
        help="Путь к видеофайлу (если не указан — берётся из VIDEO_SERVICE_INPUT_VIDEO_PATH).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = VideoServiceSettings()

    print(f"DEBUG: input_video_path = '{settings.input_video_path}'")
    print(f"DEBUG: enable_stt = {settings.enable_stt}")
    print(f"DEBUG: whisper_model_name = '{settings.whisper_model_name}'")

    input_path = args.video_path or settings.input_video_path
    context: Context = build_initial_context(settings=settings, input_path=input_path)

    handlers = build_handlers(settings)

    print("Starting video processing pipeline...")
    print("-" * 50)

    start = time.monotonic()
    try:
        context = run_pipeline(context, handlers)
        context["processing_time_seconds"] = time.monotonic() - start
        print("\nPipeline completed successfully!")
    except Exception as exc:
        context["processing_time_seconds"] = time.monotonic() - start
        print(f"\nPipeline failed: {exc}")
        raise

    print_summary(context)

    print("\n" + "=" * 80)
    print("FULL RESULTS (JSON)")
    print("=" * 80)

    jsonable_context = to_jsonable(context)
    truncated_context = truncate_large_lists(jsonable_context)
    print(json.dumps(truncated_context, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
    
    