import json
from typing import Any, Dict, List

from handlers.audio_features_handler import AudioFeaturesHandler
from handlers.base_handler import BaseHandler
from handlers.ffmpeg_extract_handler import FFmpegExtractHandler
from handlers.motion_analysis_frame_diff_handler import (
    MotionAnalysisFrameDiffHandler,
)
from handlers.motion_analysis_optical_flow_handler import (
    MotionAnalysisOpticalFlowHandler,
)
from handlers.read_file_handler import ReadFileHandler
from handlers.speech_to_text_handler import SpeechToTextHandler
from handlers.sentiment_analysis_handler import SentimentAnalysisHandler


def format_json(data: Any, indent: int = 2, max_items: int = 10) -> str:
    """Форматирует большие списки, показывая первые и последние элементы."""
    if isinstance(data, list) and len(data) > max_items:
        first = data[: max_items // 2]
        last = data[-max_items // 2 :]
        truncated = {
            "truncated": True,
            "total_length": len(data),
            "first_items": first,
            "last_items": last,
        }
        return json.dumps(truncated, indent=indent, ensure_ascii=False)
    return json.dumps(data, indent=indent, ensure_ascii=False)


def truncate_large_lists(obj: Any, max_items: int = 10) -> Any:
    """Рекурсивно обрезает большие списки в объекте."""
    if isinstance(obj, list):
        if len(obj) > max_items:
            first = obj[: max_items // 2]
            last = obj[-max_items // 2 :]
            return {
                "truncated": True,
                "total_length": len(obj),
                "first_items": first,
                "last_items": last,
            }
        return [truncate_large_lists(item, max_items) for item in obj]
    elif isinstance(obj, dict):
        return {k: truncate_large_lists(v, max_items) for k, v in obj.items()}
    return obj


def print_summary(context: Dict[str, Any]) -> None:
    """Выводит сводную информацию."""
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)

    if "motion_summary" in context:
        summary = context["motion_summary"]
        print("🎬 Motion Analysis:")
        print(f"   Mean motion:  {summary['mean']:.3f}")
        print(f"   Max motion:   {summary['max']:.3f}")
        print(f"   Seconds:      {summary['seconds']}")

    if "fps" in context:
        print(f"📹 Video FPS:     {context['fps']:.1f}")

    if "duration_seconds" in context and context["duration_seconds"]:
        print(f"⏱️  Duration:      {context['duration_seconds']:.1f}s")

    if "processing_time_seconds" in context:
        print(f"⚡ Total time:    {context.get('processing_time_seconds', 0):.2f}s")


def main() -> None:
    """Run the video processing pipeline."""
    video_path = "/Users/nikolajcerasev/Projects/video-service/videos/mstiteli.mp4"

    # Initialize context
    context: Dict[str, Any] = {"input_path": video_path}

    # Create handlers list
    handlers: List[BaseHandler] = [
        ReadFileHandler(),
        FFmpegExtractHandler(),
        MotionAnalysisFrameDiffHandler(),
        MotionAnalysisOpticalFlowHandler(),
        AudioFeaturesHandler(),
        SpeechToTextHandler(),
        SentimentAnalysisHandler(),
    ]

    # Run pipeline
    print("🚀 Starting video processing pipeline...")
    print("-" * 50)

    try:
        for i, handler in enumerate(handlers, 1):
            handler_name = handler.__class__.__name__
            print(f"[{i:2d}/{len(handlers)}] 🔄 {handler_name}")
            context = handler.handle(context)

        print("\n✅ Pipeline completed successfully!")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return

    # Print summary
    print_summary(context)

    # Print full JSON (сокращенный для больших списков)
    print("\n" + "=" * 80)
    print("📋 FULL RESULTS (JSON)")
    print("=" * 80)
    truncated_context = truncate_large_lists(context)
    print(json.dumps(truncated_context, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
