import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Callable

from config.settings import Context, VideoServiceSettings, build_initial_context
from core.pipeline import run_pipeline
from core.dag_executor import DAGExecutor, DAGNode, ExecutionResult
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
from features.nlp.block_analysis_handler import BlockAnalysisHandler
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

    # DAG timing info
    layer_timings = context.get("layer_timings")
    if layer_timings:
        print("\nLayer timings:")
        for layer_idx, timing in layer_timings.items():
            print(f"   Layer {layer_idx}: {timing:.2f}s")


def print_execution_result(result: ExecutionResult) -> None:
    """Печатает результаты DAG выполнения."""
    print("\n" + "-" * 50)
    print("DAG Execution Summary:")
    print(f"  Completed: {len(result.completed_nodes)} nodes")
    print(f"  Failed: {len(result.failed_nodes)} nodes")
    print(f"  Skipped: {len(result.skipped_nodes)} nodes")
    print(f"  Total time: {result.total_time:.2f}s")

    if result.layer_timings:
        print("\n  Layer timings:")
        for layer_idx, timing in result.layer_timings.items():
            print(f"    Layer {layer_idx + 1}: {timing:.2f}s")


def build_handlers_linear(settings: VideoServiceSettings) -> list[BaseHandler]:
    """Строит линейный список хэндлеров (legacy)."""
    from llm.settings import OllamaSettings
    from llm.ollama_client import OllamaLLMClient
    from llm.null_client import NullLLMClient

    llm_settings = OllamaSettings(
        enabled=settings.llm_enabled,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
        retries=settings.llm_retries,
        options={"temperature": settings.llm_temperature},
        max_input_chars=settings.llm_max_input_chars,
    )

    llm_client = OllamaLLMClient(llm_settings) if llm_settings.enabled else NullLLMClient()

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
    ]

    # Speech quality handler (NEW)
    if settings.enable_speech_quality:
        from features.audio.speech_quality_handler import SpeechQualityHandler
        handlers.append(SpeechQualityHandler(
            vad_threshold=settings.speech_quality_vad_threshold
        ))

    # YAMNet laughter detection
    if settings.enable_yamnet:
        from features.audio.laughter_detection_yamnet_handler import LaughterDetectionYamnetHandler
        handlers.append(LaughterDetectionYamnetHandler(threshold=settings.yamnet_threshold))

    handlers.append(
        SceneDetectionHandler(
            min_scene_duration=settings.min_scene_duration_sec,
            detector="adaptive",
            downscale=1,
            frame_skip=0,
            auto_downscale=False,
        )
    )

    if settings.enable_stt:
        handlers.append(
            SpeechToTextHandler(
                model_name=settings.whisper_model_name,
                device=settings.whisper_device
            )
        )
    if settings.enable_sentiment:
        handlers.append(
            SentimentAnalysisHandler(strategy="model", model_name=settings.sentiment_model_name)
        )
    if settings.enable_humor:
        handlers.append(HumorDetectionHandler(threshold=0.5))

    # LLM block analysis (Ollama): темы + юмор одним запросом
    if settings.llm_enabled and settings.block_analysis_enabled and settings.enable_stt:
        handlers.append(
            BlockAnalysisHandler(
                llm_client=llm_client,
                max_blocks_per_request=settings.block_analysis_max_blocks_per_request,
                max_text_per_block=settings.block_analysis_max_text_per_block,
            )
        )

    # FusionTimeline с новыми весами
    handlers.append(
        FusionTimelineHandler(
            step=1.0,
            weight_motion=settings.interest_weight_motion,
            weight_loudness=settings.interest_weight_loudness,
            weight_energy=settings.interest_weight_energy,
            weight_sentiment=settings.interest_weight_sentiment,
            weight_clarity=settings.interest_weight_clarity,
            weight_speech_prob=settings.interest_weight_speech_prob,
            weight_loud_sound=settings.interest_weight_loud_sound,
        )
    )

    # Highlights detection
    if settings.enable_viral_moments:
        from features.fusion.candidate_selection_handler import CandidateSelectionHandler
        from features.highlights.viral_moments_handler import ViralMomentsHandler
        from models.viral_moments import ViralMomentsConfig

        viral_config = ViralMomentsConfig()

        handlers.extend([
            CandidateSelectionHandler(config=viral_config),
            ViralMomentsHandler(config=viral_config),
        ])
    elif settings.enable_candidates and settings.enable_llm_refine:
        from features.fusion.candidate_selection_handler import CandidateSelectionHandler
        from features.nlp.llm_refine_candidates_handler import LLMRefineCandidatesHandler

        handlers.extend([
            CandidateSelectionHandler(),
            LLMRefineCandidatesHandler(
                llm_client=llm_client,
                max_candidates_per_request=settings.llm_refine_max_per_request,
                max_retries=settings.llm_refine_max_retries,
                backoff_base_seconds=settings.llm_refine_backoff_seconds,
                context_window_seconds=settings.llm_refine_context_window_seconds,
            ),
        ])
    else:
        # Multi-score highlights с новыми весами
        handlers.append(
            HighlightDetectionHandler(
                min_duration_seconds=settings.min_highlight_duration_sec,
                mode="multi_score",
                top_k=6,
                peak_quantile=0.90,
                snap_to_scenes=True,
                snap_window_seconds=5.0,
                weight_hook=settings.clip_weight_hook,
                weight_pace=settings.clip_weight_pace,
                weight_intensity=settings.clip_weight_intensity,
                weight_clarity=settings.clip_weight_clarity,
                weight_emotion=settings.clip_weight_emotion,
                weight_boundary=settings.clip_weight_boundary,
                weight_momentum=settings.clip_weight_momentum,
            )
        )

    # Chapter builder
    handlers.append(ChapterBuilderHandler())

    return handlers


def build_dag_nodes(settings: VideoServiceSettings) -> List[DAGNode]:
    """Строит DAG-узлы с зависимостями для параллельного выполнения."""
    from llm.settings import OllamaSettings
    from llm.ollama_client import OllamaLLMClient
    from llm.null_client import NullLLMClient

    llm_settings = OllamaSettings(
        enabled=settings.llm_enabled,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
        retries=settings.llm_retries,
        options={"temperature": settings.llm_temperature},
        max_input_chars=settings.llm_max_input_chars,
    )

    llm_client = OllamaLLMClient(llm_settings) if llm_settings.enabled else NullLLMClient()

    nodes: List[DAGNode] = []

    # Layer 1: Initialization
    nodes.append(DAGNode(
        handler=ReadFileHandler(),
        dependencies=[],
    ))
    nodes.append(DAGNode(
        handler=VideoMetaHandler(),
        dependencies=["ReadFileHandler"],
    ))
    nodes.append(DAGNode(
        handler=FFmpegExtractHandler(),
        dependencies=["VideoMetaHandler"],
    ))

    # Layer 2: Fast Features (параллельно)
    nodes.append(DAGNode(
        handler=MotionAnalysisFrameDiffHandler(
            resize_width=settings.motion_resize_width,
            frame_step=settings.motion_frame_step,
        ),
        dependencies=["VideoMetaHandler"],
    ))

    nodes.append(DAGNode(
        handler=AudioFeaturesHandler(
            target_sr=settings.audio_target_sr,
            window_size_ms=settings.audio_window_size_ms,
            hop_size_ms=settings.audio_hop_size_ms,
        ),
        dependencies=["FFmpegExtractHandler"],
    ))

    nodes.append(DAGNode(
        handler=AudioEventsHandler(),
        dependencies=["AudioFeaturesHandler"],
    ))

    # Speech quality (после audio features)
    if settings.enable_speech_quality:
        from features.audio.speech_quality_handler import SpeechQualityHandler
        nodes.append(DAGNode(
            handler=SpeechQualityHandler(vad_threshold=settings.speech_quality_vad_threshold),
            dependencies=["AudioFeaturesHandler"],
        ))

    # YAMNet (опционально)
    if settings.enable_yamnet:
        from features.audio.laughter_detection_yamnet_handler import LaughterDetectionYamnetHandler
        nodes.append(DAGNode(
            handler=LaughterDetectionYamnetHandler(threshold=settings.yamnet_threshold),
            dependencies=["FFmpegExtractHandler"],
        ))

    nodes.append(DAGNode(
        handler=SceneDetectionHandler(
            min_scene_duration=settings.min_scene_duration_sec,
            detector="adaptive",
            downscale=1,
            frame_skip=0,
            auto_downscale=False,
        ),
        dependencies=["VideoMetaHandler"],
    ))

    # Layer 3: NLP Features
    if settings.enable_stt:
        nodes.append(DAGNode(
            handler=SpeechToTextHandler(
                model_name=settings.whisper_model_name,
                device=settings.whisper_device
            ),
            dependencies=["FFmpegExtractHandler"],
        ))

        if settings.enable_sentiment:
            nodes.append(DAGNode(
                handler=SentimentAnalysisHandler(strategy="model", model_name=settings.sentiment_model_name),
                dependencies=["SpeechToTextHandler"],
            ))

        if settings.enable_humor:
            nodes.append(DAGNode(
                handler=HumorDetectionHandler(threshold=0.5),
                dependencies=["SpeechToTextHandler"],
            ))

        # LLM block analysis (Ollama): темы + юмор (внешняя Ollama)
        if settings.llm_enabled and settings.block_analysis_enabled:
            nodes.append(DAGNode(
                handler=BlockAnalysisHandler(
                    llm_client=llm_client,
                    max_blocks_per_request=settings.block_analysis_max_blocks_per_request,
                    max_text_per_block=settings.block_analysis_max_text_per_block,
                ),
                dependencies=["SpeechToTextHandler", "VideoMetaHandler"],
            ))

    # Layer 4: Fusion
    fusion_deps = [
        "MotionAnalysisFrameDiffHandler",
        "AudioFeaturesHandler",
        "AudioEventsHandler",
        "SceneDetectionHandler",
    ]
    if settings.enable_stt:
        fusion_deps.append("SpeechToTextHandler")
        if settings.enable_sentiment:
            fusion_deps.append("SentimentAnalysisHandler")
        if settings.enable_humor:
            fusion_deps.append("HumorDetectionHandler")
    if settings.enable_speech_quality:
        fusion_deps.append("SpeechQualityHandler")

    nodes.append(DAGNode(
        handler=FusionTimelineHandler(
            step=1.0,
            weight_motion=settings.interest_weight_motion,
            weight_loudness=settings.interest_weight_loudness,
            weight_energy=settings.interest_weight_energy,
            weight_sentiment=settings.interest_weight_sentiment,
            weight_clarity=settings.interest_weight_clarity,
            weight_speech_prob=settings.interest_weight_speech_prob,
            weight_loud_sound=settings.interest_weight_loud_sound,
        ),
        dependencies=fusion_deps,
    ))

    # Layer 5: Highlights
    if settings.enable_viral_moments:
        from features.fusion.candidate_selection_handler import CandidateSelectionHandler
        from features.highlights.viral_moments_handler import ViralMomentsHandler
        from models.viral_moments import ViralMomentsConfig

        viral_config = ViralMomentsConfig()

        nodes.append(DAGNode(
            handler=CandidateSelectionHandler(config=viral_config),
            dependencies=["FusionTimelineHandler"],
        ))
        nodes.append(DAGNode(
            handler=ViralMomentsHandler(config=viral_config),
            dependencies=["CandidateSelectionHandler"],
        ))
        highlight_dep = "ViralMomentsHandler"
    else:
        nodes.append(DAGNode(
            handler=HighlightDetectionHandler(
                min_duration_seconds=settings.min_highlight_duration_sec,
                mode="multi_score",
                top_k=6,
                peak_quantile=0.90,
                snap_to_scenes=True,
                snap_window_seconds=5.0,
                weight_hook=settings.clip_weight_hook,
                weight_pace=settings.clip_weight_pace,
                weight_intensity=settings.clip_weight_intensity,
                weight_clarity=settings.clip_weight_clarity,
                weight_emotion=settings.clip_weight_emotion,
                weight_boundary=settings.clip_weight_boundary,
                weight_momentum=settings.clip_weight_momentum,
            ),
            dependencies=["FusionTimelineHandler"],
        ))
        highlight_dep = "HighlightDetectionHandler"

    # Layer 6: Chapters
    nodes.append(DAGNode(
        handler=ChapterBuilderHandler(),
        dependencies=[highlight_dep, "SceneDetectionHandler"],
    ))

    return nodes


def build_minimal_dag_nodes(settings: VideoServiceSettings) -> List[DAGNode]:
    """Строит минимальный DAG для быстрого тестирования."""
    nodes: List[DAGNode] = []

    # Layer 1: Initialization
    nodes.append(DAGNode(handler=ReadFileHandler(), dependencies=[]))
    nodes.append(DAGNode(handler=VideoMetaHandler(), dependencies=["ReadFileHandler"]))
    nodes.append(DAGNode(handler=FFmpegExtractHandler(), dependencies=["VideoMetaHandler"]))

    # Layer 2: Minimal fast features
    nodes.append(DAGNode(
        handler=MotionAnalysisFrameDiffHandler(
            resize_width=settings.motion_resize_width,
            frame_step=settings.motion_frame_step,
        ),
        dependencies=["VideoMetaHandler"],
    ))

    nodes.append(DAGNode(
        handler=AudioFeaturesHandler(
            target_sr=settings.audio_target_sr,
            window_size_ms=settings.audio_window_size_ms,
            hop_size_ms=settings.audio_hop_size_ms,
        ),
        dependencies=["FFmpegExtractHandler"],
    ))

    nodes.append(DAGNode(
        handler=SceneDetectionHandler(
            min_scene_duration=settings.min_scene_duration_sec,
            detector="adaptive",
        ),
        dependencies=["VideoMetaHandler"],
    ))

    # Layer 4: Fusion
    nodes.append(DAGNode(
        handler=FusionTimelineHandler(step=1.0),
        dependencies=["MotionAnalysisFrameDiffHandler", "AudioFeaturesHandler", "SceneDetectionHandler"],
    ))

    # Layer 5: Highlights
    nodes.append(DAGNode(
        handler=HighlightDetectionHandler(
            min_duration_seconds=settings.min_highlight_duration_sec,
            mode="multi_score",
            top_k=6,
        ),
        dependencies=["FusionTimelineHandler"],
    ))

    return nodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video analysis pipeline")
    parser.add_argument(
        "video_path",
        nargs="?",
        help="Путь к видеофайлу (если не указан — берётся из VIDEO_SERVICE_INPUT_VIDEO_PATH).",
    )
    parser.add_argument(
        "--mode",
        choices=["linear", "dag", "minimal"],
        default="dag",
        help="Режим выполнения: linear (последовательный), dag (параллельный), minimal (быстрый тест)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Не выводить полный JSON результат",
    )
    return parser.parse_args()


def run_dag_pipeline(
    context: Dict[str, Any],
    settings: VideoServiceSettings,
    *,
    mode: str = "dag",
    print_plan: bool = False,
    progress_callback: "Callable[[float, str], None] | None" = None,
) -> tuple[Dict[str, Any], ExecutionResult | None]:
    """Запускает пайплайн в DAG/linear/minimal режимах (для API/UI/Celery).

    Важно: это тонкий wrapper вокруг текущей реализации из `main()`,
    чтобы импорт `run_dag_pipeline` работал стабильно.
    """
    start = time.monotonic()
    execution_result: ExecutionResult | None = None

    if mode == "linear" or not settings.enable_parallel:
        handlers = build_handlers_linear(settings)
        if progress_callback:
            progress_callback(0.25, "Запуск (linear)")
        context = run_pipeline(context, handlers)
    elif mode == "minimal":
        nodes = build_minimal_dag_nodes(settings)
        completed = 0
        total = max(1, len(nodes))
        base = 0.2
        span = 0.7

        def on_node_start(name: str) -> None:
            if progress_callback:
                progress_callback(base + span * (completed / total), f"▶ {name}")

        def on_node_complete(name: str, duration: float) -> None:
            nonlocal completed
            completed += 1
            if progress_callback:
                progress_callback(base + span * (completed / total), f"✓ {name} ({duration:.1f}s)")

        executor = DAGExecutor(
            nodes,
            max_workers=settings.num_workers,
            on_node_start=on_node_start,
            on_node_complete=on_node_complete,
        )
        if print_plan:
            executor.print_execution_plan()
        context, execution_result = executor.execute(context)
    else:
        nodes = build_dag_nodes(settings)
        completed = 0
        total = max(1, len(nodes))
        base = 0.2
        span = 0.7

        def on_node_start(name: str) -> None:
            if progress_callback:
                progress_callback(base + span * (completed / total), f"▶ {name}")

        def on_node_complete(name: str, duration: float) -> None:
            nonlocal completed
            completed += 1
            if progress_callback:
                progress_callback(base + span * (completed / total), f"✓ {name} ({duration:.1f}s)")

        executor = DAGExecutor(
            nodes,
            max_workers=settings.num_workers,
            on_node_start=on_node_start,
            on_node_complete=on_node_complete,
        )
        if print_plan:
            executor.print_execution_plan()
        context, execution_result = executor.execute(context)

    context["processing_time_seconds"] = time.monotonic() - start
    return context, execution_result


def main() -> None:
    args = parse_args()
    settings = VideoServiceSettings()

    print(f"DEBUG: input_video_path = '{settings.input_video_path}'")
    print(f"DEBUG: enable_stt = {settings.enable_stt}")
    print(f"DEBUG: whisper_model_name = '{settings.whisper_model_name}'")
    print(f"DEBUG: pipeline_mode = '{args.mode}'")
    print(f"DEBUG: enable_parallel = {settings.enable_parallel}")

    input_path = args.video_path or settings.input_video_path
    context: Context = build_initial_context(settings=settings, input_path=input_path)
    context["pipeline_mode"] = args.mode

    print(f"\nStarting video processing pipeline ({args.mode} mode)...")
    print("-" * 50)

    start = time.monotonic()
    execution_result = None

    try:
        if args.mode == "linear" or not settings.enable_parallel:
            # Legacy линейный режим
            handlers = build_handlers_linear(settings)
            context = run_pipeline(context, handlers)
        elif args.mode == "minimal":
            # Минимальный DAG для тестирования
            nodes = build_minimal_dag_nodes(settings)
            executor = DAGExecutor(nodes, max_workers=settings.num_workers)
            executor.print_execution_plan()
            context, execution_result = executor.execute(context)
        else:
            # Полный DAG с параллелизмом
            nodes = build_dag_nodes(settings)
            executor = DAGExecutor(nodes, max_workers=settings.num_workers)
            executor.print_execution_plan()
            context, execution_result = executor.execute(context)

        context["processing_time_seconds"] = time.monotonic() - start
        print("\nPipeline completed successfully!")

    except Exception as exc:
        context["processing_time_seconds"] = time.monotonic() - start
        print(f"\nPipeline failed: {exc}")
        raise

    # Print execution result for DAG mode
    if execution_result:
        print_execution_result(execution_result)

    print_summary(context)

    if not args.no_json:
        # Save JSON to results folder
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Generate filename from video name + timestamp
        video_name = Path(context.get("video_path", "output")).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = results_dir / f"{video_name}_{timestamp}.json"

        jsonable_context = to_jsonable(context)
        truncated_context = truncate_large_lists(jsonable_context)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(truncated_context, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
