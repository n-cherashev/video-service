"""Video processing handlers package."""

from handlers.audio_features_handler import AudioFeaturesHandler
from handlers.base_handler import BaseHandler
from handlers.ffmpeg_extract_handler import FFmpegExtractHandler
from handlers.motion_analysis_background_sub_handler import (
    MotionAnalysisBackgroundSubHandler,
)
from handlers.motion_analysis_frame_diff_handler import MotionAnalysisFrameDiffHandler
from handlers.motion_analysis_optical_flow_handler import (
    MotionAnalysisOpticalFlowHandler,
)
from handlers.read_file_handler import ReadFileHandler
from handlers.sentiment_analysis_handler import SentimentAnalysisHandler
from handlers.speech_to_text_handler import SpeechToTextHandler
from handlers.humor_detection_handler import HumorDetectionHandler
from handlers.topic_segmentation_handler import TopicSegmentationHandler
from handlers.audio_events_handler import AudioEventsHandler
from handlers.scene_change_handler import SceneChangeHandler
from handlers.fusion_timeline_handler import FusionTimelineHandler
from handlers.highlight_detection_handler import HighlightDetectionHandler
from handlers.chapter_builder_handler import ChapterBuilderHandler
from handlers.finalize_analysis_handler import FinalizeAnalysisHandler
from handlers.shot_boundary_handler import ShotBoundaryHandler
from handlers.scene_grouping_handler import SceneGroupingHandler

__all__ = [
    "BaseHandler",
    "ReadFileHandler",
    "FFmpegExtractHandler",
    "AudioFeaturesHandler",
    "MotionAnalysisFrameDiffHandler",
    "MotionAnalysisBackgroundSubHandler",
    "MotionAnalysisOpticalFlowHandler",
    "SpeechToTextHandler",
    "SentimentAnalysisHandler",
    "HumorDetectionHandler",
    "TopicSegmentationHandler",
    "AudioEventsHandler",
    "SceneChangeHandler",
    "ShotBoundaryHandler",
    "SceneGroupingHandler",
    "FusionTimelineHandler",
    "HighlightDetectionHandler",
    "ChapterBuilderHandler",
    "FinalizeAnalysisHandler",
]
