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
]
