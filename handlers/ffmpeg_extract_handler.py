import os
import subprocess
from pathlib import Path
from typing import Any

from handlers.base_handler import BaseHandler


class FFmpegExtractHandler(BaseHandler):
    """Handler for extracting audio from video using FFmpeg."""

    def __init__(self, temp_dir: str = "temp") -> None:
        """
        Initialize FFmpeg handler.

        Args:
            temp_dir: Directory to store temporary audio files.
        """
        self.temp_dir = temp_dir

    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Extract audio from video file using FFmpeg.

        Args:
            context: Dictionary with 'video_path' key.

        Returns:
            Updated context with 'audio_path'.

        Raises:
            FileNotFoundError: If video file doesn't exist.
            RuntimeError: If FFmpeg extraction fails.
        """
        video_path = context.get("video_path")
        if not video_path:
            raise ValueError("'video_path' not provided in context")

        # Create temp directory if it doesn't exist
        Path(self.temp_dir).mkdir(exist_ok=True)

        # Generate output audio path
        video_name = Path(video_path).stem
        audio_path = os.path.join(self.temp_dir, f"{video_name}.wav")

        # Run FFmpeg command
        try:
            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-q:a",
                "9",
                "-n",  # Don't overwrite existing files
                audio_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found. Please install ffmpeg: brew install ffmpeg"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg extraction failed: {e.stderr.decode()}")

        context["audio_path"] = audio_path

        print(f"✓ Audio extracted: {audio_path}")

        return context
