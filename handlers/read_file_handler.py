import os
from pathlib import Path
from typing import Any

from handlers.base_handler import BaseHandler


class ReadFileHandler(BaseHandler):
    """Handler for reading and validating video file."""

    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Read and validate video file from context.

        Args:
            context: Dictionary with 'input_path' key containing path to video file.

        Returns:
            Updated context with 'video_path' and 'video_size_bytes'.

        Raises:
            FileNotFoundError: If file doesn't exist.
            IsADirectoryError: If path is a directory, not a file.
        """
        input_path = context.get("input_path")
        if not input_path:
            raise ValueError("'input_path' not provided in context")

        # Normalize path
        video_path = str(Path(input_path).resolve())

        # Check if file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Check if it's a file, not a directory
        if not os.path.isfile(video_path):
            raise IsADirectoryError(f"Path is not a file: {video_path}")

        # Get file size
        video_size_bytes = os.path.getsize(video_path)

        context["video_path"] = video_path
        context["video_size_bytes"] = video_size_bytes

        print(f"✓ File read: {video_path} ({video_size_bytes} bytes)")

        return context
