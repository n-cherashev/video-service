#!/usr/bin/env python3
"""
Main entry point for video processing pipeline.

Usage:
    python main.py
"""

from typing import Any

from handlers.base_handler import BaseHandler
from handlers.ffmpeg_extract_handler import FFmpegExtractHandler
from handlers.read_file_handler import ReadFileHandler


def main() -> None:
    """Run the video processing pipeline."""
    video_path = (
        "/Users/nikolajcerasev/Projects/video-service/videos/2025-12-11 18-15-08.mkv"
    )

    # Initialize context
    context: dict[str, Any] = {"input_path": video_path}

    # Create handlers list
    handlers: list[BaseHandler] = [
        ReadFileHandler(),
        FFmpegExtractHandler(),
    ]

    # Run pipeline
    print("Starting video processing pipeline...\n")
    try:
        for handler in handlers:
            context = handler.handle(context)
        print("\n✓ Pipeline completed successfully")
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")

    # Print final context
    print("\nFinal context:")
    for key, value in context.items():
        if key != "input_path":  # Skip input_path in output
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
