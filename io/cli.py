import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Video analysis pipeline")
    parser.add_argument(
        "video_path",
        nargs="?",
        help="Путь к видеофайлу (если не указан — берётся из VIDEO_SERVICE_INPUT_VIDEO_PATH).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Путь к файлу конфигурации",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Путь для сохранения результата",
    )
    return parser.parse_args()