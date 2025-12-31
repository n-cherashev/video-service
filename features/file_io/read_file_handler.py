"""
ReadFileHandler - валидирует и регистрирует входной видео файл.

Extractor: читает файл, проверяет существование, возвращает VideoArtifact.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, ClassVar, FrozenSet

from core.base_handler import ExtractorHandler
from models.keys import Key


# Разрешённые расширения видео
ALLOWED_VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv", ".flv"
})

def _get_max_file_size_bytes() -> int:
    gb_str = os.environ.get("VIDEO_SERVICE_MAX_FILE_SIZE_GB", "15")
    try:
        gb = float(gb_str)
    except Exception:
        gb = 15.0
    if gb <= 0:
        gb = 15.0
    return int(gb * 1024 * 1024 * 1024)


# Максимальный размер файла (15GB по умолчанию, настраивается env: VIDEO_SERVICE_MAX_FILE_SIZE_GB)
MAX_FILE_SIZE_BYTES: int = _get_max_file_size_bytes()


class ReadFileHandler(ExtractorHandler):
    """Handler для чтения и валидации видео файла.

    Проверяет:
    - Существование файла
    - Что это файл, а не директория
    - Расширение файла
    - Размер файла (опционально)

    Provides:
    - VIDEO_PATH: нормализованный путь к видео
    - video_size_bytes: размер файла
    - video_fingerprint: sha256 хеш для кэширования
    """

    requires: ClassVar[FrozenSet[Key]] = frozenset({Key.INPUT_PATH})
    provides: ClassVar[FrozenSet[Key]] = frozenset({Key.VIDEO_PATH})

    def __init__(
        self,
        allowed_extensions: frozenset[str] | None = None,
        max_file_size_bytes: int | None = None,
        compute_fingerprint: bool = True,
    ) -> None:
        self.allowed_extensions = allowed_extensions or ALLOWED_VIDEO_EXTENSIONS
        self.max_file_size_bytes = max_file_size_bytes or MAX_FILE_SIZE_BYTES
        self.compute_fingerprint = compute_fingerprint

    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        print("[1] ReadFileHandler")

        input_path = context.get("input_path")
        if not input_path:
            raise ValueError("'input_path' not provided in context")

        # Нормализуем путь
        video_path = Path(input_path).resolve()
        video_path_str = str(video_path)

        # Проверка существования
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path_str}")

        # Проверка что это файл
        if not video_path.is_file():
            raise IsADirectoryError(f"Path is not a file: {video_path_str}")

        # Проверка расширения
        extension = video_path.suffix.lower()
        if extension not in self.allowed_extensions:
            raise ValueError(
                f"Unsupported file extension: {extension}. "
                f"Allowed: {', '.join(sorted(self.allowed_extensions))}"
            )

        # Получаем размер файла
        video_size_bytes = video_path.stat().st_size

        # Проверка размера
        if video_size_bytes > self.max_file_size_bytes:
            max_gb = self.max_file_size_bytes / (1024 ** 3)
            file_gb = video_size_bytes / (1024 ** 3)
            raise ValueError(
                f"File too large: {file_gb:.2f}GB (max: {max_gb:.2f}GB)"
            )

        # Вычисляем fingerprint (для кэширования)
        fingerprint = ""
        if self.compute_fingerprint:
            fingerprint = self._compute_file_fingerprint(video_path)

        # Записываем в контекст
        context["video_path"] = video_path_str
        context["video_size_bytes"] = video_size_bytes
        context["video_fingerprint"] = fingerprint
        context["video_extension"] = extension
        context["video_name"] = video_path.name

        print(f"✓ File read: {video_path.name} ({video_size_bytes / 1024 / 1024:.1f} MB)")

        return context

    def _compute_file_fingerprint(self, path: Path) -> str:
        """Вычисляет fingerprint файла (sha256 первых 64KB + size + mtime)."""
        hasher = hashlib.sha256()

        # Читаем первые 64KB
        with open(path, "rb") as f:
            chunk = f.read(65536)
            hasher.update(chunk)

        # Добавляем размер и mtime
        stat = path.stat()
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(int(stat.st_mtime)).encode())

        return hasher.hexdigest()[:16]
