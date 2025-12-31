"""
ArtifactsStore - хранилище артефактов с fingerprint-кэшированием.

Поддерживает:
- Сохранение/загрузка series в npz формате
- JSON файлы для метаданных и preview
- Fingerprint-based кэширование (input + settings hash)
- Автоматическая очистка старых артефактов
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from models.artifacts import (
    ArtifactKind,
    ArtifactRef,
    SeriesArtifact,
    TranscriptArtifact,
)


class ArtifactsStore:
    """Хранилище артефактов с fingerprint-кэшированием."""

    def __init__(
        self,
        base_dir: str = "artifacts",
        cache_enabled: bool = True,
        max_cache_size_gb: float = 10.0,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.cache_enabled = cache_enabled
        self.max_cache_size_gb = max_cache_size_gb

        # Создаём директории
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "series").mkdir(exist_ok=True)
        (self.base_dir / "meta").mkdir(exist_ok=True)
        (self.base_dir / "preview").mkdir(exist_ok=True)

    # === Fingerprint ===

    def compute_fingerprint(
        self,
        input_path: str,
        settings_dict: Optional[Dict[str, Any]] = None,
        extra: Optional[str] = None,
    ) -> str:
        """Вычисляет fingerprint для кэширования.

        Fingerprint = sha256(file_path + file_size + file_mtime + settings_hash + extra)
        """
        hasher = hashlib.sha256()

        # File info
        path = Path(input_path)
        if path.exists():
            stat = path.stat()
            hasher.update(str(input_path).encode())
            hasher.update(str(stat.st_size).encode())
            hasher.update(str(int(stat.st_mtime)).encode())
        else:
            hasher.update(str(input_path).encode())

        # Settings hash
        if settings_dict:
            settings_str = json.dumps(settings_dict, sort_keys=True, default=str)
            hasher.update(settings_str.encode())

        # Extra
        if extra:
            hasher.update(extra.encode())

        return hasher.hexdigest()[:16]  # Короткий fingerprint

    # === Series (npz) ===

    def save_series(
        self,
        kind: ArtifactKind,
        fingerprint: str,
        data: Dict[str, np.ndarray],
        step_seconds: float,
        summary: Optional[Dict[str, float]] = None,
    ) -> SeriesArtifact:
        """Сохраняет time series в npz формат.

        Args:
            kind: тип артефакта
            fingerprint: fingerprint для кэширования
            data: dict с массивами (например {"times": [...], "loudness": [...], "energy": [...]})
            step_seconds: шаг по времени
            summary: сводка (mean, max, etc.)

        Returns:
            SeriesArtifact с ссылкой на файл
        """
        filename = f"{kind.value}_{fingerprint}.npz"
        path = self.base_dir / "series" / filename

        # Сохраняем npz
        np.savez_compressed(path, **data)

        # Определяем columns и num_points
        columns = list(data.keys())
        num_points = len(next(iter(data.values()))) if data else 0

        ref = ArtifactRef(
            kind=kind,
            path=str(path),
            fingerprint=fingerprint,
            created_at=datetime.utcnow(),
        )

        return SeriesArtifact(
            ref=ref,
            step_seconds=step_seconds,
            num_points=num_points,
            columns=columns,
            summary=summary or {},
        )

    def load_series(self, artifact: SeriesArtifact) -> Dict[str, np.ndarray]:
        """Загружает series из npz файла."""
        path = Path(artifact.ref.path)
        if not path.exists():
            raise FileNotFoundError(f"Series artifact not found: {path}")

        data = np.load(path)
        return {key: data[key] for key in data.files}

    def series_exists(self, kind: ArtifactKind, fingerprint: str) -> bool:
        """Проверяет существование series артефакта."""
        filename = f"{kind.value}_{fingerprint}.npz"
        path = self.base_dir / "series" / filename
        return path.exists()

    def get_series_ref(self, kind: ArtifactKind, fingerprint: str) -> Optional[ArtifactRef]:
        """Получает ссылку на существующий series артефакт."""
        filename = f"{kind.value}_{fingerprint}.npz"
        path = self.base_dir / "series" / filename

        if not path.exists():
            return None

        return ArtifactRef(
            kind=kind,
            path=str(path),
            fingerprint=fingerprint,
            created_at=datetime.fromtimestamp(path.stat().st_mtime),
        )

    # === JSON Metadata ===

    def save_json(
        self,
        kind: ArtifactKind,
        fingerprint: str,
        data: Any,
    ) -> ArtifactRef:
        """Сохраняет данные в JSON формат."""
        filename = f"{kind.value}_{fingerprint}.json"
        path = self.base_dir / "meta" / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        return ArtifactRef(
            kind=kind,
            path=str(path),
            fingerprint=fingerprint,
            created_at=datetime.utcnow(),
        )

    def load_json(self, ref: ArtifactRef) -> Any:
        """Загружает данные из JSON файла."""
        path = Path(ref.path)
        if not path.exists():
            raise FileNotFoundError(f"JSON artifact not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # === Preview (sampled JSON) ===

    def save_series_preview(
        self,
        kind: ArtifactKind,
        fingerprint: str,
        series_data: Dict[str, np.ndarray],
        max_points: int = 100,
    ) -> str:
        """Сохраняет preview series (sampled) в JSON.

        Для больших series создаём JSON preview с max_points точками.
        """
        filename = f"{kind.value}_{fingerprint}_preview.json"
        path = self.base_dir / "preview" / filename

        # Sample data
        preview = {}
        for key, arr in series_data.items():
            if len(arr) <= max_points:
                preview[key] = arr.tolist()
            else:
                # Sample evenly
                indices = np.linspace(0, len(arr) - 1, max_points, dtype=int)
                preview[key] = arr[indices].tolist()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(preview, f)

        return str(path)

    # === Transcript ===

    def save_transcript(
        self,
        fingerprint: str,
        segments: List[Dict[str, Any]],
        language: str,
        full_text: str,
    ) -> TranscriptArtifact:
        """Сохраняет транскрипцию."""
        filename = f"transcript_{fingerprint}.json"
        path = self.base_dir / "meta" / filename

        data = {
            "segments": segments,
            "language": language,
            "full_text": full_text,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Вычисляем метрики
        total_speech_duration = sum(
            float(s.get("end", 0)) - float(s.get("start", 0))
            for s in segments
            if isinstance(s, dict)
        )

        ref = ArtifactRef(
            kind=ArtifactKind.TRANSCRIPT,
            path=str(path),
            fingerprint=fingerprint,
            created_at=datetime.utcnow(),
        )

        return TranscriptArtifact(
            ref=ref,
            language=language,
            num_segments=len(segments),
            total_speech_duration=total_speech_duration,
            total_characters=len(full_text),
        )

    def load_transcript(self, artifact: TranscriptArtifact) -> Dict[str, Any]:
        """Загружает транскрипцию."""
        return self.load_json(artifact.ref)

    # === Cache Management ===

    def get_cache_size_gb(self) -> float:
        """Возвращает размер кэша в GB."""
        total_size = 0
        for path in self.base_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size / (1024 ** 3)

    def cleanup_old_artifacts(self, max_age_days: int = 7) -> int:
        """Удаляет артефакты старше max_age_days.

        Returns:
            Количество удалённых файлов.
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=max_age_days)
        deleted = 0

        for path in self.base_dir.rglob("*"):
            if path.is_file():
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if mtime < cutoff:
                    path.unlink()
                    deleted += 1

        return deleted

    def clear_all(self) -> None:
        """Удаляет все артефакты."""
        for subdir in ["series", "meta", "preview"]:
            dir_path = self.base_dir / subdir
            if dir_path.exists():
                shutil.rmtree(dir_path)
                dir_path.mkdir()

    # === Utility ===

    def get_artifact_path(
        self,
        kind: ArtifactKind,
        fingerprint: str,
        extension: str = "npz",
    ) -> Path:
        """Возвращает путь для артефакта."""
        subdir = "series" if extension == "npz" else "meta"
        filename = f"{kind.value}_{fingerprint}.{extension}"
        return self.base_dir / subdir / filename


# Глобальный store (singleton)
_default_store: Optional[ArtifactsStore] = None


def get_artifacts_store(base_dir: str = "artifacts") -> ArtifactsStore:
    """Возвращает глобальный ArtifactsStore."""
    global _default_store
    if _default_store is None:
        _default_store = ArtifactsStore(base_dir=base_dir)
    return _default_store


def reset_artifacts_store() -> None:
    """Сбрасывает глобальный store."""
    global _default_store
    _default_store = None
