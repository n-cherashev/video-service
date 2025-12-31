"""
PipelineState V2 - новое состояние пайплайна с артефактами и контрактами.

Хранит:
- inputs: исходные артефакты (видео)
- artifacts: ссылки на все созданные артефакты
- data: key-value данные по ключам из Key enum
- metrics: тайминги, прогресс, метрики
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, Optional, Set

from typing import TYPE_CHECKING

from .keys import Key, MERGEABLE_KEYS
from .artifacts import ArtifactRef, VideoArtifact

if TYPE_CHECKING:
    from .contracts import NodePatch


PipelineStatusV2 = Literal["pending", "running", "completed", "failed"]


@dataclass
class PipelineMetrics:
    """Метрики выполнения пайплайна."""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    layer_timings: Dict[int, float] = field(default_factory=dict)
    node_timings: Dict[str, float] = field(default_factory=dict)
    peak_memory_mb: Optional[float] = None

    @property
    def total_time(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_time": self.total_time,
            "layer_timings": self.layer_timings,
            "node_timings": self.node_timings,
            "peak_memory_mb": self.peak_memory_mb,
        }


@dataclass
class PipelineStateV2:
    """Состояние пайплайна V2 с явными контрактами."""

    # Идентификация
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Конфигурация
    settings: Any = None  # VideoServiceSettings

    # Входные данные
    input_video: Optional[VideoArtifact] = None

    # Артефакты (ссылки на файлы)
    artifacts: Dict[str, ArtifactRef] = field(default_factory=dict)

    # Key-Value данные (основное состояние)
    data: Dict[Key, Any] = field(default_factory=dict)

    # Статус выполнения
    status: PipelineStatusV2 = "pending"
    current_stage: Optional[str] = None
    completed_stages: Set[str] = field(default_factory=set)
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    # Метрики
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)

    # === Методы работы с данными ===

    def get(self, key: Key, default: Any = None) -> Any:
        """Получает значение по ключу."""
        return self.data.get(key, default)

    def set(self, key: Key, value: Any) -> None:
        """Устанавливает значение по ключу."""
        self.data[key] = value

    def has(self, key: Key) -> bool:
        """Проверяет наличие ключа."""
        return key in self.data

    @property
    def available_keys(self) -> set[Key]:
        """Все доступные ключи."""
        return set(self.data.keys())

    # === Работа с артефактами ===

    def add_artifact(self, name: str, ref: ArtifactRef) -> None:
        """Добавляет артефакт."""
        self.artifacts[name] = ref

    def get_artifact(self, name: str) -> Optional[ArtifactRef]:
        """Получает артефакт по имени."""
        return self.artifacts.get(name)

    # === Статусы и прогресс ===

    def mark_started(self) -> None:
        """Помечает пайплайн как запущенный."""
        self.status = "running"
        self.metrics.start_time = time.time()

    def mark_completed(self) -> None:
        """Помечает пайплайн как завершённый."""
        self.status = "completed"
        self.metrics.end_time = time.time()

    def mark_failed(self, error: str) -> None:
        """Помечает пайплайн как упавший."""
        self.status = "failed"
        self.error = error
        self.metrics.end_time = time.time()

    def mark_stage_started(self, stage_name: str) -> None:
        """Помечает начало этапа."""
        self.current_stage = stage_name

    def mark_stage_completed(self, stage_name: str, execution_time: float = 0.0) -> None:
        """Помечает завершение этапа."""
        self.completed_stages.add(stage_name)
        self.current_stage = None
        if execution_time > 0:
            self.metrics.node_timings[stage_name] = execution_time

    def add_warning(self, warning: str) -> None:
        """Добавляет предупреждение."""
        self.warnings.append(warning)

    # === Мерджинг патчей ===

    def apply_patch(self, patch: NodePatch) -> list[str]:
        """Применяет NodePatch к состоянию.

        Возвращает список ошибок (пустой если всё ок).
        """
        errors: list[str] = []

        for key, value in patch.provides.items():
            if key in self.data and key not in MERGEABLE_KEYS:
                # Конфликт - ключ уже существует
                errors.append(
                    f"Key conflict: '{key.value}' already exists "
                    f"(from {patch.handler_name})"
                )
            else:
                # Для mergeable keys делаем специальный merge
                if key == Key.COMPLETED_STAGES and key in self.data:
                    existing = self.data[key]
                    if isinstance(existing, set) and isinstance(value, set):
                        self.data[key] = existing | value
                    else:
                        self.data[key] = value
                elif key in (Key.LAYER_TIMINGS, Key.NODE_TIMINGS) and key in self.data:
                    existing = self.data[key]
                    if isinstance(existing, dict) and isinstance(value, dict):
                        existing.update(value)
                    else:
                        self.data[key] = value
                elif key == Key.WARNINGS and key in self.data:
                    existing = self.data[key]
                    if isinstance(existing, list) and isinstance(value, list):
                        existing.extend(value)
                    else:
                        self.data[key] = value
                else:
                    self.data[key] = value

        # Добавляем артефакты
        for artifact in patch.artifacts_created:
            self.artifacts[artifact.kind.value] = artifact

        # Записываем время выполнения
        if patch.execution_time_seconds > 0:
            self.metrics.node_timings[patch.handler_name] = patch.execution_time_seconds

        # Добавляем warnings
        self.warnings.extend(patch.warnings)

        # Помечаем stage как completed
        self.mark_stage_completed(patch.handler_name, patch.execution_time_seconds)

        return errors

    # === Сериализация ===

    def to_dict(self) -> dict[str, Any]:
        """Сериализует состояние в dict."""
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "current_stage": self.current_stage,
            "completed_stages": list(self.completed_stages),
            "error": self.error,
            "warnings": self.warnings,
            "artifacts": {k: v.to_dict() for k, v in self.artifacts.items()},
            "metrics": self.metrics.to_dict(),
            # data не сериализуем целиком - только ключевые поля
            "duration_seconds": self.get(Key.DURATION_SECONDS),
            "fps": self.get(Key.FPS),
        }
