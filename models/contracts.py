"""
Handler Contracts - модели для requires/provides контрактов и NodePatch.

StateView - read-only view на состояние пайплайна для handler'а.
NodePatch - результат handler'а с явными provides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, FrozenSet, Mapping, Optional, TYPE_CHECKING

from .keys import Key
from .artifacts import ArtifactRef

if TYPE_CHECKING:
    from .pipeline_state import PipelineStateV2


@dataclass(frozen=True)
class HandlerContract:
    """Контракт handler'а: что требует и что предоставляет."""
    name: str
    requires: FrozenSet[Key]
    provides: FrozenSet[Key]
    optional_requires: FrozenSet[Key] = frozenset()

    def validate_inputs(self, available_keys: set[Key]) -> list[str]:
        """Проверяет, что все requires доступны. Возвращает список ошибок."""
        missing = self.requires - available_keys
        return [f"Handler '{self.name}' missing required key: {k.value}" for k in missing]


@dataclass
class NodePatch:
    """Результат выполнения handler'а.

    Содержит только то, что handler provides (строго по контракту).
    """
    handler_name: str
    provides: dict[Key, Any] = field(default_factory=dict)
    artifacts_created: list[ArtifactRef] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "handler_name": self.handler_name,
            "provides": {k.value: _serialize_value(v) for k, v in self.provides.items()},
            "artifacts_created": [a.to_dict() for a in self.artifacts_created],
            "execution_time_seconds": self.execution_time_seconds,
            "warnings": self.warnings,
            "error": self.error,
        }


class StateView:
    """Read-only view на состояние пайплайна.

    Handler получает StateView и НЕ может модифицировать state напрямую.
    """

    def __init__(self, state: PipelineStateV2) -> None:
        self._state = state

    def get(self, key: Key, default: Any = None) -> Any:
        """Получает значение по ключу."""
        return self._state.data.get(key, default)

    def __getitem__(self, key: Key) -> Any:
        """Получает значение по ключу (raises KeyError если нет)."""
        if key not in self._state.data:
            raise KeyError(f"Key {key.value} not found in state")
        return self._state.data[key]

    def __contains__(self, key: Key) -> bool:
        return key in self._state.data

    def has(self, key: Key) -> bool:
        return key in self._state.data

    @property
    def available_keys(self) -> set[Key]:
        return set(self._state.data.keys())

    @property
    def artifacts(self) -> Mapping[str, ArtifactRef]:
        """Все артефакты."""
        return self._state.artifacts

    @property
    def settings(self) -> Any:
        return self._state.settings

    @property
    def run_id(self) -> str:
        return self._state.run_id

    # Удобные property для часто используемых данных
    @property
    def duration_seconds(self) -> float:
        return self.get(Key.DURATION_SECONDS, 0.0)

    @property
    def fps(self) -> float:
        return self.get(Key.FPS, 25.0)

    @property
    def video_path(self) -> Optional[str]:
        return self.get(Key.VIDEO_PATH)

    @property
    def audio_path(self) -> Optional[str]:
        return self.get(Key.AUDIO_PATH)


def _serialize_value(value: Any) -> Any:
    """Сериализует значение для JSON."""
    if hasattr(value, 'to_dict'):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(str(v) for v in value)
    return value
