"""
BaseHandler V2 - базовый класс для handler'ов с поддержкой контрактов.

Поддерживает два режима:
1. Legacy: handle(context) -> context (обратная совместимость)
2. V2: run(state_view) -> NodePatch (с контрактами)

Handler должен либо:
- Переопределить `run()` для нового API
- Переопределить `handle()` для legacy API
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, FrozenSet, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.keys import Key
    from models.contracts import NodePatch, StateView, HandlerContract


class BaseHandler(ABC):
    """Base handler interface с поддержкой контрактов.

    Атрибуты класса:
        requires: ключи, которые handler требует (FrozenSet[Key])
        provides: ключи, которые handler создаёт (FrozenSet[Key])
        optional_requires: опциональные требования (FrozenSet[Key])

    Методы:
        run(state_view) -> NodePatch: новый API с контрактами
        handle(context) -> context: legacy API для обратной совместимости
    """

    # Контракт handler'а (переопределяется в подклассах)
    requires: ClassVar[FrozenSet[Key]] = frozenset()
    provides: ClassVar[FrozenSet[Key]] = frozenset()
    optional_requires: ClassVar[FrozenSet[Key]] = frozenset()

    @property
    def name(self) -> str:
        """Имя handler'а (по умолчанию имя класса)."""
        return self.__class__.__name__

    def get_contract(self) -> HandlerContract:
        """Возвращает контракт handler'а."""
        from models.contracts import HandlerContract
        return HandlerContract(
            name=self.name,
            requires=self.requires,
            provides=self.provides,
            optional_requires=self.optional_requires,
        )

    def run(self, state: StateView) -> NodePatch:
        """Выполняет handler с новым API (контракты).

        Args:
            state: Read-only view на состояние пайплайна.

        Returns:
            NodePatch с результатами (только provides).
        """
        from models.contracts import NodePatch

        # Default implementation: вызываем legacy handle()
        # и конвертируем результат в NodePatch
        start_time = time.monotonic()

        try:
            # Конвертируем StateView в dict для legacy API
            context = self._state_to_context(state)

            # Вызываем legacy handle
            result_context = self.handle(context)

            # Конвертируем результат в NodePatch
            patch = self._context_to_patch(result_context, state)
            patch.execution_time_seconds = time.monotonic() - start_time

            return patch

        except Exception as e:
            return NodePatch(
                handler_name=self.name,
                error=str(e),
                execution_time_seconds=time.monotonic() - start_time,
            )

    @abstractmethod
    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """Legacy API: обрабатывает контекст и возвращает обновлённый контекст.

        Этот метод ДОЛЖЕН быть переопределён в подклассах,
        либо вместо него переопределяется run().
        """
        raise NotImplementedError

    def _state_to_context(self, state: StateView) -> dict[str, Any]:
        """Конвертирует StateView в dict для legacy API."""
        from models.keys import Key

        context: dict[str, Any] = {}

        # Добавляем все доступные ключи
        for key in state.available_keys:
            # Конвертируем Key enum в строку для legacy API
            context[key.value] = state.get(key)

        # Добавляем settings
        if state.settings:
            context["settings"] = state.settings

        return context

    def _context_to_patch(
        self,
        context: dict[str, Any],
        original_state: StateView,
    ) -> NodePatch:
        """Конвертирует результат handle() в NodePatch."""
        from models.keys import Key
        from models.contracts import NodePatch

        # Находим новые/изменённые ключи
        provides_dict: dict[Key, Any] = {}

        for key_str, value in context.items():
            # Пропускаем служебные ключи
            if key_str in ("settings",):
                continue

            # Пробуем преобразовать строку в Key
            try:
                key = Key(key_str)
            except ValueError:
                # Ключ не в enum - пропускаем или логируем
                continue

            # Проверяем, изменилось ли значение
            if not original_state.has(key):
                provides_dict[key] = value
            elif original_state.get(key) != value:
                provides_dict[key] = value

        return NodePatch(
            handler_name=self.name,
            provides=provides_dict,
        )


class ExtractorHandler(BaseHandler):
    """Базовый класс для Extractor handlers (I/O, декодирование).

    Extractors создают артефакты на диске.
    """
    pass


class AnalyzerHandler(BaseHandler):
    """Базовый класс для Analyzer handlers (чистая аналитика).

    Analyzers работают с готовыми артефактами и создают сигналы/оценки.
    """
    pass
