from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseHandler(ABC):
    """Base handler interface."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """Process context and return updated context."""
        raise NotImplementedError
