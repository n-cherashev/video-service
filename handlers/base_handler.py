from abc import ABC, abstractmethod
from typing import Any


class BaseHandler(ABC):
    """Base handler interface."""

    @abstractmethod
    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """Process context."""
        pass
