from typing import Any, Iterable

from core.base_handler import BaseHandler


def run_pipeline(context: dict[str, Any], handlers: Iterable[BaseHandler]) -> dict[str, Any]:
    for h in handlers:
        context = h.handle(context)
    return context