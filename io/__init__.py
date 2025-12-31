"""IO module for video service.

Экспортирует:
- ArtifactsStore: хранилище артефактов с fingerprint-кэшированием
- get_artifacts_store: получение глобального store
"""

from io.artifacts_store import (
    ArtifactsStore,
    get_artifacts_store,
    reset_artifacts_store,
)

__all__ = [
    "ArtifactsStore",
    "get_artifacts_store",
    "reset_artifacts_store",
]
