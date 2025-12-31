from __future__ import annotations

from typing import Any, Mapping, Protocol


class LLMClient(Protocol):
    def score_humor(
        self,
        text: str,
        language: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> float:
        ...

    def summarize_topics(
        self,
        blocks: list[str],
        language: str | None = None,
    ) -> list[str]:
        ...
    
    def analyze_blocks(
        self,
        blocks: list[dict],
        language: str | None = None,
    ) -> list[dict]:
        ...
    
    def refine_candidates(
        self,
        candidates: list[dict],
        language: str | None = None,
    ) -> list[dict]:
        ...