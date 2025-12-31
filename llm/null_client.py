from __future__ import annotations

from typing import Any, Mapping

from llm.base import LLMClient


class NullLLMClient(LLMClient):
    def score_humor(self, text: str, language: str | None = None, metadata: Mapping[str, Any] | None = None) -> float:
        return 0.0

    def analyze_blocks(self, blocks: list[dict], language: str | None = None) -> list[dict]:
        return [
            {
                "topic_slug": "general",
                "topic_title": "General",
                "topic_confidence": 0.1,
                "humor_score": 0.0,
                "humor_label": "none"
            }
            for _ in blocks
        ]