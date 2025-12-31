from __future__ import annotations

from typing import Any, Dict, List

from core.base_handler import BaseHandler
from llm.base import LLMClient


class TopicSegmentationHandler(BaseHandler):
    """Разбивает ролик на смысловые блоки и назначает темы через LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        block_duration: float = 45.0,
        min_block_chars: int = 30,
        max_blocks: int = 64,
    ) -> None:
        self.llm_client = llm_client
        self.block_duration = float(block_duration)
        self.min_block_chars = int(min_block_chars)
        self.max_blocks = int(max_blocks)

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("[11] TopicSegmentationHandler")

        segments = context.get("transcript_segments") or []
        duration = float(context.get("duration_seconds") or 0.0)
        language = context.get("language")

        if not segments or duration <= 0:
            context["topic_segments"] = []
            print("✓ Topics: 0 segments")
            return context

        blocks = self._build_blocks(segments, duration)[: self.max_blocks]
        block_texts = [b["text"] for b in blocks]

        try:
            slugs = self.llm_client.summarize_topics(block_texts, language=language)
        except Exception as exc:
            print(f"TopicSegmentationHandler: LLM error: {exc}")
            slugs = ["general" for _ in blocks]

        if len(slugs) != len(blocks):
            slugs = (slugs[: len(blocks)] + ["general"] * len(blocks))[: len(blocks)]

        topic_segments = []
        for b, slug in zip(blocks, slugs):
            topic_segments.append({"start": b["start"], "end": b["end"], "topic": slug})

        context["topic_segments"] = topic_segments
        print(f"✓ Topics: {len(topic_segments)} segments")
        return context

    def _build_blocks(self, segments: List[Dict[str, Any]], duration: float) -> List[Dict[str, Any]]:
        segs = sorted(segments, key=lambda s: float(s.get("start") or 0.0))
        blocks: list[dict[str, Any]] = []

        t = 0.0
        while t < duration:
            end = min(t + self.block_duration, duration)

            parts: list[str] = []
            for seg in segs:
                s = float(seg.get("start") or 0.0)
                e = float(seg.get("end") or s)
                if e <= t:
                    continue
                if s >= end:
                    break
                txt = str(seg.get("text") or "").strip()
                if txt:
                    parts.append(txt)

            text = " ".join(parts).strip()
            if len(text) < self.min_block_chars:
                text = ""

            blocks.append({"start": t, "end": end, "text": text})
            t = end

        return blocks