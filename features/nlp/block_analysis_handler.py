from __future__ import annotations

from typing import Any, Dict, List

from core.base_handler import BaseHandler
from llm.base import LLMClient


class BlockAnalysisHandler(BaseHandler):
    """Анализирует блоки видео через LLM для получения тем и юмора одним запросом."""

    def __init__(
        self,
        llm_client: LLMClient,
        block_duration: float = 45.0,
        min_block_chars: int = 30,
        max_blocks: int = 64,
        max_blocks_per_request: int = 6,
        max_text_per_block: int = 500,
    ) -> None:
        self.llm_client = llm_client
        self.block_duration = float(block_duration)
        self.min_block_chars = int(min_block_chars)
        self.max_blocks = int(max_blocks)
        self.max_blocks_per_request = int(max_blocks_per_request)
        self.max_text_per_block = int(max_text_per_block)

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("[11] BlockAnalysisHandler")

        segments = context.get("transcript_segments") or []
        duration = float(context.get("duration_seconds") or 0.0)
        language = context.get("language")

        if not segments or duration <= 0:
            context["block_analysis"] = []
            context["topic_segments"] = []
            print("✓ Block analysis: 0 blocks")
            return context

        # Строим блоки по времени
        blocks = self._build_blocks(segments, duration)[: self.max_blocks]
        
        if not blocks:
            context["block_analysis"] = []
            context["topic_segments"] = []
            print("✓ Block analysis: 0 blocks")
            return context

        # LLM анализ блоков с chunking
        try:
            analysis_results = self._analyze_blocks_chunked(blocks, language)
        except Exception as exc:
            print(f"BlockAnalysisHandler: LLM error: {exc}")
            analysis_results = self._fallback_analysis(len(blocks))

        # Формируем результаты
        block_analysis = []
        topic_segments = []
        
        for i, (block, analysis) in enumerate(zip(blocks, analysis_results)):
            block_result = {
                "start": block["start"],
                "end": block["end"],
                "topic_slug": analysis["topic_slug"],
                "topic_title": analysis["topic_title"],
                "topic_confidence": analysis["topic_confidence"],
                "humor_score": analysis["humor_score"],
                "humor_label": analysis["humor_label"]
            }
            block_analysis.append(block_result)
            
            # Обратная совместимость для topic_segments
            topic_segments.append({
                "start": block["start"],
                "end": block["end"],
                "topic": analysis["topic_slug"]
            })

        context["block_analysis"] = block_analysis
        context["topic_segments"] = topic_segments

        print(f"✓ Block analysis: {len(block_analysis)} blocks analyzed")
        return context

    def _build_blocks(self, segments: List[Dict[str, Any]], duration: float) -> List[Dict[str, Any]]:
        """Строит блоки по времени из сегментов с аудио сигналами."""
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
            
            # Ограничиваем длину текста блока
            if len(text) > self.max_text_per_block:
                text = text[:self.max_text_per_block] + "..."

            blocks.append({"start": t, "end": end, "text": text})
            t = end

        return blocks

    def _analyze_blocks_chunked(self, blocks: List[Dict[str, Any]], language: str | None) -> List[Dict[str, Any]]:
        """Анализирует блоки по частям для избежания таймаутов."""
        if not blocks:
            return []
        
        all_results = []
        
        # Разбиваем на чанки
        for i in range(0, len(blocks), self.max_blocks_per_request):
            chunk = blocks[i:i + self.max_blocks_per_request]
            chunk_num = i//self.max_blocks_per_request + 1
            
            # Задержка между запросами (кроме первого)
            if i > 0:
                import time
                print(f"Waiting 5s before chunk {chunk_num}...")
                time.sleep(10)
                print(f"Waiting 5s before chunk {chunk_num}... done")
            
            try:
                chunk_results = self.llm_client.analyze_blocks(chunk, language=language)
                all_results.extend(chunk_results)
                print(f"✓ Analyzed chunk {chunk_num}: {len(chunk)} blocks")
            except Exception as exc:
                print(f"Chunk {chunk_num} failed: {exc}")
                # Fallback для этого чанка
                all_results.extend(self._fallback_analysis(len(chunk)))
        
        return all_results

    def _fallback_analysis(self, count: int) -> List[Dict[str, Any]]:
        """Fallback результат при ошибке LLM."""
        return [
            {
                "topic_slug": "general",
                "topic_title": "General",
                "topic_confidence": 0.1,
                "humor_score": 0.0,
                "humor_label": "none"
            }
            for _ in range(count)
        ]