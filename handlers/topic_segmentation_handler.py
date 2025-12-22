from __future__ import annotations

from typing import Any, Dict, List
from collections import Counter
import re

from handlers.base_handler import BaseHandler


class TopicSegmentationHandler(BaseHandler):
    """Разбивает ролик на смысловые блоки."""

    def __init__(self, block_duration: float = 45.0, min_words: int = 3) -> None:
        self.block_duration = block_duration
        self.min_words = min_words

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        segments = context.get("transcript_segments", [])
        duration = context.get("duration_seconds", 0)
        
        if not segments:
            context["topic_segments"] = []
            return context

        topic_segments = self._create_topic_blocks(segments, duration)
        context["topic_segments"] = topic_segments
        
        print(f"✓ Topics: {len(topic_segments)} segments")
        return context

    def _create_topic_blocks(self, segments: List[Dict], duration: float) -> List[Dict]:
        """Создает блоки по времени и извлекает темы."""
        blocks = []
        current_time = 0.0
        
        while current_time < duration:
            block_end = min(current_time + self.block_duration, duration)
            
            # Собираем текст для блока
            block_text = ""
            for seg in segments:
                if seg["start"] >= current_time and seg["end"] <= block_end:
                    block_text += " " + seg["text"]
            
            topic = self._extract_topic(block_text.strip())
            
            blocks.append({
                "start": current_time,
                "end": block_end,
                "topic": topic
            })
            
            current_time = block_end
        
        return blocks

    def _extract_topic(self, text: str) -> str:
        """Извлекает тему из текста."""
        if not text:
            return "silence"
        
        # Простая эвристика на основе ключевых слов
        words = re.findall(r'\b[а-яё]+\b', text.lower())
        
        # Фильтруем служебные слова
        stop_words = {"и", "в", "на", "с", "по", "для", "от", "до", "при", "о", "об", 
                     "что", "как", "где", "когда", "почему", "который", "которая", "которое"}
        
        meaningful_words = [w for w in words if len(w) > 2 and w not in stop_words]
        
        if not meaningful_words:
            return "general"
        
        # Берем самые частые слова
        word_counts = Counter(meaningful_words)
        top_words = [word for word, _ in word_counts.most_common(2)]
        
        return "_".join(top_words) if top_words else "general"