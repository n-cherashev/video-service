from __future__ import annotations

from typing import Any, Dict, List

from handlers.base_handler import BaseHandler


class HumorDetectionHandler(BaseHandler):
    """Выделяет смешные моменты на основе текста."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        segments = context.get("transcript_segments", [])
        
        humor_scores = []
        for seg in segments:
            text = seg["text"].lower()
            score = self._calculate_humor_score(text)
            time = (seg["start"] + seg["end"]) / 2
            humor_scores.append({"time": time, "score": score})

        # Сортировка по времени
        humor_scores.sort(key=lambda x: x["time"])

        # Подсчет статистики
        scores = [h["score"] for h in humor_scores]
        humor_summary = {
            "mean": sum(scores) / len(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0,
            "count_positive": sum(1 for s in scores if s > self.threshold)
        }

        context["humor_scores"] = humor_scores
        context["humor_summary"] = humor_summary
        
        print(f"✓ Humor: {len(humor_scores)} segments, {humor_summary['count_positive']} funny")
        return context

    def _calculate_humor_score(self, text: str) -> float:
        """Rule-based эвристика для определения юмора."""
        score = 0.0
        
        # Прямые маркеры смеха
        laugh_markers = ["смеется", "смех в зале", "смех", "смешно", "шутка", "прикол"]
        for marker in laugh_markers:
            if marker in text:
                score += 0.8
        
        # Междометия и разговорные конструкции
        casual_markers = ["аха", "ха-ха", "лол", "чувак", "ой", "блин", "капец"]
        for marker in casual_markers:
            if marker in text:
                score += 0.3
        
        return min(score, 1.0)