from __future__ import annotations

from typing import Any, Dict, List

from handlers.base_handler import BaseHandler


class HighlightDetectionHandler(BaseHandler):
    """Выделяет хайлайты: action, comedy, drama, dialogue."""

    def __init__(self, min_duration: float = 3.0, max_duration: float = 30.0) -> None:
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        # Пороги для разных типов хайлайтов
        self.thresholds = {
            "action": {"motion": 0.6, "audio": 0.5},
            "comedy": {"humor": 0.4},
            "drama": {"sentiment": 0.6},
            "dialogue": {"sentiment": 0.3, "is_dialogue": True}
        }

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        timeline = context.get("timeline", [])
        
        if not timeline:
            context["highlights"] = []
            return context

        highlights = self._detect_highlights(timeline)
        context["highlights"] = highlights
        
        print(f"✓ Highlights: {len(highlights)} detected")
        return context

    def _detect_highlights(self, timeline: List[Dict]) -> List[Dict]:
        """Детектирует хайлайты разных типов."""
        highlights = []
        
        # Детектируем каждый тип хайлайта
        highlights.extend(self._detect_action(timeline))
        highlights.extend(self._detect_comedy(timeline))
        highlights.extend(self._detect_drama(timeline))
        highlights.extend(self._detect_dialogue(timeline))
        
        # Сортируем по времени
        highlights.sort(key=lambda x: x["start"])
        
        # Объединяем перекрывающиеся хайлайты
        return self._merge_overlapping(highlights)

    def _detect_action(self, timeline: List[Dict]) -> List[Dict]:
        """Детектирует экшн сцены."""
        return self._detect_by_conditions(
            timeline, "action",
            lambda point: (
                point["motion"] > self.thresholds["action"]["motion"] and
                point["audio_loudness"] > self.thresholds["action"]["audio"]
            )
        )

    def _detect_comedy(self, timeline: List[Dict]) -> List[Dict]:
        """Детектирует комедийные сцены."""
        return self._detect_by_conditions(
            timeline, "comedy",
            lambda point: (
                point["humor"] > self.thresholds["comedy"]["humor"] or
                point["has_laughter"]
            )
        )

    def _detect_drama(self, timeline: List[Dict]) -> List[Dict]:
        """Детектирует драматические сцены."""
        return self._detect_by_conditions(
            timeline, "drama",
            lambda point: abs(point["sentiment"]) > self.thresholds["drama"]["sentiment"]
        )

    def _detect_dialogue(self, timeline: List[Dict]) -> List[Dict]:
        """Детектирует важные диалоги."""
        return self._detect_by_conditions(
            timeline, "dialogue",
            lambda point: (
                point["is_dialogue"] and
                abs(point["sentiment"]) > self.thresholds["dialogue"]["sentiment"]
            )
        )

    def _detect_by_conditions(self, timeline: List[Dict], highlight_type: str, 
                            condition_func) -> List[Dict]:
        """Общий метод детекции по условиям."""
        highlights = []
        current_start = None
        current_points = []
        
        for point in timeline:
            if condition_func(point):
                if current_start is None:
                    current_start = point["time"]
                current_points.append(point)
            else:
                if current_start is not None:
                    # Завершаем текущий хайлайт
                    duration = point["time"] - current_start
                    if self.min_duration <= duration <= self.max_duration:
                        score = sum(p["interest"] for p in current_points) / len(current_points)
                        highlights.append({
                            "start": current_start,
                            "end": point["time"],
                            "type": highlight_type,
                            "score": score
                        })
                    
                    current_start = None
                    current_points = []
        
        # Обрабатываем последний хайлайт, если он есть
        if current_start is not None and current_points:
            duration = timeline[-1]["time"] - current_start
            if self.min_duration <= duration <= self.max_duration:
                score = sum(p["interest"] for p in current_points) / len(current_points)
                highlights.append({
                    "start": current_start,
                    "end": timeline[-1]["time"],
                    "type": highlight_type,
                    "score": score
                })
        
        return highlights

    def _merge_overlapping(self, highlights: List[Dict]) -> List[Dict]:
        """Объединяет перекрывающиеся хайлайты."""
        if not highlights:
            return []
        
        merged = []
        current = highlights[0].copy()
        
        for next_highlight in highlights[1:]:
            # Если хайлайты перекрываются
            if next_highlight["start"] <= current["end"]:
                # Расширяем текущий хайлайт
                current["end"] = max(current["end"], next_highlight["end"])
                current["score"] = max(current["score"], next_highlight["score"])
                
                # Если типы разные, берем с большим score
                if next_highlight["score"] > current["score"]:
                    current["type"] = next_highlight["type"]
            else:
                # Сохраняем текущий и начинаем новый
                merged.append(current)
                current = next_highlight.copy()
        
        merged.append(current)
        return merged