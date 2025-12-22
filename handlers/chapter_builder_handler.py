from __future__ import annotations

from typing import Any, Dict, List

from handlers.base_handler import BaseHandler


class ChapterBuilderHandler(BaseHandler):
    """Формирует главы на основе топиков и границ сцен."""

    def __init__(self, min_chapter_duration: float = 30.0) -> None:
        self.min_chapter_duration = min_chapter_duration

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        duration = context.get("duration_seconds", 0)
        topic_segments = context.get("topic_segments", [])
        scene_boundaries = context.get("scene_boundaries", [0.0])
        timeline = context.get("timeline", [])
        
        if not topic_segments:
            # Создаем одну главу на все видео
            chapters = [{
                "start": 0.0,
                "end": duration,
                "title": "Полное видео",
                "description": "Весь контент"
            }]
        else:
            chapters = self._build_chapters(topic_segments, scene_boundaries, timeline, duration)
        
        context["chapters"] = chapters
        
        print(f"✓ Chapters: {len(chapters)} created")
        return context

    def _build_chapters(self, topic_segments: List[Dict], scene_boundaries: List[float],
                       timeline: List[Dict], duration: float) -> List[Dict]:
        """Строит главы на основе топиков."""
        chapters = []
        
        # Корректируем границы по сценам
        adjusted_segments = self._adjust_by_scenes(topic_segments, scene_boundaries)
        
        # Объединяем слишком короткие сегменты
        merged_segments = self._merge_short_segments(adjusted_segments)
        
        # Создаем главы
        for i, segment in enumerate(merged_segments):
            title = self._generate_title(segment["topic"], i + 1)
            description = self._generate_description(segment, timeline)
            
            chapters.append({
                "start": segment["start"],
                "end": segment["end"],
                "title": title,
                "description": description
            })
        
        return chapters

    def _adjust_by_scenes(self, segments: List[Dict], boundaries: List[float]) -> List[Dict]:
        """Корректирует границы сегментов по границам сцен."""
        adjusted = []
        
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            
            # Ищем ближайшие границы сцен
            closest_start = min(boundaries, key=lambda x: abs(x - start))
            closest_end = min(boundaries, key=lambda x: abs(x - end))
            
            # Корректируем только если граница близко (в пределах 5 секунд)
            if abs(closest_start - start) < 5.0:
                start = closest_start
            if abs(closest_end - end) < 5.0:
                end = closest_end
            
            adjusted.append({
                "start": start,
                "end": end,
                "topic": segment["topic"]
            })
        
        return adjusted

    def _merge_short_segments(self, segments: List[Dict]) -> List[Dict]:
        """Объединяет слишком короткие сегменты."""
        if not segments:
            return []
        
        merged = []
        current = segments[0].copy()
        
        for next_segment in segments[1:]:
            current_duration = current["end"] - current["start"]
            
            if current_duration < self.min_chapter_duration:
                # Объединяем с следующим сегментом
                current["end"] = next_segment["end"]
                current["topic"] = f"{current['topic']}_{next_segment['topic']}"
            else:
                # Сохраняем текущий и начинаем новый
                merged.append(current)
                current = next_segment.copy()
        
        merged.append(current)
        return merged

    def _generate_title(self, topic: str, chapter_num: int) -> str:
        """Генерирует заголовок главы."""
        # Простые правила для русских топиков
        topic_translations = {
            "silence": "Пауза",
            "general": "Общий контент",
            "intro": "Вступление",
            "dialog": "Диалог",
            "action": "Экшн",
            "comedy": "Комедия"
        }
        
        # Если топик содержит подчеркивания, разбиваем
        if "_" in topic:
            words = topic.split("_")
            topic_name = " ".join(words[:2])  # Берем первые 2 слова
        else:
            topic_name = topic_translations.get(topic, topic.capitalize())
        
        return f"Глава {chapter_num}: {topic_name}"

    def _generate_description(self, segment: Dict, timeline: List[Dict]) -> str:
        """Генерирует описание главы на основе timeline."""
        start_time = segment["start"]
        end_time = segment["end"]
        
        # Фильтруем timeline для данного сегмента
        segment_timeline = [
            point for point in timeline
            if start_time <= point["time"] <= end_time
        ]
        
        if not segment_timeline:
            return "Контент без особенностей"
        
        # Анализируем характеристики сегмента
        avg_motion = sum(p["motion"] for p in segment_timeline) / len(segment_timeline)
        avg_humor = sum(p["humor"] for p in segment_timeline) / len(segment_timeline)
        avg_sentiment = sum(abs(p["sentiment"]) for p in segment_timeline) / len(segment_timeline)
        has_dialogue = any(p["is_dialogue"] for p in segment_timeline)
        
        # Формируем описание
        characteristics = []
        
        if avg_motion > 0.5:
            characteristics.append("высокая активность")
        if avg_humor > 0.3:
            characteristics.append("юмористический контент")
        if avg_sentiment > 0.4:
            characteristics.append("эмоциональные моменты")
        if has_dialogue:
            characteristics.append("диалоги")
        
        if characteristics:
            return f"Сегмент с {', '.join(characteristics)}"
        else:
            return "Спокойный контент"