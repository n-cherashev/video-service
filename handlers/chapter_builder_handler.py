from __future__ import annotations

from bisect import bisect_left
from typing import Any, Dict, List

from handlers.base_handler import BaseHandler


class ChapterBuilderHandler(BaseHandler):
    """Формирует главы на основе топиков и границ сцен."""

    def __init__(self, min_chapter_duration: float = 30.0) -> None:
        self.min_chapter_duration = float(min_chapter_duration)

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        duration = float(context.get("duration_seconds", 0.0) or 0.0)
        topic_segments = context.get("topic_segments", []) or []
        scene_boundaries = sorted([float(x) for x in (context.get("scene_boundaries", [0.0]) or [0.0])])
        timeline = context.get("timeline", []) or []

        if not topic_segments:
            chapters = [{"start": 0.0, "end": duration, "title": "Полное видео", "description": "Весь контент"}]
        else:
            chapters = self._build_chapters(topic_segments, scene_boundaries, timeline, duration)

        context["chapters"] = chapters
        print(f"✓ Chapters: {len(chapters)} created")
        return context

    def _build_chapters(
        self,
        topic_segments: List[Dict[str, Any]],
        scene_boundaries: List[float],
        timeline: List[Dict[str, Any]],
        duration: float,
    ) -> List[Dict[str, Any]]:
        adjusted_segments = self._adjust_by_scenes(topic_segments, scene_boundaries)
        merged_segments = self._merge_short_segments(adjusted_segments)

        chapters: List[Dict[str, Any]] = []
        for i, seg in enumerate(merged_segments, 1):
            chapters.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "title": self._generate_title(str(seg["topic"]), i),
                "description": self._generate_description(seg, timeline),
            })
        return chapters

    def _nearest_boundary(self, boundaries: List[float], t: float) -> float:
        if not boundaries:
            return t
        pos = bisect_left(boundaries, t)
        if pos == 0:
            return boundaries[0]
        if pos >= len(boundaries):
            return boundaries[-1]
        before = boundaries[pos - 1]
        after = boundaries[pos]
        return after if abs(after - t) < abs(t - before) else before

    def _adjust_by_scenes(self, segments: List[Dict[str, Any]], boundaries: List[float]) -> List[Dict[str, Any]]:
        adjusted: List[Dict[str, Any]] = []
        for seg in segments:
            start = float(seg["start"])
            end = float(seg["end"])
            topic = seg.get("topic", "general")

            closest_start = self._nearest_boundary(boundaries, start)
            closest_end = self._nearest_boundary(boundaries, end)

            if abs(closest_start - start) < 5.0:
                start = closest_start
            if abs(closest_end - end) < 5.0:
                end = closest_end

            adjusted.append({"start": start, "end": end, "topic": topic})
        return adjusted

    def _merge_short_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not segments:
            return []

        merged: List[Dict[str, Any]] = []
        current = dict(segments[0])

        for nxt in segments[1:]:
            cur_len = float(current["end"]) - float(current["start"])
            if cur_len < self.min_chapter_duration:
                current["end"] = float(nxt["end"])
                current["topic"] = f"{current['topic']}_{nxt['topic']}"
            else:
                merged.append(current)
                current = dict(nxt)

        merged.append(current)
        return merged

    def _generate_title(self, topic: str, chapter_num: int) -> str:
        topic_translations = {
            "silence": "Пауза",
            "general": "Общий контент",
            "intro": "Вступление",
            "dialog": "Диалог",
            "action": "Экшн",
            "comedy": "Комедия",
        }

        if "_" in topic:
            words = topic.split("_")
            topic_name = " ".join(words[:2])
        else:
            topic_name = topic_translations.get(topic, topic.capitalize())

        return f"Глава {chapter_num}: {topic_name}"

    def _generate_description(self, segment: Dict[str, Any], timeline: List[Dict[str, Any]]) -> str:
        start_time = float(segment["start"])
        end_time = float(segment["end"])

        segment_tl = [p for p in timeline if start_time <= float(p["time"]) <= end_time]
        if not segment_tl:
            return "Контент без особенностей"

        avg_motion = sum(float(p["motion"]) for p in segment_tl) / len(segment_tl)
        avg_humor = sum(float(p["humor"]) for p in segment_tl) / len(segment_tl)
        avg_sent = sum(abs(float(p["sentiment"])) for p in segment_tl) / len(segment_tl)
        has_dialogue = any(bool(p.get("is_dialogue")) for p in segment_tl)

        parts = []
        if avg_motion > 0.5:
            parts.append("высокая активность")
        if avg_humor > 0.3:
            parts.append("юмористический контент")
        if avg_sent > 0.4:
            parts.append("эмоциональные моменты")
        if has_dialogue:
            parts.append("диалоги")

        return f"Сегмент с {', '.join(parts)}" if parts else "Спокойный контент"
