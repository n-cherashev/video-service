from __future__ import annotations

from bisect import bisect_left
from typing import Any, Dict, List

from core.base_handler import BaseHandler


class ChapterBuilderHandler(BaseHandler):
    """Формирует главы на основе топиков, сцен и highlights."""

    def __init__(
        self,
        min_chapter_duration: float = 60.0,
        max_chapters: int = 10,
        use_scenes_fallback: bool = True,
    ) -> None:
        self.min_chapter_duration = float(min_chapter_duration)
        self.max_chapters = max_chapters
        self.use_scenes_fallback = use_scenes_fallback

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("[14] ChapterBuilderHandler")

        duration = float(context.get("duration_seconds", 0.0) or 0.0)
        topic_segments = context.get("topic_segments", []) or []
        scenes = context.get("scenes", []) or []
        scene_boundaries = sorted([float(x) for x in (context.get("scene_boundaries", [0.0]) or [0.0])])
        timeline = context.get("timeline", []) or []
        highlights = context.get("highlights", []) or []

        if topic_segments:
            # Используем топики если есть
            chapters = self._build_from_topics(topic_segments, scene_boundaries, timeline, duration)
        elif self.use_scenes_fallback and scenes:
            # Fallback: создаём главы из сцен
            chapters = self._build_from_scenes(scenes, timeline, duration)
        elif self.use_scenes_fallback and len(scene_boundaries) > 2:
            # Fallback: создаём главы из границ сцен
            chapters = self._build_from_boundaries(scene_boundaries, timeline, duration)
        else:
            # Последний fallback: одна глава
            chapters = [{"start": 0.0, "end": duration, "title": "Полное видео", "description": "Весь контент"}]

        # Ограничиваем количество глав
        if len(chapters) > self.max_chapters:
            chapters = self._merge_to_limit(chapters, self.max_chapters)

        context["chapters"] = chapters
        print(f"✓ Chapters: {len(chapters)} created")
        return context

    def _build_from_scenes(
        self,
        scenes: List[Dict[str, Any]],
        timeline: List[Dict[str, Any]],
        duration: float,
    ) -> List[Dict[str, Any]]:
        """Создаёт главы из сцен, объединяя короткие."""
        if not scenes:
            return [{"start": 0.0, "end": duration, "title": "Полное видео", "description": "Весь контент"}]

        # Группируем сцены в главы по min_chapter_duration
        chapters = []
        current_start = 0.0
        current_scenes = []

        for scene in scenes:
            scene_start = float(scene.get("start", 0))
            scene_end = float(scene.get("end", duration))
            current_scenes.append(scene)

            chapter_duration = scene_end - current_start

            # Если накопили достаточно длинную главу
            if chapter_duration >= self.min_chapter_duration:
                chapters.append({
                    "start": current_start,
                    "end": scene_end,
                    "scenes_count": len(current_scenes),
                })
                current_start = scene_end
                current_scenes = []

        # Добавляем оставшиеся сцены
        if current_scenes:
            last_end = float(scenes[-1].get("end", duration))
            if chapters:
                # Присоединяем к последней главе если слишком короткая
                if last_end - current_start < self.min_chapter_duration / 2:
                    chapters[-1]["end"] = last_end
                    chapters[-1]["scenes_count"] += len(current_scenes)
                else:
                    chapters.append({
                        "start": current_start,
                        "end": last_end,
                        "scenes_count": len(current_scenes),
                    })
            else:
                chapters.append({
                    "start": current_start,
                    "end": last_end,
                    "scenes_count": len(current_scenes),
                })

        # Генерируем названия и описания
        result = []
        for i, chapter in enumerate(chapters, 1):
            result.append({
                "start": chapter["start"],
                "end": chapter["end"],
                "title": f"Часть {i}",
                "description": self._generate_description_from_timeline(
                    chapter["start"], chapter["end"], timeline
                ),
            })

        return result

    def _build_from_boundaries(
        self,
        boundaries: List[float],
        timeline: List[Dict[str, Any]],
        duration: float,
    ) -> List[Dict[str, Any]]:
        """Создаёт главы из границ сцен."""
        if len(boundaries) < 2:
            return [{"start": 0.0, "end": duration, "title": "Полное видео", "description": "Весь контент"}]

        # Создаём сегменты из границ
        segments = []
        for i in range(len(boundaries) - 1):
            segments.append({
                "start": boundaries[i],
                "end": boundaries[i + 1],
            })

        # Добавляем последний сегмент до конца видео
        if boundaries[-1] < duration - 1:
            segments.append({
                "start": boundaries[-1],
                "end": duration,
            })

        # Объединяем короткие сегменты
        merged = self._merge_short_segments_simple(segments)

        # Генерируем названия
        result = []
        for i, seg in enumerate(merged, 1):
            result.append({
                "start": seg["start"],
                "end": seg["end"],
                "title": f"Часть {i}",
                "description": self._generate_description_from_timeline(
                    seg["start"], seg["end"], timeline
                ),
            })

        return result

    def _merge_short_segments_simple(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Объединяет короткие сегменты."""
        if not segments:
            return []

        merged = []
        current = dict(segments[0])

        for nxt in segments[1:]:
            cur_len = current["end"] - current["start"]
            if cur_len < self.min_chapter_duration:
                current["end"] = nxt["end"]
            else:
                merged.append(current)
                current = dict(nxt)

        merged.append(current)
        return merged

    def _merge_to_limit(self, chapters: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Объединяет главы до заданного лимита."""
        while len(chapters) > limit:
            # Находим самую короткую главу
            min_idx = 0
            min_duration = float('inf')
            for i, ch in enumerate(chapters):
                dur = ch["end"] - ch["start"]
                if dur < min_duration:
                    min_duration = dur
                    min_idx = i

            # Объединяем с соседней
            if min_idx == 0:
                # Объединяем с следующей
                chapters[1]["start"] = chapters[0]["start"]
                chapters[1]["title"] = chapters[0]["title"]
                chapters.pop(0)
            elif min_idx == len(chapters) - 1:
                # Объединяем с предыдущей
                chapters[-2]["end"] = chapters[-1]["end"]
                chapters.pop()
            else:
                # Объединяем с более короткой соседней
                prev_dur = chapters[min_idx - 1]["end"] - chapters[min_idx - 1]["start"]
                next_dur = chapters[min_idx + 1]["end"] - chapters[min_idx + 1]["start"]

                if prev_dur <= next_dur:
                    chapters[min_idx - 1]["end"] = chapters[min_idx]["end"]
                    chapters.pop(min_idx)
                else:
                    chapters[min_idx + 1]["start"] = chapters[min_idx]["start"]
                    chapters[min_idx + 1]["title"] = chapters[min_idx]["title"]
                    chapters.pop(min_idx)

        return chapters

    def _build_from_topics(
        self,
        topic_segments: List[Dict[str, Any]],
        scene_boundaries: List[float],
        timeline: List[Dict[str, Any]],
        duration: float,
    ) -> List[Dict[str, Any]]:
        """Создаёт главы из топиков (оригинальная логика)."""
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
        return self._generate_description_from_timeline(
            float(segment["start"]),
            float(segment["end"]),
            timeline
        )

    def _generate_description_from_timeline(
        self,
        start_time: float,
        end_time: float,
        timeline: List[Dict[str, Any]],
    ) -> str:
        """Генерирует описание на основе timeline."""
        segment_tl = [p for p in timeline if start_time <= float(p.get("time", 0)) <= end_time]
        if not segment_tl:
            return "Контент без особенностей"

        avg_motion = sum(float(p.get("motion", 0)) for p in segment_tl) / len(segment_tl)
        avg_interest = sum(float(p.get("interest", 0)) for p in segment_tl) / len(segment_tl)
        avg_clarity = sum(float(p.get("clarity", 0)) for p in segment_tl) / len(segment_tl)
        has_dialogue = any(bool(p.get("is_dialogue")) for p in segment_tl)
        has_loud = any(bool(p.get("has_loud_sound")) for p in segment_tl)

        parts = []
        if avg_motion > 0.3:
            parts.append("активные сцены")
        if avg_interest > 0.4:
            parts.append("интересный контент")
        if avg_clarity > 0.5:
            parts.append("чёткая речь")
        if has_dialogue:
            parts.append("диалоги")
        if has_loud:
            parts.append("яркие звуки")

        duration = end_time - start_time
        duration_str = f"{int(duration // 60)}:{int(duration % 60):02d}"

        if parts:
            return f"{duration_str} — {', '.join(parts)}"
        return f"{duration_str} — спокойный контент"
