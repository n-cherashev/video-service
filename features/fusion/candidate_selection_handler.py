from __future__ import annotations

from typing import Any, Dict, List, Set
from dataclasses import dataclass, field
import numpy as np
from core.base_handler import BaseHandler
from models.viral_moments import Candidate, ViralMomentsConfig


@dataclass
class AnchorConfig:
    """Конфигурация для детекции якорей."""
    # Персентили для порогов
    interest_percentile: float = 0.80  # top 20%
    motion_percentile: float = 0.80
    audio_percentile: float = 0.80

    # Порог для sentiment changes
    sentiment_change_threshold: float = 0.5

    # Минимальное расстояние между якорями одного типа (секунды)
    min_anchor_distance: float = 5.0

    # Типы якорей для детекции
    enabled_anchor_types: List[str] = field(default_factory=lambda: [
        "interest_peak",
        "motion_peak",
        "audio_peak",
        "scene_boundary",
        "dialogue_transition",
        "sentiment_change",
    ])


class CandidateSelectionHandler(BaseHandler):
    """Генерирует кандидатов для виральных моментов на основе 6 типов якорей.

    ЯКОРЯ (согласно ТЗ):
    1. Interest peaks (top 20% локальных максимумов)
    2. Motion peaks
    3. Loudness peaks + audio events
    4. Scene boundaries
    5. Dialogue transitions (False→True)
    6. Sentiment changes (|Δsentiment| > 0.5)

    ГЕНЕРАЦИЯ КАНДИДАТОВ:
    Для каждого якоря генерируем окна разной длительности и позиционирования.
    """

    def __init__(
        self,
        config: ViralMomentsConfig = None,
        anchor_config: AnchorConfig = None,
    ):
        self.config = config or ViralMomentsConfig()
        self.anchor_config = anchor_config or AnchorConfig()

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("[Viral Candidates] CandidateSelectionHandler")

        timeline = context.get("timeline", [])
        duration = context.get("duration_seconds", 0.0)

        if not timeline or duration <= 0:
            context["viral_candidates"] = []
            context["anchors"] = []
            context["anchor_summary"] = {}
            print("✓ Viral candidates: 0 (no timeline)")
            return context

        # Детекция якорей (6 типов)
        anchors = self._detect_all_anchors(timeline)

        # Дедупликация близких якорей
        anchors = self._deduplicate_anchors(anchors)

        # Генерация кандидатов
        candidates = self._generate_candidates(anchors, duration)

        # Summary по типам якорей
        anchor_summary = self._summarize_anchors(anchors)

        context["viral_candidates"] = [self._candidate_to_dict(c) for c in candidates]
        context["anchors"] = anchors
        context["anchor_summary"] = anchor_summary

        print(f"✓ Viral candidates: {len(candidates)} from {len(anchors)} anchors")
        print(f"  Anchor breakdown: {anchor_summary}")

        return context

    def _detect_all_anchors(self, timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Детектирует все 6 типов якорей."""
        anchors = []

        # Извлекаем временные ряды
        times = np.array([item.get("time", 0) for item in timeline], dtype=float)
        interests = np.array([item.get("interest", 0) for item in timeline], dtype=float)
        motions = np.array([item.get("motion", 0) for item in timeline], dtype=float)
        audio_loudness = np.array([item.get("audio_loudness", 0) for item in timeline], dtype=float)
        sentiments = np.array([item.get("sentiment", 0) for item in timeline], dtype=float)
        is_dialogue = np.array([item.get("is_dialogue", False) for item in timeline], dtype=bool)
        is_scene_boundary = np.array([item.get("is_scene_boundary", False) for item in timeline], dtype=bool)
        has_loud_sound = np.array([item.get("has_loud_sound", False) for item in timeline], dtype=bool)

        enabled = set(self.anchor_config.enabled_anchor_types)

        # 1. INTEREST PEAKS (локальные максимумы в top 20%)
        if "interest_peak" in enabled:
            anchors.extend(
                self._detect_peak_anchors(times, interests, "interest_peak",
                                         self.anchor_config.interest_percentile)
            )

        # 2. MOTION PEAKS
        if "motion_peak" in enabled:
            anchors.extend(
                self._detect_peak_anchors(times, motions, "motion_peak",
                                         self.anchor_config.motion_percentile)
            )

        # 3. AUDIO/LOUDNESS PEAKS + loud sound events
        if "audio_peak" in enabled:
            anchors.extend(
                self._detect_peak_anchors(times, audio_loudness, "audio_peak",
                                         self.anchor_config.audio_percentile)
            )
            # Добавляем loud sound events
            for i, (t, loud) in enumerate(zip(times, has_loud_sound)):
                if loud:
                    anchors.append({
                        "time": float(t),
                        "type": "audio_peak",
                        "subtype": "loud_sound",
                        "value": 1.0,
                        "index": i,
                    })

        # 4. SCENE BOUNDARIES
        if "scene_boundary" in enabled:
            for i, (t, is_boundary) in enumerate(zip(times, is_scene_boundary)):
                if is_boundary:
                    anchors.append({
                        "time": float(t),
                        "type": "scene_boundary",
                        "subtype": "scene_start",
                        "value": 1.0,
                        "index": i,
                    })

        # 5. DIALOGUE TRANSITIONS (False → True)
        if "dialogue_transition" in enabled:
            anchors.extend(
                self._detect_dialogue_transitions(times, is_dialogue)
            )

        # 6. SENTIMENT CHANGES (|Δsentiment| > threshold)
        if "sentiment_change" in enabled:
            anchors.extend(
                self._detect_sentiment_changes(times, sentiments)
            )

        return anchors

    def _detect_peak_anchors(
        self,
        times: np.ndarray,
        values: np.ndarray,
        anchor_type: str,
        percentile: float,
    ) -> List[Dict[str, Any]]:
        """Детектирует локальные максимумы выше порога."""
        anchors = []

        if len(values) < 3:
            return anchors

        threshold = float(np.percentile(values, percentile * 100))

        for i in range(1, len(values) - 1):
            # Локальный максимум
            if values[i] >= values[i-1] and values[i] >= values[i+1]:
                if values[i] >= threshold:
                    anchors.append({
                        "time": float(times[i]),
                        "type": anchor_type,
                        "subtype": "local_max",
                        "value": float(values[i]),
                        "index": i,
                    })

        return anchors

    def _detect_dialogue_transitions(
        self,
        times: np.ndarray,
        is_dialogue: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Детектирует переходы False → True в диалоге."""
        anchors = []

        prev_dialogue = False
        for i, (t, curr_dialogue) in enumerate(zip(times, is_dialogue)):
            if curr_dialogue and not prev_dialogue:
                anchors.append({
                    "time": float(t),
                    "type": "dialogue_transition",
                    "subtype": "dialogue_start",
                    "value": 1.0,
                    "index": i,
                })
            prev_dialogue = curr_dialogue

        return anchors

    def _detect_sentiment_changes(
        self,
        times: np.ndarray,
        sentiments: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Детектирует резкие смены sentiment."""
        anchors = []

        if len(sentiments) < 2:
            return anchors

        threshold = self.anchor_config.sentiment_change_threshold

        for i in range(1, len(sentiments)):
            delta = abs(sentiments[i] - sentiments[i-1])
            if delta >= threshold:
                # Определяем направление
                if sentiments[i] > sentiments[i-1]:
                    subtype = "sentiment_rise"
                else:
                    subtype = "sentiment_drop"

                anchors.append({
                    "time": float(times[i]),
                    "type": "sentiment_change",
                    "subtype": subtype,
                    "value": float(delta),
                    "index": i,
                })

        return anchors

    def _deduplicate_anchors(self, anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Удаляет близкие якоря одного типа, сохраняя с максимальным value."""
        if not anchors:
            return anchors

        # Группируем по типу
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for anchor in anchors:
            anchor_type = anchor["type"]
            if anchor_type not in by_type:
                by_type[anchor_type] = []
            by_type[anchor_type].append(anchor)

        result = []
        min_dist = self.anchor_config.min_anchor_distance

        for anchor_type, type_anchors in by_type.items():
            # Сортируем по времени
            type_anchors.sort(key=lambda x: x["time"])

            # Жадный выбор с дедупликацией
            selected = []
            for anchor in type_anchors:
                # Проверяем расстояние до уже выбранных
                can_add = True
                for i, sel in enumerate(selected):
                    dist = abs(anchor["time"] - sel["time"])
                    if dist < min_dist:
                        # Заменяем если новый лучше
                        if anchor["value"] > sel["value"]:
                            selected[i] = anchor
                        can_add = False
                        break

                if can_add:
                    selected.append(anchor)

            result.extend(selected)

        # Сортируем по времени
        result.sort(key=lambda x: x["time"])
        return result

    def _generate_candidates(
        self,
        anchors: List[Dict[str, Any]],
        duration: float,
    ) -> List[Candidate]:
        """Генерирует кандидатов вокруг якорей с разными параметрами."""
        candidates = []
        seen: Set[tuple] = set()

        for anchor in anchors:
            anchor_time = anchor["time"]
            anchor_type = anchor["type"]
            anchor_value = anchor.get("value", 1.0)

            # Генерируем окна разной длительности
            for candidate_duration in self.config.candidate_durations:
                # Генерируем разные позиционирования
                for position in self.config.anchor_positions:
                    start, end = self._calculate_window(
                        anchor_time, candidate_duration, position, duration
                    )

                    # Проверка минимальной длины
                    if end - start < candidate_duration * 0.5:
                        continue

                    # Дедупликация по (start, end) с точностью 0.5 сек
                    key = (round(start * 2) / 2, round(end * 2) / 2)
                    if key in seen:
                        continue
                    seen.add(key)

                    candidates.append(Candidate(
                        start=start,
                        end=end,
                        anchor_time=anchor_time,
                        anchor_type=anchor_type,
                        duration=end - start,
                    ))

        return candidates

    def _calculate_window(
        self,
        anchor_time: float,
        duration: float,
        position: str,
        video_duration: float,
    ) -> tuple[float, float]:
        """Вычисляет окно кандидата относительно якоря.

        Позиции:
        - "center": якорь в центре окна
        - "start": якорь в начале окна (hook position)
        - "end": якорь в конце окна (payoff position)
        """
        if position == "center":
            start = anchor_time - duration / 2
            end = anchor_time + duration / 2
        elif position == "start":
            start = anchor_time
            end = anchor_time + duration
        elif position == "end":
            start = anchor_time - duration
            end = anchor_time
        else:
            raise ValueError(f"Unknown position: {position}")

        # Обрезка по границам видео
        start = max(0.0, start)
        end = min(video_duration, end)

        # Корректировка если окно стало слишком коротким
        actual_duration = end - start
        if actual_duration < duration * 0.5:
            if start == 0:
                end = min(duration, video_duration)
            elif end == video_duration:
                start = max(0, video_duration - duration)

        return start, end

    def _candidate_to_dict(self, c: Candidate) -> Dict[str, Any]:
        """Конвертирует Candidate в dict."""
        return {
            "start": c.start,
            "end": c.end,
            "anchor_time": c.anchor_time,
            "anchor_type": c.anchor_type,
            "duration": c.duration,
        }

    def _summarize_anchors(self, anchors: List[Dict[str, Any]]) -> Dict[str, int]:
        """Создает summary по типам якорей."""
        summary: Dict[str, int] = {}
        for anchor in anchors:
            anchor_type = anchor["type"]
            summary[anchor_type] = summary.get(anchor_type, 0) + 1
        return summary
