"""
Humor Detection Handler - определение юмора через ML-инфраструктуру.

НЕ использует hardcoded слова/маркеры!
Полагается на:
1. YAMNet laughter detection (из laughter_timeline)
2. Sentiment analysis (резкие эмоциональные переходы)
3. Audio features (энергия, громкость)
4. LLM refinement (опционально, для сложных случаев)
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional

from core.base_handler import BaseHandler


class HumorDetectionHandler(BaseHandler):
    """Детекция юмора через ML-сигналы.

    Комбинирует:
    1. Laughter detection (YAMNet) — прямой сигнал смеха
    2. Sentiment dynamics — резкие переходы эмоций часто = юмор
    3. Audio energy patterns — взрывы энергии после пауз

    НЕ использует hardcoded слова или маркеры!
    """

    def __init__(
        self,
        laughter_weight: float = 0.5,
        sentiment_weight: float = 0.3,
        energy_weight: float = 0.2,
        threshold: float = 0.3,
        sentiment_change_threshold: float = 0.4,
    ) -> None:
        self.laughter_weight = laughter_weight
        self.sentiment_weight = sentiment_weight
        self.energy_weight = energy_weight
        self.threshold = threshold
        self.sentiment_change_threshold = sentiment_change_threshold

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("[12] HumorDetectionHandler")

        # Получаем ML-сигналы из предыдущих хэндлеров
        laughter_timeline = context.get("laughter_timeline", [])
        sentiment_timeline = context.get("sentiment_timeline", [])
        audio_features = context.get("audio_features", {})
        transcript_segments = context.get("transcript_segments", [])

        duration = context.get("duration_seconds", 0.0)

        if not transcript_segments and not laughter_timeline:
            context["humor_scores"] = []
            context["humor_summary"] = {"mean": 0.0, "max": 0.0, "count_positive": 0}
            print("✓ Humor: 0 segments (no data)")
            return context

        # Вычисляем humor score для каждого сегмента или временной точки
        humor_scores = self._compute_humor_scores(
            transcript_segments=transcript_segments,
            laughter_timeline=laughter_timeline,
            sentiment_timeline=sentiment_timeline,
            audio_features=audio_features,
            duration=duration,
        )

        humor_scores.sort(key=lambda x: x["time"])

        scores = [h["score"] for h in humor_scores]
        humor_summary = {
            "mean": float(np.mean(scores)) if scores else 0.0,
            "max": float(np.max(scores)) if scores else 0.0,
            "count_positive": sum(1 for s in scores if s >= self.threshold),
        }

        context["humor_scores"] = humor_scores
        context["humor_summary"] = humor_summary

        print(f"✓ Humor: {len(humor_scores)} points, {humor_summary['count_positive']} funny (ML-based)")
        return context

    def _compute_humor_scores(
        self,
        transcript_segments: List[Dict[str, Any]],
        laughter_timeline: List[Dict[str, Any]],
        sentiment_timeline: List[Dict[str, Any]],
        audio_features: Dict[str, Any],
        duration: float,
    ) -> List[Dict[str, float]]:
        """Вычисляет humor score на основе ML-сигналов."""

        # Если есть transcript_segments — используем их как основу
        if transcript_segments:
            return self._score_segments(
                transcript_segments,
                laughter_timeline,
                sentiment_timeline,
                audio_features,
            )

        # Иначе — создаём timeline на основе laughter_timeline
        if laughter_timeline:
            return self._score_from_laughter(laughter_timeline, sentiment_timeline)

        return []

    def _score_segments(
        self,
        segments: List[Dict[str, Any]],
        laughter_timeline: List[Dict[str, Any]],
        sentiment_timeline: List[Dict[str, Any]],
        audio_features: Dict[str, Any],
    ) -> List[Dict[str, float]]:
        """Оценивает humor для каждого сегмента транскрипции."""

        # Создаём lookup для laughter по времени
        laughter_by_time = {
            int(item.get("time", 0)): item.get("prob", 0.0)
            for item in laughter_timeline
        }

        # Создаём lookup для sentiment
        sentiment_by_time = {}
        for item in sentiment_timeline:
            t = int(item.get("time", 0))
            sentiment_by_time[t] = item.get("score", 0.0)

        # Energy timeline
        energy_timeline = audio_features.get("energy", [])
        energy_by_time = {
            int(item.get("time", 0)): item.get("value", 0.0)
            for item in energy_timeline
        }

        results = []
        prev_sentiment = 0.0

        for seg in segments:
            start = float(seg.get("start", 0))
            end = float(seg.get("end", start))
            center_time = (start + end) / 2.0

            # 1. Laughter score — прямой сигнал от YAMNet
            laughter_score = self._get_max_in_range(
                laughter_by_time, int(start), int(end) + 1
            )

            # 2. Sentiment dynamics — резкие изменения
            sentiment_values = [
                sentiment_by_time.get(t, 0.0)
                for t in range(int(start), int(end) + 1)
            ]
            sentiment_change_score = self._compute_sentiment_dynamics(
                sentiment_values, prev_sentiment
            )
            if sentiment_values:
                prev_sentiment = sentiment_values[-1]

            # 3. Energy pattern — взрывы после пауз
            energy_score = self._compute_energy_pattern(
                energy_by_time, int(start), int(end)
            )

            # Комбинируем с весами
            humor_score = (
                self.laughter_weight * laughter_score +
                self.sentiment_weight * sentiment_change_score +
                self.energy_weight * energy_score
            )

            results.append({
                "time": center_time,
                "start": start,
                "end": end,
                "score": float(np.clip(humor_score, 0.0, 1.0)),
                "components": {
                    "laughter": laughter_score,
                    "sentiment_dynamics": sentiment_change_score,
                    "energy_pattern": energy_score,
                }
            })

        return results

    def _score_from_laughter(
        self,
        laughter_timeline: List[Dict[str, Any]],
        sentiment_timeline: List[Dict[str, Any]],
    ) -> List[Dict[str, float]]:
        """Создаёт humor scores напрямую из laughter timeline."""

        sentiment_by_time = {
            int(item.get("time", 0)): item.get("score", 0.0)
            for item in sentiment_timeline
        }

        results = []
        prev_sentiment = 0.0

        for item in laughter_timeline:
            t = float(item.get("time", 0))
            laughter_prob = float(item.get("prob", 0.0))

            # Sentiment change
            current_sentiment = sentiment_by_time.get(int(t), 0.0)
            sentiment_change = abs(current_sentiment - prev_sentiment)
            sentiment_score = min(1.0, sentiment_change / self.sentiment_change_threshold)
            prev_sentiment = current_sentiment

            humor_score = (
                self.laughter_weight * laughter_prob +
                self.sentiment_weight * sentiment_score
            )

            results.append({
                "time": t,
                "score": float(np.clip(humor_score, 0.0, 1.0)),
                "components": {
                    "laughter": laughter_prob,
                    "sentiment_dynamics": sentiment_score,
                }
            })

        return results

    def _get_max_in_range(
        self,
        time_dict: Dict[int, float],
        start: int,
        end: int,
    ) -> float:
        """Получает максимальное значение в временном диапазоне."""
        values = [
            time_dict.get(t, 0.0)
            for t in range(start, end)
        ]
        return max(values) if values else 0.0

    def _compute_sentiment_dynamics(
        self,
        sentiment_values: List[float],
        prev_sentiment: float,
    ) -> float:
        """Оценивает динамику sentiment — резкие изменения часто = юмор.

        Паттерны юмора:
        - Негатив -> Позитив (неожиданный поворот)
        - Нейтрал -> сильный Позитив (реакция на шутку)
        - Быстрые колебания (бэнтер, диалог)
        """
        if not sentiment_values:
            return 0.0

        # Изменение от предыдущего сегмента
        first_change = abs(sentiment_values[0] - prev_sentiment)

        # Внутренняя динамика сегмента
        if len(sentiment_values) > 1:
            diffs = np.abs(np.diff(sentiment_values))
            internal_dynamics = float(np.mean(diffs))
        else:
            internal_dynamics = 0.0

        # Переход neg->pos особенно значим
        neg_to_pos_bonus = 0.0
        if prev_sentiment < -0.2 and sentiment_values[-1] > 0.2:
            neg_to_pos_bonus = 0.3

        score = (
            min(1.0, first_change / self.sentiment_change_threshold) * 0.5 +
            min(1.0, internal_dynamics / 0.3) * 0.3 +
            neg_to_pos_bonus
        )

        return float(np.clip(score, 0.0, 1.0))

    def _compute_energy_pattern(
        self,
        energy_by_time: Dict[int, float],
        start: int,
        end: int,
    ) -> float:
        """Оценивает паттерн энергии — взрывы после пауз характерны для юмора.

        Паттерн: низкая энергия (setup) -> высокая энергия (punchline/reaction)
        """
        if not energy_by_time:
            return 0.0

        # Получаем энергию до и во время сегмента
        pre_energy = [
            energy_by_time.get(t, 0.0)
            for t in range(max(0, start - 3), start)
        ]
        segment_energy = [
            energy_by_time.get(t, 0.0)
            for t in range(start, end + 1)
        ]

        if not segment_energy:
            return 0.0

        avg_pre = float(np.mean(pre_energy)) if pre_energy else 0.3
        max_segment = float(np.max(segment_energy))

        # Контраст между pre и segment
        contrast = max_segment - avg_pre

        # Нормализуем
        score = max(0.0, contrast) / 0.5  # 0.5 — типичный большой контраст

        return float(np.clip(score, 0.0, 1.0))
