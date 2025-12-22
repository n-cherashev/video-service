from __future__ import annotations

from typing import Any, Dict, List, Literal, Sequence

import numpy as np
from transformers import Pipeline, pipeline

from handlers.base_handler import BaseHandler


class SentimentAnalysisHandler(BaseHandler):
    """
    Параметры
    ---------
    strategy : Literal["rule_based", "model"]
        Быстрый правиленный вариант или «модель» (HuggingFace).
    """

    def __init__(
        self,
        strategy: Literal["rule_based", "model"] = "rule_based",
        device: int | str = 0,  # 0 → GPU, -1 → CPU
    ) -> None:
        self.strategy = strategy
        self.device = device
        self._model: Pipeline | None = None
        if self.strategy == "model":
            self._model = pipeline(
                "sentiment-analysis", device=self.device if self.device >= 0 else -1
            )

    # ---------------------------------------------------------------------------

    def _rule_based_score(self, text: str) -> float:
        """
        Сначала на основе простого набора ключевых слов.
        Возвращает значение в диапазоне [-1.0, 1.0].
        """
        if not text:
            return 0.0

        pos_terms = {"good", "great", "happy", "joyful", "positive", "love", "awesome"}
        neg_terms = {"bad", "terrible", "sad", "unhappy", "negative", "hate", "awful"}

        words = set(text.lower().split())
        pos_cnt = len(words & pos_terms)
        neg_cnt = len(words & neg_terms)
        total = pos_cnt + neg_cnt

        if total == 0:
            return 0.0

        score = (pos_cnt - neg_cnt) / total
        return max(-1.0, min(1.0, score))

    def _model_score(self, text: str) -> float:
        """
        Бывает вариант использования HuggingFace модели.
        Возвращает [-1.0, 1.0].
        """
        if self._model is None:
            raise RuntimeError("Model pipeline not initialized")

        res = self._model(text, truncation=True)[0]
        label = res["label"].upper()
        prob = float(res["score"])

        if label == "POSITIVE":
            return max(0.0, 2 * prob - 1)
        return min(0.0, 1 - 2 * prob)

    # ---------------------------------------------------------------------------

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Вход: context['transcript_segments'] – список сегментов STT.

        Выход: добавляем `sentiment_timeline` и `sentiment_summary`.
        """
        segments: Sequence[Dict[str, Any]] = context.get("transcript_segments", [])
        if not segments:
            context["sentiment_timeline"] = []
            context["sentiment_summary"] = {"mean": 0.0, "min": 0.0, "max": 0.0}
            return context

        timeline: List[Dict[str, float]] = []

        for seg in segments:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            text = str(seg.get("text", "")).strip()
            if not text or end <= start:
                continue

            # Вычисляем балл
            if self.strategy == "rule_based":
                score = self._rule_based_score(text)
            elif self.strategy == "model":
                score = self._model_score(text)
            else:
                raise ValueError(f"Unsupported strategy: {self.strategy}")

            timeline.append({"time": (start + end) / 2, "score": score})

        if not timeline:
            context["sentiment_timeline"] = []
            context["sentiment_summary"] = {"mean": 0.0, "min": 0.0, "max": 0.0}
            return context

        timeline.sort(key=lambda d: d["time"])
        scores = np.array([d["score"] for d in timeline])

        context["sentiment_timeline"] = timeline
        context["sentiment_summary"] = {
            "mean": float(scores.mean()),
            "min": float(scores.min()),
            "max": float(scores.max()),
        }
        return context
