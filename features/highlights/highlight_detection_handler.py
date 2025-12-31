from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from core.base_handler import BaseHandler
from models.highlights import Highlight


Mode = Literal["rules", "top_interest", "multi_score"]


@dataclass
class ScoredCandidate:
    """Кандидат с полным breakdown скоринга."""
    start: float
    end: float
    score: float
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    type: str = "multi_score"
    reasons: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Candidate:
    """Legacy candidate для обратной совместимости."""
    start: float
    end: float
    score: float
    type: str = "top_interest"


class HighlightDetectionHandler(BaseHandler):
    """Обнаружение хайлайтов с многокомпонентным скорингом.

    Новая формула clip_score (согласно ТЗ):
    clip_score = (
        0.20 * hook +       # Качество первых 2-3 секунд
        0.15 * pace +       # Плотность событий
        0.15 * intensity +  # Интенсивность
        0.15 * clarity +    # Чистота диалога
        0.15 * emotion +    # Эмоциональная динамика
        0.10 * boundary +   # Близость к границам сцен
        0.10 * momentum     # Рост интенсивности к концу
    )
    """

    def __init__(
        self,
        min_duration_seconds: float = 30.0,
        max_duration_seconds: float = 90.0,
        *,
        mode: Mode = "multi_score",
        top_k: int = 6,
        peak_quantile: float = 0.90,
        snap_to_scenes: bool = True,
        snap_window_seconds: float = 5.0,
        merge_gap_seconds: float = 3.0,
        # Новые веса для 7-компонентного скоринга
        weight_hook: float = 0.20,
        weight_pace: float = 0.15,
        weight_intensity: float = 0.15,
        weight_clarity: float = 0.15,
        weight_emotion: float = 0.15,
        weight_boundary: float = 0.10,
        weight_momentum: float = 0.10,
        # Hook parameters
        hook_duration_seconds: float = 3.0,
        hook_min_threshold: float = 0.3,
        # Диверсификация
        overlap_strong_threshold: float = 0.65,
        overlap_medium_threshold: float = 0.35,
        overlap_strong_penalty: float = 0.3,
        overlap_medium_penalty: float = 0.7,
    ) -> None:
        self.min_duration = float(min_duration_seconds)
        self.max_duration = float(max_duration_seconds)
        self.mode = mode
        self.top_k = int(top_k)
        self.peak_quantile = float(peak_quantile)
        self.snap_to_scenes = bool(snap_to_scenes)
        self.snap_window_seconds = float(snap_window_seconds)
        self.merge_gap_seconds = float(merge_gap_seconds)

        # Веса для 7-компонентного скоринга
        self.weights = {
            "hook": float(weight_hook),
            "pace": float(weight_pace),
            "intensity": float(weight_intensity),
            "clarity": float(weight_clarity),
            "emotion": float(weight_emotion),
            "boundary": float(weight_boundary),
            "momentum": float(weight_momentum),
        }

        self.hook_duration = float(hook_duration_seconds)
        self.hook_min_threshold = float(hook_min_threshold)

        # Диверсификация
        self.overlap_strong_threshold = float(overlap_strong_threshold)
        self.overlap_medium_threshold = float(overlap_medium_threshold)
        self.overlap_strong_penalty = float(overlap_strong_penalty)
        self.overlap_medium_penalty = float(overlap_medium_penalty)

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("[13] HighlightDetectionHandler")

        timeline = context.get("timeline", []) or []
        if not timeline:
            context["highlights"] = []
            context["highlight_items"] = []
            return context

        duration = float(context.get("duration_seconds") or 0.0)
        scene_boundaries = sorted([float(x) for x in (context.get("scene_boundaries") or [0.0])])

        if self.mode == "multi_score":
            items = self._multi_score_highlights(timeline, duration, scene_boundaries)
        elif self.mode == "top_interest":
            items = self._top_interest_highlights(timeline, duration, scene_boundaries)
        else:
            items = []

        context["highlight_items"] = items
        context["highlights"] = [
            {
                "start": h.start,
                "end": h.end,
                "type": h.type,
                "score": h.score,
                "score_breakdown": h.score_breakdown if hasattr(h, 'score_breakdown') else {},
                "reasons": h.reasons if hasattr(h, 'reasons') else [],
            }
            for h in items
        ]

        print(f"✓ Highlights: {len(items)} clips detected")
        return context

    def _multi_score_highlights(
        self,
        timeline: List[Dict[str, Any]],
        duration: float,
        scene_boundaries: List[float],
    ) -> List[ScoredCandidate]:
        """Новый алгоритм с 7-компонентным скорингом."""

        # Извлечение временных рядов
        times = np.array([float(p["time"]) for p in timeline], dtype=float)
        interest = np.array([float(p.get("interest", 0.0) or 0.0) for p in timeline], dtype=float)
        motion = np.array([float(p.get("motion", 0.0) or 0.0) for p in timeline], dtype=float)
        loudness = np.array([float(p.get("audio_loudness", 0.0) or 0.0) for p in timeline], dtype=float)
        clarity = np.array([float(p.get("clarity", 0.5) or 0.5) for p in timeline], dtype=float)
        sentiment = np.array([float(p.get("sentiment", 0.0) or 0.0) for p in timeline], dtype=float)
        is_dialogue = np.array([bool(p.get("is_dialogue", False)) for p in timeline], dtype=bool)
        is_scene_boundary = np.array([bool(p.get("is_scene_boundary", False)) for p in timeline], dtype=bool)

        if interest.size < 3:
            return []

        # Генерация кандидатов из пиков interest
        thr = float(np.quantile(interest, self.peak_quantile))
        peaks = self._find_local_maxima(times, interest, thr)

        candidates: List[ScoredCandidate] = []

        # Генерируем окна вокруг пиков
        for t_peak in peaks:
            for win_size in [self.min_duration, (self.min_duration + self.max_duration) / 2, self.max_duration]:
                start = max(0.0, t_peak - win_size / 2.0)
                end = min(duration, start + win_size)
                start = max(0.0, end - win_size)

                if self.snap_to_scenes and scene_boundaries:
                    start = self._snap_time(start, scene_boundaries, self.snap_window_seconds)
                    end = self._snap_time(end, scene_boundaries, self.snap_window_seconds)

                # Проверка минимальной длины
                if end - start < self.min_duration * 0.5:
                    continue

                # Вычисление 7-компонентного скора
                scored = self._calculate_multi_score(
                    start, end, times,
                    interest, motion, loudness, clarity, sentiment,
                    is_dialogue, is_scene_boundary, scene_boundaries
                )

                if scored:
                    candidates.append(scored)

        # Fallback: скользящее окно если нет пиков
        if not candidates:
            start, end = 0.0, min(duration, self.min_duration)
            scored = self._calculate_multi_score(
                start, end, times,
                interest, motion, loudness, clarity, sentiment,
                is_dialogue, is_scene_boundary, scene_boundaries
            )
            if scored:
                candidates.append(scored)

        # Диверсификация и выбор финальных
        selected = self._diversify_and_select(candidates)

        # Сортировка по времени
        selected.sort(key=lambda c: c.start)

        return selected

    def _calculate_multi_score(
        self,
        start: float,
        end: float,
        times: np.ndarray,
        interest: np.ndarray,
        motion: np.ndarray,
        loudness: np.ndarray,
        clarity: np.ndarray,
        sentiment: np.ndarray,
        is_dialogue: np.ndarray,
        is_scene_boundary: np.ndarray,
        scene_boundaries: List[float],
    ) -> Optional[ScoredCandidate]:
        """Вычисляет 7-компонентный скор для кандидата."""

        mask = (times >= start) & (times <= end)
        if not np.any(mask):
            return None

        # 1. HOOK SCORE - качество первых 2-3 секунд
        hook_end = start + self.hook_duration
        hook_mask = mask & (times <= hook_end)
        if np.any(hook_mask):
            hook_interest = np.mean(interest[hook_mask])
            hook_motion = np.mean(motion[hook_mask])
            hook_loudness = np.mean(loudness[hook_mask])
            hook_dialogue = np.mean(is_dialogue[hook_mask])

            # Штраф за слабый старт
            if hook_interest < self.hook_min_threshold and hook_motion < 0.1 and hook_dialogue < 0.1:
                hook_score = 0.2
            else:
                hook_score = min(1.0, (hook_interest + hook_motion + hook_loudness) / 2.0)
        else:
            hook_score = 0.3

        # 2. PACE SCORE - плотность событий
        motion_window = motion[mask]
        loudness_window = loudness[mask]
        motion_peaks = self._count_local_peaks(motion_window, threshold=0.5)
        audio_peaks = self._count_local_peaks(loudness_window, threshold=0.5)

        expected_peaks = max(1, len(motion_window) // 10)  # ~1 пик на 10 секунд
        peak_density = (motion_peaks + audio_peaks) / (expected_peaks * 2)

        motion_var = np.std(motion_window) if len(motion_window) > 1 else 0
        audio_var = np.std(loudness_window) if len(loudness_window) > 1 else 0
        variability = (motion_var + audio_var) / 2.0

        pace_score = min(1.0, (peak_density * 0.6 + variability * 0.4))

        # 3. INTENSITY SCORE - интенсивность движения и звука
        avg_motion = np.mean(motion[mask])
        avg_loudness = np.mean(loudness[mask])
        max_motion = np.max(motion[mask])
        max_loudness = np.max(loudness[mask])

        intensity_score = min(1.0, (0.6 * avg_motion + 0.4 * avg_loudness + 0.3 * (max_motion + max_loudness) / 2) / 1.3)

        # 4. CLARITY SCORE - чистота диалога
        dialogue_ratio = np.mean(is_dialogue[mask])
        clarity_mean = np.mean(clarity[mask])

        clarity_score = clarity_mean
        if dialogue_ratio < 0.3:
            clarity_score *= 0.8  # Штраф за мало диалога

        # 5. EMOTION SCORE - эмоциональная динамика
        sentiment_window = sentiment[mask]
        avg_abs_sentiment = np.mean(np.abs(sentiment_window))

        if len(sentiment_window) > 1:
            sentiment_changes = np.abs(np.diff(sentiment_window))
            sharp_changes = np.sum(sentiment_changes > 0.5) / len(sentiment_changes)
            sentiment_range = np.max(sentiment_window) - np.min(sentiment_window)
        else:
            sharp_changes = 0
            sentiment_range = 0

        emotion_score = min(1.0, (avg_abs_sentiment + sharp_changes * 0.3 + sentiment_range * 0.2))

        # 6. BOUNDARY SCORE - близость к границам сцен
        boundary_score = 0.0

        scene_bound_times = times[is_scene_boundary]
        if len(scene_bound_times) > 0:
            start_dist = np.min(np.abs(scene_bound_times - start))
            end_dist = np.min(np.abs(scene_bound_times - end))

            if start_dist <= 3.0:
                boundary_score += 0.5
            if end_dist <= 3.0:
                boundary_score += 0.5

        # Бонус за начало/конец диалога
        dialogue_changes = np.diff(is_dialogue.astype(int))
        dialogue_boundaries = times[1:][dialogue_changes != 0]

        if len(dialogue_boundaries) > 0:
            start_dial_dist = np.min(np.abs(dialogue_boundaries - start))
            end_dial_dist = np.min(np.abs(dialogue_boundaries - end))

            if start_dial_dist <= 2.0:
                boundary_score += 0.25
            if end_dial_dist <= 2.0:
                boundary_score += 0.25

        boundary_score = min(1.0, boundary_score)

        # 7. MOMENTUM SCORE - рост интенсивности к концу
        mid_point = (start + end) / 2
        first_half_mask = mask & (times < mid_point)
        second_half_mask = mask & (times >= mid_point)

        if np.any(first_half_mask) and np.any(second_half_mask):
            first_intensity = np.mean(interest[first_half_mask])
            second_intensity = np.mean(interest[second_half_mask])

            if second_intensity > first_intensity * 1.2:
                momentum_score = 1.0
            elif first_intensity > 0:
                momentum_score = min(1.0, second_intensity / first_intensity)
            else:
                momentum_score = 0.5
        else:
            momentum_score = 0.5

        # ИТОГОВЫЙ CLIP_SCORE
        clip_score = (
            self.weights["hook"] * hook_score +
            self.weights["pace"] * pace_score +
            self.weights["intensity"] * intensity_score +
            self.weights["clarity"] * clarity_score +
            self.weights["emotion"] * emotion_score +
            self.weights["boundary"] * boundary_score +
            self.weights["momentum"] * momentum_score
        )

        score_breakdown = {
            "hook": round(hook_score, 3),
            "pace": round(pace_score, 3),
            "intensity": round(intensity_score, 3),
            "clarity": round(clarity_score, 3),
            "emotion": round(emotion_score, 3),
            "boundary": round(boundary_score, 3),
            "momentum": round(momentum_score, 3),
        }

        # Генерация reasons
        reasons = self._generate_reasons(score_breakdown)

        return ScoredCandidate(
            start=float(start),
            end=float(end),
            score=float(clip_score),
            score_breakdown=score_breakdown,
            type="multi_score",
            reasons=reasons,
        )

    def _count_local_peaks(self, signal: np.ndarray, threshold: float = 0.5) -> int:
        """Подсчитывает локальные максимумы."""
        if len(signal) < 3:
            return 0

        peaks = 0
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] >= threshold:
                peaks += 1

        return peaks

    def _generate_reasons(self, breakdown: Dict[str, float]) -> List[str]:
        """Генерирует причины высокого скора."""
        reasons = []

        if breakdown.get("hook", 0) >= 0.7:
            reasons.append("Strong hook")
        if breakdown.get("pace", 0) >= 0.7:
            reasons.append("High pace")
        if breakdown.get("intensity", 0) >= 0.7:
            reasons.append("High intensity")
        if breakdown.get("clarity", 0) >= 0.7:
            reasons.append("Clear dialogue")
        if breakdown.get("emotion", 0) >= 0.6:
            reasons.append("Emotional dynamics")
        if breakdown.get("boundary", 0) >= 0.4:
            reasons.append("Good boundaries")
        if breakdown.get("momentum", 0) >= 0.7:
            reasons.append("Building momentum")

        return reasons if reasons else ["Automated selection"]

    def _diversify_and_select(self, candidates: List[ScoredCandidate]) -> List[ScoredCandidate]:
        """Выбирает финальные клипы с диверсификацией."""
        if not candidates:
            return []

        # Сортируем по скору
        candidates.sort(key=lambda x: x.score, reverse=True)

        selected: List[ScoredCandidate] = []

        for candidate in candidates:
            if len(selected) >= self.top_k:
                break

            # Вычисляем штрафы за пересечения
            adjusted_score = candidate.score

            for sel in selected:
                overlap = self._calculate_overlap_ratio(candidate, sel)

                if overlap > self.overlap_strong_threshold:
                    adjusted_score *= self.overlap_strong_penalty
                elif overlap > self.overlap_medium_threshold:
                    adjusted_score *= self.overlap_medium_penalty

            # Добавляем если скор всё ещё достаточно высок
            if adjusted_score > 0.1:
                selected.append(candidate)

        return selected

    def _calculate_overlap_ratio(self, c1: ScoredCandidate, c2: ScoredCandidate) -> float:
        """Вычисляет коэффициент пересечения."""
        inter_start = max(c1.start, c2.start)
        inter_end = min(c1.end, c2.end)

        if inter_end <= inter_start:
            return 0.0

        intersection = inter_end - inter_start
        min_duration = min(c1.duration, c2.duration)

        return intersection / min_duration if min_duration > 0 else 0.0

    # === Legacy методы для обратной совместимости ===

    def _top_interest_highlights(
        self,
        timeline: List[Dict[str, Any]],
        duration: float,
        scene_boundaries: List[float],
    ) -> List[Highlight]:
        """Legacy метод для режима top_interest."""
        times = np.array([float(p["time"]) for p in timeline], dtype=float)
        interest = np.array([float(p.get("interest", 0.0) or 0.0) for p in timeline], dtype=float)
        if interest.size < 3:
            return []

        thr = float(np.quantile(interest, self.peak_quantile))
        peaks = self._find_local_maxima(times, interest, thr)

        cands: List[Candidate] = []
        for t_peak in peaks:
            start = max(0.0, t_peak - self.min_duration / 2.0)
            end = min(duration, start + self.min_duration)
            start = max(0.0, end - self.min_duration)

            max_d = max(self.max_duration, self.min_duration)
            if (end - start) > max_d:
                end = start + max_d

            if self.snap_to_scenes and scene_boundaries:
                start = self._snap_time(start, scene_boundaries, self.snap_window_seconds)
                end = self._snap_time(end, scene_boundaries, self.snap_window_seconds)

            score = float(self._mean_interest_in_window(times, interest, start, end))
            cands.append(Candidate(start=float(start), end=float(end), score=score))

        if not cands:
            start, end, score = self._best_sliding_window(times, interest, duration)
            if self.snap_to_scenes and scene_boundaries:
                start = self._snap_time(start, scene_boundaries, self.snap_window_seconds)
                end = self._snap_time(end, scene_boundaries, self.snap_window_seconds)
            cands = [Candidate(start=start, end=end, score=score)]

        cands.sort(key=lambda c: c.score, reverse=True)
        selected: List[Candidate] = []
        for c in cands:
            if len(selected) >= self.top_k:
                break
            if not self._overlaps_any(c, selected):
                selected.append(c)

        selected.sort(key=lambda c: c.start)
        merged = self._merge_close(selected, times, interest)

        return [Highlight(start=c.start, end=c.end, type="top_interest", score=c.score) for c in merged]

    def _merge_close(self, items: List[Candidate], times: np.ndarray, values: np.ndarray) -> List[Candidate]:
        if not items:
            return []
        out = [items[0]]
        for c in items[1:]:
            prev = out[-1]
            if c.start <= prev.end + self.merge_gap_seconds:
                new_start = prev.start
                new_end = max(prev.end, c.end)
                new_score = float(self._mean_interest_in_window(times, values, new_start, new_end))
                out[-1] = Candidate(start=new_start, end=new_end, score=new_score, type=prev.type)
            else:
                out.append(c)
        return out

    def _find_local_maxima(self, times: np.ndarray, values: np.ndarray, thr: float) -> List[float]:
        peaks: List[float] = []
        for i in range(1, len(values) - 1):
            if values[i] >= thr and values[i] >= values[i - 1] and values[i] >= values[i + 1]:
                peaks.append(float(times[i]))
        return peaks

    def _mean_interest_in_window(self, times: np.ndarray, values: np.ndarray, start: float, end: float) -> float:
        mask = (times >= start) & (times <= end)
        return float(np.mean(values[mask])) if np.any(mask) else 0.0

    def _best_sliding_window(self, times: np.ndarray, values: np.ndarray, duration: float) -> tuple[float, float, float]:
        win = self.min_duration
        best_score = -1.0
        best_start = 0.0
        best_end = min(duration, win)

        for i in range(len(times)):
            start = float(times[i])
            end = start + win
            if end > duration:
                break
            score = self._mean_interest_in_window(times, values, start, end)
            if score > best_score:
                best_score = score
                best_start, best_end = start, end

        return float(best_start), float(best_end), float(max(best_score, 0.0))

    def _snap_time(self, t: float, boundaries: List[float], window: float) -> float:
        nearest = min(boundaries, key=lambda b: abs(float(b) - t))
        return float(nearest) if abs(float(nearest) - t) <= window else float(t)

    def _overlaps_any(self, c: Candidate, selected: List[Candidate]) -> bool:
        for s in selected:
            if not (c.end <= s.start or c.start >= s.end):
                return True
        return False
