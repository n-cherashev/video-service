from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from core.base_handler import BaseHandler
from models.viral_moments import Candidate, ScoredClip, ViralMomentsConfig


class ViralMomentsHandler(BaseHandler):
    """Обнаруживает виральные моменты с универсальным скорингом и диверсификацией.

    Улучшенный local re-fit (согласно ТЗ):
    При сильном overlap > 0.65 пробуем сдвиги: [-8, -5, -3, -2, +2, +3, +5, +8] секунд
    Также добавлен momentum scoring.
    """

    def __init__(
        self,
        config: ViralMomentsConfig = None,
        enable_momentum: bool = True,
        momentum_weight: float = 0.10,
    ):
        self.config = config or ViralMomentsConfig()
        self.enable_momentum = enable_momentum
        self.momentum_weight = momentum_weight

        # Расширенные refit shifts согласно ТЗ
        if self.config.refit_shifts is None:
            self.config.refit_shifts = [-8, -5, -3, -2, 2, 3, 5, 8]

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("[Viral Moments] ViralMomentsHandler")

        timeline = context.get("timeline", [])
        candidates_data = context.get("viral_candidates", [])
        duration = context.get("duration_seconds", 10000.0)

        if not timeline or not candidates_data:
            context["viral_clips"] = []
            context["highlights"] = []  # Backward compatibility
            print("✓ Viral moments: 0 clips (no data)")
            return context

        # Конвертируем кандидатов из dict в Candidate objects
        candidates = [Candidate(**c) for c in candidates_data]

        # Фильтруем по длине клипа
        min_dur = self.config.min_clip_duration
        max_dur = self.config.max_clip_duration
        candidates = [c for c in candidates if min_dur <= c.duration <= max_dur]

        if not candidates:
            context["viral_clips"] = []
            context["highlights"] = []
            print(f"✓ Viral moments: 0 clips (no candidates in {min_dur}-{max_dur}s range)")
            return context

        # Определяем тип видео и адаптивные веса
        video_type, adaptive_weights = self._detect_video_type_and_weights(timeline)
        print(f"   Video type: {video_type}")

        # Скоринг кандидатов с адаптивными весами
        scored_clips = self._score_candidates(candidates, timeline, adaptive_weights)

        # Диверсификация и выбор финальных клипов (с video_duration для refit)
        final_clips = self._diversify_and_select(scored_clips, video_duration=duration)

        # Сохраняем результаты
        context["viral_clips"] = [clip.to_dict() for clip in final_clips]

        # Обратная совместимость
        context["highlights"] = [
            {
                "start": clip.start,
                "end": clip.end,
                "type": "viral",
                "score": clip.score,
                "score_breakdown": clip.score_breakdown,
                "reasons": clip.reasons,
            }
            for clip in final_clips
        ]

        print(f"✓ Viral moments: {len(final_clips)} clips selected from {len(scored_clips)} scored")
        return context

    def _detect_video_type_and_weights(
        self,
        timeline: List[Dict[str, Any]]
    ) -> tuple:
        """Определяет тип видео и возвращает адаптивные веса.

        Типы видео:
        - 'action': Высокая активность (motion > 0.3)
        - 'dialogue': Много диалога (dialogue_ratio > 0.6)
        - 'mixed': Смешанный контент (default)
        """
        if not timeline:
            return 'mixed', None

        avg_motion = np.mean([item.get("motion", 0) for item in timeline])
        dialogue_ratio = np.mean([item.get("is_dialogue", False) for item in timeline])
        avg_loudness = np.mean([item.get("audio_loudness", 0) for item in timeline])

        # Определяем тип видео
        if avg_motion > 0.25 and avg_loudness > 0.3:
            video_type = 'action'
            weights = {
                'hook': 0.25,
                'intensity': 0.25,  # Усиливаем для активного видео
                'pace': 0.15,
                'clarity': 0.15,
                'emotion': 0.10,
                'boundary': 0.05,
                'momentum': 0.05,
            }
        elif dialogue_ratio > 0.5:
            video_type = 'dialogue'
            weights = {
                'hook': 0.25,
                'clarity': 0.25,  # Усиливаем для диалога
                'emotion': 0.20,
                'intensity': 0.15,
                'pace': 0.10,
                'boundary': 0.03,
                'momentum': 0.02,
            }
        else:
            video_type = 'mixed'
            weights = None  # Используем default веса из config

        return video_type, weights

    def _score_candidates(
        self,
        candidates: List[Candidate],
        timeline: List[Dict[str, Any]],
        adaptive_weights: Dict[str, float] = None,
    ) -> List[ScoredClip]:
        """Вычисляет универсальный скоринг для всех кандидатов.

        Включает 7 компонентов: hook, pace, clarity, intensity, emotion, boundary, momentum.
        Поддерживает адаптивные веса в зависимости от типа видео.
        """
        scored_clips = []

        # Извлекаем временные ряды для эффективности
        times = np.array([item["time"] for item in timeline])
        interests = np.array([item.get("interest", 0) for item in timeline])
        motions = np.array([item.get("motion", 0) for item in timeline])
        audio_loudness = np.array([item.get("audio_loudness", 0) for item in timeline])
        sentiments = np.array([item.get("sentiment", 0) for item in timeline])
        clarity_values = np.array([item.get("clarity", 0.5) for item in timeline])
        is_dialogue = np.array([item.get("is_dialogue", False) for item in timeline])
        is_scene_boundary = np.array([item.get("is_scene_boundary", False) for item in timeline])

        # Используем адаптивные веса или default из config
        weights = adaptive_weights or {
            'hook': self.config.hook_weight,
            'pace': self.config.pace_weight,
            'clarity': self.config.clarity_weight,
            'intensity': self.config.intensity_weight,
            'emotion': self.config.emotion_weight,
            'boundary': self.config.boundary_weight,
            'momentum': self.momentum_weight,
        }

        for candidate in candidates:
            # Маска для временного окна кандидата
            mask = (times >= candidate.start) & (times <= candidate.end)

            if not np.any(mask):
                continue

            # Вычисляем компоненты скоринга
            hook_score = self._calculate_hook_score(candidate, times, motions, audio_loudness, is_dialogue, mask)
            pace_score = self._calculate_pace_score(motions, audio_loudness, mask)
            clarity_score = self._calculate_clarity_score_enhanced(clarity_values, is_dialogue, mask)
            intensity_score = self._calculate_intensity_score(motions, audio_loudness, mask)
            emotion_score = self._calculate_emotion_score(sentiments, mask)
            boundary_score = self._calculate_boundary_score(candidate, times, is_scene_boundary, is_dialogue)

            # Momentum score
            if self.enable_momentum:
                momentum_score = self._calculate_momentum_score(candidate, times, interests, mask)
            else:
                momentum_score = 0.5

            # Агрегируем с адаптивными весами
            weighted_scores = [
                (weights.get('hook', 0.25), hook_score),
                (weights.get('pace', 0.15), pace_score),
                (weights.get('clarity', 0.15), clarity_score),
                (weights.get('intensity', 0.15), intensity_score),
                (weights.get('emotion', 0.10), emotion_score),
                (weights.get('boundary', 0.10), boundary_score),
            ]

            if self.enable_momentum:
                weighted_scores.append((weights.get('momentum', 0.10), momentum_score))

            # Нормализованный взвешенный скор
            total_weight = sum(w for w, _ in weighted_scores)
            base_clip_score = sum(w * s for w, s in weighted_scores) / total_weight

            # Применяем нелинейное преобразование для увеличения разброса
            # Усиливаем высокие скоры и уменьшаем низкие
            clip_score = self._apply_score_spread(base_clip_score, weighted_scores)

            # Формируем breakdown и reasons
            score_breakdown = {
                "hook": round(hook_score, 3),
                "pace": round(pace_score, 3),
                "clarity": round(clarity_score, 3),
                "intensity": round(intensity_score, 3),
                "emotion": round(emotion_score, 3),
                "boundary": round(boundary_score, 3),
                "momentum": round(momentum_score, 3),
            }

            reasons = self._generate_reasons(score_breakdown)

            scored_clips.append(ScoredClip(
                start=candidate.start,
                end=candidate.end,
                score=clip_score,
                score_breakdown=score_breakdown,
                anchor_type=candidate.anchor_type,
                reasons=reasons
            ))

        return scored_clips

    def _apply_score_spread(
        self,
        base_score: float,
        weighted_scores: list,
    ) -> float:
        """Применяет нелинейное преобразование для расширения диапазона скоров.

        Улучшенная версия с:
        1. Индивидуальными нелинейными трансформациями компонентов
        2. Бонусами за сочетания компонентов
        3. Sigmoid для финального расширения диапазона [0.53-0.57] → [0.25-0.85]
        """
        # Извлекаем компоненты
        components = {}
        component_names = ['hook', 'pace', 'clarity', 'intensity', 'emotion', 'boundary', 'momentum']
        for i, (_, score) in enumerate(weighted_scores):
            if i < len(component_names):
                components[component_names[i]] = score

        # 1. Индивидуальные нелинейные трансформации
        hook_s = components.get('hook', 0.5) ** 0.9  # Слабо наказываем слабые
        pace_s = min(1.0, components.get('pace', 0.1) * 3) ** 1.2  # Усиливаем низкие pace
        clarity_s = components.get('clarity', 0.5) ** 1.1

        # Intensity: нормируем диапазон [0.3, 1.0] → [0, 1]
        intensity_raw = components.get('intensity', 0.5)
        intensity_s = max(0, (intensity_raw - 0.3) / 0.7)

        # Emotion: усиливаем экстремумы (и положительные и отрицательные)
        emotion_raw = components.get('emotion', 0.5)
        emotion_s = abs(emotion_raw - 0.5) * 2

        boundary_s = components.get('boundary', 0.5) ** 0.8
        momentum_s = components.get('momentum', 0.5) ** 1.1

        # 2. Пересчитываем взвешенный скор с новыми весами
        enhanced_base = (
            0.30 * hook_s +      # Hook — ключевой для viral
            0.20 * clarity_s +   # Clarity — важна для понимания
            0.15 * intensity_s + # Intensity — энергия
            0.15 * emotion_s +   # Emotion — эмоциональность
            0.12 * pace_s +      # Pace — динамика (теперь значимый!)
            0.05 * boundary_s +  # Boundary — естественность
            0.03 * momentum_s    # Momentum — нарастание
        )

        # 3. Бонусы за сочетания компонентов
        bonuses = 0.0

        # Бонус: сильный hook + strong intensity
        if hook_s > 0.6 and intensity_s > 0.6:
            bonuses += 0.05

        # Бонус: хорошо 3+ компонента
        good_components = sum(1 for v in [hook_s, clarity_s, intensity_s, emotion_s] if v > 0.6)
        bonuses += good_components * 0.02

        # Штраф: слабый hook
        if hook_s < 0.3:
            bonuses -= 0.10

        # Штраф: много низких компонентов
        low_count = sum(1 for v in [hook_s, clarity_s, intensity_s, emotion_s, pace_s] if v < 0.25)
        if low_count >= 3:
            bonuses -= 0.08

        # 4. Sigmoid для расширения диапазона
        pre_sigmoid = enhanced_base + bonuses

        # Sigmoid с центром в 0.5 и крутизной 5
        # Это растягивает диапазон [0.3-0.7] в [0.1-0.9]
        final_score = 1 / (1 + np.exp(-5 * (pre_sigmoid - 0.5)))

        return float(np.clip(final_score, 0.0, 1.0))

    def _calculate_hook_score(self, candidate: Candidate, times: np.ndarray, motions: np.ndarray,
                            audio_loudness: np.ndarray, is_dialogue: np.ndarray, mask: np.ndarray) -> float:
        """Оценивает качество первых 3 секунд (hook)."""
        hook_end = candidate.start + 3.0
        hook_mask = mask & (times <= hook_end)

        if not np.any(hook_mask):
            return 0.0

        # Проверяем наличие движения, звука, диалога в начале
        hook_motion = np.mean(motions[hook_mask])
        hook_audio = np.mean(audio_loudness[hook_mask])
        hook_dialogue = np.mean(is_dialogue[hook_mask])

        # Штраф за "пустой старт"
        if hook_motion < 0.1 and hook_audio < 0.1 and hook_dialogue < 0.1:
            return 0.2

        # Бонус за активное начало
        hook_score = (hook_motion + hook_audio + hook_dialogue) / 3.0
        return min(1.0, hook_score * 1.2)  # Небольшой бонус

    def _calculate_pace_score(self, motions: np.ndarray, audio_loudness: np.ndarray, mask: np.ndarray) -> float:
        """Оценивает плотность событий и вариативность.

        Улучшенная версия с scipy.signal.find_peaks и адаптивным порогом.
        Ожидаем 1 пик на 5 секунд (вместо 10) для более реалистичной оценки.
        """
        from scipy.signal import find_peaks

        if not np.any(mask):
            return 0.0

        motion_window = motions[mask]
        audio_window = audio_loudness[mask]

        # Комбинированный сигнал (60% motion + 40% audio)
        combined = 0.6 * motion_window + 0.4 * audio_window

        if len(combined) < 3:
            return 0.3  # Слишком короткое окно

        # Адаптивный порог (75 перцентиль)
        threshold = np.percentile(combined, 75)

        # Находим пики с scipy.signal.find_peaks
        # distance=3 означает минимум 3 секунды между пиками (если timeline на 1-сек шаге)
        peaks, properties = find_peaks(
            combined,
            height=threshold,
            distance=3,
            prominence=np.std(combined) * 0.3 if np.std(combined) > 0 else 0.1,
        )

        # 1 пик на 5 секунд (вместо 10) - более реалистичное ожидание
        duration_sec = len(combined)
        expected_peaks = max(1, duration_sec / 5.0)

        peak_ratio = len(peaks) / expected_peaks

        # Вариативность (стандартное отклонение)
        variability = (np.std(motion_window) + np.std(audio_window)) / 2.0

        # Комбинируем: 70% от пиков + 30% от вариативности
        pace_score = 0.7 * min(peak_ratio, 1.5) + 0.3 * min(variability * 2, 1.0)

        return float(min(1.0, pace_score))

    def _calculate_clarity_score(self, is_dialogue: np.ndarray, mask: np.ndarray) -> float:
        """Оценивает долю диалога в клипе (legacy)."""
        if not np.any(mask):
            return 0.0

        dialogue_ratio = np.mean(is_dialogue[mask])
        return float(dialogue_ratio)

    def _calculate_clarity_score_enhanced(
        self,
        clarity_values: np.ndarray,
        is_dialogue: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Улучшенная оценка clarity с учетом speech quality."""
        if not np.any(mask):
            return 0.0

        # Используем clarity из speech_quality если доступен
        avg_clarity = np.mean(clarity_values[mask])
        dialogue_ratio = np.mean(is_dialogue[mask])

        # Комбинируем: clarity * 0.6 + dialogue_ratio * 0.4
        combined = avg_clarity * 0.6 + dialogue_ratio * 0.4

        # Штраф за мало диалога
        if dialogue_ratio < 0.3:
            combined *= 0.8

        return float(min(1.0, combined))

    def _calculate_intensity_score(self, motions: np.ndarray, audio_loudness: np.ndarray, mask: np.ndarray) -> float:
        """Оценивает общую интенсивность движения и звука."""
        if not np.any(mask):
            return 0.0

        avg_motion = np.mean(motions[mask])
        avg_audio = np.mean(audio_loudness[mask])
        max_motion = np.max(motions[mask])
        max_audio = np.max(audio_loudness[mask])

        # Комбинируем средние и максимальные значения
        intensity = (avg_motion + avg_audio + max_motion + max_audio) / 4.0
        return min(1.0, intensity)

    def _calculate_emotion_score(self, sentiments: np.ndarray, mask: np.ndarray) -> float:
        """Оценивает эмоциональную динамику."""
        if not np.any(mask):
            return 0.0

        sentiment_window = sentiments[mask]

        # Средняя абсолютная эмоциональность
        avg_abs_sentiment = np.mean(np.abs(sentiment_window))

        # Количество резких смен (скачков)
        if len(sentiment_window) > 1:
            sentiment_changes = np.abs(np.diff(sentiment_window))
            sharp_changes = np.sum(sentiment_changes > 0.5) / len(sentiment_changes)
        else:
            sharp_changes = 0

        return min(1.0, (avg_abs_sentiment + sharp_changes) / 2.0)

    def _calculate_boundary_score(self, candidate: Candidate, times: np.ndarray,
                                is_scene_boundary: np.ndarray, is_dialogue: np.ndarray) -> float:
        """Оценивает близость к границам сцен и диалогов."""
        boundary_bonus = 0.0

        # Бонус за близость к границам сцен
        scene_boundaries = times[is_scene_boundary]
        if len(scene_boundaries) > 0:
            start_dist = np.min(np.abs(scene_boundaries - candidate.start))
            end_dist = np.min(np.abs(scene_boundaries - candidate.end))

            if start_dist <= 3.0 or end_dist <= 3.0:
                boundary_bonus += 0.3

        # Бонус за начало/конец диалога
        dialogue_changes = np.diff(is_dialogue.astype(int))
        dialogue_boundaries = times[1:][dialogue_changes != 0]

        if len(dialogue_boundaries) > 0:
            start_dist = np.min(np.abs(dialogue_boundaries - candidate.start))
            end_dist = np.min(np.abs(dialogue_boundaries - candidate.end))

            if start_dist <= 2.0 or end_dist <= 2.0:
                boundary_bonus += 0.2

        return min(1.0, boundary_bonus)

    def _count_local_peaks(self, signal: np.ndarray, threshold: float = 0.5) -> int:
        """Подсчитывает локальные максимумы в сигнале."""
        if len(signal) < 3:
            return 0

        peaks = 0
        for i in range(1, len(signal) - 1):
            if (signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] >= threshold):
                peaks += 1

        return peaks

    def _calculate_momentum_score(
        self,
        candidate: Candidate,
        times: np.ndarray,
        interests: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Оценивает рост интенсивности к концу клипа (momentum).

        Согласно ТЗ:
        momentum_score = 1.0 if end_intensity > start_intensity*1.2
                        else end_intensity/max(start_intensity, 0.1)
        """
        if not np.any(mask):
            return 0.5

        mid_point = (candidate.start + candidate.end) / 2
        first_half_mask = mask & (times < mid_point)
        second_half_mask = mask & (times >= mid_point)

        if np.any(first_half_mask) and np.any(second_half_mask):
            start_intensity = np.mean(interests[first_half_mask])
            end_intensity = np.mean(interests[second_half_mask])

            if end_intensity > start_intensity * 1.2:
                return 1.0
            elif start_intensity > 0.1:
                return min(1.0, end_intensity / start_intensity)
            else:
                return min(1.0, end_intensity * 5)  # Бонус если начало тихое

        return 0.5

    def _generate_reasons(self, score_breakdown: Dict[str, float]) -> List[str]:
        """Генерирует причины высокого скора."""
        reasons = []

        if score_breakdown.get("hook", 0) >= 0.7:
            reasons.append("Strong hook")
        if score_breakdown.get("pace", 0) >= 0.7:
            reasons.append("High pace")
        if score_breakdown.get("intensity", 0) >= 0.7:
            reasons.append("High intensity")
        if score_breakdown.get("clarity", 0) >= 0.7:
            reasons.append("Clear dialogue")
        if score_breakdown.get("emotion", 0) >= 0.6:
            reasons.append("Emotional dynamics")
        if score_breakdown.get("boundary", 0) >= 0.4:
            reasons.append("Good boundaries")
        if score_breakdown.get("momentum", 0) >= 0.7:
            reasons.append("Building momentum")

        return reasons if reasons else ["Automated selection"]

    def _diversify_and_select(
        self,
        scored_clips: List[ScoredClip],
        video_duration: float = 10000.0,
    ) -> List[ScoredClip]:
        """Выбирает финальные клипы с диверсификацией по времени.

        Улучшенный алгоритм:
        1. Фильтрация дубликатов (overlap > 70%)
        2. Сортировка по скору
        3. Жадный выбор с учетом overlap penalties и min_gap
        4. Local re-fit для клипов с сильным пересечением
        5. Группировка по overlap
        """
        if not scored_clips:
            return []

        # Сортируем по скору
        scored_clips.sort(key=lambda x: x.score, reverse=True)

        # Фильтруем дубликаты (оставляем только лучший из пересекающихся >70%)
        filtered_clips = self._filter_duplicates(scored_clips)

        selected = []
        group_id = 0

        for clip in filtered_clips:
            if len(selected) >= self.config.max_clips:
                break

            # Проверяем минимальный gap
            if not self._check_min_gap(clip, selected):
                continue

            # Вычисляем штрафы за пересечения
            adjusted_score = self._calculate_adjusted_score(clip, selected)

            # Если скор все еще достаточно высок, добавляем клип
            if adjusted_score > 0.15:  # Повышен минимальный порог
                # Пробуем local re-fit если есть сильные пересечения
                final_clip = self._local_refit(clip, selected, video_duration)

                # Проверяем min_gap после refit
                if not self._check_min_gap(final_clip, selected):
                    continue

                # Пересчитываем adjusted_score после refit
                final_adjusted = self._calculate_adjusted_score(final_clip, selected)

                if final_adjusted > 0.15:
                    # Назначаем группу пересечений
                    overlap_group = self._find_overlap_group(final_clip, selected)
                    if overlap_group is not None:
                        final_clip.overlap_group_id = overlap_group
                    else:
                        final_clip.overlap_group_id = group_id
                        group_id += 1

                    selected.append(final_clip)

        # Сортируем по времени для финального вывода
        selected.sort(key=lambda x: x.start)
        return selected

    def _filter_duplicates(self, clips: List[ScoredClip]) -> List[ScoredClip]:
        """Фильтрует дубликаты - клипы с overlap > duplicate_threshold.

        Оставляет только лучший клип из группы сильно пересекающихся.
        """
        if not clips:
            return []

        # Клипы уже отсортированы по score (лучшие первые)
        filtered = []

        for clip in clips:
            is_duplicate = False
            for existing in filtered:
                overlap = self._calculate_overlap_ratio(clip, existing)
                if overlap > self.config.duplicate_overlap_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(clip)

        return filtered

    def _check_min_gap(self, clip: ScoredClip, selected: List[ScoredClip]) -> bool:
        """Проверяет минимальный gap между клипом и уже выбранными."""
        min_gap = self.config.min_gap_seconds

        for existing in selected:
            # Вычисляем расстояние между клипами
            if clip.end <= existing.start:
                gap = existing.start - clip.end
            elif clip.start >= existing.end:
                gap = clip.start - existing.end
            else:
                # Пересекаются - gap = 0
                gap = 0

            if gap < min_gap and gap >= 0:
                # Разрешаем пересечения, но не слишком близкие непересекающиеся клипы
                # Если клипы пересекаются (gap < 0 невозможен в этой логике), это обрабатывается overlap
                if self._calculate_overlap_ratio(clip, existing) == 0:
                    return False

        return True

    def _calculate_adjusted_score(self, clip: ScoredClip, selected: List[ScoredClip]) -> float:
        """Вычисляет скор с учетом штрафов за пересечения."""
        adjusted_score = clip.score

        for selected_clip in selected:
            overlap_ratio = self._calculate_overlap_ratio(clip, selected_clip)

            if overlap_ratio > self.config.strong_overlap_threshold:
                adjusted_score *= self.config.strong_penalty
            elif overlap_ratio > self.config.medium_overlap_threshold:
                adjusted_score *= self.config.medium_penalty

        return adjusted_score

    def _calculate_overlap_ratio(self, clip1: ScoredClip, clip2: ScoredClip) -> float:
        """Вычисляет коэффициент пересечения двух клипов."""
        intersection_start = max(clip1.start, clip2.start)
        intersection_end = min(clip1.end, clip2.end)

        if intersection_end <= intersection_start:
            return 0.0

        intersection = intersection_end - intersection_start
        min_duration = min(clip1.duration, clip2.duration)

        return intersection / min_duration if min_duration > 0 else 0.0

    def _local_refit(
        self,
        clip: ScoredClip,
        selected: List[ScoredClip],
        video_duration: float = 10000.0,
    ) -> ScoredClip:
        """Улучшенный local re-fit с оптимальным выбором сдвига.

        Согласно ТЗ: при overlap > 0.65 пробуем сдвиги [-8, -5, -3, -2, +2, +3, +5, +8] секунд.
        Выбираем сдвиг с максимальным adjusted_score.
        """
        best_clip = clip
        best_adjusted_score = self._calculate_adjusted_score(clip, selected)
        best_shift = 0

        # Проверяем есть ли сильные пересечения
        max_overlap = 0.0
        for s in selected:
            overlap = self._calculate_overlap_ratio(clip, s)
            max_overlap = max(max_overlap, overlap)

        if max_overlap <= self.config.strong_overlap_threshold:
            return best_clip

        # Пробуем сдвиги (согласно ТЗ)
        refit_shifts = self.config.refit_shifts or [-8, -5, -3, -2, 2, 3, 5, 8]

        for shift in refit_shifts:
            shifted_start = clip.start + shift
            shifted_end = clip.end + shift

            # Проверяем границы видео
            if shifted_start < 0:
                continue
            if shifted_end > video_duration:
                continue

            # Проверяем что длина не изменилась значительно
            original_duration = clip.end - clip.start
            shifted_duration = shifted_end - shifted_start
            if abs(shifted_duration - original_duration) > 1.0:
                continue

            # Создаем сдвинутый клип
            # Штраф за сдвиг: 2% за каждую секунду сдвига
            shift_penalty = 1.0 - (abs(shift) * 0.02)

            shifted_clip = ScoredClip(
                start=shifted_start,
                end=shifted_end,
                score=clip.score * shift_penalty,
                score_breakdown=clip.score_breakdown.copy(),
                anchor_type=clip.anchor_type,
                reasons=clip.reasons.copy()
            )

            adjusted_score = self._calculate_adjusted_score(shifted_clip, selected)

            if adjusted_score > best_adjusted_score:
                best_clip = shifted_clip
                best_adjusted_score = adjusted_score
                best_shift = shift

        # Логируем если произошел refit
        if best_shift != 0:
            print(f"  Local re-fit: shifted by {best_shift}s (overlap was {max_overlap:.2f})")

        return best_clip

    def _find_overlap_group(self, clip: ScoredClip, selected: List[ScoredClip]) -> int | None:
        """Находит группу пересечений для клипа."""
        for selected_clip in selected:
            overlap_ratio = self._calculate_overlap_ratio(clip, selected_clip)
            if overlap_ratio > self.config.medium_overlap_threshold:
                return selected_clip.overlap_group_id

        return None
