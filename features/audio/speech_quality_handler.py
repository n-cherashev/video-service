from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from core.base_handler import BaseHandler


class SpeechQualityHandler(BaseHandler):
    """Анализирует качество речи в видео.

    Метрики (согласно ТЗ):
    - SNR (Signal-to-Noise Ratio): на основе VAD + loudness
    - Stability: стандартное отклонение громкости в речевых участках
    - Silence ratio: процент молчания

    Выход: clarity_timeline (per 1-second)
    """

    def __init__(
        self,
        vad_threshold: float = 0.5,
        step: float = 1.0,
        min_speech_ratio: float = 0.1,
        snr_weight: float = 0.4,
        stability_weight: float = 0.3,
        silence_penalty_weight: float = 0.3,
    ) -> None:
        self.vad_threshold = float(vad_threshold)
        self.step = float(step)
        self.min_speech_ratio = float(min_speech_ratio)

        # Веса для формулы clarity
        self.snr_weight = float(snr_weight)
        self.stability_weight = float(stability_weight)
        self.silence_penalty_weight = float(silence_penalty_weight)

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("[SpeechQuality] SpeechQualityHandler")

        audio_features = context.get("audio_features", {})
        duration = float(context.get("duration_seconds") or 0.0)

        if not audio_features or duration <= 0:
            context["speech_quality"] = self._empty_result()
            context["clarity_timeline"] = []
            print("✓ Speech quality: skipped (no audio features)")
            return context

        # Извлечение сигналов
        loudness = audio_features.get("loudness", [])
        speech_prob = audio_features.get("speech_probability", [])

        if not loudness:
            context["speech_quality"] = self._empty_result()
            context["clarity_timeline"] = []
            print("✓ Speech quality: skipped (no loudness data)")
            return context

        # Создаем временную сетку
        n_points = int(np.floor(duration / self.step)) + 1
        times = np.arange(n_points, dtype=float) * self.step

        # Интерполируем сигналы на сетку
        loudness_grid = self._interpolate_to_grid(loudness, times)
        speech_prob_grid = self._interpolate_to_grid(speech_prob, times)

        # Вычисляем метрики
        snr_timeline = self._calculate_snr_timeline(loudness_grid, speech_prob_grid, times)
        stability_timeline = self._calculate_stability_timeline(loudness_grid, speech_prob_grid, times)
        silence_timeline = self._calculate_silence_timeline(speech_prob_grid, times)

        # Комбинируем в clarity
        clarity_timeline = self._calculate_clarity_timeline(
            snr_timeline, stability_timeline, silence_timeline, times
        )

        # Общие метрики
        speech_mask = speech_prob_grid >= self.vad_threshold
        speech_ratio = float(np.mean(speech_mask))

        if speech_ratio > 0:
            speech_loudness = loudness_grid[speech_mask]
            non_speech_loudness = loudness_grid[~speech_mask]

            avg_speech_loudness = float(np.mean(speech_loudness)) if len(speech_loudness) > 0 else 0.0
            avg_noise_loudness = float(np.mean(non_speech_loudness)) if len(non_speech_loudness) > 0 else 0.0

            if avg_noise_loudness > 0:
                global_snr = avg_speech_loudness / avg_noise_loudness
            else:
                global_snr = avg_speech_loudness * 10  # Высокий SNR если нет шума

            stability = 1.0 - min(1.0, float(np.std(speech_loudness)) / max(0.01, avg_speech_loudness))
        else:
            avg_speech_loudness = 0.0
            avg_noise_loudness = float(np.mean(loudness_grid))
            global_snr = 0.0
            stability = 0.0

        speech_quality = {
            "speech_ratio": speech_ratio,
            "silence_ratio": 1.0 - speech_ratio,
            "global_snr": float(min(10.0, global_snr)),  # Нормализуем SNR
            "stability": stability,
            "avg_speech_loudness": avg_speech_loudness,
            "avg_noise_loudness": avg_noise_loudness,
            "mean_clarity": float(np.mean([c["clarity"] for c in clarity_timeline])) if clarity_timeline else 0.0,
            "vad_threshold": self.vad_threshold,
        }

        context["speech_quality"] = speech_quality
        context["clarity_timeline"] = clarity_timeline

        print(f"✓ Speech quality: speech_ratio={speech_ratio:.2f}, snr={global_snr:.2f}, stability={stability:.2f}")
        return context

    def _interpolate_to_grid(
        self,
        points: List[Dict[str, Any]],
        times: np.ndarray,
    ) -> np.ndarray:
        """Интерполирует точки на временную сетку."""
        if not points:
            return np.zeros_like(times)

        src_times = []
        src_values = []

        for p in points:
            if not isinstance(p, dict):
                continue
            t = p.get("time")
            v = p.get("value", p.get("prob", 0.0))
            if t is not None:
                src_times.append(float(t))
                src_values.append(float(v))

        if not src_times:
            return np.zeros_like(times)

        src_times = np.array(src_times)
        src_values = np.array(src_values)

        # Сортируем по времени
        sort_idx = np.argsort(src_times)
        src_times = src_times[sort_idx]
        src_values = src_values[sort_idx]

        # Линейная интерполяция
        return np.interp(times, src_times, src_values, left=src_values[0], right=src_values[-1])

    def _calculate_snr_timeline(
        self,
        loudness: np.ndarray,
        speech_prob: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Вычисляет timeline SNR (локальный)."""
        snr = np.zeros_like(times)
        window = max(1, int(5.0 / self.step))  # 5-секундное окно

        for i in range(len(times)):
            start_idx = max(0, i - window)
            end_idx = min(len(times), i + window + 1)

            local_loudness = loudness[start_idx:end_idx]
            local_speech_prob = speech_prob[start_idx:end_idx]

            speech_mask = local_speech_prob >= self.vad_threshold

            if np.any(speech_mask):
                speech_level = np.mean(local_loudness[speech_mask])
                if np.any(~speech_mask):
                    noise_level = np.mean(local_loudness[~speech_mask])
                else:
                    noise_level = 0.01

                if noise_level > 0.01:
                    snr[i] = min(10.0, speech_level / noise_level)
                else:
                    snr[i] = min(10.0, speech_level * 10)
            else:
                snr[i] = 0.0

        # Нормализуем в [0, 1]
        return np.clip(snr / 5.0, 0.0, 1.0)  # SNR > 5 считается хорошим

    def _calculate_stability_timeline(
        self,
        loudness: np.ndarray,
        speech_prob: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Вычисляет timeline стабильности громкости речи."""
        stability = np.zeros_like(times)
        window = max(1, int(3.0 / self.step))  # 3-секундное окно

        for i in range(len(times)):
            start_idx = max(0, i - window)
            end_idx = min(len(times), i + window + 1)

            local_loudness = loudness[start_idx:end_idx]
            local_speech_prob = speech_prob[start_idx:end_idx]

            speech_mask = local_speech_prob >= self.vad_threshold

            if np.sum(speech_mask) > 2:
                speech_loudness = local_loudness[speech_mask]
                mean_loudness = np.mean(speech_loudness)
                std_loudness = np.std(speech_loudness)

                if mean_loudness > 0.01:
                    # Низкое std = высокая стабильность
                    cv = std_loudness / mean_loudness  # Coefficient of variation
                    stability[i] = max(0.0, 1.0 - min(1.0, cv))
                else:
                    stability[i] = 0.5
            else:
                stability[i] = 0.5  # Нейтральное значение если мало речи

        return stability

    def _calculate_silence_timeline(
        self,
        speech_prob: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Вычисляет локальный процент молчания."""
        silence = np.zeros_like(times)
        window = max(1, int(5.0 / self.step))  # 5-секундное окно

        for i in range(len(times)):
            start_idx = max(0, i - window)
            end_idx = min(len(times), i + window + 1)

            local_speech_prob = speech_prob[start_idx:end_idx]
            speech_mask = local_speech_prob >= self.vad_threshold

            silence[i] = 1.0 - np.mean(speech_mask)

        return silence

    def _calculate_clarity_timeline(
        self,
        snr_timeline: np.ndarray,
        stability_timeline: np.ndarray,
        silence_timeline: np.ndarray,
        times: np.ndarray,
    ) -> List[Dict[str, float]]:
        """Комбинирует метрики в единый clarity score."""
        clarity_points = []

        for i, t in enumerate(times):
            # clarity = snr * w1 + stability * w2 - silence_penalty * w3
            snr_component = snr_timeline[i] * self.snr_weight
            stability_component = stability_timeline[i] * self.stability_weight
            silence_penalty = silence_timeline[i] * self.silence_penalty_weight

            clarity = snr_component + stability_component - silence_penalty
            clarity = float(np.clip(clarity, 0.0, 1.0))

            clarity_points.append({
                "time": float(t),
                "clarity": clarity,
                "snr": float(snr_timeline[i]),
                "stability": float(stability_timeline[i]),
                "silence_ratio": float(silence_timeline[i]),
            })

        return clarity_points

    def _empty_result(self) -> Dict[str, Any]:
        """Возвращает пустой результат."""
        return {
            "speech_ratio": 0.0,
            "silence_ratio": 1.0,
            "global_snr": 0.0,
            "stability": 0.0,
            "avg_speech_loudness": 0.0,
            "avg_noise_loudness": 0.0,
            "mean_clarity": 0.0,
            "vad_threshold": self.vad_threshold,
        }
