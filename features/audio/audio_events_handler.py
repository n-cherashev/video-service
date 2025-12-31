from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from core.base_handler import BaseHandler
from utils.timeseries import extract_series, interp_to_grid, normalize_01

try:
    # SciPy peak picking (recommended)
    from scipy.signal import find_peaks
except Exception:  # pragma: no cover
    find_peaks = None  # type: ignore[assignment]


@dataclass(frozen=True)
class AudioEvent:
    start: float
    end: float
    time: float
    type: str
    confidence: float
    peak_value: float


class AudioEventsHandler(BaseHandler):
    """
    Детектирует аудио-события по пикам (loud_event).
    Работает строго по таймкодам из audio_features (никаких i*0.1).
    """

    def __init__(
        self,
        combined_threshold: float = 0.75,   # порог для combined score (0..1)
        min_gap_seconds: float = 0.8,       # минимальная дистанция между событиями
        min_event_duration: float = 0.2,    # минимальная длительность события
        peak_prominence: float = 0.15,      # "выраженность" пика
    ) -> None:
        self.combined_threshold = float(combined_threshold)
        self.min_gap_seconds = float(min_gap_seconds)
        self.min_event_duration = float(min_event_duration)
        self.peak_prominence = float(peak_prominence)

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("[6] AudioEventsHandler")
        
        audio_features = context.get("audio_features") or {}
        loud_points = audio_features.get("loudness") or []
        energy_points = audio_features.get("energy") or []

        if not loud_points or not energy_points:
            context["audio_events"] = []
            return context

        loud = extract_series(loud_points)
        eng = extract_series(energy_points)

        if loud.times.size == 0:
            context["audio_events"] = []
            return context

        # выравниваем energy на сетку loudness
        eng_on_loud = interp_to_grid(eng, loud.times, fill_value=0.0)

        v_l = normalize_01(loud.values)
        v_e = normalize_01(eng_on_loud)

        # комбинированная оценка громкого события
        combined = 0.6 * v_l + 0.4 * v_e
        combined = np.clip(combined, 0.0, 1.0)

        events = self._detect_peaks(times=loud.times, score=combined)

        context["audio_events"] = [e.__dict__ for e in events]
        print(f"✓ Audio events: {len(events)} detected")
        return context

    def _detect_peaks(self, *, times: np.ndarray, score: np.ndarray) -> List[AudioEvent]:
        if times.size < 3:
            return []

        # оценка dt
        dt = float(np.median(np.diff(times))) if times.size > 1 else 0.1
        dt = max(dt, 1e-3)
        min_distance_samples = max(1, int(self.min_gap_seconds / dt))

        # SciPy find_peaks — стандартный способ искать локальные максимумы в 1D с условиями
        if find_peaks is not None:
            peaks, props = find_peaks(
                score,
                height=self.combined_threshold,
                distance=min_distance_samples,
                prominence=self.peak_prominence,
            )
            peak_heights = props.get("peak_heights", score[peaks] if len(peaks) else np.array([]))
        else:
            # fallback: наивный поиск пиков
            peaks = []
            peak_heights = []
            for i in range(1, len(score) - 1):
                if score[i] >= self.combined_threshold and score[i] > score[i - 1] and score[i] >= score[i + 1]:
                    peaks.append(i)
                    peak_heights.append(score[i])
            peaks = np.array(peaks, dtype=int)
            peak_heights = np.array(peak_heights, dtype=float)

        events: List[AudioEvent] = []
        for idx, peak_idx in enumerate(peaks):
            peak_time = float(times[int(peak_idx)])
            peak_value = float(peak_heights[idx]) if idx < len(peak_heights) else float(score[int(peak_idx)])

            # окно события: идём влево/вправо до падения ниже (threshold - небольшая дельта)
            floor = max(self.combined_threshold * 0.85, 0.05)

            left = int(peak_idx)
            while left > 0 and score[left] >= floor:
                left -= 1

            right = int(peak_idx)
            while right < len(score) - 1 and score[right] >= floor:
                right += 1

            start = float(times[left])
            end = float(times[right])

            if (end - start) < self.min_event_duration:
                # минимальное окно вокруг пика
                half = max(self.min_event_duration / 2.0, dt)
                start = max(0.0, peak_time - half)
                end = peak_time + half

            confidence = float(np.clip(peak_value, 0.0, 1.0))

            events.append(
                AudioEvent(
                    start=start,
                    end=end,
                    time=peak_time,
                    type="loud_event",
                    confidence=confidence,
                    peak_value=peak_value,
                )
            )

        return events