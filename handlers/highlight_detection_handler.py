from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal

import numpy as np

from handlers.base_handler import BaseHandler
from models.highlights import Highlight


Mode = Literal["rules", "top_interest"]


@dataclass(frozen=True, slots=True)
class Candidate:
    start: float
    end: float
    score: float
    type: str = "top_interest"


class HighlightDetectionHandler(BaseHandler):
    def __init__(
        self,
        min_duration_seconds: float = 30.0,
        max_duration_seconds: float = 90.0,
        *,
        mode: Mode = "top_interest",
        top_k: int = 6,
        peak_quantile: float = 0.90,
        snap_to_scenes: bool = True,
        snap_window_seconds: float = 5.0,
        merge_gap_seconds: float = 3.0,
    ) -> None:
        self.min_duration = float(min_duration_seconds)
        self.max_duration = float(max_duration_seconds)
        self.mode = mode
        self.top_k = int(top_k)
        self.peak_quantile = float(peak_quantile)
        self.snap_to_scenes = bool(snap_to_scenes)
        self.snap_window_seconds = float(snap_window_seconds)
        self.merge_gap_seconds = float(merge_gap_seconds)

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        timeline = context.get("timeline", []) or []
        if not timeline:
            context["highlights"] = []
            context["highlight_items"] = []
            return context

        duration = float(context.get("duration_seconds") or 0.0)
        scene_boundaries = sorted([float(x) for x in (context.get("scene_boundaries") or [0.0])])

        if self.mode == "top_interest":
            items = self._top_interest_highlights(timeline, duration, scene_boundaries)
        else:
            items = []

        context["highlight_items"] = items
        context["highlights"] = [{"start": h.start, "end": h.end, "type": h.type, "score": h.score} for h in items]
        return context

    def _top_interest_highlights(
        self,
        timeline: List[Dict[str, Any]],
        duration: float,
        scene_boundaries: List[float],
    ) -> List[Highlight]:
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
