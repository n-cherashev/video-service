from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np

from handlers.base_handler import BaseHandler
from handlers.timeseries import extract_series, interp_to_grid, normalize_01
from models.timeline import TimelinePoint


class FusionTimelineHandler(BaseHandler):
    """Создает единую временную шкалу с интегральным interest_score."""

    def __init__(
        self,
        step: float = 1.0,
        weight_motion: float = 0.25,
        weight_audio: float = 0.20,
        weight_sentiment: float = 0.15,
        weight_humor: float = 0.20,
        weight_events: float = 0.20,
        dialogue_prob_threshold: float = 0.5,
        normalize_weights: bool = True,
        event_laughter_boost: float = 0.5,
        event_loud_boost: float = 0.3,
    ) -> None:
        self.step = float(step)
        if self.step <= 0:
            raise ValueError("step must be > 0")

        self.weights = {
            "motion": float(weight_motion),
            "audio": float(weight_audio),
            "sentiment": float(weight_sentiment),
            "humor": float(weight_humor),
            "events": float(weight_events),
        }
        self.dialogue_prob_threshold = float(dialogue_prob_threshold)
        self.normalize_weights = bool(normalize_weights)

        self.event_laughter_boost = float(event_laughter_boost)
        self.event_loud_boost = float(event_loud_boost)

        self._normalize_weights_if_needed()

    def _normalize_weights_if_needed(self) -> None:
        s = sum(self.weights.values())
        if s <= 0:
            raise ValueError("sum(weights) must be > 0")
        if self.normalize_weights and abs(s - 1.0) > 1e-6:
            self.weights = {k: v / s for k, v in self.weights.items()}

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        duration = float(context.get("duration_seconds") or 0.0)
        if duration <= 0:
            context["timeline"] = []
            return context

        # Ровный грид, последняя точка <= duration
        n = int(np.floor(duration / self.step)) + 1
        times = (np.arange(n, dtype=float) * self.step)

        motion = self._series_to_grid(context.get("motion_heatmap", []), times, value_key="value")
        af = (context.get("audio_features") or {})
        loudness = self._series_to_grid(af.get("loudness", []), times, value_key="value")
        speech_prob = self._series_to_grid(af.get("speech_probability", []), times, value_key="value")

        sentiment = self._sentiment_to_grid(context.get("sentiment_timeline", []), times)
        humor = self._humor_to_grid(context.get("humor_scores", []), times)

        has_laughter, has_loud = self._event_flags(context.get("audio_events", []), times)
        is_scene_boundary = self._scene_flags(context.get("scene_boundaries", []), times)
        is_dialogue = self._dialogue_flags(times, speech_prob, context.get("transcript_segments", []))

        motion_n = normalize_01(motion)
        loud_n = normalize_01(loudness)
        humor_n = np.clip(humor, 0.0, 1.0)
        sent_abs = np.clip(np.abs(sentiment), 0.0, 1.0)

        events_component = (
            self.event_laughter_boost * has_laughter.astype(float) +
            self.event_loud_boost * has_loud.astype(float)
        )
        events_component = np.clip(events_component, 0.0, 1.0)

        interest = (
            self.weights["motion"] * motion_n +
            self.weights["audio"] * loud_n +
            self.weights["sentiment"] * sent_abs +
            self.weights["humor"] * humor_n +
            self.weights["events"] * events_component
        )
        interest = np.clip(interest, 0.0, 1.0)

        points: list[TimelinePoint] = []
        for i, t in enumerate(times):
            points.append(TimelinePoint(
                time=float(t),
                interest=float(interest[i]),
                motion=float(motion_n[i]),
                audio_loudness=float(loud_n[i]),
                sentiment=float(sentiment[i]),
                humor=float(humor_n[i]),
                has_laughter=bool(has_laughter[i]),
                has_loud_sound=bool(has_loud[i]),
                is_scene_boundary=bool(is_scene_boundary[i]),
                is_dialogue=bool(is_dialogue[i]),
            ))

        # 1) новый модельный результат
        context["timeline_points"] = points
        # 2) обратная совместимость (как раньше)
        context["timeline"] = [
            {
                "time": p.time,
                "interest": p.interest,
                "motion": p.motion,
                "audio_loudness": p.audio_loudness,
                "sentiment": p.sentiment,
                "humor": p.humor,
                "has_laughter": p.has_laughter,
                "has_loud_sound": p.has_loud_sound,
                "is_scene_boundary": p.is_scene_boundary,
                "is_dialogue": p.is_dialogue,
            }
            for p in points
        ]
        return context

    def _series_to_grid(self, points: List[Mapping[str, object]], grid: np.ndarray, *, value_key: str) -> np.ndarray:
        s = extract_series(points, value_key=value_key)
        return interp_to_grid(s, grid, fill_value=0.0)

    def _sentiment_to_grid(self, timeline: List[Dict[str, Any]], grid: np.ndarray) -> np.ndarray:
        if not timeline:
            return np.zeros_like(grid, dtype=float)
        points = []
        for p in timeline:
            if not isinstance(p, dict) or "time" not in p:
                continue
            if "score" in p:
                points.append({"time": p["time"], "value": p["score"]})
            elif "sentiment" in p:
                points.append({"time": p["time"], "value": p["sentiment"]})
        s = extract_series(points)
        return interp_to_grid(s, grid, fill_value=0.0)

    def _humor_to_grid(self, humor_scores: List[Dict[str, Any]], grid: np.ndarray) -> np.ndarray:
        if not humor_scores:
            return np.zeros_like(grid, dtype=float)
        points = [
            {"time": h["time"], "value": h["score"]}
            for h in humor_scores
            if isinstance(h, dict) and "time" in h and "score" in h
        ]
        s = extract_series(points)
        return interp_to_grid(s, grid, fill_value=0.0)

    def _time_to_index(self, grid: np.ndarray, t: float) -> int:
        step = float(grid[1] - grid[0]) if grid.size > 1 else self.step
        idx = int(round(t / step))
        return max(0, min(idx, grid.size - 1))

    def _event_flags(self, events: List[Dict[str, Any]], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        has_laughter = np.zeros(grid.size, dtype=bool)
        has_loud = np.zeros(grid.size, dtype=bool)
        for e in events:
            if not isinstance(e, dict) or "time" not in e:
                continue
            try:
                t = float(e["time"])
            except (TypeError, ValueError):
                continue
            idx = self._time_to_index(grid, t)
            etype = str(e.get("type", ""))
            if "laughter" in etype:
                has_laughter[idx] = True
            if "loud" in etype:
                has_loud[idx] = True
        return has_laughter, has_loud

    def _scene_flags(self, boundaries: List[float], grid: np.ndarray) -> np.ndarray:
        flags = np.zeros(grid.size, dtype=bool)
        for b in boundaries:
            try:
                t = float(b)
            except (TypeError, ValueError):
                continue
            flags[self._time_to_index(grid, t)] = True
        return flags

    def _dialogue_flags(
        self,
        grid: np.ndarray,
        speech_prob: np.ndarray,
        transcript_segments: List[Dict[str, Any]],
    ) -> np.ndarray:
        if speech_prob.size == grid.size and np.any(speech_prob > 0):
            return speech_prob >= self.dialogue_prob_threshold

        diff = np.zeros(grid.size + 1, dtype=int)
        for seg in transcript_segments:
            if not isinstance(seg, dict) or "start" not in seg or "end" not in seg:
                continue
            try:
                s = float(seg["start"])
                e = float(seg["end"])
            except (TypeError, ValueError):
                continue
            if e <= s:
                continue
            start_idx = self._time_to_index(grid, s)
            end_idx = self._time_to_index(grid, e)
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            diff[start_idx] += 1
            if end_idx + 1 < diff.size:
                diff[end_idx + 1] -= 1

        active = np.cumsum(diff[:-1])
        return active > 0
