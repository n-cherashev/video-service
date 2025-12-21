import time
from typing import Any

import cv2
import numpy as np

from handlers.base_handler import BaseHandler


class MotionAnalysisFrameDiffHandler(BaseHandler):
    def __init__(
        self,
        resize_width: int | None = 320,
        frame_step: int = 1,
    ) -> None:
        self.resize_width = int(resize_width) if resize_width is not None else None
        self.frame_step = int(frame_step)

    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        video_path = context.get("video_path")
        if not video_path:
            raise ValueError("'video_path' not provided in context")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        start_time = time.monotonic()
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                fps = 25.0

            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration_seconds = (
                float(frame_count) / float(fps)
                if np.isfinite(frame_count) and frame_count > 0
                else None
            )

            sum_by_sec: dict[int, float] = {}
            cnt_by_sec: dict[int, int] = {}

            prev_gray: np.ndarray | None = None
            frame_index = -1
            processed_frames = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_index += 1
                if self.frame_step > 1 and (frame_index % self.frame_step) != 0:
                    continue

                curr_gray = self._to_gray(frame)
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, curr_gray).astype(np.float32)
                    mean_diff = float(np.mean(diff)) if diff.size else 0.0

                    second = int(frame_index / fps)
                    sum_by_sec[second] = sum_by_sec.get(second, 0.0) + mean_diff
                    cnt_by_sec[second] = cnt_by_sec.get(second, 0) + 1

                prev_gray = curr_gray
                processed_frames += 1

            if processed_frames < 2:
                raise RuntimeError("Not enough frames to analyze motion")

            motion_heatmap, motion_summary = self._build_heatmap(
                sum_by_sec=sum_by_sec,
                cnt_by_sec=cnt_by_sec,
                duration_seconds=duration_seconds,
            )

            elapsed = time.monotonic() - start_time

            context["fps"] = float(fps)
            context["duration_seconds"] = duration_seconds
            context["motion_heatmap"] = motion_heatmap
            context["motion_summary"] = motion_summary
            context["motion_detection_method"] = "frame_diff"
            context["motion_processing_time_seconds"] = elapsed
            context["detection_method"] = "frame_diff"
            context["processing_time_seconds"] = elapsed

            return context
        finally:
            cap.release()

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.resize_width is None or self.resize_width <= 0:
            return gray

        h, w = gray.shape[:2]
        if w == self.resize_width:
            return gray

        new_h = int((self.resize_width / w) * h)
        return cv2.resize(gray, (self.resize_width, new_h), interpolation=cv2.INTER_AREA)

    def _build_heatmap(
        self,
        *,
        sum_by_sec: dict[int, float],
        cnt_by_sec: dict[int, int],
        duration_seconds: float | None,
    ) -> tuple[list[dict[str, float]], dict[str, float]]:
        if not sum_by_sec:
            return [], {"mean": 0.0, "max": 0.0, "seconds": 0}

        seconds_range = (
            range(0, int(duration_seconds) + 1)
            if duration_seconds is not None
            else sorted(sum_by_sec.keys())
        )

        raw_values: list[float] = []
        heatmap: list[dict[str, float]] = []

        for sec in seconds_range:
            if cnt_by_sec.get(sec, 0) > 0:
                avg = sum_by_sec[sec] / cnt_by_sec[sec]
            else:
                avg = 0.0
            raw_values.append(float(avg))

        min_val = float(np.min(raw_values)) if raw_values else 0.0
        max_val = float(np.max(raw_values)) if raw_values else 0.0
        denom = max(max_val - min_val, 1e-8)

        for idx, sec in enumerate(seconds_range):
            norm = (raw_values[idx] - min_val) / denom if denom > 1e-8 else 0.0
            heatmap.append({"time": float(sec), "value": float(norm)})

        norm_values = [point["value"] for point in heatmap]
        summary = {
            "mean": float(np.mean(norm_values)) if norm_values else 0.0,
            "max": float(np.max(norm_values)) if norm_values else 0.0,
            "seconds": len(heatmap),
        }

        return heatmap, summary

