import time
from typing import Any

import cv2
import numpy as np

from handlers.base_handler import BaseHandler


class MotionAnalysisBackgroundSubHandler(BaseHandler):
    def __init__(
        self,
        method: str = "mog2",
        resize_width: int = 320,
        frame_step: int = 1,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        learning_rate: float = -1.0,
        background_ratio: float = 0.7,
    ) -> None:
        self.method = method.lower()
        self.resize_width = int(resize_width)
        self.frame_step = int(frame_step)
        self.history = int(history)
        self.var_threshold = float(var_threshold)
        self.detect_shadows = bool(detect_shadows)
        self.learning_rate = float(learning_rate)
        self.background_ratio = float(background_ratio)

    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        video_path = context.get("video_path")
        if not video_path:
            raise ValueError("'video_path' not provided in context")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        start_time = time.time()
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

            if self.method == "mog2":
                bg_sub = cv2.createBackgroundSubtractorMOG2(
                    history=self.history,
                    varThreshold=self.var_threshold,
                    detectShadows=self.detect_shadows,
                )
                try:
                    bg_sub.setBackgroundRatio(self.background_ratio)
                except Exception:
                    pass
            elif self.method == "knn":
                bg_sub = cv2.createBackgroundSubtractorKNN(
                    history=self.history, detectShadows=self.detect_shadows
                )
            else:
                raise ValueError(
                    "Unknown background subtraction method: %s" % self.method
                )

            # Warm-up
            warmup_frames = min(200, max(0, int(self.history / 2)))
            for i in range(warmup_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                if self.resize_width > 0:
                    h, w = frame.shape[:2]
                    if w != self.resize_width:
                        new_w = self.resize_width
                        new_h = int((new_w / w) * h)
                        frame = cv2.resize(
                            frame, (new_w, new_h), interpolation=cv2.INTER_AREA
                        )
                bg_sub.apply(frame, learningRate=self.learning_rate)

            # Reset to start of analysis (we continue from current position)

            sum_by_sec: dict[int, float] = {}
            cnt_by_sec: dict[int, int] = {}

            frame_index = -1
            processed_frames = 0

            # Main loop
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_index += 1

                if self.frame_step > 1 and (frame_index % self.frame_step) != 0:
                    continue

                if self.resize_width > 0:
                    h, w = frame.shape[:2]
                    if w != self.resize_width:
                        new_w = self.resize_width
                        new_h = int((new_w / w) * h)
                        frame = cv2.resize(
                            frame, (new_w, new_h), interpolation=cv2.INTER_AREA
                        )

                fg_mask = bg_sub.apply(frame, learningRate=self.learning_rate)

                if self.detect_shadows:
                    fg_mask[fg_mask == 127] = 0

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

                foreground_pixels = int(cv2.countNonZero(fg_mask))
                total_pixels = int(fg_mask.shape[0] * fg_mask.shape[1])
                motion_value = (
                    float(foreground_pixels) / float(total_pixels)
                    if total_pixels > 0
                    else 0.0
                )

                t = float(frame_index) / float(fps)
                second = int(t)
                sum_by_sec[second] = sum_by_sec.get(second, 0.0) + motion_value
                cnt_by_sec[second] = cnt_by_sec.get(second, 0) + 1

                processed_frames += 1

            if processed_frames < 1:
                raise RuntimeError("Not enough frames to analyze motion")

            secs = sorted(sum_by_sec.keys())
            motion_heatmap: list[dict[str, float]] = []
            avg_values: list[float] = []
            for sec in secs:
                avg = sum_by_sec[sec] / cnt_by_sec[sec]
                avg_values.append(avg)

            max_avg = float(np.max(avg_values)) if avg_values else 0.0
            for i, sec in enumerate(secs):
                norm = float(avg_values[i] / max_avg) if max_avg > 0 else 0.0
                motion_heatmap.append({"time": float(sec), "value": float(norm)})

            values = [p["value"] for p in motion_heatmap]
            mean_val = float(np.mean(values)) if values else 0.0
            max_val = float(np.max(values)) if values else 0.0

            context["fps"] = float(fps)
            context["duration_seconds"] = duration_seconds
            context["motion_heatmap"] = motion_heatmap
            context["motion_summary"] = {
                "mean": mean_val,
                "max": max_val,
                "seconds": len(motion_heatmap),
            }
            context["detection_method"] = (
                "background_subtraction_mog2"
                if self.method == "mog2"
                else "background_subtraction_knn"
            )
            context["processing_time_seconds"] = time.time() - start_time

            print(
                f"✓ Motion analysis (Background Sub {self.method}): seconds={len(motion_heatmap)} mean={mean_val:.3f} max={max_val:.3f} time={context['processing_time_seconds']:.2f}s"
            )

            return context

        finally:
            cap.release()
