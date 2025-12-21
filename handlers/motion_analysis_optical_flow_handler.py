import time
from typing import Any

import cv2
import numpy as np

from handlers.base_handler import BaseHandler


class MotionAnalysisOpticalFlowHandler(BaseHandler):
    def __init__(
        self,
        resize_width: int = 320,
        frame_step: int = 1,
        num_levels: int = 5,
        pyr_scale: float = 0.5,
        win_size: int = 15,
        num_iters: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        use_gaussian: bool = False,
        motion_threshold: float = 0.1,
    ) -> None:
        self.resize_width = int(resize_width)
        self.frame_step = int(frame_step)
        self.num_levels = int(num_levels)
        self.pyr_scale = float(pyr_scale)
        self.win_size = int(win_size)
        self.num_iters = int(num_iters)
        self.poly_n = int(poly_n)
        self.poly_sigma = float(poly_sigma)
        self.use_gaussian = bool(use_gaussian)
        self.motion_threshold = float(motion_threshold)

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

            sum_by_sec: dict[int, float] = {}
            cnt_by_sec: dict[int, int] = {}

            prev_gray = None
            frame_index = -1
            processed_frames = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_index += 1
                if self.frame_step > 1 and (frame_index % self.frame_step) != 0:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if self.resize_width > 0:
                    h, w = gray.shape[:2]
                    if w != self.resize_width:
                        new_w = self.resize_width
                        new_h = int((new_w / w) * h)
                        gray = cv2.resize(
                            gray, (new_w, new_h), interpolation=cv2.INTER_AREA
                        )

                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray,
                        gray,
                        None,
                        self.pyr_scale,
                        self.num_levels,
                        self.win_size,
                        self.num_iters,
                        self.poly_n,
                        self.poly_sigma,
                        0,
                    )
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    max_mag = float(np.max(mag)) if mag.size else 0.0
                    if max_mag <= 0.0:
                        motion_value = 0.0
                    else:
                        mag_norm = mag / max_mag
                        moving_mask = mag_norm > self.motion_threshold
                        motion_value = float(np.sum(moving_mask)) / float(mag_norm.size)

                    t = float(frame_index) / float(fps)
                    second = int(t)
                    sum_by_sec[second] = sum_by_sec.get(second, 0.0) + motion_value
                    cnt_by_sec[second] = cnt_by_sec.get(second, 0) + 1

                prev_gray = gray
                processed_frames += 1

            if processed_frames < 2:
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
            context["detection_method"] = "optical_flow_farneback"
            context["processing_time_seconds"] = time.time() - start_time

            print(
                f"✓ Motion analysis (Optical Flow): seconds={len(motion_heatmap)} mean={mean_val:.3f} max={max_val:.3f} time={context['processing_time_seconds']:.2f}s"
            )

            return context

        finally:
            cap.release()
