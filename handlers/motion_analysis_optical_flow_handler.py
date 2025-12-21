import time
from typing import Any, Dict, List, Optional, TypedDict

import cv2
import cv2.typing as cv2t
import numpy as np
import numpy.typing as npt

from handlers.base_handler import BaseHandler


class MotionHeatmapEntry(TypedDict):
    time: float
    value: float


class MotionSummary(TypedDict):
    mean: float
    max: float
    seconds: int


class MotionAnalysisOpticalFlowHandler(BaseHandler):
    """Анализатор движения видео с использованием оптического потока Farneback."""

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
        self.resize_width: int = int(resize_width)
        self.frame_step: int = int(frame_step)
        self.num_levels: int = int(num_levels)
        self.pyr_scale: float = float(pyr_scale)
        self.win_size: int = int(win_size)
        self.num_iters: int = int(num_iters)
        self.poly_n: int = int(poly_n)
        self.poly_sigma: float = float(poly_sigma)
        self.use_gaussian: bool = bool(use_gaussian)
        self.motion_threshold: float = float(motion_threshold)

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        video_path: str = context.get("video_path")
        if not isinstance(video_path, str) or not video_path:
            raise ValueError("'video_path' not provided in context")

        cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        start_time: float = time.time()
        try:
            fps_raw: float = float(cap.get(cv2.CAP_PROP_FPS))
            fps: float = fps_raw if np.isfinite(fps_raw) and fps_raw > 0 else 25.0

            frame_count_raw: float = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds: Optional[float] = (
                frame_count_raw / fps
                if np.isfinite(frame_count_raw) and frame_count_raw > 0
                else None
            )

            sum_by_sec: Dict[int, float] = {}
            cnt_by_sec: Dict[int, int] = {}

            prev_gray: Optional[npt.NDArray[np.uint8]] = None
            frame_index: int = -1
            processed_frames: int = 0

            while True:
                ok: bool
                frame: cv2t.MatLike
                ok, frame = cap.read()
                if not ok:
                    break

                frame_index += 1
                if self.frame_step > 1 and (frame_index % self.frame_step) != 0:
                    continue

                # Конвертация в grayscale: (H, W, 3) -> (H, W)
                gray: npt.NDArray[np.uint8] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Опциональный ресайз
                if self.resize_width > 0:
                    h: int
                    w: int
                    h, w = gray.shape[:2]
                    if w != self.resize_width:
                        new_w: int = self.resize_width
                        new_h: int = int((new_w / float(w)) * float(h))
                        gray = cv2.resize(
                            gray,
                            (new_w, new_h),
                            interpolation=cv2.INTER_AREA,
                        )

                if prev_gray is not None:
                    # flow: (H, W, 2) с float32
                    flow: npt.NDArray[np.float32] = np.zeros(
                        (gray.shape[0], gray.shape[1], 2), dtype=np.float32
                    )
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray,
                        gray,
                        flow,
                        self.pyr_scale,
                        self.num_levels,
                        self.win_size,
                        self.num_iters,
                        self.poly_n,
                        self.poly_sigma,
                        0,
                    )

                    # mag, ang: (H, W) с float32
                    mag: npt.NDArray[np.float32]
                    ang: npt.NDArray[np.float32]
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                    max_mag: float = float(np.max(mag)) if mag.size > 0 else 0.0

                    if max_mag <= 0.0:
                        motion_value: float = 0.0
                    else:
                        mag_norm: npt.NDArray[np.float32] = mag / max_mag
                        moving_mask: npt.NDArray[np.bool_] = (
                            mag_norm > self.motion_threshold
                        )
                        motion_value: float = float(
                            np.count_nonzero(moving_mask)
                        ) / float(mag_norm.size)

                    t: float = float(frame_index) / fps
                    second: int = int(t)

                    sum_by_sec[second] = sum_by_sec.get(second, 0.0) + motion_value
                    cnt_by_sec[second] = cnt_by_sec.get(second, 0) + 1

                prev_gray = gray
                processed_frames += 1

            if processed_frames < 2:
                raise RuntimeError("Not enough frames to analyze motion")

            secs: List[int] = sorted(sum_by_sec.keys())
            motion_heatmap: List[MotionHeatmapEntry] = []
            avg_values: List[float] = []

            for sec in secs:
                total: float = sum_by_sec[sec]
                count: int = cnt_by_sec[sec]
                avg: float = total / float(count) if count > 0 else 0.0
                avg_values.append(avg)

            max_avg: float = float(np.max(avg_values)) if avg_values else 0.0

            # ИСПРАВЛЕНО: убрана аннотация i: int из цикла for
            for i, sec in enumerate(secs):
                norm: float = float(avg_values[i] / max_avg) if max_avg > 0.0 else 0.0
                motion_heatmap.append({"time": float(sec), "value": norm})

            values: List[float] = [p["value"] for p in motion_heatmap]
            mean_val: float = float(np.mean(values)) if values else 0.0
            max_val: float = float(np.max(values)) if values else 0.0

            context["fps"] = float(fps)
            context["duration_seconds"] = duration_seconds
            context["motion_heatmap"] = motion_heatmap
            context["motion_summary"] = {
                "mean": mean_val,
                "max": max_val,
                "seconds": len(motion_heatmap),
            }
            context["detection_method"] = "optical_flow_farneback"
            context["processing_time_seconds"] = float(time.time() - start_time)

            print(
                f"✓ Motion analysis (Optical Flow): "
                f"seconds={len(motion_heatmap)} "
                f"mean={mean_val:.3f} max={max_val:.3f} "
                f"time={context['processing_time_seconds']:.2f}s"
            )

            return context

        finally:
            cap.release()
