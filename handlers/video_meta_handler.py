from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from handlers.base_handler import BaseHandler


class VideoMetaHandler(BaseHandler):
    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        video_path = context.get("video_path")
        if not video_path:
            raise ValueError("'video_path' not provided in context")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 0:
                fps = 25.0

            frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = (
                frame_count / fps
                if np.isfinite(frame_count) and frame_count > 0
                else None
            )

            context["fps"] = float(fps)
            context["frame_count"] = int(frame_count) if np.isfinite(frame_count) else None
            context["duration_seconds"] = duration_seconds
            return context
        finally:
            cap.release()
