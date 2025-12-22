from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path
import numpy as np

from handlers.base_handler import BaseHandler

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class SceneChangeHandler(BaseHandler):
    """Выделяет границы сцен по визуальным изменениям."""

    def __init__(self, threshold: float = 0.3, sample_fps: float = 2.0) -> None:
        self.threshold = threshold
        self.sample_fps = sample_fps

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not CV2_AVAILABLE:
            print("⚠️ OpenCV not available, skipping scene detection")
            context["scene_boundaries"] = [0.0]
            return context

        video_path = context.get("input_path", "")
        if not Path(video_path).is_file():
            context["scene_boundaries"] = [0.0]
            return context

        boundaries = self._detect_scene_changes(video_path)
        context["scene_boundaries"] = boundaries
        
        print(f"✓ Scene changes: {len(boundaries)} boundaries")
        return context

    def _detect_scene_changes(self, video_path: str) -> List[float]:
        """Детектирует смены сцен по гистограммам."""
        boundaries = [0.0]  # Всегда начинаем с 0
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps / self.sample_fps))
            
            prev_hist = None
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Обрабатываем только каждый N-й кадр
                if frame_count % frame_interval == 0:
                    # Вычисляем гистограмму яркости
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    
                    if prev_hist is not None:
                        # Сравниваем гистограммы
                        diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                        
                        # Если корреляция низкая - сцена изменилась
                        if diff < (1.0 - self.threshold):
                            time_sec = frame_count / fps
                            boundaries.append(time_sec)
                    
                    prev_hist = hist
                
                frame_count += 1
            
            cap.release()
            
        except Exception as e:
            print(f"⚠️ Scene detection error: {e}")
        
        return sorted(boundaries)