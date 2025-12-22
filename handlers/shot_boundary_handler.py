from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

from handlers.base_handler import BaseHandler

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class ShotBoundaryHandler(BaseHandler):
    """Находит границы шотов (резкие переходы между кадрами)."""

    def __init__(
        self,
        sample_fps: float = 4.0,
        diff_threshold: float = 0.4,
        min_shot_length: float = 0.5,
    ) -> None:
        self.sample_fps = sample_fps
        self.diff_threshold = diff_threshold
        self.min_shot_length = min_shot_length

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not CV2_AVAILABLE:
            print("⚠️ OpenCV not available, skipping shot detection")
            context["shot_boundaries"] = [0.0]
            context["shot_scores"] = []
            return context

        video_path = context.get("input_path", "")
        if not Path(video_path).is_file():
            raise ValueError(f"video_path not found: {video_path}")

        shot_boundaries, shot_scores = self._detect_shots(video_path, context)
        
        context["shot_boundaries"] = shot_boundaries
        context["shot_scores"] = shot_scores
        
        if shot_scores:
            avg_score = sum(s["score"] for s in shot_scores) / len(shot_scores)
            max_score = max(s["score"] for s in shot_scores)
            print(f"✓ Shots: {len(shot_boundaries)} boundaries, avg={avg_score:.3f}, max={max_score:.3f}")
        else:
            print(f"✓ Shots: {len(shot_boundaries)} boundaries")
        
        return context

    def _detect_shots(self, video_path: str, context: Dict) -> Tuple[List[float], List[Dict]]:
        """Детектирует границы шотов."""
        cap = cv2.VideoCapture(video_path)
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        # Обновляем context если нужно
        if "fps" not in context:
            context["fps"] = video_fps
        if "duration_seconds" not in context:
            context["duration_seconds"] = duration
        
        frame_step = max(1, int(video_fps / self.sample_fps))
        
        shot_boundaries = [0.0]
        shot_scores = []
        prev_hist = None
        frame_idx = 0
        last_boundary_time = 0.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_step == 0:
                time_sec = frame_idx / video_fps
                
                # Вычисляем гистограмму
                hist = self._compute_histogram(frame)
                
                if prev_hist is not None:
                    # Вычисляем различие
                    diff = self._compute_difference(prev_hist, hist)
                    shot_scores.append({"time": time_sec, "score": diff})
                    
                    # Проверяем порог
                    if diff >= self.diff_threshold:
                        # Проверяем минимальную длину шота
                        if time_sec - last_boundary_time >= self.min_shot_length:
                            shot_boundaries.append(time_sec)
                            last_boundary_time = time_sec
                
                prev_hist = hist
            
            frame_idx += 1
        
        cap.release()
        return shot_boundaries, shot_scores

    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Вычисляет нормализованную гистограмму кадра."""
        # Уменьшаем размер для скорости
        small = cv2.resize(frame, (160, 90))
        
        # Переводим в HSV
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        
        # Вычисляем гистограмму по всем каналам
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Нормализуем
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Объединяем
        return np.concatenate([hist_h, hist_s, hist_v])

    def _compute_difference(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Вычисляет различие между гистограммами."""
        # Используем корреляцию
        correlation = cv2.compareHist(
            hist1.astype(np.float32),
            hist2.astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        
        # Преобразуем в различие (0 = одинаковые, 1 = разные)
        diff = 1.0 - correlation
        return max(0.0, min(1.0, diff))
