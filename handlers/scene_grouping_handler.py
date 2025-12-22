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


class SceneGroupingHandler(BaseHandler):
    """Объединяет шоты в сцены на основе визуальной похожести."""

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        max_scene_gap: float = 5.0,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_scene_gap = max_scene_gap

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        shot_boundaries = context.get("shot_boundaries", [])
        duration = context.get("duration_seconds", 0)
        
        if len(shot_boundaries) < 2:
            # Создаем одну сцену на все видео
            scenes = [{
                "index": 0,
                "start": 0.0,
                "end": duration,
                "shots": [0] if shot_boundaries else []
            }]
            scene_summary = {
                "count": 1,
                "mean_length": duration,
                "max_length": duration
            }
        else:
            scenes, scene_summary = self._group_shots_into_scenes(
                shot_boundaries, duration, context
            )
        
        context["scenes"] = scenes
        context["scene_summary"] = scene_summary
        
        print(f"✓ Scenes: {scene_summary['count']} scenes, avg={scene_summary['mean_length']:.1f}s")
        return context

    def _group_shots_into_scenes(
        self, shot_boundaries: List[float], duration: float, context: Dict
    ) -> Tuple[List[Dict], Dict]:
        """Группирует шоты в сцены."""
        
        if not CV2_AVAILABLE:
            # Простая группировка без анализа кадров
            return self._simple_grouping(shot_boundaries, duration)
        
        video_path = context.get("input_path", "")
        if not Path(video_path).is_file():
            return self._simple_grouping(shot_boundaries, duration)
        
        # Получаем дескрипторы шотов
        shot_descriptors = self._extract_shot_descriptors(video_path, shot_boundaries, duration)
        
        # Группируем по похожести
        scenes = []
        current_scene_shots = [0]
        current_start = shot_boundaries[0]
        
        for k in range(len(shot_boundaries) - 1):
            # Вычисляем похожесть между шотами k и k+1
            similarity = self._compute_similarity(
                shot_descriptors[k], shot_descriptors[k + 1]
            )
            
            # Вычисляем временной разрыв
            gap = shot_boundaries[k + 1] - shot_boundaries[k]
            
            # Проверяем условия для продолжения сцены
            if similarity >= self.similarity_threshold and gap <= self.max_scene_gap:
                # Продолжаем текущую сцену
                current_scene_shots.append(k + 1)
            else:
                # Заканчиваем текущую сцену
                scene_end = shot_boundaries[k + 1] if k + 1 < len(shot_boundaries) else duration
                scenes.append({
                    "index": len(scenes),
                    "start": current_start,
                    "end": scene_end,
                    "shots": current_scene_shots.copy()
                })
                
                # Начинаем новую сцену
                current_scene_shots = [k + 1]
                current_start = shot_boundaries[k + 1]
        
        # Добавляем последнюю сцену
        scenes.append({
            "index": len(scenes),
            "start": current_start,
            "end": duration,
            "shots": current_scene_shots
        })
        
        # Вычисляем статистику
        scene_lengths = [scene["end"] - scene["start"] for scene in scenes]
        scene_summary = {
            "count": len(scenes),
            "mean_length": sum(scene_lengths) / len(scene_lengths) if scene_lengths else 0.0,
            "max_length": max(scene_lengths) if scene_lengths else 0.0
        }
        
        return scenes, scene_summary

    def _simple_grouping(self, shot_boundaries: List[float], duration: float) -> Tuple[List[Dict], Dict]:
        """Простая группировка без анализа кадров."""
        # Группируем по временным интервалам
        scenes = []
        current_shots = []
        current_start = shot_boundaries[0]
        
        for i, boundary in enumerate(shot_boundaries[1:], 1):
            gap = boundary - shot_boundaries[i - 1]
            
            if gap <= self.max_scene_gap:
                current_shots.append(i - 1)
            else:
                # Заканчиваем сцену
                if not current_shots:
                    current_shots = [i - 1]
                
                scenes.append({
                    "index": len(scenes),
                    "start": current_start,
                    "end": boundary,
                    "shots": current_shots.copy()
                })
                
                current_shots = []
                current_start = boundary
        
        # Последняя сцена
        if not current_shots:
            current_shots = [len(shot_boundaries) - 1]
        
        scenes.append({
            "index": len(scenes),
            "start": current_start,
            "end": duration,
            "shots": current_shots
        })
        
        scene_lengths = [scene["end"] - scene["start"] for scene in scenes]
        scene_summary = {
            "count": len(scenes),
            "mean_length": sum(scene_lengths) / len(scene_lengths) if scene_lengths else 0.0,
            "max_length": max(scene_lengths) if scene_lengths else 0.0
        }
        
        return scenes, scene_summary

    def _extract_shot_descriptors(
        self, video_path: str, shot_boundaries: List[float], duration: float
    ) -> List[np.ndarray]:
        """Извлекает дескрипторы для каждого шота."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        descriptors = []
        
        for i, start_time in enumerate(shot_boundaries):
            # Определяем конец шота
            end_time = shot_boundaries[i + 1] if i + 1 < len(shot_boundaries) else duration
            
            # Берем кадр из середины шота
            mid_time = (start_time + end_time) / 2
            frame_idx = int(mid_time * fps)
            
            # Переходим к нужному кадру
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                descriptor = self._compute_frame_descriptor(frame)
            else:
                # Если кадр не удалось прочитать, создаем нулевой дескriptор
                descriptor = np.zeros(96)  # 32*3 для HSV гистограмм
            
            descriptors.append(descriptor)
        
        cap.release()
        return descriptors

    def _compute_frame_descriptor(self, frame: np.ndarray) -> np.ndarray:
        """Вычисляет компактный дескриптор кадра."""
        # Уменьшаем размер
        small = cv2.resize(frame, (160, 90))
        
        # Переводим в HSV
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        
        # Вычисляем гистограммы
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Нормализуем и объединяем
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        return np.concatenate([hist_h, hist_s, hist_v])

    def _compute_similarity(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """Вычисляет похожесть между дескрипторами."""
        # Используем косинусную похожесть
        dot_product = np.dot(desc1, desc2)
        norm1 = np.linalg.norm(desc1)
        norm2 = np.linalg.norm(desc2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, min(1.0, similarity))