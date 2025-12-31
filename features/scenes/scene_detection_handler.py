from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from core.base_handler import BaseHandler

DetectorType = Literal["content", "adaptive"]


@dataclass(frozen=True)
class Scene:
    index: int
    start: float
    end: float


class SceneDetectionHandler(BaseHandler):
    """
    Единый прод-хэндлер детекта сцен с максимальной точностью.

    Пишет в context:
      - scenes: list[{"index": int, "start": float, "end": float}]
      - scene_boundaries: list[float] (все start, начиная с 0.0)
      - scene_summary: {"count": int, "mean_length": float, "max_length": float}

    Особенности для максимальной точности:
      - downscale=1 (полное разрешение)
      - frame_skip=0 (каждый кадр)
      - StatsManager для экспорта метрик (тюнинг threshold)
    """

    def __init__(
        self,
        detector: DetectorType = "adaptive",
        content_threshold: float = 27.0,
        adaptive_threshold: float = 3.0,
        min_scene_len_frames: int = 15,
        downscale: int = 1,
        frame_skip: int = 0,
        auto_downscale: bool = False,  # можно оставить, но не использовать
        min_scene_duration: float = 1.0,
        ensure_start_at_zero: bool = True,
    ) -> None:
        self.detector = detector
        self.content_threshold = float(content_threshold)
        self.adaptive_threshold = float(adaptive_threshold)
        self.min_scene_len_frames = int(min_scene_len_frames)
        self.downscale = int(downscale)
        self.frame_skip = int(frame_skip)
        self.auto_downscale = bool(auto_downscale)
        self.min_scene_duration = float(min_scene_duration)
        self.ensure_start_at_zero = bool(ensure_start_at_zero)


    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("[7] SceneDetectionHandler")
        
        video_path = context.get("video_path") or context.get("input_path")
        if not isinstance(video_path, str) or not video_path:
            raise ValueError("'video_path' (or 'input_path') not provided")
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        scenes = self._detect_scenes(video_path, context)
        scenes = self._postprocess_scenes(scenes)

        scene_dicts = [{"index": s.index, "start": s.start, "end": s.end} for s in scenes]
        boundaries = [s.start for s in scenes]

        if self.ensure_start_at_zero and (not boundaries or boundaries[0] != 0.0):
            boundaries = [0.0] + [b for b in boundaries if b > 0.0]
            if scenes and scenes[0].start > 0.0:
                scenes = [Scene(0, 0.0, scenes[0].start)] + [
                    Scene(i + 1, s.start, s.end) for i, s in enumerate(scenes)
                ]
                scene_dicts = [{"index": s.index, "start": s.start, "end": s.end} for s in scenes]

        lengths = [max(0.0, s.end - s.start) for s in scenes]
        summary = {
            "count": len(scenes),
            "mean_length": sum(lengths) / len(lengths) if lengths else 0.0,
            "max_length": max(lengths) if lengths else 0.0,
        }

        context["scenes"] = scene_dicts
        context["scene_boundaries"] = boundaries
        context["scene_summary"] = summary

        print(f"✓ Scene detection: {summary['count']} scenes "
              f"(downscale={self.downscale}, frame_skip={self.frame_skip})")
        return context

    def _detect_scenes(self, video_path: str, context: Dict[str, Any]) -> List[Scene]:
        try:
            from scenedetect import (
                open_video, SceneManager,
                ContentDetector, AdaptiveDetector
            )
        except ImportError as e:
            raise RuntimeError(
                "PySceneDetect required. Install: pip install scenedetect"
            ) from e

        # Детектор
        if self.detector == "content":
            detector = ContentDetector(
                threshold=self.content_threshold,
                min_scene_len=self.min_scene_len_frames,
            )
        else:  # adaptive
            detector = AdaptiveDetector(
                adaptive_threshold=self.adaptive_threshold,
                min_scene_len=self.min_scene_len_frames,
            )

        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(detector)

        # Максимальная точность: обрабатываем каждый кадр.
        # downscale=1 означает, что даунскейла нет. auto_downscale трогать не будем.
        scene_manager.downscale = self.downscale

        scene_manager.detect_scenes(
            video=video,
            show_progress=True,
            frame_skip=self.frame_skip,  # 0 = каждый кадр
        )

        scene_list = scene_manager.get_scene_list(start_in_scene=True)
        scenes: List[Scene] = []
        for i, (start_tc, end_tc) in enumerate(scene_list):
            scenes.append(
                Scene(
                    index=i,
                    start=float(start_tc.get_seconds()),
                    end=float(end_tc.get_seconds()),
                )
            )
        return scenes

    def _postprocess_scenes(self, scenes: List[Scene]) -> List[Scene]:
        if not scenes:
            return [Scene(0, 0.0, 0.0)]

        # Фильтруем короткие
        filtered = [s for s in scenes if (s.end - s.start) >= self.min_scene_duration]
        if not filtered:
            filtered = scenes

        # Переиндексируем
        return [Scene(i, s.start, s.end) for i, s in enumerate(filtered)]