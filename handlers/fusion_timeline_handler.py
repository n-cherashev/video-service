from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from handlers.base_handler import BaseHandler


class FusionTimelineHandler(BaseHandler):
    """Создает единую временную шкалу с интегральным interest_score."""

    def __init__(self, step: float = 1.0) -> None:
        self.step = step
        # Веса для расчета interest
        self.weights = {
            "motion": 0.25,
            "audio": 0.20,
            "sentiment": 0.15,
            "humor": 0.20,
            "events": 0.20
        }

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        duration = context.get("duration_seconds", 0)
        if duration <= 0:
            context["timeline"] = []
            return context

        timeline = self._build_timeline(context, duration)
        context["timeline"] = timeline
        
        print(f"✓ Timeline: {len(timeline)} points, step={self.step}s")
        return context

    def _build_timeline(self, context: Dict, duration: float) -> List[Dict]:
        """Строит временную шкалу с интеграцией всех модальностей."""
        times = np.arange(0, duration + self.step, self.step)
        timeline = []
        
        # Интерполируем данные для каждой модальности
        motion_data = self._interpolate_motion(context, times)
        audio_data = self._interpolate_audio(context, times)
        sentiment_data = self._interpolate_sentiment(context, times)
        humor_data = self._interpolate_humor(context, times)
        
        # Флаги событий
        event_flags = self._get_event_flags(context, times)
        scene_flags = self._get_scene_flags(context, times)
        dialogue_flags = self._get_dialogue_flags(context, times)
        
        for i, time in enumerate(times):
            motion = motion_data[i] if i < len(motion_data) else 0.0
            audio_loudness = audio_data[i] if i < len(audio_data) else 0.0
            sentiment = sentiment_data[i] if i < len(sentiment_data) else 0.0
            humor = humor_data[i] if i < len(humor_data) else 0.0
            
            has_laughter = event_flags.get("laughter", {}).get(i, False)
            has_loud_sound = event_flags.get("loud", {}).get(i, False)
            is_scene_boundary = scene_flags.get(i, False)
            is_dialogue = dialogue_flags.get(i, False)
            
            # Расчет интегрального interest
            interest = self._calculate_interest(
                motion, audio_loudness, sentiment, humor,
                has_laughter, has_loud_sound
            )
            
            timeline.append({
                "time": float(time),
                "motion": motion,
                "audio_loudness": audio_loudness,
                "sentiment": sentiment,
                "humor": humor,
                "has_laughter": has_laughter,
                "has_loud_sound": has_loud_sound,
                "is_scene_boundary": is_scene_boundary,
                "is_dialogue": is_dialogue,
                "interest": interest
            })
        
        return timeline

    def _interpolate_motion(self, context: Dict, times: np.ndarray) -> List[float]:
        """Интерполирует данные движения."""
        motion_heatmap = context.get("motion_heatmap", [])
        if not motion_heatmap:
            return [0.0] * len(times)
        
        # Безопасное преобразование в числа
        numeric_motion = self._safe_numeric_array(motion_heatmap)
        if not numeric_motion:
            return [0.0] * len(times)
        
        motion_times = np.linspace(0, times[-1], len(numeric_motion))
        return np.interp(times, motion_times, numeric_motion).tolist()

    def _interpolate_audio(self, context: Dict, times: np.ndarray) -> List[float]:
        """Интерполирует аудио данные."""
        audio_features = context.get("audio_features", {})
        loudness = audio_features.get("loudness", [])
        
        if not loudness:
            return [0.0] * len(times)
        
        # Безопасное преобразование в числа
        numeric_loudness = self._safe_numeric_array(loudness)
        if not numeric_loudness:
            return [0.0] * len(times)
        
        audio_times = np.linspace(0, times[-1], len(numeric_loudness))
        return np.interp(times, audio_times, numeric_loudness).tolist()

    def _interpolate_sentiment(self, context: Dict, times: np.ndarray) -> List[float]:
        """Интерполирует данные сентимента."""
        sentiment_timeline = context.get("sentiment_timeline", [])
        if not sentiment_timeline:
            return [0.0] * len(times)
        
        sent_times = []
        sent_values = []
        
        for s in sentiment_timeline:
            if isinstance(s, dict) and "time" in s and "sentiment" in s:
                sent_times.append(float(s["time"]))
                sent_values.append(float(s["sentiment"]))
        
        if not sent_times or not sent_values:
            return [0.0] * len(times)
        
        return np.interp(times, sent_times, sent_values).tolist()

    def _interpolate_humor(self, context: Dict, times: np.ndarray) -> List[float]:
        """Интерполирует данные юмора."""
        humor_scores = context.get("humor_scores", [])
        if not humor_scores:
            return [0.0] * len(times)
        
        humor_times = []
        humor_values = []
        
        for h in humor_scores:
            if isinstance(h, dict) and "time" in h and "score" in h:
                humor_times.append(float(h["time"]))
                humor_values.append(float(h["score"]))
        
        if not humor_times or not humor_values:
            return [0.0] * len(times)
        
        return np.interp(times, humor_times, humor_values).tolist()

    def _get_event_flags(self, context: Dict, times: np.ndarray) -> Dict:
        """Получает флаги событий."""
        events = context.get("audio_events", [])
        flags = {"laughter": {}, "loud": {}}
        
        for event in events:
            event_time = event["time"]
            event_type = event["type"]
            
            # Находим ближайший индекс времени
            idx = np.argmin(np.abs(times - event_time))
            
            if "laughter" in event_type:
                flags["laughter"][idx] = True
            elif "loud" in event_type:
                flags["loud"][idx] = True
        
        return flags

    def _get_scene_flags(self, context: Dict, times: np.ndarray) -> Dict:
        """Получает флаги границ сцен."""
        boundaries = context.get("scene_boundaries", [])
        flags = {}
        
        for boundary in boundaries:
            idx = np.argmin(np.abs(times - boundary))
            flags[idx] = True
        
        return flags

    def _get_dialogue_flags(self, context: Dict, times: np.ndarray) -> Dict:
        """Получает флаги диалогов."""
        segments = context.get("transcript_segments", [])
        flags = {}
        
        for segment in segments:
            start_idx = np.argmin(np.abs(times - segment["start"]))
            end_idx = np.argmin(np.abs(times - segment["end"]))
            
            for i in range(start_idx, end_idx + 1):
                flags[i] = True
        
        return flags

    def _safe_numeric_array(self, arr: List) -> List[float]:
        """Безопасно преобразует массив в числовые значения."""
        numeric_values = []
        for val in arr:
            if isinstance(val, (int, float)):
                numeric_values.append(float(val))
            elif hasattr(val, '__float__'):
                try:
                    numeric_values.append(float(val))
                except (ValueError, TypeError):
                    numeric_values.append(0.0)
            else:
                numeric_values.append(0.0)
    def _calculate_interest(self, motion: float, audio: float, sentiment: float, 
                          humor: float, has_laughter: bool, has_loud_sound: bool) -> float:
        """Вычисляет интегральный interest score."""
        interest = (
            self.weights["motion"] * motion +
            self.weights["audio"] * audio +
            self.weights["sentiment"] * abs(sentiment) +
            self.weights["humor"] * humor +
            self.weights["events"] * (0.5 if has_laughter else 0.0) +
            self.weights["events"] * (0.3 if has_loud_sound else 0.0)
        )
        
        return min(interest, 1.0)