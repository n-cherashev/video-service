from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from handlers.base_handler import BaseHandler


class AudioEventsHandler(BaseHandler):
    """Выделяет особые звуковые события."""

    def __init__(self, loudness_threshold: float = 0.8, energy_threshold: float = 0.7) -> None:
        self.loudness_threshold = loudness_threshold
        self.energy_threshold = energy_threshold

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        audio_features = context.get("audio_features", {})
        
        if not audio_features:
            context["audio_events"] = []
            return context

        events = self._detect_events(audio_features)
        context["audio_events"] = events
        
        print(f"✓ Audio events: {len(events)} detected")
        return context

    def _detect_events(self, audio_features: Dict) -> List[Dict]:
        """Детектирует звуковые события по пикам."""
        events = []
        
        loudness = audio_features.get("loudness", [])
        energy = audio_features.get("energy", [])
        
        if not loudness or not energy:
            return events
        
        # Нормализуем значения
        loudness_norm = self._normalize_array(loudness)
        energy_norm = self._normalize_array(energy)
        
        # Ищем пики
        min_len = min(len(loudness_norm), len(energy_norm))
        for i in range(min_len):
            loud = loudness_norm[i]
            eng = energy_norm[i]
            
            confidence = 0.0
            event_type = "unknown"
            
            # Очень громкий пик
            if loud > self.loudness_threshold and eng > self.energy_threshold:
                confidence = min((loud + eng) / 2, 1.0)
                event_type = "loud_event"
            
            # Добавляем событие если confidence достаточно высокий
            if confidence > 0.5:
                time = i * 0.1  # предполагаем 10 FPS для аудио фич
                events.append({
                    "time": time,
                    "type": event_type,
                    "confidence": confidence
                })
        
        return events

    def _normalize_array(self, arr: List) -> List[float]:
        """Нормализует массив в диапазон [0, 1]."""
        if not arr:
            return []
        
        # Фильтруем только числовые значения
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
        
        if not numeric_values:
            return [0.0] * len(arr)
        
        arr_np = np.array(numeric_values)
        min_val, max_val = arr_np.min(), arr_np.max()
        
        if max_val == min_val:
            return [0.0] * len(numeric_values)
        
        return ((arr_np - min_val) / (max_val - min_val)).tolist()