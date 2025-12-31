from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class TimelinePoint:
    """Точка временной шкалы с 7-компонентной системой сигналов.

    Новые поля согласно ТЗ:
    - audio_energy: спектральная мощность аудио
    - speech_probability: вероятность речи (VAD)
    - clarity: качество речи (SNR-based)
    """
    time: float
    interest: float
    motion: float
    audio_loudness: float
    # Новые поля для 7-компонентной формулы
    audio_energy: float = 0.0
    speech_probability: float = 0.0
    clarity: float = 0.5
    # Существующие поля
    sentiment: float = 0.0
    humor: float = 0.0
    has_laughter: bool = False
    has_loud_sound: bool = False
    is_scene_boundary: bool = False
    is_dialogue: bool = False
