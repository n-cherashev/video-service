from __future__ import annotations

import os
import time
from typing import Any

import librosa
import numpy as np

from handlers.base_handler import BaseHandler
from handlers.timeseries import normalize_01, sigmoid


class AudioFeaturesHandler(BaseHandler):
    """
    Извлекает базовые аудио признаки как тайм-ряды:
      - loudness: RMS (нормированный 0..1)
      - energy: RMS^2 (нормированный 0..1)
      - speech_probability: 0..1 (плавная оценка вокруг порога)
    """

    def __init__(
        self,
        target_sr: int = 16000,
        window_size_ms: int = 200,
        hop_size_ms: int = 100,
        speech_threshold: float | None = None,
        speech_sigmoid_k: float = 0.08,  # чем меньше, тем резче переход
        smooth_window: int = 3,  # сглаживание по кадрам (не по секундам)
    ) -> None:
        self.target_sr = int(target_sr)
        self.window_size_ms = int(window_size_ms)
        self.hop_size_ms = int(hop_size_ms)
        self.speech_threshold = speech_threshold
        self.speech_sigmoid_k = float(speech_sigmoid_k)
        self.smooth_window = int(smooth_window)

    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        audio_path = context.get("audio_path")
        if not audio_path:
            raise ValueError("'audio_path' not provided in context")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        start_time = time.monotonic()

        y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        if y.size == 0:
            context["audio_features"] = {"loudness": [], "energy": [], "speech_probability": []}
            context["audio_features_meta"] = {"sr": self.target_sr, "window_ms": self.window_size_ms, "hop_ms": self.hop_size_ms}
            context["audio_processing_time_seconds"] = time.monotonic() - start_time
            return context

        duration_seconds = context.get("duration_seconds")
        if not duration_seconds:
            context["duration_seconds"] = float(len(y)) / float(sr)

        win_length = max(1, int(sr * self.window_size_ms / 1000))
        hop_length = max(1, int(sr * self.hop_size_ms / 1000))

        # RMS по фреймам (окнам) [page:1]
        rms = librosa.feature.rms(y=y, frame_length=win_length, hop_length=hop_length)[0]

        # Преобразование индексов фреймов в секунды [page:0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        loudness = normalize_01(rms.astype(float))
        energy = normalize_01((rms.astype(float) ** 2))

        if self.smooth_window >= 3:
            k = self.smooth_window
            kernel = np.ones(k, dtype=float) / float(k)
            loudness = np.convolve(loudness, kernel, mode="same")
            energy = np.convolve(energy, kernel, mode="same")

        threshold = float(self.speech_threshold) if self.speech_threshold is not None else float(np.median(loudness))
        threshold = float(np.clip(threshold, 0.0, 1.0))

        # Плавная “вероятность речи”: sigmoid вокруг порога
        k = max(self.speech_sigmoid_k, 1e-6)
        speech_probability = sigmoid((loudness - threshold) / k)

        context["audio_features"] = {
            "loudness": [{"time": float(times[i]), "value": float(loudness[i])} for i in range(len(times))],
            "energy": [{"time": float(times[i]), "value": float(energy[i])} for i in range(len(times))],
            "speech_probability": [{"time": float(times[i]), "value": float(speech_probability[i])} for i in range(len(times))],
        }
        context["audio_features_meta"] = {
            "sr": int(sr),
            "window_ms": int(self.window_size_ms),
            "hop_ms": int(self.hop_size_ms),
            "frame_length": int(win_length),
            "hop_length": int(hop_length),
            "speech_threshold": float(threshold),
        }
        context["audio_processing_time_seconds"] = time.monotonic() - start_time
        return context
