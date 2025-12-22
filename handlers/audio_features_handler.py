import os
import time
from typing import Any

import librosa
import numpy as np

from handlers.base_handler import BaseHandler


class AudioFeaturesHandler(BaseHandler):
    def __init__(
        self,
        target_sr: int = 16000,
        window_size_ms: int = 200,
        hop_size_ms: int = 100,
        speech_threshold: float | None = None,
    ) -> None:
        self.target_sr = int(target_sr)
        self.window_size_ms = int(window_size_ms)
        self.hop_size_ms = int(hop_size_ms)
        self.speech_threshold = speech_threshold

    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        audio_path = context.get("audio_path")
        if not audio_path:
            raise ValueError("'audio_path' not provided in context")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        start_time = time.monotonic()

        y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        if y.size == 0:
            context["audio_features"] = {
                "loudness": [],
                "energy": [],
                "speech_probability": [],
            }
            context["audio_processing_time_seconds"] = time.monotonic() - start_time
            return context

        duration_seconds = context.get("duration_seconds")
        if not duration_seconds:
            duration_seconds = float(len(y)) / float(sr)
            context["duration_seconds"] = duration_seconds

        win_length = max(1, int(sr * self.window_size_ms / 1000))
        hop_length = max(1, int(sr * self.hop_size_ms / 1000))

        rms = librosa.feature.rms(y=y, frame_length=win_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=hop_length
        )

        rms_min = float(np.min(rms)) if rms.size else 0.0
        rms_max = float(np.max(rms)) if rms.size else 0.0
        denom = max(rms_max - rms_min, 1e-8)
        rms_norm = (rms - rms_min) / denom
        rms_norm = np.clip(rms_norm, 0.0, 1.0)

        threshold = (
            float(self.speech_threshold)
            if self.speech_threshold is not None
            else float(np.median(rms_norm))
        )
        threshold = float(np.clip(threshold, 0.0, 1.0))

        loudness = [
            {"time": float(times[i]), "value": float(rms_norm[i])}
            for i in range(len(times))
        ]
        energy = loudness.copy()
        speech_probability = [
            {
                "time": float(times[i]),
                "value": float(1.0 if rms_norm[i] > threshold else 0.0),
            }
            for i in range(len(times))
        ]

        context["audio_features"] = {
            "loudness": loudness,
            "energy": energy,
            "speech_probability": speech_probability,
        }
        context["audio_processing_time_seconds"] = time.monotonic() - start_time
        return context


