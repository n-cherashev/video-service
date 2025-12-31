"""
AudioFeaturesHandler - извлекает базовые аудио признаки.

Hybrid (Extractor + Analyzer): читает аудио и вычисляет features.
Поддерживает streaming для длинных файлов.
"""
from __future__ import annotations

import os
import time
from typing import Any, ClassVar, FrozenSet, Generator, Tuple

import librosa
import numpy as np
import soundfile as sf

from core.base_handler import BaseHandler
from models.keys import Key
from utils.timeseries import normalize_01, sigmoid


class AudioFeaturesHandler(BaseHandler):
    """Извлекает базовые аудио признаки как тайм-ряды.

    Features:
    - loudness: RMS (нормированный 0..1)
    - energy: RMS^2 (нормированный 0..1)
    - speech_probability: 0..1 (плавная оценка вокруг порога)

    Поддерживает streaming processing для экономии RAM на длинных аудио.

    Provides:
    - audio_features: dict с loudness, energy, speech_probability
    - audio_features_meta: метаданные обработки
    """

    requires: ClassVar[FrozenSet[Key]] = frozenset({Key.AUDIO_PATH})
    provides: ClassVar[FrozenSet[Key]] = frozenset({Key.AUDIO_SERIES})

    def __init__(
        self,
        target_sr: int = 16000,
        window_size_ms: int = 200,
        hop_size_ms: int = 100,
        speech_threshold: float | None = None,
        speech_sigmoid_k: float = 0.08,
        smooth_window: int = 3,
        chunk_duration_sec: float = 30.0,
        use_streaming: bool = True,
        streaming_threshold_sec: float = 300.0,
    ) -> None:
        self.target_sr = int(target_sr)
        self.window_size_ms = int(window_size_ms)
        self.hop_size_ms = int(hop_size_ms)
        self.speech_threshold = speech_threshold
        self.speech_sigmoid_k = float(speech_sigmoid_k)
        self.smooth_window = int(smooth_window)
        self.chunk_duration_sec = float(chunk_duration_sec)
        self.use_streaming = use_streaming
        self.streaming_threshold_sec = float(streaming_threshold_sec)

    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        print("[5] AudioFeaturesHandler")

        audio_path = context.get("audio_path")
        if not audio_path:
            raise ValueError("'audio_path' not provided in context")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        start_time = time.monotonic()

        file_duration = self._get_file_duration(audio_path)

        if self.use_streaming and file_duration > self.streaming_threshold_sec:
            print(f"   Using streaming mode for {file_duration:.0f}s audio")
            result = self._process_streaming(audio_path, context)
        else:
            result = self._process_full(audio_path, context)

        result["audio_processing_time_seconds"] = time.monotonic() - start_time
        return result

    def _get_file_duration(self, audio_path: str) -> float:
        """Получает длительность аудио файла без полной загрузки."""
        try:
            info = sf.info(audio_path)
            return info.duration
        except Exception:
            return 0.0

    def _process_full(self, audio_path: str, context: dict[str, Any]) -> dict[str, Any]:
        """Обрабатывает весь файл целиком."""
        y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)

        if y.size == 0:
            context["audio_features"] = {"loudness": [], "energy": [], "speech_probability": []}
            context["audio_features_meta"] = {
                "sr": self.target_sr,
                "window_ms": self.window_size_ms,
                "hop_ms": self.hop_size_ms,
            }
            return context

        duration_seconds = context.get("duration_seconds")
        if not duration_seconds:
            context["duration_seconds"] = float(len(y)) / float(sr)

        win_length = max(1, int(sr * self.window_size_ms / 1000))
        hop_length = max(1, int(sr * self.hop_size_ms / 1000))

        rms = librosa.feature.rms(y=y, frame_length=win_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        loudness = normalize_01(rms.astype(float))
        energy = normalize_01((rms.astype(float) ** 2))

        if self.smooth_window >= 3:
            kernel = np.ones(self.smooth_window, dtype=float) / float(self.smooth_window)
            loudness = np.convolve(loudness, kernel, mode="same")
            energy = np.convolve(energy, kernel, mode="same")

        threshold = float(self.speech_threshold) if self.speech_threshold is not None else float(np.median(loudness))
        threshold = float(np.clip(threshold, 0.0, 1.0))

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
            "num_frames": len(times),
        }

        print(f"✓ Audio features: {len(times)} frames, sr={sr}Hz")
        return context

    def _process_streaming(self, audio_path: str, context: dict[str, Any]) -> dict[str, Any]:
        """Обрабатывает файл по чанкам для экономии памяти."""
        all_times = []
        all_rms = []

        chunk_samples = int(self.chunk_duration_sec * self.target_sr)
        win_length = max(1, int(self.target_sr * self.window_size_ms / 1000))
        hop_length = max(1, int(self.target_sr * self.hop_size_ms / 1000))

        time_offset = 0.0

        for chunk, sr in self._read_audio_chunks(audio_path, chunk_samples):
            if chunk.size == 0:
                continue

            rms = librosa.feature.rms(y=chunk, frame_length=win_length, hop_length=hop_length)[0]
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            times = times + time_offset

            all_times.extend(times)
            all_rms.extend(rms)

            time_offset += len(chunk) / sr

        if not all_rms:
            context["audio_features"] = {"loudness": [], "energy": [], "speech_probability": []}
            context["audio_features_meta"] = {
                "sr": self.target_sr,
                "window_ms": self.window_size_ms,
                "hop_ms": self.hop_size_ms,
            }
            return context

        all_rms = np.array(all_rms)
        all_times = np.array(all_times)

        loudness = normalize_01(all_rms.astype(float))
        energy = normalize_01((all_rms.astype(float) ** 2))

        if self.smooth_window >= 3:
            kernel = np.ones(self.smooth_window, dtype=float) / float(self.smooth_window)
            loudness = np.convolve(loudness, kernel, mode="same")
            energy = np.convolve(energy, kernel, mode="same")

        threshold = float(self.speech_threshold) if self.speech_threshold is not None else float(np.median(loudness))
        threshold = float(np.clip(threshold, 0.0, 1.0))

        k = max(self.speech_sigmoid_k, 1e-6)
        speech_probability = sigmoid((loudness - threshold) / k)

        context["duration_seconds"] = context.get("duration_seconds") or time_offset

        context["audio_features"] = {
            "loudness": [{"time": float(all_times[i]), "value": float(loudness[i])} for i in range(len(all_times))],
            "energy": [{"time": float(all_times[i]), "value": float(energy[i])} for i in range(len(all_times))],
            "speech_probability": [{"time": float(all_times[i]), "value": float(speech_probability[i])} for i in range(len(all_times))],
        }

        context["audio_features_meta"] = {
            "sr": self.target_sr,
            "window_ms": int(self.window_size_ms),
            "hop_ms": int(self.hop_size_ms),
            "frame_length": int(win_length),
            "hop_length": int(hop_length),
            "speech_threshold": float(threshold),
            "streaming": True,
            "chunk_duration": self.chunk_duration_sec,
            "num_frames": len(all_times),
        }

        print(f"✓ Audio features (streaming): {len(all_times)} frames")
        return context

    def _read_audio_chunks(
        self,
        audio_path: str,
        chunk_samples: int,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Генератор для чтения аудио по чанкам."""
        try:
            with sf.SoundFile(audio_path) as f:
                sr_native = f.samplerate

                while True:
                    chunk = f.read(chunk_samples, dtype='float32')
                    if chunk.size == 0:
                        break

                    if len(chunk.shape) > 1:
                        chunk = np.mean(chunk, axis=1)

                    if sr_native != self.target_sr:
                        chunk = librosa.resample(chunk, orig_sr=sr_native, target_sr=self.target_sr)

                    yield chunk, self.target_sr

        except Exception as e:
            print(f"Warning: streaming read failed ({e}), falling back to full load")
            y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            yield y, sr
