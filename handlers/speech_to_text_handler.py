from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

from handlers.base_handler import BaseHandler


class SpeechToTextHandler(BaseHandler):
    """
    Извлекает сегменты речи с тайм‑кодами.
    """

    def __init__(
            self,
            model_name: str = "base",
            device: str | None = None,
            max_length_seconds: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length_seconds = max_length_seconds
        self._model: Optional[whisper.Whisper] = None

    def _ensure_loaded(self) -> None:
        """Ленивая загрузка модели с диагностикой."""
        print("🔄 Loading Whisper model...")

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed. Run: pip install torch torchaudio")

        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper not installed. Run: pip install openai-whisper")

        if self._model is None:
            device_str = self.device or ("cuda" if CUDA_AVAILABLE else "cpu")
            print(f"📱 Using device: {device_str}")

            try:
                self._model = whisper.load_model(
                    self.model_name,
                    device=device_str
                )
                print(f"✅ Whisper '{self.model_name}' loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load Whisper: {e}")
                raise RuntimeError(f"Whisper load failed: {e}")

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Вход: context['audio_path']
        Выход: transcript_segments, full_transcript
        """
        audio_path_any = context.get("audio_path", "")
        if not isinstance(audio_path_any, str) or not Path(audio_path_any).is_file():
            raise ValueError("`audio_path` missing or not a file in context")

        audio_path = Path(audio_path_any)
        print(f"🎤 Transcribing: {audio_path.name}")

        self._ensure_loaded()
        if self._model is None:
            raise RuntimeError("Whisper model failed to load")

        try:
            result = self._model.transcribe(
                str(audio_path),
                language=None,  # автоопределение
                verbose=False,
                word_timestamps=False,
            )

            segments: List[Dict[str, Any]] = []
            for seg in result.get("segments", []):
                start = float(seg["start"])
                end = float(seg["end"])
                text = seg["text"].strip()

                if text and (end - start) > 0.1:  # минимум 0.1s
                    segments.append({
                        "start": start,
                        "end": end,
                        "text": text
                    })

            full_transcript = " ".join(s["text"] for s in segments)

            context["transcript_segments"] = segments
            context["full_transcript"] = full_transcript

            print(f"✓ STT: {len(segments)} segments, {len(full_transcript):,} chars")
            return context

        except Exception as e:
            print(f"❌ STT failed: {e}")
            # В случае ошибки возвращаем пустой результат
            context["transcript_segments"] = []
            context["full_transcript"] = ""
            return context
