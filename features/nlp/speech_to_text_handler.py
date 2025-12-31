from __future__ import annotations

import gc
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()

    # Detailed CUDA info
    if CUDA_AVAILABLE:
        CUDA_VERSION = torch.version.cuda
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        CUDA_VERSION = None
        GPU_NAME = None
        GPU_MEMORY_GB = 0
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    CUDA_VERSION = None
    GPU_NAME = None
    GPU_MEMORY_GB = 0

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

from core.base_handler import BaseHandler


class SpeechToTextHandler(BaseHandler):
    """
    Извлекает сегменты речи с тайм‑кодами.
    Использует faster-whisper для быстрой транскрипции.
    """

    def __init__(
            self,
            model_name: str = "base",
            device: str | None = None,
            compute_type: str = "float16",
            max_length_seconds: float = 60.0,
            num_workers: int = 4,
            release_model_after_run: bool = True,
            malloc_trim_after_run: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.max_length_seconds = max_length_seconds
        self.num_workers = num_workers
        self.release_model_after_run = release_model_after_run
        self.malloc_trim_after_run = malloc_trim_after_run
        self._model: Optional[WhisperModel] = None
        self._loaded_device: str | None = None

        print(f"STT Config: model={model_name}, device={device or 'auto'}, workers={num_workers}")

    def _release_memory(self) -> None:
        """
        Освобождает память после транскриба.

        Важно: faster-whisper / CTranslate2 держат часть памяти в нативном слое.
        Удаление объекта модели + gc.collect() обычно освобождает большую часть,
        а malloc_trim (Linux/glibc) помогает вернуть память ОС.
        """
        if self._model is None:
            return

        loaded_device = self._loaded_device
        model = self._model
        self._model = None
        self._loaded_device = None

        try:
            del model
        except Exception:
            pass

        gc.collect()

        if TORCH_AVAILABLE and loaded_device == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        if not self.malloc_trim_after_run:
            return

        if platform.system().lower() != "linux":
            return

        if os.environ.get("VIDEO_SERVICE_DISABLE_MALLOC_TRIM") == "1":
            return

        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            malloc_trim = getattr(libc, "malloc_trim", None)
            if malloc_trim is not None:
                malloc_trim(0)
        except Exception:
            pass

    def _ensure_loaded(self) -> None:
        """Ленивая загрузка модели с диагностикой."""
        print("🔄 Loading Whisper model (faster-whisper)...")

        # Детальная диагностика CUDA
        if TORCH_AVAILABLE:
            print(f"   PyTorch: {torch.__version__}")
            # Если явно выбрали CPU — не трогаем CUDA вообще (даже если она "доступна")
            if self.device == "cpu":
                print("   Device forced to CPU (CUDA disabled for this handler)")
            else:
                print(f"   CUDA available: {CUDA_AVAILABLE}")
            if CUDA_AVAILABLE:
                print(f"   CUDA version: {CUDA_VERSION}")
                print(f"   GPU: {GPU_NAME}")
                print(f"   VRAM: {GPU_MEMORY_GB:.1f} GB")

        if not WHISPER_AVAILABLE:
            raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")

        if self._model is None:
            # Resolve device with better fallback logic
            if self.device == "auto" or self.device is None:
                device_str = "cuda" if CUDA_AVAILABLE else "cpu"
            elif self.device == "cpu":
                device_str = "cpu"
            elif self.device == "cuda" and not CUDA_AVAILABLE:
                print("⚠️ CUDA requested but not available, falling back to CPU")
                device_str = "cpu"
            else:
                device_str = self.device

            # Adjust compute_type based on device
            compute_type = self.compute_type
            if device_str == "cpu" and compute_type == "float16":
                compute_type = "int8"  # float16 not supported on CPU
                print(f"⚠️ Switching to {compute_type} for CPU")

            print(f"📱 Using device: {device_str}, compute_type: {compute_type}")

            try:
                self._model = WhisperModel(
                    self.model_name,
                    device=device_str,
                    compute_type=compute_type,
                    num_workers=self.num_workers if device_str == "cuda" else 1,
                )
                self._loaded_device = device_str
                print(f"✅ Whisper '{self.model_name}' loaded successfully on {device_str}")
            except Exception as e:
                print(f"❌ Failed to load Whisper on {device_str}: {e}")
                if device_str != "cpu":
                    print("🔄 Retrying with CPU...")
                    try:
                        self._model = WhisperModel(
                            self.model_name,
                            device="cpu",
                            compute_type="int8"
                        )
                        self._loaded_device = "cpu"
                        print(f"✅ Whisper '{self.model_name}' loaded successfully on CPU (fallback)")
                    except Exception as e2:
                        raise RuntimeError(f"Whisper load failed on both {device_str} and CPU: {e2}")
                else:
                    raise RuntimeError(f"Whisper load failed: {e}")

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Вход: context['audio_path']
        Выход: transcript_segments, full_transcript
        """
        print("[8] SpeechToTextHandler")

        audio_path_any = context.get("audio_path", "")
        if not isinstance(audio_path_any, str) or not Path(audio_path_any).is_file():
            raise ValueError("`audio_path` missing or not a file in context")

        audio_path = Path(audio_path_any)
        print(f"🎤 Transcribing: {audio_path.name}")

        self._ensure_loaded()
        if self._model is None:
            raise RuntimeError("Whisper model failed to load")

        try:
            # faster-whisper returns (segments_generator, info)
            segments_gen, info = self._model.transcribe(
                str(audio_path),
                language=None,  # автоопределение
                beam_size=5,
                vad_filter=True,  # Filter out silence
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200
                )
            )

            detected_language = info.language

            segments: List[Dict[str, Any]] = []
            for seg in segments_gen:
                start = float(seg.start)
                end = float(seg.end)
                text = seg.text.strip()

                if text and (end - start) > 0.1:  # минимум 0.1s
                    segments.append({
                        "start": start,
                        "end": end,
                        "text": text
                    })

            full_transcript = " ".join(s["text"] for s in segments)

            context["transcript_segments"] = segments
            context["full_transcript"] = full_transcript
            context["language"] = detected_language

            print(f"✓ STT: {len(segments)} segments, {len(full_transcript):,} chars, language={detected_language}")
            return context

        except Exception as e:
            print(f"❌ STT failed: {e}")
            # В случае ошибки возвращаем пустой результат
            context["transcript_segments"] = []
            context["full_transcript"] = ""
            context["language"] = "unknown"
            return context
        finally:
            # Освобождаем модель и нативные буферы сразу после STT,
            # чтобы не держать память до конца пайплайна.
            if self.release_model_after_run:
                self._release_memory()
