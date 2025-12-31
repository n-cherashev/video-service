"""
LazyModelManager - менеджер для ленивой загрузки и кэширования ML моделей.

Позволяет:
- Загружать модели только когда они нужны
- Кэшировать модели для повторного использования
- Выгружать неиспользуемые модели для экономии RAM
- Отслеживать использование памяти
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Dict, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ModelInfo:
    """Информация о загруженной модели."""
    name: str
    model: Any
    device: str
    loaded_at: float
    last_used_at: float
    size_mb: float = 0.0


class LazyModelManager:
    """Менеджер для ленивой загрузки ML моделей.

    Поддерживает:
    - Whisper (faster-whisper)
    - YAMNet (TensorFlow Hub)
    - Transformers (HuggingFace)
    - Sentiment models

    Пример использования:
        manager = LazyModelManager(device="cuda")
        whisper = manager.get_model("whisper", model_name="base")
        yamnet = manager.get_model("yamnet")

        # Позже - очистка неиспользуемых
        manager.cleanup_old_models(max_age_minutes=15)
    """

    _instance: Optional["LazyModelManager"] = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern для глобального доступа."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        default_device: str = "cuda",
        max_memory_mb: float = 8000.0,  # 8GB max
    ):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._models: Dict[str, ModelInfo] = {}
        self._loaders: Dict[str, Callable] = {}
        self.default_device = default_device
        self.max_memory_mb = max_memory_mb
        self._initialized = True

        # Регистрируем стандартные загрузчики
        self._register_default_loaders()

        print(f"[ModelManager] Initialized (device={default_device}, max_memory={max_memory_mb}MB)")

    def _register_default_loaders(self) -> None:
        """Регистрирует загрузчики для стандартных моделей."""
        self._loaders["whisper"] = self._load_whisper
        self._loaders["yamnet"] = self._load_yamnet
        self._loaders["sentiment"] = self._load_sentiment

    def register_loader(self, model_type: str, loader: Callable) -> None:
        """Регистрирует кастомный загрузчик модели."""
        self._loaders[model_type] = loader

    def get_model(
        self,
        model_type: str,
        device: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Получает модель, загружая её при необходимости.

        Args:
            model_type: Тип модели ('whisper', 'yamnet', 'sentiment')
            device: Устройство (None = default_device)
            **kwargs: Дополнительные параметры для загрузчика

        Returns:
            Загруженная модель
        """
        device = device or self.default_device
        cache_key = f"{model_type}_{device}_{hash(frozenset(kwargs.items()))}"

        # Проверяем кэш
        if cache_key in self._models:
            model_info = self._models[cache_key]
            model_info.last_used_at = time.time()
            return model_info.model

        # Проверяем лимит памяти
        self._check_memory_limit()

        # Загружаем модель
        if model_type not in self._loaders:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self._loaders.keys())}")

        loader = self._loaders[model_type]
        model = loader(device=device, **kwargs)

        # Оцениваем размер модели
        size_mb = self._estimate_model_size(model)

        # Кэшируем
        now = time.time()
        self._models[cache_key] = ModelInfo(
            name=model_type,
            model=model,
            device=device,
            loaded_at=now,
            last_used_at=now,
            size_mb=size_mb,
        )

        print(f"[ModelManager] Loaded {model_type} on {device} (~{size_mb:.0f}MB)")
        return model

    def cleanup_old_models(self, max_age_minutes: float = 15.0) -> int:
        """Выгружает модели, не использованные более max_age_minutes.

        Returns:
            Количество выгруженных моделей
        """
        now = time.time()
        max_age_seconds = max_age_minutes * 60

        to_remove = []
        for key, info in self._models.items():
            age = now - info.last_used_at
            if age > max_age_seconds:
                to_remove.append(key)

        for key in to_remove:
            self._unload_model(key)

        if to_remove:
            print(f"[ModelManager] Cleaned up {len(to_remove)} old models")

        return len(to_remove)

    def _check_memory_limit(self) -> None:
        """Проверяет лимит памяти и выгружает старые модели при необходимости."""
        total_size = sum(info.size_mb for info in self._models.values())

        if total_size > self.max_memory_mb:
            # Сортируем по времени последнего использования
            sorted_models = sorted(
                self._models.items(),
                key=lambda x: x[1].last_used_at
            )

            # Выгружаем самые старые пока не освободим достаточно памяти
            for key, info in sorted_models:
                if total_size <= self.max_memory_mb * 0.8:  # 80% от лимита
                    break
                total_size -= info.size_mb
                self._unload_model(key)

    def _unload_model(self, cache_key: str) -> None:
        """Выгружает модель из памяти."""
        if cache_key not in self._models:
            return

        info = self._models.pop(cache_key)

        # Очищаем CUDA кэш если модель была на GPU
        if info.device == "cuda" and TORCH_AVAILABLE:
            del info.model
            torch.cuda.empty_cache()
        else:
            del info.model

        gc.collect()
        print(f"[ModelManager] Unloaded {info.name} (freed ~{info.size_mb:.0f}MB)")

    def _estimate_model_size(self, model: Any) -> float:
        """Оценивает размер модели в MB."""
        try:
            # Для PyTorch моделей
            if TORCH_AVAILABLE and hasattr(model, 'parameters'):
                total_params = sum(p.numel() for p in model.parameters())
                return total_params * 4 / (1024 * 1024)  # 4 bytes per float32

            # Для faster-whisper
            if hasattr(model, 'model'):
                return 500.0  # ~500MB для base модели

            # Default оценка
            return 200.0
        except Exception:
            return 200.0

    def _load_whisper(self, device: str = "cuda", model_name: str = "base", compute_type: str = "float16", **kwargs) -> Any:
        """Загружает Whisper модель."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("faster-whisper not installed. Run: pip install faster-whisper")

        # Корректируем compute_type для CPU
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"

        return WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            num_workers=4 if device == "cuda" else 1,
        )

    def _load_yamnet(self, device: str = "cpu", **kwargs) -> Any:
        """Загружает YAMNet модель.

        Note: YAMNet всегда на CPU (TensorFlow).
        """
        try:
            import tensorflow_hub as hub
        except ImportError:
            raise ImportError("tensorflow-hub not installed. Run: pip install tensorflow-hub")

        yamnet_url = "https://tfhub.dev/google/yamnet/1"
        return hub.load(yamnet_url)

    def _load_sentiment(self, device: str = "cpu", **kwargs) -> Any:
        """Загружает sentiment модель."""
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")

        return pipeline(
            "sentiment-analysis",
            model="blanchefort/rubert-base-cased-sentiment-rusentiment",
            device=0 if device == "cuda" else -1,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику загруженных моделей."""
        return {
            "loaded_models": len(self._models),
            "total_size_mb": sum(info.size_mb for info in self._models.values()),
            "models": [
                {
                    "name": info.name,
                    "device": info.device,
                    "size_mb": info.size_mb,
                    "age_minutes": (time.time() - info.loaded_at) / 60,
                }
                for info in self._models.values()
            ],
        }

    def clear_all(self) -> None:
        """Выгружает все модели."""
        keys = list(self._models.keys())
        for key in keys:
            self._unload_model(key)
        print("[ModelManager] Cleared all models")


# Глобальный экземпляр (singleton)
MODEL_MANAGER: Optional[LazyModelManager] = None


def get_model_manager(device: str = "cuda") -> LazyModelManager:
    """Получает глобальный экземпляр ModelManager."""
    global MODEL_MANAGER
    if MODEL_MANAGER is None:
        MODEL_MANAGER = LazyModelManager(default_device=device)
    return MODEL_MANAGER
