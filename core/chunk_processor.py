"""
ChunkProcessor - обработка длинных видео по частям.

Для видео > 60 минут разбивает на чанки по 10 минут,
обрабатывает каждый отдельно и объединяет результаты.

Это экономит RAM и позволяет обрабатывать очень длинные видео.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import time


@dataclass
class VideoChunk:
    """Информация о чанке видео."""
    index: int
    start_time: float
    end_time: float
    duration: float
    temp_path: Optional[str] = None


@dataclass
class ChunkResult:
    """Результат обработки чанка."""
    chunk: VideoChunk
    context: Dict[str, Any]
    processing_time: float


@dataclass
class ChunkProcessorConfig:
    """Конфигурация ChunkProcessor."""
    chunk_duration_minutes: float = 10.0  # 10 минут на чанк
    overlap_seconds: float = 5.0  # Перекрытие между чанками
    threshold_minutes: float = 60.0  # Порог для включения chunk mode
    temp_dir: Optional[str] = None  # None = системная temp
    cleanup_temp: bool = True  # Удалять временные файлы


class ChunkProcessor:
    """Обработчик длинных видео по чанкам.

    Пример использования:
        processor = ChunkProcessor(config)

        if processor.should_use_chunking(video_path):
            results = processor.process_video(video_path, pipeline_func)
            merged = processor.merge_results(results)
        else:
            result = pipeline_func(video_path)
    """

    def __init__(self, config: Optional[ChunkProcessorConfig] = None):
        self.config = config or ChunkProcessorConfig()
        self._temp_files: List[str] = []

    def should_use_chunking(self, video_path: str) -> bool:
        """Определяет нужно ли использовать chunking."""
        duration = self._get_video_duration(video_path)
        threshold_seconds = self.config.threshold_minutes * 60
        return duration > threshold_seconds

    def _get_video_duration(self, video_path: str) -> float:
        """Получает длительность видео через ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def create_chunks(self, video_path: str) -> List[VideoChunk]:
        """Создаёт список чанков для видео."""
        duration = self._get_video_duration(video_path)
        chunk_duration = self.config.chunk_duration_minutes * 60
        overlap = self.config.overlap_seconds

        chunks = []
        start_time = 0.0
        index = 0

        while start_time < duration:
            end_time = min(start_time + chunk_duration, duration)
            actual_duration = end_time - start_time

            chunks.append(VideoChunk(
                index=index,
                start_time=start_time,
                end_time=end_time,
                duration=actual_duration,
            ))

            # Следующий чанк начинается с перекрытием
            start_time = end_time - overlap
            if end_time >= duration:
                break
            index += 1

        return chunks

    def extract_chunk(self, video_path: str, chunk: VideoChunk) -> str:
        """Извлекает чанк видео во временный файл.

        Returns:
            Путь к временному файлу с чанком
        """
        temp_dir = self.config.temp_dir or tempfile.gettempdir()
        video_name = Path(video_path).stem
        temp_path = os.path.join(temp_dir, f"{video_name}_chunk_{chunk.index}.mp4")

        cmd = [
            "ffmpeg",
            "-y",  # Перезаписать
            "-ss", str(chunk.start_time),
            "-i", video_path,
            "-t", str(chunk.duration),
            "-c", "copy",  # Без перекодирования
            temp_path,
        ]

        subprocess.run(cmd, capture_output=True)

        chunk.temp_path = temp_path
        self._temp_files.append(temp_path)

        return temp_path

    def process_video(
        self,
        video_path: str,
        pipeline_func: Callable[[str, float], Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ChunkResult]:
        """Обрабатывает видео по чанкам.

        Args:
            video_path: Путь к видео
            pipeline_func: Функция пайплайна, принимает (video_path, time_offset) -> context
            progress_callback: Коллбэк для отслеживания прогресса (current, total)

        Returns:
            Список результатов для каждого чанка
        """
        chunks = self.create_chunks(video_path)
        results = []

        print(f"[ChunkProcessor] Processing {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i, len(chunks))

            print(f"[ChunkProcessor] Chunk {i+1}/{len(chunks)}: {chunk.start_time:.0f}s - {chunk.end_time:.0f}s")

            start_time = time.monotonic()

            # Извлекаем чанк
            chunk_path = self.extract_chunk(video_path, chunk)

            # Обрабатываем
            try:
                context = pipeline_func(chunk_path, chunk.start_time)
            except Exception as e:
                print(f"[ChunkProcessor] Error processing chunk {i}: {e}")
                context = {"error": str(e)}

            processing_time = time.monotonic() - start_time

            results.append(ChunkResult(
                chunk=chunk,
                context=context,
                processing_time=processing_time,
            ))

            # Удаляем временный файл если настроено
            if self.config.cleanup_temp and chunk_path and os.path.exists(chunk_path):
                os.remove(chunk_path)

        return results

    def merge_results(self, results: List[ChunkResult]) -> Dict[str, Any]:
        """Объединяет результаты всех чанков.

        Handles:
        - Списки с временными метками (timeline, segments)
        - Списки без временных меток (highlights)
        - Скалярные значения (duration, summary stats)
        """
        if not results:
            return {}

        merged: Dict[str, Any] = {}

        # Собираем все ключи
        all_keys = set()
        for r in results:
            all_keys.update(r.context.keys())

        for key in all_keys:
            values = [r.context.get(key) for r in results if key in r.context]

            if not values:
                continue

            first_value = values[0]

            # Списки с time offset
            if isinstance(first_value, list) and first_value:
                if self._is_time_keyed_list(first_value):
                    merged[key] = self._merge_time_lists(values, results)
                else:
                    # Просто конкатенируем
                    merged[key] = []
                    for v in values:
                        if isinstance(v, list):
                            merged[key].extend(v)

            # Словари (например, summary)
            elif isinstance(first_value, dict):
                merged[key] = self._merge_dicts(values)

            # Скаляры - берем последний или суммируем
            elif isinstance(first_value, (int, float)):
                if key in ('duration_seconds', 'total_duration'):
                    merged[key] = sum(v for v in values if isinstance(v, (int, float)))
                else:
                    merged[key] = values[-1]  # Последнее значение

            else:
                merged[key] = values[-1]

        # Добавляем метаданные о chunking
        merged["_chunk_info"] = {
            "num_chunks": len(results),
            "total_processing_time": sum(r.processing_time for r in results),
            "chunks": [
                {
                    "index": r.chunk.index,
                    "start": r.chunk.start_time,
                    "end": r.chunk.end_time,
                    "processing_time": r.processing_time,
                }
                for r in results
            ],
        }

        return merged

    def _is_time_keyed_list(self, lst: List) -> bool:
        """Проверяет является ли список списком с временными метками."""
        if not lst:
            return False
        first = lst[0]
        if isinstance(first, dict):
            return 'time' in first or 'start' in first
        return False

    def _merge_time_lists(
        self,
        lists: List[List],
        results: List[ChunkResult],
    ) -> List:
        """Объединяет списки с временными метками, применяя offset."""
        merged = []

        for lst, result in zip(lists, results):
            if not isinstance(lst, list):
                continue

            offset = result.chunk.start_time

            for item in lst:
                if isinstance(item, dict):
                    # Применяем offset к временным полям
                    new_item = item.copy()
                    for time_key in ('time', 'start', 'end'):
                        if time_key in new_item and isinstance(new_item[time_key], (int, float)):
                            new_item[time_key] = new_item[time_key] + offset
                    merged.append(new_item)
                else:
                    merged.append(item)

        # Сортируем по времени если возможно
        if merged and isinstance(merged[0], dict):
            time_key = 'time' if 'time' in merged[0] else 'start' if 'start' in merged[0] else None
            if time_key:
                merged.sort(key=lambda x: x.get(time_key, 0))

        return merged

    def _merge_dicts(self, dicts: List[Dict]) -> Dict:
        """Объединяет словари (например, summary статистики)."""
        merged = {}

        for d in dicts:
            if not isinstance(d, dict):
                continue
            for k, v in d.items():
                if k not in merged:
                    merged[k] = v
                elif isinstance(v, (int, float)) and isinstance(merged[k], (int, float)):
                    # Для числовых значений суммируем или берём max
                    if 'count' in k or 'total' in k:
                        merged[k] += v
                    elif 'max' in k:
                        merged[k] = max(merged[k], v)
                    elif 'min' in k:
                        merged[k] = min(merged[k], v)
                    else:
                        # Для mean - нужно пересчитать, пока берём последний
                        merged[k] = v

        return merged

    def cleanup(self) -> None:
        """Удаляет все временные файлы."""
        for path in self._temp_files:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        self._temp_files.clear()

    def __del__(self):
        """Cleanup при уничтожении объекта."""
        if self.config.cleanup_temp:
            self.cleanup()
