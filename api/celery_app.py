"""
Celery Application для асинхронной обработки видео.

Задачи:
- analyze_video: полный анализ видео
- extract_clips: извлечение клипов из видео

Запуск worker:
    celery -A api.celery_app worker --loglevel=info --concurrency=1

Мониторинг:
    celery -A api.celery_app flower
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery import Celery
from celery.result import AsyncResult

# Конфигурация
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
TEMP_DIR = Path(os.environ.get("VIDEO_SERVICE_TEMP_DIR", "temp"))
RESULTS_DIR = Path(os.environ.get("VIDEO_SERVICE_OUTPUT_DIR", "results"))

TEMP_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def _cleanup_input_if_temp(video_path: str) -> None:
    try:
        p = Path(video_path).resolve()
        temp_root = TEMP_DIR.resolve()
        if temp_root in p.parents:
            p.unlink()
    except Exception:
        return

# Добавляем корень проекта в path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Celery App
# ============================================================================

celery_app = Celery(
    "video_service",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=7200,  # 2 hours max
    task_soft_time_limit=6000,  # 100 min soft limit
    worker_prefetch_multiplier=1,  # Один таск за раз
    worker_concurrency=1,  # Один worker process (GPU bound)
    result_expires=86400,  # Результаты хранятся 24 часа
)


# ============================================================================
# Redis Meta Storage
# ============================================================================

def get_redis_client():
    """Возвращает Redis клиент."""
    try:
        import redis
        return redis.from_url(REDIS_URL)
    except ImportError:
        return None


def save_task_meta(task_id: str, meta: Dict[str, Any]) -> None:
    """Сохраняет метаданные задачи в Redis."""
    client = get_redis_client()
    if client:
        key = f"task_meta:{task_id}"
        client.setex(key, 86400, json.dumps(meta))


def _publish_task_event(task_id: str, payload: Dict[str, Any]) -> None:
    client = get_redis_client()
    if not client:
        return
    try:
        channel = f"task_events:{task_id}"
        client.publish(channel, json.dumps(payload, ensure_ascii=False))
    except Exception:
        return


def get_task_meta(task_id: str) -> Optional[Dict[str, Any]]:
    """Получает метаданные задачи из Redis."""
    client = get_redis_client()
    if client:
        key = f"task_meta:{task_id}"
        data = client.get(key)
        if data:
            return json.loads(data)
    return None


# ============================================================================
# Tasks
# ============================================================================

@celery_app.task(bind=True, name="video_service.analyze_video")
def analyze_video(
    self,
    video_path: str,
    settings: Dict[str, Any],
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Анализирует видео и возвращает результат.

    Args:
        video_path: Путь к видео файлу
        settings: Настройки анализа
        task_id: ID задачи (если задан извне)

    Returns:
        Результат анализа
    """
    celery_task_id = self.request.id
    effective_task_id = task_id or celery_task_id

    try:
        # Обновляем статус
        self.update_state(state="PROCESSING", meta={"progress": 0.1})
        _update_meta(effective_task_id, "processing", 0.1)

        from config.settings import VideoServiceSettings
        from core.dag_executor import DAGExecutor, DAGNode
        from models.serde import to_jsonable

        # Создаём настройки
        video_settings = VideoServiceSettings(
            max_viral_clips=settings.get("max_clips", 8),
            min_clip_duration=settings.get("min_duration", 30.0),
            max_clip_duration=settings.get("max_duration", 60.0),
            llm_enabled=settings.get("enable_llm", False),
            block_analysis_enabled=settings.get("enable_llm", False),
            enable_llm_refine=settings.get("enable_llm", False),
            llm_base_url=os.environ.get("VIDEO_SERVICE_LLM_BASE_URL", "http://host.docker.internal:11434"),
            llm_model=os.environ.get("VIDEO_SERVICE_LLM_MODEL", "qwen2.5-coder:14b"),
        )

        self.update_state(state="PROCESSING", meta={"progress": 0.2})
        _update_meta(effective_task_id, "processing", 0.2)

        # Запускаем пайплайн
        from main import run_dag_pipeline

        context = {
            "input_path": video_path,
            "settings": video_settings,
            "task_id": effective_task_id,
        }

        start_time = time.monotonic()
        result_context, exec_result = run_dag_pipeline(context, video_settings)
        processing_time = time.monotonic() - start_time

        self.update_state(state="PROCESSING", meta={"progress": 0.9})
        _update_meta(effective_task_id, "processing", 0.9)

        # Конвертируем результат
        result = to_jsonable(result_context)
        result["processing_time_seconds"] = processing_time
        result["task_id"] = effective_task_id

        # Сохраняем public результат
        public_path = RESULTS_DIR / f"{effective_task_id}.json"
        public_data = result.get("public", result)
        with open(public_path, "w", encoding="utf-8") as f:
            json.dump(public_data, f, indent=2, ensure_ascii=False)

        # Сохраняем full результат
        full_path = RESULTS_DIR / f"{effective_task_id}.full.json"
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Обновляем метаданные
        _update_meta(effective_task_id, "completed", 1.0, completed=True)

        return {
            "status": "completed",
            "task_id": effective_task_id,
            "duration_seconds": result.get("duration_seconds", 0),
            "processing_time_seconds": processing_time,
            "viral_clips_count": len(result.get("viral_clips", [])),
            "result_path": str(public_path),
        }

    except Exception as e:
        _update_meta(effective_task_id, "failed", 0.0, error=str(e))
        raise

    finally:
        # Очищаем временный файл
        _cleanup_input_if_temp(video_path)


@celery_app.task(bind=True, name="video_service.extract_clips")
def extract_clips(
    self,
    video_path: str,
    clips: List[Dict[str, Any]],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Извлекает клипы из видео через ffmpeg.

    Args:
        video_path: Путь к исходному видео
        clips: Список клипов [{start, end, name}, ...]
        output_dir: Директория для сохранения

    Returns:
        Информация об извлечённых клипах
    """
    import subprocess

    task_id = self.request.id
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extracted = []

    for i, clip in enumerate(clips):
        start = clip.get("start", 0)
        end = clip.get("end", 0)
        name = clip.get("name", f"clip_{i+1}")

        output_file = output_path / f"{name}.mp4"

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(end - start),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "fast",
                str(output_file),
            ]

            subprocess.run(cmd, capture_output=True, check=True)

            extracted.append({
                "name": name,
                "path": str(output_file),
                "start": start,
                "end": end,
                "duration": end - start,
            })

        except subprocess.CalledProcessError as e:
            extracted.append({
                "name": name,
                "error": str(e),
            })

        # Обновляем прогресс
        progress = (i + 1) / len(clips)
        self.update_state(state="PROCESSING", meta={"progress": progress})

    return {
        "status": "completed",
        "task_id": task_id,
        "clips_extracted": len([c for c in extracted if "path" in c]),
        "clips": extracted,
    }


def _update_meta(
    task_id: str,
    status: str,
    progress: float,
    error: Optional[str] = None,
    completed: bool = False,
) -> None:
    """Обновляет метаданные задачи в Redis."""
    meta = get_task_meta(task_id) or {}
    meta["status"] = status
    meta["progress"] = progress
    if error:
        meta["error"] = error
    if completed:
        meta["completed_at"] = datetime.utcnow().isoformat()
    save_task_meta(task_id, meta)
    _publish_task_event(
        task_id,
        {
            "task_id": task_id,
            "status": status,
            "progress": progress,
            "error": error,
            "message": meta.get("message"),
        },
    )


# ============================================================================
# Utility Functions
# ============================================================================

def get_task_status(task_id: str) -> Dict[str, Any]:
    """Получает статус задачи Celery."""
    result = AsyncResult(task_id, app=celery_app)

    status = {
        "task_id": task_id,
        "status": result.state,
    }

    if result.state == "PROCESSING":
        status["progress"] = result.info.get("progress", 0) if result.info else 0
    elif result.state == "SUCCESS":
        status["result"] = result.result
    elif result.state == "FAILURE":
        status["error"] = str(result.result)

    return status


def revoke_task(task_id: str, terminate: bool = True) -> bool:
    """Отменяет задачу."""
    celery_app.control.revoke(task_id, terminate=terminate)
    return True


# Alias for backwards compatibility
app = celery_app
