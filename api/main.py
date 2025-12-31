"""
Video Service - FastAPI Application

Production-ready REST API для анализа видео.
Использует Celery + Redis для асинхронной обработки.

Запуск:
    uvicorn api.main:app --reload

Документация:
    http://localhost:8000/docs
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
import asyncio
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Конфигурация
TEMP_DIR = Path(os.environ.get("VIDEO_SERVICE_TEMP_DIR", "temp"))
RESULTS_DIR = Path(os.environ.get("VIDEO_SERVICE_OUTPUT_DIR", "results"))
USE_CELERY = os.environ.get("USE_CELERY", "false").lower() == "true"
ALLOWED_INPUT_DIR_RAW = str(os.environ.get("VIDEO_SERVICE_ALLOWED_INPUT_DIR", "") or "").strip()
ALLOWED_INPUT_DIR = Path(ALLOWED_INPUT_DIR_RAW).resolve() if ALLOWED_INPUT_DIR_RAW else None

TEMP_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def _is_allowed_input_path(path: Path) -> bool:
    if ALLOWED_INPUT_DIR is None:
        return False
    try:
        return path.resolve().is_relative_to(ALLOWED_INPUT_DIR)
    except Exception:
        return False


def _cleanup_input_if_temp(video_path: str) -> None:
    try:
        p = Path(video_path).resolve()
        temp_root = TEMP_DIR.resolve()
        if temp_root in p.parents:
            p.unlink()
    except Exception:
        return


# ============================================================================
# Pydantic Models
# ============================================================================

class AnalysisRequest(BaseModel):
    """Запрос на анализ видео."""
    max_clips: int = Field(8, ge=1, le=20, description="Максимальное количество клипов")
    min_duration: float = Field(30.0, ge=10, le=300, description="Мин. длина клипа (сек)")
    max_duration: float = Field(60.0, ge=15, le=600, description="Макс. длина клипа (сек)")
    enable_llm: bool = Field(False, description="Использовать LLM refinement")


class TaskStatus(BaseModel):
    """Статус задачи анализа."""
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: float = 0.0
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_url: Optional[str] = None
    error: Optional[str] = None


class ViralClipResponse(BaseModel):
    """Viral клип."""
    id: str = ""
    start: float
    end: float
    duration: float = 0.0
    score: float
    score_breakdown: Dict[str, float] = {}
    anchor_type: str = ""
    reasons: List[str] = []


class ChapterResponse(BaseModel):
    """Глава."""
    id: str = ""
    start: float
    end: float
    duration: float = 0.0
    title: str
    description: str = ""


class AnalysisSummaryResponse(BaseModel):
    """Сводка анализа."""
    total_scenes: int = 0
    total_speech_duration: float = 0.0
    speech_ratio: float = 0.0
    mean_motion: float = 0.0
    mean_loudness: float = 0.0
    mean_interest: float = 0.0
    detected_language: Optional[str] = None


class AnalysisResultResponse(BaseModel):
    """Результат анализа."""
    task_id: str
    video_name: str
    duration_seconds: float
    processing_time_seconds: float
    viral_clips: List[ViralClipResponse]
    chapters: List[ChapterResponse] = []
    summary: AnalysisSummaryResponse = AnalysisSummaryResponse()
    created_at: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
    gpu_available: bool
    celery_enabled: bool


class AnalyzePathRequest(BaseModel):
    """Запрос на анализ файла, доступного на диске сервера (без upload)."""
    path: str = Field(..., min_length=1, description="Путь к видеофайлу внутри контейнера/сервера")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Video Analyzer API",
    description="REST API для автоматического поиска viral-моментов в видео",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production ограничить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Task Status via Redis
# ============================================================================

def get_redis_client():
    """Возвращает Redis клиент."""
    try:
        import redis
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        return redis.from_url(redis_url)
    except ImportError:
        return None


def save_task_meta(task_id: str, meta: Dict[str, Any]) -> None:
    """Сохраняет метаданные задачи в Redis."""
    client = get_redis_client()
    if client:
        key = f"task_meta:{task_id}"
        client.setex(key, 86400, json.dumps(meta))  # TTL 24h


def get_task_meta(task_id: str) -> Optional[Dict[str, Any]]:
    """Получает метаданные задачи из Redis."""
    client = get_redis_client()
    if client:
        key = f"task_meta:{task_id}"
        data = client.get(key)
        if data:
            return json.loads(data)
    return None


def _publish_task_event(task_id: str, payload: Dict[str, Any]) -> None:
    """Публикует событие прогресса задачи в Redis pubsub (для WebSocket)."""
    client = get_redis_client()
    if not client:
        return
    try:
        channel = f"task_events:{task_id}"
        client.publish(channel, json.dumps(payload, ensure_ascii=False))
    except Exception:
        # pubsub не должен ломать пайплайн
        return


def update_task_progress(
    task_id: str,
    progress: float,
    status: str = "processing",
    message: Optional[str] = None,
) -> None:
    """Обновляет прогресс задачи + публикует событие для WebSocket."""
    meta = get_task_meta(task_id) or {}
    meta["progress"] = progress
    meta["status"] = status
    if message:
        meta["message"] = message
    save_task_meta(task_id, meta)
    _publish_task_event(
        task_id,
        {
            "task_id": task_id,
            "status": status,
            "progress": progress,
            "message": message,
        },
    )


@app.websocket("/ws/tasks/{task_id}")
async def ws_task_progress(websocket: WebSocket, task_id: str) -> None:
    """WebSocket канал прогресса задачи (без polling со стороны UI)."""
    await websocket.accept()

    # Отправляем начальный снэпшот
    meta = get_task_meta(task_id)
    if meta:
        await websocket.send_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "status": meta.get("status", "unknown"),
                    "progress": meta.get("progress", 0.0),
                    "message": meta.get("message"),
                    "error": meta.get("error"),
                },
                ensure_ascii=False,
            )
        )

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis.asyncio as redis  # type: ignore
    except Exception:
        redis = None  # type: ignore

    if not redis:
        # Fallback: серверный polling Redis (UI не делает HTTP polling)
        try:
            while True:
                await asyncio.sleep(1.0)
                meta = get_task_meta(task_id) or {}
                await websocket.send_text(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "status": meta.get("status", "unknown"),
                            "progress": meta.get("progress", 0.0),
                            "message": meta.get("message"),
                            "error": meta.get("error"),
                        },
                        ensure_ascii=False,
                    )
                )
        except WebSocketDisconnect:
            return

    client = redis.from_url(redis_url, decode_responses=True)
    pubsub = client.pubsub()
    channel = f"task_events:{task_id}"

    try:
        await pubsub.subscribe(channel)

        while True:
            try:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg and msg.get("data"):
                    await websocket.send_text(str(msg["data"]))
                else:
                    await asyncio.sleep(0.1)
            except WebSocketDisconnect:
                break
    finally:
        try:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
        except Exception:
            pass
        try:
            await client.close()
        except Exception:
            pass


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False

    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=datetime.utcnow().isoformat(),
        gpu_available=gpu_available,
        celery_enabled=USE_CELERY,
    )


@app.post("/analyze", response_model=TaskStatus, tags=["Analysis"])
async def create_analysis(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    max_clips: int = Query(8, ge=1, le=20),
    min_duration: float = Query(30.0, ge=10, le=300),
    max_duration: float = Query(60.0, ge=15, le=600),
    enable_llm: bool = Query(False),
):
    """
    Создаёт задачу анализа видео.

    Возвращает task_id для отслеживания прогресса.
    Анализ выполняется асинхронно (Celery или BackgroundTasks).
    """
    if not video.filename:
        raise HTTPException(400, "Filename is required")

    allowed_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    ext = Path(video.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(400, f"Unsupported format. Allowed: {allowed_extensions}")

    task_id = str(uuid.uuid4())
    video_path = TEMP_DIR / f"{task_id}{ext}"

    try:
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
    except Exception as e:
        raise HTTPException(500, f"Failed to save video: {e}")

    settings = {
        "max_clips": max_clips,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "enable_llm": enable_llm,
    }

    created_at = datetime.utcnow().isoformat()

    # Сохраняем метаданные задачи
    task_meta = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "created_at": created_at,
        "video_path": str(video_path),
        "video_name": video.filename,
        "settings": settings,
    }
    save_task_meta(task_id, task_meta)

    # Запускаем анализ
    if USE_CELERY:
        from api.celery_app import analyze_video
        analyze_video.delay(str(video_path), settings, task_id)
    else:
        background_tasks.add_task(run_analysis_task, task_id, str(video_path), settings)

    return TaskStatus(
        task_id=task_id,
        status="pending",
        progress=0.0,
        created_at=created_at,
    )


@app.post("/analyze-path", response_model=TaskStatus, tags=["Analysis"])
async def create_analysis_by_path(
    background_tasks: BackgroundTasks,
    request: AnalyzePathRequest,
    max_clips: int = Query(8, ge=1, le=20),
    min_duration: float = Query(30.0, ge=10, le=300),
    max_duration: float = Query(60.0, ge=15, le=600),
    enable_llm: bool = Query(False),
):
    """
    Создаёт задачу анализа видео по пути на диске (без upload).

    Для безопасности путь должен находиться внутри директории,
    заданной env `VIDEO_SERVICE_ALLOWED_INPUT_DIR`.
    """
    if ALLOWED_INPUT_DIR is None:
        raise HTTPException(
            403,
            "analyze-path is disabled. Set VIDEO_SERVICE_ALLOWED_INPUT_DIR to enable it.",
        )

    video_path = Path(request.path).expanduser().resolve()
    if not video_path.exists():
        raise HTTPException(404, f"Video not found: {video_path}")
    if not video_path.is_file():
        raise HTTPException(400, f"Path is not a file: {video_path}")
    if not _is_allowed_input_path(video_path):
        raise HTTPException(403, f"Path is outside allowed directory: {ALLOWED_INPUT_DIR}")

    allowed_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    ext = video_path.suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(400, f"Unsupported format. Allowed: {allowed_extensions}")

    task_id = str(uuid.uuid4())
    settings = {
        "max_clips": max_clips,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "enable_llm": enable_llm,
    }

    created_at = datetime.utcnow().isoformat()
    task_meta = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "created_at": created_at,
        "video_path": str(video_path),
        "video_name": video_path.name,
        "settings": settings,
        "source": "path",
    }
    save_task_meta(task_id, task_meta)

    if USE_CELERY:
        from api.celery_app import analyze_video
        analyze_video.delay(str(video_path), settings, task_id)
    else:
        background_tasks.add_task(run_analysis_task, task_id, str(video_path), settings)

    return TaskStatus(
        task_id=task_id,
        status="pending",
        progress=0.0,
        created_at=created_at,
    )


@app.get("/tasks/{task_id}", response_model=TaskStatus, tags=["Analysis"])
async def get_task_status_endpoint(task_id: str):
    """Получает статус задачи анализа."""
    if USE_CELERY:
        from api.celery_app import get_task_status as get_celery_status
        status = get_celery_status(task_id)

        celery_to_api = {
            "PENDING": "pending",
            "STARTED": "processing",
            "PROCESSING": "processing",
            "SUCCESS": "completed",
            "FAILURE": "failed",
        }

        return TaskStatus(
            task_id=task_id,
            status=celery_to_api.get(status["status"], status["status"]),
            progress=status.get("progress", 0.0),
            result_url=f"/results/{task_id}" if status["status"] == "SUCCESS" else None,
            error=status.get("error"),
        )
    else:
        meta = get_task_meta(task_id)
        if not meta:
            raise HTTPException(404, "Task not found")

        return TaskStatus(
            task_id=task_id,
            status=meta.get("status", "unknown"),
            progress=meta.get("progress", 0.0),
            created_at=meta.get("created_at"),
            completed_at=meta.get("completed_at"),
            result_url=f"/results/{task_id}" if meta.get("status") == "completed" else None,
            error=meta.get("error"),
        )


@app.get("/results/{task_id}", response_model=AnalysisResultResponse, tags=["Results"])
async def get_analysis_result(task_id: str):
    """Получает результат анализа."""
    # Проверяем статус
    meta = get_task_meta(task_id)
    if meta and meta.get("status") != "completed":
        raise HTTPException(400, f"Task not completed. Status: {meta.get('status')}")

    # Читаем public результат
    result_path = RESULTS_DIR / f"{task_id}.json"
    if not result_path.exists():
        raise HTTPException(404, "Result not found")

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(500, f"Failed to read result: {e}")

    # Извлекаем public часть если есть
    public = data.get("public", data)

    viral_clips = []
    for i, c in enumerate(public.get("viral_clips", [])):
        viral_clips.append(ViralClipResponse(
            id=c.get("id", f"clip_{i}"),
            start=c.get("start", 0),
            end=c.get("end", 0),
            duration=c.get("duration", c.get("end", 0) - c.get("start", 0)),
            score=c.get("score", 0),
            score_breakdown=c.get("score_breakdown", {}),
            anchor_type=c.get("anchor_type", ""),
            reasons=c.get("reasons", []),
        ))

    chapters = []
    for i, ch in enumerate(public.get("chapters", [])):
        chapters.append(ChapterResponse(
            id=ch.get("id", f"chapter_{i}"),
            start=ch.get("start", 0),
            end=ch.get("end", 0),
            duration=ch.get("duration", ch.get("end", 0) - ch.get("start", 0)),
            title=ch.get("title", ""),
            description=ch.get("description", ""),
        ))

    summary_data = public.get("summary", {})
    summary = AnalysisSummaryResponse(
        total_scenes=summary_data.get("total_scenes", 0),
        total_speech_duration=summary_data.get("total_speech_duration", 0),
        speech_ratio=summary_data.get("speech_ratio", 0),
        mean_motion=summary_data.get("mean_motion", 0),
        mean_loudness=summary_data.get("mean_loudness", 0),
        mean_interest=summary_data.get("mean_interest", 0),
        detected_language=summary_data.get("detected_language"),
    )

    return AnalysisResultResponse(
        task_id=public.get("task_id", task_id),
        video_name=public.get("video_name", "video"),
        duration_seconds=public.get("duration_seconds", 0),
        processing_time_seconds=public.get("processing_time_seconds", 0),
        viral_clips=viral_clips,
        chapters=chapters,
        summary=summary,
        created_at=public.get("created_at"),
    )


@app.get("/results/{task_id}/full", tags=["Results"])
async def get_full_result(task_id: str):
    """Получает полный результат анализа (включая timeline и т.д.)."""
    full_path = RESULTS_DIR / f"{task_id}.full.json"
    if full_path.exists():
        return FileResponse(
            full_path,
            media_type="application/json",
            filename=f"analysis_full_{task_id}.json",
        )

    # Fallback на основной файл
    result_path = RESULTS_DIR / f"{task_id}.json"
    if not result_path.exists():
        raise HTTPException(404, "Result not found")

    return FileResponse(
        result_path,
        media_type="application/json",
        filename=f"analysis_{task_id}.json",
    )


@app.get("/results/{task_id}/download", tags=["Results"])
async def download_result(task_id: str):
    """Скачивает результат в формате JSON."""
    result_path = RESULTS_DIR / f"{task_id}.json"
    if not result_path.exists():
        raise HTTPException(404, "Result file not found")

    return FileResponse(
        result_path,
        media_type="application/json",
        filename=f"analysis_{task_id}.json",
    )


@app.delete("/tasks/{task_id}", tags=["Analysis"])
async def delete_task(task_id: str):
    """Удаляет задачу и связанные файлы."""
    meta = get_task_meta(task_id)

    # Удаляем временные файлы
    if meta:
        video_path = Path(meta.get("video_path", ""))
        if video_path.exists():
            video_path.unlink()

    # Удаляем результаты
    for suffix in [".json", ".full.json"]:
        result_path = RESULTS_DIR / f"{task_id}{suffix}"
        if result_path.exists():
            result_path.unlink()

    # Удаляем метаданные из Redis
    client = get_redis_client()
    if client:
        client.delete(f"task_meta:{task_id}")

    return {"message": "Task deleted"}


@app.get("/tasks", response_model=List[TaskStatus], tags=["Analysis"])
async def list_tasks(limit: int = Query(10, ge=1, le=100), status: Optional[str] = None):
    """Список задач."""
    # Получаем все ключи задач из Redis
    client = get_redis_client()
    if not client:
        return []

    keys = client.keys("task_meta:*")
    tasks = []

    for key in keys[:limit * 2]:  # Берём с запасом для фильтрации
        data = client.get(key)
        if data:
            meta = json.loads(data)
            if status and meta.get("status") != status:
                continue
            tasks.append(TaskStatus(
                task_id=meta.get("task_id", ""),
                status=meta.get("status", "unknown"),
                progress=meta.get("progress", 0.0),
                created_at=meta.get("created_at"),
                completed_at=meta.get("completed_at"),
                result_url=f"/results/{meta.get('task_id')}" if meta.get("status") == "completed" else None,
                error=meta.get("error"),
            ))

    # Сортируем по дате
    tasks.sort(key=lambda x: x.created_at or "", reverse=True)

    return tasks[:limit]


# ============================================================================
# Background Task (non-Celery mode)
# ============================================================================

def run_analysis_task(task_id: str, video_path: str, settings: Dict[str, Any]) -> None:
    """Выполняет анализ видео в фоне (без Celery)."""
    import sys
    import time
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        update_task_progress(task_id, 0.1, "processing")

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

        update_task_progress(task_id, 0.2, "processing", "Подготовка пайплайна")

        # Запускаем пайплайн (с прогрессом по нодам)
        from main import run_dag_pipeline

        context = {"input_path": video_path, "settings": video_settings, "task_id": task_id}

        start_time = time.monotonic()
        def on_progress(p: float, msg: str) -> None:
            update_task_progress(task_id, p, "processing", msg)

        result_context, exec_result = run_dag_pipeline(context, video_settings, progress_callback=on_progress)
        processing_time = time.monotonic() - start_time

        update_task_progress(task_id, 0.95, "processing", "Сохранение результатов")

        # Сохраняем результаты
        result = to_jsonable(result_context)
        result["processing_time_seconds"] = processing_time
        result["task_id"] = task_id

        # Public result
        public_path = RESULTS_DIR / f"{task_id}.json"
        with open(public_path, "w", encoding="utf-8") as f:
            json.dump(result.get("public", result), f, indent=2, ensure_ascii=False)

        # Full result
        full_path = RESULTS_DIR / f"{task_id}.full.json"
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Обновляем статус
        meta = get_task_meta(task_id) or {}
        meta["status"] = "completed"
        meta["progress"] = 1.0
        meta["completed_at"] = datetime.utcnow().isoformat()
        save_task_meta(task_id, meta)
        _publish_task_event(
            task_id,
            {
                "task_id": task_id,
                "status": "completed",
                "progress": 1.0,
                "message": "Готово",
            },
        )

    except Exception as e:
        meta = get_task_meta(task_id) or {}
        meta["status"] = "failed"
        meta["error"] = str(e)
        meta["completed_at"] = datetime.utcnow().isoformat()
        save_task_meta(task_id, meta)
        _publish_task_event(
            task_id,
            {
                "task_id": task_id,
                "status": "failed",
                "progress": meta.get("progress", 0.0),
                "error": str(e),
                "message": "Ошибка",
            },
        )
        print(f"Task {task_id} failed: {e}")

    finally:
        # Очищаем временный файл
        _cleanup_input_if_temp(video_path)


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Действия при запуске."""
    print("🚀 Video Analyzer API started")
    print(f"   Version: 2.0.0")
    print(f"   Temp dir: {TEMP_DIR}")
    print(f"   Results dir: {RESULTS_DIR}")
    print(f"   Celery: {'enabled' if USE_CELERY else 'disabled'}")


@app.on_event("shutdown")
async def shutdown_event():
    """Действия при остановке."""
    print("👋 Video Analyzer API shutting down")
