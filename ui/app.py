"""
Video Analyzer - Streamlit Web UI (v2 - simplified)

Минимальная версия с логированием для диагностики.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# ============================================================================
# LOGGING - в самом начале, до любых тяжёлых импортов
# ============================================================================
print("[UI] ========== STARTING UI ==========", flush=True)
print(f"[UI] Python: {sys.version}", flush=True)
print(f"[UI] CWD: {os.getcwd()}", flush=True)
print(f"[UI] __file__: {__file__}", flush=True)

# ============================================================================
# Streamlit import
# ============================================================================
print("[UI] Importing streamlit...", flush=True)
try:
    import streamlit as st
    print("[UI] ✓ streamlit imported", flush=True)
except Exception as e:
    print(f"[UI] ✗ Failed to import streamlit: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# HTTP client import
# ============================================================================
print("[UI] Importing httpx...", flush=True)
try:
    import httpx
    print("[UI] ✓ httpx imported", flush=True)
except Exception as e:
    print(f"[UI] ✗ Failed to import httpx: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Page config - MUST be first st.* call
# ============================================================================
print("[UI] Setting page config...", flush=True)
try:
    st.set_page_config(
        page_title="Video Analyzer",
        page_icon="🎬",
        layout="wide",
    )
    print("[UI] ✓ page config set", flush=True)
except Exception as e:
    print(f"[UI] ✗ Failed to set page config: {e}", flush=True)
    traceback.print_exc()


# ============================================================================
# Helper functions
# ============================================================================
def get_api_url() -> str:
    """Возвращает URL API из окружения."""
    return (os.environ.get("API_URL") or "").rstrip("/")


def get_videos_dir() -> Path:
    """Возвращает путь к папке с видео."""
    return Path(os.environ.get("VIDEO_INPUT_DIR", "/app/videos"))


def api_analyze_path(
    api_url: str,
    video_path: str,
    max_clips: int = 8,
    min_duration: float = 30.0,
    max_duration: float = 60.0,
    enable_llm: bool = False,
) -> str:
    """Отправляет запрос на анализ видео по пути."""
    print(f"[UI] api_analyze_path: {video_path}", flush=True)

    params = {
        "max_clips": max_clips,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "enable_llm": enable_llm,
    }
    payload = {"path": video_path}

    timeout = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0)
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{api_url}/analyze-path", params=params, json=payload)
        r.raise_for_status()
        data = r.json()
        task_id = data.get("task_id")
        if not task_id:
            raise RuntimeError(f"API did not return task_id: {data}")
        print(f"[UI] ✓ task_id: {task_id}", flush=True)
        return str(task_id)


def api_get_task_status(api_url: str, task_id: str) -> dict[str, Any]:
    """Получает статус задачи."""
    timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)
    with httpx.Client(timeout=timeout) as client:
        r = client.get(f"{api_url}/tasks/{task_id}")
        r.raise_for_status()
        return r.json()


def api_get_result(api_url: str, task_id: str) -> dict[str, Any]:
    """Получает результат анализа."""
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0)
    with httpx.Client(timeout=timeout) as client:
        r = client.get(f"{api_url}/results/{task_id}")
        r.raise_for_status()
        return r.json()


def format_time(seconds: float) -> str:
    """Форматирует секунды в MM:SS или HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


# ============================================================================
# MAIN UI
# ============================================================================
print("[UI] Starting main UI render...", flush=True)

try:
    # Header
    st.title("🎬 Video Analyzer")
    st.markdown("**Автоматический поиск viral-моментов в видео**")
    print("[UI] ✓ header rendered", flush=True)

    # API URL check
    api_url = get_api_url()
    if api_url:
        st.success(f"✅ API: {api_url}")
    else:
        st.error("❌ API_URL не задан. В Docker он должен быть http://api:8000")
        st.stop()

    print(f"[UI] API URL: {api_url}", flush=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Настройки")
        max_clips = st.slider("Макс. клипов", 1, 20, 8)
        min_duration = st.slider("Мин. длина клипа (сек)", 10, 120, 30)
        max_duration = st.slider("Макс. длина клипа (сек)", 30, 300, 60)
        enable_llm = st.checkbox("Использовать LLM", value=False)

    print("[UI] ✓ sidebar rendered", flush=True)

    # Tabs
    tab1, tab2 = st.tabs(["📤 Анализ", "📊 Результаты"])

    with tab1:
        st.subheader("Выберите видео для анализа")

        videos_dir = get_videos_dir()
        st.caption(f"Папка видео: `{videos_dir}`")

        if not videos_dir.exists():
            st.warning(f"Папка `{videos_dir}` не найдена. Положи видео в `video-service/videos/`")
            st.stop()

        # Список видео
        video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
        video_files = sorted(
            [f for f in videos_dir.iterdir() if f.is_file() and f.suffix.lower() in video_extensions],
            key=lambda p: p.name.lower(),
        )

        if not video_files:
            st.info("В папке нет видеофайлов.")
            st.stop()

        # Выбор файла
        selected_file = st.selectbox(
            "Выберите файл",
            options=video_files,
            format_func=lambda p: f"{p.name} ({p.stat().st_size / (1024**3):.2f} GB)",
        )

        if selected_file:
            st.info(f"📁 Выбран: **{selected_file.name}**")

            if st.button("🚀 Запустить анализ", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.info("⏳ Отправляю запрос на анализ...")
                    print(f"[UI] Starting analysis: {selected_file}", flush=True)

                    task_id = api_analyze_path(
                        api_url=api_url,
                        video_path=str(selected_file),
                        max_clips=max_clips,
                        min_duration=min_duration,
                        max_duration=max_duration,
                        enable_llm=enable_llm,
                    )
                    st.session_state["task_id"] = task_id
                    st.session_state["video_name"] = selected_file.name

                    # Polling loop
                    import time
                    for i in range(3600):  # max 1 hour
                        status = api_get_task_status(api_url, task_id)
                        state = status.get("status", "unknown")
                        progress = float(status.get("progress", 0) or 0)

                        progress_bar.progress(min(int(progress * 100), 100))
                        status_text.info(f"Статус: {state} | Прогресс: {progress * 100:.0f}%")

                        if state in ("completed", "SUCCESS"):
                            result = api_get_result(api_url, task_id)
                            st.session_state["analysis_result"] = result
                            progress_bar.progress(100)
                            status_text.success("✅ Анализ завершён!")
                            print(f"[UI] ✓ Analysis completed: {task_id}", flush=True)
                            break

                        if state in ("failed", "FAILURE"):
                            err = status.get("error", "unknown error")
                            status_text.error(f"❌ Ошибка: {err}")
                            print(f"[UI] ✗ Analysis failed: {err}", flush=True)
                            break

                        time.sleep(1.0)

                except Exception as e:
                    status_text.error(f"❌ Ошибка: {e}")
                    print(f"[UI] ✗ Exception: {e}", flush=True)
                    traceback.print_exc()

    with tab2:
        if "analysis_result" not in st.session_state:
            st.info("👆 Сначала запустите анализ видео")
        else:
            result = st.session_state["analysis_result"]
            video_name = st.session_state.get("video_name", "video")

            st.subheader(f"📊 Результаты: {video_name}")

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                duration = result.get("duration_seconds", 0)
                st.metric("⏱️ Длительность", format_time(duration))
            with col2:
                clips = result.get("viral_clips", [])
                st.metric("🎬 Клипов", len(clips))
            with col3:
                proc_time = result.get("processing_time_seconds", 0)
                st.metric("⚡ Время анализа", f"{proc_time:.1f}s")

            st.divider()

            # Clips
            if clips:
                st.subheader("🎬 Viral клипы")
                for i, clip in enumerate(clips):
                    start = clip.get("start", 0)
                    end = clip.get("end", 0)
                    score = clip.get("score", 0)

                    with st.expander(f"Клип #{i+1}: {format_time(start)} - {format_time(end)} (score: {score:.2f})"):
                        st.json(clip)
            else:
                st.warning("Клипы не найдены")

            st.divider()

            # Download JSON
            json_str = json.dumps(result, indent=2, ensure_ascii=False)
            st.download_button(
                "📥 Скачать JSON",
                data=json_str,
                file_name=f"{Path(video_name).stem}_analysis.json",
                mime="application/json",
            )

    print("[UI] ✓ Main UI render complete", flush=True)

except Exception as e:
    print(f"[UI] ✗ FATAL ERROR: {e}", flush=True)
    traceback.print_exc()
    st.error(f"Критическая ошибка UI: {e}")
    st.code(traceback.format_exc())

print("[UI] ========== UI SCRIPT END ==========", flush=True)
