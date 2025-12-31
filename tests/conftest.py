"""Pytest configuration and fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Добавляем корень проекта в path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_context() -> dict:
    """Sample context for testing handlers."""
    return {
        "input_path": "/path/to/video.mp4",
        "video_path": "/path/to/video.mp4",
        "duration_seconds": 120.0,
        "fps": 30.0,
        "frame_count": 3600,
    }


@pytest.fixture
def sample_timeline() -> list:
    """Sample timeline data."""
    import numpy as np

    points = []
    for i in range(100):
        t = float(i)
        points.append({
            "time": t,
            "interest": float(np.random.random()),
            "motion": float(np.random.random()),
            "audio_loudness": float(np.random.random()),
            "clarity": 0.5 + float(np.random.random()) * 0.3,
            "sentiment": float(np.random.random()) * 2 - 1,
            "has_laughter": np.random.random() > 0.9,
            "is_scene_boundary": np.random.random() > 0.95,
        })
    return points


@pytest.fixture
def sample_audio_features() -> dict:
    """Sample audio features data."""
    import numpy as np

    n_points = 500
    times = [float(i * 0.1) for i in range(n_points)]

    return {
        "loudness": [{"time": t, "value": float(np.random.random())} for t in times],
        "energy": [{"time": t, "value": float(np.random.random())} for t in times],
        "speech_probability": [{"time": t, "value": float(np.random.random())} for t in times],
    }


@pytest.fixture
def sample_transcript_segments() -> list:
    """Sample transcript segments."""
    return [
        {"start": 0.0, "end": 5.0, "text": "Hello, welcome to this video."},
        {"start": 5.5, "end": 10.0, "text": "Today we're going to talk about something interesting."},
        {"start": 12.0, "end": 18.0, "text": "Let's get started with the main topic."},
        {"start": 20.0, "end": 28.0, "text": "This is a really important point to remember."},
        {"start": 30.0, "end": 35.0, "text": "And that concludes our discussion."},
    ]


@pytest.fixture
def sample_scored_clip():
    """Sample scored clip for testing."""
    from models.candidates import ScoredClipV2, ScoreComponents, AnchorType

    return ScoredClipV2(
        id="test_clip_1",
        start=10.0,
        end=40.0,
        score=0.75,
        components=ScoreComponents(
            hook=0.8,
            pace=0.6,
            clarity=0.7,
            intensity=0.5,
            emotion=0.4,
            boundary=0.6,
            momentum=0.3,
        ),
        anchor_type=AnchorType.INTEREST_PEAK,
        reasons=["High hook score", "Good clarity"],
    )


@pytest.fixture
def quality_gate_config():
    """Sample quality gate config."""
    from core.quality_gates import QualityGateConfig

    return QualityGateConfig(
        min_hook_for_llm=0.3,
        min_clarity_for_llm=0.25,
        min_score_for_llm=0.4,
        min_score_for_refit=0.35,
        max_candidates_for_llm=15,
    )
