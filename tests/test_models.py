"""Tests for models package."""
from __future__ import annotations

import pytest
from datetime import datetime

from models.keys import Key, MERGEABLE_KEYS, ARTIFACT_KEYS
from models.artifacts import ArtifactKind, ArtifactRef, VideoMeta
from models.contracts import HandlerContract, NodePatch, StateView
from models.pipeline_state import PipelineStateV2, PipelineMetrics
from models.results import (
    ScoreBreakdown,
    ViralClipResult,
    ChapterResult,
    AnalysisSummary,
    AnalysisResultPublic,
)
from models.candidates import (
    AnchorType,
    Anchor,
    CandidateWindow,
    ScoreComponents,
    ScoredClipV2,
)


class TestKeys:
    """Tests for Key enum and key sets."""

    def test_key_enum_values(self) -> None:
        """Test that Key enum has expected values."""
        assert Key.INPUT_PATH.value == "input_path"
        assert Key.VIDEO_PATH.value == "video_path"
        assert Key.AUDIO_PATH.value == "audio_path"
        assert Key.TIMELINE.value == "timeline"

    def test_mergeable_keys(self) -> None:
        """Test MERGEABLE_KEYS contains expected keys."""
        assert Key.COMPLETED_STAGES in MERGEABLE_KEYS
        assert Key.LAYER_TIMINGS in MERGEABLE_KEYS
        assert Key.WARNINGS in MERGEABLE_KEYS
        # Non-mergeable keys
        assert Key.VIDEO_PATH not in MERGEABLE_KEYS

    def test_artifact_keys(self) -> None:
        """Test ARTIFACT_KEYS contains expected keys."""
        assert Key.MOTION_SERIES in ARTIFACT_KEYS
        assert Key.AUDIO_SERIES in ARTIFACT_KEYS
        assert Key.TIMELINE in ARTIFACT_KEYS


class TestArtifacts:
    """Tests for artifact models."""

    def test_artifact_ref_creation(self) -> None:
        """Test ArtifactRef creation and serialization."""
        ref = ArtifactRef(
            kind=ArtifactKind.VIDEO,
            path="/path/to/video.mp4",
            fingerprint="abc123",
        )
        assert ref.kind == ArtifactKind.VIDEO
        assert ref.path == "/path/to/video.mp4"
        assert ref.fingerprint == "abc123"

        d = ref.to_dict()
        assert d["kind"] == "video"
        assert d["path"] == "/path/to/video.mp4"

    def test_video_meta(self) -> None:
        """Test VideoMeta creation."""
        meta = VideoMeta(
            duration_seconds=120.5,
            fps=30.0,
            frame_count=3615,
            width=1920,
            height=1080,
            has_audio=True,
        )
        assert meta.duration_seconds == 120.5
        assert meta.fps == 30.0
        assert meta.frame_count == 3615


class TestContracts:
    """Tests for handler contracts."""

    def test_handler_contract_validation(self) -> None:
        """Test HandlerContract validation."""
        contract = HandlerContract(
            name="TestHandler",
            requires=frozenset({Key.VIDEO_PATH}),
            provides=frozenset({Key.AUDIO_PATH}),
        )

        # Valid - has required key
        errors = contract.validate_inputs({Key.VIDEO_PATH, Key.FPS})
        assert len(errors) == 0

        # Invalid - missing required key
        errors = contract.validate_inputs({Key.FPS})
        assert len(errors) == 1
        assert "missing required key" in errors[0].lower()

    def test_node_patch_creation(self) -> None:
        """Test NodePatch creation."""
        patch = NodePatch(
            handler_name="TestHandler",
            provides={Key.AUDIO_PATH: "/path/to/audio.wav"},
            execution_time_seconds=1.5,
        )
        assert patch.handler_name == "TestHandler"
        assert patch.success
        assert patch.error is None

    def test_node_patch_with_error(self) -> None:
        """Test NodePatch with error."""
        patch = NodePatch(
            handler_name="FailingHandler",
            error="Something went wrong",
        )
        assert not patch.success
        assert patch.error == "Something went wrong"


class TestPipelineState:
    """Tests for PipelineStateV2."""

    def test_state_creation(self) -> None:
        """Test PipelineStateV2 creation."""
        state = PipelineStateV2()
        assert state.status == "pending"
        assert len(state.completed_stages) == 0
        assert state.error is None

    def test_state_set_get(self) -> None:
        """Test setting and getting values."""
        state = PipelineStateV2()
        state.set(Key.VIDEO_PATH, "/path/to/video.mp4")
        assert state.get(Key.VIDEO_PATH) == "/path/to/video.mp4"
        assert state.has(Key.VIDEO_PATH)
        assert not state.has(Key.AUDIO_PATH)

    def test_state_apply_patch(self) -> None:
        """Test applying NodePatch."""
        state = PipelineStateV2()

        patch = NodePatch(
            handler_name="TestHandler",
            provides={Key.VIDEO_PATH: "/path/to/video.mp4"},
            execution_time_seconds=1.0,
        )

        errors = state.apply_patch(patch)
        assert len(errors) == 0
        assert state.get(Key.VIDEO_PATH) == "/path/to/video.mp4"
        assert "TestHandler" in state.completed_stages

    def test_state_apply_patch_conflict(self) -> None:
        """Test patch conflict detection."""
        state = PipelineStateV2()
        state.set(Key.VIDEO_PATH, "/original/path.mp4")

        patch = NodePatch(
            handler_name="ConflictHandler",
            provides={Key.VIDEO_PATH: "/new/path.mp4"},
        )

        errors = state.apply_patch(patch)
        # Should report conflict for non-mergeable key
        assert len(errors) == 1
        assert "conflict" in errors[0].lower()

    def test_state_merge_warnings(self) -> None:
        """Test merging warnings (mergeable key)."""
        state = PipelineStateV2()
        state.set(Key.WARNINGS, ["warning1"])

        patch = NodePatch(
            handler_name="Handler2",
            provides={Key.WARNINGS: ["warning2"]},
        )

        errors = state.apply_patch(patch)
        assert len(errors) == 0
        # Warnings should be merged
        warnings = state.get(Key.WARNINGS)
        assert "warning1" in warnings
        assert "warning2" in warnings


class TestResults:
    """Tests for result models."""

    def test_score_breakdown(self) -> None:
        """Test ScoreBreakdown."""
        breakdown = ScoreBreakdown(
            hook=0.8,
            pace=0.6,
            clarity=0.7,
            intensity=0.5,
            emotion=0.4,
            boundary=0.3,
            momentum=0.2,
        )
        d = breakdown.to_dict()
        assert d["hook"] == 0.8
        assert d["clarity"] == 0.7

    def test_viral_clip_result(self) -> None:
        """Test ViralClipResult."""
        clip = ViralClipResult(
            id="clip_1",
            start=10.0,
            end=40.0,
            score=0.85,
            score_breakdown=ScoreBreakdown(hook=0.9),
            anchor_type="interest_peak",
            reasons=["High hook score", "Good clarity"],
        )
        assert clip.duration == 30.0
        d = clip.to_dict()
        assert d["id"] == "clip_1"
        assert d["duration"] == 30.0

    def test_analysis_result_public(self) -> None:
        """Test AnalysisResultPublic."""
        result = AnalysisResultPublic(
            task_id="test-123",
            video_name="test.mp4",
            duration_seconds=120.0,
            processing_time_seconds=30.0,
            viral_clips=[],
            chapters=[],
            summary=AnalysisSummary(),
        )
        d = result.to_dict()
        assert d["task_id"] == "test-123"
        assert d["duration_seconds"] == 120.0


class TestCandidates:
    """Tests for candidate models."""

    def test_anchor(self) -> None:
        """Test Anchor creation."""
        anchor = Anchor(
            time=30.5,
            type=AnchorType.INTEREST_PEAK,
            value=0.85,
        )
        assert anchor.time == 30.5
        assert anchor.type == AnchorType.INTEREST_PEAK

    def test_candidate_window(self) -> None:
        """Test CandidateWindow."""
        candidate = CandidateWindow(
            id="c_1",
            start=10.0,
            end=40.0,
            anchor_time=25.0,
            anchor_type=AnchorType.MOTION_PEAK,
        )
        assert candidate.duration == 30.0

    def test_score_components_weighted_sum(self) -> None:
        """Test ScoreComponents weighted sum."""
        components = ScoreComponents(
            hook=1.0,
            pace=0.5,
            clarity=0.5,
        )
        weights = {"hook": 0.5, "pace": 0.25, "clarity": 0.25}
        result = components.weighted_sum(weights)
        # (1.0 * 0.5 + 0.5 * 0.25 + 0.5 * 0.25) / (0.5 + 0.25 + 0.25) = 0.75
        assert abs(result - 0.75) < 0.001

    def test_scored_clip_v2(self) -> None:
        """Test ScoredClipV2."""
        clip = ScoredClipV2(
            id="sc_1",
            start=10.0,
            end=50.0,
            score=0.78,
            components=ScoreComponents(hook=0.8, clarity=0.7),
            anchor_type=AnchorType.AUDIO_PEAK,
            reasons=["High energy"],
        )
        assert clip.duration == 40.0
        assert clip.get_effective_score() == 0.78

        # With LLM score
        clip.final_score = 0.85
        assert clip.get_effective_score() == 0.85
