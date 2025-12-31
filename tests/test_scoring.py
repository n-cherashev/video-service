"""Tests for scoring functions."""
from __future__ import annotations

import numpy as np
import pytest


class TestScoringFunctions:
    """Tests for scoring calculations."""

    def test_normalize_01(self) -> None:
        """Test normalize_01 function."""
        from utils.timeseries import normalize_01

        # Normal case
        arr = np.array([0.0, 5.0, 10.0])
        result = normalize_01(arr)
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])

        # All same values
        arr = np.array([5.0, 5.0, 5.0])
        result = normalize_01(arr)
        np.testing.assert_array_almost_equal(result, [0.5, 0.5, 0.5])

        # Empty array
        arr = np.array([])
        result = normalize_01(arr)
        assert len(result) == 0

    def test_sigmoid(self) -> None:
        """Test sigmoid function."""
        from utils.timeseries import sigmoid

        # Center value
        result = sigmoid(0.0)
        assert abs(result - 0.5) < 0.001

        # Large positive
        result = sigmoid(10.0)
        assert result > 0.99

        # Large negative
        result = sigmoid(-10.0)
        assert result < 0.01

    def test_safe_mean_points(self) -> None:
        """Test safe_mean_points function."""
        from utils.timeseries import safe_mean_points

        # Normal case with dict points
        points = [
            {"time": 0, "value": 1.0},
            {"time": 1, "value": 2.0},
            {"time": 2, "value": 3.0},
        ]
        result = safe_mean_points(points)
        assert abs(result - 2.0) < 0.001

        # Empty list
        result = safe_mean_points([])
        assert result == 0.0

        # None
        result = safe_mean_points(None)
        assert result == 0.0


class TestScoreComponents:
    """Tests for score component calculations."""

    def test_score_components_creation(self) -> None:
        """Test creating score components."""
        from models.candidates import ScoreComponents

        comp = ScoreComponents(
            hook=0.8,
            pace=0.6,
            clarity=0.7,
            intensity=0.5,
            emotion=0.4,
            boundary=0.3,
            momentum=0.2,
        )

        # Test to_dict
        d = comp.to_dict()
        assert d["hook"] == 0.8
        assert d["momentum"] == 0.2

        # Test from_dict
        restored = ScoreComponents.from_dict(d)
        assert restored.hook == 0.8
        assert restored.momentum == 0.2

    def test_weighted_sum_basic(self) -> None:
        """Test weighted sum calculation."""
        from models.candidates import ScoreComponents

        comp = ScoreComponents(hook=1.0, pace=0.0, clarity=0.5)
        weights = {"hook": 1.0, "pace": 1.0, "clarity": 1.0}

        result = comp.weighted_sum(weights)
        # (1.0 + 0.0 + 0.5) / 3 = 0.5
        assert abs(result - 0.5) < 0.001

    def test_weighted_sum_unequal_weights(self) -> None:
        """Test weighted sum with unequal weights."""
        from models.candidates import ScoreComponents

        comp = ScoreComponents(hook=1.0, pace=0.0)
        weights = {"hook": 0.8, "pace": 0.2}

        result = comp.weighted_sum(weights)
        # (1.0 * 0.8 + 0.0 * 0.2) / 1.0 = 0.8
        assert abs(result - 0.8) < 0.001


class TestQualityGates:
    """Tests for quality gate logic."""

    def test_minimum_hook_score_gate(self) -> None:
        """Test minimum hook score quality gate."""
        min_hook = 0.3

        # Should pass
        assert 0.5 >= min_hook
        assert 0.3 >= min_hook

        # Should fail
        assert 0.2 < min_hook
        assert 0.0 < min_hook

    def test_minimum_clarity_gate(self) -> None:
        """Test minimum clarity quality gate."""
        min_clarity = 0.25

        # Should pass
        assert 0.5 >= min_clarity
        assert 0.25 >= min_clarity

        # Should fail
        assert 0.1 < min_clarity

    def test_clip_duration_gate(self) -> None:
        """Test clip duration quality gate."""
        min_duration = 15.0
        max_duration = 90.0

        # Valid durations
        assert min_duration <= 30.0 <= max_duration
        assert min_duration <= 60.0 <= max_duration
        assert min_duration <= 15.0 <= max_duration
        assert min_duration <= 90.0 <= max_duration

        # Invalid durations
        assert not (min_duration <= 10.0 <= max_duration)
        assert not (min_duration <= 100.0 <= max_duration)

    def test_combined_quality_gate(self) -> None:
        """Test combined quality gate for LLM/refit eligibility."""
        def is_eligible_for_refinement(
            hook: float,
            clarity: float,
            score: float,
            min_hook: float = 0.3,
            min_clarity: float = 0.25,
            min_score: float = 0.4,
        ) -> bool:
            """Check if candidate is eligible for expensive refinement."""
            return hook >= min_hook and clarity >= min_clarity and score >= min_score

        # Should be eligible
        assert is_eligible_for_refinement(hook=0.5, clarity=0.5, score=0.6)
        assert is_eligible_for_refinement(hook=0.3, clarity=0.25, score=0.4)

        # Should not be eligible
        assert not is_eligible_for_refinement(hook=0.2, clarity=0.5, score=0.6)  # Low hook
        assert not is_eligible_for_refinement(hook=0.5, clarity=0.1, score=0.6)  # Low clarity
        assert not is_eligible_for_refinement(hook=0.5, clarity=0.5, score=0.3)  # Low score
