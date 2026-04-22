"""
Tests for evaluation metrics.

This module tests:
- EpisodeMetrics dataclass
- compute_spl() function
- compute_success_rate() function
- EvaluationReport dataclass
- aggregate_metrics() function
"""

import pytest
from dataclasses import asdict
from typing import List

import sys
sys.path.insert(0, "src")

from evaluation.metrics import (
    EpisodeMetrics,
    EvaluationReport,
    compute_spl,
    compute_success_rate,
    aggregate_metrics,
)


class TestEpisodeMetrics:
    """Test the EpisodeMetrics dataclass."""

    def test_success_metrics(self):
        """Test EpisodeMetrics creation for successful episode."""
        metrics = EpisodeMetrics(
            success=True,
            shortest_path_distance=5.0,
            actual_distance=6.0,
            total_steps=10,
            planning_efficiency=0.8
        )

        assert metrics.success is True
        assert metrics.shortest_path_distance == 5.0
        assert metrics.actual_distance == 6.0
        assert metrics.total_steps == 10
        assert metrics.planning_efficiency == 0.8

    def test_failure_metrics(self):
        """Test EpisodeMetrics creation for failed episode."""
        metrics = EpisodeMetrics(
            success=False,
            shortest_path_distance=5.0,
            actual_distance=15.0,
            total_steps=50,
            planning_efficiency=0.2
        )

        assert metrics.success is False
        assert metrics.shortest_path_distance == 5.0
        assert metrics.actual_distance == 15.0
        assert metrics.total_steps == 50
        assert metrics.planning_efficiency == 0.2

    def test_spl_property_success(self):
        """Test SPL property calculation for successful episode."""
        metrics = EpisodeMetrics(
            success=True,
            shortest_path_distance=5.0,
            actual_distance=6.0,
            total_steps=10,
            planning_efficiency=0.8
        )

        # SPL = success * (l_i / max(p_i, l_i)) = 1 * (5.0 / 6.0) = 0.833...
        expected_spl = 5.0 / 6.0
        assert abs(metrics.spl - expected_spl) < 0.001

    def test_spl_property_failure(self):
        """Test SPL property calculation for failed episode."""
        metrics = EpisodeMetrics(
            success=False,
            shortest_path_distance=5.0,
            actual_distance=15.0,
            total_steps=50,
            planning_efficiency=0.2
        )

        # SPL = 0 * ... = 0 (failure means SPL is 0)
        assert metrics.spl == 0.0

    def test_spl_property_shortest_path_larger(self):
        """Test SPL when shortest path is larger than actual (edge case)."""
        metrics = EpisodeMetrics(
            success=True,
            shortest_path_distance=10.0,
            actual_distance=8.0,
            total_steps=10,
            planning_efficiency=1.0
        )

        # SPL = 1 * (10.0 / max(8.0, 10.0)) = 10.0 / 10.0 = 1.0
        assert metrics.spl == 1.0

    def test_spl_property_zero_distance(self):
        """Test SPL when distances are zero (already at goal)."""
        metrics = EpisodeMetrics(
            success=True,
            shortest_path_distance=0.0,
            actual_distance=0.0,
            total_steps=0,
            planning_efficiency=1.0
        )

        # Edge case: both distances 0, SPL should be 1.0 for success
        assert metrics.spl == 1.0

    def test_metrics_to_dict(self):
        """Test EpisodeMetrics can be converted to dict."""
        metrics = EpisodeMetrics(
            success=True,
            shortest_path_distance=5.0,
            actual_distance=6.0,
            total_steps=10,
            planning_efficiency=0.8
        )

        metrics_dict = asdict(metrics)
        assert metrics_dict["success"] is True
        assert metrics_dict["shortest_path_distance"] == 5.0
        # spl is a property, not a field, so it's not in asdict()
        # Test it separately via the property
        assert metrics.spl == 5.0 / 6.0


class TestComputeSPL:
    """Test the compute_spl() function."""

    def test_spl_single_episode_success(self):
        """Test SPL computation for single successful episode."""
        metrics = EpisodeMetrics(
            success=True,
            shortest_path_distance=5.0,
            actual_distance=6.0,
            total_steps=10,
            planning_efficiency=0.8
        )

        spl = compute_spl(metrics)
        expected = 5.0 / 6.0
        assert abs(spl - expected) < 0.001

    def test_spl_single_episode_failure(self):
        """Test SPL computation for single failed episode."""
        metrics = EpisodeMetrics(
            success=False,
            shortest_path_distance=5.0,
            actual_distance=15.0,
            total_steps=50,
            planning_efficiency=0.2
        )

        spl = compute_spl(metrics)
        assert spl == 0.0

    def test_spl_batch(self):
        """Test SPL computation for batch of episodes."""
        metrics_list = [
            EpisodeMetrics(success=True, shortest_path_distance=5.0, actual_distance=5.0, total_steps=10, planning_efficiency=1.0),  # SPL = 1.0
            EpisodeMetrics(success=True, shortest_path_distance=4.0, actual_distance=8.0, total_steps=15, planning_efficiency=0.8),  # SPL = 0.5
            EpisodeMetrics(success=False, shortest_path_distance=3.0, actual_distance=6.0, total_steps=20, planning_efficiency=0.0),  # SPL = 0.0
        ]

        spl = compute_spl(metrics_list)
        # Average SPL = (1.0 + 0.5 + 0.0) / 3 = 0.5
        expected = (1.0 + 0.5 + 0.0) / 3
        assert abs(spl - expected) < 0.001

    def test_spl_empty_batch(self):
        """Test SPL computation with empty list returns 0."""
        spl = compute_spl([])
        assert spl == 0.0

    def test_spl_batch_all_failures(self):
        """Test SPL computation when all episodes fail."""
        metrics_list = [
            EpisodeMetrics(success=False, shortest_path_distance=5.0, actual_distance=10.0, total_steps=20, planning_efficiency=0.0),
            EpisodeMetrics(success=False, shortest_path_distance=3.0, actual_distance=8.0, total_steps=15, planning_efficiency=0.0),
        ]

        spl = compute_spl(metrics_list)
        assert spl == 0.0

    def test_spl_batch_all_success(self):
        """Test SPL computation when all episodes succeed."""
        metrics_list = [
            EpisodeMetrics(success=True, shortest_path_distance=4.0, actual_distance=4.0, total_steps=8, planning_efficiency=1.0),  # SPL = 1.0
            EpisodeMetrics(success=True, shortest_path_distance=6.0, actual_distance=6.0, total_steps=12, planning_efficiency=1.0),  # SPL = 1.0
        ]

        spl = compute_spl(metrics_list)
        assert spl == 1.0


class TestComputeSuccessRate:
    """Test the compute_success_rate() function."""

    def test_success_rate_all_success(self):
        """Test success rate when all episodes succeed."""
        metrics_list = [
            EpisodeMetrics(success=True, shortest_path_distance=5.0, actual_distance=5.0, total_steps=10, planning_efficiency=1.0),
            EpisodeMetrics(success=True, shortest_path_distance=4.0, actual_distance=4.0, total_steps=8, planning_efficiency=1.0),
            EpisodeMetrics(success=True, shortest_path_distance=3.0, actual_distance=3.0, total_steps=6, planning_efficiency=1.0),
        ]

        rate = compute_success_rate(metrics_list)
        assert rate == 1.0

    def test_success_rate_mixed(self):
        """Test success rate with mixed results."""
        metrics_list = [
            EpisodeMetrics(success=True, shortest_path_distance=5.0, actual_distance=5.0, total_steps=10, planning_efficiency=1.0),
            EpisodeMetrics(success=False, shortest_path_distance=4.0, actual_distance=8.0, total_steps=20, planning_efficiency=0.5),
            EpisodeMetrics(success=True, shortest_path_distance=3.0, actual_distance=4.0, total_steps=8, planning_efficiency=0.8),
        ]

        rate = compute_success_rate(metrics_list)
        assert rate == 2 / 3

    def test_success_rate_all_failure(self):
        """Test success rate when all episodes fail."""
        metrics_list = [
            EpisodeMetrics(success=False, shortest_path_distance=5.0, actual_distance=10.0, total_steps=20, planning_efficiency=0.0),
            EpisodeMetrics(success=False, shortest_path_distance=4.0, actual_distance=8.0, total_steps=15, planning_efficiency=0.0),
        ]

        rate = compute_success_rate(metrics_list)
        assert rate == 0.0

    def test_success_rate_empty_list(self):
        """Test success rate with empty list returns 0."""
        rate = compute_success_rate([])
        assert rate == 0.0


class TestEvaluationReport:
    """Test the EvaluationReport dataclass."""

    def test_report_creation(self):
        """Test EvaluationReport creation."""
        report = EvaluationReport(
            total_episodes=10,
            success_rate=0.7,
            avg_spl=0.65,
            avg_steps=15.5,
            avg_distance=12.3,
            avg_planning_efficiency=0.75,
            vision_only_success_rate=0.5,
            vision_only_avg_spl=0.45
        )

        assert report.total_episodes == 10
        assert report.success_rate == 0.7
        assert report.avg_spl == 0.65
        assert report.avg_steps == 15.5
        assert report.avg_distance == 12.3
        assert report.avg_planning_efficiency == 0.75
        assert report.vision_only_success_rate == 0.5
        assert report.vision_only_avg_spl == 0.45

    def test_report_to_dict(self):
        """Test EvaluationReport.to_dict() method."""
        report = EvaluationReport(
            total_episodes=10,
            success_rate=0.7,
            avg_spl=0.65,
            avg_steps=15.5,
            avg_distance=12.3,
            avg_planning_efficiency=0.75,
            vision_only_success_rate=0.5,
            vision_only_avg_spl=0.45
        )

        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert report_dict["total_episodes"] == 10
        assert report_dict["success_rate"] == 0.7
        assert report_dict["avg_spl"] == 0.65
        assert report_dict["vision_only_success_rate"] == 0.5

    def test_report_str(self):
        """Test EvaluationReport.__str__() method."""
        report = EvaluationReport(
            total_episodes=10,
            success_rate=0.7,
            avg_spl=0.65,
            avg_steps=15.5,
            avg_distance=12.3,
            avg_planning_efficiency=0.75,
            vision_only_success_rate=0.5,
            vision_only_avg_spl=0.45
        )

        report_str = str(report)
        assert "10" in report_str
        assert "70.0%" in report_str or "0.7" in report_str
        assert "SPL" in report_str

    def test_report_defaults(self):
        """Test EvaluationReport with default vision_only fields."""
        report = EvaluationReport(
            total_episodes=5,
            success_rate=0.6,
            avg_spl=0.5,
            avg_steps=12.0,
            avg_distance=10.0,
            avg_planning_efficiency=0.7
        )

        assert report.vision_only_success_rate is None
        assert report.vision_only_avg_spl is None


class TestAggregateMetrics:
    """Test the aggregate_metrics() function."""

    def test_aggregate_single_episode(self):
        """Test aggregation with single episode."""
        metrics = EpisodeMetrics(
            success=True,
            shortest_path_distance=5.0,
            actual_distance=6.0,
            total_steps=10,
            planning_efficiency=0.8
        )

        report = aggregate_metrics([metrics])

        assert report.total_episodes == 1
        assert report.success_rate == 1.0
        assert abs(report.avg_spl - (5.0 / 6.0)) < 0.001
        assert report.avg_steps == 10
        assert report.avg_distance == 6.0
        assert report.avg_planning_efficiency == 0.8

    def test_aggregate_multiple_episodes(self):
        """Test aggregation with multiple episodes."""
        metrics_list = [
            EpisodeMetrics(success=True, shortest_path_distance=5.0, actual_distance=5.0, total_steps=10, planning_efficiency=1.0),
            EpisodeMetrics(success=True, shortest_path_distance=4.0, actual_distance=8.0, total_steps=15, planning_efficiency=0.8),
            EpisodeMetrics(success=False, shortest_path_distance=3.0, actual_distance=6.0, total_steps=20, planning_efficiency=0.0),
        ]

        report = aggregate_metrics(metrics_list)

        assert report.total_episodes == 3
        assert report.success_rate == 2 / 3
        # SPL = (1.0 + 0.5 + 0.0) / 3
        expected_spl = (1.0 + 0.5 + 0.0) / 3
        assert abs(report.avg_spl - expected_spl) < 0.001
        assert report.avg_steps == (10 + 15 + 20) / 3
        assert report.avg_distance == (5.0 + 8.0 + 6.0) / 3
        assert report.avg_planning_efficiency == (1.0 + 0.8 + 0.0) / 3

    def test_aggregate_empty_list(self):
        """Test aggregation with empty list."""
        report = aggregate_metrics([])

        assert report.total_episodes == 0
        assert report.success_rate == 0.0
        assert report.avg_spl == 0.0
        assert report.avg_steps == 0.0
        assert report.avg_distance == 0.0
        assert report.avg_planning_efficiency == 0.0

    def test_aggregate_with_vision_only(self):
        """Test aggregation with vision_only metrics."""
        metrics_list = [
            EpisodeMetrics(success=True, shortest_path_distance=5.0, actual_distance=5.0, total_steps=10, planning_efficiency=1.0),
        ]
        vision_only_metrics = [
            EpisodeMetrics(success=False, shortest_path_distance=5.0, actual_distance=10.0, total_steps=25, planning_efficiency=0.5),
        ]

        report = aggregate_metrics(metrics_list, vision_only_metrics)

        assert report.total_episodes == 1
        assert report.success_rate == 1.0
        assert report.vision_only_success_rate == 0.0
        assert report.vision_only_avg_spl == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
