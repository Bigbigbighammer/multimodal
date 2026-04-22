"""
Evaluation module for the Vision-Language Embodied Agent.

This module provides metrics, visualization, and evaluation runners
for assessing agent performance on navigation and interaction tasks.

Key components:
- EpisodeMetrics: Metrics for a single episode
- EvaluationReport: Aggregated metrics across episodes
- EvaluationRunner: Orchestrates episode execution and metrics collection
- TrajectoryVisualizer: Creates trajectory and summary plots
"""

from src.evaluation.metrics import (
    EpisodeMetrics,
    EvaluationReport,
    compute_spl,
    compute_success_rate,
    aggregate_metrics,
)
from src.evaluation.runner import (
    EpisodeConfig,
    EpisodeResult,
    EvaluationRunner,
)
from src.evaluation.visualizer import (
    TrajectoryVisualizer,
    TrajectoryPoint,
)

__all__ = [
    # Metrics
    "EpisodeMetrics",
    "EvaluationReport",
    "compute_spl",
    "compute_success_rate",
    "aggregate_metrics",
    # Runner
    "EpisodeConfig",
    "EpisodeResult",
    "EvaluationRunner",
    # Visualizer
    "TrajectoryVisualizer",
    "TrajectoryPoint",
]
