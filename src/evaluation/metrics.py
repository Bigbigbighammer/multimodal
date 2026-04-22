"""
Evaluation metrics for vision-language embodied agent.

This module provides metrics for evaluating navigation and interaction tasks:
- SPL (Success weighted by Path Length)
- Success Rate
- Planning Efficiency
- EvaluationReport for aggregate metrics

SPL Formula:
    SPL = (1/N) * sum(S_i * l_i / max(p_i, l_i))
    - S_i = 1 if success, 0 if failure
    - l_i = shortest path distance
    - p_i = actual path distance
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union


@dataclass
class EpisodeMetrics:
    """
    Metrics for a single episode.

    Attributes:
        success: Whether the episode was successful
        shortest_path_distance: Distance of optimal path (meters)
        actual_distance: Actual distance traveled (meters)
        total_steps: Number of actions taken
        planning_efficiency: Ratio of completed subgoals to total subgoals
    """
    success: bool
    shortest_path_distance: float
    actual_distance: float
    total_steps: int
    planning_efficiency: float

    @property
    def spl(self) -> float:
        """
        Compute SPL for this episode.

        SPL = success * (l_i / max(p_i, l_i))

        Returns:
            SPL value between 0 and 1
        """
        if not self.success:
            return 0.0

        # Handle edge case where both distances are 0
        if self.shortest_path_distance == 0 and self.actual_distance == 0:
            return 1.0

        max_distance = max(self.actual_distance, self.shortest_path_distance)
        if max_distance == 0:
            return 1.0

        return self.shortest_path_distance / max_distance


@dataclass
class EvaluationReport:
    """
    Aggregated evaluation report across multiple episodes.

    Attributes:
        total_episodes: Total number of episodes evaluated
        success_rate: Fraction of successful episodes
        avg_spl: Average SPL across all episodes
        avg_steps: Average number of steps per episode
        avg_distance: Average distance traveled per episode
        avg_planning_efficiency: Average planning efficiency
        vision_only_success_rate: Success rate for vision-only baseline
        vision_only_avg_spl: Average SPL for vision-only baseline
    """
    total_episodes: int
    success_rate: float
    avg_spl: float
    avg_steps: float
    avg_distance: float
    avg_planning_efficiency: float
    vision_only_success_rate: Optional[float] = None
    vision_only_avg_spl: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert report to dictionary format.

        Returns:
            Dictionary containing all metrics
        """
        return {
            "total_episodes": self.total_episodes,
            "success_rate": self.success_rate,
            "avg_spl": self.avg_spl,
            "avg_steps": self.avg_steps,
            "avg_distance": self.avg_distance,
            "avg_planning_efficiency": self.avg_planning_efficiency,
            "vision_only_success_rate": self.vision_only_success_rate,
            "vision_only_avg_spl": self.vision_only_avg_spl,
        }

    def __str__(self) -> str:
        """
        Human-readable string representation.

        Returns:
            Formatted string with key metrics
        """
        lines = [
            "=== Evaluation Report ===",
            f"Total Episodes: {self.total_episodes}",
            f"Success Rate: {self.success_rate * 100:.1f}%",
            f"Average SPL: {self.avg_spl:.3f}",
            f"Average Steps: {self.avg_steps:.1f}",
            f"Average Distance: {self.avg_distance:.2f}m",
            f"Average Planning Efficiency: {self.avg_planning_efficiency:.2f}",
        ]

        if self.vision_only_success_rate is not None:
            lines.extend([
                "",
                "--- Vision-Only Baseline ---",
                f"Success Rate: {self.vision_only_success_rate * 100:.1f}%",
                f"Average SPL: {self.vision_only_avg_spl:.3f}" if self.vision_only_avg_spl is not None else "Average SPL: N/A",
            ])

        return "\n".join(lines)


def compute_spl(metrics: Union[EpisodeMetrics, List[EpisodeMetrics]]) -> float:
    """
    Compute SPL (Success weighted by Path Length).

    For a single episode:
        SPL = success * (l_i / max(p_i, l_i))

    For multiple episodes:
        SPL = (1/N) * sum(S_i * l_i / max(p_i, l_i))

    Args:
        metrics: Single EpisodeMetrics or list of EpisodeMetrics

    Returns:
        SPL value between 0 and 1
    """
    if isinstance(metrics, EpisodeMetrics):
        return metrics.spl

    if not metrics:
        return 0.0

    total_spl = sum(m.spl for m in metrics)
    return total_spl / len(metrics)


def compute_success_rate(metrics: List[EpisodeMetrics]) -> float:
    """
    Compute success rate across episodes.

    Args:
        metrics: List of EpisodeMetrics

    Returns:
        Success rate between 0 and 1
    """
    if not metrics:
        return 0.0

    successful = sum(1 for m in metrics if m.success)
    return successful / len(metrics)


def aggregate_metrics(
    metrics: List[EpisodeMetrics],
    vision_only_metrics: Optional[List[EpisodeMetrics]] = None
) -> EvaluationReport:
    """
    Aggregate metrics from multiple episodes into a report.

    Args:
        metrics: List of EpisodeMetrics from main evaluation
        vision_only_metrics: Optional list of EpisodeMetrics from vision-only baseline

    Returns:
        EvaluationReport with aggregated statistics
    """
    if not metrics:
        report = EvaluationReport(
            total_episodes=0,
            success_rate=0.0,
            avg_spl=0.0,
            avg_steps=0.0,
            avg_distance=0.0,
            avg_planning_efficiency=0.0
        )
        if vision_only_metrics:
            report.vision_only_success_rate = compute_success_rate(vision_only_metrics)
            report.vision_only_avg_spl = compute_spl(vision_only_metrics)
        return report

    n = len(metrics)
    success_rate = compute_success_rate(metrics)
    avg_spl = compute_spl(metrics)
    avg_steps = sum(m.total_steps for m in metrics) / n
    avg_distance = sum(m.actual_distance for m in metrics) / n
    avg_planning_efficiency = sum(m.planning_efficiency for m in metrics) / n

    report = EvaluationReport(
        total_episodes=n,
        success_rate=success_rate,
        avg_spl=avg_spl,
        avg_steps=avg_steps,
        avg_distance=avg_distance,
        avg_planning_efficiency=avg_planning_efficiency
    )

    if vision_only_metrics:
        report.vision_only_success_rate = compute_success_rate(vision_only_metrics)
        report.vision_only_avg_spl = compute_spl(vision_only_metrics)

    return report
