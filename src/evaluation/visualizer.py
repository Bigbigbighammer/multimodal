"""
Trajectory visualization for the Vision-Language Embodied Agent.

This module provides visualization tools for navigation trajectories,
including trajectory plots and summary statistics visualization.

Key components:
- TrajectoryVisualizer: Class for creating trajectory and summary plots
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    """A single point in the navigation trajectory."""
    x: float
    z: float
    step: int


class TrajectoryVisualizer:
    """
    Visualizer for agent navigation trajectories.

    Creates plots for individual episode trajectories and summary
    statistics across multiple episodes.

    Example:
        >>> visualizer = TrajectoryVisualizer("results/trajectories")
        >>> positions = [{"x": 0, "z": 0}, {"x": 1, "z": 0}, {"x": 1, "z": 1}]
        >>> visualizer.plot_trajectory(positions, {"x": 2, "z": 2}, "ep_001", True)
    """

    def __init__(self, output_dir: str):
        """
        Initialize the TrajectoryVisualizer.

        Args:
            output_dir: Directory to save visualization outputs.
        """
        self._output_dir = output_dir
        self._matplotlib_available = False

        # Check for matplotlib availability
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            self._plt = plt
            self._matplotlib_available = True
            logger.info("Matplotlib initialized for trajectory visualization")
        except ImportError:
            logger.warning(
                "matplotlib not installed. Visualization will be skipped. "
                "Install with: pip install matplotlib"
            )
            self._plt = None

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def plot_trajectory(
        self,
        positions: List[Dict[str, float]],
        target_position: Optional[Dict[str, float]],
        episode_id: str,
        success: bool
    ) -> Optional[str]:
        """
        Plot a single navigation trajectory.

        Args:
            positions: List of position dicts with 'x' and 'z' keys.
            target_position: Target position dict, or None.
            episode_id: Episode identifier for filename.
            success: Whether the episode was successful.

        Returns:
            Path to saved plot file, or None if visualization unavailable.
        """
        if not self._matplotlib_available:
            logger.debug("Skipping trajectory plot: matplotlib not available")
            return None

        if not positions:
            logger.warning(f"No positions to plot for episode {episode_id}")
            return None

        try:
            fig, ax = self._plt.subplots(figsize=(10, 8))

            # Extract coordinates
            xs = [p["x"] for p in positions]
            zs = [p["z"] for p in positions]

            # Plot trajectory
            color = "green" if success else "red"
            ax.plot(xs, zs, "o-", color=color, linewidth=2, markersize=4,
                    label="Agent trajectory", alpha=0.7)

            # Mark start position
            ax.scatter([xs[0]], [zs[0]], color="blue", s=150, marker="^",
                      label="Start", zorder=5)

            # Mark end position
            ax.scatter([xs[-1]], [zs[-1]], color=color, s=150, marker="s",
                      label="End", zorder=5)

            # Mark target if provided
            if target_position:
                ax.scatter([target_position["x"]], [target_position["z"]],
                          color="gold", s=200, marker="*", label="Target",
                          edgecolors="black", linewidths=1, zorder=5)

            # Configure plot
            ax.set_xlabel("X (meters)", fontsize=12)
            ax.set_ylabel("Z (meters)", fontsize=12)
            ax.set_title(
                f"Episode {episode_id} - {'Success' if success else 'Failure'}",
                fontsize=14
            )
            ax.legend(loc="best", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", adjustable="box")

            # Add padding
            x_range = max(xs) - min(xs) if len(xs) > 1 else 1
            z_range = max(zs) - min(zs) if len(zs) > 1 else 1
            padding = max(x_range, z_range) * 0.1 + 0.5
            ax.set_xlim(min(xs) - padding, max(xs) + padding)
            ax.set_ylim(min(zs) - padding, max(zs) + padding)

            # Save plot
            filename = f"trajectory_{episode_id}.png"
            filepath = os.path.join(self._output_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            self._plt.close(fig)

            logger.debug(f"Saved trajectory plot: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to create trajectory plot: {e}")
            return None

    def create_summary_plot(
        self,
        success_rates: List[float],
        spl_values: List[float],
        labels: List[str],
        title: str = "Evaluation Summary"
    ) -> Optional[str]:
        """
        Create a summary bar plot of evaluation metrics.

        Args:
            success_rates: List of success rates (0.0 to 1.0).
            spl_values: List of SPL values (0.0 to 1.0).
            labels: Labels for each evaluation run.
            title: Plot title.

        Returns:
            Path to saved plot file, or None if visualization unavailable.
        """
        if not self._matplotlib_available:
            logger.debug("Skipping summary plot: matplotlib not available")
            return None

        if not success_rates or not spl_values or not labels:
            logger.warning("Missing data for summary plot")
            return None

        if len(success_rates) != len(spl_values) or len(success_rates) != len(labels):
            logger.warning("Mismatched lengths for summary plot data")
            return None

        try:
            import numpy as np

            fig, ax = self._plt.subplots(figsize=(12, 6))

            x = np.arange(len(labels))
            width = 0.35

            # Create bars
            bars1 = ax.bar(x - width/2, [s * 100 for s in success_rates],
                          width, label="Success Rate (%)", color="steelblue")
            bars2 = ax.bar(x + width/2, [s * 100 for s in spl_values],
                          width, label="SPL (%)", color="coral")

            # Configure plot
            ax.set_xlabel("Episode / Run", fontsize=12)
            ax.set_ylabel("Percentage (%)", fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.legend(loc="upper right", fontsize=10)
            ax.set_ylim(0, 105)
            ax.grid(True, axis="y", alpha=0.3)

            # Add value labels on bars
            def autolabel(bars):
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f"{height:.1f}",
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha="center", va="bottom", fontsize=8)

            autolabel(bars1)
            autolabel(bars2)

            # Save plot
            filename = "summary_plot.png"
            filepath = os.path.join(self._output_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            self._plt.close(fig)

            logger.debug(f"Saved summary plot: {filepath}")
            return filepath

        except ImportError:
            logger.warning("numpy not installed, using simple summary plot")
            return self._create_simple_summary_plot(
                success_rates, spl_values, labels, title
            )
        except Exception as e:
            logger.error(f"Failed to create summary plot: {e}")
            return None

    def _create_simple_summary_plot(
        self,
        success_rates: List[float],
        spl_values: List[float],
        labels: List[str],
        title: str
    ) -> Optional[str]:
        """
        Create a simple summary plot without numpy.

        Args:
            success_rates: List of success rates.
            spl_values: List of SPL values.
            labels: Labels for each run.
            title: Plot title.

        Returns:
            Path to saved plot file, or None on failure.
        """
        try:
            fig, ax = self._plt.subplots(figsize=(12, 6))

            x = range(len(labels))
            width = 0.35
            offset = width / 2

            # Create bars
            bars1 = ax.bar(
                [i - offset for i in x],
                [s * 100 for s in success_rates],
                width,
                label="Success Rate (%)",
                color="steelblue"
            )
            bars2 = ax.bar(
                [i + offset for i in x],
                [s * 100 for s in spl_values],
                width,
                label="SPL (%)",
                color="coral"
            )

            # Configure plot
            ax.set_xlabel("Episode / Run", fontsize=12)
            ax.set_ylabel("Percentage (%)", fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.legend(loc="upper right", fontsize=10)
            ax.set_ylim(0, 105)
            ax.grid(True, axis="y", alpha=0.3)

            # Save plot
            filename = "summary_plot.png"
            filepath = os.path.join(self._output_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            self._plt.close(fig)

            return filepath

        except Exception as e:
            logger.error(f"Failed to create simple summary plot: {e}")
            return None

    def create_comparison_plot(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: str = "Method Comparison"
    ) -> Optional[str]:
        """
        Create a comparison bar plot between different methods.

        Args:
            metrics: Dict mapping method names to metric dicts.
                    Example: {"LLM Planner": {"success_rate": 0.8, "spl": 0.6}}
            title: Plot title.

        Returns:
            Path to saved plot file, or None if visualization unavailable.
        """
        if not self._matplotlib_available:
            return None

        if not metrics:
            return None

        try:
            methods = list(metrics.keys())
            success_rates = [metrics[m].get("success_rate", 0) * 100 for m in methods]
            spls = [metrics[m].get("spl", 0) * 100 for m in methods]

            return self.create_summary_plot(
                success_rates=success_rates,
                spl_values=spls,
                labels=methods,
                title=title
            )
        except Exception as e:
            logger.error(f"Failed to create comparison plot: {e}")
            return None

    def is_available(self) -> bool:
        """
        Check if visualization is available.

        Returns:
            True if matplotlib is available, False otherwise.
        """
        return self._matplotlib_available
