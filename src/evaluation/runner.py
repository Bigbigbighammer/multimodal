"""
Evaluation runner for the Vision-Language Embodied Agent.

This module provides the EvaluationRunner class that orchestrates
episode execution, metrics collection, and report generation.

Key components:
- EpisodeConfig: Configuration for a single evaluation episode
- EvaluationRunner: Main class for running evaluations
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
import math
import os
import yaml

from src.agent.controller import ThorController, ThorObservation
from src.agent.planner import PlannerAgent
from src.agent.navigator import NavigatorAgent
from src.evaluation.metrics import (
    EpisodeMetrics,
    EvaluationReport,
    aggregate_metrics
)
from src.evaluation.visualizer import TrajectoryVisualizer
from src.config.settings import Settings, default_settings

logger = logging.getLogger(__name__)


@dataclass
class EpisodeConfig:
    """
    Configuration for a single evaluation episode.

    Attributes:
        scene: Scene name (e.g., "FloorPlan1").
        instruction: Task instruction string.
        initial_position: Starting position dict with 'x', 'y', 'z'.
        initial_rotation: Starting rotation in degrees.
        target_position: Target position for distance calculation.
    """
    scene: str
    instruction: str
    initial_position: Dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    initial_rotation: float = 0.0
    target_position: Optional[Dict[str, float]] = None


@dataclass
class EpisodeResult:
    """
    Result of running a single episode.

    Attributes:
        episode_id: Unique identifier for the episode.
        config: Episode configuration used.
        success: Whether the task completed successfully.
        metrics: EpisodeMetrics with detailed metrics.
        action_history: List of actions taken.
        position_history: List of positions visited.
        error: Error message if episode failed, None otherwise.
    """
    episode_id: str
    config: EpisodeConfig
    success: bool
    metrics: Optional[EpisodeMetrics] = None
    action_history: List[str] = field(default_factory=list)
    position_history: List[Dict[str, float]] = field(default_factory=list)
    error: Optional[str] = None


class EvaluationRunner:
    """
    Runner for evaluating the embodied agent on navigation tasks.

    This class handles:
    - Loading episode configurations from YAML files
    - Running episodes with the PlannerAgent
    - Computing metrics and generating reports
    - Optional trajectory visualization

    Example:
        >>> runner = EvaluationRunner(task_file="tasks/navigation.yaml")
        >>> report = runner.run_all()
        >>> print(report)
    """

    def __init__(
        self,
        task_config: Optional[Dict[str, Any]] = None,
        task_file: Optional[str] = None,
        use_llm_planner: bool = True,
        settings: Optional[Settings] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the EvaluationRunner.

        Args:
            task_config: Task configuration dict. Use task_file to load from file.
            task_file: Path to YAML file with task configuration.
            use_llm_planner: Whether to use LLM for planning.
            settings: Settings instance. Uses default_settings if None.
            output_dir: Directory for outputs. Uses settings.evaluation.results_dir if None.
        """
        self._settings = settings or default_settings
        self._use_llm_planner = use_llm_planner
        self._output_dir = output_dir or self._settings.evaluation.results_dir

        # Load task configuration
        if task_file:
            self._task_config = self._load_task_file(task_file)
        elif task_config:
            self._task_config = task_config
        else:
            self._task_config = {"episodes": []}

        # Parse episodes
        self._episodes = self._parse_episodes(self._task_config)

        # Controller (initialized lazily)
        self._controller: Optional[ThorController] = None

        # Results storage
        self._results: List[EpisodeResult] = []

        logger.info(
            f"EvaluationRunner initialized with {len(self._episodes)} episodes, "
            f"use_llm_planner={use_llm_planner}"
        )

    def _load_task_file(self, filepath: str) -> Dict[str, Any]:
        """
        Load task configuration from YAML file.

        Args:
            filepath: Path to YAML file.

        Returns:
            Parsed configuration dict.
        """
        try:
            with open(filepath, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded task configuration from {filepath}")
            return config or {}
        except Exception as e:
            logger.error(f"Failed to load task file {filepath}: {e}")
            return {"episodes": []}

    def _parse_episodes(self, config: Dict[str, Any]) -> List[EpisodeConfig]:
        """
        Parse episode configurations from task config.

        Args:
            config: Task configuration dict.

        Returns:
            List of EpisodeConfig instances.
        """
        episodes = []

        raw_episodes = config.get("episodes", [])
        if isinstance(raw_episodes, list):
            for i, ep_data in enumerate(raw_episodes):
                if isinstance(ep_data, dict):
                    episode = EpisodeConfig(
                        scene=ep_data.get("scene", "FloorPlan1"),
                        instruction=ep_data.get("instruction", ""),
                        initial_position=ep_data.get("initial_position", {"x": 0.0, "y": 0.0, "z": 0.0}),
                        initial_rotation=ep_data.get("initial_rotation", 0.0),
                        target_position=ep_data.get("target_position")
                    )
                    episodes.append(episode)

        if not episodes:
            # Create default episode if none specified
            episodes.append(EpisodeConfig(
                scene="FloorPlan1",
                instruction="find the target"
            ))

        return episodes

    def _init_controller(self) -> ThorController:
        """
        Initialize the THOR controller.

        Returns:
            Initialized ThorController instance.
        """
        if self._controller is None:
            self._controller = ThorController(
                use_thor=not self._settings.headless,
                settings=self._settings
            )
        return self._controller

    def run_all(self) -> EvaluationReport:
        """
        Run all episodes and generate an evaluation report.

        Returns:
            EvaluationReport with aggregated metrics.
        """
        logger.info(f"Starting evaluation of {len(self._episodes)} episodes")

        self._results = []
        all_metrics: List[EpisodeMetrics] = []

        for i, episode_config in enumerate(self._episodes):
            episode_id = f"episode_{i+1:03d}"
            logger.info(f"Running episode {i+1}/{len(self._episodes)}: {episode_id}")

            try:
                result = self.run_episode(episode_config, episode_id)
                self._results.append(result)

                if result.metrics:
                    all_metrics.append(result.metrics)

            except Exception as e:
                logger.error(f"Episode {episode_id} failed with error: {e}")
                self._results.append(EpisodeResult(
                    episode_id=episode_id,
                    config=episode_config,
                    success=False,
                    error=str(e)
                ))

        # Aggregate metrics
        report = aggregate_metrics(all_metrics)

        logger.info(
            f"Evaluation complete: {report.total_episodes} episodes, "
            f"success_rate={report.success_rate:.2%}, "
            f"avg_spl={report.avg_spl:.3f}"
        )

        return report

    def run_episode(
        self,
        episode: EpisodeConfig,
        episode_id: str
    ) -> EpisodeResult:
        """
        Run a single evaluation episode.

        Args:
            episode: Episode configuration.
            episode_id: Unique identifier for the episode.

        Returns:
            EpisodeResult with success status and metrics.
        """
        controller = self._init_controller()

        # Reset environment
        observation = controller.reset(
            scene_name=episode.scene,
            initial_position=episode.initial_position
        )

        # Track positions
        position_history: List[Dict[str, float]] = []
        initial_state = controller.get_current_state()
        position_history.append(dict(initial_state.position))

        # Create agent
        agent = PlannerAgent(
            controller=controller,
            use_llm_planner=self._use_llm_planner,
            settings=self._settings
        )

        # Execute task
        result = agent.execute_task(episode.instruction)

        # Track final position
        final_state = controller.get_current_state()
        position_history.append(dict(final_state.position))

        # Compute metrics
        success = result.get("success", False)
        action_history = result.get("executed_actions", [])

        # Calculate path length
        path_length = self._compute_path_length(position_history)

        # Calculate shortest path (estimated)
        shortest_path = self._estimate_shortest_path(
            episode.initial_position,
            episode.target_position
        )

        # Calculate planning efficiency
        planning_efficiency = self._compute_planning_efficiency(result)

        metrics = EpisodeMetrics(
            success=success,
            shortest_path_distance=shortest_path,
            actual_distance=path_length,
            total_steps=len(action_history),
            planning_efficiency=planning_efficiency
        )

        logger.info(
            f"Episode {episode_id}: success={success}, "
            f"steps={metrics.total_steps}, spl={metrics.spl:.3f}"
        )

        return EpisodeResult(
            episode_id=episode_id,
            config=episode,
            success=success,
            metrics=metrics,
            action_history=action_history,
            position_history=position_history,
            error=result.get("error")
        )

    def _compute_path_length(
        self,
        positions: List[Dict[str, float]]
    ) -> float:
        """
        Compute the total path length from position history.

        Args:
            positions: List of position dicts.

        Returns:
            Total distance traveled in meters.
        """
        if len(positions) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(positions)):
            dx = positions[i]["x"] - positions[i-1]["x"]
            dz = positions[i]["z"] - positions[i-1]["z"]
            total_distance += math.sqrt(dx * dx + dz * dz)

        return total_distance

    def _estimate_shortest_path(
        self,
        start: Dict[str, float],
        target: Optional[Dict[str, float]]
    ) -> float:
        """
        Estimate the shortest path distance.

        For now, uses straight-line distance if target is known.
        In a full implementation, this would use path planning.

        Args:
            start: Starting position.
            target: Target position, or None.

        Returns:
            Estimated shortest path distance.
        """
        if target is None:
            # Unknown target - return a default estimate
            return 5.0

        dx = target["x"] - start["x"]
        dz = target["z"] - start["z"]
        return math.sqrt(dx * dx + dz * dz)

    def _compute_planning_efficiency(self, result: Dict[str, Any]) -> float:
        """
        Compute planning efficiency from execution result.

        Planning efficiency is the ratio of successful subgoals
        to total subgoals attempted.

        Args:
            result: Execution result dict.

        Returns:
            Planning efficiency between 0 and 1.
        """
        verification_results = result.get("verification_results", [])
        if not verification_results:
            return 1.0 if result.get("success", False) else 0.0

        successful = sum(1 for v in verification_results if v.get("success", False))
        total = len(verification_results)

        return successful / total if total > 0 else 0.0

    def get_results(self) -> List[EpisodeResult]:
        """
        Get all episode results.

        Returns:
            List of EpisodeResult instances.
        """
        return self._results

    def save_report(
        self,
        report: EvaluationReport,
        output_path: Optional[str] = None
    ) -> str:
        """
        Save evaluation report to a file.

        Args:
            report: EvaluationReport to save.
            output_path: Output file path. Uses default if None.

        Returns:
            Path to saved report.
        """
        if output_path is None:
            os.makedirs(self._output_dir, exist_ok=True)
            output_path = os.path.join(self._output_dir, "evaluation_report.yaml")

        # Prepare report data
        report_data = {
            "summary": report.to_dict(),
            "episodes": [
                {
                    "episode_id": r.episode_id,
                    "success": r.success,
                    "scene": r.config.scene,
                    "instruction": r.config.instruction,
                    "steps": r.metrics.total_steps if r.metrics else 0,
                    "spl": r.metrics.spl if r.metrics else 0.0,
                    "error": r.error
                }
                for r in self._results
            ]
        }

        try:
            with open(output_path, "w") as f:
                yaml.dump(report_data, f, default_flow_style=False)
            logger.info(f"Saved evaluation report to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

        return output_path

    def generate_visualizations(
        self,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Generate trajectory visualizations for all episodes.

        Args:
            output_dir: Directory for visualizations. Uses self._output_dir if None.

        Returns:
            List of paths to generated plot files.
        """
        output_dir = output_dir or self._output_dir
        os.makedirs(output_dir, exist_ok=True)

        visualizer = TrajectoryVisualizer(output_dir)
        plot_paths: List[str] = []

        for result in self._results:
            if result.position_history:
                path = visualizer.plot_trajectory(
                    positions=result.position_history,
                    target_position=result.config.target_position,
                    episode_id=result.episode_id,
                    success=result.success
                )
                if path:
                    plot_paths.append(path)

        # Generate summary plot
        if self._results:
            success_rates = [
                1.0 if r.success else 0.0
                for r in self._results
            ]
            spl_values = [
                r.metrics.spl if r.metrics else 0.0
                for r in self._results
            ]
            labels = [r.episode_id for r in self._results]

            summary_path = visualizer.create_summary_plot(
                success_rates=success_rates,
                spl_values=spl_values,
                labels=labels
            )
            if summary_path:
                plot_paths.append(summary_path)

        return plot_paths

    def close(self) -> None:
        """Clean up resources."""
        if self._controller is not None:
            self._controller.close()
            self._controller = None
        logger.info("EvaluationRunner closed")
