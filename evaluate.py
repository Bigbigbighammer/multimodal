"""
Evaluation entry point for the Vision-Language Embodied Agent.

This module provides the entry point for running batch evaluations
and generating evaluation reports.

Usage:
    Basic evaluation:
        python evaluate.py --task-file tasks/navigation.yaml

    With output directory:
        python evaluate.py --task-file tasks/navigation.yaml --output-dir results/eval_001

    Vision-only baseline:
        python evaluate.py --task-file tasks/navigation.yaml --vision-only

    Compare LLM vs vision-only:
        python evaluate.py --task-file tasks/navigation.yaml --compare
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

from src.config.settings import Settings
from src.evaluation import (
    EvaluationRunner,
    EvaluationReport,
    EpisodeResult,
    aggregate_metrics,
    TrajectoryVisualizer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_evaluation(
    task_file: str,
    use_llm_planner: bool = True,
    settings: Optional[Settings] = None,
    output_dir: Optional[str] = None
) -> EvaluationReport:
    """
    Run evaluation from task file.

    Args:
        task_file: Path to YAML task configuration file.
        use_llm_planner: Whether to use LLM for planning.
        settings: Settings instance.
        output_dir: Output directory for results.

    Returns:
        EvaluationReport with aggregated metrics.
    """
    settings = settings or Settings.from_env()

    # Set log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    planner_type = "LLM" if use_llm_planner else "Vision-Only"
    logger.info(f"Starting evaluation with {planner_type} planner")
    logger.info(f"Task file: {task_file}")

    # Create output directory with timestamp
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            settings.evaluation.results_dir,
            f"eval_{timestamp}"
        )

    os.makedirs(output_dir, exist_ok=True)

    # Create runner
    runner = EvaluationRunner(
        task_file=task_file,
        use_llm_planner=use_llm_planner,
        settings=settings,
        output_dir=output_dir
    )

    try:
        # Run evaluation
        report = runner.run_all()

        # Save report
        report_path = runner.save_report(report)

        # Generate visualizations
        if settings.evaluation.save_trajectory_images:
            plot_paths = runner.generate_visualizations()
            logger.info(f"Generated {len(plot_paths)} visualization plots")

        # Save detailed episode results
        episodes_path = os.path.join(output_dir, "episodes.json")
        save_episode_results(runner.get_results(), episodes_path)

        return report

    finally:
        runner.close()


def run_comparison(
    task_file: str,
    settings: Optional[Settings] = None,
    output_dir: Optional[str] = None
) -> Dict[str, EvaluationReport]:
    """
    Run comparison between LLM and vision-only planners.

    Args:
        task_file: Path to task configuration file.
        settings: Settings instance.
        output_dir: Output directory for results.

    Returns:
        Dict with 'llm' and 'vision_only' EvaluationReport instances.
    """
    settings = settings or Settings.from_env()

    # Set log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    logger.info("Running comparison evaluation: LLM vs Vision-Only")

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            settings.evaluation.results_dir,
            f"comparison_{timestamp}"
        )

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Run LLM planner evaluation
    logger.info("=== Running LLM Planner Evaluation ===")
    llm_output_dir = os.path.join(output_dir, "llm_planner")
    results["llm"] = run_evaluation(
        task_file=task_file,
        use_llm_planner=True,
        settings=settings,
        output_dir=llm_output_dir
    )

    # Run vision-only evaluation
    logger.info("=== Running Vision-Only Evaluation ===")
    vision_output_dir = os.path.join(output_dir, "vision_only")
    results["vision_only"] = run_evaluation(
        task_file=task_file,
        use_llm_planner=False,
        settings=settings,
        output_dir=vision_output_dir
    )

    # Generate comparison visualization
    visualizer = TrajectoryVisualizer(output_dir)
    comparison_metrics = {
        "LLM Planner": {
            "success_rate": results["llm"].success_rate,
            "spl": results["llm"].avg_spl
        },
        "Vision-Only": {
            "success_rate": results["vision_only"].success_rate,
            "spl": results["vision_only"].avg_spl
        }
    }
    visualizer.create_comparison_plot(comparison_metrics, "LLM vs Vision-Only Comparison")

    # Save comparison summary
    comparison_summary = {
        "task_file": task_file,
        "timestamp": datetime.now().isoformat(),
        "llm_planner": results["llm"].to_dict(),
        "vision_only": results["vision_only"].to_dict(),
        "improvement": {
            "success_rate": results["llm"].success_rate - results["vision_only"].success_rate,
            "spl": results["llm"].avg_spl - results["vision_only"].avg_spl
        }
    }

    summary_path = os.path.join(output_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(comparison_summary, f, indent=2)

    logger.info(f"Comparison saved to {output_dir}")
    print_comparison_summary(results)

    return results


def save_episode_results(
    results: List[EpisodeResult],
    output_path: str
) -> None:
    """
    Save detailed episode results to JSON.

    Args:
        results: List of EpisodeResult instances.
        output_path: Path to output JSON file.
    """
    episodes_data = []
    for r in results:
        episode_data = {
            "episode_id": r.episode_id,
            "scene": r.config.scene,
            "instruction": r.config.instruction,
            "initial_position": r.config.initial_position,
            "success": r.success,
            "metrics": {
                "spl": r.metrics.spl if r.metrics else None,
                "total_steps": r.metrics.total_steps if r.metrics else None,
                "actual_distance": r.metrics.actual_distance if r.metrics else None,
                "planning_efficiency": r.metrics.planning_efficiency if r.metrics else None
            } if r.metrics else None,
            "action_history": r.action_history,
            "error": r.error
        }
        episodes_data.append(episode_data)

    with open(output_path, "w") as f:
        json.dump(episodes_data, f, indent=2)

    logger.info(f"Episode results saved to {output_path}")


def print_comparison_summary(results: Dict[str, EvaluationReport]) -> None:
    """Print a formatted comparison summary."""
    print("\n" + "=" * 60)
    print("EVALUATION COMPARISON SUMMARY")
    print("=" * 60)

    llm = results["llm"]
    vo = results["vision_only"]

    print(f"\n{'Metric':<25} {'LLM Planner':>15} {'Vision-Only':>15}")
    print("-" * 55)
    print(f"{'Success Rate':<25} {llm.success_rate*100:>14.1f}% {vo.success_rate*100:>14.1f}%")
    print(f"{'Average SPL':<25} {llm.avg_spl:>15.3f} {vo.avg_spl:>15.3f}")
    print(f"{'Average Steps':<25} {llm.avg_steps:>15.1f} {vo.avg_steps:>15.1f}")
    print(f"{'Avg Distance (m)':<25} {llm.avg_distance:>15.2f} {vo.avg_distance:>15.2f}")
    print(f"{'Avg Planning Eff.':<25} {llm.avg_planning_efficiency:>15.2f} {vo.avg_planning_efficiency:>15.2f}")

    print("\n" + "-" * 55)
    print("Improvement (LLM over Vision-Only):")
    print(f"  Success Rate: {(llm.success_rate - vo.success_rate)*100:+.1f}%")
    print(f"  SPL: {llm.avg_spl - vo.avg_spl:+.3f}")
    print("=" * 60 + "\n")


def main() -> int:
    """
    Main entry point for evaluation.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Vision-Language Embodied Agent - Evaluation Entry Point"
    )

    # Required arguments
    parser.add_argument(
        "--task-file",
        type=str,
        required=True,
        help="Path to YAML task configuration file"
    )

    # Mode options
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="Run vision-only mode (no LLM planner)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison between LLM and vision-only planners"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base results directory"
    )

    # Debug options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without THOR rendering window"
    )

    args = parser.parse_args()

    # Validate task file exists
    if not os.path.exists(args.task_file):
        logger.error(f"Task file not found: {args.task_file}")
        return 1

    # Configure settings
    settings = Settings.from_env()
    if args.debug:
        settings.debug = True
        settings.log_level = "DEBUG"
    if args.headless:
        settings.headless = True
    if args.results_dir:
        settings.evaluation.results_dir = args.results_dir

    try:
        if args.compare:
            # Run comparison
            results = run_comparison(
                task_file=args.task_file,
                settings=settings,
                output_dir=args.output_dir
            )
            # Return success if LLM planner does better
            llm_wins = (
                results["llm"].success_rate >= results["vision_only"].success_rate
                or results["llm"].avg_spl >= results["vision_only"].avg_spl
            )
            return 0 if llm_wins else 1

        else:
            # Run single evaluation
            use_llm_planner = not args.vision_only
            report = run_evaluation(
                task_file=args.task_file,
                use_llm_planner=use_llm_planner,
                settings=settings,
                output_dir=args.output_dir
            )

            # Print summary
            print("\n" + str(report))

            return 0 if report.success_rate > 0 else 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if settings.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
