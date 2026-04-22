"""
Main entry point for the Vision-Language Embodied Agent.

This module provides the main entry point for running single episodes
or batch evaluations of the embodied agent on navigation tasks.

Usage:
    Single episode:
        python main.py --scene FloorPlan1 --instruction "find the red mug"

    From task file:
        python main.py --task-file tasks/navigation.yaml

    Vision-only mode (no LLM planner):
        python main.py --scene FloorPlan1 --instruction "find the chair" --vision-only
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

from src.agent.controller import ThorController, ThorObservation
from src.agent.planner import PlannerAgent
from src.agent.navigator import NavigatorAgent
from src.perception.visual_encoder import VisualEncoder
from src.perception.detector import ObjectDetector
from src.config.settings import Settings, default_settings
from src.evaluation import EvaluationRunner, EvaluationReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_logging(settings: Settings) -> None:
    """Configure logging based on settings."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)


def run_single_episode(
    scene: str,
    instruction: str,
    initial_position: Optional[Dict[str, float]] = None,
    initial_rotation: float = 0.0,
    use_llm_planner: bool = True,
    settings: Optional[Settings] = None
) -> Dict[str, Any]:
    """
    Run a single episode in the specified scene.

    Args:
        scene: Scene name (e.g., "FloorPlan1").
        instruction: Task instruction.
        initial_position: Starting position with 'x', 'y', 'z' keys.
        initial_rotation: Starting rotation in degrees.
        use_llm_planner: Whether to use LLM for planning.
        settings: Settings instance. Uses default_settings if None.

    Returns:
        Dict with execution results including success status and metrics.
    """
    settings = settings or default_settings
    setup_logging(settings)

    logger.info(f"Starting episode: scene={scene}, instruction='{instruction}'")
    logger.info(f"use_llm_planner={use_llm_planner}")

    # Initialize controller
    controller = ThorController(
        use_thor=not settings.headless,
        settings=settings
    )

    try:
        # Reset environment
        if initial_position is None:
            initial_position = {"x": 0.0, "y": 0.0, "z": 0.0}

        observation = controller.reset(
            scene_name=scene,
            initial_position=initial_position
        )

        logger.info(f"Environment reset: {scene}")
        logger.info(f"Initial position: {observation.agent_state.position}")

        # Create planner agent
        agent = PlannerAgent(
            controller=controller,
            use_llm_planner=use_llm_planner,
            settings=settings
        )

        # Execute task
        result = agent.execute_task(instruction)

        # Get final state
        final_state = controller.get_current_state()

        # Prepare output
        output = {
            "success": result.get("success", False),
            "scene": scene,
            "instruction": instruction,
            "initial_position": initial_position,
            "final_position": final_state.position,
            "executed_actions": result.get("executed_actions", []),
            "verification_results": result.get("verification_results", []),
            "decomposition": result.get("decomposition"),
            "error": result.get("error"),
            "use_llm_planner": use_llm_planner
        }

        # Log result
        if output["success"]:
            logger.info(f"Episode completed successfully!")
        else:
            logger.warning(f"Episode failed: {output.get('error', 'unknown error')}")

        return output

    except Exception as e:
        logger.error(f"Episode failed with exception: {e}")
        return {
            "success": False,
            "scene": scene,
            "instruction": instruction,
            "error": str(e)
        }

    finally:
        controller.close()


def run_from_task_file(
    task_file: str,
    use_llm_planner: bool = True,
    settings: Optional[Settings] = None,
    output_dir: Optional[str] = None
) -> EvaluationReport:
    """
    Run episodes from a task configuration file.

    Args:
        task_file: Path to YAML task configuration file.
        use_llm_planner: Whether to use LLM for planning.
        settings: Settings instance. Uses default_settings if None.
        output_dir: Directory for output files.

    Returns:
        EvaluationReport with aggregated metrics.
    """
    settings = settings or default_settings
    setup_logging(settings)

    logger.info(f"Running evaluation from task file: {task_file}")

    # Create evaluation runner
    runner = EvaluationRunner(
        task_file=task_file,
        use_llm_planner=use_llm_planner,
        settings=settings,
        output_dir=output_dir
    )

    try:
        # Run all episodes
        report = runner.run_all()

        # Save report
        report_path = runner.save_report(report)
        logger.info(f"Report saved to: {report_path}")

        # Generate visualizations
        plot_paths = runner.generate_visualizations()
        if plot_paths:
            logger.info(f"Generated {len(plot_paths)} visualization plots")

        return report

    finally:
        runner.close()


def main() -> int:
    """
    Main entry point for command-line usage.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Vision-Language Embodied Agent - Main Entry Point"
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--scene",
        type=str,
        help="Scene name for single episode mode (e.g., FloorPlan1)"
    )
    mode_group.add_argument(
        "--task-file",
        type=str,
        help="Path to YAML task configuration file for batch mode"
    )

    # Single episode options
    parser.add_argument(
        "--instruction",
        type=str,
        default="find the target",
        help="Task instruction for single episode mode"
    )
    parser.add_argument(
        "--initial-x",
        type=float,
        default=0.0,
        help="Initial X position"
    )
    parser.add_argument(
        "--initial-z",
        type=float,
        default=0.0,
        help="Initial Z position"
    )
    parser.add_argument(
        "--initial-rotation",
        type=float,
        default=0.0,
        help="Initial rotation in degrees"
    )

    # Mode flags
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="Use vision-only mode (no LLM planner)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without THOR rendering window"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for reports and visualizations"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Path to save results as JSON"
    )

    # Debug options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Configure settings
    settings = Settings.from_env()
    if args.debug:
        settings.debug = True
        settings.log_level = "DEBUG"
    if args.headless:
        settings.headless = True

    use_llm_planner = not args.vision_only

    try:
        if args.task_file:
            # Batch mode
            report = run_from_task_file(
                task_file=args.task_file,
                use_llm_planner=use_llm_planner,
                settings=settings,
                output_dir=args.output_dir
            )

            # Print summary
            print("\n" + str(report))

            # Save JSON if requested
            if args.output_json:
                with open(args.output_json, "w") as f:
                    json.dump(report.to_dict(), f, indent=2)
                logger.info(f"Results saved to {args.output_json}")

            return 0 if report.success_rate > 0 else 1

        else:
            # Single episode mode
            initial_position = {
                "x": args.initial_x,
                "y": 0.0,
                "z": args.initial_z
            }

            result = run_single_episode(
                scene=args.scene,
                instruction=args.instruction,
                initial_position=initial_position,
                initial_rotation=args.initial_rotation,
                use_llm_planner=use_llm_planner,
                settings=settings
            )

            # Print result
            print("\n=== Episode Result ===")
            print(f"Scene: {result['scene']}")
            print(f"Instruction: {result['instruction']}")
            print(f"Success: {result['success']}")
            print(f"Actions taken: {len(result.get('executed_actions', []))}")

            if result.get("error"):
                print(f"Error: {result['error']}")

            # Save JSON if requested
            if args.output_json:
                with open(args.output_json, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Results saved to {args.output_json}")

            return 0 if result["success"] else 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if settings.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
