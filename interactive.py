"""
Interactive REPL for the Vision-Language Embodied Agent.

This module provides an interactive command-line interface for real-time
control of the embodied agent.

Usage:
    python interactive.py --scene FloorPlan1

Commands:
    Navigation: go <target>, find <object>, explore
    Actions: pick <object>, put <object>, open <object>, close <object>
    Vision: look, scan, turn <left|right>, turn <degrees>
    Control: status, help, stop, reset, quit
"""

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system env vars

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from src.agent.controller import ThorController, ThorObservation
from src.agent.navigator import NavigatorAgent
from src.agent.planner import PlannerAgent
from src.perception.visual_encoder import VisualEncoder
from src.perception.detector import ObjectDetector
from src.config.settings import Settings, default_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CommandType(Enum):
    """Types of commands."""
    NAVIGATE = "navigate"
    ACTION = "action"
    VISION = "vision"
    CONTROL = "control"
    UNKNOWN = "unknown"


@dataclass
class ParsedCommand:
    """Parsed command result."""
    type: CommandType
    action: str
    target: Optional[str] = None
    args: Dict = None

    def __post_init__(self):
        if self.args is None:
            self.args = {}


class CommandParser:
    """Parser for natural language commands."""

    # Command patterns
    PATTERNS = {
        # Navigation
        r"^go\s+(?:to\s+)?(.+)$": ("navigate", "go"),
        r"^find\s+(.+)$": ("navigate", "find"),
        r"^navigate\s+(?:to\s+)?(.+)$": ("navigate", "go"),
        r"^explore$": ("navigate", "explore"),

        # Actions
        r"^pick\s+(?:up\s+)?(.+)$": ("action", "pickup"),
        r"^grab\s+(.+)$": ("action", "pickup"),
        r"^put\s+(.+)$": ("action", "put"),
        r"^place\s+(.+)$": ("action", "put"),
        r"^open\s+(.+)$": ("action", "open"),
        r"^close\s+(.+)$": ("action", "close"),
        r"^toggle\s+(.+)$": ("action", "toggle"),
        r"^slice\s+(.+)$": ("action", "slice"),
        r"^clean\s+(.+)$": ("action", "clean"),

        # Vision
        r"^look$": ("vision", "look"),
        r"^scan$": ("vision", "scan"),
        r"^what\s+do\s+you\s+see$": ("vision", "scan"),
        r"^turn\s+(left|right)$": ("vision", "turn"),
        r"^turn\s+(-?\d+)$": ("vision", "turn_deg"),
        r"^rotate\s+(left|right)$": ("vision", "turn"),
        r"^face\s+(.+)$": ("vision", "face"),

        # Control
        r"^status$": ("control", "status"),
        r"^help$": ("control", "help"),
        r"^stop$": ("control", "stop"),
        r"^reset$": ("control", "reset"),
        r"^quit$": ("control", "quit"),
        r"^exit$": ("control", "quit"),
    }

    def parse(self, text: str) -> ParsedCommand:
        """
        Parse a command string.

        Args:
            text: Input command string.

        Returns:
            ParsedCommand with type, action, and target.
        """
        text = text.strip().lower()

        for pattern, (cmd_type, action) in self.PATTERNS.items():
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                target = match.group(1) if match.groups() else None

                # Determine command type
                type_map = {
                    "navigate": CommandType.NAVIGATE,
                    "action": CommandType.ACTION,
                    "vision": CommandType.VISION,
                    "control": CommandType.CONTROL,
                }

                return ParsedCommand(
                    type=type_map.get(cmd_type, CommandType.UNKNOWN),
                    action=action,
                    target=target,
                    args={"match_groups": match.groups()}
                )

        return ParsedCommand(type=CommandType.UNKNOWN, action="unknown")


class InteractiveSession:
    """
    Interactive REPL session for the embodied agent.

    This class maintains the THOR environment and provides real-time
    command execution.

    Example:
        >>> session = InteractiveSession("FloorPlan1")
        >>> session.run()
    """

    def __init__(
        self,
        scene: str,
        settings: Optional[Settings] = None,
        use_llm_planner: bool = True
    ):
        """
        Initialize the interactive session.

        Args:
            scene: Scene name to load.
            settings: Settings instance.
            use_llm_planner: Whether to use LLM for planning.
        """
        self._scene = scene
        self._settings = settings or Settings.from_env()
        self._use_llm_planner = use_llm_planner

        # Components
        self._controller: Optional[ThorController] = None
        self._navigator: Optional[NavigatorAgent] = None
        self._planner: Optional[PlannerAgent] = None
        self._visual_encoder: Optional[VisualEncoder] = None
        self._detector: Optional[ObjectDetector] = None
        self._parser = CommandParser()

        # State
        self._running = True
        self._current_task = None
        self._command_history: List[str] = []

        # Prompt session
        self._prompt_session: Optional[PromptSession] = None

    def start(self) -> None:
        """Initialize the session and load the scene."""
        print(f"\n{'='*60}")
        print(f"Vision-Language Embodied Agent - Interactive Mode")
        print(f"{'='*60}")
        print(f"Scene: {self._scene}")
        print(f"LLM Planner: {'Enabled' if self._use_llm_planner else 'Disabled'}")
        print(f"{'='*60}\n")

        # Initialize controller
        print("Initializing environment...")
        self._controller = ThorController(
            use_thor=not self._settings.headless,
            settings=self._settings
        )

        # Reset to scene
        observation = self._controller.reset(scene_name=self._scene)
        print(f"Scene loaded: {self._scene}")
        print(f"Position: {observation.agent_state.position}")

        # Initialize perception
        print("Initializing perception models...")
        self._visual_encoder = VisualEncoder(
            model_name=self._settings.perception.clip_model,
            pretrained=self._settings.perception.clip_pretrained
        )
        self._detector = ObjectDetector(
            model_name=self._settings.perception.yolo_model
        )

        # Initialize navigator
        self._navigator = NavigatorAgent(
            controller=self._controller,
            visual_encoder=self._visual_encoder,
            detector=self._detector,
            settings=self._settings
        )

        # Initialize LLM planner if enabled
        if self._use_llm_planner:
            print("Initializing LLM planner...")
            self._planner = PlannerAgent(
                controller=self._controller,
                use_llm_planner=True,
                settings=self._settings
            )

        print("Ready!\n")

        # Setup prompt
        self._setup_prompt()

    def _setup_prompt(self) -> None:
        """Setup the prompt_toolkit session."""
        # Command completions
        commands = [
            # Navigation
            "go to", "find", "navigate to", "explore",
            # Actions
            "pick up", "grab", "put", "place", "open", "close", "toggle",
            # Vision
            "look", "scan", "turn left", "turn right", "face",
            # Control
            "status", "help", "stop", "reset", "quit",
        ]

        completer = WordCompleter(commands, ignore_case=True)

        # Custom style
        style = Style.from_dict({
            'prompt': '#00aa00 bold',
        })

        self._prompt_session = PromptSession(
            completer=completer,
            style=style,
            history=FileHistory('.interactive_history'),
        )

    def run(self) -> None:
        """Run the interactive REPL loop."""
        self.start()

        print("Enter natural language commands. Type 'help' for examples, 'quit' to exit.\n")

        while self._running:
            try:
                # Get input
                user_input = self._prompt_session.prompt(">>> ").strip()

                if not user_input:
                    continue

                # Handle special control commands
                lower_input = user_input.lower()
                if lower_input in ("quit", "exit"):
                    self._quit()
                    break
                elif lower_input == "help":
                    self._print_help()
                    continue
                elif lower_input == "status":
                    self._print_status()
                    continue
                elif lower_input == "reset":
                    print("[Resetting environment...]")
                    self._controller.reset(scene_name=self._scene)
                    print("Environment reset.")
                    continue

                # All other input goes to LLM planner
                self._execute_natural_language(user_input)

            except KeyboardInterrupt:
                print("\nUse 'quit' to exit.")
                continue

            except EOFError:
                self._quit()
                break

    def _execute_natural_language(self, instruction: str) -> None:
        """
        Execute a natural language instruction using LLM planner.

        Args:
            instruction: Natural language instruction from user.
        """
        self._command_history.append(instruction)

        print(f"\n[Executing: {instruction}]")

        if self._use_llm_planner and self._planner:
            # Use LLM planner for intelligent task execution
            result = self._planner.execute_task(instruction)

            if result.get("success"):
                steps = len(result.get("executed_actions", []))
                decomposition = result.get("decomposition")
                if decomposition:
                    subgoals = decomposition.get("subgoals", [])
                    print(f"✓ Task completed!")
                    print(f"  Subgoals: {len(subgoals)}, Steps: {steps}")
                else:
                    print(f"✓ Task completed in {steps} steps")
            else:
                error = result.get("error", "unknown error")
                print(f"✗ Failed: {error}")
        else:
            # Fallback: try to use navigator for simple navigation
            print("[No LLM planner available, using basic navigation...]")
            result = self._navigator.navigate_to_target(instruction)

            if result.get("success"):
                distance = result.get("distance", 0)
                steps = result.get("steps", 0)
                print(f"✓ Reached target (distance: {distance:.2f}m, steps: {steps})")
            else:
                error = result.get("error", "unknown error")
                print(f"✗ Failed: {error}")

        print()

    def _print_help(self) -> None:
        """Print help message."""
        help_text = """
╔══════════════════════════════════════════════════════════╗
║              NATURAL LANGUAGE INTERACTION                  ║
╠══════════════════════════════════════════════════════════╣
║ Just type what you want to do in natural language!         ║
║                                                             ║
║ Examples:                                                   ║
║   "find the chair"          - Navigate to a chair          ║
║   "go to the kitchen"       - Navigate to kitchen          ║
║   "pick up the apple"       - Pick up an apple             ║
║   "open the fridge"         - Open the refrigerator        ║
║   "look around"             - Scan the environment         ║
║   "what can you see"        - List visible objects         ║
║                                                             ║
╠══════════════════════════════════════════════════════════╣
║ CONTROL COMMANDS                                            ║
║   status   - Show agent status                              ║
║   reset    - Reset environment                              ║
║   help     - Show this help                                 ║
║   quit     - Exit interactive mode                          ║
╚══════════════════════════════════════════════════════════╝
"""
        print(help_text)

    def _print_status(self) -> None:
        """Print current status."""
        observation = self._controller.get_current_observation()
        state = observation.agent_state

        print(f"\n{'='*40}")
        print("AGENT STATUS:")
        print(f"{'='*40}")
        print(f"Scene: {self._scene}")
        print(f"Position: ({state.position['x']:.2f}, {state.position['y']:.2f}, {state.position['z']:.2f})")
        print(f"Rotation: ({state.rotation['x']:.1f}°, {state.rotation['y']:.1f}°, {state.rotation['z']:.1f}°)")
        print(f"Visible objects: {len(observation.visible_objects)}")
        print(f"Held object: {observation.agent_state.held_object or 'None'}")
        print(f"Commands executed: {len(self._command_history)}")
        print(f"{'='*40}\n")

    def _quit(self) -> None:
        """Quit the session."""
        print("\nClosing environment...")
        if self._controller:
            self._controller.close()
        self._running = False
        print("Goodbye!")

    def close(self) -> None:
        """Close the session."""
        if self._controller:
            self._controller.close()


def main() -> int:
    """
    Main entry point for interactive mode.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Vision-Language Embodied Agent - Interactive Mode"
    )

    parser.add_argument(
        "--scene",
        type=str,
        default="FloorPlan1",
        help="Scene name to load (default: FloorPlan1)"
    )
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="Disable LLM planner"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without THOR rendering window"
    )
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
        session = InteractiveSession(
            scene=args.scene,
            settings=settings,
            use_llm_planner=use_llm_planner
        )
        session.run()
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    except Exception as e:
        logger.error(f"Session failed: {e}")
        if settings.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
