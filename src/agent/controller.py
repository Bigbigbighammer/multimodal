"""
ThorController for AI2-THOR environment interaction.

This module provides the main controller interface for interacting with the
AI2-THOR simulation environment. It wraps the AI2-THOR controller and provides
a clean API with error handling and support for mock mode (testing without THOR).

Key design principles:
- StepResult always returns (success, observation, error) - never raises exceptions
- Support mock mode (use_thor=False) for testing without AI2-THOR installed
- Lazy initialization of THOR controller on first reset()
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import logging

# Handle AI2-THOR availability
try:
    import ai2thor.controller
    THOR_AVAILABLE = True
except ImportError:
    THOR_AVAILABLE = False
    ai2thor = None

from config.settings import Settings, default_settings

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """
    Represents the current state of the agent in the environment.

    Attributes:
        position: Dict with 'x', 'y', 'z' coordinates in meters
        rotation: Dict with 'x', 'y', 'z' rotation angles in degrees
        held_object: Name of object currently held, or None
    """
    position: Dict[str, float]
    rotation: Dict[str, float]
    held_object: Optional[str] = None


@dataclass
class ThorObservation:
    """
    Observation returned by the THOR environment after each action.

    Attributes:
        rgb: RGB image as bytes (or numpy array if THOR available)
        depth: Depth image as bytes/None
        instance_mask: Instance segmentation mask as bytes/None
        agent_state: Current agent state
        visible_objects: List of visible object names
    """
    rgb: Any  # bytes in mock mode, numpy array with THOR
    depth: Optional[Any]  # bytes in mock mode, numpy array with THOR
    instance_mask: Optional[Any]  # bytes in mock mode, numpy array with THOR
    agent_state: AgentState
    visible_objects: List[str] = field(default_factory=list)


@dataclass
class StepResult:
    """
    Result of a step action in the environment.

    Design principle: Never raises exceptions. All errors are captured
    in the error field with success=False.

    Attributes:
        success: Whether the action succeeded
        observation: ThorObservation if success, None if failure
        error: Error message if success=False, None otherwise
    """
    success: bool
    observation: Optional[ThorObservation]
    error: Optional[str]


class ThorController:
    """
    Controller for AI2-THOR environment interaction.

    Provides a clean interface for:
    - Initializing and resetting the environment
    - Executing navigation and manipulation actions
    - Getting observations and agent state
    - Handling errors gracefully

    Supports mock mode for testing without AI2-THOR installed.

    Example:
        >>> controller = ThorController(use_thor=False)  # Mock mode
        >>> observation = controller.reset("FloorPlan1", {"x": 0, "y": 0, "z": 0})
        >>> result = controller.step("MoveAhead")
        >>> if result.success:
        ...     print(result.observation.agent_state.position)
        >>> controller.close()
    """

    VALID_ACTIONS: Set[str] = {
        # Movement actions
        "MoveAhead", "MoveBack", "MoveLeft", "MoveRight",
        # Rotation actions
        "RotateLeft", "RotateRight",
        # Camera actions
        "LookUp", "LookDown",
        # Object interaction actions
        "PickupObject", "PutObject", "OpenObject", "CloseObject",
        # Teleport
        "Teleport",
        # Additional actions
        "Stand", "Crouch", "LieDown", "MoveHand", "Touch",
        "CastSpell", "Drink", "FillObjectWithLiquid", "EmptyLiquidFromObject",
        "Create", "Destroy"
    }

    def __init__(self, use_thor: bool = True, settings: Optional[Settings] = None):
        """
        Initialize the ThorController.

        Args:
            use_thor: If True, use real AI2-THOR. If False, use mock mode.
            settings: Settings instance. If None, uses default_settings.
        """
        self._use_thor = use_thor and THOR_AVAILABLE
        self._settings = settings or default_settings
        self._controller: Optional[Any] = None
        self._initialized = False
        self._current_scene: Optional[str] = None

        # Mock state (used when use_thor=False)
        self._mock_position: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._mock_rotation: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._mock_horizon: float = 0.0  # Camera pitch
        self._mock_held_object: Optional[str] = None
        self._mock_reachable_positions: List[Dict[str, float]] = []

        if self._use_thor and not THOR_AVAILABLE:
            logger.warning("AI2-THOR requested but not installed. Falling back to mock mode.")
            self._use_thor = False

    def _init_controller(self) -> None:
        """
        Initialize the AI2-THOR controller with settings.

        Called lazily on first reset() call.
        """
        if not self._use_thor:
            return

        if THOR_AVAILABLE:
            try:
                self._controller = ai2thor.controller.Controller(
                    width=self._settings.thor.width,
                    height=self._settings.thor.height,
                    renderDepthImage=self._settings.thor.render_depth,
                    renderInstanceSegmentation=self._settings.thor.render_instance_segmentation,
                    visibilityDistance=self._settings.thor.visibility_distance,
                    snapToGrid=self._settings.thor.snap_to_grid,
                    rotateStepDegrees=self._settings.thor.rotate_step_degrees,
                    horizonStepDegrees=self._settings.thor.horizon_step_degrees,
                    gridSize=self._settings.thor.grid_size,
                )
                logger.info("AI2-THOR controller initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AI2-THOR: {e}")
                raise RuntimeError(f"Failed to initialize AI2-THOR: {e}")

    def reset(
        self,
        scene_name: str,
        initial_position: Optional[Dict[str, float]] = None
    ) -> ThorObservation:
        """
        Reset the environment to a scene with optional initial position.

        Args:
            scene_name: Name of the scene (e.g., "FloorPlan1")
            initial_position: Starting position with 'x', 'y', 'z' keys.
                            If None, uses scene default.

        Returns:
            ThorObservation with initial state
        """
        if initial_position is None:
            initial_position = {"x": 0.0, "y": 0.0, "z": 0.0}

        if self._use_thor:
            return self._reset_thor(scene_name, initial_position)
        else:
            return self._reset_mock(scene_name, initial_position)

    def _reset_thor(
        self,
        scene_name: str,
        initial_position: Dict[str, float]
    ) -> ThorObservation:
        """Reset with real AI2-THOR."""
        if not self._initialized:
            self._init_controller()
            self._initialized = True

        try:
            self._controller.reset(scene_name)
            self._current_scene = scene_name

            # Teleport to initial position
            event = self._controller.step(
                action="Teleport",
                x=initial_position["x"],
                y=initial_position.get("y", 0.0),
                z=initial_position["z"],
                rotation=initial_position.get("rotation", 0.0)
            )

            return self._get_observation_thor(event)
        except Exception as e:
            logger.error(f"Failed to reset THOR: {e}")
            # Return mock observation on failure
            return self._get_mock_observation()

    def _reset_mock(
        self,
        scene_name: str,
        initial_position: Dict[str, float]
    ) -> ThorObservation:
        """Reset in mock mode."""
        self._current_scene = scene_name
        self._mock_position = dict(initial_position)
        self._mock_rotation = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._mock_horizon = 0.0
        self._mock_held_object = None

        # Generate mock reachable positions
        self._mock_reachable_positions = self._generate_mock_reachable_positions()

        return self._get_mock_observation()

    def _generate_mock_reachable_positions(self) -> List[Dict[str, float]]:
        """Generate mock reachable positions for testing."""
        positions = []
        grid_size = self._settings.thor.grid_size

        # Generate a grid of positions centered around origin
        for i in range(-5, 6):
            for j in range(-5, 6):
                positions.append({
                    "x": i * grid_size,
                    "y": 0.0,
                    "z": j * grid_size
                })
        return positions

    def step(self, action: str, **kwargs) -> StepResult:
        """
        Execute an action in the environment.

        Args:
            action: Action name (e.g., "MoveAhead", "PickupObject")
            **kwargs: Action-specific parameters (e.g., objectId for PickupObject)

        Returns:
            StepResult with success status, observation (if success), and error (if failure)
        """
        # Validate action
        if action not in self.VALID_ACTIONS:
            return StepResult(
                success=False,
                observation=None,
                error=f"Invalid action: '{action}'. Valid actions: {sorted(self.VALID_ACTIONS)}"
            )

        # Check if initialized
        if self._current_scene is None:
            return StepResult(
                success=False,
                observation=None,
                error="Environment not initialized. Call reset() first."
            )

        if self._use_thor:
            return self._step_thor(action, **kwargs)
        else:
            return self._step_mock(action, **kwargs)

    def _step_thor(self, action: str, **kwargs) -> StepResult:
        """Execute action with real AI2-THOR."""
        try:
            # Build action dict
            action_dict = {"action": action}
            action_dict.update(kwargs)

            event = self._controller.step(action_dict)

            if not event:
                return StepResult(
                    success=False,
                    observation=None,
                    error=f"THOR returned null event for action: {action}"
                )

            observation = self._get_observation_thor(event)
            return StepResult(
                success=True,
                observation=observation,
                error=None
            )
        except Exception as e:
            logger.error(f"THOR step failed: {e}")
            return StepResult(
                success=False,
                observation=None,
                error=f"THOR step failed: {str(e)}"
            )

    def _step_mock(self, action: str, **kwargs) -> StepResult:
        """Execute action in mock mode."""
        try:
            # Simulate action effects
            if action == "MoveAhead":
                self._mock_move_forward()
            elif action == "MoveBack":
                self._mock_move_backward()
            elif action == "MoveLeft":
                self._mock_move_left()
            elif action == "MoveRight":
                self._mock_move_right()
            elif action == "RotateLeft":
                self._mock_rotate(-self._settings.thor.rotate_step_degrees)
            elif action == "RotateRight":
                self._mock_rotate(self._settings.thor.rotate_step_degrees)
            elif action == "LookUp":
                self._mock_horizon = max(-60.0, self._mock_horizon - self._settings.thor.horizon_step_degrees)
            elif action == "LookDown":
                self._mock_horizon = min(60.0, self._mock_horizon + self._settings.thor.horizon_step_degrees)
            elif action == "PickupObject":
                object_id = kwargs.get("objectId")
                if object_id:
                    # Extract object name from objectId (format: "ObjectName|x|y|z")
                    self._mock_held_object = object_id.split("|")[0]
                else:
                    return StepResult(
                        success=False,
                        observation=None,
                        error="PickupObject requires 'objectId' parameter"
                    )
            elif action == "PutObject":
                # Put down the held object
                self._mock_held_object = None
            elif action == "OpenObject":
                pass  # No state change in mock
            elif action == "CloseObject":
                pass  # No state change in mock
            elif action == "Teleport":
                self._mock_position = {
                    "x": kwargs.get("x", self._mock_position["x"]),
                    "y": kwargs.get("y", self._mock_position["y"]),
                    "z": kwargs.get("z", self._mock_position["z"])
                }
                if "rotation" in kwargs:
                    self._mock_rotation["y"] = kwargs["rotation"]

            observation = self._get_mock_observation()
            return StepResult(
                success=True,
                observation=observation,
                error=None
            )
        except Exception as e:
            return StepResult(
                success=False,
                observation=None,
                error=f"Mock step failed: {str(e)}"
            )

    def _mock_move_forward(self) -> None:
        """Move forward in the direction the agent is facing."""
        import math
        angle_rad = math.radians(self._mock_rotation["y"])
        distance = self._settings.thor.grid_size

        # Forward is negative z in THOR convention, rotated by y-angle
        self._mock_position["x"] += distance * math.sin(angle_rad)
        self._mock_position["z"] += distance * math.cos(angle_rad)

    def _mock_move_backward(self) -> None:
        """Move backward (opposite of facing direction)."""
        import math
        angle_rad = math.radians(self._mock_rotation["y"])
        distance = self._settings.thor.grid_size

        self._mock_position["x"] -= distance * math.sin(angle_rad)
        self._mock_position["z"] -= distance * math.cos(angle_rad)

    def _mock_move_left(self) -> None:
        """Move left (strafe left)."""
        import math
        angle_rad = math.radians(self._mock_rotation["y"] + 90)
        distance = self._settings.thor.grid_size

        self._mock_position["x"] += distance * math.sin(angle_rad)
        self._mock_position["z"] += distance * math.cos(angle_rad)

    def _mock_move_right(self) -> None:
        """Move right (strafe right)."""
        import math
        angle_rad = math.radians(self._mock_rotation["y"] - 90)
        distance = self._settings.thor.grid_size

        self._mock_position["x"] += distance * math.sin(angle_rad)
        self._mock_position["z"] += distance * math.cos(angle_rad)

    def _mock_rotate(self, degrees: float) -> None:
        """Rotate the agent by the given degrees."""
        self._mock_rotation["y"] = (self._mock_rotation["y"] + degrees) % 360.0

    def get_reachable_positions(self) -> List[Dict[str, float]]:
        """
        Get all reachable positions in the current scene.

        Returns:
            List of position dicts with 'x', 'y', 'z' keys
        """
        if self._current_scene is None:
            return []

        if self._use_thor and self._controller is not None:
            try:
                event = self._controller.step(action="GetReachablePositions")
                if event and hasattr(event, "metadata"):
                    positions = event.metadata.get("actionReturn", [])
                    return [
                        {"x": p["x"], "y": p["y"], "z": p["z"]}
                        for p in positions
                    ]
            except Exception as e:
                logger.error(f"Failed to get reachable positions: {e}")
                return self._mock_reachable_positions

        return self._mock_reachable_positions

    def get_current_state(self) -> AgentState:
        """
        Get the current agent state.

        Returns:
            AgentState with position, rotation, and held_object
        """
        if self._use_thor and self._controller is not None:
            try:
                event = self._controller.last_event
                if event and hasattr(event, "metadata"):
                    metadata = event.metadata
                    agent = metadata.get("agent", {})
                    position = agent.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
                    rotation = agent.get("rotation", {"x": 0.0, "y": 0.0, "z": 0.0})

                    # Get held object
                    held_object = None
                    inventory = agent.get("inventory", [])
                    if inventory:
                        held_object = inventory[0].get("objectType", None)

                    return AgentState(
                        position={"x": position.get("x", 0.0),
                                 "y": position.get("y", 0.0),
                                 "z": position.get("z", 0.0)},
                        rotation={"x": rotation.get("x", 0.0),
                                 "y": rotation.get("y", 0.0),
                                 "z": rotation.get("z", 0.0)},
                        held_object=held_object
                    )
            except Exception as e:
                logger.error(f"Failed to get current state: {e}")

        # Return mock state
        return AgentState(
            position=dict(self._mock_position),
            rotation=dict(self._mock_rotation),
            held_object=self._mock_held_object
        )

    def _get_observation(self) -> ThorObservation:
        """Get current observation (dispatches to THOR or mock)."""
        if self._use_thor and self._controller is not None:
            try:
                return self._get_observation_thor(self._controller.last_event)
            except Exception as e:
                logger.error(f"Failed to get THOR observation: {e}")

        return self._get_mock_observation()

    def _get_observation_thor(self, event: Any) -> ThorObservation:
        """Extract observation from THOR event."""
        metadata = event.metadata if event else {}
        agent = metadata.get("agent", {})

        position = agent.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
        rotation = agent.get("rotation", {"x": 0.0, "y": 0.0, "z": 0.0})

        # Get held object
        held_object = None
        inventory = agent.get("inventory", [])
        if inventory:
            held_object = inventory[0].get("objectType", None)

        agent_state = AgentState(
            position={"x": position.get("x", 0.0),
                     "y": position.get("y", 0.0),
                     "z": position.get("z", 0.0)},
            rotation={"x": rotation.get("x", 0.0),
                     "y": rotation.get("y", 0.0),
                     "z": rotation.get("z", 0.0)},
            held_object=held_object
        )

        # Get visible objects
        visible_objects = []
        objects = metadata.get("objects", [])
        for obj in objects:
            if obj.get("visible", False):
                visible_objects.append(obj.get("objectType", "Unknown"))

        # Get image data
        rgb = None
        depth = None
        instance_mask = None

        if hasattr(event, "frame") and event.frame is not None:
            rgb = event.frame
        if hasattr(event, "depth_frame") and event.depth_frame is not None:
            depth = event.depth_frame
        if hasattr(event, "instance_segmentation_frame") and event.instance_segmentation_frame is not None:
            instance_mask = event.instance_segmentation_frame

        return ThorObservation(
            rgb=rgb if rgb is not None else b"",
            depth=depth,
            instance_mask=instance_mask,
            agent_state=agent_state,
            visible_objects=visible_objects
        )

    def _get_mock_observation(self) -> ThorObservation:
        """Generate mock observation for testing."""
        agent_state = AgentState(
            position=dict(self._mock_position),
            rotation=dict(self._mock_rotation),
            held_object=self._mock_held_object
        )

        # Return dummy byte data for images
        return ThorObservation(
            rgb=b"mock_rgb_data",
            depth=b"mock_depth_data" if self._settings.thor.render_depth else None,
            instance_mask=b"mock_mask_data" if self._settings.thor.render_instance_segmentation else None,
            agent_state=agent_state,
            visible_objects=["MockObject1", "MockObject2"]
        )

    def close(self) -> None:
        """Close the controller and release resources."""
        if self._controller is not None:
            try:
                self._controller.stop()
            except Exception as e:
                logger.warning(f"Error closing THOR controller: {e}")
            finally:
                self._controller = None

        self._initialized = False
        self._current_scene = None
        self._mock_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._mock_rotation = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._mock_held_object = None

        logger.info("ThorController closed")

    def __enter__(self) -> "ThorController":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
