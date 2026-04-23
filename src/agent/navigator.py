"""
NavigatorAgent for visual navigation in the embodied agent.

This module provides the NavigatorAgent class that combines perception
(visual encoding, object detection) with spatial reasoning (topological map)
to navigate to target objects in AI2-THOR environments.

Key components:
- NavigationResult: Dataclass for navigation results
- ObjectInfo: Dataclass for detected object information
- NavigatorAgent: Main agent class for navigation tasks
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import math

from src.agent.controller import ThorController, StepResult
from src.memory.spatial_map import TopologicalMap, Position, euclidean_distance
from src.memory.working_memory import WorkingMemory
from src.perception.visual_encoder import VisualEncoder
from src.perception.detector import ObjectDetector, Detection
from src.config.settings import Settings, default_settings

logger = logging.getLogger(__name__)


@dataclass
class NavigationResult:
    """
    Result of a navigation attempt.

    Attributes:
        success: Whether navigation completed successfully.
        final_position: The agent's final position after navigation.
        steps_taken: Number of steps executed during navigation.
        distance_to_target: Distance to target at navigation end.
        error: Error message if navigation failed, None otherwise.
    """
    success: bool
    final_position: Dict[str, float]
    steps_taken: int
    distance_to_target: float
    error: Optional[str] = None


@dataclass
class ObjectInfo:
    """
    Information about a detected object.

    Attributes:
        object_id: Unique identifier from THOR (e.g., "Chair|1|2|3").
        object_type: Type of the object (e.g., "Chair", "Table").
        position: 3D position as dict with x, y, z keys.
        distance: Distance from agent to object.
        detection: Detection object from the detector (if any).
        clip_similarity: CLIP similarity score if matched against text.
    """
    object_id: str
    object_type: str
    position: Dict[str, float]
    distance: float
    detection: Optional[Detection] = None
    clip_similarity: float = 0.0


class NavigatorAgent:
    """
    Agent for visual navigation using CLIP and YOLO.

    This agent combines:
    - ThorController for environment interaction
    - VisualEncoder (CLIP) for visual feature matching
    - ObjectDetector (YOLO) for object detection
    - TopologicalMap for spatial reasoning
    - WorkingMemory for state tracking

    Example:
        >>> controller = ThorController(use_thor=False)
        >>> encoder = VisualEncoder(use_mock=True)
        >>> detector = ObjectDetector(use_mock=True)
        >>> agent = NavigatorAgent(controller, encoder, detector)
        >>> controller.reset("FloorPlan1")
        >>> result = agent.navigate_to("red chair", max_steps=50)
    """

    def __init__(
        self,
        controller: ThorController,
        visual_encoder: VisualEncoder,
        detector: ObjectDetector,
        settings: Optional[Settings] = None
    ):
        """
        Initialize the NavigatorAgent.

        Args:
            controller: ThorController for environment interaction.
            visual_encoder: VisualEncoder for CLIP-based matching.
            detector: ObjectDetector for YOLO-based detection.
            settings: Settings instance. Uses default_settings if None.
        """
        self._controller = controller
        self._visual_encoder = visual_encoder
        self._detector = detector
        self._settings = settings or default_settings

        # Initialize memory components
        self._spatial_map: Optional[TopologicalMap] = None
        self._working_memory = WorkingMemory(
            max_steps=self._settings.memory.working_memory_max_steps
        )

        # Navigation state
        self._current_position: Position = (0.0, 0.0)
        self._visited_positions: List[Position] = []
        self._exploration_direction = 0  # Current facing direction for exploration

    def build_spatial_map(self) -> TopologicalMap:
        """
        Build a spatial map from reachable positions in the environment.

        Gets reachable positions from the THOR controller and builds
        a topological map with nodes and edges for path planning.

        Returns:
            TopologicalMap with nodes at reachable positions.
        """
        self._spatial_map = TopologicalMap()

        # Get reachable positions from controller
        reachable = self._controller.get_reachable_positions()

        # Convert to Position tuples (x, z)
        positions: List[Position] = []
        for pos in reachable:
            position: Position = (pos["x"], pos["z"])
            positions.append(position)

        # Build map with edge threshold from settings
        edge_threshold = self._settings.memory.map_edge_distance_threshold
        self._spatial_map.build_from_positions(positions, edge_threshold)

        logger.info(
            f"Built spatial map with {len(self._spatial_map.nodes)} nodes "
            f"and {len(self._spatial_map.edges)} edges"
        )

        return self._spatial_map

    def find_object(self, description: str) -> Optional[ObjectInfo]:
        """
        Find an object matching the description.

        Uses CLIP to match the description against visible objects.

        Args:
            description: Text description of the object to find.

        Returns:
            ObjectInfo for the best matching object, or None if not found.
        """
        # Get visible objects from THOR
        visible_objects = self._get_visible_objects_from_thor()

        if not visible_objects:
            logger.info(f"No visible objects found")
            return None

        # Find best match using CLIP
        best_match = self._find_best_match_thor(description, visible_objects)

        if best_match is not None:
            logger.info(
                f"Found object: {best_match.object_type} "
                f"(similarity: {best_match.clip_similarity:.2f})"
            )

        return best_match

    def navigate_to(
        self,
        target_description: str,
        max_steps: int = 50,
        verify_callback: Optional[Callable[[], bool]] = None
    ) -> NavigationResult:
        """
        Navigate to a target object described by text.

        This method combines exploration and goal-directed navigation
        to find and reach a target object.

        Args:
            target_description: Text description of the target object.
            max_steps: Maximum number of navigation steps.
            verify_callback: Optional callback to verify arrival at target.

        Returns:
            NavigationResult with success status and final state.
        """
        # Ensure spatial map is initialized
        if self._spatial_map is None:
            self.build_spatial_map()

        # Get initial state
        initial_state = self._controller.get_current_state()
        initial_pos = initial_state.position
        self._current_position = (initial_pos["x"], initial_pos["z"])

        steps_taken = 0
        target_object: Optional[ObjectInfo] = None

        for step in range(max_steps):
            steps_taken = step + 1

            # Try to find the target
            target_object = self.find_object(target_description)

            if target_object is not None:
                # Check if we're close enough
                success_distance = self._settings.navigation.success_distance

                if target_object.distance <= success_distance:
                    logger.info(
                        f"Navigation successful: reached target in {steps_taken} steps"
                    )
                    return NavigationResult(
                        success=True,
                        final_position=dict(initial_state.position),
                        steps_taken=steps_taken,
                        distance_to_target=target_object.distance,
                        error=None
                    )

                # Try to move towards target
                target_pos = target_object.position
                move_result = self._move_towards(target_pos)

                if move_result.success:
                    # Update current position
                    state = self._controller.get_current_state()
                    self._current_position = (state.position["x"], state.position["z"])
                    continue

            # Target not found or can't move towards it - explore
            explore_result = self._explore_step()

            if explore_result.success:
                state = self._controller.get_current_state()
                self._current_position = (state.position["x"], state.position["z"])
            else:
                # Try a different direction
                self._exploration_direction = (self._exploration_direction + 1) % 4

        # Ran out of steps
        final_state = self._controller.get_current_state()
        final_distance = target_object.distance if target_object else float('inf')

        return NavigationResult(
            success=False,
            final_position=dict(final_state.position),
            steps_taken=steps_taken,
            distance_to_target=final_distance,
            error=f"Max steps ({max_steps}) reached without finding target"
        )

    def _get_visible_objects_from_thor(self) -> List[ObjectInfo]:
        """
        Get visible objects from the THOR environment.

        Returns:
            List of ObjectInfo for visible objects.
        """
        objects: List[ObjectInfo] = []

        # Get current observation
        state = self._controller.get_current_state()

        # Step to get fresh observation
        # In mock mode, we need to query for visible objects
        # In real THOR, objects come from event metadata

        # Since controller doesn't expose objects directly,
        # we use the visible_objects from the last observation
        # For now, return mock objects for testing

        # In a full implementation, this would:
        # 1. Get the last event from controller
        # 2. Extract visible objects from metadata
        # 3. Create ObjectInfo for each

        # Mock implementation for testing
        # Real implementation would query THOR metadata
        return objects

    def _find_best_match_thor(
        self,
        description: str,
        objects: List[ObjectInfo]
    ) -> Optional[ObjectInfo]:
        """
        Find the best matching object using CLIP.

        Args:
            description: Text description to match.
            objects: List of ObjectInfo candidates.

        Returns:
            Best matching ObjectInfo, or None if no match above threshold.
        """
        if not objects:
            return None

        # Encode the description
        text_feature = self._visual_encoder.encode_text(description)

        best_match: Optional[ObjectInfo] = None
        best_similarity = -1.0

        for obj in objects:
            # Calculate similarity (would need image features in real implementation)
            # For now, use object type matching with CLIP
            type_feature = self._visual_encoder.encode_text(obj.object_type)
            similarity = self._visual_encoder.compute_similarity(text_feature, type_feature)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = obj
                obj.clip_similarity = similarity

        threshold = self._settings.perception.clip_match_threshold
        if best_similarity >= threshold:
            return best_match

        return None

    def _move_towards(self, target_pos: Dict[str, float]) -> StepResult:
        """
        Move the agent towards a target position.

        Uses simple heuristic: rotate to face target, then move forward.

        Args:
            target_pos: Target position with x, y, z keys.

        Returns:
            StepResult from the movement action.
        """
        # Get current state
        current_state = self._controller.get_current_state()
        current_pos = current_state.position
        current_rot = current_state.rotation

        # Calculate angle to target
        dx = target_pos["x"] - current_pos["x"]
        dz = target_pos["z"] - current_pos["z"]
        target_angle = math.degrees(math.atan2(dx, dz))

        # Normalize to [0, 360)
        target_angle = target_angle % 360
        current_angle = current_rot["y"] % 360

        # Calculate rotation difference
        angle_diff = (target_angle - current_angle + 180) % 360 - 180

        # Rotate if needed (with tolerance)
        rotation_threshold = 45.0  # degrees

        if abs(angle_diff) > rotation_threshold:
            if angle_diff > 0:
                return self._controller.step("RotateLeft")
            else:
                return self._controller.step("RotateRight")

        # Move forward
        return self._controller.step("MoveAhead")

    def _explore_step(self) -> StepResult:
        """
        Take an exploration step to discover new areas.

        Uses a simple exploration strategy: rotate and move to
        unvisited directions.

        Returns:
            StepResult from the exploration action.
        """
        # Try different actions in order
        # Simple strategy: rotate to face unexplored direction, then move

        exploration_actions = [
            "MoveAhead",
            "RotateLeft",
            "MoveAhead",
            "RotateRight",
            "MoveAhead"
        ]

        action_idx = self._exploration_direction % len(exploration_actions)
        action = exploration_actions[action_idx]

        result = self._controller.step(action)

        if not result.success:
            # If movement failed, try rotating
            self._exploration_direction += 1
            return self._controller.step("RotateRight")

        return result

    def reset(self, scene_name: Optional[str] = None) -> None:
        """
        Reset the navigator for a new episode.

        Clears the spatial map and working memory.

        Args:
            scene_name: Optional scene name to reset to.
        """
        self._spatial_map = None
        self._working_memory.reset()
        self._current_position = (0.0, 0.0)
        self._visited_positions = []
        self._exploration_direction = 0

        logger.info("NavigatorAgent reset")

    def get_spatial_map(self) -> Optional[TopologicalMap]:
        """
        Get the current spatial map.

        Returns:
            TopologicalMap if built, None otherwise.
        """
        return self._spatial_map

    def get_current_position(self) -> Position:
        """
        Get the agent's current position.

        Returns:
            Current position as (x, z) tuple.
        """
        return self._current_position

    def navigate_to_target(
        self,
        target: str,
        max_steps: int = 50
    ) -> Dict[str, Any]:
        """
        Navigate to a target object (wrapper for interactive mode).

        Args:
            target: Target object description.
            max_steps: Maximum navigation steps.

        Returns:
            Dict with success, distance, steps, and error.
        """
        result = self.navigate_to(target, max_steps=max_steps)
        return {
            "success": result.success,
            "distance": result.distance_to_target,
            "steps": result.steps_taken,
            "error": result.error
        }

    def explore(self, max_steps: int = 20) -> Dict[str, Any]:
        """
        Explore the environment.

        Args:
            max_steps: Maximum exploration steps.

        Returns:
            Dict with steps taken and positions visited.
        """
        positions_visited = []
        steps_taken = 0

        for _ in range(max_steps):
            result = self._explore_step()
            steps_taken += 1

            state = self._controller.get_current_state()
            pos = (state.position["x"], state.position["z"])
            positions_visited.append(pos)

            if not result.success:
                # Try rotating
                self._controller.step("RotateRight")

        return {
            "steps": steps_taken,
            "positions": positions_visited
        }
