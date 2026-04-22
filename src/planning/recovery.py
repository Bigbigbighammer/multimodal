"""
Recovery strategies for the Vision-Language Embodied Agent.

This module provides recovery mechanisms when subgoals fail,
including retry, replan, and search behaviors.

Key components:
- RecoveryAction: Enum defining available recovery actions
- RecoveryResult: Dataclass for recovery attempt results
- RecoveryStrategy: Class that selects appropriate recovery strategies
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    """
    Available recovery actions when a subgoal fails.

    Actions are ordered by increasing cost/complexity:
    - RETRY: Try the same action again (e.g., after rotation)
    - LOCAL_SEARCH: Search nearby for the target
    - REPLAN: Decompose the task again with updated context
    - ABORT: Give up on the task
    """
    RETRY = "retry"
    LOCAL_SEARCH = "local_search"
    REPLAN = "replan"
    ABORT = "abort"


@dataclass
class RecoveryResult:
    """
    Result of a recovery attempt.

    Attributes:
        action: The recovery action that was attempted.
        success: Whether the recovery succeeded.
        message: Human-readable message about the recovery.
        new_subgoal: New subgoal if replanning, None otherwise.
    """
    action: RecoveryAction
    success: bool
    message: str
    new_subgoal: Optional[str] = None


class RecoveryStrategy:
    """
    Selects and executes recovery strategies when subgoals fail.

    The strategy selection is based on:
    - Number of retry attempts for the current subgoal
    - Total number of replans for the task
    - Type of failure encountered

    Example:
        >>> strategy = RecoveryStrategy(max_retries=3, max_replans=2)
        >>> action = strategy.select_strategy(retry_count=1, replan_count=0)
        >>> print(action)  # RecoveryAction.RETRY
    """

    def __init__(
        self,
        max_retries: int = 3,
        max_replans: int = 2,
        retry_rotation_degrees: float = 90.0,
        local_search_radius: float = 2.0
    ):
        """
        Initialize the RecoveryStrategy.

        Args:
            max_retries: Maximum retry attempts per subgoal.
            max_replans: Maximum global replans for the task.
            retry_rotation_degrees: Degrees to rotate before retry.
            local_search_radius: Radius for local search in meters.
        """
        self.max_retries = max_retries
        self.max_replans = max_replans
        self.retry_rotation_degrees = retry_rotation_degrees
        self.local_search_radius = local_search_radius

    def select_strategy(
        self,
        retry_count: int,
        replan_count: int,
        failure_type: str = "unknown"
    ) -> RecoveryAction:
        """
        Select the appropriate recovery action based on failure context.

        Strategy selection logic:
        1. If retries exhausted, try local search
        2. If local search exhausted, try replan
        3. If replans exhausted, abort

        Args:
            retry_count: Number of retries attempted for current subgoal.
            replan_count: Number of global replans attempted.
            failure_type: Type of failure (e.g., "navigation", "manipulation").

        Returns:
            RecoveryAction to attempt next.
        """
        # Check if we've exhausted all recovery options
        if replan_count >= self.max_replans:
            logger.warning(f"All recovery options exhausted: replan_count={replan_count}")
            return RecoveryAction.ABORT

        # If we haven't exhausted retries, try again
        if retry_count < self.max_retries:
            logger.info(
                f"Selecting RETRY: retry_count={retry_count}/{self.max_retries}"
            )
            return RecoveryAction.RETRY

        # If retries exhausted but replans available, try local search first
        if retry_count >= self.max_retries and replan_count < self.max_replans:
            # For navigation failures, local search might help
            if failure_type in ("navigation", "object_not_found"):
                logger.info("Selecting LOCAL_SEARCH: navigation failure with retries exhausted")
                return RecoveryAction.LOCAL_SEARCH

            # For other failures, go straight to replan
            logger.info(f"Selecting REPLAN: {failure_type} failure with retries exhausted")
            return RecoveryAction.REPLAN

        # Default to abort if we somehow get here
        return RecoveryAction.ABORT

    def get_retry_actions(self, current_rotation: float = 0.0) -> List[str]:
        """
        Get actions to execute for a retry recovery.

        Args:
            current_rotation: Current agent rotation in degrees.

        Returns:
            List of action names to execute for retry.
        """
        # Simple retry: rotate and try again
        # This helps when the agent was facing the wrong direction
        return ["RotateLeft"]  # Rotate to get a different view

    def get_local_search_actions(
        self,
        search_radius: Optional[float] = None
    ) -> List[Tuple[str, dict]]:
        """
        Get actions for local search recovery.

        Args:
            search_radius: Search radius in meters. Uses default if None.

        Returns:
            List of (action_name, kwargs) tuples for local search.
        """
        radius = search_radius or self.local_search_radius

        # Local search pattern: look around and move to nearby positions
        actions = [
            ("RotateLeft", {}),
            ("RotateLeft", {}),
            ("RotateLeft", {}),
            ("RotateLeft", {}),  # Full 360 rotation
            ("MoveAhead", {}),
            ("RotateLeft", {}),
            ("RotateLeft", {}),
            ("MoveAhead", {}),
        ]
        return actions

    def should_replan(
        self,
        retry_count: int,
        replan_count: int,
        failure_context: Optional[dict] = None
    ) -> bool:
        """
        Determine if replanning is needed.

        Args:
            retry_count: Number of retries for current subgoal.
            replan_count: Number of global replans.
            failure_context: Optional context about the failure.

        Returns:
            True if replanning is recommended.
        """
        if replan_count >= self.max_replans:
            return False

        # Replan if retries are exhausted
        if retry_count >= self.max_retries:
            return True

        # Could add more sophisticated logic based on failure_context
        return False
