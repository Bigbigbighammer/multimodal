"""Working memory for task state tracking in the vision-language embodied agent.

This module provides data structures for tracking the agent's current task state,
including action history, subgoal progress, and relevant context for LLM planning.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ActionRecord:
    """A record of a single action taken by the agent.

    Attributes:
        action: The name of the action taken (e.g., "MoveAhead", "Pickup").
        success: Whether the action was successful.
        step: The step number when this action was taken.
        observation_summary: A brief summary of what was observed after the action.
    """

    action: str
    success: bool
    step: int
    observation_summary: str = ""


@dataclass
class SubGoal:
    """A subgoal within the agent's current plan.

    Attributes:
        description: Human-readable description of the subgoal.
        type: Type of subgoal (e.g., "search", "navigate", "interact").
        completed: Whether this subgoal has been completed.
        target_object: The target object for this subgoal, if applicable.
    """

    description: str
    type: str
    completed: bool = False
    target_object: Optional[str] = None


@dataclass
class WorkingMemory:
    """Working memory for tracking task state during execution.

    This class maintains the agent's current understanding of the task,
    including the original instruction, current plan, action history,
    and various state tracking fields.

    Attributes:
        max_steps: Maximum number of action steps to keep before summarization.
        original_instruction: The original task instruction from the user.
        current_plan: List of subgoals representing the current plan.
        plan_revision_count: Number of times the plan has been revised.
        action_history: List of recent actions taken by the agent.
        held_object: The object currently held by the agent, if any.
        current_step: The current step number in execution.
        current_visible_objects: List of objects currently visible to the agent.
        target_object_location: Location of the target object if known.
        current_goal: The current subgoal being worked on.
    """

    max_steps: int = 10
    original_instruction: str = ""
    current_plan: List[SubGoal] = field(default_factory=list)
    plan_revision_count: int = 0
    action_history: List[ActionRecord] = field(default_factory=list)
    held_object: Optional[str] = None
    current_step: int = 0
    current_visible_objects: List[str] = field(default_factory=list)
    target_object_location: Optional[Tuple[float, float]] = None
    current_goal: Optional[SubGoal] = None

    def set_instruction(self, instruction: str) -> None:
        """Set the original task instruction.

        Args:
            instruction: The task instruction from the user.
        """
        self.original_instruction = instruction

    def set_plan(self, plan: List[SubGoal]) -> None:
        """Set a new plan for the current task.

        Increments the plan revision count.

        Args:
            plan: List of subgoals representing the new plan.
        """
        self.current_plan = plan
        self.plan_revision_count += 1

    def add_action(
        self, action: str, success: bool, observation_summary: str = ""
    ) -> None:
        """Add an action to the action history.

        Increments the current step counter and creates an ActionRecord.

        Args:
            action: The name of the action taken.
            success: Whether the action was successful.
            observation_summary: Brief summary of observations after the action.
        """
        self.current_step += 1
        record = ActionRecord(
            action=action,
            success=success,
            step=self.current_step,
            observation_summary=observation_summary,
        )
        self.action_history.append(record)

    def set_current_goal(self, goal: SubGoal) -> None:
        """Set the current goal being worked on.

        Args:
            goal: The subgoal to set as the current goal.
        """
        self.current_goal = goal

    def mark_subgoal_completed(self, index: int) -> None:
        """Mark a subgoal as completed by its index.

        Args:
            index: The index of the subgoal in current_plan to mark complete.
        """
        if 0 <= index < len(self.current_plan):
            self.current_plan[index].completed = True

    def get_current_subgoal(self) -> Optional[SubGoal]:
        """Get the first incomplete subgoal from the current plan.

        Returns:
            The first incomplete SubGoal, or None if all subgoals are complete.
        """
        for subgoal in self.current_plan:
            if not subgoal.completed:
                return subgoal
        return None

    def should_summarize(self) -> bool:
        """Check if action history should be summarized.

        Returns:
            True if the number of actions in history equals or exceeds max_steps.
        """
        return len(self.action_history) >= self.max_steps

    def get_context_for_llm(self) -> Dict[str, Any]:
        """Generate a context dictionary for LLM planning.

        Returns:
            A dictionary containing relevant context for the LLM, including:
            - original_instruction: The task instruction
            - current_plan: List of subgoal descriptions
            - action_history: Recent action records
            - held_object: Currently held object
            - current_step: Step number
            - current_visible_objects: Visible objects
            - current_goal: The current subgoal being worked on
        """
        return {
            "original_instruction": self.original_instruction,
            "current_plan": [
                {
                    "description": sg.description,
                    "type": sg.type,
                    "completed": sg.completed,
                    "target_object": sg.target_object,
                }
                for sg in self.current_plan
            ],
            "action_history": [
                {
                    "action": r.action,
                    "success": r.success,
                    "step": r.step,
                    "observation_summary": r.observation_summary,
                }
                for r in self.action_history
            ],
            "held_object": self.held_object,
            "current_step": self.current_step,
            "current_visible_objects": self.current_visible_objects,
            "current_goal": (
                {
                    "description": self.current_goal.description,
                    "type": self.current_goal.type,
                    "completed": self.current_goal.completed,
                    "target_object": self.current_goal.target_object,
                }
                if self.current_goal
                else None
            ),
        }

    def summarize_history(self) -> Dict[str, Any]:
        """Create a summary of the action history.

        Returns:
            A dictionary with summary statistics including:
            - total_actions: Total number of actions
            - successful_actions: Number of successful actions
            - failed_actions: Number of failed actions
            - last_n_actions: The most recent actions (up to 5)
        """
        successful = sum(1 for r in self.action_history if r.success)
        failed = len(self.action_history) - successful

        return {
            "total_actions": len(self.action_history),
            "successful_actions": successful,
            "failed_actions": failed,
            "last_n_actions": [
                {"action": r.action, "success": r.success}
                for r in self.action_history[-5:]
            ],
        }

    def reset(self) -> None:
        """Reset all memory state to initial values.

        Clears instruction, plan, action history, and all tracking fields.
        """
        self.original_instruction = ""
        self.current_plan = []
        self.plan_revision_count = 0
        self.action_history = []
        self.held_object = None
        self.current_step = 0
        self.current_visible_objects = []
        self.target_object_location = None
        self.current_goal = None
