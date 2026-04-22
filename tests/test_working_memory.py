"""Tests for WorkingMemory and related dataclasses."""

import pytest
from dataclasses import dataclass
from typing import Optional

from src.memory.working_memory import ActionRecord, SubGoal, WorkingMemory
from src.config.settings import Settings


class TestActionRecord:
    """Tests for ActionRecord dataclass."""

    def test_action_record_creation(self):
        """Test creating an ActionRecord with all fields."""
        record = ActionRecord(
            action="MoveAhead",
            success=True,
            step=1,
            observation_summary="Saw a chair and a table"
        )
        assert record.action == "MoveAhead"
        assert record.success is True
        assert record.step == 1
        assert record.observation_summary == "Saw a chair and a table"

    def test_action_record_defaults(self):
        """Test ActionRecord with minimal required fields."""
        record = ActionRecord(
            action="RotateLeft",
            success=False,
            step=2,
            observation_summary=""
        )
        assert record.action == "RotateLeft"
        assert record.success is False
        assert record.step == 2
        assert record.observation_summary == ""


class TestSubGoal:
    """Tests for SubGoal dataclass."""

    def test_subgoal_creation(self):
        """Test creating a SubGoal with all fields."""
        subgoal = SubGoal(
            description="Find the red mug",
            type="search",
            completed=False,
            target_object="Mug"
        )
        assert subgoal.description == "Find the red mug"
        assert subgoal.type == "search"
        assert subgoal.completed is False
        assert subgoal.target_object == "Mug"

    def test_subgoal_with_none_target(self):
        """Test SubGoal with no target object."""
        subgoal = SubGoal(
            description="Navigate to kitchen",
            type="navigate",
            completed=False,
            target_object=None
        )
        assert subgoal.description == "Navigate to kitchen"
        assert subgoal.target_object is None


class TestWorkingMemory:
    """Tests for WorkingMemory dataclass."""

    @pytest.fixture
    def settings(self):
        """Create default settings for testing."""
        return Settings()

    @pytest.fixture
    def memory(self, settings):
        """Create a WorkingMemory instance for testing."""
        return WorkingMemory(max_steps=settings.memory.working_memory_max_steps)

    def test_empty_memory(self, memory):
        """Test newly created memory is empty."""
        assert memory.original_instruction == ""
        assert memory.current_plan == []
        assert memory.plan_revision_count == 0
        assert memory.action_history == []
        assert memory.held_object is None
        assert memory.current_step == 0
        assert memory.current_visible_objects == []
        assert memory.target_object_location is None
        assert memory.current_goal is None

    def test_set_instruction(self, memory):
        """Test setting the original instruction."""
        instruction = "Find the red mug and put it on the table"
        memory.set_instruction(instruction)
        assert memory.original_instruction == instruction

    def test_set_plan(self, memory):
        """Test setting a plan."""
        plan = [
            SubGoal("Find red mug", "search", False, "Mug"),
            SubGoal("Pick up mug", "interact", False, "Mug"),
            SubGoal("Put mug on table", "interact", False, "Table")
        ]
        memory.set_plan(plan)
        assert len(memory.current_plan) == 3
        assert memory.current_plan[0].description == "Find red mug"
        assert memory.plan_revision_count == 1

    def test_add_action(self, memory):
        """Test adding actions to history."""
        memory.add_action("MoveAhead", True, "Moved forward successfully")
        assert len(memory.action_history) == 1
        assert memory.action_history[0].action == "MoveAhead"
        assert memory.action_history[0].success is True
        assert memory.current_step == 1

        memory.add_action("RotateLeft", True, "Rotated left")
        assert len(memory.action_history) == 2
        assert memory.current_step == 2

    def test_should_summarize_threshold(self, settings):
        """Test that should_summarize returns True when threshold is reached."""
        # Create memory with max_steps=3 for testing
        memory = WorkingMemory(max_steps=3)

        # Add actions up to threshold
        memory.add_action("MoveAhead", True, "step 1")
        assert memory.should_summarize() is False

        memory.add_action("MoveAhead", True, "step 2")
        assert memory.should_summarize() is False

        memory.add_action("MoveAhead", True, "step 3")
        assert memory.should_summarize() is True

    def test_should_summarize_default_settings(self, memory):
        """Test should_summarize with default settings (max_steps=10)."""
        # Add 10 actions
        for i in range(10):
            memory.add_action("MoveAhead", True, f"step {i+1}")

        assert memory.should_summarize() is True

    def test_get_context_for_llm(self, memory):
        """Test generating context for LLM."""
        memory.set_instruction("Find the mug")
        memory.set_plan([
            SubGoal("Find mug", "search", False, "Mug")
        ])
        memory.add_action("MoveAhead", True, "Saw a table")
        memory.set_current_goal(SubGoal("Find mug", "search", False, "Mug"))

        context = memory.get_context_for_llm()

        assert "original_instruction" in context
        assert context["original_instruction"] == "Find the mug"
        assert "current_plan" in context
        assert "action_history" in context
        assert len(context["action_history"]) == 1
        assert "current_goal" in context
        assert context["current_goal"]["description"] == "Find mug"
        assert "current_step" in context
        assert context["current_step"] == 1

    def test_subgoal_tracking(self, memory):
        """Test setting and completing subgoals."""
        subgoal1 = SubGoal("Find mug", "search", False, "Mug")
        subgoal2 = SubGoal("Pick up mug", "interact", False, "Mug")

        # Set plan with subgoals
        memory.set_plan([subgoal1, subgoal2])

        # Get current subgoal (should be first incomplete)
        current = memory.get_current_subgoal()
        assert current is not None
        assert current.description == "Find mug"

        # Mark first subgoal as completed
        memory.mark_subgoal_completed(0)
        assert memory.current_plan[0].completed is True

        # Current subgoal should now be second
        current = memory.get_current_subgoal()
        assert current is not None
        assert current.description == "Pick up mug"

    def test_set_current_goal(self, memory):
        """Test setting the current goal."""
        goal = SubGoal("Navigate to kitchen", "navigate", False, None)
        memory.set_current_goal(goal)
        assert memory.current_goal is not None
        assert memory.current_goal.description == "Navigate to kitchen"

    def test_reset(self, memory):
        """Test resetting memory to initial state."""
        memory.set_instruction("Find the mug")
        memory.add_action("MoveAhead", True, "step 1")
        memory.set_plan([SubGoal("Find mug", "search", False, "Mug")])
        memory.set_current_goal(SubGoal("Find mug", "search", False, "Mug"))

        memory.reset()

        assert memory.original_instruction == ""
        assert memory.current_plan == []
        assert memory.plan_revision_count == 0
        assert memory.action_history == []
        assert memory.held_object is None
        assert memory.current_step == 0
        assert memory.current_visible_objects == []
        assert memory.target_object_location is None
        assert memory.current_goal is None

    def test_summarize_history(self, memory):
        """Test history summarization."""
        # Add some actions
        for i in range(5):
            memory.add_action("MoveAhead", True, f"step {i+1}")

        # Get summary
        summary = memory.summarize_history()
        assert "total_actions" in summary
        assert summary["total_actions"] == 5
        assert "successful_actions" in summary
        assert summary["successful_actions"] == 5
        assert "failed_actions" in summary
        assert summary["failed_actions"] == 0

    def test_summarize_history_with_failures(self, memory):
        """Test history summarization with failed actions."""
        memory.add_action("MoveAhead", True, "step 1")
        memory.add_action("Pickup", False, "Object not reachable")
        memory.add_action("MoveAhead", True, "step 2")

        summary = memory.summarize_history()
        assert summary["total_actions"] == 3
        assert summary["successful_actions"] == 2
        assert summary["failed_actions"] == 1

    def test_all_subgoals_completed(self, memory):
        """Test when all subgoals are completed."""
        memory.set_plan([
            SubGoal("Find mug", "search", True, "Mug"),
            SubGoal("Pick up mug", "interact", True, "Mug")
        ])

        current = memory.get_current_subgoal()
        assert current is None  # No incomplete subgoals

    def test_held_object_tracking(self, memory):
        """Test tracking held object."""
        assert memory.held_object is None

        memory.held_object = "Mug"
        assert memory.held_object == "Mug"

        memory.held_object = None
        assert memory.held_object is None

    def test_visible_objects_tracking(self, memory):
        """Test tracking visible objects."""
        memory.current_visible_objects = ["Chair", "Table", "Mug"]
        assert len(memory.current_visible_objects) == 3
        assert "Mug" in memory.current_visible_objects

    def test_target_object_location_tracking(self, memory):
        """Test tracking target object location."""
        assert memory.target_object_location is None

        memory.target_object_location = (1.5, 2.0)
        assert memory.target_object_location == (1.5, 2.0)

    def test_get_context_for_llm_empty_memory(self, memory):
        """Test context generation with empty memory."""
        context = memory.get_context_for_llm()

        assert context["original_instruction"] == ""
        assert context["current_plan"] == []
        assert context["action_history"] == []
        assert context["current_goal"] is None
        assert context["held_object"] is None
        assert context["current_step"] == 0
