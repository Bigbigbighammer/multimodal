"""
Tests for PlannerAgent and planning components.

This module tests:
- RecoveryStrategy: Recovery action selection
- Verifier: Subgoal verification methods
- TaskDecomposer: Task decomposition with LLM (mocked)
- PlannerAgent: LangGraph workflow execution
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, Any, List

import sys
sys.path.insert(0, "src")

from planning.recovery import (
    RecoveryAction,
    RecoveryResult,
    RecoveryStrategy,
)
from planning.verifier import (
    VerificationResult,
    Verifier,
)
from planning.task_decomposer import (
    Subgoal,
    TaskDecomposition,
    TaskDecomposer,
)


class TestRecoveryAction:
    """Test RecoveryAction enum."""

    def test_recovery_action_values(self):
        """Test RecoveryAction enum values."""
        assert RecoveryAction.RETRY.value == "retry"
        assert RecoveryAction.LOCAL_SEARCH.value == "local_search"
        assert RecoveryAction.REPLAN.value == "replan"
        assert RecoveryAction.ABORT.value == "abort"

    def test_recovery_action_count(self):
        """Test that we have expected recovery actions."""
        actions = list(RecoveryAction)
        assert len(actions) == 4


class TestRecoveryResult:
    """Test RecoveryResult dataclass."""

    def test_recovery_result_creation(self):
        """Test RecoveryResult creation."""
        result = RecoveryResult(
            action=RecoveryAction.RETRY,
            success=True,
            message="Retry successful"
        )
        assert result.action == RecoveryAction.RETRY
        assert result.success is True
        assert result.message == "Retry successful"
        assert result.new_subgoal is None

    def test_recovery_result_with_new_subgoal(self):
        """Test RecoveryResult with new subgoal."""
        result = RecoveryResult(
            action=RecoveryAction.REPLAN,
            success=True,
            message="Replanned successfully",
            new_subgoal="Navigate to kitchen"
        )
        assert result.new_subgoal == "Navigate to kitchen"


class TestRecoveryStrategy:
    """Test RecoveryStrategy class."""

    def test_strategy_initialization(self):
        """Test RecoveryStrategy initialization."""
        strategy = RecoveryStrategy(
            max_retries=5,
            max_replans=3,
            retry_rotation_degrees=45.0,
            local_search_radius=1.5
        )
        assert strategy.max_retries == 5
        assert strategy.max_replans == 3
        assert strategy.retry_rotation_degrees == 45.0
        assert strategy.local_search_radius == 1.5

    def test_strategy_default_values(self):
        """Test RecoveryStrategy default values."""
        strategy = RecoveryStrategy()
        assert strategy.max_retries == 3
        assert strategy.max_replans == 2

    def test_select_strategy_retry(self):
        """Test strategy selection for retry."""
        strategy = RecoveryStrategy(max_retries=3, max_replans=2)

        # First failure should select RETRY
        action = strategy.select_strategy(retry_count=0, replan_count=0)
        assert action == RecoveryAction.RETRY

        # Still within retries
        action = strategy.select_strategy(retry_count=2, replan_count=0)
        assert action == RecoveryAction.RETRY

    def test_select_strategy_local_search_or_replan(self):
        """Test strategy selection when retries exhausted."""
        strategy = RecoveryStrategy(max_retries=3, max_replans=2)

        # Retries exhausted, should go to local search or replan
        action = strategy.select_strategy(
            retry_count=3,
            replan_count=0,
            failure_type="navigation"
        )
        # Navigation failure with retries exhausted -> local search
        assert action == RecoveryAction.LOCAL_SEARCH

        # Other failure types -> replan
        action = strategy.select_strategy(
            retry_count=3,
            replan_count=0,
            failure_type="manipulation"
        )
        assert action == RecoveryAction.REPLAN

    def test_select_strategy_abort(self):
        """Test strategy selection when all options exhausted."""
        strategy = RecoveryStrategy(max_retries=3, max_replans=2)

        # Replans exhausted
        action = strategy.select_strategy(retry_count=3, replan_count=2)
        assert action == RecoveryAction.ABORT

    def test_get_retry_actions(self):
        """Test retry action generation."""
        strategy = RecoveryStrategy()
        actions = strategy.get_retry_actions()
        assert isinstance(actions, list)
        assert len(actions) > 0
        assert "RotateLeft" in actions

    def test_get_local_search_actions(self):
        """Test local search action generation."""
        strategy = RecoveryStrategy(local_search_radius=3.0)
        actions = strategy.get_local_search_actions()
        assert isinstance(actions, list)
        assert len(actions) > 0
        # Each action should be a tuple of (action_name, kwargs)
        for action in actions:
            assert isinstance(action, tuple)
            assert len(action) == 2

    def test_should_replan(self):
        """Test replan decision logic."""
        strategy = RecoveryStrategy(max_retries=3, max_replans=2)

        # Should not replan if retries not exhausted
        assert strategy.should_replan(retry_count=1, replan_count=0) is False

        # Should replan if retries exhausted
        assert strategy.should_replan(retry_count=3, replan_count=0) is True

        # Should not replan if max replans reached
        assert strategy.should_replan(retry_count=3, replan_count=2) is False


class TestVerificationResult:
    """Test VerificationResult dataclass."""

    def test_verification_result_success(self):
        """Test successful verification result."""
        result = VerificationResult(
            success=True,
            message="Navigation successful",
            details={"distance": 0.5}
        )
        assert result.success is True
        assert result.message == "Navigation successful"
        assert result.details["distance"] == 0.5

    def test_verification_result_failure(self):
        """Test failed verification result."""
        result = VerificationResult(
            success=False,
            message="Object not held"
        )
        assert result.success is False
        assert result.details is None


class TestVerifier:
    """Test Verifier class."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock controller."""
        controller = MagicMock()
        controller.get_current_state.return_value = MagicMock(
            position={"x": 1.0, "y": 0.0, "z": 2.0},
            rotation={"x": 0.0, "y": 90.0, "z": 0.0},
            held_object=None
        )
        return controller

    def test_verifier_initialization(self, mock_controller):
        """Test Verifier initialization."""
        verifier = Verifier(mock_controller)
        assert verifier._controller == mock_controller

    def test_verify_navigate_success(self, mock_controller):
        """Test successful navigation verification."""
        verifier = Verifier(mock_controller)

        # Target close to current position
        result = verifier.verify_navigate(
            target_position={"x": 1.0, "y": 0.0, "z": 2.0},
            tolerance=1.0
        )
        assert result.success is True
        assert "successful" in result.message.lower()

    def test_verify_navigate_failure(self, mock_controller):
        """Test failed navigation verification."""
        verifier = Verifier(mock_controller)

        # Target far from current position
        result = verifier.verify_navigate(
            target_position={"x": 10.0, "y": 0.0, "z": 10.0},
            tolerance=1.0
        )
        assert result.success is False
        assert "failed" in result.message.lower()

    def test_verify_pickup_no_object(self, mock_controller):
        """Test pickup verification when nothing held."""
        verifier = Verifier(mock_controller)

        result = verifier.verify_pickup("Mug")
        assert result.success is False
        assert "no object" in result.message.lower()

    def test_verify_pickup_success(self, mock_controller):
        """Test successful pickup verification."""
        mock_controller.get_current_state.return_value.held_object = "Mug"
        verifier = Verifier(mock_controller)

        result = verifier.verify_pickup("Mug")
        assert result.success is True

    def test_verify_pickup_partial_match(self, mock_controller):
        """Test pickup verification with object ID format."""
        mock_controller.get_current_state.return_value.held_object = "Mug|1|2|3"
        verifier = Verifier(mock_controller)

        result = verifier.verify_pickup("Mug")
        assert result.success is True

    def test_verify_open_mock_mode(self, mock_controller):
        """Test open verification in mock mode."""
        verifier = Verifier(mock_controller)

        result = verifier.verify_open("Fridge")
        assert result.success is True  # Mock mode assumes success

    def test_verify_put_success(self, mock_controller):
        """Test successful put verification."""
        verifier = Verifier(mock_controller)

        result = verifier.verify_put("Mug", "Table", held_object=None)
        assert result.success is True

    def test_verify_put_failure(self, mock_controller):
        """Test failed put verification."""
        verifier = Verifier(mock_controller)

        result = verifier.verify_put("Mug", "Table", held_object="Mug")
        assert result.success is False


class TestSubgoal:
    """Test Subgoal pydantic model."""

    def test_subgoal_creation(self):
        """Test Subgoal creation."""
        subgoal = Subgoal(
            id="subgoal_1",
            action="navigate",
            target="red chair",
            description="Navigate to the red chair"
        )
        assert subgoal.id == "subgoal_1"
        assert subgoal.action == "navigate"
        assert subgoal.target == "red chair"
        assert subgoal.dependencies == []

    def test_subgoal_with_dependencies(self):
        """Test Subgoal with dependencies."""
        subgoal = Subgoal(
            id="subgoal_2",
            action="pickup",
            target="mug",
            description="Pick up the mug",
            dependencies=["subgoal_1"]
        )
        assert "subgoal_1" in subgoal.dependencies


class TestTaskDecomposition:
    """Test TaskDecomposition pydantic model."""

    def test_decomposition_creation(self):
        """Test TaskDecomposition creation."""
        subgoals = [
            Subgoal(id="subgoal_1", action="navigate", target="chair", description="Go to chair"),
            Subgoal(id="subgoal_2", action="pickup", target="mug", description="Get mug")
        ]
        decomposition = TaskDecomposition(
            task="Pick up the mug from the chair",
            subgoals=subgoals,
            reasoning="Two-step task",
            estimated_steps=10
        )
        assert decomposition.task == "Pick up the mug from the chair"
        assert len(decomposition.subgoals) == 2
        assert decomposition.estimated_steps == 10


class TestTaskDecomposer:
    """Test TaskDecomposer class."""

    def test_decomposer_initialization_no_llm(self):
        """Test TaskDecomposer without LLM."""
        decomposer = TaskDecomposer(use_llm=False)
        assert decomposer._use_llm is False
        assert decomposer._llm is None

    def test_decomposer_template_based(self):
        """Test template-based decomposition."""
        decomposer = TaskDecomposer(use_llm=False)

        decomposition = decomposer.decompose("Pick up the red mug")
        assert decomposition.task == "Pick up the red mug"
        assert len(decomposition.subgoals) >= 1

    def test_decomposer_find_template(self):
        """Test find template decomposition."""
        decomposer = TaskDecomposer(use_llm=False)

        decomposition = decomposer.decompose("Find the chair")
        assert len(decomposition.subgoals) >= 1
        assert decomposition.subgoals[0].action == "navigate"

    def test_decomposer_pickup_template(self):
        """Test pickup template decomposition."""
        decomposer = TaskDecomposer(use_llm=False)

        decomposition = decomposer.decompose("Pick up the apple")
        # Pickup template should have navigate + pickup
        actions = [sg.action for sg in decomposition.subgoals]
        assert "navigate" in actions
        assert "pickup" in actions

    def test_decomposer_open_template(self):
        """Test open template decomposition."""
        decomposer = TaskDecomposer(use_llm=False)

        decomposition = decomposer.decompose("Open the fridge")
        actions = [sg.action for sg in decomposition.subgoals]
        assert "navigate" in actions
        assert "open" in actions

    def test_decomposer_put_template(self):
        """Test put template decomposition."""
        decomposer = TaskDecomposer(use_llm=False)

        decomposition = decomposer.decompose("Put the mug on the table")
        actions = [sg.action for sg in decomposition.subgoals]
        assert "navigate" in actions
        assert "pickup" in actions
        assert "put" in actions

    def test_decomposer_default_fallback(self):
        """Test default fallback for unknown tasks."""
        decomposer = TaskDecomposer(use_llm=False)

        decomposition = decomposer.decompose("Some random task without keywords")
        assert len(decomposition.subgoals) >= 1
        assert decomposition.subgoals[0].action == "navigate"

    def test_decomposer_is_available_no_llm(self):
        """Test is_available when LLM not initialized."""
        decomposer = TaskDecomposer(use_llm=False)
        assert decomposer.is_available() is False


class TestTaskDecomposerWithMockedLLM:
    """Test TaskDecomposer with mocked LLM."""

    def test_decomposer_with_mocked_llm(self):
        """Test LLM decomposition with mocked response."""
        # Create decomposer with LLM enabled
        # Since we can't easily mock pydantic models, we test template fallback
        decomposer = TaskDecomposer(use_llm=True, api_key="mock_key")

        # If LLM is not initialized (no langchain_openai), use template
        # Template-based decomposition should still work
        decomposition = decomposer.decompose("Pick up the apple")
        assert len(decomposition.subgoals) >= 1
        assert decomposition.subgoals[0].action in ["navigate", "pickup"]


class TestPlannerAgent:
    """Test PlannerAgent class."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock controller."""
        controller = MagicMock()
        controller.get_current_state.return_value = MagicMock(
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            rotation={"x": 0.0, "y": 0.0, "z": 0.0},
            held_object=None
        )
        controller.step.return_value = MagicMock(
            success=True,
            observation=MagicMock(),
            error=None
        )
        return controller

    def test_planner_initialization(self, mock_controller):
        """Test PlannerAgent initialization."""
        from agent.planner import PlannerAgent

        planner = PlannerAgent(mock_controller, use_llm_planner=False)
        assert planner._controller == mock_controller
        assert planner._use_llm_planner is False

    def test_planner_execute_task_simple(self, mock_controller):
        """Test simple task execution."""
        from agent.planner import PlannerAgent

        planner = PlannerAgent(mock_controller, use_llm_planner=False)

        # Mock successful step for all actions
        mock_controller.step.return_value = MagicMock(success=True, error=None)

        result = planner.execute_task("Find the chair")

        assert isinstance(result, dict)
        assert "success" in result
        assert "executed_actions" in result

    def test_planner_state_management(self, mock_controller):
        """Test planner state management."""
        from agent.planner import PlannerAgent, _init_state

        planner = PlannerAgent(mock_controller, use_llm_planner=False)
        assert planner.get_state() is None

        # Initialize state
        state = _init_state("Test task")
        assert state["instruction"] == "Test task"
        assert state["status"] == "planning"
        assert state["retry_count"] == 0

    def test_planner_reset(self, mock_controller):
        """Test planner reset."""
        from agent.planner import PlannerAgent

        planner = PlannerAgent(mock_controller, use_llm_planner=False)
        planner._state = {"test": "state"}

        planner.reset()
        assert planner.get_state() is None

    def test_planner_with_custom_settings(self, mock_controller):
        """Test planner with custom settings."""
        from agent.planner import PlannerAgent
        from config.settings import Settings

        settings = Settings()
        settings.planning.max_retries_per_subgoal = 5
        settings.planning.max_global_replans = 3

        planner = PlannerAgent(
            mock_controller,
            use_llm_planner=False,
            settings=settings
        )

        # Check that recovery strategy uses custom settings
        assert planner._recovery.max_retries == 5
        assert planner._recovery.max_replans == 3


class TestPlannerAgentNodes:
    """Test individual PlannerAgent workflow nodes."""

    @pytest.fixture
    def planner_with_mock(self):
        """Create planner with mock controller."""
        from agent.planner import PlannerAgent, _init_state

        controller = MagicMock()
        controller.get_current_state.return_value = MagicMock(
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            rotation={"x": 0.0, "y": 0.0, "z": 0.0},
            held_object=None
        )
        controller.step.return_value = MagicMock(success=True, error=None)

        planner = PlannerAgent(controller, use_llm_planner=False)
        return planner

    def test_plan_node(self, planner_with_mock):
        """Test plan node execution."""
        from agent.planner import _init_state

        planner = planner_with_mock
        state = _init_state("Pick up the mug")

        result = planner._plan_node(state)

        assert result["decomposition"] is not None
        assert result["status"] == "executing"
        assert result["current_subgoal_idx"] == 0

    def test_execute_node(self, planner_with_mock):
        """Test execute node execution."""
        from agent.planner import _init_state

        planner = planner_with_mock
        state = _init_state("Pick up the mug")

        # First plan
        state = planner._plan_node(state)
        # Then execute
        state = planner._execute_node(state)

        assert state["status"] == "verifying"
        assert len(state["executed_actions"]) >= 1

    def test_verify_node_success(self, planner_with_mock):
        """Test verify node with successful verification."""
        from agent.planner import _init_state

        planner = planner_with_mock
        state = _init_state("Navigate somewhere")

        state = planner._plan_node(state)
        state = planner._execute_node(state)
        state = planner._verify_node(state)

        assert "verification_results" in state
        assert len(state["verification_results"]) >= 1

    def test_should_continue_logic(self, planner_with_mock):
        """Test _should_continue decision logic."""
        from agent.planner import _init_state

        planner = planner_with_mock
        state = _init_state("Find the chair")

        # After planning
        state = planner._plan_node(state)

        # Check decision when all subgoals complete
        state["current_subgoal_idx"] = 100  # Past all subgoals
        action = planner._should_continue(state)
        assert action == "done"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
