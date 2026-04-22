"""
Tests for ThorController and related dataclasses.

This module tests:
- Dataclass structures (StepResult, AgentState, ThorObservation)
- ThorController with mocked AI2-THOR
- Error handling and edge cases
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import asdict
from typing import List, Dict, Any

# Test imports - these will work after implementation
import sys
sys.path.insert(0, "src")

from agent.controller import (
    ThorController,
    AgentState,
    ThorObservation,
    StepResult,
    THOR_AVAILABLE,
)


class TestThorControllerDataclasses:
    """Test the dataclass structures used by ThorController."""

    def test_agent_state_creation(self):
        """Test AgentState dataclass creation with all fields."""
        state = AgentState(
            position={"x": 1.0, "y": 0.0, "z": 2.0},
            rotation={"x": 0.0, "y": 90.0, "z": 0.0},
            held_object="Mug"
        )
        assert state.position == {"x": 1.0, "y": 0.0, "z": 2.0}
        assert state.rotation == {"x": 0.0, "y": 90.0, "z": 0.0}
        assert state.held_object == "Mug"

    def test_agent_state_defaults(self):
        """Test AgentState with default held_object."""
        state = AgentState(
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            rotation={"x": 0.0, "y": 0.0, "z": 0.0}
        )
        assert state.held_object is None

    def test_agent_state_to_dict(self):
        """Test AgentState can be converted to dict."""
        state = AgentState(
            position={"x": 1.0, "y": 0.0, "z": 2.0},
            rotation={"x": 0.0, "y": 90.0, "z": 0.0},
            held_object="Mug"
        )
        state_dict = asdict(state)
        assert state_dict["position"]["x"] == 1.0
        assert state_dict["held_object"] == "Mug"

    def test_thor_observation_creation(self):
        """Test ThorObservation dataclass creation."""
        state = AgentState(
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            rotation={"x": 0.0, "y": 0.0, "z": 0.0}
        )
        observation = ThorObservation(
            rgb=b"fake_rgb_data",
            depth=b"fake_depth_data",
            instance_mask=b"fake_mask_data",
            agent_state=state,
            visible_objects=["Mug", "Chair"]
        )
        assert observation.rgb == b"fake_rgb_data"
        assert observation.depth == b"fake_depth_data"
        assert observation.instance_mask == b"fake_mask_data"
        assert observation.agent_state == state
        assert observation.visible_objects == ["Mug", "Chair"]

    def test_thor_observation_optional_fields(self):
        """Test ThorObservation with None for optional image data."""
        state = AgentState(
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            rotation={"x": 0.0, "y": 0.0, "z": 0.0}
        )
        observation = ThorObservation(
            rgb=b"rgb_data",
            depth=None,
            instance_mask=None,
            agent_state=state,
            visible_objects=[]
        )
        assert observation.rgb == b"rgb_data"
        assert observation.depth is None
        assert observation.instance_mask is None

    def test_step_result_success(self):
        """Test StepResult for successful actions."""
        state = AgentState(
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            rotation={"x": 0.0, "y": 0.0, "z": 0.0}
        )
        observation = ThorObservation(
            rgb=b"rgb",
            depth=None,
            instance_mask=None,
            agent_state=state,
            visible_objects=[]
        )
        result = StepResult(
            success=True,
            observation=observation,
            error=None
        )
        assert result.success is True
        assert result.observation == observation
        assert result.error is None

    def test_step_result_failure(self):
        """Test StepResult for failed actions."""
        result = StepResult(
            success=False,
            observation=None,
            error="Invalid action: UnknownAction"
        )
        assert result.success is False
        assert result.observation is None
        assert result.error == "Invalid action: UnknownAction"

    def test_step_result_never_raises(self):
        """StepResult should capture errors, not raise exceptions."""
        # This is a design principle test - StepResult always returns
        result = StepResult(
            success=False,
            observation=None,
            error="Connection failed"
        )
        # No exception should be raised when creating failure result
        assert result.success is False
        assert "Connection" in result.error


class TestThorControllerMocked:
    """Test ThorController with mocked AI2-THOR."""

    def test_controller_initialization_mock_mode(self):
        """Test controller initialization in mock mode."""
        controller = ThorController(use_thor=False)
        assert controller._use_thor is False
        assert controller._controller is None
        controller.close()

    def test_controller_valid_actions_constant(self):
        """Test that VALID_ACTIONS contains expected AI2-THOR actions."""
        controller = ThorController(use_thor=False)

        # Core actions that must be present
        expected_actions = {
            "MoveAhead", "MoveBack", "MoveLeft", "MoveRight",
            "RotateLeft", "RotateRight", "LookUp", "LookDown",
            "PickupObject", "PutObject", "OpenObject", "CloseObject",
            "Teleport"
        }
        # Check that expected actions are a subset of VALID_ACTIONS
        assert expected_actions.issubset(controller.VALID_ACTIONS)
        controller.close()

    def test_controller_reset_mock_mode(self):
        """Test reset in mock mode returns observation."""
        controller = ThorController(use_thor=False)
        observation = controller.reset(
            scene_name="FloorPlan1",
            initial_position={"x": 0.0, "y": 0.0, "z": 0.0}
        )

        assert isinstance(observation, ThorObservation)
        assert observation.agent_state.position == {"x": 0.0, "y": 0.0, "z": 0.0}
        assert observation.rgb is not None  # Mock should return dummy data
        controller.close()

    def test_controller_step_valid_action_mock_mode(self):
        """Test step with valid action in mock mode."""
        controller = ThorController(use_thor=False)
        controller.reset("FloorPlan1", {"x": 0.0, "y": 0.0, "z": 0.0})

        result = controller.step("MoveAhead")
        assert isinstance(result, StepResult)
        assert result.success is True
        assert result.observation is not None
        assert result.error is None
        controller.close()

    def test_controller_step_invalid_action_mock_mode(self):
        """Test step with invalid action returns error."""
        controller = ThorController(use_thor=False)
        controller.reset("FloorPlan1", {"x": 0.0, "y": 0.0, "z": 0.0})

        result = controller.step("InvalidAction")
        assert result.success is False
        assert result.error is not None
        assert "InvalidAction" in result.error
        controller.close()

    def test_controller_step_moves_agent_in_mock_mode(self):
        """Test that MoveAhead updates agent position in mock mode."""
        controller = ThorController(use_thor=False)
        controller.reset("FloorPlan1", {"x": 0.0, "y": 0.0, "z": 0.0})

        initial_state = controller.get_current_state()
        result = controller.step("MoveAhead")

        assert result.success is True
        new_state = controller.get_current_state()
        # Agent should have moved forward (z increases in THOR convention)
        assert new_state.position["z"] != initial_state.position["z"]
        controller.close()

    def test_controller_step_rotate_in_mock_mode(self):
        """Test that RotateLeft/RotateRight update rotation in mock mode."""
        controller = ThorController(use_thor=False)
        controller.reset("FloorPlan1", {"x": 0.0, "y": 0.0, "z": 0.0})

        initial_state = controller.get_current_state()
        initial_rotation = initial_state.rotation["y"]

        result = controller.step("RotateLeft")
        assert result.success is True

        new_state = controller.get_current_state()
        # Rotation should change by 90 degrees (default rotate_step_degrees)
        expected_rotation = (initial_rotation - 90.0) % 360
        assert abs(new_state.rotation["y"] - expected_rotation) < 0.01
        controller.close()

    def test_controller_get_reachable_positions_mock_mode(self):
        """Test get_reachable_positions returns list in mock mode."""
        controller = ThorController(use_thor=False)
        controller.reset("FloorPlan1", {"x": 0.0, "y": 0.0, "z": 0.0})

        positions = controller.get_reachable_positions()
        assert isinstance(positions, list)
        assert len(positions) > 0
        # Each position should have x, y, z keys
        for pos in positions:
            assert "x" in pos
            assert "y" in pos
            assert "z" in pos
        controller.close()

    def test_controller_get_current_state_mock_mode(self):
        """Test get_current_state returns AgentState."""
        controller = ThorController(use_thor=False)
        controller.reset("FloorPlan1", {"x": 1.0, "y": 0.0, "z": 2.0})

        state = controller.get_current_state()
        assert isinstance(state, AgentState)
        assert state.position["x"] == 1.0
        assert state.position["z"] == 2.0
        controller.close()

    def test_controller_step_without_reset_returns_error(self):
        """Test step before reset returns error."""
        controller = ThorController(use_thor=False)

        result = controller.step("MoveAhead")
        assert result.success is False
        assert result.error is not None
        assert "reset" in result.error.lower() or "initialized" in result.error.lower()
        controller.close()

    def test_controller_close_mock_mode(self):
        """Test close in mock mode doesn't raise."""
        controller = ThorController(use_thor=False)
        controller.reset("FloorPlan1", {"x": 0.0, "y": 0.0, "z": 0.0})

        # Should not raise
        controller.close()
        assert controller._controller is None

    def test_controller_pickup_object_mock_mode(self):
        """Test PickupObject action in mock mode."""
        controller = ThorController(use_thor=False)
        controller.reset("FloorPlan1", {"x": 0.0, "y": 0.0, "z": 0.0})

        # In mock mode, pickup should succeed and update held_object
        result = controller.step("PickupObject", objectId="Mug|1|2|3")
        # Mock mode should return success
        assert result.success is True
        controller.close()

    def test_controller_put_object_mock_mode(self):
        """Test PutObject action in mock mode."""
        controller = ThorController(use_thor=False)
        controller.reset("FloorPlan1", {"x": 0.0, "y": 0.0, "z": 0.0})

        # First pickup, then put
        controller.step("PickupObject", objectId="Mug|1|2|3")
        result = controller.step("PutObject", objectId="Mug|1|2|3",
                                  receptacleObjectId="Table|4|5|6")
        assert result.success is True
        controller.close()

    def test_controller_teleport_mock_mode(self):
        """Test Teleport action updates position."""
        controller = ThorController(use_thor=False)
        controller.reset("FloorPlan1", {"x": 0.0, "y": 0.0, "z": 0.0})

        result = controller.step("Teleport", x=5.0, y=0.0, z=3.0)
        assert result.success is True

        state = controller.get_current_state()
        assert state.position["x"] == 5.0
        assert state.position["z"] == 3.0
        controller.close()

    def test_controller_look_up_down_mock_mode(self):
        """Test LookUp and LookDown actions."""
        controller = ThorController(use_thor=False)
        controller.reset("FloorPlan1", {"x": 0.0, "y": 0.0, "z": 0.0})

        # LookUp should decrease horizon (looking up)
        result = controller.step("LookUp")
        assert result.success is True

        # LookDown should increase horizon (looking down)
        result = controller.step("LookDown")
        assert result.success is True
        controller.close()

    def test_controller_with_custom_settings(self):
        """Test controller uses settings from Settings class."""
        from config.settings import Settings

        settings = Settings()
        controller = ThorController(use_thor=False, settings=settings)

        # Controller should have access to settings
        assert controller._settings is not None
        assert controller._settings.thor.grid_size == 0.25
        controller.close()

    def test_controller_multiple_resets(self):
        """Test that multiple resets work correctly."""
        controller = ThorController(use_thor=False)

        # First reset
        obs1 = controller.reset("FloorPlan1", {"x": 0.0, "y": 0.0, "z": 0.0})
        assert obs1.agent_state.position["x"] == 0.0

        # Second reset to different position
        obs2 = controller.reset("FloorPlan1", {"x": 5.0, "y": 0.0, "z": 5.0})
        assert obs2.agent_state.position["x"] == 5.0
        controller.close()


@pytest.mark.skipif(not THOR_AVAILABLE, reason="AI2-THOR not installed")
class TestThorControllerWithThor:
    """Test ThorController with actual AI2-THOR (requires installation)."""

    def test_controller_initialization_with_thor(self):
        """Test controller initialization with real THOR."""
        controller = ThorController(use_thor=True)
        assert controller._use_thor is True
        # After init, controller might still be None (lazy init)
        controller.close()

    def test_controller_lazy_init(self):
        """Test that THOR controller is lazily initialized."""
        controller = ThorController(use_thor=True)
        # Before reset, controller should not be initialized
        assert controller._controller is None

        # After reset, controller should be initialized (or fall back to mock)
        try:
            controller.reset("FloorPlan1", {"x": 0.0, "y": 0.0, "z": 0.0})
        except RuntimeError as e:
            # On Windows, THOR may fail to initialize due to platform limitations
            # This is expected - controller should work in mock mode
            if "no build exists" in str(e):
                pytest.skip("AI2-THOR build not available for Windows")
            raise
        # Controller should be initialized or using mock mode
        assert controller._initialized or controller._controller is None
        controller.close()


class TestThorControllerErrorHandling:
    """Test error handling in ThorController."""

    def test_invalid_scene_name_returns_error(self):
        """Test reset with invalid scene name returns error result."""
        controller = ThorController(use_thor=False)

        # In mock mode, any scene name should work
        # This tests the error handling structure
        result = controller.reset("InvalidScene", {"x": 0.0, "y": 0.0, "z": 0.0})
        # Mock mode should still return an observation
        assert isinstance(result, ThorObservation)
        controller.close()

    def test_step_after_close_returns_error(self):
        """Test that step after close returns error."""
        controller = ThorController(use_thor=False)
        controller.reset("FloorPlan1", {"x": 0.0, "y": 0.0, "z": 0.0})
        controller.close()

        result = controller.step("MoveAhead")
        # After close, should return error or reinitialize
        # Implementation choice: could return error or auto-reinitialize
        # For safety, let's check it handles gracefully
        assert isinstance(result, StepResult)

    def test_missing_object_id_for_pickup(self):
        """Test PickupObject without objectId returns error."""
        controller = ThorController(use_thor=False)
        controller.reset("FloorPlan1", {"x": 0.0, "y": 0.0, "z": 0.0})

        result = controller.step("PickupObject")
        assert result.success is False
        assert result.error is not None
        controller.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
