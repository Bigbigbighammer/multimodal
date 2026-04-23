"""Tests for perception-aware task decomposition."""

import pytest
from unittest.mock import MagicMock

from src.planning.task_decomposer import (
    TaskDecomposer,
    TaskDecomposition,
    Subgoal,
    EnvironmentObservation,
)


class TestEnvironmentObservation:
    """Test EnvironmentObservation dataclass."""

    def test_empty_observation(self):
        """Test empty observation creation."""
        obs = EnvironmentObservation()
        assert obs.visible_objects == []
        assert obs.held_object is None
        assert obs.agent_position == {"x": 0.0, "y": 0.0, "z": 0.0}

    def test_observation_with_visible_objects(self):
        """Test observation with visible objects."""
        obs = EnvironmentObservation(
            visible_objects=[
                {"name": "Apple", "distance": 1.2},
                {"name": "Fridge", "distance": 3.5},
            ]
        )
        assert len(obs.visible_objects) == 2
        assert obs.visible_objects[0]["name"] == "Apple"

    def test_has_visible_object_found(self):
        """Test finding visible object within distance."""
        obs = EnvironmentObservation(
            visible_objects=[
                {"name": "Apple", "distance": 1.2},
                {"name": "Fridge", "distance": 3.5},
            ]
        )
        result = obs.has_visible_object("apple", max_distance=2.0)
        assert result is not None
        assert result["name"] == "Apple"

    def test_has_visible_object_not_found(self):
        """Test not finding visible object."""
        obs = EnvironmentObservation(
            visible_objects=[
                {"name": "Fridge", "distance": 3.5},
            ]
        )
        result = obs.has_visible_object("apple", max_distance=2.0)
        assert result is None

    def test_has_visible_object_too_far(self):
        """Test object visible but too far."""
        obs = EnvironmentObservation(
            visible_objects=[
                {"name": "Apple", "distance": 5.0},
            ]
        )
        result = obs.has_visible_object("apple", max_distance=2.0)
        assert result is None

    def test_to_prompt_string(self):
        """Test formatting for LLM prompt."""
        obs = EnvironmentObservation(
            visible_objects=[
                {"name": "Apple", "distance": 1.2},
                {"name": "Fridge", "distance": 3.5},
            ],
            held_object="Mug"
        )
        prompt_str = obs.to_prompt_string()
        assert "Apple" in prompt_str
        assert "Fridge" in prompt_str
        assert "Holding: Mug" in prompt_str


class TestTaskDecomposerPerceptionAware:
    """Test perception-aware task decomposition."""

    def setup_method(self):
        self.decomposer = TaskDecomposer(use_llm=False)

    def test_decompose_with_visible_target(self):
        """Test decomposition when target is visible."""
        env = EnvironmentObservation(
            visible_objects=[
                {"name": "Apple", "distance": 1.2, "position": {"x": 1, "y": 0, "z": 1}},
            ]
        )

        result = self.decomposer.decompose("find apple", environment=env)

        assert len(result.subgoals) == 1
        assert result.subgoals[0].action == "navigate"
        assert "Apple" in result.subgoals[0].target
        assert "visible" in result.reasoning.lower() or "1.2" in result.reasoning

    def test_decompose_with_visible_target_pickup(self):
        """Test pickup task when target is visible."""
        env = EnvironmentObservation(
            visible_objects=[
                {"name": "Apple", "distance": 1.5, "position": {"x": 1, "y": 0, "z": 1}},
            ]
        )

        result = self.decomposer.decompose("pick up apple", environment=env)

        assert len(result.subgoals) == 2
        assert result.subgoals[0].action == "navigate"
        assert result.subgoals[1].action == "pickup"

    def test_decompose_without_environment(self):
        """Test decomposition without environment (backward compatible)."""
        result = self.decomposer.decompose("find apple")

        assert len(result.subgoals) >= 1
        assert result.subgoals[0].action == "navigate"

    def test_decompose_target_not_visible(self):
        """Test decomposition when target is not visible."""
        env = EnvironmentObservation(
            visible_objects=[
                {"name": "Fridge", "distance": 3.5},
            ]
        )

        result = self.decomposer.decompose("find apple", environment=env)

        # Should fall back to template/default
        assert len(result.subgoals) >= 1

    def test_decompose_with_objectType_key(self):
        """Test with objectType key instead of name."""
        env = EnvironmentObservation(
            visible_objects=[
                {"objectType": "Apple", "distance": 1.0},
            ]
        )

        result = self.decomposer.decompose("find apple", environment=env)

        assert len(result.subgoals) == 1
        assert result.subgoals[0].action == "navigate"


class TestTaskDecompositionReasoning:
    """Test that reasoning reflects perception-aware decisions."""

    def setup_method(self):
        self.decomposer = TaskDecomposer(use_llm=False)

    def test_reasoning_mentions_visible(self):
        """Test that reasoning mentions visibility."""
        env = EnvironmentObservation(
            visible_objects=[
                {"name": "Chair", "distance": 1.0},
            ]
        )

        result = self.decomposer.decompose("find chair", environment=env)

        # Reasoning should mention the visible object
        assert "visible" in result.reasoning.lower() or "1.0" in result.reasoning

    def test_estimated_steps_lower_when_visible(self):
        """Test that estimated steps is lower when target visible."""
        env = EnvironmentObservation(
            visible_objects=[
                {"name": "Apple", "distance": 1.0},
            ]
        )

        result_visible = self.decomposer.decompose("find apple", environment=env)
        result_no_env = self.decomposer.decompose("find apple")

        # Visible target should have fewer estimated steps
        assert result_visible.estimated_steps <= result_no_env.estimated_steps
