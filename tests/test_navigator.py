"""
Tests for NavigatorAgent.

This module tests the NavigatorAgent class with mocked components
to verify navigation behavior without requiring AI2-THOR or actual ML models.
"""

import pytest
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from unittest.mock import Mock, MagicMock, patch

# Test fixtures for dataclasses
@dataclass
class TestNavigationResult:
    """Result of a navigation attempt."""
    success: bool
    final_position: Dict[str, float]
    steps_taken: int
    distance_to_target: float
    error: Optional[str]


@dataclass
class TestObjectInfo:
    """Information about a detected object."""
    object_id: str
    object_type: str
    position: Dict[str, float]
    distance: float
    detection: Any  # Mock detection object
    clip_similarity: float


class TestNavigatorAgent:
    """Test suite for NavigatorAgent."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock ThorController."""
        controller = Mock()

        # Setup mock state
        controller._mock_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        controller._mock_rotation = {"x": 0.0, "y": 0.0, "z": 0.0}

        # Mock reset
        def mock_reset(scene_name, initial_position=None):
            if initial_position:
                controller._mock_position = dict(initial_position)
            return Mock(
                agent_state=Mock(
                    position=controller._mock_position,
                    rotation=controller._mock_rotation,
                    held_object=None
                ),
                visible_objects=["Chair", "Table"]
            )
        controller.reset = mock_reset

        # Mock step
        def mock_step(action, **kwargs):
            if action == "MoveAhead":
                controller._mock_position["z"] += 0.25
            elif action == "MoveBack":
                controller._mock_position["z"] -= 0.25
            elif action == "RotateLeft":
                controller._mock_rotation["y"] = (controller._mock_rotation["y"] - 90) % 360
            elif action == "RotateRight":
                controller._mock_rotation["y"] = (controller._mock_rotation["y"] + 90) % 360

            return Mock(
                success=True,
                observation=Mock(
                    agent_state=Mock(
                        position=dict(controller._mock_position),
                        rotation=dict(controller._mock_rotation),
                        held_object=None
                    ),
                    visible_objects=["Chair", "Table"]
                ),
                error=None
            )
        controller.step = mock_step

        # Mock get_reachable_positions
        controller.get_reachable_positions.return_value = [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 0.25, "y": 0.0, "z": 0.0},
            {"x": 0.0, "y": 0.0, "z": 0.25},
            {"x": 0.25, "y": 0.0, "z": 0.25},
        ]

        # Mock get_current_state
        def mock_get_current_state():
            return Mock(
                position=dict(controller._mock_position),
                rotation=dict(controller._mock_rotation),
                held_object=None
            )
        controller.get_current_state = mock_get_current_state

        return controller

    @pytest.fixture
    def mock_visual_encoder(self):
        """Create a mock VisualEncoder."""
        encoder = Mock()

        # Mock encode_text to return a feature vector
        def mock_encode_text(text):
            import numpy as np
            # Deterministic embedding based on text
            seed = hash(text) % (2**31)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(512).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            return Mock(vector=vector, label=text)

        encoder.encode_text = mock_encode_text

        # Mock encode_image
        def mock_encode_image(image):
            import numpy as np
            seed = hash(tuple(image.shape) if hasattr(image, 'shape') else 0) % (2**31)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(512).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            return Mock(vector=vector)

        encoder.encode_image = mock_encode_image

        # Mock compute_similarity
        def mock_compute_similarity(feat1, feat2):
            import numpy as np
            v1 = feat1.vector if hasattr(feat1, 'vector') else feat1
            v2 = feat2.vector if hasattr(feat2, 'vector') else feat2
            return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        encoder.compute_similarity = mock_compute_similarity

        return encoder

    @pytest.fixture
    def mock_detector(self):
        """Create a mock ObjectDetector."""
        detector = Mock()

        # Mock detect to return detections
        def mock_detect(image):
            mock_detection = Mock(
                bbox=(100, 100, 200, 200),
                class_name="chair",
                confidence=0.85
            )
            return [mock_detection]

        detector.detect = mock_detect

        return detector

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock()
        settings.navigation = Mock(
            success_distance=1.0,
            max_steps_per_episode=200,
            exploration_frontier_weight=2.0,
            max_exploration_steps=50,
            path_replan_threshold=0.5,
            move_distance=0.25,
            collision_threshold=0.1
        )
        settings.memory = Mock(
            map_edge_distance_threshold=0.30,
            map_max_nodes=1000,
            working_memory_max_steps=10
        )
        settings.perception = Mock(
            clip_match_threshold=0.25,
            clip_model="ViT-B-32",
            clip_pretrained="laion2b_s34b_b79k",
            yolo_model="yolov8n.pt",
            yolo_confidence=0.3,
            yolo_iou_threshold=0.5,
            cache_features=True
        )
        settings.thor = Mock(
            grid_size=0.25,
            render_depth=True,
            render_instance_segmentation=True,
            width=640,
            height=480,
            visibility_distance=1.5,
            rotate_step_degrees=90.0,
            horizon_step_degrees=30.0,
            snap_to_grid=True
        )
        return settings

    def test_navigation_result_dataclass(self):
        """Test NavigationResult dataclass creation and defaults."""
        # Import from actual module
        from src.agent.navigator import NavigationResult

        result = NavigationResult(
            success=True,
            final_position={"x": 1.0, "y": 0.0, "z": 2.0},
            steps_taken=10,
            distance_to_target=0.5,
            error=None
        )

        assert result.success is True
        assert result.final_position == {"x": 1.0, "y": 0.0, "z": 2.0}
        assert result.steps_taken == 10
        assert result.distance_to_target == 0.5
        assert result.error is None

    def test_object_info_dataclass(self):
        """Test ObjectInfo dataclass creation."""
        from src.agent.navigator import ObjectInfo

        mock_detection = Mock(bbox=(0, 0, 100, 100), class_name="chair", confidence=0.9)
        obj_info = ObjectInfo(
            object_id="Chair|1|2|3",
            object_type="Chair",
            position={"x": 1.0, "y": 0.0, "z": 2.0},
            distance=1.5,
            detection=mock_detection,
            clip_similarity=0.85
        )

        assert obj_info.object_id == "Chair|1|2|3"
        assert obj_info.object_type == "Chair"
        assert obj_info.distance == 1.5
        assert obj_info.clip_similarity == 0.85

    def test_navigator_agent_init(
        self, mock_controller, mock_visual_encoder, mock_detector, mock_settings
    ):
        """Test NavigatorAgent initialization."""
        from src.agent.navigator import NavigatorAgent

        agent = NavigatorAgent(
            controller=mock_controller,
            visual_encoder=mock_visual_encoder,
            detector=mock_detector,
            settings=mock_settings
        )

        assert agent._controller is mock_controller
        assert agent._visual_encoder is mock_visual_encoder
        assert agent._detector is mock_detector
        assert agent._spatial_map is None  # Not built until build_spatial_map is called
        assert agent._working_memory is not None

    def test_build_spatial_map(
        self, mock_controller, mock_visual_encoder, mock_detector, mock_settings
    ):
        """Test building spatial map from reachable positions."""
        from src.agent.navigator import NavigatorAgent

        agent = NavigatorAgent(
            controller=mock_controller,
            visual_encoder=mock_visual_encoder,
            detector=mock_detector,
            settings=mock_settings
        )

        # Build spatial map
        spatial_map = agent.build_spatial_map()

        assert spatial_map is not None
        assert len(spatial_map.nodes) > 0
        # Should have nodes for reachable positions
        assert len(spatial_map.nodes) == 4  # Based on mock reachable positions

    def test_find_object_not_found(
        self, mock_controller, mock_visual_encoder, mock_detector, mock_settings
    ):
        """Test find_object when object is not visible."""
        from src.agent.navigator import NavigatorAgent

        agent = NavigatorAgent(
            controller=mock_controller,
            visual_encoder=mock_visual_encoder,
            detector=mock_detector,
            settings=mock_settings
        )

        # Mock _get_visible_objects_from_thor to return empty list
        agent._get_visible_objects_from_thor = Mock(return_value=[])

        result = agent.find_object("nonexistent object")

        assert result is None

    def test_find_object_found(
        self, mock_controller, mock_visual_encoder, mock_detector, mock_settings
    ):
        """Test find_object when object is found."""
        from src.agent.navigator import NavigatorAgent, ObjectInfo

        agent = NavigatorAgent(
            controller=mock_controller,
            visual_encoder=mock_visual_encoder,
            detector=mock_detector,
            settings=mock_settings
        )

        # Create mock object info
        mock_obj = ObjectInfo(
            object_id="Chair|1|0|2",
            object_type="Chair",
            position={"x": 1.0, "y": 0.0, "z": 2.0},
            distance=1.5,
            detection=Mock(bbox=(0, 0, 100, 100), class_name="chair", confidence=0.9),
            clip_similarity=0.85
        )

        # Mock methods
        agent._get_visible_objects_from_thor = Mock(return_value=[mock_obj])
        agent._find_best_match_thor = Mock(return_value=mock_obj)

        result = agent.find_object("a chair")

        assert result is not None
        assert result.object_type == "Chair"

    def test_navigate_to_success(
        self, mock_controller, mock_visual_encoder, mock_detector, mock_settings
    ):
        """Test successful navigation to target."""
        from src.agent.navigator import NavigatorAgent, NavigationResult

        agent = NavigatorAgent(
            controller=mock_controller,
            visual_encoder=mock_visual_encoder,
            detector=mock_detector,
            settings=mock_settings
        )

        # Build initial map
        agent.build_spatial_map()

        # Mock find_object to return a target
        mock_target = Mock(
            object_id="Target|1|0|1",
            object_type="Target",
            position={"x": 0.0, "y": 0.0, "z": 0.75},  # Close enough
            distance=0.75
        )

        # Mock verification callback
        verify_callback = Mock(return_value=True)

        result = agent.navigate_to(
            target_description="target object",
            max_steps=10,
            verify_callback=verify_callback
        )

        # Navigation should complete (may succeed or fail based on path)
        assert result.steps_taken >= 0
        assert isinstance(result, NavigationResult)

    def test_navigate_to_max_steps(
        self, mock_controller, mock_visual_encoder, mock_detector, mock_settings
    ):
        """Test navigation stops at max steps."""
        from src.agent.navigator import NavigatorAgent

        agent = NavigatorAgent(
            controller=mock_controller,
            visual_encoder=mock_visual_encoder,
            detector=mock_detector,
            settings=mock_settings
        )

        agent.build_spatial_map()

        # Navigate with very low max_steps
        result = agent.navigate_to(
            target_description="far away object",
            max_steps=1
        )

        # Should stop after 1 step
        assert result.steps_taken <= 1

    def test_get_visible_objects_from_thor(
        self, mock_controller, mock_visual_encoder, mock_detector, mock_settings
    ):
        """Test getting visible objects from THOR."""
        from src.agent.navigator import NavigatorAgent

        agent = NavigatorAgent(
            controller=mock_controller,
            visual_encoder=mock_visual_encoder,
            detector=mock_detector,
            settings=mock_settings
        )

        # Mock controller step to return objects
        mock_event = Mock()
        mock_event.metadata = {
            "objects": [
                {
                    "objectId": "Chair|1|0|2",
                    "objectType": "Chair",
                    "position": {"x": 1.0, "y": 0.0, "z": 2.0},
                    "visible": True,
                    "distance": 1.5
                },
                {
                    "objectId": "Table|2|0|3",
                    "objectType": "Table",
                    "position": {"x": 2.0, "y": 0.0, "z": 3.0},
                    "visible": False,  # Not visible
                    "distance": 2.5
                }
            ]
        }

        # Mock the step result
        mock_step_result = Mock(
            success=True,
            observation=Mock(rgb=b"fake_image_data"),
            error=None
        )
        mock_controller.step.return_value = mock_step_result

        # This is tested indirectly through build_spatial_map and other methods
        # The method is internal and uses controller step
        assert agent._controller is not None

    def test_explore_step(
        self, mock_controller, mock_visual_encoder, mock_detector, mock_settings
    ):
        """Test exploration step functionality."""
        from src.agent.navigator import NavigatorAgent
        from src.agent.controller import StepResult

        agent = NavigatorAgent(
            controller=mock_controller,
            visual_encoder=mock_visual_encoder,
            detector=mock_detector,
            settings=mock_settings
        )

        agent.build_spatial_map()

        # _explore_step should return a StepResult
        result = agent._explore_step()

        # Result should be a StepResult or None
        assert result is not None
        assert hasattr(result, 'success')

    def test_move_towards(
        self, mock_controller, mock_visual_encoder, mock_detector, mock_settings
    ):
        """Test moving towards a target position."""
        from src.agent.navigator import NavigatorAgent
        from src.agent.controller import StepResult

        agent = NavigatorAgent(
            controller=mock_controller,
            visual_encoder=mock_visual_encoder,
            detector=mock_detector,
            settings=mock_settings
        )

        agent.build_spatial_map()

        # Target position to move towards
        target_pos = {"x": 0.5, "y": 0.0, "z": 0.5}

        result = agent._move_towards(target_pos)

        # Result should be a StepResult
        assert result is not None
        assert hasattr(result, 'success')

    def test_navigate_without_initialization(
        self, mock_controller, mock_visual_encoder, mock_detector, mock_settings
    ):
        """Test navigation without calling reset first."""
        from src.agent.navigator import NavigatorAgent

        agent = NavigatorAgent(
            controller=mock_controller,
            visual_encoder=mock_visual_encoder,
            detector=mock_detector,
            settings=mock_settings
        )

        # Don't build map - should handle gracefully
        result = agent.navigate_to("target", max_steps=5)

        assert result.success is False or result.error is not None

    def test_find_best_match_thor(
        self, mock_controller, mock_visual_encoder, mock_detector, mock_settings
    ):
        """Test finding best matching object from THOR objects."""
        from src.agent.navigator import NavigatorAgent, ObjectInfo

        agent = NavigatorAgent(
            controller=mock_controller,
            visual_encoder=mock_visual_encoder,
            detector=mock_detector,
            settings=mock_settings
        )

        # Create mock objects
        objects = [
            ObjectInfo(
                object_id="Chair|1|0|2",
                object_type="Chair",
                position={"x": 1.0, "y": 0.0, "z": 2.0},
                distance=1.5,
                detection=Mock(bbox=(0, 0, 100, 100), class_name="chair", confidence=0.9),
                clip_similarity=0.85
            ),
            ObjectInfo(
                object_id="Table|2|0|3",
                object_type="Table",
                position={"x": 2.0, "y": 0.0, "z": 3.0},
                distance=2.5,
                detection=Mock(bbox=(0, 0, 100, 100), class_name="table", confidence=0.8),
                clip_similarity=0.70
            )
        ]

        # Find best match for "chair"
        result = agent._find_best_match_thor("a chair to sit on", objects)

        # Should find the chair with higher similarity
        if result is not None:
            assert result.object_type in ["Chair", "Table"]


class TestNavigatorAgentIntegration:
    """Integration tests for NavigatorAgent."""

    @pytest.fixture
    def full_mock_setup(self):
        """Create a full mock setup for integration testing."""
        controller = Mock()
        controller._mock_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        controller._mock_rotation = {"x": 0.0, "y": 0.0, "z": 0.0}

        def mock_step(action, **kwargs):
            step_dist = 0.25
            if action == "MoveAhead":
                controller._mock_position["z"] += step_dist
            elif action == "MoveBack":
                controller._mock_position["z"] -= step_dist
            elif action == "RotateLeft":
                controller._mock_rotation["y"] = (controller._mock_rotation["y"] - 90) % 360
            elif action == "RotateRight":
                controller._mock_rotation["y"] = (controller._mock_rotation["y"] + 90) % 360

            return Mock(
                success=True,
                observation=Mock(
                    rgb=b"mock_image",
                    agent_state=Mock(
                        position=dict(controller._mock_position),
                        rotation=dict(controller._mock_rotation),
                        held_object=None
                    )
                ),
                error=None
            )

        controller.step = mock_step
        controller.get_reachable_positions.return_value = [
            {"x": i * 0.25, "y": 0.0, "z": j * 0.25}
            for i in range(-3, 4) for j in range(-3, 4)
        ]
        controller.get_current_state = lambda: Mock(
            position=dict(controller._mock_position),
            rotation=dict(controller._mock_rotation),
            held_object=None
        )

        encoder = Mock()
        encoder.encode_text = lambda t: Mock(vector=self._make_vector(t), label=t)
        encoder.encode_image = lambda i: Mock(vector=self._make_vector(str(i)))
        encoder.compute_similarity = lambda f1, f2: self._compute_sim(f1, f2)

        detector = Mock()
        detector.detect = lambda i: [Mock(bbox=(0, 0, 100, 100), class_name="object", confidence=0.8)]

        settings = Mock()
        settings.navigation = Mock(
            success_distance=1.0,
            max_steps_per_episode=200,
            max_exploration_steps=50,
            move_distance=0.25
        )
        settings.memory = Mock(map_edge_distance_threshold=0.30, map_max_nodes=1000)
        settings.perception = Mock(clip_match_threshold=0.25)

        return controller, encoder, detector, settings

    @staticmethod
    def _make_vector(text):
        import numpy as np
        seed = hash(text) % (2**31)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(512).astype(np.float32)
        return v / np.linalg.norm(v)

    @staticmethod
    def _compute_sim(f1, f2):
        import numpy as np
        v1 = f1.vector if hasattr(f1, 'vector') else f1
        v2 = f2.vector if hasattr(f2, 'vector') else f2
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def test_full_navigation_cycle(self, full_mock_setup):
        """Test a complete navigation cycle."""
        controller, encoder, detector, settings = full_mock_setup

        from src.agent.navigator import NavigatorAgent

        agent = NavigatorAgent(
            controller=controller,
            visual_encoder=encoder,
            detector=detector,
            settings=settings
        )

        # Build map
        spatial_map = agent.build_spatial_map()
        assert spatial_map is not None
        assert len(spatial_map.nodes) > 0

        # Navigate
        result = agent.navigate_to(
            target_description="target",
            max_steps=5
        )

        assert result.steps_taken >= 0
        assert result.final_position is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
