"""
Tests for YOLO object detector.

This module tests:
- Detection dataclass
- ObjectDetector with mocked YOLO
- Error handling and edge cases
"""

import pytest
import numpy as np
from dataclasses import asdict

import sys
sys.path.insert(0, "src")

from perception.detector import Detection, ObjectDetector, YOLO_AVAILABLE


class TestDetection:
    """Test the Detection dataclass."""

    def test_detection_creation(self):
        """Test Detection dataclass creation with all fields."""
        detection = Detection(
            bbox=(10, 20, 100, 200),
            class_name="cup",
            confidence=0.85
        )
        assert detection.bbox == (10, 20, 100, 200)
        assert detection.class_name == "cup"
        assert detection.confidence == 0.85

    def test_detection_center_property(self):
        """Test Detection center property calculation."""
        detection = Detection(
            bbox=(0, 0, 100, 200),
            class_name="object",
            confidence=0.5
        )
        assert detection.center == (50, 100)

    def test_detection_center_property_odd_dimensions(self):
        """Test Detection center property with odd dimensions (integer division)."""
        detection = Detection(
            bbox=(0, 0, 99, 199),
            class_name="object",
            confidence=0.5
        )
        assert detection.center == (49, 99)

    def test_detection_width_property(self):
        """Test Detection width property calculation."""
        detection = Detection(
            bbox=(10, 0, 110, 100),
            class_name="object",
            confidence=0.5
        )
        assert detection.width == 100

    def test_detection_height_property(self):
        """Test Detection height property calculation."""
        detection = Detection(
            bbox=(0, 20, 100, 120),
            class_name="object",
            confidence=0.5
        )
        assert detection.height == 100

    def test_detection_crop(self):
        """Test Detection crop method extracts correct region."""
        # Create a simple test image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[20:40, 30:50] = 255  # White square in the image

        detection = Detection(
            bbox=(30, 20, 50, 40),
            class_name="object",
            confidence=0.5
        )

        crop = detection.crop(image)
        assert crop.shape == (20, 20, 3)
        assert np.all(crop == 255)  # Should be all white

    def test_detection_crop_boundary_clipping(self):
        """Test Detection crop clips to image boundaries."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Bounding box extends beyond image boundaries
        detection = Detection(
            bbox=(-10, -10, 150, 150),
            class_name="object",
            confidence=0.5
        )

        crop = detection.crop(image)
        # Should be clipped to image size
        assert crop.shape == (100, 100, 3)

    def test_detection_to_dict(self):
        """Test Detection can be converted to dict via dataclasses.asdict."""
        detection = Detection(
            bbox=(10, 20, 100, 200),
            class_name="cup",
            confidence=0.85
        )
        d = asdict(detection)
        assert d["bbox"] == (10, 20, 100, 200)
        assert d["class_name"] == "cup"
        assert d["confidence"] == 0.85


class TestObjectDetector:
    """Test the ObjectDetector class."""

    def test_detector_initialization_mock_mode(self):
        """Test detector initialization in mock mode."""
        detector = ObjectDetector(use_mock=True)
        assert detector._use_mock is True
        assert detector.model is None

    def test_detector_initialization_with_custom_params(self):
        """Test detector initialization with custom parameters."""
        detector = ObjectDetector(
            model_name="yolov8s.pt",
            confidence_threshold=0.5,
            iou_threshold=0.6,
            use_mock=True
        )
        assert detector._model_name == "yolov8s.pt"
        assert detector.confidence_threshold == 0.5
        assert detector.iou_threshold == 0.6

    def test_detector_confidence_threshold_clamping(self):
        """Test confidence threshold is clamped to [0, 1]."""
        detector = ObjectDetector(use_mock=True)

        detector.confidence_threshold = 1.5
        assert detector.confidence_threshold == 1.0

        detector.confidence_threshold = -0.5
        assert detector.confidence_threshold == 0.0

        detector.confidence_threshold = 0.5
        assert detector.confidence_threshold == 0.5

    def test_detector_iou_threshold_clamping(self):
        """Test IoU threshold is clamped to [0, 1]."""
        detector = ObjectDetector(use_mock=True)

        detector.iou_threshold = 2.0
        assert detector.iou_threshold == 1.0

        detector.iou_threshold = -1.0
        assert detector.iou_threshold == 0.0

    def test_detector_detect_mock_mode(self):
        """Test detect method in mock mode returns mock detection."""
        detector = ObjectDetector(use_mock=True)
        image = np.zeros((640, 480, 3), dtype=np.uint8)

        detections = detector.detect(image)
        assert isinstance(detections, list)
        assert len(detections) == 1
        assert isinstance(detections[0], Detection)
        assert detections[0].class_name == "object"
        assert detections[0].confidence == 0.5

    def test_detector_detect_mock_mode_centered_bbox(self):
        """Test mock detection bbox is centered in image."""
        detector = ObjectDetector(use_mock=True)
        h, w = 400, 600
        image = np.zeros((h, w, 3), dtype=np.uint8)

        detections = detector.detect(image)
        detection = detections[0]

        # Mock detection should be centered
        expected_center = (w // 2, h // 2)
        assert detection.center == expected_center

    def test_detector_detect_classes_mock_mode(self):
        """Test detect_classes method filters by class names."""
        detector = ObjectDetector(use_mock=True)
        image = np.zeros((640, 480, 3), dtype=np.uint8)

        # Mock returns "object" class
        detections = detector.detect_classes(image, ["object"])
        assert len(detections) == 1

        # Different class name - should filter out
        detections = detector.detect_classes(image, ["cup", "chair"])
        assert len(detections) == 0

    def test_detector_detect_classes_empty_list(self):
        """Test detect_classes with empty class list returns empty list."""
        detector = ObjectDetector(use_mock=True)
        image = np.zeros((640, 480, 3), dtype=np.uint8)

        detections = detector.detect_classes(image, [])
        assert len(detections) == 0

    def test_detector_detect_returns_sorted_by_confidence(self):
        """Test detect returns detections sorted by confidence descending."""
        detector = ObjectDetector(use_mock=True)
        image = np.zeros((640, 480, 3), dtype=np.uint8)

        # Mock mode returns single detection, but real mode should sort
        detections = detector.detect(image)
        # Verify list is sorted
        for i in range(len(detections) - 1):
            assert detections[i].confidence >= detections[i + 1].confidence

    def test_detector_detect_classes_returns_sorted_by_confidence(self):
        """Test detect_classes returns detections sorted by confidence."""
        detector = ObjectDetector(use_mock=True)
        image = np.zeros((640, 480, 3), dtype=np.uint8)

        detections = detector.detect_classes(image, ["object"])
        for i in range(len(detections) - 1):
            assert detections[i].confidence >= detections[i + 1].confidence


@pytest.mark.skipif(not YOLO_AVAILABLE, reason="YOLO (ultralytics) not installed")
class TestObjectDetectorWithYolo:
    """Test ObjectDetector with actual YOLO model (requires installation)."""

    def test_detector_initialization_with_yolo(self):
        """Test detector initialization with real YOLO model."""
        detector = ObjectDetector(model_name="yolov8n.pt", use_mock=False)
        # Model might load successfully or fall back to mock
        assert detector.model is not None or detector._use_mock

    def test_detector_detect_with_yolo(self):
        """Test detect method with real YOLO model."""
        detector = ObjectDetector(model_name="yolov8n.pt", use_mock=False)

        # Create a simple test image
        image = np.zeros((640, 480, 3), dtype=np.uint8)

        detections = detector.detect(image)
        assert isinstance(detections, list)
        # YOLO should detect something (or return empty list)
        for d in detections:
            assert isinstance(d, Detection)


class TestObjectDetectorWithSettings:
    """Test ObjectDetector integration with Settings."""

    def test_detector_with_settings(self):
        """Test detector can be created from Settings."""
        from config.settings import Settings

        settings = Settings()
        detector = ObjectDetector(
            model_name=settings.perception.yolo_model,
            confidence_threshold=settings.perception.yolo_confidence,
            use_mock=True
        )

        assert detector._model_name == settings.perception.yolo_model
        assert detector.confidence_threshold == settings.perception.yolo_confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
