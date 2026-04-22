"""
YOLO object detection wrapper for the Vision-Language Embodied Agent.

This module provides a Detection dataclass and ObjectDetector class that wrap
YOLO (YOLOv8) functionality for object detection in the embodied agent context.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, TYPE_CHECKING

# Handle YOLO import gracefully
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image


@dataclass
class Detection:
    """
    Represents a single object detection result.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        class_name: Class label of the detected object.
        confidence: Detection confidence score (0.0 to 1.0).
    """
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_name: str
    confidence: float

    @property
    def center(self) -> Tuple[int, int]:
        """Calculate the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        """Calculate the width of the bounding box."""
        x1, _, x2, _ = self.bbox
        return x2 - x1

    @property
    def height(self) -> int:
        """Calculate the height of the bounding box."""
        _, y1, _, y2 = self.bbox
        return y2 - y1

    def crop(self, image: "np.ndarray") -> "np.ndarray":
        """
        Crop the detection region from the given image.

        Args:
            image: Input image as numpy array (H, W, C).

        Returns:
            Cropped image region corresponding to the bounding box.
        """
        x1, y1, x2, y2 = self.bbox
        # Ensure bounds are within image dimensions
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        return image[y1:y2, x1:x2]


class ObjectDetector:
    """
    YOLO-based object detector for the embodied agent.

    This class wraps the YOLOv8 model and provides a simple interface for
    object detection with configurable confidence thresholds.

    Attributes:
        model: The YOLO model instance.
        confidence_threshold: Minimum confidence for detections.
        iou_threshold: IoU threshold for non-maximum suppression.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        use_mock: bool = False
    ):
        """
        Initialize the ObjectDetector.

        Args:
            model_name: Name or path of the YOLO model weights.
            confidence_threshold: Minimum confidence score for detections.
            iou_threshold: IoU threshold for non-maximum suppression.
            use_mock: If True, use mock mode without loading actual model.
        """
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._use_mock = use_mock
        self._model: Optional[Any] = None

        if not use_mock and YOLO_AVAILABLE:
            try:
                self._model = YOLO(model_name)
            except Exception:
                # Fall back to mock mode if model loading fails
                self._use_mock = True

    @property
    def model(self) -> Optional[Any]:
        """Get the underlying YOLO model, if available."""
        return self._model

    @property
    def confidence_threshold(self) -> float:
        """Get the confidence threshold for detections."""
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        """Set the confidence threshold for detections."""
        self._confidence_threshold = max(0.0, min(1.0, value))

    @property
    def iou_threshold(self) -> float:
        """Get the IoU threshold for non-maximum suppression."""
        return self._iou_threshold

    @iou_threshold.setter
    def iou_threshold(self, value: float) -> None:
        """Set the IoU threshold for non-maximum suppression."""
        self._iou_threshold = max(0.0, min(1.0, value))

    def detect(self, image: "np.ndarray") -> List[Detection]:
        """
        Detect objects in the given image.

        Args:
            image: Input image as numpy array (H, W, C) in RGB format.

        Returns:
            List of Detection objects sorted by confidence (descending).
        """
        if self._use_mock or self._model is None:
            return self._detect_mock(image)

        try:
            results = self._model(
                image,
                conf=self._confidence_threshold,
                iou=self._iou_threshold,
                verbose=False
            )

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = result.names.get(class_id, f"class_{class_id}")

                    detection = Detection(
                        bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                        class_name=class_name,
                        confidence=confidence
                    )
                    detections.append(detection)

            # Sort by confidence descending
            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections

        except Exception:
            # Fall back to mock on error
            return self._detect_mock(image)

    def detect_classes(
        self,
        image: "np.ndarray",
        class_names: List[str]
    ) -> List[Detection]:
        """
        Detect objects of specific classes in the given image.

        Args:
            image: Input image as numpy array (H, W, C) in RGB format.
            class_names: List of class names to filter detections.

        Returns:
            List of Detection objects for the specified classes,
            sorted by confidence (descending).
        """
        all_detections = self.detect(image)
        class_set = set(class_names)

        filtered = [d for d in all_detections if d.class_name in class_set]
        filtered.sort(key=lambda d: d.confidence, reverse=True)
        return filtered

    def _detect_mock(self, image: "np.ndarray") -> List[Detection]:
        """
        Generate mock detections for testing.

        Args:
            image: Input image as numpy array.

        Returns:
            List of mock Detection objects.
        """
        h, w = image.shape[:2]

        # Return a mock detection at the center of the image
        center_x, center_y = w // 2, h // 2
        box_size = min(w, h) // 4

        mock_detection = Detection(
            bbox=(
                center_x - box_size // 2,
                center_y - box_size // 2,
                center_x + box_size // 2,
                center_y + box_size // 2
            ),
            class_name="object",
            confidence=0.5
        )

        return [mock_detection]
