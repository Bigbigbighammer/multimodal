"""
Perception module for the Vision-Language Embodied Agent.

This module provides:
- ObjectDetector: YOLO-based object detection
- Detection: Object detection result dataclass
- VisualEncoder: CLIP-based visual encoding
- FeatureVector: Feature vector dataclass
"""

from perception.detector import Detection, ObjectDetector, YOLO_AVAILABLE
from perception.visual_encoder import FeatureVector, VisualEncoder, CLIP_AVAILABLE

__all__ = [
    "Detection",
    "ObjectDetector",
    "FeatureVector",
    "VisualEncoder",
    "YOLO_AVAILABLE",
    "CLIP_AVAILABLE",
]
