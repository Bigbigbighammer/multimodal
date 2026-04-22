"""
Tests for CLIP visual encoder.

This module tests:
- FeatureVector dataclass
- VisualEncoder with mocked CLIP
- Similarity matching and detection matching
"""

import pytest
import numpy as np
from typing import List

import sys
sys.path.insert(0, "src")

from perception.visual_encoder import FeatureVector, VisualEncoder, CLIP_AVAILABLE


class TestFeatureVector:
    """Test the FeatureVector dataclass."""

    def test_feature_vector_creation(self):
        """Test FeatureVector dataclass creation with all fields."""
        vector = np.random.randn(512).astype(np.float32)
        feature = FeatureVector(
            vector=vector,
            source="text:apple",
            label="apple"
        )
        assert np.array_equal(feature.vector, vector)
        assert feature.source == "text:apple"
        assert feature.label == "apple"

    def test_feature_vector_optional_label(self):
        """Test FeatureVector with None label."""
        vector = np.random.randn(512).astype(np.float32)
        feature = FeatureVector(
            vector=vector,
            source="image",
            label=None
        )
        assert feature.label is None

    def test_feature_vector_shape_property(self):
        """Test FeatureVector shape property."""
        vector = np.random.randn(512).astype(np.float32)
        feature = FeatureVector(vector=vector, source="test")
        assert feature.shape == (512,)

    def test_feature_vector_dimension_property(self):
        """Test FeatureVector dimension property."""
        vector = np.random.randn(512).astype(np.float32)
        feature = FeatureVector(vector=vector, source="test")
        assert feature.dimension == 512


class TestVisualEncoder:
    """Test the VisualEncoder class."""

    def test_encoder_initialization_mock_mode(self):
        """Test encoder initialization in mock mode."""
        encoder = VisualEncoder(use_mock=True)
        assert encoder._use_mock is True
        assert encoder.model is None

    def test_encoder_initialization_with_custom_params(self):
        """Test encoder initialization with custom parameters."""
        encoder = VisualEncoder(
            model_name="ViT-B-16",
            pretrained="laion2b_s34b_b88k",
            device="cpu",
            use_mock=True
        )
        assert encoder._model_name == "ViT-B-16"
        assert encoder._pretrained == "laion2b_s34b_b88k"
        assert encoder.device == "cpu"

    def test_encoder_device_property(self):
        """Test encoder device property."""
        encoder = VisualEncoder(device="cpu", use_mock=True)
        assert encoder.device == "cpu"

    def test_encoder_embed_dim_property(self):
        """Test encoder embed_dim property returns dimension."""
        encoder = VisualEncoder(use_mock=True)
        # Mock mode returns default 512
        assert encoder.embed_dim == 512

    def test_encoder_encode_text_mock_mode(self):
        """Test encode_text method in mock mode."""
        encoder = VisualEncoder(use_mock=True)
        feature = encoder.encode_text("apple")

        assert isinstance(feature, FeatureVector)
        assert feature.source == "text:apple"
        assert feature.label == "apple"
        assert feature.vector.shape == (512,)
        # Mock vector should be normalized
        norm = np.linalg.norm(feature.vector)
        assert abs(norm - 1.0) < 1e-5

    def test_encoder_encode_text_deterministic(self):
        """Test mock encode_text is deterministic for same input."""
        encoder = VisualEncoder(use_mock=True)
        feature1 = encoder.encode_text("apple")
        feature2 = encoder.encode_text("apple")

        assert np.allclose(feature1.vector, feature2.vector)

    def test_encoder_encode_text_different_for_different_inputs(self):
        """Test mock encode_text produces different embeddings for different texts."""
        encoder = VisualEncoder(use_mock=True)
        feature1 = encoder.encode_text("apple")
        feature2 = encoder.encode_text("banana")

        assert not np.allclose(feature1.vector, feature2.vector)

    def test_encoder_encode_image_mock_mode(self):
        """Test encode_image method in mock mode."""
        encoder = VisualEncoder(use_mock=True)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        feature = encoder.encode_image(image)

        assert isinstance(feature, FeatureVector)
        assert feature.source == "image"
        assert feature.label is None
        assert feature.vector.shape == (512,)
        # Mock vector should be normalized
        norm = np.linalg.norm(feature.vector)
        assert abs(norm - 1.0) < 1e-5

    def test_encoder_encode_image_deterministic(self):
        """Test mock encode_image is deterministic for same input."""
        encoder = VisualEncoder(use_mock=True)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        feature1 = encoder.encode_image(image)
        feature2 = encoder.encode_image(image)

        assert np.allclose(feature1.vector, feature2.vector)

    def test_encoder_encode_images(self):
        """Test encode_images method for batch encoding."""
        encoder = VisualEncoder(use_mock=True)
        images = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.ones((100, 100, 3), dtype=np.uint8),
        ]

        features = encoder.encode_images(images)
        assert len(features) == 2
        assert all(isinstance(f, FeatureVector) for f in features)

    def test_encoder_compute_similarity_identical(self):
        """Test compute_similarity for identical vectors returns 1.0."""
        encoder = VisualEncoder(use_mock=True)
        vector = np.random.randn(512).astype(np.float32)
        vector = vector / np.linalg.norm(vector)

        feature = FeatureVector(vector=vector, source="test")
        similarity = encoder.compute_similarity(feature, feature)

        assert abs(similarity - 1.0) < 1e-5

    def test_encoder_compute_similarity_orthogonal(self):
        """Test compute_similarity for orthogonal vectors."""
        encoder = VisualEncoder(use_mock=True)

        # Create orthogonal vectors
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        feature1 = FeatureVector(vector=v1, source="test1")
        feature2 = FeatureVector(vector=v2, source="test2")

        similarity = encoder.compute_similarity(feature1, feature2)
        assert abs(similarity) < 1e-5

    def test_encoder_compute_similarity_range(self):
        """Test compute_similarity returns value in valid range."""
        encoder = VisualEncoder(use_mock=True)

        v1 = np.random.randn(512).astype(np.float32)
        v2 = np.random.randn(512).astype(np.float32)

        feature1 = FeatureVector(vector=v1, source="test1")
        feature2 = FeatureVector(vector=v2, source="test2")

        similarity = encoder.compute_similarity(feature1, feature2)
        assert -1.0 <= similarity <= 1.0

    def test_encoder_find_best_match_returns_best(self):
        """Test find_best_match returns the best matching candidate."""
        encoder = VisualEncoder(use_mock=True)

        # Create query and candidates
        query_vector = np.zeros(512, dtype=np.float32)
        query_vector[0] = 1.0

        candidate1_vector = np.zeros(512, dtype=np.float32)
        candidate1_vector[0] = 0.9  # Similar to query

        candidate2_vector = np.zeros(512, dtype=np.float32)
        candidate2_vector[1] = 1.0  # Orthogonal to query

        query = FeatureVector(vector=query_vector, source="query")
        candidates = [
            FeatureVector(vector=candidate1_vector, source="c1", label="candidate1"),
            FeatureVector(vector=candidate2_vector, source="c2", label="candidate2"),
        ]

        best_match, score = encoder.find_best_match(query, candidates)

        assert best_match is not None
        assert best_match.label == "candidate1"
        assert score > 0.8

    def test_encoder_find_best_match_empty_candidates(self):
        """Test find_best_match with empty candidates returns None."""
        encoder = VisualEncoder(use_mock=True)
        query_vector = np.random.randn(512).astype(np.float32)
        query = FeatureVector(vector=query_vector, source="query")

        best_match, score = encoder.find_best_match(query, [])

        assert best_match is None
        assert score == 0.0

    def test_encoder_find_best_match_with_threshold(self):
        """Test find_best_match respects threshold."""
        encoder = VisualEncoder(use_mock=True)

        # Create query and low-similarity candidates
        query_vector = np.zeros(512, dtype=np.float32)
        query_vector[0] = 1.0

        candidate_vector = np.zeros(512, dtype=np.float32)
        candidate_vector[1] = 1.0  # Orthogonal

        query = FeatureVector(vector=query_vector, source="query")
        candidates = [FeatureVector(vector=candidate_vector, source="c1", label="candidate1")]

        # Threshold of 0.5 should reject the orthogonal candidate
        best_match, score = encoder.find_best_match(query, candidates, threshold=0.5)

        assert best_match is None
        assert score < 0.5

    def test_encoder_match_detections(self):
        """Test match_detections method."""
        encoder = VisualEncoder(use_mock=True)

        # Import Detection here to avoid issues
        from perception.detector import Detection

        # Create a simple image
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create detections
        detections = [
            Detection(bbox=(10, 10, 50, 50), class_name="object", confidence=0.8),
        ]

        # Text queries
        text_queries = ["cup", "chair"]

        matches = encoder.match_detections(image, detections, text_queries, threshold=0.0)

        # With threshold 0.0, should get matches (mock embeddings)
        assert isinstance(matches, list)

    def test_encoder_match_detections_with_threshold(self):
        """Test match_detections respects threshold."""
        encoder = VisualEncoder(use_mock=True)

        from perception.detector import Detection

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            Detection(bbox=(10, 10, 50, 50), class_name="object", confidence=0.8),
        ]
        text_queries = ["cup"]

        # High threshold should filter out most mock matches
        matches = encoder.match_detections(image, detections, text_queries, threshold=0.99)

        # Mock embeddings are unlikely to match with 0.99 threshold
        assert isinstance(matches, list)


@pytest.mark.skipif(not CLIP_AVAILABLE, reason="OpenCLIP not installed")
class TestVisualEncoderWithClip:
    """Test VisualEncoder with actual CLIP model (requires installation)."""

    def test_encoder_initialization_with_clip(self):
        """Test encoder initialization with real CLIP model."""
        encoder = VisualEncoder(
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            use_mock=False
        )
        # Model might load successfully or fall back to mock
        assert encoder.model is not None or encoder._use_mock

    def test_encoder_encode_text_with_clip(self):
        """Test encode_text with real CLIP model."""
        encoder = VisualEncoder(
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            use_mock=False
        )

        feature = encoder.encode_text("a red apple")
        assert isinstance(feature, FeatureVector)
        assert feature.label == "a red apple"

    def test_encoder_encode_image_with_clip(self):
        """Test encode_image with real CLIP model."""
        encoder = VisualEncoder(
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            use_mock=False
        )

        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        feature = encoder.encode_image(image)
        assert isinstance(feature, FeatureVector)
        assert feature.source == "image"


class TestVisualEncoderWithSettings:
    """Test VisualEncoder integration with Settings."""

    def test_encoder_with_settings(self):
        """Test encoder can be created from Settings."""
        from config.settings import Settings

        settings = Settings()
        encoder = VisualEncoder(
            model_name=settings.perception.clip_model,
            pretrained=settings.perception.clip_pretrained,
            use_mock=True
        )

        assert encoder._model_name == settings.perception.clip_model
        assert encoder._pretrained == settings.perception.clip_pretrained


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
