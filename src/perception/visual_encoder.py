"""
CLIP visual encoder for the Vision-Language Embodied Agent.

This module provides a FeatureVector dataclass and VisualEncoder class that wrap
OpenCLIP functionality for encoding images and text into shared embedding space.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, TYPE_CHECKING

# Handle open_clip import gracefully
try:
    import open_clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    open_clip = None
    torch = None

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image


@dataclass
class FeatureVector:
    """
    Represents a feature vector from the visual encoder.

    Attributes:
        vector: The embedding vector as a numpy array.
        source: Description of the source (e.g., "image" or "text:label").
        label: Optional label for the feature (e.g., class name).
    """
    vector: "np.ndarray"
    source: str
    label: Optional[str] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the feature vector."""
        return self.vector.shape

    @property
    def dimension(self) -> int:
        """Get the dimension of the feature vector."""
        return len(self.vector)


class VisualEncoder:
    """
    CLIP-based visual encoder for the embodied agent.

    This class wraps the OpenCLIP model and provides methods for encoding
    images and text into a shared embedding space for similarity matching.

    Attributes:
        model: The OpenCLIP model instance.
        preprocess: Image preprocessing pipeline.
        tokenizer: Text tokenizer.
        device: Device for inference (CPU or CUDA).
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
        use_mock: bool = False
    ):
        """
        Initialize the VisualEncoder.

        Args:
            model_name: Name of the CLIP model architecture.
            pretrained: Name of the pretrained weights.
            device: Device for inference ("cuda", "cpu", or None for auto).
            use_mock: If True, use mock mode without loading actual model.
        """
        self._model_name = model_name
        self._pretrained = pretrained
        self._use_mock = use_mock
        self._model: Optional[Any] = None
        self._preprocess: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

        # Determine device
        if device is None:
            if CLIP_AVAILABLE and torch is not None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = "cpu"
        else:
            self._device = device

        if not use_mock and CLIP_AVAILABLE:
            try:
                self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                    model_name,
                    pretrained=pretrained
                )
                self._model = self._model.to(self._device)
                self._model.eval()
                self._tokenizer = open_clip.get_tokenizer(model_name)
            except Exception:
                # Fall back to mock mode if model loading fails
                self._use_mock = True

    @property
    def model(self) -> Optional[Any]:
        """Get the underlying OpenCLIP model, if available."""
        return self._model

    @property
    def device(self) -> str:
        """Get the device used for inference."""
        return self._device

    @property
    def embed_dim(self) -> int:
        """Get the embedding dimension of the model."""
        if self._use_mock or self._model is None:
            return 512  # Default CLIP embedding dimension
        return self._model.visual.output_dim

    def encode_text(self, text: str) -> FeatureVector:
        """
        Encode text into a feature vector.

        Args:
            text: Input text string to encode.

        Returns:
            FeatureVector containing the text embedding.
        """
        if self._use_mock or self._model is None or self._tokenizer is None:
            return self._encode_text_mock(text)

        try:
            import numpy as np

            tokens = self._tokenizer([text])
            tokens = tokens.to(self._device)

            with torch.no_grad():
                text_features = self._model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            vector = text_features.cpu().numpy()[0]
            return FeatureVector(
                vector=vector,
                source=f"text:{text}",
                label=text
            )

        except Exception:
            return self._encode_text_mock(text)

    def encode_image(self, image: "np.ndarray") -> FeatureVector:
        """
        Encode a single image into a feature vector.

        Args:
            image: Input image as numpy array (H, W, C) in RGB format.

        Returns:
            FeatureVector containing the image embedding.
        """
        if self._use_mock or self._model is None or self._preprocess is None:
            return self._encode_image_mock(image)

        try:
            import numpy as np
            from PIL import Image as PILImage

            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = image

            # Preprocess image
            image_tensor = self._preprocess(pil_image).unsqueeze(0).to(self._device)

            with torch.no_grad():
                image_features = self._model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            vector = image_features.cpu().numpy()[0]
            return FeatureVector(
                vector=vector,
                source="image",
                label=None
            )

        except Exception:
            return self._encode_image_mock(image)

    def encode_images(self, images: List["np.ndarray"]) -> List[FeatureVector]:
        """
        Encode multiple images into feature vectors.

        Args:
            images: List of input images as numpy arrays (H, W, C) in RGB format.

        Returns:
            List of FeatureVector objects for each image.
        """
        return [self.encode_image(img) for img in images]

    def compute_similarity(
        self,
        feature1: FeatureVector,
        feature2: FeatureVector
    ) -> float:
        """
        Compute cosine similarity between two feature vectors.

        Args:
            feature1: First feature vector.
            feature2: Second feature vector.

        Returns:
            Cosine similarity score (between -1 and 1, typically 0-1 for normalized features).
        """
        import numpy as np

        v1 = feature1.vector
        v2 = feature2.vector

        # Cosine similarity
        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(similarity)

    def find_best_match(
        self,
        query: FeatureVector,
        candidates: List[FeatureVector],
        threshold: Optional[float] = None
    ) -> Tuple[Optional[FeatureVector], float]:
        """
        Find the best matching candidate for a query feature.

        Args:
            query: Query feature vector.
            candidates: List of candidate feature vectors.
            threshold: Optional minimum similarity threshold.

        Returns:
            Tuple of (best matching FeatureVector or None, similarity score).
            Returns (None, 0.0) if no candidates meet the threshold.
        """
        if not candidates:
            return None, 0.0

        best_match: Optional[FeatureVector] = None
        best_score: float = -1.0

        for candidate in candidates:
            score = self.compute_similarity(query, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate

        if threshold is not None and best_score < threshold:
            return None, best_score

        return best_match, best_score

    def match_detections(
        self,
        image: "np.ndarray",
        detections: List["Detection"],
        text_queries: List[str],
        threshold: float = 0.25
    ) -> List[Tuple["Detection", str, float]]:
        """
        Match detections to text queries based on visual similarity.

        Args:
            image: Source image as numpy array.
            detections: List of Detection objects to match.
            text_queries: List of text query strings.
            threshold: Minimum similarity threshold for matches.

        Returns:
            List of (Detection, matched_query, similarity_score) tuples
            for matches above the threshold.
        """
        # Encode text queries
        text_features = [self.encode_text(q) for q in text_queries]

        # Encode each detection crop
        matches: List[Tuple["Detection", str, float]] = []

        for detection in detections:
            crop = detection.crop(image)
            image_feature = self.encode_image(crop)

            # Find best matching text query
            best_match, score = self.find_best_match(
                image_feature,
                text_features,
                threshold=threshold
            )

            if best_match is not None and best_match.label is not None:
                matches.append((detection, best_match.label, score))

        return matches

    def _encode_text_mock(self, text: str) -> FeatureVector:
        """
        Generate a mock text embedding for testing.

        Args:
            text: Input text string.

        Returns:
            Mock FeatureVector with deterministic embedding.
        """
        import numpy as np

        # Generate deterministic mock embedding based on text hash
        seed = hash(text) % (2**31)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(512).astype(np.float32)
        vector = vector / np.linalg.norm(vector)

        return FeatureVector(
            vector=vector,
            source=f"text:{text}",
            label=text
        )

    def _encode_image_mock(self, image: "np.ndarray") -> FeatureVector:
        """
        Generate a mock image embedding for testing.

        Args:
            image: Input image as numpy array.

        Returns:
            Mock FeatureVector with deterministic embedding.
        """
        import numpy as np

        # Generate deterministic mock embedding based on image shape
        seed = hash(tuple(image.shape)) % (2**31)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(512).astype(np.float32)
        vector = vector / np.linalg.norm(vector)

        return FeatureVector(
            vector=vector,
            source="image",
            label=None
        )


# Note: Detection is imported at runtime in match_detections to avoid circular import
