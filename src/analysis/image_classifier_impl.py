import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from .image_classifier import ImageClassifier

class ImageClassifierImpl(ImageClassifier):
    def __init__(self, model_name: str = "MockImageClassifier", num_classes: int = 1000, confidence_threshold: float = 0.5, input_size: Tuple[int, int] = (224, 224)):
        super().__init__(model_name, num_classes, confidence_threshold, input_size)
        self.is_initialized = True
        self.class_names = [f"class_{i}" for i in range(num_classes)]

    def initialize(self) -> bool:
        print(f"Mock ImageClassifier {self.model_name} initialized.")
        return True

    def classify_image(self, image: np.ndarray) -> Dict[str, Any]:
        print(f"Mock ImageClassifier {self.model_name} classifying image.")
        # Mock implementation: always returns a fixed class
        return {
            'predictions': [self.class_names[0]],
            'probabilities': [0.99],
            'top_class': self.class_names[0],
            'confidence': 0.99
        }

    def classify_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        return [self.classify_image(img) for img in images]

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        print(f"Mock ImageClassifier {self.model_name} preprocessing image.")
        return image # No actual preprocessing for mock

