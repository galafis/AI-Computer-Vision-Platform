import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from .emotion_analyzer import EmotionAnalyzer, EmotionType

class EmotionAnalyzerImpl(EmotionAnalyzer):
    def __init__(self, model_name: str = "MockEmotionAnalyzer", confidence_threshold: float = 0.6):
        super().__init__(model_name, confidence_threshold)
        self.is_initialized = True

    def initialize(self) -> bool:
        print(f"Mock EmotionAnalyzer {self.model_name} initialized.")
        return True

    def analyze_emotion(self, image: np.ndarray, face_region: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        print(f"Mock EmotionAnalyzer {self.model_name} analyzing emotion.")
        # Mock implementation: always returns 'happiness'
        return {
            'emotions': {EmotionType.HAPPINESS.value: 0.95, EmotionType.NEUTRAL.value: 0.05},
            'primary_emotion': EmotionType.HAPPINESS.value,
            'confidence': 0.95,
            'face_detected': True,
            'landmarks': [],
            'valence': 0.8,
            'arousal': 0.6
        }

    def analyze_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        return [self.analyze_emotion(img) for img in images]

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        return [(10, 10, 100, 100)] # Mock face detection

    def extract_facial_landmarks(self, image: np.ndarray, face_region: Tuple[int, int, int, int]) -> np.ndarray:
        return np.array([]) # Mock landmarks
