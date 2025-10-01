import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from abc import ABC, abstractmethod

class TextRecognition(ABC):
    def __init__(self, ocr_engine: str = "tesseract", language: str = "eng", confidence_threshold: float = 0.5):
        self.ocr_engine = ocr_engine
        self.language = language
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def detect_text(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        pass

    @abstractmethod
    def extract_text(self, image: np.ndarray, text_regions: Optional[List[Tuple[int, int, int, int]]] = None) -> List[str]:
        pass

    @abstractmethod
    def get_text_confidence(self, image: np.ndarray) -> List[float]:
        pass

    def recognize_text(self, image_path: str, preprocess: bool = True) -> Dict[str, Any]:
        return {
            'text_regions': [],
            'extracted_text': ['MockText'],
            'confidence_scores': [0.9],
            'full_text': 'MockText',
            'num_text_regions': 1
        }

class TextRecognizer(TextRecognition):
    def __init__(self):
        super().__init__()

    def detect_text(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        return [(0, 0, 100, 20)] # Mock bounding box

    def extract_text(self, image: np.ndarray, text_regions: Optional[List[Tuple[int, int, int, int]]] = None) -> List[str]:
        return ["MockText"]

    def get_text_confidence(self, image: np.ndarray) -> List[float]:
        return [0.9]

