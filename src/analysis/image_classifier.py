"""Image Classifier Module.

This module provides advanced image classification functionality using state-of-the-art
deep learning models. It supports multiple architectures including CNNs, Vision Transformers,
and custom models for various classification tasks.

Classes:
    ImageClassifier: Main class for image classification operations

Author: AI Computer Vision Platform Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
from pathlib import Path


class ImageClassifier(ABC):
    """Advanced image classification using deep learning models.
    
    This class provides a comprehensive interface for image classification
    tasks using various deep learning architectures. It supports multi-class
    classification, confidence scoring, and batch processing.
    
    Attributes:
        model_name (str): Name of the classification model
        num_classes (int): Number of classification classes
        class_names (List[str]): Names of the classification classes
        confidence_threshold (float): Minimum confidence score for valid predictions
        is_initialized (bool): Whether the model has been properly initialized
        input_size (Tuple[int, int]): Expected input image dimensions
    """
    
    def __init__(self, model_name: str = "BaseImageClassifier", 
                 num_classes: int = 1000,
                 confidence_threshold: float = 0.5,
                 input_size: Tuple[int, int] = (224, 224)):
        """Initialize the image classifier instance.
        
        Args:
            model_name (str): Name identifier for this model
            num_classes (int): Number of classes to classify
            confidence_threshold (float): Minimum confidence for valid predictions
            input_size (Tuple[int, int]): Expected input image size (width, height)
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.class_names = []
        self.confidence_threshold = confidence_threshold
        self.is_initialized = False
        self.input_size = input_size
        self._model = None
        self._preprocessing_params = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the image classification model.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
        
    @abstractmethod
    def classify_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Classify a single image.
        
        Args:
            image (np.ndarray): Input image array
            
        Returns:
            Dict[str, Any]: Classification results containing:
                - 'predictions': List of class predictions
                - 'probabilities': Probability scores for each class
                - 'top_class': Most likely class name
                - 'confidence': Confidence score of top prediction
        """
        pass
        
    @abstractmethod
    def classify_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Classify multiple images in batch.
        
        Args:
            images (List[np.ndarray]): List of input images
            
        Returns:
            List[Dict[str, Any]]: List of classification results for each image
        """
        pass
        
    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for classification.
        
        Args:
            image (np.ndarray): Raw input image
            
        Returns:
            np.ndarray: Preprocessed image ready for classification
        """
        pass
        
    def validate_image(self, image: np.ndarray) -> bool:
        """Validate input image format and dimensions.
        
        Args:
            image (np.ndarray): Input image to validate
            
        Returns:
            bool: True if image is valid, False otherwise
        """
        if image is None:
            return False
            
        if not isinstance(image, np.ndarray):
            return False
            
        if len(image.shape) not in [2, 3]:
            return False
            
        if image.size == 0:
            return False
            
        return True
        
    def load_class_names(self, class_names_path: Union[str, Path]) -> bool:
        """Load class names from file.
        
        Args:
            class_names_path (Union[str, Path]): Path to class names file
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            path = Path(class_names_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
                return len(self.class_names) == self.num_classes
            return False
        except Exception:
            return False
            
    def set_confidence_threshold(self, threshold: float) -> None:
        """Update the confidence threshold for predictions.
        
        Args:
            threshold (float): New confidence threshold (0.0 to 1.0)
            
        Raises:
            ValueError: If threshold is not between 0.0 and 1.0
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
            
    def get_top_k_predictions(self, probabilities: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Get top-k predictions from probability scores.
        
        Args:
            probabilities (np.ndarray): Probability scores for all classes
            k (int): Number of top predictions to return
            
        Returns:
            List[Dict[str, Any]]: Top-k predictions with class names and scores
        """
        if len(probabilities) != self.num_classes:
            raise ValueError(f"Probabilities length {len(probabilities)} doesn't match num_classes {self.num_classes}")
            
        top_k_indices = np.argsort(probabilities)[::-1][:k]
        top_k_predictions = []
        
        for idx in top_k_indices:
            prediction = {
                'class_index': int(idx),
                'class_name': self.class_names[idx] if self.class_names else f"class_{idx}",
                'probability': float(probabilities[idx]),
                'confidence': float(probabilities[idx])
            }
            top_k_predictions.append(prediction)
            
        return top_k_predictions
        
    def resize_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Resize image to target dimensions.
        
        Args:
            image (np.ndarray): Input image
            target_size (Optional[Tuple[int, int]]): Target size (width, height)
                                                   If None, uses self.input_size
            
        Returns:
            np.ndarray: Resized image
        """
        if target_size is None:
            target_size = self.input_size
            
        return cv2.resize(image, target_size)
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dict[str, Any]: Model information including name, status, and settings
        """
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'is_initialized': self.is_initialized,
            'confidence_threshold': self.confidence_threshold,
            'input_size': self.input_size,
            'has_class_names': bool(self.class_names)
        }
        
    def __repr__(self) -> str:
        """String representation of the image classifier instance."""
        return (f"{self.__class__.__name__}(model_name='{self.model_name}', "
                f"num_classes={self.num_classes}, "
                f"confidence_threshold={self.confidence_threshold}, "
                f"initialized={self.is_initialized})")
