"""Gesture Recognition Module.

This module provides the base class and utilities for gesture recognition
functionality within the AI Computer Vision Platform. It defines the core
interface and common methods for implementing various gesture recognition
algorithms and techniques.

Classes:
    GestureRecognition: Base class for gesture recognition implementations

Author: AI Computer Vision Platform Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class GestureRecognition(ABC):
    """Base class for gesture recognition implementations.
    
    This abstract base class defines the standard interface for gesture
    recognition algorithms. All concrete gesture recognition implementations
    should inherit from this class and implement the required abstract methods.
    
    Attributes:
        model_name (str): Name of the gesture recognition model
        confidence_threshold (float): Minimum confidence score for valid detections
        is_initialized (bool): Whether the model has been properly initialized
    """
    
    def __init__(self, model_name: str = "BaseGestureRecognition", 
                 confidence_threshold: float = 0.5):
        """Initialize the gesture recognition instance.
        
        Args:
            model_name (str): Name identifier for this model
            confidence_threshold (float): Minimum confidence for valid detections
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.is_initialized = False
        self._model = None
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the gesture recognition model.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
        
    @abstractmethod
    def recognize_gesture(self, frame: np.ndarray) -> Dict[str, Any]:
        """Recognize gestures in the given frame.
        
        Args:
            frame (np.ndarray): Input image frame
            
        Returns:
            Dict[str, Any]: Recognition results containing:
                - 'gestures': List of detected gestures
                - 'confidence_scores': Confidence scores for each gesture
                - 'bounding_boxes': Bounding boxes if applicable
                - 'landmarks': Hand/body landmarks if available
        """
        pass
        
    @abstractmethod
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess the input frame for gesture recognition.
        
        Args:
            frame (np.ndarray): Raw input frame
            
        Returns:
            np.ndarray: Preprocessed frame ready for recognition
        """
        pass
        
    def validate_frame(self, frame: np.ndarray) -> bool:
        """Validate input frame format and dimensions.
        
        Args:
            frame (np.ndarray): Input frame to validate
            
        Returns:
            bool: True if frame is valid, False otherwise
        """
        if frame is None:
            return False
            
        if not isinstance(frame, np.ndarray):
            return False
            
        if len(frame.shape) not in [2, 3]:
            return False
            
        if frame.size == 0:
            return False
            
        return True
        
    def set_confidence_threshold(self, threshold: float) -> None:
        """Update the confidence threshold for gesture detection.
        
        Args:
            threshold (float): New confidence threshold (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dict[str, Any]: Model information including name, status, and settings
        """
        return {
            'model_name': self.model_name,
            'is_initialized': self.is_initialized,
            'confidence_threshold': self.confidence_threshold
        }
        
    def __repr__(self) -> str:
        """String representation of the gesture recognition instance."""
        return (f"{self.__class__.__name__}(model_name='{self.model_name}', "
                f"confidence_threshold={self.confidence_threshold}, "
                f"initialized={self.is_initialized})")
