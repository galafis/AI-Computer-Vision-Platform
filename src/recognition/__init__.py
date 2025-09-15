"""Recognition module for AI Computer Vision Platform.

This module contains classes and functions for various recognition tasks including:
- Face recognition
- Text recognition (OCR)
- Gesture recognition

Author: Gabriel Demetrios Lafis
Email: gabrieldemetrios@gmail.com
Date: September 2025
"""

from .face_recognition import FaceRecognition
from .text_recognition import TextRecognition
from .gesture_recognition import GestureRecognition

__all__ = [
    'FaceRecognition',
    'TextRecognition', 
    'GestureRecognition'
]
