"""
Recognition module for AI Computer Vision Platform.

This module contains classes and functions for various recognition tasks including:
- Face recognition
- Text recognition (OCR)
- Gesture recognition

Author: Gabriel Demetrios Lafis
Email: gabrieldemetrios@gmail.com
Date: September 2025
"""

from .face_recognition import FaceRecognition, FaceRecognizer
from .text_recognition import TextRecognition, TextRecognizer
from .gesture_recognition import GestureRecognition
from .gesture_recognizer_impl import GestureRecognizer

__all__ = [
    'FaceRecognition',
    'FaceRecognizer',
    'TextRecognition',
    'TextRecognizer',
    'GestureRecognition',
    'GestureRecognizer'
]

