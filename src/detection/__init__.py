"""
Detection Module - AI Computer Vision Platform

This module contains various detection algorithms and models for:
- Object detection (YOLO, R-CNN, SSD)
- Face detection
- Pose detection
- Real-time detection capabilities

Author: Gabriel Demetrios Lafis
License: MIT
"""

from .object_detector import ObjectDetector
from .face_detector import FaceDetector
from .pose_detector import PoseDetector

__all__ = [
    "ObjectDetector",
    "FaceDetector",
    "PoseDetector",
]
