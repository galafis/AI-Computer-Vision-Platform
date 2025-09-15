"""Analysis Package.

This package provides comprehensive analysis and classification functionality
for the AI Computer Vision Platform. It includes modules for image classification,
emotion analysis, and scene understanding.

Modules:
    image_classifier: Advanced image classification using deep learning models
    emotion_analyzer: Emotion detection and analysis from facial expressions
    scene_analyzer: Scene understanding and contextual analysis

Author: AI Computer Vision Platform Team
Version: 1.0.0
"""

from .image_classifier import ImageClassifier
from .emotion_analyzer import EmotionAnalyzer
from .scene_analyzer import SceneAnalyzer

__all__ = [
    'ImageClassifier',
    'EmotionAnalyzer',
    'SceneAnalyzer'
]

__version__ = '1.0.0'
__author__ = 'AI Computer Vision Platform Team'
