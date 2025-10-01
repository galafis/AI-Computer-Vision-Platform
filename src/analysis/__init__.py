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
from .image_classifier_impl import ImageClassifierImpl
from .emotion_analyzer import EmotionAnalyzer
from .emotion_analyzer_impl import EmotionAnalyzerImpl
from .scene_analyzer import SceneAnalyzer
from .scene_analyzer_impl import SceneAnalyzerImpl

__all__ = [
    'ImageClassifier',
    'ImageClassifierImpl',
    'EmotionAnalyzer',
    'EmotionAnalyzerImpl',
    'SceneAnalyzer',
    'SceneAnalyzerImpl'
]

__version__ = '1.0.0'
__author__ = 'AI Computer Vision Platform Team'

