"""Processing Package for AI Computer Vision Platform.

This package provides comprehensive image and video processing capabilities
including filtering, transformation, and analysis operations.

Modules:
    image_processor: Image processing operations and transformations
    video_processor: Video processing and frame manipulation
    filters: Collection of image filters and enhancement algorithms

Author: AI Computer Vision Platform Team
Version: 1.0.0
"""

from .image_processor import ImageProcessor
from .video_processor import VideoProcessor
from .filters import Filters

__all__ = [
    'ImageProcessor',
    'VideoProcessor', 
    'Filters'
]

__version__ = '1.0.0'
__author__ = 'AI Computer Vision Platform Team'
