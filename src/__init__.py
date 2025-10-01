"""
AI Computer Vision Platform - Core Package

This package contains all the core modules for the AI Computer Vision Platform,
including detection, recognition, analysis, processing, and utility modules.

Author: Gabriel Demetrios Lafis
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Gabriel Demetrios Lafis"
__email__ = "gabrieldemetrios@gmail.com"

# Core modules
from . import detection
from . import recognition
from . import analysis
from . import processing
from . import utils

__all__ = [
    "detection",
    "recognition", 
    "analysis",
    "processing",
    "utils",
]

