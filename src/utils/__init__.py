"""Utils Package

This package provides utility classes and functions for the AI Computer Vision Platform.
It includes configuration management, logging, and helper utilities.

Modules:
    config: Configuration management utilities
    logger: Logging system implementation
    helpers: General helper functions and utilities

Author: AI Computer Vision Platform Team
Version: 1.0.0
"""

from .config import Config
from .logger import Logger
from .helpers import Helpers

__all__ = ['Config', 'Logger', 'Helpers']
__version__ = '1.0.0'
