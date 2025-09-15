"""Logger module for AI Computer Vision Platform.

This module provides logging functionality with configurable levels
and formatters for the AI Computer Vision Platform.

Author: AI Computer Vision Platform Team
Date: September 2025
"""

import logging
import os
from datetime import datetime
from typing import Optional


class Logger:
    """Logger class for handling application logging.
    
    This class provides a centralized logging mechanism with
    configurable log levels, formatters, and output destinations.
    
    Attributes:
        name (str): The name of the logger instance.
        level (str): The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        logger (logging.Logger): The underlying Python logger instance.
    
    Example:
        >>> logger = Logger("my_app")
        >>> logger.info("Application started")
        >>> logger.error("An error occurred")
    """
    
    def __init__(self, name: str = "ai_cv_platform", level: str = "INFO"):
        """Initialize the Logger instance.
        
        Args:
            name (str, optional): The name for the logger. Defaults to "ai_cv_platform".
            level (str, optional): The logging level. Defaults to "INFO".
        
        Raises:
            ValueError: If an invalid logging level is provided.
        """
        self.name = name
        self.level = level.upper()
        
        # Create logger instance
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(getattr(logging, self.level, logging.INFO))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up logging handlers with appropriate formatters."""
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.level, logging.INFO))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str):
        """Log a debug message.
        
        Args:
            message (str): The debug message to log.
        """
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log an info message.
        
        Args:
            message (str): The info message to log.
        """
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log a warning message.
        
        Args:
            message (str): The warning message to log.
        """
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log an error message.
        
        Args:
            message (str): The error message to log.
        """
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log a critical message.
        
        Args:
            message (str): The critical message to log.
        """
        self.logger.critical(message)
