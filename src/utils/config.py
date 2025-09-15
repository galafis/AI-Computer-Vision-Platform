"""Configuration Management Module

This module provides configuration management utilities for the AI Computer Vision Platform.
It handles loading, validation, and management of application settings from various sources
including environment variables, configuration files, and default values.

Classes:
    Config: Main configuration management class that handles all configuration operations

Author: AI Computer Vision Platform Team
Version: 1.0.0
Last Modified: September 2025
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path


class Config:
    """Configuration Management Class
    
    This class provides comprehensive configuration management for the AI Computer Vision Platform.
    It supports loading configurations from multiple sources including environment variables,
    JSON files, YAML files, and provides default values with validation.
    
    Attributes:
        config_data (Dict[str, Any]): Dictionary containing all configuration values
        config_file (Optional[str]): Path to the configuration file if loaded from file
        
    Example:
        >>> config = Config()
        >>> config.load_from_file('config.yaml')
        >>> api_key = config.get('api_key', 'default_key')
        >>> config.set('debug_mode', True)
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the Config class
        
        Args:
            config_file (Optional[str]): Path to configuration file to load on initialization
        """
        self.config_data: Dict[str, Any] = {}
        self.config_file: Optional[str] = config_file
        
        # Load default configuration
        self._load_defaults()
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
            
        # Override with environment variables
        self._load_from_env()
    
    def _load_defaults(self) -> None:
        """Load default configuration values"""
        self.config_data.update({
            'debug_mode': False,
            'log_level': 'INFO',
            'max_threads': 4,
            'timeout': 30,
            'cache_enabled': True,
            'cache_size': 1000,
            'model_path': './models',
            'output_path': './output',
            'temp_path': './temp'
        })
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables"""
        env_mappings = {
            'DEBUG_MODE': ('debug_mode', bool),
            'LOG_LEVEL': ('log_level', str),
            'MAX_THREADS': ('max_threads', int),
            'TIMEOUT': ('timeout', int),
            'CACHE_ENABLED': ('cache_enabled', bool),
            'CACHE_SIZE': ('cache_size', int),
            'MODEL_PATH': ('model_path', str),
            'OUTPUT_PATH': ('output_path', str),
            'TEMP_PATH': ('temp_path', str)
        }
        
        for env_var, (config_key, data_type) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    if data_type == bool:
                        self.config_data[config_key] = value.lower() in ('true', '1', 'yes', 'on')
                    elif data_type == int:
                        self.config_data[config_key] = int(value)
                    else:
                        self.config_data[config_key] = value
                except ValueError:
                    pass  # Skip invalid values
    
    def load_from_file(self, file_path: str) -> bool:
        """Load configuration from a file
        
        Args:
            file_path (str): Path to the configuration file (JSON or YAML)
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return False
                
            with open(path, 'r', encoding='utf-8') as file:
                if path.suffix.lower() in ['.yml', '.yaml']:
                    data = yaml.safe_load(file)
                elif path.suffix.lower() == '.json':
                    data = json.load(file)
                else:
                    return False
                    
            if isinstance(data, dict):
                self.config_data.update(data)
                self.config_file = file_path
                return True
                
        except Exception:
            pass
            
        return False
    
    def save_to_file(self, file_path: str, format_type: str = 'yaml') -> bool:
        """Save current configuration to a file
        
        Args:
            file_path (str): Path where to save the configuration
            format_type (str): Format type ('yaml' or 'json')
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as file:
                if format_type.lower() == 'json':
                    json.dump(self.config_data, file, indent=2)
                else:
                    yaml.dump(self.config_data, file, default_flow_style=False)
                    
            return True
            
        except Exception:
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value
        
        Args:
            key (str): Configuration key
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value or default
        """
        return self.config_data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value
        
        Args:
            key (str): Configuration key
            value (Any): Value to set
        """
        self.config_data[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with a dictionary
        
        Args:
            config_dict (Dict[str, Any]): Dictionary with configuration updates
        """
        self.config_data.update(config_dict)
    
    def has(self, key: str) -> bool:
        """Check if a configuration key exists
        
        Args:
            key (str): Configuration key to check
            
        Returns:
            bool: True if key exists, False otherwise
        """
        return key in self.config_data
    
    def remove(self, key: str) -> bool:
        """Remove a configuration key
        
        Args:
            key (str): Configuration key to remove
            
        Returns:
            bool: True if key was removed, False if key didn't exist
        """
        if key in self.config_data:
            del self.config_data[key]
            return True
        return False
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values
        
        Returns:
            Dict[str, Any]: Dictionary containing all configuration values
        """
        return self.config_data.copy()
    
    def clear(self) -> None:
        """Clear all configuration values"""
        self.config_data.clear()
        self._load_defaults()
    
    def validate(self) -> Dict[str, str]:
        """Validate current configuration
        
        Returns:
            Dict[str, str]: Dictionary with validation errors (empty if valid)
        """
        errors = {}
        
        # Validate required paths exist
        path_keys = ['model_path', 'output_path', 'temp_path']
        for key in path_keys:
            if key in self.config_data:
                path = Path(self.config_data[key])
                if not path.exists():
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        errors[key] = f"Cannot create directory: {path}"
        
        # Validate numeric ranges
        if self.get('max_threads', 1) < 1:
            errors['max_threads'] = "Must be at least 1"
            
        if self.get('timeout', 1) < 1:
            errors['timeout'] = "Must be at least 1 second"
            
        if self.get('cache_size', 1) < 1:
            errors['cache_size'] = "Must be at least 1"
        
        return errors
    
    def __str__(self) -> str:
        """String representation of the configuration"""
        return f"Config(file={self.config_file}, keys={len(self.config_data)})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the configuration"""
        return f"Config(config_data={self.config_data}, config_file='{self.config_file}')"
