"""Helper utilities module for AI Computer Vision Platform.

This module provides utility functions and helper classes that support
various operations across the AI Computer Vision Platform.

Author: AI Computer Vision Platform Team
Date: September 2025
"""

import os
import json
import hashlib
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class Helpers:
    """Helper class providing various utility functions.
    
    This class contains static and instance methods for common operations
    such as file handling, data validation, string manipulation, and
    other utility functions used throughout the platform.
    
    Example:
        >>> helpers = Helpers()
        >>> is_valid = helpers.validate_file_extension("image.jpg", [".jpg", ".png"])
        >>> hash_val = Helpers.generate_hash("some_data")
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize the Helpers instance.
        
        Args:
            base_path (str, optional): Base path for file operations.
                                     Defaults to current working directory.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.timestamp = datetime.now()
    
    def validate_file_extension(self, filename: str, allowed_extensions: List[str]) -> bool:
        """Validate if a file has an allowed extension.
        
        Args:
            filename (str): The filename to validate.
            allowed_extensions (List[str]): List of allowed extensions (e.g., [".jpg", ".png"]).
        
        Returns:
            bool: True if the file extension is allowed, False otherwise.
        
        Example:
            >>> helpers = Helpers()
            >>> helpers.validate_file_extension("image.jpg", [".jpg", ".png"])
            True
        """
        if not filename or not isinstance(filename, str):
            return False
        
        file_ext = Path(filename).suffix.lower()
        return file_ext in [ext.lower() for ext in allowed_extensions]
    
    def ensure_directory_exists(self, directory_path: Union[str, Path]) -> bool:
        """Ensure a directory exists, create if it doesn't.
        
        Args:
            directory_path (Union[str, Path]): Path to the directory.
        
        Returns:
            bool: True if directory exists or was created successfully.
        
        Raises:
            OSError: If directory creation fails.
        """
        try:
            path = Path(directory_path)
            path.mkdir(parents=True, exist_ok=True)
            return True
        except OSError as e:
            raise OSError(f"Failed to create directory {directory_path}: {e}")
    
    def read_json_file(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Read and parse a JSON file.
        
        Args:
            file_path (Union[str, Path]): Path to the JSON file.
        
        Returns:
            Optional[Dict[str, Any]]: Parsed JSON data or None if error occurs.
        
        Example:
            >>> helpers = Helpers()
            >>> data = helpers.read_json_file("config.json")
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
    
    def write_json_file(self, data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
        """Write data to a JSON file.
        
        Args:
            data (Dict[str, Any]): Data to write to the file.
            file_path (Union[str, Path]): Path to the JSON file.
        
        Returns:
            bool: True if successful, False otherwise.
        
        Example:
            >>> helpers = Helpers()
            >>> success = helpers.write_json_file({"key": "value"}, "output.json")
        """
        try:
            # Ensure directory exists
            file_path = Path(file_path)
            self.ensure_directory_exists(file_path.parent)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
            return True
        except (OSError, TypeError):
            return False
    
    @staticmethod
    def generate_hash(data: Union[str, bytes], algorithm: str = "sha256") -> str:
        """Generate a hash for the given data.
        
        Args:
            data (Union[str, bytes]): Data to hash.
            algorithm (str, optional): Hash algorithm to use. Defaults to "sha256".
        
        Returns:
            str: Hexadecimal representation of the hash.
        
        Raises:
            ValueError: If an unsupported algorithm is specified.
        
        Example:
            >>> hash_val = Helpers.generate_hash("some_data")
            >>> print(hash_val)
        """
        if algorithm not in hashlib.algorithms_available:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(data)
        return hash_obj.hexdigest()
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> Optional[int]:
        """Get the size of a file in bytes.
        
        Args:
            file_path (Union[str, Path]): Path to the file.
        
        Returns:
            Optional[int]: File size in bytes, or None if file doesn't exist.
        
        Example:
            >>> size = Helpers.get_file_size("image.jpg")
            >>> print(f"File size: {size} bytes")
        """
        try:
            return Path(file_path).stat().st_size
        except (FileNotFoundError, OSError):
            return None
    
    @staticmethod
    def format_bytes(bytes_size: int, decimal_places: int = 2) -> str:
        """Format bytes into human readable format.
        
        Args:
            bytes_size (int): Size in bytes.
            decimal_places (int, optional): Number of decimal places. Defaults to 2.
        
        Returns:
            str: Formatted size string (e.g., "1.5 MB").
        
        Example:
            >>> formatted = Helpers.format_bytes(1536000)
            >>> print(formatted)  # "1.46 MB"
        """
        if bytes_size == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB", "PB"]
        i = 0
        while bytes_size >= 1024.0 and i < len(size_names) - 1:
            bytes_size /= 1024.0
            i += 1
        
        return f"{bytes_size:.{decimal_places}f} {size_names[i]}"
    
    @staticmethod
    def get_timestamp(format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Get current timestamp as formatted string.
        
        Args:
            format_string (str, optional): Timestamp format. 
                                         Defaults to "%Y-%m-%d %H:%M:%S".
        
        Returns:
            str: Formatted timestamp string.
        
        Example:
            >>> timestamp = Helpers.get_timestamp()
            >>> print(timestamp)  # "2025-09-15 15:30:45"
        """
        return datetime.now().strftime(format_string)
    
    def get_relative_path(self, target_path: Union[str, Path]) -> str:
        """Get relative path from base path to target path.
        
        Args:
            target_path (Union[str, Path]): Target path.
        
        Returns:
            str: Relative path string.
        
        Example:
            >>> helpers = Helpers(base_path="/home/user")
            >>> rel_path = helpers.get_relative_path("/home/user/documents/file.txt")
            >>> print(rel_path)  # "documents/file.txt"
        """
        try:
            return str(Path(target_path).relative_to(self.base_path))
        except ValueError:
            return str(Path(target_path).absolute())
