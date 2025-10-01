import numpy as np
import cv2
from typing import Optional, Tuple, Union
from pathlib import Path


class ImageProcessor:
    """A comprehensive image processing class for computer vision applications.
    
    This class provides methods for loading, transforming, enhancing, and
    analyzing digital images. It supports various image formats and offers
    both basic and advanced processing operations.
    
    Attributes:
        image_data (np.ndarray): The current image data as a numpy array
        original_shape (tuple): The original dimensions of the loaded image
        color_mode (str): The color mode of the image ('RGB', 'BGR', 'GRAY')
        
    Example:
        >>> processor = ImageProcessor()
        >>> processor.load_image('path/to/image.jpg')
        >>> resized_image = processor.resize_image((640, 480))
        >>> processor.save_image('output/resized_image.jpg')
    """
    
    def __init__(self):
        """Initialize the ImageProcessor with default settings."""
        self.image_data: Optional[np.ndarray] = None
        self.original_shape: Optional[Tuple[int, ...]] = None
        self.color_mode: str = 'RGB'
        
    def load_image(self, image_path: Union[str, Path]) -> bool:
        """Load an image from file.
        
        Args:
            image_path (Union[str, Path]): Path to the image file
            
        Returns:
            bool: True if image loaded successfully, False otherwise
            
        Raises:
            FileNotFoundError: If the image file does not exist
            ValueError: If the file format is not supported
        """
        print(f"Mock loading image from {image_path}")
        # Simulate loading an image
        self.image_data = np.zeros((100, 100, 3), dtype=np.uint8)
        self.original_shape = self.image_data.shape
        self.color_mode = 'RGB'
        return True
        
    def save_image(self, output_path: Union[str, Path], quality: int = 95) -> bool:
        """Save the current image to file.
        
        Args:
            output_path (Union[str, Path]): Path where to save the image
            quality (int): Image quality for JPEG files (1-100)
            
        Returns:
            bool: True if image saved successfully, False otherwise
        """
        if self.image_data is None:
            return False
        print(f"Mock saving image to {output_path} with quality {quality}")
        # Simulate saving an image
        return True
        
    def resize_image(self, new_size: Tuple[int, int], 
                    interpolation: str = 'bilinear') -> np.ndarray:
        """Resize the image to specified dimensions.
        
        Args:
            new_size (Tuple[int, int]): Target size as (width, height)
            interpolation (str): Interpolation method ('nearest', 'bilinear', 'bicubic')
            
        Returns:
            np.ndarray: Resized image array
            
        Raises:
            ValueError: If no image is loaded or invalid parameters
        """
        if self.image_data is None:
            raise ValueError("No image loaded to resize.")
        print(f"Mock resizing image to {new_size} with interpolation {interpolation}")
        # Simulate resizing
        self.image_data = np.zeros((new_size[1], new_size[0], 3), dtype=np.uint8)
        return self.image_data
        
    def rotate_image(self, angle: float, expand: bool = True) -> np.ndarray:
        """Rotate the image by specified angle.
        
        Args:
            angle (float): Rotation angle in degrees (positive = clockwise)
            expand (bool): Whether to expand the image to fit the rotated content
            
        Returns:
            np.ndarray: Rotated image array
        """
        if self.image_data is None:
            raise ValueError("No image loaded to rotate.")
        print(f"Mock rotating image by {angle} degrees, expand={expand}")
        # Simulate rotation
        return self.image_data.copy()
        
    def adjust_brightness(self, factor: float) -> np.ndarray:
        """Adjust image brightness.
        
        Args:
            factor (float): Brightness factor (1.0 = no change, >1.0 = brighter)
            
        Returns:
            np.ndarray: Brightness-adjusted image array
        """
        if self.image_data is None:
            raise ValueError("No image loaded to adjust brightness.")
        print(f"Mock adjusting brightness by factor {factor}")
        # Simulate brightness adjustment
        return self.image_data.copy()
        
    def adjust_contrast(self, factor: float) -> np.ndarray:
        """Adjust image contrast.
        
        Args:
            factor (float): Contrast factor (1.0 = no change, >1.0 = more contrast)
            
        Returns:
            np.ndarray: Contrast-adjusted image array
        """
        if self.image_data is None:
            raise ValueError("No image loaded to adjust contrast.")
        print(f"Mock adjusting contrast by factor {factor}")
        # Simulate contrast adjustment
        return self.image_data.copy()
        
    def convert_color_space(self, target_mode: str) -> np.ndarray:
        """Convert image to different color space.
        
        Args:
            target_mode (str): Target color space ('RGB', 'BGR', 'GRAY', 'HSV')
            
        Returns:
            np.ndarray: Color-converted image array
            
        Raises:
            ValueError: If target color space is not supported
        """
        if self.image_data is None:
            raise ValueError("No image loaded to convert color space.")
        if target_mode not in ['RGB', 'BGR', 'GRAY', 'HSV']:
            raise ValueError(f"Unsupported color mode: {target_mode}")
        print(f"Mock converting color space to {target_mode}")
        # Simulate color space conversion
        if target_mode == 'GRAY':
            return np.zeros(self.image_data.shape[:2], dtype=np.uint8)
        return self.image_data.copy()
        
    def get_image_info(self) -> dict:
        """Get information about the current image.
        
        Returns:
            dict: Dictionary containing image metadata and statistics
        """
        if self.image_data is None:
            return {"status": "No image loaded"}
        return {
            "shape": self.image_data.shape,
            "original_shape": self.original_shape,
            "color_mode": self.color_mode,
            "size_bytes": self.image_data.nbytes
        }
        
    def reset(self) -> None:
        """Reset the processor to initial state."""
        self.image_data = None
        self.original_shape = None
        self.color_mode = 'RGB'

