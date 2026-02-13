import numpy as np
from typing import Optional, Tuple, Union
from enum import Enum


class FilterType(Enum):
    """Enumeration of available filter types."""
    GAUSSIAN_BLUR = "gaussian_blur"
    BOX_BLUR = "box_blur"
    MEDIAN = "median"
    SHARPEN = "sharpen"
    EDGE_DETECTION = "edge_detection"
    EMBOSS = "emboss"
    SOBEL_X = "sobel_x"
    SOBEL_Y = "sobel_y"
    LAPLACIAN = "laplacian"
    BILATERAL = "bilateral"


class Filters:
    """A comprehensive collection of image filters for computer vision applications.
    
    This class provides various filtering operations including blurring, sharpening,
    edge detection, noise reduction, and artistic effects. All filters operate on
    numpy arrays representing image data.
    
    Example:
        >>> filters = Filters()
        >>> blurred_image = filters.gaussian_blur(image, kernel_size=5, sigma=1.5)
        >>> edges = filters.sobel_edge_detection(image)
        >>> sharpened = filters.sharpen(image, intensity=1.2)
    """
    
    def __init__(self):
        """Initialize the Filters with default settings."""
        self._kernel_cache = {}
        
    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5, 
                     sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian blur filter to the image.
        
        Args:
            image (np.ndarray): Input image array
            kernel_size (int): Size of the Gaussian kernel (must be odd)
            sigma (float): Standard deviation for Gaussian kernel
            
        Returns:
            np.ndarray: Blurred image array
            
        Raises:
            ValueError: If kernel_size is even or sigma is negative
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if sigma < 0:
            raise ValueError("sigma cannot be negative")
        print(f"Mock applying Gaussian blur with kernel_size={kernel_size}, sigma={sigma}")
        return image.copy() # Mock implementation
        
    def box_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply box blur (average) filter to the image.
        
        Args:
            image (np.ndarray): Input image array
            kernel_size (int): Size of the box kernel (must be odd)
            
        Returns:
            np.ndarray: Blurred image array
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        print(f"Mock applying box blur with kernel_size={kernel_size}")
        return image.copy() # Mock implementation
        
    def median_filter(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply median filter for noise reduction.
        
        Args:
            image (np.ndarray): Input image array
            kernel_size (int): Size of the median kernel (must be odd)
            
        Returns:
            np.ndarray: Filtered image array
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        print(f"Mock applying median filter with kernel_size={kernel_size}")
        return image.copy() # Mock implementation
        
    def sharpen(self, image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """Apply sharpening filter to enhance image details.
        
        Args:
            image (np.ndarray): Input image array
            intensity (float): Sharpening intensity (1.0 = standard sharpening)
            
        Returns:
            np.ndarray: Sharpened image array
        """
        print(f"Mock applying sharpen filter with intensity={intensity}")
        return image.copy() # Mock implementation
        
    def sobel_edge_detection(self, image: np.ndarray, 
                           direction: str = 'both') -> np.ndarray:
        """Apply Sobel edge detection filter.
        
        Args:
            image (np.ndarray): Input image array (should be grayscale)
            direction (str): Edge direction ('x', 'y', or 'both')
            
        Returns:
            np.ndarray: Edge-detected image array
            
        Raises:
            ValueError: If direction is not valid
        """
        if direction not in ['x', 'y', 'both']:
            raise ValueError("Direction must be 'x', 'y', or 'both'")
        print(f"Mock applying Sobel edge detection in {direction} direction")
        return np.zeros_like(image) # Mock implementation
        
    def laplacian_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """Apply Laplacian edge detection filter.
        
        Args:
            image (np.ndarray): Input image array (should be grayscale)
            
        Returns:
            np.ndarray: Edge-detected image array
        """
        print("Mock applying Laplacian edge detection")
        return np.zeros_like(image) # Mock implementation
        
    def emboss(self, image: np.ndarray, angle: float = 45.0) -> np.ndarray:
        """Apply emboss effect to create 3D appearance.
        
        Args:
            image (np.ndarray): Input image array
            angle (float): Emboss angle in degrees
            
        Returns:
            np.ndarray: Embossed image array
        """
        print(f"Mock applying emboss effect with angle={angle}")
        return image.copy() # Mock implementation
        
    def bilateral_filter(self, image: np.ndarray, d: int = 9, 
                        sigma_color: float = 75.0, 
                        sigma_space: float = 75.0) -> np.ndarray:
        """Apply bilateral filter for noise reduction while preserving edges.
        
        Args:
            image (np.ndarray): Input image array
            d (int): Diameter of each pixel neighborhood
            sigma_color (float): Filter sigma in the color space
            sigma_space (float): Filter sigma in the coordinate space
            
        Returns:
            np.ndarray: Filtered image array
        """
        print(f"Mock applying bilateral filter with d={d}, sigma_color={sigma_color}, sigma_space={sigma_space}")
        return image.copy() # Mock implementation
        
    def apply_custom_kernel(self, image: np.ndarray, 
                          kernel: np.ndarray) -> np.ndarray:
        """Apply a custom convolution kernel to the image.
        
        Args:
            image (np.ndarray): Input image array
            kernel (np.ndarray): Custom kernel matrix
            
        Returns:
            np.ndarray: Filtered image array
            
        Raises:
            ValueError: If kernel dimensions are invalid
        """
        if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise ValueError("Kernel dimensions must be odd")
        print("Mock applying custom kernel")
        return image.copy() # Mock implementation
        
    def create_gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Create a Gaussian kernel for convolution operations.
        
        Args:
            size (int): Kernel size (must be odd)
            sigma (float): Standard deviation
            
        Returns:
            np.ndarray: Gaussian kernel matrix
        """
        if size % 2 == 0:
            raise ValueError("Kernel size must be odd")
        if sigma < 0:
            raise ValueError("Sigma cannot be negative")
        print(f"Mock creating Gaussian kernel with size={size}, sigma={sigma}")
        return np.ones((size, size)) # Mock implementation
        
    def normalize_image(self, image: np.ndarray, 
                       target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        """Normalize image values to specified range.
        
        Args:
            image (np.ndarray): Input image array
            target_range (Tuple[float, float]): Target value range (min, max)
            
        Returns:
            np.ndarray: Normalized image array
        """
        print(f"Mock normalizing image to range {target_range}")
        return image.astype(np.float32) / 255.0 # Simple mock normalization
        
    def get_available_filters(self) -> list:
        """Get list of available filter methods.
        
        Returns:
            list: List of available filter method names
        """
        return [
            'gaussian_blur',
            'box_blur', 
            'median_filter',
            'sharpen',
            'sobel_edge_detection',
            'laplacian_edge_detection',
            'emboss',
            'bilateral_filter',
            'apply_custom_kernel'
        ]
        
    def clear_cache(self) -> None:
        """Clear the kernel cache to free memory."""
        print("Mock clearing kernel cache")
        self._kernel_cache.clear()

