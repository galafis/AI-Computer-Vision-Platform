"""Text Recognition (OCR) module for AI Computer Vision Platform.

This module provides text recognition capabilities including text detection,
text extraction, and document analysis using state-of-the-art OCR technologies.

Author: Gabriel Demetrios Lafis
Email: gabrieldemetrios@gmail.com
Date: September 2025
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from abc import ABC, abstractmethod


class TextRecognition(ABC):
    """Base class for text recognition and OCR operations.
    
    This class provides a standard interface for text recognition tasks including
    text detection, text extraction, and document analysis.
    
    Attributes:
        ocr_engine (str): OCR engine being used (tesseract, easyocr, paddle)
        language (str): Language code for text recognition
        confidence_threshold (float): Minimum confidence score for text detection
        
    Methods:
        detect_text: Detect text regions in an image
        extract_text: Extract text content from detected regions
        recognize_text: Complete text recognition pipeline
        preprocess_image: Prepare image for optimal OCR results
        get_text_confidence: Get confidence scores for recognized text
    """
    
    def __init__(self, 
                 ocr_engine: str = "tesseract",
                 language: str = "eng",
                 confidence_threshold: float = 0.5):
        """Initialize the TextRecognition system.
        
        Args:
            ocr_engine (str): OCR engine to use ('tesseract', 'easyocr', 'paddle')
            language (str): Language code for text recognition
            confidence_threshold (float): Minimum confidence for text detection
        """
        self.ocr_engine = ocr_engine
        self.language = language
        self.confidence_threshold = confidence_threshold
        self.text_detector = None
        self.text_recognizer = None
        
        self._initialize_ocr_engine()
    
    def _initialize_ocr_engine(self) -> None:
        """Initialize the selected OCR engine."""
        # Implementation will be added based on chosen engine
        pass
    
    @abstractmethod
    def detect_text(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect text regions in an image.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            List[Tuple[int, int, int, int]]: List of text bounding boxes (x, y, w, h)
        """
        pass
    
    @abstractmethod
    def extract_text(self, image: np.ndarray, 
                    text_regions: Optional[List[Tuple[int, int, int, int]]] = None) -> List[str]:
        """Extract text content from image or specific regions.
        
        Args:
            image (np.ndarray): Input image in BGR format
            text_regions (Optional[List[Tuple]]): Specific regions to extract text from
            
        Returns:
            List[str]: List of extracted text strings
        """
        pass
    
    @abstractmethod
    def get_text_confidence(self, image: np.ndarray) -> List[float]:
        """Get confidence scores for recognized text.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            List[float]: Confidence scores for each detected text element
        """
        pass
    
    def preprocess_image(self, image: np.ndarray, 
                        enhancement_type: str = "default") -> np.ndarray:
        """Preprocess image for optimal OCR results.
        
        Args:
            image (np.ndarray): Input image in BGR format
            enhancement_type (str): Type of enhancement to apply
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if enhancement_type == "denoise":
            # Apply denoising
            gray = cv2.fastNlMeansDenoising(gray)
        elif enhancement_type == "sharpen":
            # Apply sharpening kernel
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            gray = cv2.filter2D(gray, -1, kernel)
        elif enhancement_type == "threshold":
            # Apply adaptive thresholding
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        return gray
    
    def recognize_text(self, image_path: str, 
                      preprocess: bool = True) -> Dict[str, Any]:
        """Complete text recognition pipeline.
        
        Args:
            image_path (str): Path to the input image
            preprocess (bool): Whether to preprocess the image
            
        Returns:
            Dict[str, Any]: Results containing detected text and metadata
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Preprocess if requested
        if preprocess:
            processed_image = self.preprocess_image(image)
        else:
            processed_image = image
        
        # Detect text regions
        text_regions = self.detect_text(processed_image)
        
        # Extract text
        extracted_text = self.extract_text(processed_image, text_regions)
        
        # Get confidence scores
        confidence_scores = self.get_text_confidence(processed_image)
        
        return {
            'text_regions': text_regions,
            'extracted_text': extracted_text,
            'confidence_scores': confidence_scores,
            'full_text': ' '.join(extracted_text),
            'num_text_regions': len(text_regions)
        }
    
    def extract_structured_data(self, image: np.ndarray, 
                               data_type: str = "table") -> Dict[str, Any]:
        """Extract structured data from image.
        
        Args:
            image (np.ndarray): Input image
            data_type (str): Type of structured data ('table', 'form', 'receipt')
            
        Returns:
            Dict[str, Any]: Structured data extraction results
        """
        # Base implementation - to be extended in subclasses
        text_result = self.recognize_text(image)
        
        return {
            'data_type': data_type,
            'raw_text': text_result['full_text'],
            'structured_data': {},  # To be implemented based on data_type
            'confidence': np.mean(text_result['confidence_scores']) if text_result['confidence_scores'] else 0.0
        }
    
    def get_engine_info(self) -> Dict[str, str]:
        """Get information about the current OCR engine.
        
        Returns:
            Dict[str, str]: OCR engine information
        """
        return {
            'ocr_engine': self.ocr_engine,
            'language': self.language,
            'confidence_threshold': str(self.confidence_threshold)
        }
