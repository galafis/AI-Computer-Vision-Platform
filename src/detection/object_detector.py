"""
Object Detection Module - AI Computer Vision Platform

This module implements various object detection algorithms including YOLO, R-CNN, SSD
for real-time and high-accuracy object detection tasks.

Supported models:
- YOLOv5/YOLOv8: Real-time object detection
- Faster R-CNN: High accuracy object detection  
- SSD MobileNet: Lightweight detection for mobile/edge devices

Author: Gabriel Demetrios Lafis
License: MIT
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Data class for storing object detection results."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    class_id: int


class ObjectDetector:
    """Object detection class supporting multiple detection models.
    
    This class provides a unified interface for various object detection
    algorithms with support for different model architectures.
    
    Attributes:
        model_type (str): Type of detection model ('yolov5', 'yolov8', 'rcnn', 'ssd')
        confidence_threshold (float): Minimum confidence score for detections
        nms_threshold (float): Non-maximum suppression threshold
        input_size (Tuple[int, int]): Input image size for the model
    """
    
    def __init__(
        self, 
        model: str = 'yolov5',
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        device: str = 'cpu'
    ):
        """Initialize the ObjectDetector.
        
        Args:
            model: Model type to use for detection
            confidence_threshold: Minimum confidence score
            nms_threshold: NMS threshold for filtering overlapping boxes
            input_size: Target input size for the model
            device: Device to run inference on ('cpu', 'cuda')
        """
        self.model_type = model
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.device = device
        
        self.model = None
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the specified detection model.
        
        Raises:
            ValueError: If unsupported model type is specified
        """
        if self.model_type.lower() in ['yolov5', 'yolov8']:
            self._load_yolo_model()
        elif self.model_type.lower() == 'rcnn':
            self._load_rcnn_model()
        elif self.model_type.lower() == 'ssd':
            self._load_ssd_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _load_yolo_model(self) -> None:
        """Load YOLO model implementation (mock).
        In a real scenario, this would load a YOLO model (e.g., from Ultralytics).
        """
        print(f"Mock loading {self.model_type} model...")
        self.model = "MockYOLOModel"
    
    def _load_rcnn_model(self) -> None:
        """Load R-CNN model implementation (mock).
        In a real scenario, this would load an R-CNN model.
        """
        print("Mock loading R-CNN model...")
        self.model = "MockRCNNModel"
    
    def _load_ssd_model(self) -> None:
        """Load SSD model implementation (mock).
        In a real scenario, this would load an SSD model.
        """
        print("Mock loading SSD model...")
        self.model = "MockSSDModel"
    
    def detect(
        self, 
        image_path: Optional[str] = None, 
        image: Optional[np.ndarray] = None
    ) -> List[DetectionResult]:
        """Perform object detection on an image.
        
        Args:
            image_path: Path to the image file
            image: Image array (BGR format)
            
        Returns:
            List of DetectionResult objects containing detected objects
            
        Raises:
            ValueError: If neither image_path nor image is provided
        """
        if image_path is None and image is None:
            raise ValueError("Either image_path or image must be provided")
        
        if image is None:
            image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Run inference
        detections = self._run_inference(processed_image)
        
        # Post-process results
        results = self._postprocess_detections(detections, image.shape)
        
        return results
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model inference.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image array
        """
        # Resize image to model input size
        resized = cv2.resize(image, self.input_size)
        
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def _run_inference(self, image: np.ndarray) -> np.ndarray:
        """Run model inference on preprocessed image (mock).
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Raw model predictions (mock data)
        """
        print(f"Mock running inference with {self.model_type}...")
        # Mock detection data: [x_center, y_center, width, height, confidence, class_id]
        # For simplicity, returning a fixed mock detection for testing purposes.
        # In a real scenario, this would be the actual model output.
        mock_output = np.array([
            [0.5, 0.5, 0.2, 0.2, 0.9, 0],  # Mock detection: person
            [0.1, 0.1, 0.1, 0.1, 0.8, 2]   # Mock detection: car
        ])
        return mock_output
    
    def _postprocess_detections(
        self, 
        detections: np.ndarray, 
        original_shape: Tuple[int, int, int]
    ) -> List[DetectionResult]:
        """Post-process model predictions to extract detection results (mock).
        
        Args:
            detections: Raw model predictions (mock data)
            original_shape: Original image shape (height, width, channels)
            
        Returns:
            List of processed DetectionResult objects
        """
        results = []
        img_height, img_width, _ = original_shape
        
        for det in detections:
            x_center, y_center, w, h, confidence, class_id = det
            
            if confidence >= self.confidence_threshold:
                # Convert normalized center-width-height to pixel x, y, w, h
                x = int((x_center - w / 2) * img_width)
                y = int((y_center - h / 2) * img_height)
                w = int(w * img_width)
                h = int(h * img_height)
                
                class_name = self.class_names[int(class_id)] if int(class_id) < len(self.class_names) else "unknown"
                
                results.append(DetectionResult(
                    class_name=class_name,
                    confidence=float(confidence),
                    bbox=(x, y, w, h),
                    class_id=int(class_id)
                ))
        
        # Apply Non-Maximum Suppression (NMS) - mock for now
        # In a real scenario, NMS would be applied here to filter overlapping boxes.
        
        return results
    
    def detect_video(
        self, 
        video_path: str, 
        output_path: Optional[str] = None
    ) -> None:
        """Perform object detection on a video file.
        
        Args:
            video_path: Path to the input video
            output_path: Path for the output video with detections
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Process only a few frames for mock testing to avoid long execution
            if frame_count % 30 == 0: # Process every 30th frame
                # Detect objects in frame
                detections = self.detect(image=frame)
                
                # Draw detections on frame
                annotated_frame = self._draw_detections(frame, detections)
            else:
                annotated_frame = frame # Use original frame if not processing

            if output_path:
                out.write(annotated_frame)
        
        cap.release()
        if output_path:
            out.release()
    
    def _draw_detections(
        self, 
        image: np.ndarray, 
        detections: List[DetectionResult]
    ) -> np.ndarray:
        """Draw detection results on image.
        
        Args:
            image: Input image
            detections: List of detection results
            
        Returns:
            Image with drawn detections
        """
        annotated_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(
                annotated_image, 
                label, 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
        
        return annotated_image
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': self.model_type,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'input_size': self.input_size,
            'device': self.device,
            'num_classes': len(self.class_names)
        }

