import pytest
import numpy as np
import cv2
from src.detection.object_detector import ObjectDetector, DetectionResult

# Mock image for testing
@pytest.fixture
def mock_image():
    return np.zeros((480, 640, 3), dtype=np.uint8)

def test_object_detector_initialization():
    detector = ObjectDetector(model="yolov5")
    assert detector.model_type == "yolov5"
    assert detector.confidence_threshold == 0.5
    assert detector.nms_threshold == 0.4
    assert detector.input_size == (640, 640)
    assert detector.model is not None

def test_object_detector_unsupported_model():
    with pytest.raises(ValueError, match="Unsupported model type: invalid_model"):
        ObjectDetector(model="invalid_model")

def test_object_detector_detect_image_path(tmp_path, mock_image):
    # Create a dummy image file
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), mock_image)

    detector = ObjectDetector(model="yolov5")
    results = detector.detect(image_path=str(img_path))
    assert isinstance(results, list)
    assert len(results) > 0  # Expecting mock detections
    assert all(isinstance(res, DetectionResult) for res in results)
    assert results[0].class_name == "person"
    assert results[0].confidence == 0.9

def test_object_detector_detect_image_array(mock_image):
    detector = ObjectDetector(model="yolov5")
    results = detector.detect(image=mock_image)
    assert isinstance(results, list)
    assert len(results) > 0  # Expecting mock detections
    assert all(isinstance(res, DetectionResult) for res in results)
    assert results[0].class_name == "person"
    assert results[0].confidence == 0.9

def test_object_detector_detect_no_input():
    detector = ObjectDetector(model="yolov5")
    with pytest.raises(ValueError, match="Either image_path or image must be provided"):
        detector.detect()

def test_object_detector_preprocess_image(mock_image):
    detector = ObjectDetector(model="yolov5")
    processed_image = detector._preprocess_image(mock_image)
    assert processed_image.shape == (1, 640, 640, 3) # Batch, H, W, C
    assert processed_image.dtype == np.float32
    assert np.all(processed_image >= 0) and np.all(processed_image <= 1)

def test_object_detector_postprocess_detections(mock_image):
    detector = ObjectDetector(model="yolov5")
    # Mock detections from _run_inference
    mock_raw_detections = np.array([
        [0.5, 0.5, 0.2, 0.2, 0.9, 0],  # Mock detection: person
        [0.1, 0.1, 0.1, 0.1, 0.8, 2]   # Mock detection: car
    ])
    original_shape = mock_image.shape
    results = detector._postprocess_detections(mock_raw_detections, original_shape)
    assert len(results) == 2
    assert results[0].class_name == "person"
    assert results[1].class_name == "car"
    assert results[0].confidence == 0.9
    assert results[1].confidence == 0.8

def test_object_detector_get_model_info():
    detector = ObjectDetector(model="ssd", confidence_threshold=0.7)
    info = detector.get_model_info()
    assert info["model_type"] == "ssd"
    assert info["confidence_threshold"] == 0.7
    assert "num_classes" in info

