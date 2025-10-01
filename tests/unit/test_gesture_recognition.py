import pytest
import numpy as np
from src.recognition.gesture_recognition import GestureRecognition
from src.recognition.gesture_recognizer_impl import GestureRecognizer

@pytest.fixture
def mock_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)

def test_gesture_recognizer_initialization():
    recognizer = GestureRecognizer()
    assert isinstance(recognizer, GestureRecognition)
    assert recognizer.model_name == "MockGestureRecognizer"
    assert recognizer.confidence_threshold == 0.5
    assert recognizer.is_initialized == True

def test_gesture_recognizer_initialize_method():
    recognizer = GestureRecognizer()
    assert recognizer.initialize() == True

def test_gesture_recognizer_recognize_gesture(mock_frame):
    recognizer = GestureRecognizer()
    results = recognizer.recognize_gesture(mock_frame)
    assert isinstance(results, dict)
    assert "gestures" in results
    assert results["gestures"] == ["mock_gesture"]
    assert "confidence_scores" in results
    assert results["confidence_scores"] == [0.95]
    assert "bounding_boxes" in results
    assert results["bounding_boxes"] == [(10, 10, 50, 50)]

def test_gesture_recognizer_preprocess_frame(mock_frame):
    recognizer = GestureRecognizer()
    processed_frame = recognizer.preprocess_frame(mock_frame)
    assert np.array_equal(processed_frame, mock_frame)

def test_gesture_recognizer_validate_frame(mock_frame):
    recognizer = GestureRecognizer()
    assert recognizer.validate_frame(mock_frame) == True
    assert recognizer.validate_frame(None) == False
    assert recognizer.validate_frame(np.array([])) == False

def test_gesture_recognizer_set_confidence_threshold():
    recognizer = GestureRecognizer()
    recognizer.set_confidence_threshold(0.7)
    assert recognizer.confidence_threshold == 0.7
    with pytest.raises(ValueError):
        recognizer.set_confidence_threshold(1.1)

def test_gesture_recognizer_get_model_info():
    recognizer = GestureRecognizer(model_name="TestModel", confidence_threshold=0.8)
    info = recognizer.get_model_info()
    assert info["model_name"] == "TestModel"
    assert info["is_initialized"] == True
    assert info["confidence_threshold"] == 0.8

