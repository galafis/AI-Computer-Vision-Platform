"""Unit tests for the GestureRecognizerImpl class."""

import pytest
import numpy as np
from src.recognition.gesture_recognition import GestureRecognition
from src.recognition.gesture_recognizer_impl import GestureRecognizer

@pytest.fixture
def mock_frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)

class TestGestureRecognizer:

    def test_initialization(self):
        recognizer = GestureRecognizer()
        assert isinstance(recognizer, GestureRecognition)
        assert recognizer.model_name == "MockGestureRecognizer"
        assert recognizer.confidence_threshold == 0.5
        assert recognizer.is_initialized == True

    def test_initialize_method(self, capsys):
        recognizer = GestureRecognizer()
        assert recognizer.initialize() == True
        captured = capsys.readouterr()
        assert "Mock GestureRecognizer MockGestureRecognizer initialized." in captured.out

    def test_recognize_gesture_returns_expected_format(self, mock_frame):
        recognizer = GestureRecognizer()
        results = recognizer.recognize_gesture(mock_frame)
        assert isinstance(results, dict)
        assert "gestures" in results
        assert "confidence_scores" in results
        assert "bounding_boxes" in results
        assert "landmarks" in results
        assert results["gestures"] == ["mock_gesture"]
        assert results["confidence_scores"] == [0.95]

    def test_preprocess_frame(self, mock_frame, capsys):
        recognizer = GestureRecognizer()
        processed_frame = recognizer.preprocess_frame(mock_frame)
        assert np.array_equal(processed_frame, mock_frame) # Mock does not change frame
        captured = capsys.readouterr()
        assert "Mock GestureRecognizer MockGestureRecognizer preprocessing frame." in captured.out

    def test_validate_frame(self, mock_frame):
        recognizer = GestureRecognizer()
        assert recognizer.validate_frame(mock_frame) == True
        assert recognizer.validate_frame(None) == False
        assert recognizer.validate_frame(np.array([])) == False

    def test_set_confidence_threshold(self):
        recognizer = GestureRecognizer()
        recognizer.set_confidence_threshold(0.75)
        assert recognizer.confidence_threshold == 0.75
        with pytest.raises(ValueError):
            recognizer.set_confidence_threshold(1.1)

    def test_get_model_info(self):
        recognizer = GestureRecognizer(model_name="TestGestureModel", confidence_threshold=0.8)
        info = recognizer.get_model_info()
        assert info["model_name"] == "TestGestureModel"
        assert info["is_initialized"] == True
        assert info["confidence_threshold"] == 0.8

