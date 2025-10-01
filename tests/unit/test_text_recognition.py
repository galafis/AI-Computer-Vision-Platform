import pytest
import numpy as np
import cv2
from src.recognition.text_recognition import TextRecognizer, TextRecognition

# Mock image for testing
@pytest.fixture
def mock_image():
    return np.zeros((100, 200, 3), dtype=np.uint8)

def test_text_recognizer_initialization():
    recognizer = TextRecognizer()
    assert isinstance(recognizer, TextRecognition)
    assert recognizer.ocr_engine == "tesseract"
    assert recognizer.language == "eng"
    assert recognizer.confidence_threshold == 0.5

def test_text_recognizer_detect_text(mock_image):
    recognizer = TextRecognizer()
    text_regions = recognizer.detect_text(mock_image)
    assert isinstance(text_regions, list)
    assert len(text_regions) == 1
    assert text_regions[0] == (0, 0, 100, 20)

def test_text_recognizer_extract_text(mock_image):
    recognizer = TextRecognizer()
    extracted_text = recognizer.extract_text(mock_image)
    assert isinstance(extracted_text, list)
    assert len(extracted_text) == 1
    assert extracted_text[0] == "MockText"

def test_text_recognizer_get_text_confidence(mock_image):
    recognizer = TextRecognizer()
    confidence_scores = recognizer.get_text_confidence(mock_image)
    assert isinstance(confidence_scores, list)
    assert len(confidence_scores) == 1
    assert confidence_scores[0] == 0.9

def test_text_recognizer_recognize_text(tmp_path, mock_image):
    # Create a dummy image file
    img_path = tmp_path / "test_text_image.jpg"
    cv2.imwrite(str(img_path), mock_image)

    recognizer = TextRecognizer()
    results = recognizer.recognize_text(image_path=str(img_path))
    assert isinstance(results, dict)
    assert results["full_text"] == "MockText"
    assert results["num_text_regions"] == 1
    assert results["extracted_text"] == ["MockText"]
    assert results["confidence_scores"] == [0.9]

