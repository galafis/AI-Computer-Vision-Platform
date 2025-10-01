import pytest
import numpy as np
from src.recognition.face_recognition import FaceRecognizer, FaceRecognition

@pytest.fixture
def mock_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)

def test_face_recognizer_initialization():
    recognizer = FaceRecognizer()
    assert isinstance(recognizer, FaceRecognition)
    assert recognizer.model_name == "dlib"
    assert recognizer.confidence_threshold == 0.5
    assert recognizer.encoding_model == "facenet"
    assert recognizer.known_faces == {}

def test_face_recognizer_detect_faces(mock_image):
    recognizer = FaceRecognizer()
    faces = recognizer.detect_faces(mock_image)
    assert isinstance(faces, list)
    assert len(faces) == 1
    assert faces[0] == (10, 10, 50, 50)

def test_face_recognizer_encode_faces(mock_image):
    recognizer = FaceRecognizer()
    face_locations = [(10, 10, 50, 50)]
    encodings = recognizer.encode_faces(mock_image, face_locations)
    assert isinstance(encodings, list)
    assert len(encodings) == 1
    assert isinstance(encodings[0], np.ndarray)
    assert encodings[0].shape == (128,)

def test_face_recognizer_recognize_faces():
    recognizer = FaceRecognizer()
    mock_encodings = [np.random.rand(128)]
    recognized_names = recognizer.recognize_faces(mock_encodings)
    assert isinstance(recognized_names, list)
    assert len(recognized_names) == 1
    assert recognized_names[0] == "MockPerson"

def test_face_recognizer_load_known_faces():
    recognizer = FaceRecognizer()
    known_faces_dict = {"John Doe": np.random.rand(128)}
    recognizer.load_known_faces(known_faces_dict)
    assert "John Doe" in recognizer.known_faces
    assert isinstance(recognizer.known_faces["John Doe"], np.ndarray)

def test_face_recognizer_add_known_face():
    recognizer = FaceRecognizer()
    new_encoding = np.random.rand(128)
    recognizer.add_known_face("Jane Doe", new_encoding)
    assert "Jane Doe" in recognizer.known_faces
    assert np.array_equal(recognizer.known_faces["Jane Doe"], new_encoding)

