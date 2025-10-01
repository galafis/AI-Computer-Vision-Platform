
import pytest
import numpy as np
import cv2
from src.analysis.emotion_analyzer import EmotionAnalyzer, EmotionType
from src.analysis.emotion_analyzer_impl import EmotionAnalyzerImpl

@pytest.fixture
def mock_image():
    return np.zeros((480, 640, 3), dtype=np.uint8)

class TestEmotionAnalyzerImpl:

    def test_initialization(self):
        analyzer = EmotionAnalyzerImpl()
        assert isinstance(analyzer, EmotionAnalyzer)
        assert analyzer.model_name == "MockEmotionAnalyzer"
        assert analyzer.confidence_threshold == 0.6
        assert analyzer.is_initialized == True
        assert analyzer.face_detection_enabled == True
        assert analyzer.landmark_detection_enabled == True

    def test_initialize_method(self, capsys):
        analyzer = EmotionAnalyzerImpl()
        assert analyzer.initialize() == True
        captured = capsys.readouterr()
        assert "Mock EmotionAnalyzer MockEmotionAnalyzer initialized." in captured.out

    def test_analyze_emotion_returns_expected_format(self, mock_image):
        analyzer = EmotionAnalyzerImpl()
        results = analyzer.analyze_emotion(mock_image)
        assert isinstance(results, dict)
        assert "emotions" in results
        assert "primary_emotion" in results
        assert "confidence" in results
        assert results["primary_emotion"] == EmotionType.HAPPINESS.value
        assert results["confidence"] == 0.95
        assert results["face_detected"] == True

    def test_analyze_batch(self, mock_image):
        analyzer = EmotionAnalyzerImpl()
        images = [mock_image, mock_image]
        results_batch = analyzer.analyze_batch(images)
        assert isinstance(results_batch, list)
        assert len(results_batch) == 2
        assert all(isinstance(res, dict) for res in results_batch)
        assert results_batch[0]["primary_emotion"] == EmotionType.HAPPINESS.value

    def test_detect_faces(self, mock_image):
        analyzer = EmotionAnalyzerImpl()
        faces = analyzer.detect_faces(mock_image)
        assert isinstance(faces, list)
        assert len(faces) == 1
        assert faces[0] == (10, 10, 100, 100)

    def test_extract_facial_landmarks(self, mock_image):
        analyzer = EmotionAnalyzerImpl()
        face_region = (10, 10, 100, 100)
        landmarks = analyzer.extract_facial_landmarks(mock_image, face_region)
        assert isinstance(landmarks, np.ndarray)
        assert landmarks.size == 0

    def test_validate_image(self, mock_image):
        analyzer = EmotionAnalyzerImpl()
        assert analyzer.validate_image(mock_image) == True
        assert analyzer.validate_image(None) == False
        assert analyzer.validate_image(np.array([])) == False

    def test_preprocess_face_region(self, mock_image):
        analyzer = EmotionAnalyzerImpl()
        face_region = (10, 10, 100, 100)
        processed_face = analyzer.preprocess_face_region(mock_image, face_region, target_size=(64, 64))
        assert processed_face.shape == (64, 64, 3)

    def test_calculate_emotion_intensity(self):
        analyzer = EmotionAnalyzerImpl()
        emotion_scores = {EmotionType.HAPPINESS.value: 0.8, EmotionType.SADNESS.value: 0.1, EmotionType.NEUTRAL.value: 0.1}
        intensity = analyzer.calculate_emotion_intensity(emotion_scores)
        assert "valence" in intensity
        assert "arousal" in intensity
        assert "intensity" in intensity
        assert intensity["intensity"] == 0.8

    def test_get_dominant_emotion(self):
        analyzer = EmotionAnalyzerImpl()
        emotion_scores = {EmotionType.HAPPINESS.value: 0.9, EmotionType.ANGER.value: 0.1}
        dominant_emotion, confidence = analyzer.get_dominant_emotion(emotion_scores)
        assert dominant_emotion == EmotionType.HAPPINESS.value
        assert confidence == 0.9

    def test_filter_low_confidence_emotions(self):
        analyzer = EmotionAnalyzerImpl(confidence_threshold=0.7)
        emotion_scores = {EmotionType.HAPPINESS.value: 0.8, EmotionType.SADNESS.value: 0.6, EmotionType.NEUTRAL.value: 0.9}
        filtered_emotions = analyzer.filter_low_confidence_emotions(emotion_scores)
        assert EmotionType.HAPPINESS.value in filtered_emotions
        assert EmotionType.SADNESS.value not in filtered_emotions
        assert EmotionType.NEUTRAL.value in filtered_emotions

    def test_set_confidence_threshold(self):
        analyzer = EmotionAnalyzerImpl()
        analyzer.set_confidence_threshold(0.75)
        assert analyzer.confidence_threshold == 0.75
        with pytest.raises(ValueError):
            analyzer.set_confidence_threshold(1.1)

    def test_get_model_info(self):
        analyzer = EmotionAnalyzerImpl(model_name="TestEmotionModel", confidence_threshold=0.8)
        info = analyzer.get_model_info()
        assert info["model_name"] == "TestEmotionModel"
        assert info["is_initialized"] == True
        assert info["confidence_threshold"] == 0.8


