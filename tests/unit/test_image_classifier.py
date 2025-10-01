import pytest
import numpy as np
import cv2
from src.analysis.image_classifier import ImageClassifier
from src.analysis.image_classifier_impl import ImageClassifierImpl

@pytest.fixture
def mock_image():
    return np.zeros((224, 224, 3), dtype=np.uint8)

def test_image_classifier_impl_initialization():
    classifier = ImageClassifierImpl()
    assert isinstance(classifier, ImageClassifier)
    assert classifier.model_name == "MockImageClassifier"
    assert classifier.num_classes == 1000
    assert classifier.confidence_threshold == 0.5
    assert classifier.is_initialized == True
    assert classifier.input_size == (224, 224)
    assert len(classifier.class_names) == 1000

def test_image_classifier_impl_initialize_method():
    classifier = ImageClassifierImpl()
    assert classifier.initialize() == True

def test_image_classifier_impl_classify_image(mock_image):
    classifier = ImageClassifierImpl()
    results = classifier.classify_image(mock_image)
    assert isinstance(results, dict)
    assert "predictions" in results
    assert results["predictions"] == ["class_0"]
    assert "probabilities" in results
    assert results["probabilities"] == [0.99]
    assert "top_class" in results
    assert results["top_class"] == "class_0"
    assert "confidence" in results
    assert results["confidence"] == 0.99

def test_image_classifier_impl_classify_batch(mock_image):
    classifier = ImageClassifierImpl()
    images = [mock_image, mock_image]
    results_batch = classifier.classify_batch(images)
    assert isinstance(results_batch, list)
    assert len(results_batch) == 2
    assert results_batch[0]["top_class"] == "class_0"

def test_image_classifier_impl_preprocess_image(mock_image):
    classifier = ImageClassifierImpl()
    processed_image = classifier.preprocess_image(mock_image)
    assert np.array_equal(processed_image, mock_image)

def test_image_classifier_impl_validate_image(mock_image):
    classifier = ImageClassifierImpl()
    assert classifier.validate_image(mock_image) == True
    assert classifier.validate_image(None) == False
    assert classifier.validate_image(np.array([])) == False

def test_image_classifier_impl_load_class_names(tmp_path):
    classifier = ImageClassifierImpl(num_classes=2)
    class_names_file = tmp_path / "classes.txt"
    class_names_file.write_text("cat\ndog")
    assert classifier.load_class_names(str(class_names_file)) == True
    assert classifier.class_names == ["cat", "dog"]

def test_image_classifier_impl_set_confidence_threshold():
    classifier = ImageClassifierImpl()
    classifier.set_confidence_threshold(0.75)
    assert classifier.confidence_threshold == 0.75
    with pytest.raises(ValueError):
        classifier.set_confidence_threshold(1.1)

def test_image_classifier_impl_get_top_k_predictions():
    classifier = ImageClassifierImpl(num_classes=3)
    classifier.class_names = ["class_A", "class_B", "class_C"]
    probabilities = np.array([0.1, 0.8, 0.5])
    top_predictions = classifier.get_top_k_predictions(probabilities, k=2)
    assert len(top_predictions) == 2
    assert top_predictions[0]["class_name"] == "class_B"
    assert top_predictions[1]["class_name"] == "class_C"

def test_image_classifier_impl_resize_image(mock_image):
    classifier = ImageClassifierImpl()
    resized_image = classifier.resize_image(mock_image, target_size=(100, 100))
    assert resized_image.shape == (100, 100, 3)

def test_image_classifier_impl_get_model_info():
    classifier = ImageClassifierImpl(model_name="TestClassifier", confidence_threshold=0.8)
    info = classifier.get_model_info()
    assert info["model_name"] == "TestClassifier"
    assert info["is_initialized"] == True
    assert info["confidence_threshold"] == 0.8

