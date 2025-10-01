import pytest
import numpy as np
from pathlib import Path
from src.processing.image_processor import ImageProcessor

@pytest.fixture
def image_processor():
    return ImageProcessor()

def test_image_processor_initialization(image_processor):
    assert image_processor.image_data is None
    assert image_processor.original_shape is None
    assert image_processor.color_mode == 'RGB'

def test_load_image(image_processor, tmp_path):
    image_path = tmp_path / "test_image.jpg"
    image_path.touch()
    assert image_processor.load_image(image_path) == True
    assert isinstance(image_processor.image_data, np.ndarray)
    assert image_processor.original_shape == (100, 100, 3)
    assert image_processor.color_mode == 'RGB'

def test_save_image(image_processor, tmp_path):
    image_processor.image_data = np.zeros((100, 100, 3), dtype=np.uint8)
    output_path = tmp_path / "output.jpg"
    assert image_processor.save_image(output_path) == True

def test_resize_image(image_processor):
    image_processor.image_data = np.zeros((100, 100, 3), dtype=np.uint8)
    resized_image = image_processor.resize_image((50, 50))
    assert resized_image.shape == (50, 50, 3)
    with pytest.raises(ValueError, match="No image loaded to resize."):
        ImageProcessor().resize_image((50, 50))

def test_rotate_image(image_processor):
    image_processor.image_data = np.zeros((100, 100, 3), dtype=np.uint8)
    rotated_image = image_processor.rotate_image(90)
    assert rotated_image.shape == (100, 100, 3)
    with pytest.raises(ValueError, match="No image loaded to rotate."):
        ImageProcessor().rotate_image(90)

def test_adjust_brightness(image_processor):
    image_processor.image_data = np.zeros((100, 100, 3), dtype=np.uint8)
    adjusted_image = image_processor.adjust_brightness(1.5)
    assert adjusted_image.shape == (100, 100, 3)
    with pytest.raises(ValueError, match="No image loaded to adjust brightness."):
        ImageProcessor().adjust_brightness(1.5)

def test_adjust_contrast(image_processor):
    image_processor.image_data = np.zeros((100, 100, 3), dtype=np.uint8)
    adjusted_image = image_processor.adjust_contrast(1.5)
    assert adjusted_image.shape == (100, 100, 3)
    with pytest.raises(ValueError, match="No image loaded to adjust contrast."):
        ImageProcessor().adjust_contrast(1.5)

def test_convert_color_space(image_processor):
    image_processor.image_data = np.zeros((100, 100, 3), dtype=np.uint8)
    gray_image = image_processor.convert_color_space('GRAY')
    assert gray_image.shape == (100, 100)
    with pytest.raises(ValueError, match="No image loaded to convert color space."):
        ImageProcessor().convert_color_space('GRAY')
    with pytest.raises(ValueError, match="Unsupported color mode: YUV"):
        image_processor.convert_color_space('YUV')

def test_get_image_info(image_processor):
    assert image_processor.get_image_info() == {"status": "No image loaded"}
    image_processor.image_data = np.zeros((100, 100, 3), dtype=np.uint8)
    image_processor.original_shape = (100, 100, 3)
    info = image_processor.get_image_info()
    assert info["shape"] == (100, 100, 3)
    assert info["original_shape"] == (100, 100, 3)
    assert info["color_mode"] == 'RGB'
    assert info["size_bytes"] > 0

def test_reset(image_processor):
    image_processor.image_data = np.zeros((100, 100, 3), dtype=np.uint8)
    image_processor.original_shape = (100, 100, 3)
    image_processor.color_mode = 'BGR'
    image_processor.reset()
    assert image_processor.image_data is None
    assert image_processor.original_shape is None
    assert image_processor.color_mode == 'RGB'

