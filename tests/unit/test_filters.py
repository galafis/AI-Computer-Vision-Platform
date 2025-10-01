import pytest
import numpy as np
from src.processing.filters import Filters

@pytest.fixture
def mock_image():
    return np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

def test_filters_initialization():
    filters = Filters()
    assert isinstance(filters._kernel_cache, dict)

def test_gaussian_blur(mock_image):
    filters = Filters()
    blurred_image = filters.gaussian_blur(mock_image, kernel_size=5, sigma=1.0)
    assert blurred_image.shape == mock_image.shape
    with pytest.raises(ValueError, match="kernel_size must be odd"):
        filters.gaussian_blur(mock_image, kernel_size=4)
    with pytest.raises(ValueError, match="sigma cannot be negative"):
        filters.gaussian_blur(mock_image, sigma=-1.0)

def test_box_blur(mock_image):
    filters = Filters()
    blurred_image = filters.box_blur(mock_image, kernel_size=3)
    assert blurred_image.shape == mock_image.shape
    with pytest.raises(ValueError, match="kernel_size must be odd"):
        filters.box_blur(mock_image, kernel_size=2)

def test_median_filter(mock_image):
    filters = Filters()
    filtered_image = filters.median_filter(mock_image, kernel_size=3)
    assert filtered_image.shape == mock_image.shape
    with pytest.raises(ValueError, match="kernel_size must be odd"):
        filters.median_filter(mock_image, kernel_size=2)

def test_sharpen(mock_image):
    filters = Filters()
    sharpened_image = filters.sharpen(mock_image, intensity=1.5)
    assert sharpened_image.shape == mock_image.shape

def test_sobel_edge_detection(mock_image):
    filters = Filters()
    edges_image = filters.sobel_edge_detection(mock_image, direction='x')
    assert edges_image.shape == mock_image.shape
    assert np.all(edges_image == 0) # Mock returns zeros_like
    with pytest.raises(ValueError, match="Direction must be 'x', 'y', or 'both'"):
        filters.sobel_edge_detection(mock_image, direction='invalid')

def test_laplacian_edge_detection(mock_image):
    filters = Filters()
    edges_image = filters.laplacian_edge_detection(mock_image)
    assert edges_image.shape == mock_image.shape
    assert np.all(edges_image == 0) # Mock returns zeros_like

def test_emboss(mock_image):
    filters = Filters()
    embossed_image = filters.emboss(mock_image, angle=90.0)
    assert embossed_image.shape == mock_image.shape

def test_bilateral_filter(mock_image):
    filters = Filters()
    filtered_image = filters.bilateral_filter(mock_image, d=5, sigma_color=50.0, sigma_space=50.0)
    assert filtered_image.shape == mock_image.shape

def test_apply_custom_kernel(mock_image):
    filters = Filters()
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filtered_image = filters.apply_custom_kernel(mock_image, kernel)
    assert filtered_image.shape == mock_image.shape
    with pytest.raises(ValueError, match="Kernel dimensions must be odd"):
        filters.apply_custom_kernel(mock_image, np.array([[1, 1], [1, 1]]))

def test_create_gaussian_kernel():
    filters = Filters()
    kernel = filters.create_gaussian_kernel(size=3, sigma=1.0)
    assert kernel.shape == (3, 3)
    assert np.all(kernel == 1) # Mock returns ones
    with pytest.raises(ValueError, match="Kernel size must be odd"):
        filters.create_gaussian_kernel(size=2, sigma=1.0)
    with pytest.raises(ValueError, match="Sigma cannot be negative"):
        filters.create_gaussian_kernel(size=3, sigma=-1.0)

def test_normalize_image(mock_image):
    filters = Filters()
    normalized_image = filters.normalize_image(mock_image, target_range=(0.0, 1.0))
    assert normalized_image.shape == mock_image.shape
    assert normalized_image.dtype == np.float32
    assert np.all(normalized_image >= 0.0) and np.all(normalized_image <= 1.0)

def test_get_available_filters():
    filters = Filters()
    available_filters = filters.get_available_filters()
    expected_filters = [
        'gaussian_blur',
        'box_blur', 
        'median_filter',
        'sharpen',
        'sobel_edge_detection',
        'laplacian_edge_detection',
        'emboss',
        'bilateral_filter',
        'apply_custom_kernel'
    ]
    assert sorted(available_filters) == sorted(expected_filters)

def test_clear_cache():
    filters = Filters()
    filters._kernel_cache["test"] = "value"
    filters.clear_cache()
    assert filters._kernel_cache == {}

