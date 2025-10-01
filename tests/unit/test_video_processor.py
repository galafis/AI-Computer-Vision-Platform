import pytest
import numpy as np
from pathlib import Path
from src.processing.video_processor import VideoProcessor

@pytest.fixture
def video_processor():
    return VideoProcessor()

@pytest.fixture
def mock_video_file(tmp_path):
    video_path = tmp_path / "mock_video.mp4"
    video_path.touch() # Create an empty file to simulate a video file
    return video_path

def test_video_processor_initialization(video_processor):
    assert video_processor.video_path is None
    assert video_processor.frame_count == 0
    assert video_processor.fps == 0.0
    assert video_processor.resolution == (0, 0)
    assert video_processor.current_frame == 0

def test_load_video(video_processor, mock_video_file):
    assert video_processor.load_video(mock_video_file) == True
    assert video_processor.video_path == mock_video_file
    assert video_processor.frame_count == 100
    assert video_processor.fps == 30.0
    assert video_processor.resolution == (640, 480)

def test_load_video_not_found(video_processor):
    with pytest.raises(FileNotFoundError):
        video_processor.load_video("non_existent_video.mp4")

def test_get_frame(video_processor, mock_video_file):
    video_processor.load_video(mock_video_file)
    frame = video_processor.get_frame(50)
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)
    assert video_processor.current_frame == 50
    assert video_processor.get_frame(101) is None # Invalid frame index

def test_get_frame_no_video_loaded(video_processor):
    with pytest.raises(ValueError, match="No video loaded."):
        video_processor.get_frame(0)

def test_get_frame_sequence(video_processor, mock_video_file):
    video_processor.load_video(mock_video_file)
    frames = video_processor.get_frame_sequence(10, 20)
    assert isinstance(frames, list)
    assert len(frames) == 10
    assert frames[0].shape == (480, 640, 3)

def test_get_frame_sequence_invalid_indices(video_processor, mock_video_file):
    video_processor.load_video(mock_video_file)
    with pytest.raises(ValueError, match="Invalid frame indices."):
        video_processor.get_frame_sequence(20, 10) # start > end
    with pytest.raises(ValueError, match="Invalid frame indices."):
        video_processor.get_frame_sequence(0, 101) # end out of bounds

def test_save_video_segment(video_processor, mock_video_file, tmp_path):
    video_processor.load_video(mock_video_file)
    output_path = tmp_path / "segment.mp4"
    assert video_processor.save_video_segment(output_path, 10, 20) == True

def test_resize_video(video_processor, mock_video_file, tmp_path):
    video_processor.load_video(mock_video_file)
    output_path = tmp_path / "resized.mp4"
    assert video_processor.resize_video((320, 240), output_path) == True
    assert video_processor.resolution == (320, 240)

def test_change_fps(video_processor, mock_video_file, tmp_path):
    video_processor.load_video(mock_video_file)
    output_path = tmp_path / "fps_changed.mp4"
    assert video_processor.change_fps(60.0, output_path) == True
    assert video_processor.fps == 60.0

def test_apply_frame_filter(video_processor, mock_video_file, tmp_path):
    video_processor.load_video(mock_video_file)
    output_path = tmp_path / "filtered.mp4"
    def mock_filter(frame): return frame # Dummy filter
    assert video_processor.apply_frame_filter(mock_filter, output_path) == True

def test_extract_audio(video_processor, mock_video_file, tmp_path):
    video_processor.load_video(mock_video_file)
    output_path = tmp_path / "audio.mp3"
    assert video_processor.extract_audio(output_path) == True

def test_get_video_info(video_processor, mock_video_file):
    assert video_processor.get_video_info() == {"status": "No video loaded"}
    video_processor.load_video(mock_video_file)
    info = video_processor.get_video_info()
    assert info["video_path"] == str(mock_video_file)
    assert info["frame_count"] == 100
    assert info["fps"] == 30.0
    assert info["resolution"] == (640, 480)

def test_release_video(video_processor, mock_video_file):
    video_processor.load_video(mock_video_file)
    video_processor.release_video()
    # No direct assertion, but ensures no errors and prints mock message

def test_reset(video_processor, mock_video_file):
    video_processor.load_video(mock_video_file)
    video_processor.reset()
    assert video_processor.video_path is None
    assert video_processor.frame_count == 0
    assert video_processor.fps == 0.0
    assert video_processor.resolution == (0, 0)
    assert video_processor.current_frame == 0

