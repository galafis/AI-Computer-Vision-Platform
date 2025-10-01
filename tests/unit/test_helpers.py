
import pytest
import numpy as np
import json
import hashlib
from unittest.mock import patch, mock_open
from pathlib import Path
from src.utils.helpers import Helpers

@pytest.fixture
def helpers_instance():
    return Helpers()

class TestHelpers:

    def test_validate_file_extension(self, helpers_instance):
        assert helpers_instance.validate_file_extension("image.jpg", [".jpg", ".png"]) == True
        assert helpers_instance.validate_file_extension("document.pdf", [".jpg", ".png"]) == False
        assert helpers_instance.validate_file_extension("archive.tar.gz", [".gz"]) == True
        assert helpers_instance.validate_file_extension("no_extension", [".txt"]) == False
        assert helpers_instance.validate_file_extension(None, [".txt"]) == False
        assert helpers_instance.validate_file_extension("file.TXT", [".txt"]) == True

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists", return_value=False)
    def test_ensure_directory_exists_creates_new(self, mock_exists, mock_mkdir, helpers_instance):
        path = Path("new_dir")
        assert helpers_instance.ensure_directory_exists(path) == True
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists", return_value=True)
    def test_ensure_directory_exists_already_exists(self, mock_exists, mock_mkdir, helpers_instance):
        path = Path("existing_dir")
        assert helpers_instance.ensure_directory_exists(path) == True
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("builtins.open", new_callable=mock_open, read_data=
'{"key": "value"}'
)
    @patch("pathlib.Path.exists", return_value=True)
    def test_read_json_file_success(self, mock_exists, mock_file, helpers_instance):
        data = helpers_instance.read_json_file("config.json")
        assert data == {"key": "value"}

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists", return_value=False)
    def test_read_json_file_not_found(self, mock_exists, mock_file, helpers_instance):
        data = helpers_instance.read_json_file("non_existent.json")
        assert data is None

    @patch("builtins.open", new_callable=mock_open, read_data=
'invalid json'
)
    @patch("pathlib.Path.exists", return_value=True)
    def test_read_json_file_invalid_json(self, mock_exists, mock_file, helpers_instance):
        data = helpers_instance.read_json_file("invalid.json")
        assert data is None

    @patch("src.utils.helpers.Helpers.ensure_directory_exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_write_json_file_success(self, mock_json_dump, mock_file, mock_ensure_dir, helpers_instance):
        data = {"test": "data"}
        assert helpers_instance.write_json_file(data, "output.json") == True
        mock_file.assert_called_once_with(Path("output.json"), 'w', encoding='utf-8')
        mock_json_dump.assert_called_once_with(data, mock_file(), indent=2, ensure_ascii=False)

    @patch("src.utils.helpers.Helpers.ensure_directory_exists", return_value=True)
    @patch("builtins.open", side_effect=OSError)
    def test_write_json_file_os_error(self, mock_open_error, mock_ensure_dir, helpers_instance):
        data = {"test": "data"}
        assert helpers_instance.write_json_file(data, "output.json") == False

    def test_generate_hash_sha256(self):
        data = "test string"
        expected_hash = hashlib.sha256(data.encode("utf-8")).hexdigest()
        assert Helpers.generate_hash(data) == expected_hash

    def test_generate_hash_md5(self):
        data = b"binary data"
        expected_hash = hashlib.md5(data).hexdigest()
        assert Helpers.generate_hash(data, algorithm="md5") == expected_hash

    def test_generate_hash_unsupported_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported hash algorithm: invalid_algo"):
            Helpers.generate_hash("data", algorithm="invalid_algo")

    @patch("pathlib.Path.stat")
    @patch("pathlib.Path.exists", return_value=True)
    def test_get_file_size_success(self, mock_exists, mock_stat):
        mock_stat.return_value.st_size = 1024
        assert Helpers.get_file_size("file.txt") == 1024

    @patch("pathlib.Path.exists", return_value=False)
    def test_get_file_size_not_found(self, mock_exists):
        assert Helpers.get_file_size("non_existent.txt") is None

    def test_format_bytes(self):
        assert Helpers.format_bytes(0) == "0 B"
        assert Helpers.format_bytes(500) == "500.00 B"
        assert Helpers.format_bytes(1024) == "1.00 KB"
        assert Helpers.format_bytes(1536) == "1.50 KB"
        assert Helpers.format_bytes(1048576) == "1.00 MB"
        assert Helpers.format_bytes(1073741824, decimal_places=1) == "1.0 GB"

    @patch("src.utils.helpers.datetime")
    def test_get_timestamp(self, mock_datetime):
        mock_datetime.now.return_value.strftime.return_value = "2025-10-01 10:30:00"
        assert Helpers.get_timestamp() == "2025-10-01 10:30:00"
        mock_datetime.now.return_value.strftime.assert_called_once_with("%Y-%m-%d %H:%M:%S")

    def test_get_relative_path_success(self):
        helpers = Helpers(base_path="/home/user")
        target_path = "/home/user/documents/file.txt"
        assert helpers.get_relative_path(target_path) == "documents/file.txt"

    def test_get_relative_path_outside_base(self):
        helpers = Helpers(base_path="/home/user")
        target_path = "/opt/another_file.txt"
        assert helpers.get_relative_path(target_path) == "/opt/another_file.txt"

    def test_get_relative_path_same_path(self):
        helpers = Helpers(base_path="/home/user")
        target_path = "/home/user"
        assert helpers.get_relative_path(target_path) == "."

