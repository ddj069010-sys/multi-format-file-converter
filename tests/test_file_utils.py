"""
Tests for file_utils module
"""
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from file_utils import (
    get_file_type,
    safe_filename,
    format_file_size,
    get_compression_ratio
)


def test_get_file_type():
    """Test file type detection"""
    assert get_file_type('.jpg') == 'image'
    assert get_file_type('.png') == 'image'
    assert get_file_type('.pdf') == 'document'
    assert get_file_type('.docx') == 'document'
    assert get_file_type('.mp3') == 'media'
    assert get_file_type('.mp4') == 'media'
    assert get_file_type('.zip') == 'archive'
    assert get_file_type('.unknown') == 'unknown'


def test_safe_filename():
    """Test filename sanitization"""
    assert safe_filename('test.txt') == 'test.txt'
    assert safe_filename('../../../etc/passwd') == 'etc_passwd'
    assert safe_filename('file<>:"|?*.txt') == 'file_______.txt'
    assert safe_filename('normal_file-name.txt') == 'normal_file-name.txt'
    # Test long filenames
    long_name = 'a' * 300 + '.txt'
    result = safe_filename(long_name)
    assert len(result) <= 200
    assert result.endswith('.txt')


def test_format_file_size():
    """Test file size formatting"""
    assert format_file_size(0) == '0.00 B'
    assert format_file_size(1024) == '1.00 KB'
    assert format_file_size(1024 * 1024) == '1.00 MB'
    assert format_file_size(1024 * 1024 * 1024) == '1.00 GB'
    assert format_file_size(500) == '500.00 B'


def test_get_compression_ratio():
    """Test compression ratio calculation"""
    assert get_compression_ratio(1000, 500) == 50.0
    assert get_compression_ratio(1000, 1000) == 0.0
    assert get_compression_ratio(1000, 0) == 100.0
    assert get_compression_ratio(0, 0) == 0
    assert get_compression_ratio(100, 75) == 25.0

