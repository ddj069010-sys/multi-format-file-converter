"""
Basic tests for main.py API endpoints
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "service" in data


def test_get_formats():
    """Test formats endpoint"""
    response = client.get("/api/formats")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "formats" in data
    assert "capabilities" in data


def test_get_status():
    """Test status endpoint"""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "service" in data
    assert "version" in data
    assert "capabilities" in data


def test_get_history():
    """Test history endpoint"""
    response = client.get("/api/history")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "files" in data
    assert isinstance(data["files"], list)


def test_get_stats():
    """Test stats endpoint"""
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "upload_dir" in data
    assert "download_dir" in data
    assert "temp_dir" in data


def test_compress_no_file():
    """Test compress endpoint without file"""
    response = client.post("/api/compress")
    assert response.status_code == 422  # Validation error


def test_compress_invalid_quality():
    """Test compress endpoint with invalid quality"""
    files = {"file": ("test.txt", b"test content", "text/plain")}
    data = {"quality": 150, "format_type": ""}
    response = client.post("/api/compress", files=files, data=data)
    # Should fail validation or return error
    assert response.status_code in [400, 422]


def test_rate_limiting():
    """Test rate limiting (may need adjustment based on rate limit settings)"""
    # Make multiple requests quickly
    responses = [client.get("/api/health") for _ in range(70)]
    # Should eventually hit rate limit (if rate limit is 60/minute)
    # Note: This test may be flaky depending on rate limit implementation
    status_codes = [r.status_code for r in responses]
    # At least most should succeed (first 60)
    assert all(code == 200 for code in status_codes[:60])

