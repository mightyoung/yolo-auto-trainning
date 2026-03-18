# Training API Unit Tests

import sys
from pathlib import Path

# Add project root to sys.path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from unittest.mock import Mock, patch


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_returns_200(self):
        """Health check returns 200."""
        from fastapi.testclient import TestClient
        from src.api.gateway import app

        with patch('src.api.gateway.get_redis_client'):
            client = TestClient(app)
            response = client.get("/health")

            assert response.status_code == 200

    def test_health_check_returns_gpu_status(self):
        """Health check returns GPU status."""
        from fastapi.testclient import TestClient
        from src.api.gateway import app

        with patch('src.api.gateway.get_redis_client'):
            client = TestClient(app)
            response = client.get("/health")

            data = response.json()
            assert "gpu" in data
            assert data["status"] == "healthy"


class TestTrainingEndpoints:
    """Test Training API endpoints."""

    def test_start_training(self):
        """Training start endpoint works."""
        from fastapi.testclient import TestClient
        from src.api.gateway import app

        client = TestClient(app)

        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "test_123",
                "model": "yolo11n",
                "data_yaml": "/data/test.yaml",
                "epochs": 10
            },
            headers={"X-API-Key": "default-internal-key"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test_123"

    def test_get_training_status(self):
        """Training status endpoint works."""
        from fastapi.testclient import TestClient
        from src.api.gateway import app

        client = TestClient(app)

        # First create a task
        client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "test_456",
                "model": "yolo11n",
                "data_yaml": "/data/test.yaml",
                "epochs": 10
            },
            headers={"X-API-Key": "default-internal-key"}
        )

        # Then get status
        response = client.get("/api/v1/internal/train/status/test_456")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test_456"

    def test_start_training_without_api_key(self):
        """Training start without API key fails."""
        from fastapi.testclient import TestClient
        from src.api.gateway import app

        client = TestClient(app)

        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "test_789",
                "model": "yolo11n",
                "data_yaml": "/data/test.yaml",
                "epochs": 10
            }
        )

        assert response.status_code == 401


class TestCallbackClient:
    """Test Callback Client."""

    def test_notify_task_complete(self):
        """Test sending task completion notification."""
        from src.api.callback_client import CallbackClient
        from unittest.mock import patch, Mock

        with patch('httpx.Client') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {"received": True}
            mock_response.raise_for_status = Mock()

            mock_client.return_value.__enter__.return_value.post.return_value = mock_response

            client = CallbackClient(
                business_api_url="http://localhost:8000",
                api_key="test-key"
            )

            result = client.notify_task_complete(
                task_id="test_123",
                status="completed",
                metrics={"mAP50": 0.75}
            )

            assert result.get("received") is True
