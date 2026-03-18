# Training API Integration Tests

import pytest
from unittest.mock import Mock, patch


class TestTrainingFlow:
    """Integration tests for training flow."""

    def test_training_lifecycle(self):
        """Test complete training lifecycle."""
        from fastapi.testclient import TestClient
        from src.api.gateway import app

        client = TestClient(app)

        # Start training
        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "test_flow_001",
                "model": "yolo11n",
                "data_yaml": "/data/test.yaml",
                "epochs": 10
            },
            headers={"X-API-Key": "default-internal-key"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test_flow_001"

        # Check status
        response = client.get("/api/v1/internal/train/status/test_flow_001")
        assert response.status_code == 200


class TestExportFlow:
    """Integration tests for export flow."""

    def test_export_lifecycle(self):
        """Test complete export lifecycle."""
        from fastapi.testclient import TestClient
        from src.api.gateway import app

        client = TestClient(app)

        # Start export
        response = client.post(
            "/api/v1/internal/export/start",
            json={
                "task_id": "export_001",
                "model_path": "/models/test.pt",
                "platform": "jetson_orin"
            },
            headers={"X-API-Key": "default-internal-key"}
        )

        assert response.status_code == 200

        # Check status
        response = client.get("/api/v1/internal/export/status/export_001")
        assert response.status_code == 200


class TestModelManagement:
    """Integration tests for model management."""

    def test_list_models(self):
        """Test listing models."""
        from fastapi.testclient import TestClient
        from src.api.gateway import app

        client = TestClient(app)

        response = client.get("/api/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total" in data


class TestAuthentication:
    """Integration tests for authentication."""

    def test_protected_endpoint_requires_api_key(self):
        """Test that protected endpoints require API key."""
        from fastapi.testclient import TestClient
        from src.api.gateway import app

        client = TestClient(app)

        # Try without API key
        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "test_auth",
                "model": "yolo11n",
                "data_yaml": "/data/test.yaml",
                "epochs": 10
            }
        )

        assert response.status_code == 401

    def test_protected_endpoint_accepts_valid_api_key(self):
        """Test that protected endpoints accept valid API key."""
        from fastapi.testclient import TestClient
        from src.api.gateway import app

        client = TestClient(app)

        # Try with API key
        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "test_auth_valid",
                "model": "yolo11n",
                "data_yaml": "/data/test.yaml",
                "epochs": 10
            },
            headers={"X-API-Key": "default-internal-key"}
        )

        assert response.status_code == 200
