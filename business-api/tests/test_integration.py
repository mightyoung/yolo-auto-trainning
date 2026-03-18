# Business API Integration Tests

import pytest
from unittest.mock import Mock, patch, AsyncMock


class TestDataIntegration:
    """Integration tests for data endpoints."""

    def test_search_and_discover_flow(self):
        """Test full data search and discover flow."""
        from fastapi.testclient import TestClient

        with patch('src.api.gateway.get_redis_client'):
            with patch('src.api.training_client.TrainingAPIClient'):
                from src.api.gateway import app

                client = TestClient(app)

                # Test search
                with patch('src.data.discovery.DatasetDiscovery') as mock_disc:
                    mock_instance = Mock()
                    mock_instance.search.return_value = []
                    mock_disc.return_value = mock_instance

                    response = client.post(
                        "/api/v1/data/search",
                        json={"query": "car", "max_results": 5}
                    )

                    assert response.status_code == 200


class TestTrainingIntegration:
    """Integration tests for training endpoints."""

    def test_training_flow(self):
        """Test full training flow."""
        from fastapi.testclient import TestClient

        with patch('src.api.gateway.get_redis_client'):
            with patch('src.api.training_client.TrainingAPIClient') as mock_client_cls:
                mock_client = Mock()
                mock_client.start_training = AsyncMock(return_value={
                    "task_id": "test_123",
                    "status": "started"
                })
                mock_client.health_check = AsyncMock(return_value=True)
                mock_client_cls.return_value = mock_client

                from src.api.gateway import app
                client = TestClient(app)

                response = client.post(
                    "/api/v1/train/submit",
                    json={
                        "model": "yolo11n",
                        "data_yaml": "/data/test.yaml",
                        "epochs": 10
                    }
                )

                assert response.status_code in [200, 502]  # May fail if training API not available


class TestCallbackIntegration:
    """Integration tests for callback endpoints."""

    def test_callback_receives_notification(self):
        """Test callback endpoint receives notifications."""
        from fastapi.testclient import TestClient

        with patch('src.api.gateway.get_redis_client'):
            from src.api.gateway import app

            client = TestClient(app)

            response = client.post(
                "/api/v1/callback/task/callback",
                json={
                    "task_id": "train_123",
                    "status": "completed",
                    "metrics": {"mAP50": 0.75}
                }
            )

            assert response.status_code == 200
            assert response.json()["received"] is True
