# Business API Unit Tests

import pytest
from unittest.mock import Mock, patch, AsyncMock


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

    def test_health_check_returns_healthy_status(self):
        """Health check returns healthy status."""
        from fastapi.testclient import TestClient
        from src.api.gateway import app

        with patch('src.api.gateway.get_redis_client'):
            client = TestClient(app)
            response = client.get("/health")

            data = response.json()
            assert data["status"] == "healthy"


class TestTrainingClient:
    """Test Training API Client."""

    @pytest.mark.asyncio
    async def test_start_training(self):
        """Test starting a training job."""
        from src.api.training_client import TrainingAPIClient

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                "task_id": "test_123",
                "status": "started"
            }
            mock_response.raise_for_status = Mock()

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            client = TrainingAPIClient(
                base_url="http://localhost:8001",
                api_key="test-key"
            )

            result = await client.start_training(
                task_id="test_123",
                model="yolo11n",
                data_yaml="/data/test.yaml",
                epochs=10
            )

            assert result["task_id"] == "test_123"

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check."""
        from src.api.training_client import TrainingAPIClient

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            client = TrainingAPIClient(
                base_url="http://localhost:8001",
                api_key="test-key"
            )

            result = await client.health_check()
            assert result is True


class TestDataEndpoints:
    """Test Data API endpoints."""

    def test_search_endpoint_works(self):
        """Data search endpoint returns results."""
        from fastapi.testclient import TestClient
        from unittest.mock import Mock, patch

        with patch('src.api.gateway.get_redis_client'):
            with patch('src.api.training_client.TrainingAPIClient'):
                from src.api.gateway import app

                client = TestClient(app)

                with patch('src.data.discovery.DatasetDiscovery') as mock_discovery:
                    mock_instance = Mock()
                    mock_instance.search.return_value = []
                    mock_discovery.return_value = mock_instance

                    response = client.post(
                        "/api/v1/data/search",
                        json={"query": "test", "max_results": 5}
                    )

                    assert response.status_code == 200
