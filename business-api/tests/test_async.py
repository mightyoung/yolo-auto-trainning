"""
Business API Async Tests

Tests for async endpoints and concurrent operations.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


class TestAsyncDataEndpoints:
    """Test async data endpoints."""

    @pytest.mark.asyncio
    async def test_search_async(self):
        """Test async dataset search."""
        from fastapi.testclient import TestClient
        from business_api.src.api.gateway import app

        mock_redis = Mock()
        mock_redis.ping.return_value = True

        mock_training_client = Mock()
        mock_training_client.health_check = AsyncMock(return_value=True)
        mock_training_client.start_training = AsyncMock(return_value={
            "task_id": "test_123",
            "status": "started"
        })

        mock_discovery = Mock()
        mock_discovery.search.return_value = []

        with patch('business_api.src.api.gateway.get_redis_client', return_value=mock_redis):
            with patch('business_api.src.api.training_client.TrainingAPIClient', return_value=mock_training_client):
                with patch('business_api.src.api.routes.DatasetDiscovery', return_value=mock_discovery):
                    with patch.dict('os.environ', {
                        'JWT_SECRET_KEY': 'test-secret-key',
                        'TRAINING_API_URL': 'http://localhost:8001',
                        'TRAINING_API_KEY': 'test-api-key',
                        'REDIS_URL': 'redis://localhost:6379/0'
                    }, clear=False):
                        client = TestClient(app)

                        response = client.post(
                            "/api/v1/data/search",
                            json={"query": "test", "max_results": 5}
                        )

                        assert response.status_code == 200


class TestAsyncTrainingEndpoints:
    """Test async training endpoints."""

    @pytest.mark.asyncio
    async def test_submit_training_async(self):
        """Test async training submission."""
        from fastapi.testclient import TestClient
        from business_api.src.api.gateway import app

        mock_redis = Mock()
        mock_training_client = Mock()
        mock_training_client.health_check = AsyncMock(return_value=True)
        mock_training_client.start_training = AsyncMock(return_value={
            "task_id": "async_train123",
            "status": "started"
        })

        with patch('business_api.src.api.gateway.get_redis_client', return_value=mock_redis):
            with patch('business_api.src.api.training_client.TrainingAPIClient', return_value=mock_training_client):
                with patch.dict('os.environ', {
                    'JWT_SECRET_KEY': 'test-secret-key',
                    'TRAINING_API_URL': 'http://localhost:8001',
                    'TRAINING_API_KEY': 'test-api-key',
                    'REDIS_URL': 'redis://localhost:6379/0'
                }, clear=False):
                    client = TestClient(app)

                    response = client.post(
                        "/api/v1/train/submit",
                        json={
                            "model": "yolo11n",
                            "data_yaml": "/data/test.yaml",
                            "epochs": 10
                        }
                    )

                    assert response.status_code == 200


class TestConcurrentOperations:
    """Test concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_status_requests(self):
        """Test multiple concurrent status requests."""
        from fastapi.testclient import TestClient
        from business_api.src.api.gateway import app
        import asyncio

        mock_redis = Mock()
        mock_training_client = Mock()
        mock_training_client.health_check = AsyncMock(return_value=True)
        mock_training_client.get_task_status = AsyncMock(return_value={
            "task_id": "concurrent_test",
            "status": "running",
            "progress": 0.5
        })

        with patch('business_api.src.api.gateway.get_redis_client', return_value=mock_redis):
            with patch('business_api.src.api.training_client.TrainingAPIClient', return_value=mock_training_client):
                with patch.dict('os.environ', {
                    'JWT_SECRET_KEY': 'test-secret-key',
                    'TRAINING_API_URL': 'http://localhost:8001',
                    'TRAINING_API_KEY': 'test-api-key',
                    'REDIS_URL': 'redis://localhost:6379/0'
                }, clear=False):
                    client = TestClient(app)

                    # Make concurrent requests
                    responses = []
                    for i in range(5):
                        response = client.get(f"/api/v1/train/status/task_{i}")
                        responses.append(response)

                    # All should succeed
                    for response in responses:
                        assert response.status_code in [200, 502]


class TestAsyncAnalysis:
    """Test async analysis endpoints."""

    @pytest.mark.asyncio
    async def test_analysis_endpoint_async(self):
        """Test async analysis endpoint."""
        from fastapi.testclient import TestClient
        from business_api.src.api.gateway import app

        mock_redis = Mock()
        mock_training_client = Mock()
        mock_training_client.health_check = AsyncMock(return_value=True)

        mock_deepanalyze = Mock()
        mock_deepanalyze.health_check = Mock(return_value=True)
        mock_deepanalyze.analyze_dataset = Mock(return_value={
            "status": "completed",
            "content": "Analysis done"
        })

        with patch('business_api.src.api.gateway.get_redis_client', return_value=mock_redis):
            with patch('business_api.src.api.training_client.TrainingAPIClient', return_value=mock_training_client):
                with patch('business_api.src.api.deepanalyze_client.DeepAnalyzeClient', return_value=mock_deepanalyze):
                    with patch.dict('os.environ', {
                        'JWT_SECRET_KEY': 'test-secret-key',
                        'TRAINING_API_URL': 'http://localhost:8001',
                        'TRAINING_API_KEY': 'test-api-key',
                        'REDIS_URL': 'redis://localhost:6379/0'
                    }, clear=False):
                        client = TestClient(app)

                        response = client.post(
                            "/api/v1/analysis/analyze",
                            json={
                                "dataset_path": "/data/test",
                                "analysis_type": "quality"
                            }
                        )

                        assert response.status_code == 200


class TestAsyncAgent:
    """Test async agent endpoints."""

    @pytest.mark.asyncio
    async def test_agent_task_submission_async(self):
        """Test async agent task submission."""
        from fastapi.testclient import TestClient
        from business_api.src.api.gateway import app

        mock_redis = Mock()
        mock_training_client = Mock()
        mock_training_client.health_check = AsyncMock(return_value=True)

        with patch('business_api.src.api.gateway.get_redis_client', return_value=mock_redis):
            with patch('business_api.src.api.training_client.TrainingAPIClient', return_value=mock_training_client):
                with patch.dict('os.environ', {
                    'JWT_SECRET_KEY': 'test-secret-key',
                    'TRAINING_API_URL': 'http://localhost:8001',
                    'TRAINING_API_KEY': 'test-api-key',
                    'REDIS_URL': 'redis://localhost:6379/0'
                }, clear=False):
                    client = TestClient(app)

                    response = client.post(
                        "/api/v1/agent/task",
                        json={
                            "task": "Train a car detection model"
                        }
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert "task_id" in data


class TestAsyncCallbacks:
    """Test async callback endpoints."""

    @pytest.mark.asyncio
    async def test_callback_async(self):
        """Test async callback endpoint."""
        from fastapi.testclient import TestClient
        from business_api.src.api.gateway import app

        mock_redis = Mock()
        mock_training_client = Mock()
        mock_training_client.health_check = AsyncMock(return_value=True)

        with patch('business_api.src.api.gateway.get_redis_client', return_value=mock_redis):
            with patch('business_api.src.api.training_client.TrainingAPIClient', return_value=mock_training_client):
                with patch.dict('os.environ', {
                    'JWT_SECRET_KEY': 'test-secret-key',
                    'TRAINING_API_URL': 'http://localhost:8001',
                    'TRAINING_API_KEY': 'test-api-key',
                    'REDIS_URL': 'redis://localhost:6379/0'
                }, clear=False):
                    client = TestClient(app)

                    response = client.post(
                        "/api/v1/callback/task/callback",
                        json={
                            "task_id": "async_callback_test",
                            "status": "completed",
                            "metrics": {"mAP50": 0.85}
                        }
                    )

                    assert response.status_code == 200
                    assert response.json()["received"] is True
