# Integration Tests - API Routes

import pytest
from pathlib import Path
import sys
from unittest.mock import Mock, patch

# Add src to path - handle both direct and package execution
test_dir = Path(__file__).parent
project_root = test_dir.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Skip if dependencies not available
pytestmark = pytest.mark.integration


# ==================== Test Health Endpoint ====================

class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_returns_200(self):
        """Health check returns 200."""
        from fastapi.testclient import TestClient
        from api.gateway import app

        with patch('src.api.gateway.get_redis_client'):
            client = TestClient(app)
            response = client.get("/health")

            assert response.status_code == 200

    def test_health_check_returns_healthy(self):
        """Health check returns healthy status."""
        from fastapi.testclient import TestClient
        from api.gateway import app

        with patch('src.api.gateway.get_redis_client'):
            client = TestClient(app)
            response = client.get("/health")

            data = response.json()
            assert data["status"] == "healthy"

    def test_health_check_returns_version(self):
        """Health check returns version."""
        from fastapi.testclient import TestClient
        from api.gateway import app

        with patch('src.api.gateway.get_redis_client'):
            client = TestClient(app)
            response = client.get("/health")

            data = response.json()
            assert "version" in data


# ==================== Test Data Endpoints ====================

class TestDataEndpoints:
    """Test data API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from api.gateway import app

        # Patch before creating TestClient
        with patch('src.data.discovery.DatasetDiscovery') as mock:
            self._mock_discovery = mock
            mock_instance = Mock()
            mock_instance.search.return_value = []
            mock.return_value = mock_instance
            client = TestClient(app)
            return client

    def test_search_datasets_endpoint(self, client):
        """Data search endpoint works."""
        with patch('src.data.discovery.DatasetDiscovery') as mock_discovery:
            mock_instance = Mock()
            mock_instance.search.return_value = []
            mock_discovery.return_value = mock_instance

            response = client.post(
                "/api/v1/data/search",
                json={"query": "car detection", "max_results": 5}
            )

            assert response.status_code == 200
            assert "datasets" in response.json()

    def test_search_datasets_with_results(self, client):
        """Data search returns results."""
        from src.data.discovery import DatasetInfo

        with patch('src.data.discovery.DatasetDiscovery') as mock_discovery:
            mock_instance = Mock()
            mock_instance.search.return_value = [
                DatasetInfo(
                    source="roboflow",
                    name="car-detection",
                    url="https://example.com",
                    license="MIT",
                    annotations="coco",
                    images=1000,
                    categories=["car"],
                    relevance_score=0.9,
                )
            ]
            mock_discovery.return_value = mock_instance

            response = client.post(
                "/api/v1/data/search",
                json={"query": "car detection", "max_results": 10}
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["datasets"]) == 1
            assert data["datasets"][0]["name"] == "car-detection"


# ==================== Test Training Endpoints ====================

class TestTrainingEndpoints:
    """Test training API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from api.gateway import app
        return TestClient(app)

    def test_train_start_endpoint(self, client):
        """Training start endpoint works."""
        with patch('src.api.tasks.training_task'):
            response = client.post(
                "/api/v1/train/start",
                json={
                    "data_yaml": "/data/dataset.yaml",
                    "model": "yolo11m",
                    "epochs": 100,
                    "imgsz": 640,
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert "task_id" in data
            assert data["status"] == "submitted"

    def test_train_status_endpoint(self, client):
        """Training status endpoint works."""
        response = client.get("/api/v1/train/status/test_task_123")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_train_results_endpoint(self, client):
        """Training results endpoint works."""
        response = client.get("/api/v1/train/results/test_task_123")

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data


# ==================== Test Export Endpoints ====================

class TestExportEndpoints:
    """Test export API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from api.gateway import app
        return TestClient(app)

    def test_export_endpoint(self, client):
        """Export endpoint works."""
        with patch('src.api.tasks.export_task'):
            response = client.post(
                "/api/v1/deploy/export",
                json={
                    "model_path": "/models/best.pt",
                    "platform": "jetson_orin",
                    "imgsz": 640,
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert "task_id" in data

    def test_export_status_endpoint(self, client):
        """Export status endpoint works."""
        response = client.get("/api/v1/deploy/export/status/export_task_123")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data


# ==================== Test API Models ====================

class TestAPIModels:
    """Test API request/response models."""

    def test_dataset_search_request(self):
        """DatasetSearchRequest model works."""
        from api.routes import DatasetSearchRequest

        request = DatasetSearchRequest(
            query="car detection",
            max_results=10
        )

        assert request.query == "car detection"
        assert request.max_results == 10

    def test_train_request(self):
        """TrainRequest model works."""
        from api.routes import TrainRequest

        request = TrainRequest(
            data_yaml="/data/dataset.yaml",
            model="yolo11m",
            epochs=100,
            imgsz=640,
        )

        assert request.data_yaml == "/data/dataset.yaml"
        assert request.model == "yolo11m"
        assert request.epochs == 100

    def test_train_response(self):
        """TrainResponse model works."""
        from api.routes import TrainResponse

        response = TrainResponse(
            task_id="test_123",
            status="submitted",
            message="Training started"
        )

        assert response.task_id == "test_123"
        assert response.status == "submitted"
