# Integration Tests - API Routes
#
# NOTE: Uses httpx.AsyncClient instead of Starlette TestClient to avoid
# Python 3.14 metaclass conflict when tests are collected together with
# modules that mock sys.modules (e.g., test_agents.py)

import os
import pytest
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src to path - handle both direct and package execution
test_dir = Path(__file__).parent
project_root = test_dir.parent.parent
src_path = project_root / "src"
business_api_path = project_root / "business-api"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(business_api_path) not in sys.path:
    sys.path.insert(0, str(business_api_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set required environment variables BEFORE importing gateway
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("TRAINING_API_URL", "http://localhost:8001")
os.environ.setdefault("TRAINING_API_KEY", "test-api-key")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

# Note: redis and celery are NOT mocked at module level because:
# 1. redis IS installed in the test environment (real redis package)
# 2. celery is NOT installed, but the integration tests use patch() on
#    src.api.gateway.get_redis_client rather than importing gateway directly
# 3. Module-level mocks persisted in sys.modules and polluted subsequent tests

pytestmark = pytest.mark.integration


# ==================== Fixtures ====================

_original_integration_modules = {}


@pytest.fixture(autouse=True)
def _setup_integration_mocks():
    """Set up redis/celery mocks at test execution time, clean up after.

    This fixture mocks redis and celery for tests that need them.
    """
    global _original_integration_modules

    _original_integration_modules['redis'] = sys.modules.get('redis')
    _original_integration_modules['celery'] = sys.modules.get('celery')

    sys.modules['redis'] = MagicMock()
    sys.modules['celery'] = MagicMock()

    yield

    for mod_name in ('redis', 'celery'):
        orig = _original_integration_modules.get(mod_name)
        if orig is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = orig

    _original_integration_modules.clear()


# ==================== Test Health Endpoint ====================

class TestHealthEndpoint:
    """Test health check endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client using httpx AsyncClient to avoid Starlette metaclass conflict."""
        import httpx
        from api.gateway import app
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    @pytest.mark.asyncio
    async def test_health_check_returns_200(self, client):
        """Health check returns 200."""
        with patch('src.api.gateway.get_redis_client'):
            response = await client.get("/health")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_check_returns_healthy(self, client):
        """Health check returns healthy status."""
        with patch('src.api.gateway.get_redis_client'):
            response = await client.get("/health")
            data = response.json()
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_returns_version(self, client):
        """Health check returns version."""
        with patch('src.api.gateway.get_redis_client'):
            response = await client.get("/health")
            data = response.json()
            assert "version" in data


# ==================== Test Data Endpoints ====================

class TestDataEndpoints:
    """Test data API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client using httpx AsyncClient."""
        import httpx
        from api.gateway import app
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    @pytest.mark.asyncio
    async def test_search_datasets_endpoint(self, client):
        """Data search endpoint works."""
        with patch('src.data.discovery.DatasetDiscovery') as mock_discovery:
            mock_instance = Mock()
            mock_instance.search.return_value = []
            mock_discovery.return_value = mock_instance

            response = await client.post(
                "/api/v1/data/search",
                json={"query": "car detection", "max_results": 5}
            )

            assert response.status_code == 200
            assert "datasets" in response.json()

    @pytest.mark.asyncio
    async def test_search_datasets_with_results(self, client):
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

            response = await client.post(
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
        """Create test client using httpx AsyncClient."""
        import httpx
        from api.gateway import app
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    @pytest.mark.asyncio
    async def test_train_start_endpoint(self, client):
        """Training start endpoint works."""
        response = await client.post(
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

    @pytest.mark.asyncio
    async def test_train_status_endpoint(self, client):
        """Training status endpoint works."""
        response = await client.get("/api/v1/train/status/test_task_123")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    @pytest.mark.asyncio
    async def test_train_results_endpoint(self, client):
        """Training results endpoint works."""
        response = await client.get("/api/v1/train/results/test_task_123")

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data


# ==================== Test Export Endpoints ====================

class TestExportEndpoints:
    """Test export API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client using httpx AsyncClient."""
        import httpx
        from api.gateway import app
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    @pytest.mark.asyncio
    async def test_export_endpoint(self, client):
        """Export endpoint works."""
        response = await client.post(
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

    @pytest.mark.asyncio
    async def test_export_status_endpoint(self, client):
        """Export status endpoint works."""
        response = await client.get("/api/v1/deploy/export/status/export_task_123")

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
