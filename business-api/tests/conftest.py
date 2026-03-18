# Business API Test Configuration

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, Any

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ==================== Mock Dataset ====================

class MockDataset:
    """Mock dataset object for testing."""
    def __init__(
        self,
        name: str = "test-dataset",
        source: str = "roboflow",
        url: str = "https://example.com/dataset",
        license: str = "MIT",
        images: int = 1000,
        relevance_score: float = 0.95
    ):
        self.name = name
        self.source = source
        self.url = url
        self.license = license
        self.images = images
        self.relevance_score = relevance_score


# ==================== Fixtures ====================

@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    client = Mock()
    client.ping.return_value = True
    client.get.return_value = None
    client.set.return_value = True
    client.hset.return_value = True
    client.hget.return_value = None
    client.delete.return_value = True
    return client


@pytest.fixture
def mock_training_client():
    """Mock Training API client."""
    client = Mock()
    client.health_check = AsyncMock(return_value=True)
    client.start_training = AsyncMock(return_value={
        "task_id": "test_123",
        "status": "started"
    })
    client.start_hpo = AsyncMock(return_value={
        "task_id": "hpo_123",
        "status": "started"
    })
    client.start_export = AsyncMock(return_value={
        "task_id": "export_123",
        "status": "started"
    })
    client.get_task_status = AsyncMock(return_value={
        "task_id": "test_123",
        "status": "running",
        "progress": 0.5,
        "current_epoch": 50,
        "total_epochs": 100,
        "metrics": {"mAP50": 0.75}
    })
    client.cancel_task = AsyncMock(return_value={
        "task_id": "test_123",
        "status": "cancelled"
    })
    return client


@pytest.fixture
def mock_dataset_discovery():
    """Mock Dataset Discovery."""
    discovery = Mock()
    discovery.search = Mock(return_value=[
        MockDataset(
            name="car-detection-dataset",
            source="roboflow",
            images=5000,
            relevance_score=0.95
        ),
        MockDataset(
            name="vehicle-detection",
            source="kaggle",
            images=3000,
            relevance_score=0.85
        ),
        MockDataset(
            name="traffic-detection",
            source="huggingface",
            images=4500,
            relevance_score=0.80
        ),
    ])
    return discovery


@pytest.fixture
def mock_deepanalyze_client():
    """Mock DeepAnalyze client."""
    client = Mock()
    client.health_check = Mock(return_value=True)
    client.analyze_dataset = Mock(return_value={
        "status": "completed",
        "content": "Analysis completed successfully",
        "files": [
            {"name": "report.json", "path": "/tmp/report.json"}
        ]
    })
    client.generate_report = Mock(return_value={
        "status": "completed",
        "content": "Report generated successfully",
        "files": [
            {"name": "report.md", "path": "/tmp/report.md"}
        ]
    })
    return client


@pytest.fixture
def app_with_mocks(mock_redis, mock_training_client, mock_dataset_discovery, mock_deepanalyze_client):
    """Create FastAPI app with all mocks."""
    from unittest.mock import patch, MagicMock
    from fastapi.testclient import TestClient
    from business_api.src.api import gateway

    # Create mock app state
    mock_app = MagicMock()
    mock_app.state = MagicMock()
    mock_app.state.redis = mock_redis
    mock_app.state.training_client = mock_training_client

    with patch.object(gateway, 'get_redis_client', return_value=mock_redis):
        with patch('business_api.src.api.routes.DatasetDiscovery', return_value=mock_dataset_discovery):
            with patch('business_api.src.api.training_client.TrainingAPIClient', return_value=mock_training_client):
                with patch('business_api.src.api.deepanalyze_client.DeepAnalyzeClient', return_value=mock_deepanalyze_client):
                    # Set environment variables for gateway
                    with patch.dict('os.environ', {
                        'JWT_SECRET_KEY': 'test-secret-key',
                        'TRAINING_API_URL': 'http://localhost:8001',
                        'TRAINING_API_KEY': 'test-api-key',
                        'REDIS_URL': 'redis://localhost:6379/0'
                    }, clear=False):
                        from business_api.src.api.gateway import app
                        yield app


@pytest.fixture
def client(app_with_mocks, mock_training_client):
    """Create TestClient with mocked dependencies."""
    from fastapi.testclient import TestClient
    return TestClient(app_with_mocks)
