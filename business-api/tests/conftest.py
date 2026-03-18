# Business API Test Configuration

import pytest
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    from unittest.mock import Mock
    client = Mock()
    client.ping.return_value = True
    return client


@pytest.fixture
def mock_training_client():
    """Mock Training API client."""
    from unittest.mock import Mock
    client = Mock()
    client.health_check.return_value = True
    client.start_training.return_value = {
        "task_id": "test_123",
        "status": "started"
    }
    client.get_task_status.return_value = {
        "task_id": "test_123",
        "status": "running",
        "progress": 0.5
    }
    return client
