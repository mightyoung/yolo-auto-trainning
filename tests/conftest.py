# Pytest Configuration and Shared Fixtures

import pytest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ==================== Fixtures ====================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_yaml(temp_dir):
    """Create sample dataset YAML for testing."""
    yaml_content = """
path: .
train: train/images
val: val/images
nc: 1
names: ['object']
"""
    yaml_path = temp_dir / "data.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    with patch('src.api.gateway.get_redis_client') as mock:
        client = Mock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_requests():
    """Mock requests library."""
    with patch('src.data.discovery.requests') as mock:
        yield mock


# ==================== Module Fixtures ====================

@pytest.fixture
def discovery_instance(temp_dir):
    """Create DatasetDiscovery instance."""
    from src.data.discovery import DatasetDiscovery
    return DatasetDiscovery(output_dir=temp_dir)


@pytest.fixture
def trainer_instance(temp_dir):
    """Create YOLOTrainer instance."""
    from src.training.runner import YOLOTrainer
    return YOLOTrainer(model="yolo11n", output_dir=temp_dir)


@pytest.fixture
def data_merger_instance():
    """Create DataMerger instance."""
    from src.data.discovery import DataMerger
    return DataMerger(max_synthetic_ratio=0.3)


# ==================== Auth Fixtures ====================

@pytest.fixture
def sample_token_payload():
    """Sample JWT token payload."""
    return {"sub": "test_user", "type": "access"}


@pytest.fixture
def sample_user_data():
    """Sample user data for authentication."""
    return {"user_id": "test_user", "email": "test@example.com"}
