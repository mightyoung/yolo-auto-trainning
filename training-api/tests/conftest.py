# Training API Test Configuration

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
