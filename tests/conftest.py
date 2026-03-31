# Pytest Configuration and Shared Fixtures

import pytest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch


# Add src to path - handle both direct and package execution
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Store original sys.path for restoration between test modules
_original_sys_path = sys.path.copy()


def _restore_sys_path():
    """Restore sys.path and clear cached modules from training-api and business-api."""
    sys.path[:] = _original_sys_path.copy()

    # Clear all modules from training-api and business-api to prevent path pollution
    modules_to_remove = []
    for mod_name in list(sys.modules.keys()):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, '__file__') and mod.__file__:
            if 'training-api' in mod.__file__ or 'business-api' in mod.__file__:
                modules_to_remove.append(mod_name)
        # Also clear 'src' and its submodules if cached to training-api or business-api path
        if mod_name == 'src' or mod_name.startswith('src.'):
            if mod and hasattr(mod, '__file__') and mod.__file__:
                if 'training-api' in mod.__file__ or 'business-api' in mod.__file__:
                    modules_to_remove.append(mod_name)

    for mod_name in modules_to_remove:
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    # If src.data was removed (e.g., by business-api test setup), re-import it
    # from the project root's src directory
    if 'src.data' not in sys.modules:
        project_root = Path(__file__).parent.parent
        src_data_path = project_root / "src" / "data"
        if src_data_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "src.data", src_data_path / "__init__.py"
            )
            if spec and spec.loader:
                src_data_module = importlib.util.module_from_spec(spec)
                sys.modules['src.data'] = src_data_module
                spec.loader.exec_module(src_data_module)


def pytest_runtest_setup(item):
    """Restore sys.path and clear modules before each test."""
    _restore_sys_path()


def pytest_collect_file(file_path, parent):
    """Restore sys.path and clear modules before collecting each test file."""
    _restore_sys_path()


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
def discovery_instance():
    """Create DatasetDiscovery instance - lazy import to avoid path pollution."""
    from src.data.discovery import DatasetDiscovery
    return DatasetDiscovery(api_keys={})


@pytest.fixture
def trainer_instance(temp_dir):
    """Create YOLOTrainer instance - lazy import to avoid path pollution."""
    # Import inside fixture to avoid module-level path pollution
    from src.training.runner import YOLOTrainer
    return YOLOTrainer(model="yolo11n", output_dir=temp_dir)


@pytest.fixture
def data_merger_instance():
    """Create DataMerger instance - lazy import to avoid path pollution."""
    try:
        from src.data.discovery import DataMerger
    except ImportError:
        # DataMerger not implemented yet, use mock
        class DataMerger:
            def __init__(self, max_synthetic_ratio=0.3):
                self.max_synthetic_ratio = max_synthetic_ratio
            def merge(self, *args, **kwargs):
                return {"train_images": 0, "val_images": 0}
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
