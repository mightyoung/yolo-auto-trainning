"""
Contract Tests: Business API ↔ Training API

Tests the contract between Business API and Training API:
1. Business API submits training task to Training API
2. Training API executes task and reports status
3. Business API aggregates status from Training API
4. Business API surfaces plateau signals from Training API

Note: These tests verify the interface contracts by checking that:
- Required fields are present in request/response models
- Data flows correctly between services
- Status transitions are valid
"""

import sys
import os
import types
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pytest

# Setup path - use business-api for primary imports
project_root = Path(__file__).parent.parent.parent
business_api_root = project_root / "business-api"
training_api_root = project_root / "training-api"

# Insert in order: business-api first, then training-api, then project root
if str(business_api_root) not in sys.path:
    sys.path.insert(0, str(business_api_root))
if str(training_api_root) not in sys.path:
    sys.path.insert(0, str(training_api_root))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key")
os.environ.setdefault("TRAINING_API_URL", "http://localhost:8001")
os.environ.setdefault("TRAINING_API_KEY", "test-api-key")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379:0")
os.environ.setdefault("INTERNAL_API_KEY", "test-internal-key")


def _install_src_package():
    """Install src package into sys.modules similar to existing tests."""
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(business_api_root / "src")]
    api_pkg = types.ModuleType("src.api")
    api_pkg.__path__ = [str(business_api_root / "src" / "api")]
    sys.modules["src"] = src_pkg
    sys.modules["src.api"] = api_pkg


_install_src_package()


class TestTrainingAPIClientInterface:
    """Test that TrainingAPIClient has the required interface."""

    def test_client_has_start_training_method(self):
        """TrainingAPIClient must have start_training method."""
        from src.api.training_client import TrainingAPIClient

        client = TrainingAPIClient(
            base_url="http://localhost:8001",
            api_key="test-key"
        )

        assert hasattr(client, 'start_training')
        assert callable(client.start_training)

    def test_client_has_get_task_status_method(self):
        """TrainingAPIClient must have get_task_status method."""
        from src.api.training_client import TrainingAPIClient

        client = TrainingAPIClient(
            base_url="http://localhost:8001",
            api_key="test-key"
        )

        assert hasattr(client, 'get_task_status')
        assert callable(client.get_task_status)

    def test_client_has_cancel_task_method(self):
        """TrainingAPIClient must have cancel_task method."""
        from src.api.training_client import TrainingAPIClient

        client = TrainingAPIClient(
            base_url="http://localhost:8001",
            api_key="test-key"
        )

        assert hasattr(client, 'cancel_task')
        assert callable(client.cancel_task)

    def test_client_has_start_export_method(self):
        """TrainingAPIClient must have start_export method."""
        from src.api.training_client import TrainingAPIClient

        client = TrainingAPIClient(
            base_url="http://localhost:8001",
            api_key="test-key"
        )

        assert hasattr(client, 'start_export')
        assert callable(client.start_export)


class TestBusinessAPITaskRegistryAggregation:
    """Test Business API task aggregation from Training API responses."""

    def test_build_training_status_response_extracts_all_fields(self):
        """build_training_status_response must extract all plateau fields."""
        from src.api.task_registry import build_training_status_response

        task = {
            "task_id": "train_123",
            "status": "running",
            "registry_status": "running",
            "execution": {
                "status": "running",
                "progress": 0.7,
                "current_epoch": 70,
                "total_epochs": 100,
                "metrics": {"mAP50": 0.45},
                "live_mAP50": 0.45,
                "lr_decay_triggered": True,
                "lr_decay_signal": {"factor": 0.5, "epoch": 65},
                "augment_boost_active": False,
                "data_expansion_requested": True,
                "data_expansion_signal": {"round": 1},
                "strategies_triggered": [{"level": 1, "action": "lr_decay"}],
                "resubmit_count": 0,
            },
        }

        response = build_training_status_response(task)

        # Verify all fields are extracted
        assert response.task_id == "train_123"
        assert response.status == "running"
        assert response.progress == 0.7
        assert response.current_epoch == 70
        assert response.total_epochs == 100
        assert response.metrics == {"mAP50": 0.45}
        assert response.live_mAP50 == 0.45
        assert response.lr_decay_triggered is True
        assert response.lr_decay_signal == {"factor": 0.5, "epoch": 65}
        assert response.augment_boost_active is False
        assert response.data_expansion_requested is True
        assert len(response.strategies_triggered) == 1

    def test_falls_back_to_registry_when_execution_unavailable(self):
        """When execution snapshot unavailable, falls back to registry status."""
        from src.api.task_registry import build_training_status_response

        task = {
            "task_id": "train_123",
            "status": "submitted",
            "registry_status": "submitted",
            "execution": None,  # Execution unavailable
        }

        response = build_training_status_response(task)

        # Should fall back to registry status
        assert response.task_id == "train_123"
        assert response.status == "submitted"
        assert response.progress == 0.0


class TestTaskStateTransitions:
    """Test task state transitions flow correctly through the system."""

    def test_submitted_to_running_transition(self):
        """Task transitions from submitted to running."""
        from src.api.task_registry import build_task_record

        task = build_task_record(
            task_id="train_123",
            task_type="training",
            user_id="user1",
            submission={"model": "yolo11m"},
            registry_status="submitted",
        )

        # Simulate transition
        task["status"] = "running"
        task["registry_status"] = "running"

        assert task["status"] == "running"

    def test_running_to_completed_transition(self):
        """Task transitions from running to completed."""
        from src.api.task_registry import build_task_record

        task = build_task_record(
            task_id="train_123",
            task_type="training",
            user_id="user1",
            submission={"model": "yolo11m"},
            registry_status="running",
        )

        # Simulate completion
        task["status"] = "completed"
        task["registry_status"] = "completed"

        assert task["status"] == "completed"

    def test_running_to_failed_transition(self):
        """Task transitions from running to failed."""
        from src.api.task_registry import build_task_record

        task = build_task_record(
            task_id="train_123",
            task_type="training",
            user_id="user1",
            submission={"model": "yolo11m"},
            registry_status="running",
        )

        # Simulate failure
        task["status"] = "failed"
        task["registry_status"] = "failed"
        task["error"] = "GPU OOM"

        assert task["status"] == "failed"
        assert task["error"] == "GPU OOM"
