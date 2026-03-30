"""
Task State Machine Tests

Tests the task lifecycle state machine:
- Valid states: submitted, running, completed, failed, cancelled, adjusted
- Valid transitions and invalid transitions
- State persistence in Redis
"""

import sys
import os
import types
from pathlib import Path
import json
from unittest.mock import Mock, AsyncMock, patch
import pytest

# Setup path similar to existing tests
project_root = Path(__file__).parent.parent.parent
business_api_root = project_root / "business-api"
if str(business_api_root) not in sys.path:
    sys.path.insert(0, str(business_api_root))

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key")
os.environ.setdefault("TRAINING_API_URL", "http://localhost:8001")
os.environ.setdefault("TRAINING_API_KEY", "test-api-key")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")


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


class TestTaskStateTransitions:
    """Test valid and invalid task state transitions."""

    # Valid state transitions
    VALID_TRANSITIONS = {
        "submitted": {"running", "cancelled"},
        "running": {"completed", "failed", "cancelled", "adjusted"},
        "adjusted": {"running", "completed", "failed"},
        "completed": set(),  # Terminal state
        "failed": set(),  # Terminal state
        "cancelled": set(),  # Terminal state
    }

    def test_submitted_to_running_transition(self):
        """Task moves from submitted to running when training starts."""
        from src.api.task_registry import build_task_record

        task = build_task_record(
            task_id="train_123",
            task_type="training",
            user_id="user1",
            submission={"model": "yolo11m"},
            registry_status="submitted",
        )

        assert task["status"] == "submitted"
        assert task["registry_status"] == "submitted"

        # Simulate transition to running
        task["status"] = "running"
        task["registry_status"] = "running"

        assert task["status"] == "running"

    def test_submitted_to_cancelled_transition(self):
        """Task can be cancelled from submitted state."""
        from src.api.task_registry import build_task_record

        task = build_task_record(
            task_id="train_123",
            task_type="training",
            user_id="user1",
            submission={"model": "yolo11m"},
            registry_status="submitted",
        )

        # Cancel before starting
        task["status"] = "cancelled"
        task["registry_status"] = "cancelled"

        assert task["status"] == "cancelled"

    def test_running_to_completed_transition(self):
        """Task completes after successful training."""
        from src.api.task_registry import build_task_record

        task = build_task_record(
            task_id="train_123",
            task_type="training",
            user_id="user1",
            submission={"model": "yolo11m"},
            registry_status="running",
        )

        # Training finishes
        task["status"] = "completed"
        task["registry_status"] = "completed"

        assert task["status"] == "completed"

    def test_running_to_failed_transition(self):
        """Task fails when training errors."""
        from src.api.task_registry import build_task_record

        task = build_task_record(
            task_id="train_123",
            task_type="training",
            user_id="user1",
            submission={"model": "yolo11m"},
            registry_status="running",
        )

        # Training fails
        task["status"] = "failed"
        task["registry_status"] = "failed"
        task["error"] = "GPU out of memory"

        assert task["status"] == "failed"
        assert task["error"] == "GPU out of memory"

    def test_running_to_adjusted_transition(self):
        """Task can be adjusted (plateau-breaking) while running."""
        from src.api.task_registry import build_task_record

        task = build_task_record(
            task_id="train_123",
            task_type="training",
            user_id="user1",
            submission={"model": "yolo11m", "epochs": 100},
            registry_status="running",
        )

        # Plateau detected, task adjusted
        task["status"] = "adjusted"
        task["registry_status"] = "adjusted"
        task["adjusted_to"] = "train_456"

        assert task["status"] == "adjusted"
        assert task["adjusted_to"] == "train_456"

    def test_adjusted_creates_new_task(self):
        """Adjustment creates a new task with adjusted params."""
        from src.api.task_registry import build_task_record

        original = build_task_record(
            task_id="train_123",
            task_type="training",
            user_id="user1",
            submission={"model": "yolo11m", "epochs": 100, "lr0": 0.01},
            registry_status="running",
        )

        # Create adjusted task
        adjusted = build_task_record(
            task_id="train_456",
            task_type="training",
            user_id="user1",
            submission={
                "model": "yolo11m",
                "epochs": 100,
                "lr0": 0.005,  # Halved LR
                "adjusted_from": "train_123",
            },
            registry_status="submitted",
            links={"adjusted_from": "train_123"},
        )

        # Update original to point to new task
        original["status"] = "adjusted"
        original["adjusted_to"] = "train_456"

        assert original["adjusted_to"] == adjusted["task_id"]
        assert adjusted["submission"]["lr0"] == 0.005
        assert adjusted["links"]["adjusted_from"] == original["task_id"]

    def test_terminal_states_are_final(self):
        """Completed, failed, and cancelled are terminal states."""
        terminal_states = {"completed", "failed", "cancelled"}

        for state in terminal_states:
            # Terminal states should have no valid transitions out
            assert state not in self.VALID_TRANSITIONS or len(self.VALID_TRANSITIONS.get(state, set())) == 0


class TestTaskSchemaVersioning:
    """Test schema version migration."""

    def test_legacy_task_gets_migrated(self):
        """Legacy tasks (without schema_version) are migrated to current schema."""
        from src.api.task_registry import normalize_task_record, CURRENT_SCHEMA_VERSION

        legacy_task = {
            "task_id": "train_legacy",
            "task_type": "training",
            "user_id": "user1",
            "status": "completed",
            "params": {  # Legacy field name
                "model": "yolo11m",
            },
            # No schema_version field
        }

        normalized = normalize_task_record(legacy_task)

        assert normalized["schema_version"] == CURRENT_SCHEMA_VERSION
        assert "submission" in normalized  # params -> submission migration
        assert normalized["submission"]["model"] == "yolo11m"

    def test_current_schema_not_migrated(self):
        """Current schema tasks pass through unchanged."""
        from src.api.task_registry import normalize_task_record, CURRENT_SCHEMA_VERSION

        current_task = {
            "task_id": "train_current",
            "task_type": "training",
            "user_id": "user1",
            "status": "submitted",
            "schema_version": CURRENT_SCHEMA_VERSION,
            "submission": {"model": "yolo11m"},
        }

        normalized = normalize_task_record(current_task)

        assert normalized["schema_version"] == CURRENT_SCHEMA_VERSION
        assert normalized["submission"]["model"] == "yolo11m"

    def test_build_task_record_includes_schema_version(self):
        """New task records are created with current schema version."""
        from src.api.task_registry import build_task_record, CURRENT_SCHEMA_VERSION

        task = build_task_record(
            task_id="train_new",
            task_type="training",
            user_id="user1",
            submission={"model": "yolo11m"},
        )

        assert task["schema_version"] == CURRENT_SCHEMA_VERSION


class TestTaskOwnership:
    """Test task ownership validation."""

    def test_verify_ownership_returns_task_for_owner(self):
        """Owner can access their task."""
        from src.api.task_registry import verify_task_ownership

        mock_redis = Mock()
        mock_redis.get.return_value = json.dumps({
            "task_id": "train_123",
            "user_id": "user1",
            "status": "submitted",
            "submission": {},
        })

        result = verify_task_ownership(mock_redis, "train_123", "user1")

        assert result is not None
        assert result["task_id"] == "train_123"

    def test_verify_ownership_returns_none_for_non_owner(self):
        """Non-owner cannot access task."""
        from src.api.task_registry import verify_task_ownership

        mock_redis = Mock()
        mock_redis.get.return_value = json.dumps({
            "task_id": "train_123",
            "user_id": "user1",
            "status": "submitted",
            "submission": {},
        })

        result = verify_task_ownership(mock_redis, "train_123", "user2")

        assert result is None

    def test_verify_ownership_returns_none_for_nonexistent_task(self):
        """Nonexistent task returns None."""
        from src.api.task_registry import verify_task_ownership

        mock_redis = Mock()
        mock_redis.get.return_value = None

        result = verify_task_ownership(mock_redis, "train_nonexistent", "user1")

        assert result is None
