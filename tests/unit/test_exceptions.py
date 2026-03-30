"""
Exception Hierarchy Tests

Tests the 4-tier exception classification:
1. BusinessError - Business logic errors
2. ExternalDependencyError - External service failures
3. StateConflictError - State conflicts
4. ConfigurationError - Configuration errors
"""

import pytest
import sys
from pathlib import Path

# Add business-api/src to path for imports
business_api_root = Path(__file__).parent.parent.parent / "business-api" / "src"
sys.path.insert(0, str(business_api_root))

from api.exceptions import (
    BusinessError,
    ExternalDependencyError,
    StateConflictError,
    ConfigurationError,
    task_not_found,
    task_not_owned,
    training_api_unavailable,
    redis_unavailable,
    invalid_task_state,
)


class TestBusinessError:
    """Test BusinessError exception."""

    def test_creation_with_message(self):
        err = BusinessError("Invalid model name")
        assert str(err) == "Invalid model name"
        assert err.code == "BUSINESS_ERROR"
        assert err.details == {}

    def test_creation_with_code_and_details(self):
        err = BusinessError(
            "Invalid training params",
            code="INVALID_PARAMS",
            details={"field": "epochs", "reason": "must be positive"},
        )
        assert err.code == "INVALID_PARAMS"
        assert err.details["field"] == "epochs"

    def test_isinstance_check(self):
        err = BusinessError("test")
        assert isinstance(err, Exception)


class TestExternalDependencyError:
    """Test ExternalDependencyError exception."""

    def test_creation_with_service(self):
        err = ExternalDependencyError(
            "Training API timeout",
            service="training-api",
        )
        assert str(err) == "Training API timeout"
        assert err.service == "training-api"
        assert err.code == "EXTERNAL_ERROR"
        assert err.retry_after is None

    def test_creation_with_retry_after(self):
        err = ExternalDependencyError(
            "Redis connection failed",
            service="redis",
            retry_after=5,
        )
        assert err.retry_after == 5
        assert err.service == "redis"


class TestStateConflictError:
    """Test StateConflictError exception."""

    def test_creation_with_resource_info(self):
        err = StateConflictError(
            "Task already completed",
            resource_type="task",
            resource_id="train_123",
            current_state="completed",
        )
        assert str(err) == "Task already completed"
        assert err.resource_type == "task"
        assert err.resource_id == "train_123"
        assert err.current_state == "completed"
        assert err.code == "STATE_CONFLICT"


class TestConfigurationError:
    """Test ConfigurationError exception."""

    def test_creation_with_config_key(self):
        err = ConfigurationError(
            "JWT_SECRET_KEY not set",
            config_key="JWT_SECRET_KEY",
            config_source="environment",
        )
        assert str(err) == "JWT_SECRET_KEY not set"
        assert err.config_key == "JWT_SECRET_KEY"
        assert err.config_source == "environment"
        assert err.code == "CONFIG_ERROR"


class TestConvenienceFunctions:
    """Test convenience functions for common error patterns."""

    def test_task_not_found(self):
        err = task_not_found("train_456")
        assert "train_456" in str(err)
        assert err.resource_type == "task"
        assert err.resource_id == "train_456"

    def test_task_not_owned(self):
        err = task_not_owned("train_456", "user2")
        assert "train_456" in str(err)
        assert "user2" in str(err)
        assert err.resource_type == "task"
        assert err.resource_id == "train_456"

    def test_training_api_unavailable(self):
        err = training_api_unavailable()
        assert err.service == "training-api"
        assert err.retry_after == 5

    def test_redis_unavailable(self):
        err = redis_unavailable()
        assert err.service == "redis"
        assert err.retry_after == 2

    def test_invalid_task_state(self):
        err = invalid_task_state("train_123", "completed", ["running", "submitted"])
        assert "train_123" in str(err)
        assert "completed" in str(err)
        assert err.current_state == "completed"
        assert err.resource_type == "task"
