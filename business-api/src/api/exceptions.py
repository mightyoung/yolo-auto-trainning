"""
Business API Exception Hierarchy
================================

Unified exception classification for the Business API.
All exceptions are categorized into one of four groups:

1. BusinessError - Business logic errors (validation, state transitions)
2. ExternalDependencyError - External service failures (Training API, Redis, MLflow)
3. StateConflictError - Concurrent modification or state conflicts
4. ConfigurationError - Missing or invalid configuration

Usage:
    from src.api.exceptions import BusinessError, ExternalDependencyError

    try:
        result = await client.start_training(...)
    except ExternalDependencyError as e:
        logger.error(f"Training API unavailable: {e}")
        raise HTTPException(status_code=503, detail="Training service temporarily unavailable")
    except BusinessError as e:
        logger.warning(f"Invalid training request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
"""



class BusinessError(Exception):
    """Business logic errors - invalid input, state violations, etc.

    These errors indicate the request was fundamentally invalid,
    not a system or external problem.
    """

    def __init__(self, message: str, *, code: str | None = None, details: dict | None = None):
        super().__init__(message)
        self.code = code or "BUSINESS_ERROR"
        self.details = details or {}


class ExternalDependencyError(Exception):
    """External service failures - Training API, Redis, MLflow, etc.

    These errors indicate a downstream service is unavailable or returned an error.
    The client may retry these after a delay.
    """

    def __init__(self, message: str, *, service: str | None = None, code: str | None = None, retry_after: int | None = None):
        super().__init__(message)
        self.service = service
        self.code = code or "EXTERNAL_ERROR"
        self.retry_after = retry_after  # seconds


class StateConflictError(Exception):
    """Concurrent modification or state conflicts.

    These errors occur when:
    - Two requests modify the same resource simultaneously
    - A resource is in an unexpected state for the requested operation
    - Optimistic locking fails
    """

    def __init__(self, message: str, *, resource_type: str | None = None, resource_id: str | None = None, current_state: str | None = None):
        super().__init__(message)
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.current_state = current_state
        self.code = "STATE_CONFLICT"


class ConfigurationError(Exception):
    """Configuration errors - missing env vars, invalid values, etc.

    These errors indicate the application is misconfigured and cannot start
    or the request includes invalid configuration.
    """

    def __init__(self, message: str, *, config_key: str | None = None, config_source: str | None = None):
        super().__init__(message)
        self.config_key = config_key
        self.config_source = config_source
        self.code = "CONFIG_ERROR"


# Convenience functions for common error patterns

def task_not_found(task_id: str) -> StateConflictError:
    """Create a TaskNotFound error."""
    return StateConflictError(
        f"Task not found: {task_id}",
        resource_type="task",
        resource_id=task_id,
    )


def task_not_owned(task_id: str, user_id: str) -> StateConflictError:
    """Create a TaskNotOwned error."""
    return StateConflictError(
        f"Task {task_id} does not belong to user {user_id}",
        resource_type="task",
        resource_id=task_id,
    )


def training_api_unavailable(detail: str = "Training API unavailable") -> ExternalDependencyError:
    """Create a TrainingAPIUnavailable error."""
    return ExternalDependencyError(
        detail,
        service="training-api",
        code="TRAINING_API_UNAVAILABLE",
        retry_after=5,
    )


def redis_unavailable(detail: str = "Redis unavailable") -> ExternalDependencyError:
    """Create a RedisUnavailable error."""
    return ExternalDependencyError(
        detail,
        service="redis",
        code="REDIS_UNAVAILABLE",
        retry_after=2,
    )


def invalid_task_state(task_id: str, current_state: str, expected_states: list[str]) -> StateConflictError:
    """Create an InvalidTaskState error."""
    return StateConflictError(
        f"Task {task_id} is in state '{current_state}', expected one of {expected_states}",
        resource_type="task",
        resource_id=task_id,
        current_state=current_state,
    )
