"""Task registry storage and aggregation helpers for the Business API."""

from __future__ import annotations

import json
from datetime import datetime

try:
    from ..agents.event_graph import append_graph_event, normalize_event_graph
except ImportError:  # pragma: no cover - compatibility with direct package imports
    from agents.event_graph import append_graph_event, normalize_event_graph
from .task_models import ExportStatusResponse, TrainStatusResponse

TASK_TTL_SECONDS = 7 * 24 * 60 * 60

# Schema version for Redis task records
# Increment this when the schema changes and add migration logic below
CURRENT_SCHEMA_VERSION = "v3"
TASK_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


# ==================== Task State Machine ====================
# Valid task statuses
TASK_STATUSES = frozenset([
    "submitted",
    "running",
    "awaiting_confirmation",
    "awaiting_training_confirmation",
    "downloading_dataset",
    "completed",
    "failed",
    "cancelled",
])

# Valid state transitions: current_status -> set of allowed next statuses
# Based on the actual workflow: submitted -> running -> completed/failed/cancelled
# or: submitted -> awaiting_confirmation -> running -> completed/failed/cancelled
VALID_TRANSITIONS: dict[str, frozenset] = {
    # Initial states
    "submitted": frozenset(["running", "awaiting_confirmation", "awaiting_training_confirmation", "downloading_dataset", "failed"]),
    # Running states
    "running": frozenset(["completed", "failed", "cancelled"]),
    "downloading_dataset": frozenset(["running", "awaiting_training_confirmation", "failed"]),
    "awaiting_training_confirmation": frozenset(["running", "cancelled"]),
    "awaiting_confirmation": frozenset(["running", "cancelled"]),
    # Terminal states (no outgoing transitions)
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, task_id: str, current_status: str, target_status: str):
        self.task_id = task_id
        self.current_status = current_status
        self.target_status = target_status
        allowed = VALID_TRANSITIONS.get(current_status, frozenset())
        super().__init__(
            f"Task {task_id}: invalid transition {current_status} -> {target_status}. "
            f"Allowed transitions: {set(allowed) if allowed else 'none (terminal state)'}"
        )


def is_terminal_task_status(status: str | None) -> bool:
    """Return True when the task status is terminal."""
    return bool(status and status.lower() in TASK_TERMINAL_STATUSES)


def validate_transition(current_status: str, target_status: str) -> bool:
    """Validate a status transition.

    Args:
        current_status: The current status of the task
        target_status: The target status to transition to

    Returns:
        True if the transition is valid

    Raises:
        InvalidTransitionError: If the transition is not valid
    """
    # Normalize status values
    current = current_status.lower() if current_status else ""
    target = target_status.lower() if target_status else ""

    # Allow same-state transitions (no-op)
    if current == target:
        return True

    # Check if current status is valid
    if current not in TASK_STATUSES:
        # Unknown current status - allow transition but log warning
        return True

    # Check if target status is valid
    if target not in TASK_STATUSES:
        raise InvalidTransitionError("", current, target)

    # Check if transition is allowed
    allowed = VALID_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise InvalidTransitionError("", current, target)

    return True


def assert_transition(task_id: str, current_status: str, target_status: str) -> None:
    """Assert that a status transition is valid, raising an exception if not.

    Args:
        task_id: The task ID (for error messages)
        current_status: The current status of the task
        target_status: The target status to transition to

    Raises:
        InvalidTransitionError: If the transition is not valid
    """
    validate_transition(current_status, target_status)


def get_allowed_transitions(current_status: str) -> frozenset:
    """Get the set of allowed transitions from the current status.

    Args:
        current_status: The current status of the task

    Returns:
        Frozenset of allowed target statuses
    """
    return VALID_TRANSITIONS.get(current_status.lower() if current_status else "", frozenset())


def migrate_task_record(task_data: dict) -> dict:
    """Migrate a task record to the current schema version.

    This function handles migrations from older schema versions to newer ones.
    Add new migration cases as the schema evolves.
    """
    schema_version = task_data.get("schema_version", "legacy")

    if schema_version == CURRENT_SCHEMA_VERSION:
        return task_data

    # Migration: legacy/v1/v2 -> v3
    if schema_version in {"legacy", "v1", "v2"}:
        # Legacy records had "params" instead of "submission"
        if "params" in task_data and "submission" not in task_data:
            task_data["submission"] = task_data.pop("params")
        task_data.setdefault("attempt_memory", [])
        task_data.setdefault("latest_attempt", None)
        task_data.setdefault("event_graph", {})
        task_data["schema_version"] = CURRENT_SCHEMA_VERSION
        return task_data

    # If we don't know the schema version, try to normalize what we can
    if schema_version not in (CURRENT_SCHEMA_VERSION, "legacy", "v1", "v2"):
        # Normalize fields but don't recursively call migrate
        if "params" in task_data and "submission" not in task_data:
            task_data["submission"] = task_data.pop("params")
        task_data.setdefault("attempt_memory", [])
        task_data.setdefault("latest_attempt", None)
        task_data.setdefault("event_graph", {})
        task_data["schema_version"] = CURRENT_SCHEMA_VERSION

    return task_data


def normalize_task_record(task_data: dict | None) -> dict | None:
    """Normalize legacy task records into registry + execution-link shape.

    Also applies schema migration if needed.
    """
    if task_data is None:
        return None

    # Apply schema migration first
    normalized = migrate_task_record(dict(task_data))
    task_type = normalized.get("task_type", "unknown")
    submission = dict(normalized.get("submission") or normalized.get("params") or {})
    links = dict(normalized.get("links") or {})

    adjusted_from = submission.get("adjusted_from") or normalized.get("adjusted_from")
    adjusted_to = normalized.get("adjusted_to")
    if adjusted_from:
        links.setdefault("adjusted_from", adjusted_from)
    if adjusted_to:
        links.setdefault("adjusted_to", adjusted_to)

    if task_type in {"training", "export"}:
        links.setdefault("execution_task_id", normalized.get("task_id"))

    normalized["submission"] = submission
    normalized["links"] = links
    normalized["registry_status"] = normalized.get(
        "registry_status",
        normalized.get("status", "submitted"),
    )
    normalized["attempt_memory"] = list(normalized.get("attempt_memory") or [])
    normalized["latest_attempt"] = normalized.get("latest_attempt")
    normalized["event_graph"] = normalize_event_graph(normalized.get("event_graph"))
    normalized["output_path"] = normalized.get("output_path")
    normalized["output_offset"] = int(normalized.get("output_offset") or 0)
    normalized["output_summary"] = normalized.get("output_summary")
    normalized["output_capped"] = bool(normalized.get("output_capped", False))
    return normalized


def build_task_record(
    *,
    task_id: str,
    task_type: str,
    user_id: str,
    submission: dict,
    registry_status: str = "submitted",
    links: dict | None = None,
) -> dict:
    """Build a business-side task registry record."""
    record_links = dict(links or {})
    if task_type in {"training", "export"}:
        record_links.setdefault("execution_task_id", task_id)

    return normalize_task_record(
        {
            "task_id": task_id,
            "task_type": task_type,
            "user_id": user_id,
            "status": registry_status,
            "registry_status": registry_status,
            "schema_version": CURRENT_SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(),
            "submission": submission,
            "links": record_links,
            "attempt_memory": [],
            "latest_attempt": None,
            "event_graph": {},
            "output_path": None,
            "output_offset": 0,
            "output_summary": None,
            "output_capped": False,
        }
    )


def append_task_attempt(task_data: dict, attempt_record: dict, max_entries: int = 20) -> dict:
    """Append a typed attempt record to a task record."""
    normalized = normalize_task_record(task_data) or {}
    history = list(normalized.get("attempt_memory") or [])
    history.append(attempt_record)
    normalized["attempt_memory"] = history[-max_entries:]
    normalized["latest_attempt"] = attempt_record
    return normalized


def append_task_event(
    task_data: dict,
    *,
    source: str,
    target: str,
    relation: str,
    node_type: str | None = None,
    label: str | None = None,
    metadata: dict | None = None,
    target_metadata: dict | None = None,
) -> dict:
    """Append a bounded graph event to a task record."""
    normalized = normalize_task_record(task_data) or {}
    normalized["event_graph"] = append_graph_event(
        normalized.get("event_graph"),
        source=source,
        target=target,
        relation=relation,
        node_type=node_type,
        label=label,
        metadata=metadata,
        target_metadata=target_metadata,
    )
    return normalized


def store_task_in_redis(redis_client, task_data: dict) -> None:
    """Store task in Redis with user_id index."""
    if redis_client is None:
        return

    task_data = normalize_task_record(task_data)
    task_id = task_data["task_id"]
    user_id = task_data["user_id"]

    redis_client.set(
        f"task:{task_id}",
        json.dumps(task_data),
        ex=TASK_TTL_SECONDS,
    )
    redis_client.sadd(f"user:{user_id}:tasks", task_id)


def get_task_from_redis(redis_client, task_id: str) -> dict | None:
    """Get task from Redis."""
    if redis_client is None:
        return None

    data = redis_client.get(f"task:{task_id}")
    if data:
        return normalize_task_record(json.loads(data))
    return None


def get_user_tasks_from_redis(redis_client, user_id: str) -> list[dict]:
    """Get all tasks for a user from Redis."""
    if redis_client is None:
        return []

    task_ids = redis_client.smembers(f"user:{user_id}:tasks")
    tasks = []
    for task_id in task_ids:
        task_data = get_task_from_redis(redis_client, task_id)
        if task_data:
            tasks.append(task_data)

    tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return tasks


def verify_task_ownership(redis_client, task_id: str, user_id: str) -> dict | None:
    """Verify that a task belongs to the user."""
    task = get_task_from_redis(redis_client, task_id)
    if task is None or task.get("user_id") != user_id:
        return None
    return task


def delete_task_from_redis(redis_client, task_id: str, user_id: str) -> bool:
    """Delete a task from Redis if owned by user."""
    task = verify_task_ownership(redis_client, task_id, user_id)
    if task is None:
        return False

    redis_client.delete(f"task:{task_id}")
    redis_client.srem(f"user:{user_id}:tasks", task_id)
    return True


def build_result_summary(task: dict) -> dict | None:
    """Build a stable summary for synchronous business-owned task results."""
    task = normalize_task_record(task)
    if task.get("task_type") not in {"analysis", "report"}:
        return None

    result = task.get("result") or {}
    return {
        "status": task.get("registry_status"),
        "content_preview": (result.get("content") or "")[:120] or None,
        "file_count": len(result.get("files") or []),
    }


async def attach_execution_snapshot(task: dict, training_client) -> dict:
    """Attach execution status from Training API for execution-backed tasks."""
    task = normalize_task_record(task)
    execution_task_id = task.get("links", {}).get("execution_task_id")
    task["execution_summary"] = {
        "status": task.get("registry_status", task.get("status")),
        "progress": None,
        "updated_at": task.get("updated_at") or task.get("created_at"),
    }
    if task.get("task_type") not in {"training", "export"} or not execution_task_id:
        task["status"] = task.get("registry_status", task.get("status"))
        return task

    try:
        execution = await training_client.get_task_status(execution_task_id)
        task["execution"] = execution
        task["status"] = execution.get("status", task.get("registry_status"))
        task["execution_summary"] = {
            "status": execution.get("status", task.get("registry_status")),
            "progress": execution.get("progress"),
            "updated_at": execution.get("completed_at")
            or execution.get("started_at")
            or task.get("updated_at")
            or task.get("created_at"),
            "error": execution.get("error"),
        }
    except Exception:
        task["status"] = task.get("registry_status", task.get("status"))

    return task


async def get_aggregated_task(
    redis_client,
    training_client,
    task_id: str,
    user_id: str,
) -> dict | None:
    """Load a task registry record and enrich it with execution state."""
    task = verify_task_ownership(redis_client, task_id, user_id)
    if task is None:
        return None
    task = await attach_execution_snapshot(task, training_client)
    task["result_summary"] = build_result_summary(task)
    return task


def build_training_status_response(task: dict) -> TrainStatusResponse:
    """Map an aggregated training task into the public status response shape."""
    task = normalize_task_record(task)
    result = dict(task.get("execution") or {})
    status_value = task.get("status", task.get("registry_status", "unknown"))

    return TrainStatusResponse(
        task_id=task.get("task_id", ""),
        status=status_value,
        progress=result.get("progress", 0.0),
        current_epoch=result.get("current_epoch"),
        total_epochs=result.get("total_epochs"),
        metrics=result.get("metrics"),
        error=result.get("error", task.get("error")),
        live_mAP50=result.get("live_mAP50"),
        lr_decay_triggered=result.get("lr_decay_triggered"),
        lr_decay_signal=result.get("lr_decay_signal"),
        augment_boost_active=result.get("augment_boost_active"),
        augment_boost_signal=result.get("augment_boost_signal"),
        data_expansion_requested=result.get("data_expansion_requested"),
        data_expansion_signal=result.get("data_expansion_signal"),
        strategies_triggered=result.get("strategies_triggered"),
        resubmit_count=result.get("resubmit_count"),
        last_resubmitted_at=result.get("last_resubmitted_at"),
        resubmit_reason=result.get("resubmit_reason"),
        output_path=task.get("output_path"),
        output_offset=task.get("output_offset"),
        output_summary=task.get("output_summary"),
        output_capped=task.get("output_capped"),
    )


def build_export_status_response(task: dict) -> ExportStatusResponse:
    """Map an aggregated export task into the public status response shape."""
    task = normalize_task_record(task)
    result = dict(task.get("execution") or {})
    submission = dict(task.get("submission") or {})
    status_value = task.get("status", task.get("registry_status", "unknown"))

    return ExportStatusResponse(
        task_id=task.get("task_id", ""),
        status=status_value,
        progress=result.get("progress", 0.0),
        model_path=result.get("model_path", submission.get("model_path")),
        platform=result.get("platform", submission.get("platform")),
        imgsz=result.get("imgsz", submission.get("imgsz")),
        formats=result.get("formats"),
        int8_quantize=result.get("int8_quantize"),
        export_path=result.get("export_path"),
        error=result.get("error", task.get("error")),
        started_at=result.get("started_at"),
        completed_at=result.get("completed_at"),
        output_path=task.get("output_path"),
        output_offset=task.get("output_offset"),
        output_summary=task.get("output_summary"),
        output_capped=task.get("output_capped"),
    )
