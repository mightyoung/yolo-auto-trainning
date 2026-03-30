"""Task registry storage and aggregation helpers for the Business API."""

from __future__ import annotations

from datetime import datetime
import json
from typing import Optional

from .task_models import ExportStatusResponse, TrainStatusResponse


TASK_TTL_SECONDS = 7 * 24 * 60 * 60

# Schema version for Redis task records
# Increment this when the schema changes and add migration logic below
CURRENT_SCHEMA_VERSION = "v1"


def migrate_task_record(task_data: dict) -> dict:
    """Migrate a task record to the current schema version.

    This function handles migrations from older schema versions to newer ones.
    Add new migration cases as the schema evolves.
    """
    schema_version = task_data.get("schema_version", "legacy")

    if schema_version == CURRENT_SCHEMA_VERSION:
        return task_data

    # Migration: legacy -> v1
    if schema_version == "legacy":
        # Legacy records had "params" instead of "submission"
        if "params" in task_data and "submission" not in task_data:
            task_data["submission"] = task_data.pop("params")
        # Legacy records didn't have schema_version
        task_data["schema_version"] = CURRENT_SCHEMA_VERSION
        return task_data

    # If we don't know the schema version, try to normalize what we can
    if schema_version not in (CURRENT_SCHEMA_VERSION, "legacy"):
        # Normalize fields but don't recursively call migrate
        if "params" in task_data and "submission" not in task_data:
            task_data["submission"] = task_data.pop("params")
        task_data["schema_version"] = CURRENT_SCHEMA_VERSION

    return task_data


def normalize_task_record(task_data: Optional[dict]) -> Optional[dict]:
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
    return normalized


def build_task_record(
    *,
    task_id: str,
    task_type: str,
    user_id: str,
    submission: dict,
    registry_status: str = "submitted",
    links: Optional[dict] = None,
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
        }
    )


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


def get_task_from_redis(redis_client, task_id: str) -> Optional[dict]:
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


def verify_task_ownership(redis_client, task_id: str, user_id: str) -> Optional[dict]:
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


def build_result_summary(task: dict) -> Optional[dict]:
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
) -> Optional[dict]:
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
    )
