"""Typed worker-memory helpers for agent orchestration."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .coordinator_summary import build_compact_summary


MAX_ATTEMPT_HISTORY = 20
SUMMARY_TEXT_LIMIT = 240

_SUMMARY_KEYS = {
    "status",
    "progress",
    "current_epoch",
    "total_epochs",
    "error",
    "live_mAP50",
    "lr_decay_triggered",
    "lr_decay_signal",
    "augment_boost_active",
    "augment_boost_signal",
    "data_expansion_requested",
    "data_expansion_signal",
    "strategies_triggered",
    "curriculum_stage",
    "curriculum_stage_mAP",
    "curriculum_stage_history",
    "resubmit_count",
    "resubmit_reason",
}


def _truncate_text(value: Any, limit: int = SUMMARY_TEXT_LIMIT) -> str | None:
    """Return a bounded string representation."""
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def sanitize_training_status(status_data: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only bounded, schema-safe fields from worker/training status."""
    if not status_data:
        return {}

    summary: dict[str, Any] = {}
    for key in _SUMMARY_KEYS:
        if key not in status_data:
            continue
        value = status_data[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value
        elif isinstance(value, dict):
            summary[key] = {
                nested_key: nested_value
                for nested_key, nested_value in value.items()
                if isinstance(nested_value, (str, int, float, bool)) or nested_value is None
            }
        elif isinstance(value, list):
            bounded_items = []
            for item in value[:10]:
                if isinstance(item, dict):
                    bounded_items.append({
                        nested_key: nested_value
                        for nested_key, nested_value in item.items()
                        if isinstance(nested_value, (str, int, float, bool)) or nested_value is None
                    })
                elif isinstance(item, (str, int, float, bool)) or item is None:
                    bounded_items.append(item)
            summary[key] = bounded_items

    return summary


def build_attempt_record(
    *,
    attempt_type: str,
    stage: str,
    outcome: str,
    source: str,
    action: str | None = None,
    error: str | None = None,
    training_task_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a typed attempt/reflection record."""
    summary = build_compact_summary(
        kind=attempt_type,
        stage=stage,
        outcome=outcome,
        action=action,
        detail=error,
    )
    return {
        "timestamp": datetime.now().isoformat(),
        "attempt_type": attempt_type,
        "stage": stage,
        "outcome": outcome,
        "source": source,
        "action": action,
        "summary": summary,
        "error": _truncate_text(error),
        "training_task_id": training_task_id,
        "details": details or {},
    }


def append_attempt_history(
    redis_client,
    redis_key: str,
    record: dict[str, Any],
    *,
    field: str = "attempt_history",
    latest_field: str = "latest_attempt",
    max_entries: int = MAX_ATTEMPT_HISTORY,
) -> list[dict[str, Any]]:
    """Append a typed attempt record into a Redis hash field."""
    if redis_client is None:
        return [record]

    raw = redis_client.hget(redis_key, field)
    history = []
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                history = parsed
        except json.JSONDecodeError:
            history = []

    history.append(record)
    history = history[-max_entries:]
    redis_client.hset(
        redis_key,
        mapping={
            field: json.dumps(history),
            latest_field: json.dumps(record),
        },
    )
    return history


def append_agent_attempt(redis_client, task_id: str, record: dict[str, Any]) -> list[dict[str, Any]]:
    """Append attempt history to the main agent hash."""
    return append_attempt_history(redis_client, f"agent:{task_id}", record)


def append_autoadjust_attempt(redis_client, task_id: str, record: dict[str, Any]) -> list[dict[str, Any]]:
    """Append attempt history to the auto-adjust hash."""
    return append_attempt_history(
        redis_client,
        f"autoadjust:{task_id}",
        record,
        field="attempt_history",
        latest_field="latest_attempt",
    )


def extract_agent_submission(task_data: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize training submission params from mixed legacy/new agent state."""
    if not task_data:
        return {}

    for key in ("submission", "params"):
        raw = task_data.get(key)
        if not raw:
            continue
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return {}
