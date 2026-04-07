"""Task storage for Training API.

Redis-backed task storage with L1 in-memory cache.
On reads: check local dict first, then Redis, populate cache on miss.
On writes: write-through to both local dict and Redis.
Key pattern in Redis: training:task:{task_id}
"""

import json
import threading
from datetime import datetime

from .event_graph import append_graph_event, normalize_event_graph

# Module-level cache and locks
_redis_client = None
_tasks_cache: dict = {}
_tasks_lock = threading.Lock()
MAX_ATTEMPT_HISTORY = 20


def _resolve_redis_client():
    """Resolve the Redis client lazily to avoid import-time circular imports."""
    global _redis_client
    if _redis_client is None:
        try:
            from ..gateway import get_redis_client

            _redis_client = get_redis_client()
        except Exception:
            _redis_client = None
    return _redis_client


def normalize_task_record(task: dict | None) -> dict | None:
    """Normalize a task record so attempt-memory fields are always present."""
    if task is None:
        return None
    normalized = dict(task)
    normalized["attempt_memory"] = list(normalized.get("attempt_memory") or [])
    normalized["latest_attempt"] = normalized.get("latest_attempt")
    normalized["event_graph"] = normalize_event_graph(normalized.get("event_graph"))
    return normalized


def build_attempt_record(
    *,
    attempt_type: str,
    stage: str,
    outcome: str,
    source: str,
    action: str,
    error: str | None = None,
    training_task_id: str | None = None,
    details: dict | None = None,
) -> dict:
    """Build a typed attempt record for bounded retry/failure history."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "attempt_type": attempt_type,
        "stage": stage,
        "outcome": outcome,
        "source": source,
        "action": action,
    }
    if error is not None:
        record["error"] = error
    if training_task_id is not None:
        record["training_task_id"] = training_task_id
    if details is not None:
        record["details"] = details
    return record


def append_task_attempt(task_id: str, attempt_record: dict, max_entries: int = MAX_ATTEMPT_HISTORY) -> dict | None:
    """Append a bounded attempt record to a task and persist the update."""
    task = _task_get(task_id)
    if task is None:
        return None
    history = list(task.get("attempt_memory") or [])
    history.append(attempt_record)
    task["attempt_memory"] = history[-max_entries:]
    task["latest_attempt"] = attempt_record
    _task_set(task_id, task)
    return task


def append_task_event(
    task_id: str,
    *,
    source: str,
    target: str,
    relation: str,
    node_type: str | None = None,
    label: str | None = None,
    metadata: dict | None = None,
    target_metadata: dict | None = None,
) -> dict | None:
    """Append a bounded graph event to a task and persist the update."""
    task = _task_get(task_id)
    if task is None:
        return None
    task["event_graph"] = append_graph_event(
        task.get("event_graph"),
        source=source,
        target=target,
        relation=relation,
        node_type=node_type,
        label=label,
        metadata=metadata,
        target_metadata=target_metadata,
    )
    _task_set(task_id, task)
    return task


def _task_get(task_id: str) -> dict | None:
    """Read a task. L1 dict cache, then Redis."""
    with _tasks_lock:
        if task_id in _tasks_cache:
            cached = normalize_task_record(_tasks_cache[task_id])
            if cached is not None:
                _tasks_cache[task_id] = cached
            return cached
    redis_client = _resolve_redis_client()
    if redis_client is None:
        return None
    try:
        key = f"training:task:{task_id}"
        raw = redis_client.get(key)
        if raw:
            task = normalize_task_record(json.loads(raw))
            with _tasks_lock:
                _tasks_cache[task_id] = task
            return task
    except Exception:
        pass
    return None


def _task_set(task_id: str, task: dict) -> None:
    """Write a task. Write-through to local cache and Redis."""
    task = normalize_task_record(task) or {}
    with _tasks_lock:
        _tasks_cache[task_id] = task
    redis_client = _resolve_redis_client()
    if redis_client is None:
        return
    try:
        key = f"training:task:{task_id}"
        redis_client.set(key, json.dumps(task))
    except Exception as e:
        print(f"[_task_set] Redis write failed for {task_id}: {e}")


def _task_del(task_id: str) -> None:
    """Delete a task from local cache and Redis."""
    with _tasks_lock:
        _tasks_cache.pop(task_id, None)
    redis_client = _resolve_redis_client()
    if redis_client is None:
        return
    try:
        redis_client.delete(f"training:task:{task_id}")
    except Exception as e:
        print(f"[_task_del] Redis delete failed for {task_id}: {e}")


# Cancellation registry: task_id -> threading.Event
# Stored separately from task records so Event objects aren't JSON-serialised.
_cancel_events: dict[str, threading.Event] = {}
_cancel_lock = threading.Lock()


def get_cancel_event(task_id: str) -> threading.Event | None:
    """Get cancellation event for a task."""
    with _cancel_lock:
        return _cancel_events.get(task_id)


def set_cancel_event(task_id: str, event: threading.Event) -> None:
    """Set cancellation event for a task."""
    with _cancel_lock:
        _cancel_events[task_id] = event


def clear_cancel_event(task_id: str) -> None:
    """Remove cancellation event for a task."""
    with _cancel_lock:
        _cancel_events.pop(task_id, None)


def get_tasks_cache() -> dict:
    """Get the tasks cache for reading."""
    return _tasks_cache


def get_tasks_lock() -> threading.Lock:
    """Get the tasks lock for external synchronization."""
    return _tasks_lock
