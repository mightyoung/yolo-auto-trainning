"""Task storage for Training API.

Redis-backed task storage with L1 in-memory cache.
On reads: check local dict first, then Redis, populate cache on miss.
On writes: write-through to both local dict and Redis.
Key pattern in Redis: training:task:{task_id}
"""

import json
import threading
from typing import Optional, Dict

from ..gateway import get_redis_client

# Module-level cache and locks
_redis_client = get_redis_client()
_tasks_cache: dict = {}
_tasks_lock = threading.Lock()


def _task_get(task_id: str) -> Optional[dict]:
    """Read a task. L1 dict cache, then Redis."""
    with _tasks_lock:
        if task_id in _tasks_cache:
            return _tasks_cache[task_id]
    if _redis_client is None:
        return None
    try:
        key = f"training:task:{task_id}"
        raw = _redis_client.get(key)
        if raw:
            task = json.loads(raw)
            with _tasks_lock:
                _tasks_cache[task_id] = task
            return task
    except Exception:
        pass
    return None


def _task_set(task_id: str, task: dict) -> None:
    """Write a task. Write-through to local cache and Redis."""
    with _tasks_lock:
        _tasks_cache[task_id] = task
    if _redis_client is None:
        return
    try:
        key = f"training:task:{task_id}"
        _redis_client.set(key, json.dumps(task))
    except Exception as e:
        print(f"[_task_set] Redis write failed for {task_id}: {e}")


def _task_del(task_id: str) -> None:
    """Delete a task from local cache and Redis."""
    with _tasks_lock:
        _tasks_cache.pop(task_id, None)
    if _redis_client is None:
        return
    try:
        _redis_client.delete(f"training:task:{task_id}")
    except Exception as e:
        print(f"[_task_del] Redis delete failed for {task_id}: {e}")


# Cancellation registry: task_id -> threading.Event
# Stored separately from task records so Event objects aren't JSON-serialised.
_cancel_events: Dict[str, threading.Event] = {}
_cancel_lock = threading.Lock()


def get_cancel_event(task_id: str) -> Optional[threading.Event]:
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
