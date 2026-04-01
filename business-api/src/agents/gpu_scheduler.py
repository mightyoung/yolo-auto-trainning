"""
GPU Task Queue Scheduler for Business API.

Autonomous GPU task scheduling - when a GPU is free and tasks are queued,
automatically dispatch the next task.
"""

import threading
import time
import logging
import httpx
import json
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Redis helpers ──────────────────────────────────────────────
def _get_redis():
    import redis
    url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    pw = os.environ.get('REDIS_PASSWORD')  # No default - must be configured
    if pw:
        return redis.from_url(url, password=pw, decode_responses=True)
    return redis.from_url(url, decode_responses=True)

# ── Queue Operations ──────────────────────────────────────────
def enqueue_task(task_metadata: dict) -> int:
    """Add task to queue. Returns queue length."""
    r = _get_redis()
    return r.lpush('training:queue', json.dumps(task_metadata))

def dequeue_task() -> Optional[dict]:
    """Pop next task from queue. Returns None if empty."""
    r = _get_redis()
    raw = r.rpop('training:queue')
    if raw:
        return json.loads(raw)
    return None

def peek_queue() -> list:
    """View all queued tasks without removing."""
    r = _get_redis()
    items = r.lrange('training:queue', 0, -1)
    return [json.loads(x) for x in reversed(items)]

# ── GPU Status ────────────────────────────────────────────────
def get_free_gpu_slots() -> list:
    """Query Training API for free GPU slots."""
    training_url = os.environ.get('TRAINING_API_URL', 'http://localhost:8001')
    api_key = os.environ.get('TRAINING_API_KEY', 'default-key')
    try:
        r = httpx.get(
            f'{training_url}/api/v1/internal/gpu/status',
            headers={'X-API-Key': api_key},
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            free = [g for g in data.get('gpus', []) if g['utilization'] < 10]
            return free
    except Exception as e:
        logger.warning(f"Failed to query GPU status: {e}")
    return []

# ── Task Submission ───────────────────────────────────────────
def dispatch_task(task_metadata: dict) -> Optional[str]:
    """Submit task to Training API. Returns task_id."""
    training_url = os.environ.get('TRAINING_API_URL', 'http://localhost:8001')
    api_key = os.environ.get('TRAINING_API_KEY', 'default-key')
    try:
        r = httpx.post(
            f'{training_url}/api/v1/internal/train/curriculum',
            json={
                'data_yaml': task_metadata['data_yaml'],
                'output_dir': task_metadata.get('output_dir', '/home/wangxin/runs'),
                'device': task_metadata.get('device', 'cuda:0'),
                'epochs_per_stage': task_metadata.get('epochs_per_stage', 100),
            },
            headers={'X-API-Key': api_key, 'Content-Type': 'application/json'},
            timeout=30
        )
        if r.status_code == 200:
            return r.json().get('task_id')
    except Exception as e:
        logger.error(f"Failed to dispatch task: {e}")
    return None

# ── Scheduler Loop ─────────────────────────────────────────────
_scheduler_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()

def start_scheduler():
    """Start the GPU scheduler in a background thread."""
    global _scheduler_thread, _stop_event
    if _scheduler_thread and _scheduler_thread.is_alive():
        logger.info("Scheduler already running")
        return
    _stop_event.clear()
    _scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True, name="GPUScheduler")
    _scheduler_thread.start()
    logger.info("GPU scheduler started")

def stop_scheduler():
    """Stop the GPU scheduler."""
    global _stop_event
    _stop_event.set()
    if _scheduler_thread:
        _scheduler_thread.join(timeout=5)

def _scheduler_loop():
    """Main polling loop."""
    while not _stop_event.is_set():
        try:
            free_gpus = get_free_gpu_slots()
            queue_size = _get_redis().llen('training:queue')
            if free_gpus and queue_size > 0:
                task = dequeue_task()
                if task:
                    task_id = dispatch_task(task)
                    if task_id:
                        logger.info(f"Auto-dispatched task {task_id} to free GPU")
                    else:
                        # Re-queue on failure
                        enqueue_task(task)
                        logger.error("Dispatch failed, re-queued task")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
        _stop_event.wait(60)  # Poll every 60s
