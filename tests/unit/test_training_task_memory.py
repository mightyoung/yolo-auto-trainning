import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def training_api_modules():
    project_root = Path(__file__).parent.parent.parent
    training_api_src = project_root / "training-api" / "src"

    original_sys_path = sys.path.copy()
    for p in list(sys.path):
        if "training-api" in p:
            sys.path.remove(p)
    sys.path.insert(0, str(training_api_src))

    modules_to_remove = [k for k in list(sys.modules.keys()) if k == "api" or k.startswith("api.")]
    for mod in modules_to_remove:
        del sys.modules[mod]

    task_store = importlib.import_module("api.store.task_store")
    shared = importlib.import_module("api.services._shared")
    original_redis_client = task_store._redis_client

    yield task_store, shared

    task_store._redis_client = original_redis_client
    task_store._tasks_cache.clear()
    sys.path[:] = original_sys_path


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value

    def delete(self, key):
        self.store.pop(key, None)


def test_training_task_store_normalizes_attempt_memory(training_api_modules):
    task_store, _ = training_api_modules
    fake_redis = FakeRedis()
    task_store._redis_client = fake_redis
    task_store._tasks_cache.clear()

    task_store._task_set("task-1", {"task_id": "task-1", "status": "submitted"})

    stored = json.loads(fake_redis.get("training:task:task-1"))
    assert stored["attempt_memory"] == []
    assert stored["latest_attempt"] is None

    updated = task_store.append_task_attempt(
        "task-1",
        task_store.build_attempt_record(
            attempt_type="training",
            stage="train",
            outcome="completed",
            source="training_runner",
            action="complete",
            training_task_id="task-1",
            details={"metrics": {"mAP50": 0.91}},
        ),
    )

    assert updated["latest_attempt"]["outcome"] == "completed"
    assert updated["attempt_memory"][-1]["training_task_id"] == "task-1"


def test_training_task_store_caps_attempt_history(training_api_modules):
    task_store, _ = training_api_modules
    fake_redis = FakeRedis()
    task_store._redis_client = fake_redis
    task_store._tasks_cache.clear()

    task_store._task_set("task-2", {"task_id": "task-2", "status": "submitted"})

    for idx in range(25):
        task_store.append_task_attempt(
            "task-2",
            task_store.build_attempt_record(
                attempt_type="training_retry",
                stage="train",
                outcome="retrying",
                source="training_runner",
                action="retry_after_failure",
                error=f"err-{idx}",
                training_task_id="task-2",
            ),
        )

    updated = task_store._task_get("task-2")
    assert len(updated["attempt_memory"]) == task_store.MAX_ATTEMPT_HISTORY
    assert updated["latest_attempt"]["error"] == "err-24"


def test_shared_training_attempt_helper_builds_structured_records(training_api_modules):
    _, shared = training_api_modules
    captured = {}

    def fake_append(task_id, record, max_entries=20):
        captured["task_id"] = task_id
        captured["record"] = record
        captured["max_entries"] = max_entries

    original_append = shared.append_task_attempt
    shared.append_task_attempt = fake_append
    try:
        shared._record_training_attempt(
            "task-3",
            attempt_type="training",
            stage="train",
            outcome="failed",
            source="training_runner",
            action="finish",
            error="OOM",
            details={"attempt": 2},
        )
    finally:
        shared.append_task_attempt = original_append

    assert captured["task_id"] == "task-3"
    assert captured["record"]["attempt_type"] == "training"
    assert captured["record"]["outcome"] == "failed"
    assert captured["record"]["error"] == "OOM"
    assert captured["record"]["details"]["attempt"] == 2
