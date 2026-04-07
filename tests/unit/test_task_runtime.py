import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def business_modules():
    project_root = Path(__file__).parent.parent.parent
    business_api_src = project_root / "business-api" / "src"

    original_sys_path = sys.path.copy()
    for p in list(sys.path):
        if "business-api" in p:
            sys.path.remove(p)
    sys.path.insert(0, str(business_api_src))

    modules_to_remove = [k for k in list(sys.modules.keys()) if k == "api" or k.startswith("api.") or k == "agents" or k.startswith("agents.")]
    for mod in modules_to_remove:
        del sys.modules[mod]

    task_registry = importlib.import_module("api.task_registry")
    yield task_registry

    sys.path[:] = original_sys_path


@pytest.fixture()
def training_modules():
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
    yield task_store

    sys.path[:] = original_sys_path


def test_business_task_registry_includes_output_fields_and_terminal_helper(business_modules):
    task_registry = business_modules
    record = task_registry.build_task_record(
        task_id="train_1",
        task_type="training",
        user_id="user-1",
        submission={"model": "yolo11m"},
    )

    assert record["output_path"] is None
    assert record["output_offset"] == 0
    assert record["output_summary"] is None
    assert task_registry.is_terminal_task_status("completed") is True
    assert task_registry.is_terminal_task_status("running") is False


def test_training_task_store_includes_output_fields_and_terminal_helper(training_modules):
    task_store = training_modules
    fake_redis = type(
        "FakeRedis",
        (),
        {
            "__init__": lambda self: setattr(self, "store", {}),
            "get": lambda self, key: self.store.get(key),
            "set": lambda self, key, value: self.store.__setitem__(key, value),
            "delete": lambda self, key: self.store.pop(key, None),
        },
    )()
    task_store._redis_client = fake_redis
    task_store._tasks_cache.clear()

    task_store._task_set("task-1", {"task_id": "task-1", "status": "submitted"})
    stored = json.loads(fake_redis.get("training:task:task-1"))

    assert stored["output_path"] is None
    assert stored["output_offset"] == 0
    assert stored["output_summary"] is None
    assert task_store.is_terminal_task_status("completed") is True
    assert task_store.is_terminal_task_status("retrying") is False
