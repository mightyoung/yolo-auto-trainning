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
    event_graph = importlib.import_module("agents.event_graph")

    yield task_registry, event_graph

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


def test_business_task_registry_appends_bounded_event_graph(business_modules):
    task_registry, _ = business_modules

    record = task_registry.build_task_record(
        task_id="task-1",
        task_type="training",
        user_id="user-1",
        submission={"model": "yolo11m"},
    )
    updated = task_registry.append_task_event(
        record,
        source="task-1",
        target="train-1",
        relation="training_started",
        node_type="task",
        label="running",
        metadata={"status": "running"},
    )

    assert updated["event_graph"]["edges"][-1]["type"] == "training_started"
    assert updated["event_graph"]["latest_edge"]["target"] == "train-1"
    assert updated["event_graph"]["latest_node"]["id"] == "train-1"


def test_business_event_graph_is_bounded(business_modules):
    task_registry, _ = business_modules

    record = task_registry.build_task_record(
        task_id="task-2",
        task_type="training",
        user_id="user-1",
        submission={"model": "yolo11m"},
    )
    for idx in range(30):
        record = task_registry.append_task_event(
            record,
            source=f"src-{idx}",
            target=f"dst-{idx}",
            relation="step",
            metadata={"idx": idx},
        )

    assert len(record["event_graph"]["edges"]) == 20
    assert len(record["event_graph"]["nodes"]) <= 20
    assert record["event_graph"]["latest_edge"]["target"] == "dst-29"


def test_training_task_store_appends_event_graph(training_modules):
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

    task_store._task_set("task-3", {"task_id": "task-3", "status": "submitted"})
    updated = task_store.append_task_event(
        "task-3",
        source="task-3",
        target="task-3:running",
        relation="training_started",
        node_type="training",
        label="running",
        metadata={"status": "running"},
    )

    persisted = json.loads(fake_redis.get("training:task:task-3"))
    assert updated["event_graph"]["latest_edge"]["type"] == "training_started"
    assert persisted["event_graph"]["latest_edge"]["target"] == "task-3:running"


def test_business_task_registry_materializes_strategy_ledger(business_modules):
    task_registry, _ = business_modules

    record = task_registry.build_task_record(
        task_id="task-ledger",
        task_type="training",
        user_id="user-1",
        submission={"model": "yolo11m", "epochs": 100},
    )
    record = task_registry.append_task_event(
        record,
        source="train-1",
        target="task-ledger",
        relation="strategy_change_proposed",
        node_type="strategy",
        label="lr_decay",
        metadata={
            "proposal_id": "proposal-1",
            "parent_run_id": "task-ledger",
            "child_training_task_id": "train-1",
            "trigger_signal": {"lr_decay_count": 1},
            "rationale": "selected=lr_decay",
            "change_set": {"action": "lr_decay"},
            "decision": "lr_decay",
            "timestamp": "2026-04-14T09:45:00",
        },
    )
    record = task_registry.append_task_event(
        record,
        source="train-1",
        target="task-ledger",
        relation="strategy_change_committed",
        node_type="strategy",
        label="lr_decay",
        metadata={
            "proposal_id": "proposal-1",
            "commit_id": "proposal-1",
            "parent_run_id": "task-ledger",
            "child_training_task_id": "train-next",
            "sequence": 1,
            "trigger_signal": {"lr_decay_count": 1},
            "rationale": "selected=lr_decay",
            "change_set": {"action": "lr_decay"},
            "decision": "lr_decay",
            "timestamp": "2026-04-14T09:46:00",
        },
    )
    record = task_registry.append_task_event(
        record,
        source="train-next",
        target="task-ledger",
        relation="strategy_stop",
        node_type="strategy",
        label="budget_exhausted",
        metadata={
            "proposal_id": "proposal-1",
            "commit_id": "proposal-1",
            "parent_run_id": "task-ledger",
            "child_training_task_id": "train-next",
            "sequence": 2,
            "trigger_signal": {},
            "rationale": "budget reached",
            "change_set": {"action": "stop"},
            "decision": "stopped",
            "stop_reason": "budget_exhausted",
            "timestamp": "2026-04-14T09:47:00",
        },
    )

    assert len(record["strategy_ledger"]) == 2
    assert record["strategy_ledger"][0]["relation"] == "strategy_change_committed"
    assert record["strategy_ledger"][0]["sequence"] == 1
    assert record["strategy_ledger"][1]["relation"] == "strategy_stop"
    assert record["strategy_ledger"][1]["stop_reason"] == "budget_exhausted"
