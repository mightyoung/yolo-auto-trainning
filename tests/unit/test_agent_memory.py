import importlib
import json
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent.parent
biz_api_src = project_root / "business-api" / "src"
if str(biz_api_src) not in sys.path:
    sys.path.insert(0, str(biz_api_src))

worker_memory = importlib.import_module("agents.worker_memory")
task_registry = importlib.import_module("api.task_registry")


class FakeRedisHash:
    def __init__(self):
        self.hashes = {}

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def hset(self, key, mapping):
        bucket = self.hashes.setdefault(key, {})
        bucket.update(mapping)


def test_sanitize_training_status_drops_unbounded_fields():
    status = {
        "status": "running",
        "progress": 0.4,
        "live_mAP50": 0.62,
        "curriculum_stage": "stage2",
        "strategies_triggered": [{"level": 1, "action": "lr_decay", "verbose": "x" * 500}],
        "unsafe_blob": {"huge": "payload"},
        "log_text": "x" * 500,
    }

    summary = worker_memory.sanitize_training_status(status)

    assert summary["status"] == "running"
    assert summary["progress"] == 0.4
    assert summary["curriculum_stage"] == "stage2"
    assert "unsafe_blob" not in summary
    assert "log_text" not in summary
    assert summary["strategies_triggered"][0]["action"] == "lr_decay"
    assert "verbose" in summary["strategies_triggered"][0]


def test_append_agent_attempt_caps_history_and_updates_latest():
    redis_client = FakeRedisHash()
    task_id = "agent_123"

    for idx in range(25):
        record = worker_memory.build_attempt_record(
            attempt_type="auto_adjust",
            stage="lr_decay",
            outcome="completed",
            source="test",
            details={"idx": idx},
        )
        worker_memory.append_agent_attempt(redis_client, task_id, record)

    stored = redis_client.hashes[f"agent:{task_id}"]
    history = json.loads(stored["attempt_history"])
    latest = json.loads(stored["latest_attempt"])

    assert len(history) == worker_memory.MAX_ATTEMPT_HISTORY
    assert history[0]["details"]["idx"] == 5
    assert latest["details"]["idx"] == 24


def test_extract_agent_submission_prefers_submission_over_legacy_params():
    task_data = {
        "submission": json.dumps({"model": "yolo11x", "batch": 8}),
        "params": json.dumps({"model": "yolo11n", "batch": 16}),
    }

    submission = worker_memory.extract_agent_submission(task_data)

    assert submission["model"] == "yolo11x"
    assert submission["batch"] == 8


def test_task_registry_migrates_and_appends_attempt_memory():
    legacy = {
        "task_id": "train_123",
        "task_type": "training",
        "user_id": "user1",
        "status": "submitted",
        "params": {"model": "yolo11m"},
    }

    normalized = task_registry.normalize_task_record(legacy)
    assert normalized["schema_version"] == task_registry.CURRENT_SCHEMA_VERSION
    assert normalized["attempt_memory"] == []
    assert normalized["latest_attempt"] is None

    updated = task_registry.append_task_attempt(
        normalized,
        {"attempt_type": "training_completion", "outcome": "failed"},
        max_entries=2,
    )
    updated = task_registry.append_task_attempt(
        updated,
        {"attempt_type": "training_completion", "outcome": "completed"},
        max_entries=2,
    )

    assert len(updated["attempt_memory"]) == 2
    assert updated["latest_attempt"]["outcome"] == "completed"
