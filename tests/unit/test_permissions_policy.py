import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def policy_module():
    project_root = Path(__file__).parent.parent.parent
    business_api_src = project_root / "business-api" / "src"

    original_sys_path = sys.path.copy()
    for p in list(sys.path):
        if "business-api" in p:
            sys.path.remove(p)
    sys.path.insert(0, str(business_api_src))

    modules_to_remove = [k for k in list(sys.modules.keys()) if k == "agents" or k.startswith("agents.")]
    for mod in modules_to_remove:
        del sys.modules[mod]

    module = importlib.import_module("agents.operation_policy")
    yield module

    sys.path[:] = original_sys_path


def test_operation_policy_denies_missing_ssh_credentials(policy_module, monkeypatch):
    monkeypatch.delenv("GPU_SERVER_HOST", raising=False)
    monkeypatch.delenv("GPU_SERVER_USER", raising=False)
    monkeypatch.delenv("GPU_SERVER_PASS", raising=False)

    decision = policy_module.evaluate_operation("ssh_dataset_download", context={"source": "roboflow"})

    assert decision.behavior == policy_module.DENY
    assert "missing SSH credentials" in decision.reason


def test_operation_policy_allows_training_submit_when_env_is_present(policy_module, monkeypatch):
    monkeypatch.setenv("TRAINING_API_URL", "http://localhost:8001")
    monkeypatch.setenv("TRAINING_API_KEY", "test-key")

    decision = policy_module.evaluate_operation("gpu_training_submit", context={"task_id": "task-1"})

    assert decision.behavior == policy_module.ALLOW


def test_operation_policy_can_ask_for_confirmation(policy_module, monkeypatch):
    monkeypatch.setenv("OPERATION_POLICY_MODE", "ask")
    monkeypatch.setenv("TRAINING_API_URL", "http://localhost:8001")
    monkeypatch.setenv("TRAINING_API_KEY", "test-key")

    decision = policy_module.evaluate_operation("model_export", context={"platform": "jetson_orin"})

    assert decision.behavior == policy_module.ASK
