import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def business_output_module():
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

    module = importlib.import_module("agents.task_output")
    yield module

    sys.path[:] = original_sys_path


@pytest.fixture()
def training_output_module():
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

    module = importlib.import_module("api.store.task_output")
    yield module

    sys.path[:] = original_sys_path


def _exercise_spool(module, monkeypatch):
    module._TASK_OUTPUTS.clear()
    spool = module.get_task_output_spool("task-1")
    monkeypatch.setattr(module, "MAX_TASK_OUTPUT_BYTES", 64)
    snapshot = spool.append("hello world", summary="hello summary")
    assert snapshot["output_path"].endswith("task-1.log")
    assert snapshot["output_offset"] > 0
    assert snapshot["output_summary"] == "hello summary"

    capped = None
    for _ in range(10):
        capped = spool.append("x" * 32, summary="overflow summary")
    assert capped["output_capped"] is True
    assert capped["output_summary"] in {"overflow summary", "x" * 32}


def test_business_task_output_spool_is_bounded(business_output_module, monkeypatch):
    _exercise_spool(business_output_module, monkeypatch)


def test_training_task_output_spool_is_bounded(training_output_module, monkeypatch):
    _exercise_spool(training_output_module, monkeypatch)
