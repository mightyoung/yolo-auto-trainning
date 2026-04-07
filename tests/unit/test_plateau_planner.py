import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def plateau_planner():
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

    module = importlib.import_module("agents.plateau_planner")

    yield module

    sys.path[:] = original_sys_path


def test_build_plateau_decision_prefers_data_expansion_when_gap_is_large(plateau_planner):
    status_summary = {
        "live_mAP50": 0.70,
        "lr_decay_triggered": True,
        "lr_decay_signal": {
            "lr_decay_count": 1,
            "current_mAP50": 0.70,
            "target_mAP50": 0.90,
            "factor": 0.5,
        },
        "data_expansion_requested": True,
        "data_expansion_signal": {
            "current_mAP50": 0.70,
            "target_mAP50": 0.90,
            "recommendation": "expand dataset",
        },
    }

    decision = plateau_planner.build_plateau_decision(status_summary, [])

    assert decision is not None
    assert decision["selected"]["action"] == "data_expansion"
    assert decision["candidate_count"] == 2
    assert len(decision["rejected"]) == 1


def test_build_plateau_decision_prefers_lr_decay_when_gap_is_small(plateau_planner):
    status_summary = {
        "live_mAP50": 0.87,
        "lr_decay_triggered": True,
        "lr_decay_signal": {
            "lr_decay_count": 1,
            "current_mAP50": 0.87,
            "target_mAP50": 0.90,
            "factor": 0.5,
        },
        "data_expansion_requested": True,
        "data_expansion_signal": {
            "current_mAP50": 0.87,
            "target_mAP50": 0.90,
        },
    }

    decision = plateau_planner.build_plateau_decision(status_summary, [])

    assert decision is not None
    assert decision["selected"]["action"] == "lr_decay"
    assert decision["candidate_count"] == 2
    assert decision["selected"]["score"] > decision["rejected"][0]["score"]


def test_build_plateau_decision_returns_none_without_actionable_signals(plateau_planner):
    status_summary = {
        "live_mAP50": 0.88,
        "augment_boost_active": True,
    }

    decision = plateau_planner.build_plateau_decision(status_summary, [])

    assert decision is None


def test_build_plateau_attempt_record_includes_selected_and_rejected(plateau_planner):
    decision = {
        "selected": {
            "action": "lr_decay",
            "stage": "lr_decay",
            "score": 0.79,
            "reason": "example",
            "details": {"decay_count": 1},
        },
        "rejected": [
            {
                "action": "data_expansion",
                "stage": "data_expansion",
                "score": 0.64,
                "reason": "example",
                "details": {},
            }
        ],
        "candidate_count": 2,
        "rationale": "selected=lr_decay score=0.79; rejected=1",
        "signal_bundle": {"live_mAP50": 0.87},
    }

    record = plateau_planner.build_plateau_attempt_record(
        task_id="task-1",
        training_task_id="train-1",
        decision=decision,
    )

    assert record["attempt_type"] == "plateau_search"
    assert record["action"] == "lr_decay"
    assert record["training_task_id"] == "train-1"
    assert record["details"]["selected"]["action"] == "lr_decay"
    assert record["details"]["rejected"][0]["action"] == "data_expansion"
