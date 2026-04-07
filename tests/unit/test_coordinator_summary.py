import importlib
import sys
from pathlib import Path


def test_compact_summary_is_bounded():
    project_root = Path(__file__).parent.parent.parent
    business_api_src = project_root / "business-api" / "src"

    original_sys_path = sys.path.copy()
    for p in list(sys.path):
        if "business-api" in p:
            sys.path.remove(p)
    sys.path.insert(0, str(business_api_src))

    for mod in [k for k in list(sys.modules.keys()) if k == "agents" or k.startswith("agents.")]:
        del sys.modules[mod]

    try:
        summary_module = importlib.import_module("agents.coordinator_summary")
        summary = summary_module.build_compact_summary(
            kind="auto_adjust",
            stage="data_expansion",
            outcome="completed",
            action="expand_dataset",
            detail="x" * 500,
            limit=120,
        )

        assert len(summary) <= 120
        assert "auto_adjust" in summary
        assert "completed" in summary

        attempt = summary_module.summarize_attempt(
            "training",
            "poll",
            "failed",
            action="status_check",
            error="OOM",
        )
        assert "training" in attempt
        assert "failed" in attempt
        assert "OOM" in attempt
    finally:
        sys.path[:] = original_sys_path
