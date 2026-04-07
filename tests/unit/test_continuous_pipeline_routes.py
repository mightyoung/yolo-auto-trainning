import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient


project_root = Path(__file__).parent.parent.parent
training_api_src = project_root / "training-api" / "src"
continuous_module_path = training_api_src / "api" / "routes" / "continuous.py"

_original_modules: dict[str, object | None] = {}


def _install_training_api_namespace() -> None:
    for name in (
        "src",
        "src.api",
        "src.api.gateway",
        "src.api.routes",
        "src.api.routes.continuous",
        "src.pipeline",
        "src.pipeline.continuous_training",
    ):
        _original_modules[name] = sys.modules.get(name)
        sys.modules.pop(name, None)

    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(training_api_src)]  # type: ignore[attr-defined]
    api_pkg = types.ModuleType("src.api")
    api_pkg.__path__ = [str(training_api_src / "api")]  # type: ignore[attr-defined]
    api_pkg.model_router = None
    routes_pkg = types.ModuleType("src.api.routes")
    routes_pkg.__path__ = [str(training_api_src / "api" / "routes")]  # type: ignore[attr-defined]
    routes_pkg.router = FastAPI().router
    pipeline_pkg = types.ModuleType("src.pipeline")
    pipeline_pkg.__path__ = [str(training_api_src / "pipeline")]  # type: ignore[attr-defined]
    model_routes_pkg = types.ModuleType("src.api.model_routes")
    model_routes_pkg.model_router = FastAPI().router

    sys.modules["src"] = src_pkg
    sys.modules["src.api"] = api_pkg
    sys.modules["src.api.model_routes"] = model_routes_pkg
    sys.modules["src.api.routes"] = routes_pkg
    sys.modules["src.pipeline"] = pipeline_pkg


def _restore_modules() -> None:
    for name, original in _original_modules.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    _original_modules.clear()


def _load_continuous_module():
    _install_training_api_namespace()
    spec = importlib.util.spec_from_file_location("src.api.routes.continuous", continuous_module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["src.api.routes.continuous"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_continuous_pipeline_routes_are_registered_and_work(monkeypatch):
    module = _load_continuous_module()
    try:
        fake_pipeline = MagicMock()
        fake_pipeline.start_retrain_pipeline.return_value = "pipeline_123"
        fake_pipeline.get_status.return_value = {"stage": "idle"}
        fake_pipeline.check_drift_and_decide.return_value = MagicMock(
            action="retrain",
            drift_score=0.42,
            message="drift detected",
        )
        fake_pipeline._ab_test_results = [types.SimpleNamespace(to_dict=lambda: {"winner": "model_b"})]

        monkeypatch.setattr(module, "verify_internal_api_key", lambda key: True)
        monkeypatch.setattr(module, "_get_continuous_pipeline", lambda: fake_pipeline)

        gateway = importlib.import_module("src.api.gateway")
        monkeypatch.setattr(gateway, "get_redis_client", lambda: None)

        app = FastAPI()
        app.include_router(module.router, prefix="/api/v1")
        client = TestClient(app)

        response = client.post(
            "/api/v1/pipeline/continuous/start",
            headers={"X-API-Key": "test-key"},
            json={"task_id": "task-1", "model_name": "yolo11m", "output_dir": "/tmp/out"},
        )
        assert response.status_code == 200
        assert response.json()["pipeline_id"] == "pipeline_123"

        status_response = client.get(
            "/api/v1/pipeline/continuous/status",
            headers={"X-API-Key": "test-key"},
        )
        assert status_response.status_code == 200
        assert status_response.json() == {"stage": "idle"}
    finally:
        _restore_modules()
