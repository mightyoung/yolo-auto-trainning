import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient


project_root = Path(__file__).parent.parent.parent
training_api_src = project_root / "training-api" / "src"
deploy_module_path = training_api_src / "api" / "routes" / "deploy.py"

_original_modules: dict[str, object | None] = {}


def _install_training_api_namespace() -> None:
    for name in (
        "src",
        "src.api",
        "src.api.gateway",
        "src.api.routes",
        "src.api.routes.deploy",
        "src.api.model_routes",
        "src.deployment",
        "src.deployment.edge_config",
        "src.monitoring",
        "src.monitoring.drift_detector",
    ):
        _original_modules[name] = sys.modules.get(name)
        sys.modules.pop(name, None)

    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(training_api_src)]  # type: ignore[attr-defined]
    api_pkg = types.ModuleType("src.api")
    api_pkg.__path__ = [str(training_api_src / "api")]  # type: ignore[attr-defined]
    routes_pkg = types.ModuleType("src.api.routes")
    routes_pkg.__path__ = [str(training_api_src / "api" / "routes")]  # type: ignore[attr-defined]
    routes_pkg.router = FastAPI().router
    model_routes_pkg = types.ModuleType("src.api.model_routes")
    model_routes_pkg.model_router = FastAPI().router
    deployment_pkg = types.ModuleType("src.deployment")
    deployment_pkg.__path__ = [str(training_api_src / "deployment")]  # type: ignore[attr-defined]
    monitoring_pkg = types.ModuleType("src.monitoring")
    monitoring_pkg.__path__ = [str(training_api_src / "monitoring")]  # type: ignore[attr-defined]

    sys.modules["src"] = src_pkg
    sys.modules["src.api"] = api_pkg
    sys.modules["src.api.routes"] = routes_pkg
    sys.modules["src.api.model_routes"] = model_routes_pkg
    sys.modules["src.deployment"] = deployment_pkg
    sys.modules["src.monitoring"] = monitoring_pkg


def _restore_modules() -> None:
    for name, original in _original_modules.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    _original_modules.clear()


def _load_deploy_module():
    _install_training_api_namespace()
    spec = importlib.util.spec_from_file_location("src.api.routes.deploy", deploy_module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["src.api.routes.deploy"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_deploy_routes_are_registered_and_work(tmp_path, monkeypatch):
    module = _load_deploy_module()
    try:
        monkeypatch.setattr(module, "verify_internal_api_key", lambda key: True)

        edge_module = types.ModuleType("src.deployment.edge_config")

        class _EdgeProfileGenerator:
            def generate_config(self, device, model_path, imgsz):
                return {
                    "device": device,
                    "model_path": model_path,
                    "batch_size": 4,
                    "stream_count": 1,
                    "workspace_mb": 512,
                    "recommended_format": "onnx",
                    "fallback_formats": ["engine-fp16"],
                    "precision": "fp16",
                    "dynamic_batch": False,
                    "imgsz": imgsz,
                    "export_kwargs": {"format": "onnx"},
                    "notes": ["ok"],
                }

            def list_devices(self):
                return ["jetson_orin", "generic"]

        edge_module.EdgeProfileGenerator = _EdgeProfileGenerator
        sys.modules["src.deployment.edge_config"] = edge_module

        drift_module = types.ModuleType("src.monitoring.drift_detector")

        class _Report:
            data_drift_score = 0.21
            concept_drift_detected = True
            recommendation = "retrain"
            feature_drift = {"mAP50": 0.13}
            timestamp = "2026-04-07T00:00:00"

        class _DriftDetector:
            def __init__(self, psi_threshold):
                self.psi_threshold = psi_threshold

            def check_drift(self, **kwargs):
                return _Report()

        drift_module.DriftDetector = _DriftDetector
        sys.modules["src.monitoring.drift_detector"] = drift_module

        gateway = importlib.import_module("src.api.gateway")
        monkeypatch.setattr(gateway, "get_redis_client", lambda: None)

        ref_dir = tmp_path / "reference"
        cur_dir = tmp_path / "current"
        ref_dir.mkdir()
        cur_dir.mkdir()
        (ref_dir / "a.jpg").write_bytes(b"1")
        (cur_dir / "b.jpg").write_bytes(b"1")

        app = FastAPI()
        app.include_router(module.router, prefix="/api/v1")
        client = TestClient(app)

        edge_response = client.get(
            "/api/v1/deploy/edge-devices",
            headers={"X-API-Key": "test-key"},
        )
        assert edge_response.status_code == 200
        assert edge_response.json()["devices"] == ["jetson_orin", "generic"]

        drift_response = client.post(
            "/api/v1/monitor/drift-check",
            headers={"X-API-Key": "test-key"},
            json={
                "model_name": "yolo11m",
                "reference_image_dir": str(ref_dir),
                "current_image_dir": str(cur_dir),
                "metrics_history": [0.9, 0.88],
                "psi_threshold": 0.2,
            },
        )
        assert drift_response.status_code == 200
        assert drift_response.json()["concept_drift_detected"] is True
    finally:
        _restore_modules()
