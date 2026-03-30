import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient


project_root = Path(__file__).parent.parent.parent
business_api_root = project_root / "business-api"
if str(business_api_root) not in sys.path:
    sys.path.insert(0, str(business_api_root))


os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key")
os.environ.setdefault("TRAINING_API_URL", "http://localhost:8001")
os.environ.setdefault("TRAINING_API_KEY", "test-api-key")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")


def _install_business_src_package():
    _install_src_package(business_api_root)
    _prime_business_task_registry()


def _install_src_package(package_root: Path):
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]

    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(package_root / "src")]
    api_pkg = types.ModuleType("src.api")
    api_pkg.__path__ = [str(package_root / "src" / "api")]
    sys.modules["src"] = src_pkg
    sys.modules["src.api"] = api_pkg


def _prime_business_task_registry():
    from src.api import task_models, task_registry

    for name in (
        "TaskExecutionSummaryResponse",
        "TaskRecordResponse",
        "TaskListResponse",
        "TaskDetailResponse",
        "TrainStatusResponse",
        "ExportStatusResponse",
    ):
        setattr(task_registry, name, getattr(task_models, name))


_install_business_src_package()


def _build_client(mock_redis, mock_training_client):
    from src.api import gateway
    from src.api import routes

    with patch.object(gateway, "get_redis_client", return_value=mock_redis):
        routes.get_redis_client = lambda request=None: gateway.get_redis_client()
        app = gateway.app
        app.state.redis = mock_redis
        app.state.training_client = mock_training_client
        return TestClient(app)


def _auth_headers():
    return {"Authorization": "Bearer test-token"}


def _mock_current_user():
    return type("CurrentUser", (), {"user_id": "test-user"})()


def test_submit_training_persists_device_and_batch():
    mock_redis = Mock()
    mock_training_client = Mock()
    mock_training_client.start_training = AsyncMock(return_value={"task_id": "train_123", "status": "started"})

    from src.api import gateway
    from src.api import routes

    gateway.app.dependency_overrides[routes.get_current_user] = _mock_current_user
    gateway.app.dependency_overrides[routes.check_rate_limit] = lambda: None
    try:
        client = _build_client(mock_redis, mock_training_client)
        response = client.post(
            "/api/v1/train/submit",
            json={
                "model": "yolo11x",
                "data_yaml": "/data/test.yaml",
                "epochs": 20,
                "imgsz": 1280,
                "batch": 8,
                "device": "cuda:1",
            },
            headers=_auth_headers(),
        )
    finally:
        gateway.app.dependency_overrides.clear()

    assert response.status_code == 200
    kwargs = mock_training_client.start_training.await_args.kwargs
    assert kwargs["batch"] == 8
    assert kwargs["device"] == "cuda:1"

    stored_payload = json.loads(mock_redis.set.call_args.args[1])
    assert "params" not in stored_payload
    assert stored_payload["submission"]["batch"] == 8
    assert stored_payload["submission"]["device"] == "cuda:1"


def test_adjust_training_updates_original_task_and_reuses_original_params():
    mock_redis = Mock()
    mock_training_client = Mock()
    mock_training_client.cancel_task = AsyncMock(return_value={"task_id": "train_old", "status": "cancelled"})
    mock_training_client.start_training = AsyncMock(return_value={"task_id": "train_new", "status": "started"})
    original_task = {
        "task_id": "train_old",
        "task_type": "training",
        "user_id": "test-user",
        "status": "running",
        "created_at": "2026-03-30T09:00:00",
        "params": {
            "model": "yolo11x",
            "data_yaml": "/data/test.yaml",
            "epochs": 100,
            "imgsz": 1280,
            "batch": 8,
            "device": "cuda:1",
        },
    }
    mock_redis.get.return_value = json.dumps(original_task)

    from src.api import gateway
    from src.api import routes

    gateway.app.dependency_overrides[routes.get_current_user] = _mock_current_user
    gateway.app.dependency_overrides[routes.check_rate_limit] = lambda: None
    try:
        client = _build_client(mock_redis, mock_training_client)
        response = client.post(
            "/api/v1/train/adjust/train_old",
            json={"additional_epochs": 20, "resume_from": "/tmp/best.pt"},
            headers=_auth_headers(),
        )
    finally:
        gateway.app.dependency_overrides.clear()

    assert response.status_code == 200
    kwargs = mock_training_client.start_training.await_args.kwargs
    assert kwargs["batch"] == 8
    assert kwargs["device"] == "cuda:1"

    updated_original = json.loads(mock_redis.set.call_args_list[-1].args[1])
    assert updated_original["status"] == "adjusted"
    assert updated_original["adjusted_to"].startswith("train_")
    assert updated_original["submission"]["model"] == "yolo11x"


def test_business_status_exposes_resubmit_metadata():
    mock_redis = Mock()
    owned_task = {
        "task_id": "train_123",
        "task_type": "training",
        "user_id": "test-user",
        "status": "submitted",
        "created_at": "2026-03-30T09:00:00",
    }
    mock_redis.get.return_value = json.dumps(owned_task)
    mock_training_client = Mock()
    mock_training_client.get_task_status = AsyncMock(return_value={
        "task_id": "train_123",
        "status": "running",
        "progress": 0.5,
        "resubmit_count": 2,
        "last_resubmitted_at": "2026-03-30T10:00:00",
        "resubmit_reason": "failed_task",
    })

    from src.api import gateway
    from src.api import routes

    gateway.app.dependency_overrides[routes.get_current_user] = _mock_current_user
    gateway.app.dependency_overrides[routes.check_rate_limit] = lambda: None
    try:
        client = _build_client(mock_redis, mock_training_client)
        response = client.get("/api/v1/train/status/train_123", headers=_auth_headers())
    finally:
        gateway.app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["resubmit_count"] == 2
    assert data["resubmit_reason"] == "failed_task"


def test_list_tasks_returns_registry_and_execution_views():
    mock_redis = Mock()
    mock_redis.smembers.return_value = {"train_123"}
    mock_redis.get.return_value = json.dumps({
        "task_id": "train_123",
        "task_type": "training",
        "user_id": "test-user",
        "status": "submitted",
        "created_at": "2026-03-30T09:00:00",
        "params": {"model": "yolo11n", "data_yaml": "/data/test.yaml"},
    })
    mock_training_client = Mock()
    mock_training_client.get_task_status = AsyncMock(return_value={
        "task_id": "train_123",
        "status": "running",
        "progress": 0.4,
    })

    from src.api import gateway
    from src.api import routes

    gateway.app.dependency_overrides[routes.get_current_user] = _mock_current_user
    gateway.app.dependency_overrides[routes.check_rate_limit] = lambda: None
    try:
        client = _build_client(mock_redis, mock_training_client)
        response = client.get("/api/v1/train/tasks", headers=_auth_headers())
    finally:
        gateway.app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    task = payload["tasks"][0]
    assert task["registry_status"] == "submitted"
    assert task["status"] == "running"
    assert task["submission"]["model"] == "yolo11n"
    assert task["links"]["execution_task_id"] == "train_123"
    assert task["execution"]["progress"] == 0.4
    assert task["execution_summary"]["status"] == "running"
    assert task["execution_summary"]["progress"] == 0.4
    assert "params" not in task


def test_export_status_uses_aggregated_execution_snapshot():
    mock_redis = Mock()
    mock_redis.get.return_value = json.dumps({
        "task_id": "export_123",
        "task_type": "export",
        "user_id": "test-user",
        "status": "submitted",
        "created_at": "2026-03-30T09:00:00",
        "submission": {"model_path": "/tmp/best.pt", "platform": "jetson_orin"},
    })
    mock_training_client = Mock()
    mock_training_client.get_task_status = AsyncMock(return_value={
        "task_id": "export_123",
        "status": "completed",
        "export_path": "/tmp/best.onnx",
    })

    from src.api import gateway
    from src.api import routes

    gateway.app.dependency_overrides[routes.get_current_user] = _mock_current_user
    gateway.app.dependency_overrides[routes.check_rate_limit] = lambda: None
    try:
        client = _build_client(mock_redis, mock_training_client)
        response = client.get("/api/v1/deploy/export/status/export_123", headers=_auth_headers())
    finally:
        gateway.app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["export_path"] == "/tmp/best.onnx"


def test_list_tasks_falls_back_to_registry_summary_when_execution_unavailable():
    mock_redis = Mock()
    mock_redis.smembers.return_value = {"train_123"}
    mock_redis.get.return_value = json.dumps({
        "task_id": "train_123",
        "task_type": "training",
        "user_id": "test-user",
        "registry_status": "submitted",
        "status": "submitted",
        "created_at": "2026-03-30T09:00:00",
        "submission": {"model": "yolo11n"},
    })
    mock_training_client = Mock()
    mock_training_client.get_task_status = AsyncMock(side_effect=RuntimeError("training api unavailable"))

    from src.api import gateway
    from src.api import routes

    gateway.app.dependency_overrides[routes.get_current_user] = _mock_current_user
    gateway.app.dependency_overrides[routes.check_rate_limit] = lambda: None
    try:
        client = _build_client(mock_redis, mock_training_client)
        response = client.get("/api/v1/train/tasks", headers=_auth_headers())
    finally:
        gateway.app.dependency_overrides.clear()

    assert response.status_code == 200
    task = response.json()["tasks"][0]
    assert task["status"] == "submitted"
    assert task["execution_summary"]["status"] == "submitted"
    assert task["execution_summary"]["progress"] is None


def test_analysis_task_uses_normalized_submission_schema():
    mock_redis = Mock()
    mock_training_client = Mock()

    from src.api import gateway
    from src.api import routes

    class FakeDeepAnalyzeClient:
        def __init__(self, *args, **kwargs):
            pass

        def health_check(self):
            return True

        def analyze_dataset(self, dataset_path, analysis_type):
            return {"content": "ok", "files": []}

    gateway.app.dependency_overrides[routes.get_current_user] = _mock_current_user
    gateway.app.dependency_overrides[routes.check_rate_limit] = lambda: None
    try:
        with patch("src.api.deepanalyze_client.DeepAnalyzeClient", FakeDeepAnalyzeClient):
            client = _build_client(mock_redis, mock_training_client)
            response = client.post(
                "/api/v1/analysis/analyze",
                json={"dataset_path": "/data/train.csv", "analysis_type": "quality"},
                headers=_auth_headers(),
            )
    finally:
        gateway.app.dependency_overrides.clear()

    assert response.status_code == 200
    stored_payload = json.loads(mock_redis.set.call_args.args[1])
    assert stored_payload["task_type"] == "analysis"
    assert stored_payload["submission"]["dataset_path"] == "/data/train.csv"
    assert stored_payload["registry_status"] == "completed"
    assert "params" not in stored_payload


def test_report_task_is_persisted_with_normalized_schema():
    mock_redis = Mock()
    mock_training_client = Mock()

    from src.api import gateway
    from src.api import routes

    class FakeDeepAnalyzeClient:
        def __init__(self, *args, **kwargs):
            pass

        def health_check(self):
            return True

        def generate_report(self, data_description, analysis_goals):
            return {"content": "report", "files": []}

    gateway.app.dependency_overrides[routes.get_current_user] = _mock_current_user
    gateway.app.dependency_overrides[routes.check_rate_limit] = lambda: None
    try:
        with patch("src.api.deepanalyze_client.DeepAnalyzeClient", FakeDeepAnalyzeClient):
            client = _build_client(mock_redis, mock_training_client)
            response = client.post(
                "/api/v1/analysis/report",
                json={"data_description": "sales", "analysis_goals": ["trend"]},
                headers=_auth_headers(),
            )
    finally:
        gateway.app.dependency_overrides.clear()

    assert response.status_code == 200
    stored_payload = json.loads(mock_redis.set.call_args.args[1])
    assert stored_payload["task_type"] == "report"
    assert stored_payload["submission"]["analysis_goals"] == ["trend"]
    assert stored_payload["registry_status"] == "completed"
    assert "params" not in stored_payload


def test_list_tasks_includes_result_summary_for_analysis_tasks():
    mock_redis = Mock()
    mock_redis.smembers.return_value = {"analyze_123"}
    mock_redis.get.return_value = json.dumps({
        "task_id": "analyze_123",
        "task_type": "analysis",
        "user_id": "test-user",
        "registry_status": "completed",
        "status": "completed",
        "created_at": "2026-03-30T09:00:00",
        "submission": {"dataset_path": "/data/train.csv", "analysis_type": "quality"},
        "result": {"content": "quality report", "files": [{"name": "a.json"}]},
    })
    mock_training_client = Mock()

    from src.api import gateway
    from src.api import routes

    gateway.app.dependency_overrides[routes.get_current_user] = _mock_current_user
    gateway.app.dependency_overrides[routes.check_rate_limit] = lambda: None
    try:
        client = _build_client(mock_redis, mock_training_client)
        response = client.get("/api/v1/train/tasks", headers=_auth_headers())
    finally:
        gateway.app.dependency_overrides.clear()

    assert response.status_code == 200
    task = response.json()["tasks"][0]
    assert task["result_summary"]["status"] == "completed"
    assert task["result_summary"]["file_count"] == 1
    assert task["result_summary"]["content_preview"] == "quality report"


def test_task_detail_returns_aggregated_training_task():
    mock_redis = Mock()
    mock_redis.get.return_value = json.dumps({
        "task_id": "train_123",
        "task_type": "training",
        "user_id": "test-user",
        "registry_status": "submitted",
        "status": "submitted",
        "created_at": "2026-03-30T09:00:00",
        "submission": {"model": "yolo11n"},
    })
    mock_training_client = Mock()
    mock_training_client.get_task_status = AsyncMock(return_value={
        "task_id": "train_123",
        "status": "running",
        "progress": 0.4,
    })

    from src.api import gateway
    from src.api import routes

    gateway.app.dependency_overrides[routes.get_current_user] = _mock_current_user
    gateway.app.dependency_overrides[routes.check_rate_limit] = lambda: None
    try:
        client = _build_client(mock_redis, mock_training_client)
        response = client.get("/api/v1/train/tasks/train_123", headers=_auth_headers())
    finally:
        gateway.app.dependency_overrides.clear()

    assert response.status_code == 200
    task = response.json()["task"]
    assert task["task_id"] == "train_123"
    assert task["status"] == "running"
    assert task["execution_summary"]["progress"] == 0.4


def test_task_detail_returns_analysis_result_summary():
    mock_redis = Mock()
    mock_redis.get.return_value = json.dumps({
        "task_id": "analyze_123",
        "task_type": "analysis",
        "user_id": "test-user",
        "registry_status": "completed",
        "status": "completed",
        "created_at": "2026-03-30T09:00:00",
        "submission": {"dataset_path": "/data/train.csv", "analysis_type": "quality"},
        "result": {"content": "quality report", "files": [{"name": "a.json"}]},
    })
    mock_training_client = Mock()

    from src.api import gateway
    from src.api import routes

    gateway.app.dependency_overrides[routes.get_current_user] = _mock_current_user
    gateway.app.dependency_overrides[routes.check_rate_limit] = lambda: None
    try:
        client = _build_client(mock_redis, mock_training_client)
        response = client.get("/api/v1/train/tasks/analyze_123", headers=_auth_headers())
    finally:
        gateway.app.dependency_overrides.clear()

    assert response.status_code == 200
    task = response.json()["task"]
    assert task["task_type"] == "analysis"
    assert task["result_summary"]["file_count"] == 1
    assert task["result_summary"]["content_preview"] == "quality report"


def test_training_status_matches_task_detail_execution_snapshot():
    mock_redis = Mock()
    mock_redis.get.return_value = json.dumps({
        "task_id": "train_123",
        "task_type": "training",
        "user_id": "test-user",
        "registry_status": "submitted",
        "status": "submitted",
        "created_at": "2026-03-30T09:00:00",
        "submission": {"model": "yolo11n"},
    })
    execution_snapshot = {
        "task_id": "train_123",
        "status": "running",
        "progress": 0.75,
        "current_epoch": 15,
        "total_epochs": 20,
        "metrics": {"mAP50": 0.63},
        "live_mAP50": 0.63,
        "lr_decay_triggered": True,
        "lr_decay_signal": {"factor": 0.5},
        "augment_boost_active": False,
        "data_expansion_requested": True,
        "data_expansion_signal": {"query": "hard cases"},
        "strategies_triggered": ["lr_decay", "data_expansion"],
        "resubmit_count": 1,
        "last_resubmitted_at": "2026-03-30T10:00:00",
        "resubmit_reason": "submitted_timeout",
    }
    mock_training_client = Mock()
    mock_training_client.get_task_status = AsyncMock(return_value=execution_snapshot)

    from src.api import gateway
    from src.api import routes

    gateway.app.dependency_overrides[routes.get_current_user] = _mock_current_user
    gateway.app.dependency_overrides[routes.check_rate_limit] = lambda: None
    try:
        client = _build_client(mock_redis, mock_training_client)
        status_response = client.get("/api/v1/train/status/train_123", headers=_auth_headers())
        detail_response = client.get("/api/v1/train/tasks/train_123", headers=_auth_headers())
    finally:
        gateway.app.dependency_overrides.clear()

    assert status_response.status_code == 200
    assert detail_response.status_code == 200

    status_payload = status_response.json()
    task_payload = detail_response.json()["task"]
    execution = task_payload["execution"]

    assert status_payload["status"] == task_payload["status"] == execution["status"]
    assert status_payload["progress"] == execution["progress"]
    assert status_payload["current_epoch"] == execution["current_epoch"]
    assert status_payload["total_epochs"] == execution["total_epochs"]
    assert status_payload["metrics"] == execution["metrics"]
    assert status_payload["live_mAP50"] == execution["live_mAP50"]
    assert status_payload["lr_decay_signal"] == execution["lr_decay_signal"]
    assert status_payload["data_expansion_signal"] == execution["data_expansion_signal"]
    assert status_payload["strategies_triggered"] == execution["strategies_triggered"]
    assert status_payload["resubmit_count"] == execution["resubmit_count"]
    assert status_payload["resubmit_reason"] == execution["resubmit_reason"]


def test_training_status_falls_back_to_registry_state_when_execution_unavailable():
    mock_redis = Mock()
    mock_redis.get.return_value = json.dumps({
        "task_id": "train_123",
        "task_type": "training",
        "user_id": "test-user",
        "registry_status": "submitted",
        "status": "submitted",
        "created_at": "2026-03-30T09:00:00",
        "submission": {"model": "yolo11n"},
    })
    mock_training_client = Mock()
    mock_training_client.get_task_status = AsyncMock(side_effect=RuntimeError("training api unavailable"))

    from src.api import gateway
    from src.api import routes

    gateway.app.dependency_overrides[routes.get_current_user] = _mock_current_user
    gateway.app.dependency_overrides[routes.check_rate_limit] = lambda: None
    try:
        client = _build_client(mock_redis, mock_training_client)
        response = client.get("/api/v1/train/status/train_123", headers=_auth_headers())
    finally:
        gateway.app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "submitted"
    assert payload["progress"] == 0.0
    assert payload["error"] is None


def test_training_gateway_imports_without_required_env_vars():
    try:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JWT_SECRET_KEY", None)
            os.environ.pop("INTERNAL_API_KEY", None)
            _install_src_package(project_root / "training-api")
            from src.api import gateway

        assert gateway.verify_internal_api_key("anything") is False
    finally:
        _install_business_src_package()


def test_business_gateway_imports_without_training_env_vars():
    try:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRAINING_API_URL", None)
            os.environ.pop("TRAINING_API_KEY", None)
            _install_business_src_package()
            from src.api import gateway

        assert gateway.TRAINING_API_URL is None
        assert gateway.TRAINING_API_KEY is None
    finally:
        _install_business_src_package()


def test_export_status_matches_task_detail_execution_snapshot():
    mock_redis = Mock()
    mock_redis.get.return_value = json.dumps({
        "task_id": "export_123",
        "task_type": "export",
        "user_id": "test-user",
        "registry_status": "submitted",
        "status": "submitted",
        "created_at": "2026-03-30T09:00:00",
        "submission": {
            "model_path": "/tmp/best.pt",
            "platform": "jetson_orin",
            "imgsz": 640,
        },
    })
    execution_snapshot = {
        "task_id": "export_123",
        "status": "running",
        "progress": 0.6,
        "model_path": "/tmp/best.pt",
        "platform": "jetson_orin",
        "imgsz": 640,
        "formats": ["onnx"],
        "int8_quantize": False,
        "started_at": "2026-03-30T10:00:00",
        "export_path": "/tmp/best.onnx",
    }
    mock_training_client = Mock()
    mock_training_client.get_task_status = AsyncMock(return_value=execution_snapshot)

    from src.api import gateway
    from src.api import routes

    gateway.app.dependency_overrides[routes.get_current_user] = _mock_current_user
    gateway.app.dependency_overrides[routes.check_rate_limit] = lambda: None
    try:
        client = _build_client(mock_redis, mock_training_client)
        status_response = client.get("/api/v1/deploy/export/status/export_123", headers=_auth_headers())
        detail_response = client.get("/api/v1/train/tasks/export_123", headers=_auth_headers())
    finally:
        gateway.app.dependency_overrides.clear()

    assert status_response.status_code == 200
    assert detail_response.status_code == 200

    status_payload = status_response.json()
    task_payload = detail_response.json()["task"]
    execution = task_payload["execution"]

    assert status_payload["status"] == task_payload["status"] == execution["status"]
    assert status_payload["progress"] == execution["progress"]
    assert status_payload["model_path"] == execution["model_path"]
    assert status_payload["platform"] == execution["platform"]
    assert status_payload["imgsz"] == execution["imgsz"]
    assert status_payload["formats"] == execution["formats"]
    assert status_payload["export_path"] == execution["export_path"]


def test_export_status_falls_back_to_registry_submission_when_execution_unavailable():
    mock_redis = Mock()
    mock_redis.get.return_value = json.dumps({
        "task_id": "export_123",
        "task_type": "export",
        "user_id": "test-user",
        "registry_status": "submitted",
        "status": "submitted",
        "created_at": "2026-03-30T09:00:00",
        "submission": {
            "model_path": "/tmp/best.pt",
            "platform": "jetson_orin",
            "imgsz": 640,
        },
    })
    mock_training_client = Mock()
    mock_training_client.get_task_status = AsyncMock(side_effect=RuntimeError("training api unavailable"))

    from src.api import gateway
    from src.api import routes

    gateway.app.dependency_overrides[routes.get_current_user] = _mock_current_user
    gateway.app.dependency_overrides[routes.check_rate_limit] = lambda: None
    try:
        client = _build_client(mock_redis, mock_training_client)
        response = client.get("/api/v1/deploy/export/status/export_123", headers=_auth_headers())
    finally:
        gateway.app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "submitted"
    assert payload["progress"] == 0.0
    assert payload["model_path"] == "/tmp/best.pt"
    assert payload["platform"] == "jetson_orin"
    assert payload["imgsz"] == 640
