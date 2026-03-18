# Training API Integration Tests - Comprehensive
# Tests all internal routes and model routes with success/error cases

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path - for training-api
# training-api/ is the package root, parent is project root
training_api_root = Path(__file__).parent.parent.resolve()  # training-api/
training_api_src = training_api_root / "src"  # training-api/src

# Add training_api_root to sys.path FIRST to prioritize it over project root
if str(training_api_root) not in sys.path:
    sys.path.insert(0, str(training_api_root))
if str(training_api_src) not in sys.path:
    sys.path.insert(1, str(training_api_src))

# Set environment variables before importing
os.environ["INTERNAL_API_KEY"] = "test-internal-key"
os.environ["JWT_SECRET_KEY"] = "test-secret-key"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"


# ==================== Test Fixtures ====================

@pytest.fixture(scope="module")
def client():
    """Create test client with mocked dependencies."""
    from fastapi.testclient import TestClient
    from src.api.gateway import app

    # Patch after importing app
    with patch('src.api.gateway.get_redis_client'):
        test_client = TestClient(app)
        yield test_client


@pytest.fixture
def auth_headers():
    """Standard auth headers for testing."""
    return {"X-API-Key": "test-internal-key"}


@pytest.fixture
def api_key():
    """Test API key."""
    return "test-internal-key"


# ==================== Training Endpoint Tests ====================

class TestTrainingEndpoints:
    """Test training endpoints."""

    def test_start_training_success(self, client, auth_headers):
        """Test successful training start."""
        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "train_001",
                "model": "yolo11n",
                "data_yaml": "/data/coco.yaml",
                "epochs": 100,
                "imgsz": 640,
                "batch": 16,
                "device": "cuda:0"
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "train_001"
        assert data["status"] == "started"
        assert "worker_id" in data
        assert "message" in data

    def test_start_training_with_defaults(self, client, auth_headers):
        """Test training start with default values."""
        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "train_002",
                "data_yaml": "/data/coco.yaml"
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "train_002"

    def test_start_training_without_api_key(self, client):
        """Test training start without API key returns 401."""
        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "train_003",
                "data_yaml": "/data/coco.yaml"
            }
        )

        assert response.status_code == 401

    def test_start_training_without_task_id(self, client, auth_headers):
        """Test training start without task_id returns 422."""
        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "model": "yolo11n",
                "data_yaml": "/data/coco.yaml"
            },
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_start_training_without_data_yaml(self, client, auth_headers):
        """Test training start without data_yaml returns 422."""
        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "train_004"
            },
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_get_training_status_success(self, client, auth_headers):
        """Test getting training status."""
        # First create a task
        client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "status_001",
                "model": "yolo11n",
                "data_yaml": "/data/coco.yaml",
                "epochs": 100
            },
            headers=auth_headers
        )

        # Then get status
        response = client.get("/api/v1/internal/train/status/status_001")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "status_001"
        assert "status" in data
        assert "progress" in data

    def test_get_training_status_not_found(self, client):
        """Test getting status for non-existent task."""
        response = client.get("/api/v1/internal/train/status/nonexistent")

        assert response.status_code == 404

    def test_cancel_training_success(self, client, auth_headers):
        """Test cancelling a training task."""
        # First create a task
        client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "cancel_001",
                "model": "yolo11n",
                "data_yaml": "/data/coco.yaml",
                "epochs": 100
            },
            headers=auth_headers
        )

        # Cancel the task
        response = client.post("/api/v1/internal/train/cancel/cancel_001")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "cancel_001"
        assert data["status"] == "cancelled"

    def test_cancel_training_not_found(self, client):
        """Test cancelling non-existent task."""
        response = client.post("/api/v1/internal/train/cancel/nonexistent")

        assert response.status_code == 404


# ==================== HPO Endpoint Tests ====================

class TestHPOEndpoints:
    """Test HPO endpoints."""

    def test_start_hpo_success(self, client):
        """Test successful HPO start."""
        response = client.post(
            "/api/v1/internal/hpo/start",
            json={
                "task_id": "hpo_001",
                "model": "yolo11m",
                "data_yaml": "/data/coco.yaml",
                "n_trials": 10,
                "epochs_per_trial": 50
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "hpo_001"
        assert data["status"] == "started"

    def test_start_hpo_with_defaults(self, client):
        """Test HPO start with default values."""
        response = client.post(
            "/api/v1/internal/hpo/start",
            json={
                "task_id": "hpo_002",
                "data_yaml": "/data/coco.yaml"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "hpo_002"

    def test_start_hpo_without_task_id(self, client):
        """Test HPO start without task_id returns 422."""
        response = client.post(
            "/api/v1/internal/hpo/start",
            json={
                "data_yaml": "/data/coco.yaml"
            }
        )

        assert response.status_code == 422

    def test_start_hpo_without_data_yaml(self, client):
        """Test HPO start without data_yaml returns 422."""
        response = client.post(
            "/api/v1/internal/hpo/start",
            json={
                "task_id": "hpo_003"
            }
        )

        assert response.status_code == 422

    def test_get_hpo_status_success(self, client):
        """Test getting HPO status."""
        # First create an HPO task
        client.post(
            "/api/v1/internal/hpo/start",
            json={
                "task_id": "hpo_status_001",
                "model": "yolo11m",
                "data_yaml": "/data/coco.yaml",
                "n_trials": 5
            }
        )

        # Get status
        response = client.get("/api/v1/internal/hpo/status/hpo_status_001")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "hpo_status_001"
        assert data["type"] == "hpo"

    def test_get_hpo_status_not_found(self, client):
        """Test getting status for non-existent HPO task."""
        response = client.get("/api/v1/internal/hpo/status/nonexistent")

        assert response.status_code == 404


# ==================== Export Endpoint Tests ====================

class TestExportEndpoints:
    """Test export endpoints."""

    def test_start_export_success(self, client):
        """Test successful export start."""
        response = client.post(
            "/api/v1/internal/export/start",
            json={
                "task_id": "export_001",
                "model_path": "/models/train_001/weights.pt",
                "platform": "jetson_orin",
                "imgsz": 640
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "export_001"
        assert data["status"] == "started"

    def test_start_export_with_defaults(self, client):
        """Test export start with default platform."""
        response = client.post(
            "/api/v1/internal/export/start",
            json={
                "task_id": "export_002",
                "model_path": "/models/train_001/weights.pt"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "export_002"

    def test_start_export_without_task_id(self, client):
        """Test export start without task_id returns 422."""
        response = client.post(
            "/api/v1/internal/export/start",
            json={
                "model_path": "/models/train_001/weights.pt"
            }
        )

        assert response.status_code == 422

    def test_start_export_without_model_path(self, client):
        """Test export start without model_path returns 422."""
        response = client.post(
            "/api/v1/internal/export/start",
            json={
                "task_id": "export_003"
            }
        )

        assert response.status_code == 422

    def test_get_export_status_success(self, client):
        """Test getting export status."""
        # First create an export task
        client.post(
            "/api/v1/internal/export/start",
            json={
                "task_id": "export_status_001",
                "model_path": "/models/train_001/weights.pt",
                "platform": "jetson_orin"
            }
        )

        # Get status
        response = client.get("/api/v1/internal/export/status/export_status_001")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "export_status_001"
        assert data["type"] == "export"

    def test_get_export_status_not_found(self, client):
        """Test getting status for non-existent export task."""
        response = client.get("/api/v1/internal/export/status/nonexistent")

        assert response.status_code == 404


# ==================== Labeling Endpoint Tests ====================

class TestLabelingEndpoints:
    """Test auto-labeling endpoints."""

    def test_submit_labeling_success(self, client):
        """Test successful labeling submission."""
        response = client.post(
            "/api/v1/internal/label/submit",
            json={
                "task_id": "label_001",
                "input_folder": "/data/images",
                "classes": ["person", "car", "dog"],
                "base_model": "grounded_sam",
                "conf_threshold": 0.3
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "label_001"
        assert data["status"] == "submitted"

    def test_submit_labeling_with_defaults(self, client):
        """Test labeling with default base model."""
        response = client.post(
            "/api/v1/internal/label/submit",
            json={
                "task_id": "label_002",
                "input_folder": "/data/images",
                "classes": ["cat", "dog"]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "label_002"

    def test_submit_labeling_without_task_id(self, client):
        """Test labeling without task_id returns 422."""
        response = client.post(
            "/api/v1/internal/label/submit",
            json={
                "input_folder": "/data/images",
                "classes": ["person"]
            }
        )

        assert response.status_code == 422

    def test_submit_labeling_without_input_folder(self, client):
        """Test labeling without input_folder returns 422."""
        response = client.post(
            "/api/v1/internal/label/submit",
            json={
                "task_id": "label_003",
                "classes": ["person"]
            }
        )

        assert response.status_code == 422

    def test_submit_labeling_without_classes(self, client):
        """Test labeling without classes returns 422."""
        response = client.post(
            "/api/v1/internal/label/submit",
            json={
                "task_id": "label_004",
                "input_folder": "/data/images"
            }
        )

        assert response.status_code == 422

    def test_get_labeling_status_success(self, client):
        """Test getting labeling status."""
        # First submit a labeling task
        client.post(
            "/api/v1/internal/label/submit",
            json={
                "task_id": "label_status_001",
                "input_folder": "/data/images",
                "classes": ["person", "car"]
            }
        )

        # Get status
        response = client.get("/api/v1/internal/label/status/label_status_001")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "label_status_001"
        assert data["type"] == "labeling"

    def test_get_labeling_status_not_found(self, client):
        """Test getting status for non-existent labeling task."""
        response = client.get("/api/v1/internal/label/status/nonexistent")

        assert response.status_code == 404


# ==================== Distillation Endpoint Tests ====================

class TestDistillationEndpoints:
    """Test distillation endpoints."""

    def test_start_distillation_success(self, client):
        """Test successful distillation start."""
        response = client.post(
            "/api/v1/internal/train/distill",
            json={
                "task_id": "distill_001",
                "data_yaml": "/data/autolabeled.yaml",
                "target_model": "yolov8",
                "model_size": "n",
                "epochs": 100,
                "device": "cuda:0"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "distill_001"
        assert data["status"] == "submitted"

    def test_start_distillation_with_defaults(self, client):
        """Test distillation with default values."""
        response = client.post(
            "/api/v1/internal/train/distill",
            json={
                "task_id": "distill_002",
                "data_yaml": "/data/autolabeled.yaml"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "distill_002"

    def test_start_distillation_without_task_id(self, client):
        """Test distillation without task_id returns 422."""
        response = client.post(
            "/api/v1/internal/train/distill",
            json={
                "data_yaml": "/data/autolabeled.yaml"
            }
        )

        assert response.status_code == 422

    def test_start_distillation_without_data_yaml(self, client):
        """Test distillation without data_yaml returns 422."""
        response = client.post(
            "/api/v1/internal/train/distill",
            json={
                "task_id": "distill_003"
            }
        )

        assert response.status_code == 422


# ==================== Model Management Tests ====================

class TestModelManagementEndpoints:
    """Test model management endpoints."""

    def test_list_models_success(self, client):
        """Test listing models."""
        response = client.get("/api/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total" in data

    def test_list_models_with_limit(self, client):
        """Test listing models with limit parameter."""
        response = client.get("/api/v1/models?limit=10")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data

    def test_get_model_not_found(self, client):
        """Test getting non-existent model."""
        response = client.get("/api/v1/models/nonexistent")

        assert response.status_code == 404

    def test_delete_model_without_api_key(self, client):
        """Test deleting model without API key."""
        response = client.delete("/api/v1/models/train_001")

        assert response.status_code == 401

    def test_export_model_without_api_key(self, client):
        """Test exporting model without API key."""
        response = client.post("/api/v1/models/train_001/export")

        assert response.status_code == 401


# ==================== Authentication Tests ====================

class TestAuthentication:
    """Test authentication and authorization."""

    def test_invalid_api_key(self, client):
        """Test invalid API key is rejected."""
        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "auth_test",
                "data_yaml": "/data/coco.yaml"
            },
            headers={"X-API-Key": "invalid-key"}
        )

        assert response.status_code == 401

    def test_missing_api_key_protected_endpoints(self, client):
        """Test protected endpoints require API key."""
        # Test training start
        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "auth_test_2",
                "data_yaml": "/data/coco.yaml"
            }
        )
        assert response.status_code == 401

        # Test model delete
        response = client.delete("/api/v1/models/test")
        assert response.status_code == 401

        # Test model export
        response = client.post("/api/v1/models/test/export")
        assert response.status_code == 401


# ==================== Health and Root Endpoints ====================

class TestHealthEndpoints:
    """Test health check and root endpoints."""

    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "gpu" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "docs" in data


# ==================== Edge Cases and Error Handling ====================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_task_id_with_special_characters(self, client, auth_headers):
        """Test task ID with special characters."""
        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "task_with_underscores-dashes.and.dots",
                "data_yaml": "/data/coco.yaml"
            },
            headers=auth_headers
        )

        assert response.status_code == 200

    def test_invalid_json_body(self, client, auth_headers):
        """Test invalid JSON body."""
        response = client.post(
            "/api/v1/internal/train/start",
            data="not valid json",
            headers={**auth_headers, "Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_empty_task_id(self, client, auth_headers):
        """Test empty task ID."""
        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "",
                "data_yaml": "/data/coco.yaml"
            },
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_negative_epochs(self, client, auth_headers):
        """Test negative epochs value."""
        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "neg_epochs",
                "data_yaml": "/data/coco.yaml",
                "epochs": -1
            },
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_large_epochs_value(self, client, auth_headers):
        """Test very large epochs value."""
        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "large_epochs",
                "data_yaml": "/data/coco.yaml",
                "epochs": 1000000
            },
            headers=auth_headers
        )

        # Should accept but might be impractical
        assert response.status_code == 200

    def test_invalid_device_format(self, client, auth_headers):
        """Test invalid device format."""
        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "invalid_device",
                "data_yaml": "/data/coco.yaml",
                "device": "invalid_device_name"
            },
            headers=auth_headers
        )

        # Should accept but might fail at runtime
        assert response.status_code == 200


# ==================== Training Lifecycle Tests ====================

class TestTrainingLifecycle:
    """Test complete training lifecycle."""

    def test_full_training_lifecycle(self, client, auth_headers):
        """Test complete training lifecycle: start -> status -> cancel."""
        task_id = "lifecycle_test"

        # Start training
        start_response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": task_id,
                "model": "yolo11n",
                "data_yaml": "/data/coco.yaml",
                "epochs": 100
            },
            headers=auth_headers
        )
        assert start_response.status_code == 200

        # Get status
        status_response = client.get(f"/api/v1/internal/train/status/{task_id}")
        assert status_response.status_code == 200

        # Cancel training
        cancel_response = client.post(f"/api/v1/internal/train/cancel/{task_id}")
        assert cancel_response.status_code == 200

        # Verify cancelled status
        final_status = client.get(f"/api/v1/internal/train/status/{task_id}")
        assert final_status.json()["status"] == "cancelled"

    def test_multiple_trainings_same_task_id(self, client, auth_headers):
        """Test starting multiple trainings with same task ID."""
        task_id = "duplicate_test"

        # First start
        response1 = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": task_id,
                "data_yaml": "/data/coco.yaml"
            },
            headers=auth_headers
        )
        assert response1.status_code == 200

        # Second start - should succeed (overwrites)
        response2 = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": task_id,
                "data_yaml": "/data/coco.yaml"
            },
            headers=auth_headers
        )
        assert response2.status_code == 200


# ==================== Parameter Validation Tests ====================

class TestParameterValidation:
    """Test parameter validation."""

    def test_invalid_batch_size(self, client, auth_headers):
        """Test invalid batch size."""
        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "invalid_batch",
                "data_yaml": "/data/coco.yaml",
                "batch": 0
            },
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_invalid_image_size(self, client, auth_headers):
        """Test invalid image size."""
        response = client.post(
            "/api/v1/internal/train/start",
            json={
                "task_id": "invalid_imgsz",
                "data_yaml": "/data/coco.yaml",
                "imgsz": -1
            },
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_invalid_conf_threshold(self, client):
        """Test invalid confidence threshold."""
        response = client.post(
            "/api/v1/internal/label/submit",
            json={
                "task_id": "invalid_conf",
                "input_folder": "/data/images",
                "classes": ["person"],
                "conf_threshold": 1.5  # Should be 0-1
            }
        )

        assert response.status_code == 422

    def test_invalid_n_trials(self, client):
        """Test invalid number of trials."""
        response = client.post(
            "/api/v1/internal/hpo/start",
            json={
                "task_id": "invalid_trials",
                "data_yaml": "/data/coco.yaml",
                "n_trials": 0
            }
        )

        assert response.status_code == 422
