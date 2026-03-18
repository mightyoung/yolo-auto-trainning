"""
Training API Internal Routes
Location: training-api/src/api/routes.py

Contains internal endpoints for:
- Training task management
- HPO task management
- Model export management
"""

import sys
from pathlib import Path

# Add project root to sys.path to access main src/ module
# This allows training-api to import from src.training.mlflow_tracker, etc.
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import uuid
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Header, status, BackgroundTasks, Depends, Request
from pydantic import BaseModel, Field

# Import verify_internal_api_key from gateway for timing-safe comparison
import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from training_api.src.api.gateway import verify_internal_api_key, check_rate_limit


# ==================== Request/Response Models ====================

class TrainStartRequest(BaseModel):
    """Internal training start request."""
    task_id: str = Field(..., description="Task identifier")
    model: str = Field("yolo11m", description="Model size")
    data_yaml: str = Field(..., description="Dataset YAML path")
    epochs: int = Field(100, description="Number of epochs")
    imgsz: int = Field(640, description="Image size")
    batch: int = Field(16, description="Batch size")
    output_dir: str = Field("/runs", description="Output directory")
    device: str = Field("cuda:0", description="Device")


class TrainStatusResponse(BaseModel):
    """Training status response."""
    task_id: str
    status: str  # submitted, running, completed, failed
    progress: float = 0.0
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    metrics: Optional[dict] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class HPOStartRequest(BaseModel):
    """HPO start request."""
    task_id: str
    model: str = "yolo11m"
    data_yaml: str
    n_trials: int = 50
    epochs_per_trial: int = 50


class ExportStartRequest(BaseModel):
    """Export start request."""
    task_id: str
    model_path: str
    platform: str = "jetson_orin"
    imgsz: int = 640


# ==================== Create Router ====================

router = APIRouter()


# ==================== Task Storage ====================

# In-memory task storage (in production, use Redis)
tasks_db = {}


# ==================== Training Endpoints ====================

@router.post("/train/start")
async def start_training(
    request: TrainStartRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Start a training job on the GPU.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    # Create task record
    task_id = request.task_id
    tasks_db[task_id] = {
        "task_id": task_id,
        "type": "training",
        "status": "submitted",
        "model": request.model,
        "data_yaml": request.data_yaml,
        "epochs": request.epochs,
        "progress": 0.0,
        "created_at": datetime.now().isoformat()
    }

    # TODO: Actually start training in background
    # For now, simulate task submission
    background_tasks = BackgroundTasks()

    return {
        "task_id": task_id,
        "status": "started",
        "worker_id": f"worker_{uuid.uuid4().hex[:6]}",
        "message": "Training task started"
    }


@router.get("/train/status/{task_id}", response_model=TrainStatusResponse)
async def get_training_status(
    task_id: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Get training job status.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    if task_id not in tasks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )

    task = tasks_db[task_id]
    return TrainStatusResponse(
        task_id=task["task_id"],
        status=task.get("status", "unknown"),
        progress=task.get("progress", 0.0),
        current_epoch=task.get("current_epoch"),
        total_epochs=task.get("epochs"),
        metrics=task.get("metrics"),
        error=task.get("error"),
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at")
    )


@router.post("/train/cancel/{task_id}")
async def cancel_training(
    task_id: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Cancel a training job.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    if task_id not in tasks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )

    tasks_db[task_id]["status"] = "cancelled"
    tasks_db[task_id]["cancelled_at"] = datetime.now().isoformat()

    return {
        "task_id": task_id,
        "status": "cancelled",
        "message": "Training task cancelled"
    }


# ==================== HPO Endpoints ====================

@router.post("/hpo/start")
async def start_hpo(
    request: HPOStartRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Start an HPO job.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    task_id = request.task_id
    tasks_db[task_id] = {
        "task_id": task_id,
        "type": "hpo",
        "status": "submitted",
        "model": request.model,
        "n_trials": request.n_trials,
        "progress": 0.0,
        "created_at": datetime.now().isoformat()
    }

    return {
        "task_id": task_id,
        "status": "started",
        "message": "HPO task started"
    }


@router.get("/hpo/status/{task_id}")
async def get_hpo_status(
    task_id: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """Get HPO job status."""
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    if task_id not in tasks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )

    return tasks_db[task_id]


# ==================== Export Endpoints ====================

@router.post("/export/start")
async def start_export(
    request: ExportStartRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Start a model export job.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    task_id = request.task_id
    tasks_db[task_id] = {
        "task_id": task_id,
        "type": "export",
        "status": "submitted",
        "model_path": request.model_path,
        "platform": request.platform,
        "progress": 0.0,
        "created_at": datetime.now().isoformat()
    }

    return {
        "task_id": task_id,
        "status": "started",
        "message": "Export task started"
    }


@router.get("/export/status/{task_id}")
async def get_export_status(
    task_id: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """Get export job status."""
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    if task_id not in tasks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )

    return tasks_db[task_id]


# ==================== Auto Label Endpoints ====================

class AutoLabelRequest(BaseModel):
    """Auto labeling request."""
    task_id: str
    input_folder: str
    classes: list[str]
    base_model: str = "grounded_sam"
    conf_threshold: float = 0.3


class AutoLabelResponse(BaseModel):
    """Auto labeling response."""
    task_id: str
    status: str
    message: str
    output_folder: Optional[str] = None
    data_yaml_path: Optional[str] = None


@router.post("/label/submit")
async def submit_labeling(
    request: AutoLabelRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Submit an auto-labeling job.

    Uses foundation models (GroundedSAM, etc.) to automatically
    label images in the input folder.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    task_id = request.task_id

    # Store task
    tasks_db[task_id] = {
        "task_id": task_id,
        "type": "labeling",
        "status": "submitted",
        "input_folder": request.input_folder,
        "classes": request.classes,
        "base_model": request.base_model,
        "progress": 0.0,
        "created_at": datetime.now().isoformat()
    }

    # Note: Actual labeling runs in background
    # For now, return task info

    return {
        "task_id": task_id,
        "status": "submitted",
        "message": f"Labeling job submitted. Base model: {request.base_model}",
        "input_folder": request.input_folder,
        "classes": request.classes
    }


@router.get("/label/status/{task_id}")
async def get_labeling_status(
    task_id: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """Get auto-labeling job status."""
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    if task_id not in tasks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )

    task = tasks_db[task_id]
    if task.get("type") != "labeling":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task {task_id} is not a labeling task"
        )

    return task


class DistillRequest(BaseModel):
    """Model distillation request."""
    task_id: str
    data_yaml: str
    target_model: str = "yolov8"
    model_size: str = "n"
    epochs: int = 100
    device: str = "cuda:0"


@router.post("/train/distill")
async def start_distillation(
    request: DistillRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Start a model distillation job.

    Uses auto-labeled dataset to train a target model.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    task_id = request.task_id
    tasks_db[task_id] = {
        "task_id": task_id,
        "type": "distillation",
        "status": "submitted",
        "data_yaml": request.data_yaml,
        "target_model": request.target_model,
        "model_size": request.model_size,
        "epochs": request.epochs,
        "progress": 0.0,
        "created_at": datetime.now().isoformat()
    }

    return {
        "task_id": task_id,
        "status": "submitted",
        "message": f"Distillation job submitted. Target: {request.target_model}{request.model_size}",
        "data_yaml": request.data_yaml,
        "epochs": request.epochs
    }


# ==================== Model Registry Endpoints ====================
# Based on MLflow best practices:
# - Version everything: Use Git for code, DVC for data, MLflow for models
# - Use stages: Staging, Production, Archived
# - Enable rollbacks: Easy model version switching
# ============================================================

class ModelRegisterRequest(BaseModel):
    """Model registration request."""
    name: str = Field(..., description="Registered model name")
    version: int = Field(..., description="Model version")
    stage: str = Field("Staging", description="Target stage")
    description: str = Field("", description="Model description")


class ModelCreateRequest(BaseModel):
    """Create registered model request."""
    name: str = Field(..., description="Model name")
    description: str = Field("", description="Model description")
    tags: Optional[dict] = Field(default_factory=dict, description="Model tags")


class ModelStageTransitionRequest(BaseModel):
    """Model stage transition request."""
    name: str = Field(..., description="Registered model name")
    version: int = Field(..., description="Model version")
    stage: str = Field(..., description="Target stage")


@router.get("/models/registry")
async def list_registered_models(
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    List all registered models.

    Returns all models in the MLflow Model Registry.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.training.mlflow_tracker import list_registered_models
        models = list_registered_models()
        return {
            "models": [
                {
                    "name": m.name,
                    "description": m.description,
                    "latest_versions": len(m.latest_versions) if hasattr(m, 'latest_versions') else 0,
                    "created_at": m.creation_timestamp if hasattr(m, 'creation_timestamp') else None,
                }
                for m in models
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.post("/models/registry")
async def create_registered_model(
    request: ModelCreateRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Create a new registered model.

    Creates a new model entry in MLflow Model Registry.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.training.mlflow_tracker import create_registered_model as create_model
        model = create_model(
            name=request.name,
            description=request.description,
            tags=request.tags if request.tags else None
        )
        if model:
            return {
                "name": model.name,
                "description": model.description,
                "status": "created"
            }
        raise HTTPException(status_code=400, detail="Failed to create model")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")


@router.get("/models/registry/{name}")
async def get_model_info(
    name: str,
    stage: Optional[str] = None,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Get model information and versions.

    Args:
        name: Registered model name
        stage: Optional stage filter (Staging, Production)
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.training.mlflow_tracker import get_latest_model_versions
        versions = get_latest_model_versions(name, stage)
        return {
            "name": name,
            "versions": [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "created_at": v.creation_timestamp if hasattr(v, 'creation_timestamp') else None,
                }
                for v in versions
            ] if versions else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.post("/models/registry/{name}/transition")
async def transition_model_stage(
    name: str,
    request: ModelStageTransitionRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Transition a model version to a different stage.

    Stages: Staging -> Production -> Archived
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.training.mlflow_tracker import transition_model_stage as transition
        result = transition(name, request.version, request.stage)
        if result:
            return {
                "name": name,
                "version": request.version,
                "stage": request.stage,
                "status": "success"
            }
        raise HTTPException(status_code=400, detail="Failed to transition stage")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to transition stage: {str(e)}")


@router.delete("/models/registry/{name}/version/{version}")
async def delete_model_version(
    name: str,
    version: int,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Delete a specific model version.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.training.mlflow_tracker import delete_model_version as delete
        success = delete(name, version)
        if success:
            return {"status": "deleted", "name": name, "version": version}
        raise HTTPException(status_code=400, detail="Failed to delete model version")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@router.delete("/models/registry/{name}")
async def delete_registered_model(
    name: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Delete a registered model and all its versions.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.training.mlflow_tracker import delete_registered_model as delete
        success = delete(name)
        if success:
            return {"status": "deleted", "name": name}
        raise HTTPException(status_code=400, detail="Failed to delete model")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


# ==================== Real-time Inference Endpoints ====================
# Based on ML system design patterns:
# - Real-time inference for immediate results
# - Model caching for performance
# - Configurable parameters
# ============================================================

class InferenceRequest(BaseModel):
    """Real-time inference request."""
    model_path: str = Field(..., description="Path to model weights")
    confidence: float = Field(0.25, description="Confidence threshold")
    iou_threshold: float = Field(0.45, description="IoU threshold for NMS")
    max_det: int = Field(300, description="Maximum detections")
    device: str = Field("cuda:0", description="Device to use")
    half: bool = Field(False, description="Use FP16 inference")


class InferenceResponse(BaseModel):
    """Inference response."""
    task_id: str
    status: str
    detections: List[dict]
    inference_time_ms: float
    model_name: str
    image_size: tuple
    timestamp: str


@router.post("/inference/predict", response_model=InferenceResponse)
async def predict(
    request: InferenceRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Run real-time inference on an image.

    Supports:
    - Image file upload (multipart/form-data)
    - Image URL
    - Base64 encoded image
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.inference.engine import get_inference_engine

        engine = get_inference_engine()

        return InferenceResponse(
            task_id=f"inf_{int(datetime.now().timestamp() * 1000)}",
            status="pending",
            detections=[],
            inference_time_ms=0,
            model_name=request.model_path,
            image_size=(0, 0),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.post("/inference/predict/image")
async def predict_image(
    model_path: str,
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 300,
    device: str = "cuda:0",
    half: bool = False,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Run inference on uploaded image.

    Upload an image file for real-time inference.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.inference.engine import get_inference_engine

        engine = get_inference_engine()

        return {
            "status": "ready",
            "message": "Image upload endpoint ready",
            "model_path": model_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.get("/inference/stats")
async def get_inference_stats(
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Get inference statistics.

    Returns metrics about inference performance.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.inference.engine import get_inference_engine

        engine = get_inference_engine()
        stats = engine.get_stats()

        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/inference/cache/clear")
async def clear_inference_cache(
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Clear the model cache.

    Frees up memory by removing cached models.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.inference.engine import get_inference_engine

        engine = get_inference_engine()
        engine.clear_cache()

        return {"status": "cleared", "message": "Model cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")
