"""Training routes for Business API."""

import asyncio
import json
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..audit import audit_logger
from ..auth import CurrentUser, check_rate_limit, get_current_user
from ..exceptions import (
    ExternalDependencyError,
    StateConflictError,
    task_not_found,
    task_not_owned,
)
from ..task_models import (
    TaskDetailResponse,
    TaskListResponse,
    TrainStatusResponse,
)
from ..task_registry import (
    attach_execution_snapshot,
    build_result_summary,
    build_task_record,
    build_training_status_response,
    delete_task_from_redis,
    get_aggregated_task,
    get_user_tasks_from_redis,
    normalize_task_record,
    store_task_in_redis,
    verify_task_ownership,
)

# Import model registry routes to include in this router
# This keeps model registry endpoints at /api/v1/train/models/registry
from .model_registry_routes import router as _model_registry_router

router = APIRouter()

# Include model registry routes into train_router so they share the same prefix
router.include_router(_model_registry_router)


# ==================== Request/Response Models ====================

class TrainRequest(BaseModel):
    """Training request."""
    model: str = Field("yolo11m", description="Model size (n/s/m/l/x)")
    device: str = Field("cuda:0", description="CUDA device for training")
    data_yaml: str = Field(..., description="Path to dataset YAML")
    epochs: int = Field(100, description="Number of epochs")
    imgsz: int = Field(640, description="Image size")
    batch: int = Field(16, description="Batch size")
    task_type: str = Field("training", description="Task type: training/hpo")
    project: str = "/models/auto-detect"


class AdjustRequest(BaseModel):
    """Request to adjust training parameters mid-run (plateau-breaking)."""
    lr0: float | None = Field(None, description="New initial learning rate")
    augmentation_preset: str | None = Field(None, description="Augmentation preset: balanced|strong")
    resume_from: str | None = Field(None, description="Path to best.pt from previous run")
    additional_epochs: int = Field(0, description="Extra epochs to add to original schedule")


class TrainResponse(BaseModel):
    """Training response."""
    task_id: str
    status: str
    message: str
    gpu_server: str
    estimated_time_minutes: int | None = None


# ==================== Training Endpoints ====================

def get_redis_client(request: Request):
    """Get Redis client from request app state."""
    return request.app.state.redis


@router.post("/submit", response_model=TrainResponse)
async def submit_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Submit a training job to the GPU server."""
    task_id = f"train_{uuid.uuid4().hex[:8]}"

    try:
        client = http_request.app.state.training_client

        result = await client.start_training(
            task_id=task_id,
            model=request.model,
            data_yaml=request.data_yaml,
            epochs=request.epochs,
            imgsz=request.imgsz,
            output_dir=f"/home/wangxin/runs/{task_id}",
            batch=request.batch,
            device=request.device,
        )

        task_data = build_task_record(
            task_id=task_id,
            task_type="training",
            user_id=current_user.user_id,
            submission={
                "model": request.model,
                "data_yaml": request.data_yaml,
                "epochs": request.epochs,
                "imgsz": request.imgsz,
                "batch": request.batch,
                "device": request.device,
                "output_dir": f"/home/wangxin/runs/{task_id}",
            },
        )
        redis_client = get_redis_client(http_request)
        store_task_in_redis(redis_client, task_data)

        estimated_time = request.epochs * 2

        audit_logger.log_training(
            user_id=current_user.user_id,
            action="submit",
            task_id=task_id,
            request=http_request,
            details={"model": request.model, "epochs": request.epochs, "imgsz": request.imgsz}
        )

        return TrainResponse(
            task_id=task_id,
            status="submitted",
            message="Training job submitted to GPU server",
            gpu_server=os.getenv("TRAINING_API_URL", "http://localhost:8001"),
            estimated_time_minutes=estimated_time
        )

    except ExternalDependencyError as e:
        audit_logger.log_training(
            user_id=current_user.user_id,
            action="submit_failed",
            task_id=task_id,
            request=http_request,
            details={"model": request.model, "epochs": request.epochs, "error": str(e)}
        )
        raise HTTPException(
            status_code=503,
            detail=f"Training service unavailable: {str(e)}"
        )
    except Exception as e:
        audit_logger.log_training(
            user_id=current_user.user_id,
            action="submit_failed",
            task_id=task_id,
            request=http_request,
            details={"model": request.model, "epochs": request.epochs, "error": str(e)}
        )
        raise HTTPException(
            status_code=502,
            detail=f"Failed to submit training job: {str(e)}"
        )


@router.get("/status/{task_id}", response_model=TrainStatusResponse)
async def get_training_status(
    task_id: str,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Get training job status from the GPU server."""
    try:
        redis_client = get_redis_client(http_request)
        client = http_request.app.state.training_client
        task = await get_aggregated_task(redis_client, client, task_id, current_user.user_id)
        if task is None:
            raise task_not_found(task_id)

        audit_logger.log_training(
            user_id=current_user.user_id,
            action="status_check",
            task_id=task_id,
            request=http_request,
            details={"status": task.get("status")}
        )

        return build_training_status_response(task)

    except StateConflictError:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    except ExternalDependencyError as e:
        raise HTTPException(status_code=503, detail=f"Training service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to get training status: {str(e)}"
        )


@router.post("/cancel/{task_id}")
async def cancel_training(
    task_id: str,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Cancel a running training job."""
    try:
        redis_client = get_redis_client(http_request)
        task = verify_task_ownership(redis_client, task_id, current_user.user_id)
        if task is None:
            raise task_not_owned(task_id, current_user.user_id)

        client = http_request.app.state.training_client
        result = await client.cancel_task(task_id)

        audit_logger.log_training(
            user_id=current_user.user_id,
            action="cancel",
            task_id=task_id,
            request=http_request
        )

        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Training job cancelled"
        }

    except StateConflictError:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    except ExternalDependencyError as e:
        raise HTTPException(status_code=503, detail=f"Training service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to cancel training job: {str(e)}"
        )


@router.post("/adjust/{task_id}")
async def adjust_training(
    task_id: str,
    request: AdjustRequest,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Adjust training parameters mid-run to break plateau."""
    try:
        redis_client = get_redis_client(http_request)
        task = verify_task_ownership(redis_client, task_id, current_user.user_id)
        if task is None:
            raise task_not_owned(task_id, current_user.user_id)

        client = http_request.app.state.training_client

        task = normalize_task_record(task)
        submission = task.get("submission", {})
        model = submission.get("model", "yolo11m")
        data_yaml = submission.get("data_yaml", "")
        original_epochs = submission.get("epochs", 100)
        imgsz = submission.get("imgsz", 640)
        batch = submission.get("batch", 16)
        device = submission.get("device", "cuda:0")

        try:
            await client.cancel_task(task_id)
        except Exception:
            pass

        new_lr0 = request.lr0
        if new_lr0 is None and request.resume_from:
            new_lr0 = 0.005

        augmentation_preset = request.augmentation_preset
        if augmentation_preset is None and request.additional_epochs > 0:
            augmentation_preset = "strong"

        new_epochs = original_epochs + request.additional_epochs
        new_task_id = f"train_{uuid.uuid4().hex[:8]}"

        result = await client.start_training(
            task_id=new_task_id,
            model=model,
            data_yaml=data_yaml,
            epochs=new_epochs,
            imgsz=imgsz,
            output_dir=f"/home/wangxin/runs/{new_task_id}",
            batch=batch,
            device=device,
            augmentation_preset=augmentation_preset,
            resume_from=request.resume_from,
        )

        task_data = build_task_record(
            task_id=new_task_id,
            task_type="training",
            user_id=current_user.user_id,
            submission={
                "model": model,
                "data_yaml": data_yaml,
                "epochs": new_epochs,
                "imgsz": imgsz,
                "batch": batch,
                "device": device,
                "output_dir": f"/home/wangxin/runs/{new_task_id}",
                "adjusted_from": task_id,
                "lr0": new_lr0,
                "augmentation_preset": augmentation_preset,
                "resume_from": request.resume_from,
            },
            links={"adjusted_from": task_id},
        )
        store_task_in_redis(redis_client, task_data)

        task = normalize_task_record(task)
        task.update({
            "status": "adjusted",
            "registry_status": "adjusted",
            "adjusted_to": new_task_id,
            "adjusted_at": datetime.now().isoformat(),
        })
        task["links"]["adjusted_to"] = new_task_id
        redis_client.set(f"task:{task_id}", json.dumps(task), ex=7 * 24 * 60 * 60)

        audit_logger.log_training(
            user_id=current_user.user_id,
            action="adjust",
            task_id=new_task_id,
            request=http_request,
            details={
                "original_task": task_id,
                "lr0": new_lr0,
                "augmentation": augmentation_preset,
                "resume_from": request.resume_from,
                "new_epochs": new_epochs,
            }
        )

        return {
            "task_id": new_task_id,
            "status": "submitted",
            "message": f"Training adjusted and restarted. New lr0={new_lr0}, augmentation={augmentation_preset}, epochs={new_epochs}",
            "original_task_id": task_id,
            "gpu_server": "http://192.168.11.3:8001",
        }

    except StateConflictError:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    except ExternalDependencyError as e:
        raise HTTPException(status_code=503, detail=f"Training service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to adjust training: {str(e)}"
        )


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """List all tasks for the current user."""
    redis_client = get_redis_client(request)
    tasks = get_user_tasks_from_redis(redis_client, current_user.user_id)
    training_client = request.app.state.training_client
    tasks = await asyncio.gather(*[
        attach_execution_snapshot(task, training_client) for task in tasks
    ]) if tasks else []
    tasks = [{**task, "result_summary": build_result_summary(task)} for task in tasks]

    return TaskListResponse(
        tasks=tasks,
        total=len(tasks)
    )


@router.get("/tasks/{task_id}", response_model=TaskDetailResponse)
async def get_task_detail(
    task_id: str,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Get a single task detail view."""
    try:
        redis_client = get_redis_client(request)
        training_client = request.app.state.training_client
        task = await get_aggregated_task(redis_client, training_client, task_id, current_user.user_id)

        if task is None:
            raise task_not_found(task_id)

        return TaskDetailResponse(task=task)
    except StateConflictError:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    except ExternalDependencyError as e:
        raise HTTPException(status_code=503, detail=f"Training service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to get task detail: {str(e)}"
        )


@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Delete a task."""
    try:
        redis_client = get_redis_client(request)
        success = delete_task_from_redis(redis_client, task_id, current_user.user_id)

        if not success:
            raise task_not_owned(task_id, current_user.user_id)

        return {
            "task_id": task_id,
            "status": "deleted",
            "message": "Task deleted successfully"
        }
    except StateConflictError:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to delete task: {str(e)}"
        )


