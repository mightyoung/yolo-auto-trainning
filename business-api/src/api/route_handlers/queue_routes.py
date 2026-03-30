"""GPU Queue routes for Business API."""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..auth import get_current_user, CurrentUser, check_rate_limit

router = APIRouter()


class QueueTaskRequest(BaseModel):
    """GPU task queue request."""
    data_yaml: str = Field(..., description="Path to dataset YAML")
    output_dir: Optional[str] = Field("/home/wangxin/runs", description="Output directory")
    device: str = Field("cuda:0", description="CUDA device")
    epochs_per_stage: int = Field(100, description="Epochs per curriculum stage")
    model: Optional[str] = Field(None, description="Model override")


@router.post("/enqueue")
async def enqueue_training(
    request: QueueTaskRequest,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Enqueue a training task for autonomous GPU scheduling."""
    from ..gpu_scheduler import enqueue_task as _enqueue_task

    task_metadata = {
        "task_id": f"queue_{uuid.uuid4().hex[:8]}",
        "user_id": current_user.user_id,
        "data_yaml": request.data_yaml,
        "output_dir": request.output_dir,
        "device": request.device,
        "epochs_per_stage": request.epochs_per_stage,
    }
    if request.model:
        task_metadata["model"] = request.model

    queue_len = _enqueue_task(task_metadata)

    return {
        "status": "queued",
        "task_id": task_metadata["task_id"],
        "queue_length": queue_len,
        "message": "Task added to GPU queue and will dispatch when a GPU is free"
    }


@router.get("/status")
async def get_queue_status(
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """View all queued tasks without removing them."""
    from ..gpu_scheduler import peek_queue as _peek_queue

    tasks = _peek_queue()
    visible_tasks = [t for t in tasks if t.get("user_id") == current_user.user_id]

    return {
        "queue_length": len(tasks),
        "tasks": visible_tasks,
    }
