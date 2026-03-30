"""Callback routes for Business API - receives callbacks from Training API."""

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..auth import get_current_user, CurrentUser

router = APIRouter()


class TaskCallbackRequest(BaseModel):
    """Task callback from training API."""
    task_id: str
    status: str  # completed, failed
    metrics: Optional[dict] = None
    model_path: Optional[str] = None
    error: Optional[str] = None
    completed_at: Optional[str] = None


class TaskCallbackResponse(BaseModel):
    """Task callback response."""
    received: bool
    task_id: str


@router.post("/task/callback")
async def task_callback(
    request: TaskCallbackRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Receive callback from Training API when task completes.

    This endpoint is called by the Training API when:
    - Training completes
    - Export completes
    - HPO completes
    - Any task fails
    """
    print(f"Received callback for task {request.task_id}: {request.status}")

    return TaskCallbackResponse(
        received=True,
        task_id=request.task_id
    )
