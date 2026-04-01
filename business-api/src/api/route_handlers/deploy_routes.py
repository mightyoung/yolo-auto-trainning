"""Deployment/Export routes for Business API."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..audit import audit_logger
from ..auth import CurrentUser, check_rate_limit, get_current_user
from ..exceptions import (
    ExternalDependencyError,
    StateConflictError,
)
from ..task_models import ExportStatusResponse
from ..task_registry import (
    build_export_status_response,
    build_task_record,
    get_aggregated_task,
    store_task_in_redis,
)

router = APIRouter()


class ExportRequest(BaseModel):
    """Model export request."""
    model_path: str = Field(..., description="Path to trained model")
    platform: str = Field("jetson_orin", description="Target platform")
    imgsz: int = Field(640, description="Input image size")


class ExportResponse(BaseModel):
    """Export response."""
    task_id: str
    status: str
    message: str


def get_redis_client(request: Request):
    """Get Redis client from request app state."""
    return request.app.state.redis


@router.post("/export", response_model=ExportResponse)
async def export_model(
    request: ExportRequest,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Submit a model export job to the GPU server."""
    task_id = f"export_{uuid.uuid4().hex[:8]}"

    try:
        client = http_request.app.state.training_client

        result = await client.start_export(
            task_id=task_id,
            model_path=request.model_path,
            platform=request.platform,
            imgsz=request.imgsz
        )

        task_data = build_task_record(
            task_id=task_id,
            task_type="export",
            user_id=current_user.user_id,
            submission={
                "model_path": request.model_path,
                "platform": request.platform,
                "imgsz": request.imgsz
            },
        )
        redis_client = get_redis_client(http_request)
        store_task_in_redis(redis_client, task_data)

        audit_logger.log(
            action="export",
            user_id=current_user.user_id,
            resource=f"export/{task_id}",
            request=http_request,
            details={"model_path": request.model_path, "platform": request.platform, "imgsz": request.imgsz},
            status="success"
        )

        return ExportResponse(
            task_id=task_id,
            status="submitted",
            message="Export job submitted to GPU server"
        )

    except ExternalDependencyError as e:
        audit_logger.log(
            action="export",
            user_id=current_user.user_id,
            resource=f"export/{task_id}",
            request=http_request,
            details={"model_path": request.model_path, "platform": request.platform, "error": str(e)},
            status="failure"
        )
        raise HTTPException(
            status_code=503,
            detail=f"Training service unavailable: {str(e)}"
        )
    except Exception as e:
        audit_logger.log(
            action="export",
            user_id=current_user.user_id,
            resource=f"export/{task_id}",
            request=http_request,
            details={"model_path": request.model_path, "platform": request.platform, "error": str(e)},
            status="failure"
        )
        raise HTTPException(
            status_code=502,
            detail=f"Failed to submit export job: {str(e)}"
        )


@router.get("/export/status/{task_id}", response_model=ExportStatusResponse)
async def get_export_status(
    task_id: str,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Get export job status."""
    try:
        redis_client = get_redis_client(http_request)
        client = http_request.app.state.training_client
        task = await get_aggregated_task(redis_client, client, task_id, current_user.user_id)
        if task is None:
            raise StateConflictError(
                f"Task not found: {task_id}",
                resource_type="task",
                resource_id=task_id,
            )
        return build_export_status_response(task)

    except StateConflictError:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    except ExternalDependencyError as e:
        raise HTTPException(status_code=503, detail=f"Training service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to get export status: {str(e)}"
        )
