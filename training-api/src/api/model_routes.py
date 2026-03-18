"""
Model Management Routes
Location: training-api/src/api/model_routes.py

Contains:
- Model version management
- Model storage and retrieval
"""

import sys
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Header, status, Request, Depends
from pydantic import BaseModel, Field

# Import verify_internal_api_key from gateway for timing-safe comparison
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from training_api.src.api.gateway import verify_internal_api_key, check_rate_limit


# ==================== Request/Response Models ====================

class ModelMetadata(BaseModel):
    """Model metadata."""
    model_id: str
    task_id: str
    created_at: str
    model_path: str
    metadata: dict = {}


class ModelListResponse(BaseModel):
    """Model list response."""
    models: List[ModelMetadata]
    total: int


# ==================== Create Router ====================

model_router = APIRouter()


# ==================== Model Endpoints ====================

@model_router.get("/models", response_model=ModelListResponse)
async def list_models(
    limit: int = 100,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    List all stored models.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    from .model_manager import ModelManager

    manager = ModelManager()
    models = manager.list_models(limit=limit)

    return ModelListResponse(
        models=models,
        total=len(models)
    )


@model_router.get("/models/{task_id}")
async def get_model(
    task_id: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Get model metadata by task ID.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    from .model_manager import ModelManager

    manager = ModelManager()
    model = manager.get_model(task_id)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {task_id} not found"
        )

    return model


@model_router.delete("/models/{task_id}")
async def delete_model(
    task_id: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Delete a model by task ID.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    from .model_manager import ModelManager

    manager = ModelManager()
    success = manager.delete_model(task_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {task_id} not found"
        )

    return {"task_id": task_id, "status": "deleted"}


@model_router.post("/models/{task_id}/export")
async def export_model(
    task_id: str,
    format: str = "onnx",
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Export model to specified format.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    from .model_manager import ModelManager

    manager = ModelManager()

    try:
        export_path = manager.export_model(
            task_id=task_id,
            export_path=f"/models/{task_id}/export.{format}",
            format=format
        )
        return {
            "task_id": task_id,
            "status": "exported",
            "format": format,
            "path": export_path
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
