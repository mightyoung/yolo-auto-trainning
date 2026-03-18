"""
Model Management Routes
Location: training-api/src/api/model_routes.py

Contains:
- Model version management
- Model storage and retrieval
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Header, status
from pydantic import BaseModel, Field


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
async def list_models(limit: int = 100):
    """
    List all stored models.
    """
    from .model_manager import ModelManager

    manager = ModelManager()
    models = manager.list_models(limit=limit)

    return ModelListResponse(
        models=models,
        total=len(models)
    )


@model_router.get("/models/{task_id}")
async def get_model(task_id: str):
    """
    Get model metadata by task ID.
    """
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
    x_api_key: str = Header(None, alias="X-API-Key")
):
    """
    Delete a model by task ID.
    """
    # Verify API key
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
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
    x_api_key: str = Header(None, alias="X-API-Key")
):
    """
    Export model to specified format.
    """
    # Verify API key
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
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
