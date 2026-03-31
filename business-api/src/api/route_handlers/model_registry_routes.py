"""Model registry routes for Business API.

Endpoints for managing MLflow model registry.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ..auth import get_current_user, CurrentUser, check_rate_limit
from ..exceptions import ExternalDependencyError, BusinessError

router = APIRouter()


# ==================== Request Models ====================

class ModelCreateRequest(BaseModel):
    """Create registered model request."""
    name: str
    description: str = ""
    tags: Optional[dict] = {}


class ModelTransitionRequest(BaseModel):
    """Model stage transition request."""
    version: int
    stage: str


# ==================== Model Registry Endpoints ====================

@router.get("/models/registry")
async def list_models(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """List all registered models."""
    try:
        from src.training.mlflow_tracker import list_registered_models
        models = list_registered_models()
        return {
            "models": [
                {
                    "name": m.name,
                    "description": m.description,
                    "latest_versions": len(m.latest_versions) if hasattr(m, 'latest_versions') else 0,
                }
                for m in models
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )


@router.post("/models/registry")
async def create_model(
    model_request: ModelCreateRequest,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Create a new registered model."""
    try:
        from src.training.mlflow_tracker import create_registered_model as create_model_func
        model = create_model_func(
            name=model_request.name,
            description=model_request.description,
            tags=model_request.tags if model_request.tags else None
        )
        if model:
            return {"name": model.name, "status": "created"}
        raise HTTPException(status_code=400, detail="Failed to create model")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")


@router.get("/models/registry/{name}")
async def get_model(
    name: str,
    stage: str = None,
    http_request: Request = None,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Get model versions."""
    try:
        from src.training.mlflow_tracker import get_latest_model_versions
        versions = get_latest_model_versions(name, stage)
        return {
            "name": name,
            "versions": [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                }
                for v in versions
            ] if versions else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model: {str(e)}")


@router.post("/models/registry/{name}/transition")
async def transition_model(
    name: str,
    model_request: ModelTransitionRequest,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Transition model to different stage."""
    try:
        from src.training.mlflow_tracker import transition_model_stage
        result = transition_model_stage(name, model_request.version, model_request.stage)
        if result:
            return {"name": name, "version": model_request.version, "stage": model_request.stage, "status": "success"}
        raise HTTPException(status_code=400, detail="Failed to transition")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to transition: {str(e)}")


@router.delete("/models/registry/{name}")
async def delete_model(
    name: str,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Delete a registered model."""
    try:
        from src.training.mlflow_tracker import delete_registered_model
        success = delete_registered_model(name)
        if success:
            return {"status": "deleted", "name": name}
        raise HTTPException(status_code=400, detail="Failed to delete model")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")
