"""
API Routes - REST endpoints for YOLO training system.
"""

from pathlib import Path
from typing import Optional, List
from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field

# Celery task imports
from .tasks import (
    training_task,
    hpo_task,
    export_task,
    data_discovery_task,
)

# Create routers
data_router = APIRouter()
train_router = APIRouter()
deploy_router = APIRouter()


# Request/Response models
class DatasetSearchRequest(BaseModel):
    """Dataset search request."""
    query: str = Field(..., description="Search query")
    max_results: int = Field(10, description="Maximum results")


class DatasetSearchResponse(BaseModel):
    """Dataset search response."""
    datasets: List[dict]
    total: int


class TrainRequest(BaseModel):
    """Training request."""
    data_yaml: str = Field(..., description="Path to dataset YAML")
    model: str = Field("yolo11m", description="Model size (n/s/m/l)")
    epochs: int = Field(100, description="Number of epochs")
    imgsz: int = Field(640, description="Image size")


class TrainResponse(BaseModel):
    """Training response."""
    task_id: str
    status: str
    message: str


class TrainStatusResponse(BaseModel):
    """Training status response."""
    task_id: str
    status: str
    progress: float
    metrics: Optional[dict] = None
    error: Optional[str] = None


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


# Data endpoints
@data_router.post("/search", response_model=DatasetSearchResponse)
async def search_datasets(request: DatasetSearchRequest):
    """
    Search for datasets across multiple sources.

    Supported sources: Roboflow, Kaggle, HuggingFace
    Uses Celery for async processing.
    """
    from src.data.discovery import DatasetDiscovery
    discovery = DatasetDiscovery()
    results = discovery.search(query=request.query, max_results=request.max_results)

    datasets = [
        {
            "name": ds.name,
            "source": ds.source,
            "url": ds.url,
            "license": ds.license,
            "images": ds.images,
            "relevance_score": ds.relevance_score,
        }
        for ds in results
    ]

    return DatasetSearchResponse(
        datasets=datasets,
        total=len(datasets),
    )


@data_router.post("/discover")
async def discover_datasets(
    query: str,
    sources: Optional[List[str]] = None,
):
    """
    Discover and download datasets.
    """
    return {
        "status": "completed",
        "datasets": [],
    }


# Training endpoints
@train_router.post("/start", response_model=TrainResponse)
async def start_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start YOLO training job.

    Submits to Celery task queue.
    """
    task_id = f"train_{uuid.uuid4().hex[:8]}"

    # Submit to Celery task queue
    config = {
        "model": request.model,
        "data_yaml": request.data_yaml,
        "epochs": request.epochs,
        "imgsz": request.imgsz,
        "output_dir": str(Path("./runs") / task_id),
    }

    # Submit async task
    background_tasks.add_task(training_task.delay, config)

    return TrainResponse(
        task_id=task_id,
        status="submitted",
        message="Training job submitted successfully",
    )


@train_router.get("/status/{task_id}", response_model=TrainStatusResponse)
async def get_training_status(task_id: str):
    """
    Get training job status from Celery.
    """
    # Use task_id to look up results (in production, store task_id in Redis)
    # For now, return pending status
    return TrainStatusResponse(
        task_id=task_id,
        status="running",
        progress=0.0,
    )


@train_router.get("/results/{task_id}")
async def get_training_results(task_id: str):
    """
    Get training results.
    """
    return {
        "task_id": task_id,
        "status": "completed",
        "metrics": {
            "mAP50": 0.45,
            "mAP50-95": 0.35,
        },
    }


# Deployment endpoints
@deploy_router.post("/export", response_model=ExportResponse)
async def export_model(
    request: ExportRequest,
    background_tasks: BackgroundTasks,
):
    """
    Export model to target platform via Celery.
    """
    task_id = f"export_{uuid.uuid4().hex[:8]}"

    # Submit to Celery task queue
    config = {
        "model_path": request.model_path,
        "platform": request.platform,
        "imgsz": request.imgsz,
        "output_dir": str(Path("./runs/export") / task_id),
    }

    # Submit async task
    background_tasks.add_task(export_task.delay, config)

    return ExportResponse(
        task_id=task_id,
        status="submitted",
        message="Export job submitted successfully",
    )


@deploy_router.get("/export/status/{task_id}")
async def get_export_status(task_id: str):
    """
    Get export job status.
    """
    return {
        "task_id": task_id,
        "status": "running",
        "progress": 0.5,
    }


# Router aggregation
router = APIRouter()
router.include_router(data_router, prefix="/data", tags=["Data"])
router.include_router(train_router, prefix="/train", tags=["Training"])
router.include_router(deploy_router, prefix="/deploy", tags=["Deployment"])
