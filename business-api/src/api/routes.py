"""
Business API Routes
Location: business-api/src/api/routes.py

Contains:
- Data Discovery endpoints
- Task Management endpoints
- Agent orchestration endpoints
"""

from pathlib import Path
from typing import Optional, List
from datetime import datetime
import uuid
import asyncio

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status, Request
from pydantic import BaseModel, Field


# ==================== Request/Response Models ====================

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

class DatasetSearchRequest(BaseModel):
    """Dataset search request."""
    query: str = Field(..., description="Search query")
    max_results: int = Field(10, description="Maximum results")
    sources: Optional[List[str]] = Field(None, description="Data sources to search")
    min_images: Optional[int] = Field(None, description="Minimum images")
    license: Optional[str] = Field(None, description="License filter")


class DatasetSearchResponse(BaseModel):
    """Dataset search response."""
    datasets: List[dict]
    total: int
    query_time_ms: int = 0


class TrainRequest(BaseModel):
    """Training request."""
    model: str = Field("yolo11m", description="Model size (n/s/m/l/x)")
    data_yaml: str = Field(..., description="Path to dataset YAML")
    epochs: int = Field(100, description="Number of epochs")
    imgsz: int = Field(640, description="Image size")
    task_type: str = Field("training", description="Task type: training/hpo")


class TrainResponse(BaseModel):
    """Training response."""
    task_id: str
    status: str
    message: str
    gpu_server: str
    estimated_time_minutes: Optional[int] = None


class TrainStatusResponse(BaseModel):
    """Training status response."""
    task_id: str
    status: str
    progress: float
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
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


class DataAnalysisRequest(BaseModel):
    """Data analysis request for DeepAnalyze."""
    dataset_path: str = Field(..., description="Path to dataset file or directory")
    analysis_type: str = Field("quality", description="Type of analysis: quality, distribution, anomalies, full")
    prompt: Optional[str] = Field(None, description="Custom analysis prompt")


class DataAnalysisResponse(BaseModel):
    """Data analysis response."""
    task_id: str
    status: str
    content: Optional[str] = None
    files: Optional[List[dict]] = None
    error: Optional[str] = None


class ReportRequest(BaseModel):
    """Report generation request."""
    data_description: str = Field(..., description="Description of the data")
    analysis_goals: List[str] = Field(..., description="List of analysis objectives")


class ReportResponse(BaseModel):
    """Report generation response."""
    task_id: str
    status: str
    content: Optional[str] = None
    files: Optional[List[dict]] = None
    error: Optional[str] = None


# ==================== Create Routers ====================

data_router = APIRouter()
train_router = APIRouter()
deploy_router = APIRouter()
callback_router = APIRouter()
analysis_router = APIRouter()


# ==================== Task Callback Endpoints ====================

@callback_router.post("/task/callback")
async def task_callback(request: TaskCallbackRequest):
    """
    Receive callback from Training API when task completes.

    This endpoint is called by the Training API when:
    - Training completes
    - Export completes
    - HPO completes
    - Any task fails
    """
    # Store callback data in Redis or database
    # For now, just log and acknowledge
    print(f"Received callback for task {request.task_id}: {request.status}")

    return TaskCallbackResponse(
        received=True,
        task_id=request.task_id
    )


# ==================== Data Endpoints ====================

@data_router.post("/search", response_model=DatasetSearchResponse)
async def search_datasets(request: DatasetSearchRequest):
    """
    Search for datasets across multiple sources.

    Supported sources: Roboflow, Kaggle, HuggingFace
    """
    import time
    start_time = time.time()

    # Import data discovery module
    try:
        from src.data.discovery import DatasetDiscovery
        discovery = DatasetDiscovery()
        # Run sync discovery.search in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, discovery.search, request.query, request.max_results
        )

        # Filter by sources if specified
        if request.sources:
            results = [r for r in results if r.source in request.sources]

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

        query_time_ms = int((time.time() - start_time) * 1000)

        return DatasetSearchResponse(
            datasets=datasets,
            total=len(datasets),
            query_time_ms=query_time_ms
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


# ==================== Training Endpoints ====================

@train_router.post("/submit", response_model=TrainResponse)
async def submit_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    request_state = None  # Will be replaced with app state
):
    """
    Submit a training job to the GPU server.
    """
    task_id = f"train_{uuid.uuid4().hex[:8]}"

    try:
        # Use training client from app state (initialized in gateway.py)
        client = request.app.state.training_client

        # Submit training task
        result = await client.start_training(
            task_id=task_id,
            model=request.model,
            data_yaml=request.data_yaml,
            epochs=request.epochs,
            imgsz=request.imgsz
        )

        # Estimate training time
        estimated_time = request.epochs * 2  # rough estimate: 2 min per epoch

        return TrainResponse(
            task_id=task_id,
            status="submitted",
            message="Training job submitted to GPU server",
            gpu_server=os.getenv("TRAINING_API_URL", "http://localhost:8001"),
            estimated_time_minutes=estimated_time
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to submit training job: {str(e)}"
        )


@train_router.get("/status/{task_id}", response_model=TrainStatusResponse)
async def get_training_status(task_id: str, request: Request):
    """
    Get training job status from the GPU server.
    """
    try:
        # Use training client from app state (initialized in gateway.py)
        client = request.app.state.training_client

        result = await client.get_task_status(task_id)

        return TrainStatusResponse(
            task_id=result.get("task_id", task_id),
            status=result.get("status", "unknown"),
            progress=result.get("progress", 0.0),
            current_epoch=result.get("current_epoch"),
            total_epochs=result.get("total_epochs"),
            metrics=result.get("metrics"),
            error=result.get("error")
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to get training status: {str(e)}"
        )


@train_router.post("/cancel/{task_id}")
async def cancel_training(task_id: str, request: Request):
    """
    Cancel a running training job.
    """
    try:
        # Use training client from app state (initialized in gateway.py)
        client = request.app.state.training_client

        result = await client.cancel_task(task_id)

        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Training job cancelled"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to cancel training job: {str(e)}"
        )


# ==================== Model Registry Endpoints ====================
# Based on MLflow best practices:
# - Version everything: Use MLflow Model Registry
# - Use stages: Staging, Production, Archived
# - Enable rollbacks
# ============================================================

@train_router.get("/models/registry")
async def list_models():
    """
    List all registered models.
    """
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


class ModelCreateRequest(BaseModel):
    """Create registered model request."""
    name: str
    description: str = ""
    tags: Optional[dict] = {}


@train_router.post("/models/registry")
async def create_model(request: ModelCreateRequest):
    """
    Create a new registered model.
    """
    try:
        from src.training.mlflow_tracker import create_registered_model as create_model
        model = create_model(
            name=request.name,
            description=request.description,
            tags=request.tags if request.tags else None
        )
        if model:
            return {"name": model.name, "status": "created"}
        raise HTTPException(status_code=400, detail="Failed to create model")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")


@train_router.get("/models/registry/{name}")
async def get_model(name: str, stage: Optional[str] = None):
    """
    Get model versions.
    """
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


class ModelTransitionRequest(BaseModel):
    """Model stage transition request."""
    version: int
    stage: str


@train_router.post("/models/registry/{name}/transition")
async def transition_model(name: str, request: ModelTransitionRequest):
    """
    Transition model to different stage.
    """
    try:
        from src.training.mlflow_tracker import transition_model_stage
        result = transition_model_stage(name, request.version, request.stage)
        if result:
            return {"name": name, "version": request.version, "stage": request.stage, "status": "success"}
        raise HTTPException(status_code=400, detail="Failed to transition")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to transition: {str(e)}")


@train_router.delete("/models/registry/{name}")
async def delete_model(name: str):
    """
    Delete a registered model.
    """
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


# ==================== Export Endpoints ====================

@deploy_router.post("/export", response_model=ExportResponse)
async def export_model(request: ExportRequest):
    """
    Submit a model export job to the GPU server.
    """
    task_id = f"export_{uuid.uuid4().hex[:8]}"

    try:
        # Use training client from app state (initialized in gateway.py)
        client = request.app.state.training_client

        result = await client.start_export(
            task_id=task_id,
            model_path=request.model_path,
            platform=request.platform,
            imgsz=request.imgsz
        )

        return ExportResponse(
            task_id=task_id,
            status="submitted",
            message="Export job submitted to GPU server"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to submit export job: {str(e)}"
        )


@deploy_router.get("/export/status/{task_id}")
async def get_export_status(task_id: str, request: Request):
    """
    Get export job status.
    """
    try:
        # Use training client from app state (initialized in gateway.py)
        client = request.app.state.training_client

        result = await client.get_task_status(task_id)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to get export status: {str(e)}"
        )


# ==================== Data Analysis Endpoints (DeepAnalyze) ====================

@analysis_router.post("/health")
async def check_analysis_api():
    """
    Check if DeepAnalyze API is available.
    """
    try:
        from .deepanalyze_client import DeepAnalyzeClient
        import os

        client = DeepAnalyzeClient(
            base_url=os.getenv("DEEPANALYZE_API_URL", "http://localhost:8200/v1"),
            api_key=os.getenv("DEEPANALYZE_API_KEY")
        )

        available = client.health_check()

        return {
            "status": "available" if available else "unavailable",
            "service": "DeepAnalyze"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@analysis_router.post("/analyze", response_model=DataAnalysisResponse)
async def analyze_dataset(request: DataAnalysisRequest):
    """
    Analyze dataset using DeepAnalyze.

    Supports:
    - quality: Data quality analysis (missing values, outliers, duplicates)
    - distribution: Data distribution analysis (statistics, correlations)
    - anomalies: Anomaly detection
    - full: Comprehensive analysis
    """
    task_id = f"analyze_{uuid.uuid4().hex[:8]}"

    try:
        from .deepanalyze_client import DeepAnalyzeClient
        import os

        client = DeepAnalyzeClient(
            base_url=os.getenv("DEEPANALYZE_API_URL", "http://localhost:8200/v1"),
            api_key=os.getenv("DEEPANALYZE_API_KEY")
        )

        # Check if API is available
        if not client.health_check():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="DeepAnalyze API is not available. Please ensure the service is running."
            )

        # Perform analysis
        result = client.analyze_dataset(
            dataset_path=request.dataset_path,
            analysis_type=request.analysis_type
        )

        if "error" in result:
            return DataAnalysisResponse(
                task_id=task_id,
                status="failed",
                error=result["error"]
            )

        return DataAnalysisResponse(
            task_id=task_id,
            status="completed",
            content=result.get("content"),
            files=result.get("files")
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@analysis_router.post("/report", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """
    Generate a comprehensive data science report using DeepAnalyze.

    Args:
        data_description: Description of the data to analyze
        analysis_goals: List of analysis objectives
    """
    task_id = f"report_{uuid.uuid4().hex[:8]}"

    try:
        from .deepanalyze_client import DeepAnalyzeClient
        import os

        client = DeepAnalyzeClient(
            base_url=os.getenv("DEEPANALYZE_API_URL", "http://localhost:8200/v1"),
            api_key=os.getenv("DEEPANALYZE_API_KEY")
        )

        # Check if API is available
        if not client.health_check():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="DeepAnalyze API is not available. Please ensure the service is running."
            )

        # Generate report
        result = client.generate_report(
            data_description=request.data_description,
            analysis_goals=request.analysis_goals
        )

        if "error" in result:
            return ReportResponse(
                task_id=task_id,
                status="failed",
                error=result["error"]
            )

        return ReportResponse(
            task_id=task_id,
            status="completed",
            content=result.get("content"),
            files=result.get("files")
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}"
        )
