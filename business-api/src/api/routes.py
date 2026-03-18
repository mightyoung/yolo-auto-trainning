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
import json
import os

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status, Request
from pydantic import BaseModel, Field

# Import authentication from auth module
from .auth import get_current_user, CurrentUser, create_access_token, check_rate_limit
from .audit import audit_logger


# ==================== Task Storage Utilities ====================

def get_redis_client(request: Request):
    """Get Redis client from app state."""
    return request.app.state.redis


def store_task_in_redis(redis_client, task_data: dict) -> None:
    """Store task in Redis with user_id index."""
    if redis_client is None:
        return

    task_id = task_data["task_id"]
    user_id = task_data["user_id"]

    # Store task data
    redis_client.set(
        f"task:{task_id}",
        json.dumps(task_data),
        ex=7 * 24 * 60 * 60  # 7 days TTL
    )

    # Add to user's task index
    redis_client.sadd(f"user:{user_id}:tasks", task_id)


def get_task_from_redis(redis_client, task_id: str) -> Optional[dict]:
    """Get task from Redis."""
    if redis_client is None:
        return None

    data = redis_client.get(f"task:{task_id}")
    if data:
        return json.loads(data)
    return None


def get_user_tasks_from_redis(redis_client, user_id: str) -> List[dict]:
    """Get all tasks for a user from Redis."""
    if redis_client is None:
        return []

    task_ids = redis_client.smembers(f"user:{user_id}:tasks")
    tasks = []

    for task_id in task_ids:
        task_data = get_task_from_redis(redis_client, task_id)
        if task_data:
            tasks.append(task_data)

    # Sort by created_at descending
    tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return tasks


def verify_task_ownership(redis_client, task_id: str, user_id: str) -> Optional[dict]:
    """Verify that a task belongs to the user. Returns task if owned, None otherwise."""
    task = get_task_from_redis(redis_client, task_id)
    if task is None:
        return None

    if task.get("user_id") != user_id:
        return None

    return task


def delete_task_from_redis(redis_client, task_id: str, user_id: str) -> bool:
    """Delete a task from Redis if owned by user."""
    task = verify_task_ownership(redis_client, task_id, user_id)
    if task is None:
        return False

    redis_client.delete(f"task:{task_id}")
    redis_client.srem(f"user:{user_id}:tasks", task_id)
    return True


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


class TaskListResponse(BaseModel):
    """Task list response with user isolation."""
    tasks: List[dict]
    total: int


class TaskDetailResponse(BaseModel):
    """Task detail response."""
    task_id: str
    task_type: str
    status: str
    user_id: str
    created_at: str
    progress: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None


# ==================== Create Routers ====================

data_router = APIRouter()
train_router = APIRouter()
deploy_router = APIRouter()
callback_router = APIRouter()
analysis_router = APIRouter()


# ==================== Task Callback Endpoints ====================

@callback_router.post("/task/callback")
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

    Supports both JWT and API key authentication.
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
async def search_datasets(
    request: DatasetSearchRequest,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Search for datasets across multiple sources.

    Supported sources: Roboflow, Kaggle, HuggingFace

    Requires authentication.
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

        # Log data access
        audit_logger.log_data_access(
            user_id=current_user.user_id,
            dataset_id=request.query,
            action="search",
            request=http_request,
            details={"query": request.query, "max_results": request.max_results, "sources": request.sources}
        )

        return DatasetSearchResponse(
            datasets=datasets,
            total=len(datasets),
            query_time_ms=query_time_ms
        )

    except Exception as e:
        # Log failed search
        audit_logger.log_data_access(
            user_id=current_user.user_id,
            dataset_id=request.query,
            action="search_failed",
            request=http_request,
            details={"query": request.query, "error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


# ==================== Training Endpoints ====================

@train_router.post("/submit", response_model=TrainResponse)
async def submit_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Submit a training job to the GPU server.

    Requires authentication.
    """
    task_id = f"train_{uuid.uuid4().hex[:8]}"

    try:
        # Use training client from app state (initialized in gateway.py)
        client = http_request.app.state.training_client

        # Submit training task
        result = await client.start_training(
            task_id=task_id,
            model=request.model,
            data_yaml=request.data_yaml,
            epochs=request.epochs,
            imgsz=request.imgsz
        )

        # Store task in Redis with user_id for isolation
        task_data = {
            "task_id": task_id,
            "task_type": "training",
            "user_id": current_user.user_id,
            "status": "submitted",
            "created_at": datetime.now().isoformat(),
            "params": {
                "model": request.model,
                "data_yaml": request.data_yaml,
                "epochs": request.epochs,
                "imgsz": request.imgsz
            }
        }
        redis_client = get_redis_client(http_request)
        store_task_in_redis(redis_client, task_data)

        # Estimate training time
        estimated_time = request.epochs * 2  # rough estimate: 2 min per epoch

        # Log training submission
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

    except Exception as e:
        # Log failed training submission
        audit_logger.log_training(
            user_id=current_user.user_id,
            action="submit_failed",
            task_id=task_id,
            request=http_request,
            details={"model": request.model, "epochs": request.epochs, "error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to submit training job: {str(e)}"
        )


@train_router.get("/status/{task_id}", response_model=TrainStatusResponse)
async def get_training_status(
    task_id: str,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Get training job status from the GPU server.

    Requires authentication. Verifies task ownership.
    """
    try:
        # Verify task ownership
        redis_client = get_redis_client(http_request)
        task = verify_task_ownership(redis_client, task_id, current_user.user_id)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found or not authorized"
            )

        # Use training client from app state (initialized in gateway.py)
        client = http_request.app.state.training_client

        result = await client.get_task_status(task_id)

        # Log status check
        audit_logger.log_training(
            user_id=current_user.user_id,
            action="status_check",
            task_id=task_id,
            request=request,
            details={"status": result.get("status")}
        )

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
async def cancel_training(
    task_id: str,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Cancel a running training job.

    Requires authentication. Verifies task ownership.
    """
    try:
        # Verify task ownership
        redis_client = get_redis_client(http_request)
        task = verify_task_ownership(redis_client, task_id, current_user.user_id)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found or not authorized"
            )

        # Use training client from app state (initialized in gateway.py)
        client = http_request.app.state.training_client

        result = await client.cancel_task(task_id)

        # Log cancellation
        audit_logger.log_training(
            user_id=current_user.user_id,
            action="cancel",
            task_id=task_id,
            request=request
        )

        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Training job cancelled"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to cancel training job: {str(e)}"
        )


@train_router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    List all tasks for the current user.

    Requires authentication. Returns only tasks owned by the current user.
    """
    redis_client = get_redis_client(http_request)
    tasks = get_user_tasks_from_redis(redis_client, current_user.user_id)

    return TaskListResponse(
        tasks=tasks,
        total=len(tasks)
    )


@train_router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Delete a task.

    Requires authentication. Verifies task ownership.
    """
    redis_client = get_redis_client(http_request)

    # Verify ownership and delete
    success = delete_task_from_redis(redis_client, task_id, current_user.user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found or not authorized"
        )

    return {
        "task_id": task_id,
        "status": "deleted",
        "message": "Task deleted successfully"
    }


# ==================== Model Registry Endpoints ====================
# Based on MLflow best practices:
# - Version everything: Use MLflow Model Registry
# - Use stages: Staging, Production, Archived
# - Enable rollbacks
# ============================================================

@train_router.get("/models/registry")
async def list_models(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    List all registered models.

    Requires authentication.
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
async def create_model(
    request: ModelCreateRequest,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Create a new registered model.

    Requires authentication.
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
async def get_model(
    name: str,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Get model versions.

    Requires authentication.
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
async def transition_model(
    name: str,
    request: ModelTransitionRequest,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Transition model to different stage.

    Requires authentication.
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
async def delete_model(
    name: str,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Delete a registered model.

    Requires authentication.
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
async def export_model(
    request: ExportRequest,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Submit a model export job to the GPU server.

    Requires authentication.
    """
    task_id = f"export_{uuid.uuid4().hex[:8]}"

    try:
        # Use training client from app state (initialized in gateway.py)
        client = http_request.app.state.training_client

        result = await client.start_export(
            task_id=task_id,
            model_path=request.model_path,
            platform=request.platform,
            imgsz=request.imgsz
        )

        # Store task in Redis with user_id for isolation
        task_data = {
            "task_id": task_id,
            "task_type": "export",
            "user_id": current_user.user_id,
            "status": "submitted",
            "created_at": datetime.now().isoformat(),
            "params": {
                "model_path": request.model_path,
                "platform": request.platform,
                "imgsz": request.imgsz
            }
        }
        redis_client = get_redis_client(http_request)
        store_task_in_redis(redis_client, task_data)

        # Log export submission
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

    except Exception as e:
        # Log failed export
        audit_logger.log(
            action="export",
            user_id=current_user.user_id,
            resource=f"export/{task_id}",
            request=http_request,
            details={"model_path": request.model_path, "platform": request.platform, "error": str(e)},
            status="failure"
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to submit export job: {str(e)}"
        )


@deploy_router.get("/export/status/{task_id}")
async def get_export_status(
    task_id: str,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Get export job status.

    Requires authentication. Verifies task ownership.
    """
    try:
        # Verify task ownership
        redis_client = get_redis_client(http_request)
        task = verify_task_ownership(redis_client, task_id, current_user.user_id)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found or not authorized"
            )

        # Use training client from app state (initialized in gateway.py)
        client = http_request.app.state.training_client

        result = await client.get_task_status(task_id)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to get export status: {str(e)}"
        )


# ==================== Data Analysis Endpoints (DeepAnalyze) ====================

@analysis_router.post("/health")
async def check_analysis_api(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Check if DeepAnalyze API is available.

    Requires authentication.
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
async def analyze_dataset(
    request: DataAnalysisRequest,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Analyze dataset using DeepAnalyze.

    Supports:
    - quality: Data quality analysis (missing values, outliers, duplicates)
    - distribution: Data distribution analysis (statistics, correlations)
    - anomalies: Anomaly detection
    - full: Comprehensive analysis

    Requires authentication.
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

        # Store task in Redis with user_id for isolation
        task_data = {
            "task_id": task_id,
            "task_type": "analysis",
            "user_id": current_user.user_id,
            "status": "completed" if "error" not in result else "failed",
            "created_at": datetime.now().isoformat(),
            "params": {
                "dataset_path": request.dataset_path,
                "analysis_type": request.analysis_type
            },
            "result": result if "error" not in result else None,
            "error": result.get("error") if "error" in result else None
        }
        redis_client = get_redis_client(http_request)
        store_task_in_redis(redis_client, task_data)

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
async def generate_report(
    request: ReportRequest,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Generate a comprehensive data science report using DeepAnalyze.

    Args:
        data_description: Description of the data to analyze
        analysis_goals: List of analysis objectives

    Requires authentication.
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
