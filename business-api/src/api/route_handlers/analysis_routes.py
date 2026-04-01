"""Analysis routes for Business API."""

import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..auth import CurrentUser, check_rate_limit, get_current_user
from ..exceptions import ExternalDependencyError
from ..task_registry import (
    build_task_record,
    store_task_in_redis,
)

router = APIRouter()


class DataAnalysisRequest(BaseModel):
    """Data analysis request for DeepAnalyze."""
    dataset_path: str = Field(..., description="Path to dataset file or directory")
    analysis_type: str = Field("quality", description="Type of analysis: quality, distribution, anomalies, full")
    prompt: str | None = Field(None, description="Custom analysis prompt")


class DataAnalysisResponse(BaseModel):
    """Data analysis response."""
    task_id: str
    status: str
    content: str | None = None
    files: list[dict] | None = None
    error: str | None = None


class ReportRequest(BaseModel):
    """Report generation request."""
    data_description: str = Field(..., description="Description of the data")
    analysis_goals: list[str] = Field(..., description="List of analysis objectives")


class ReportResponse(BaseModel):
    """Report generation response."""
    task_id: str
    status: str
    content: str | None = None
    files: list[dict] | None = None
    error: str | None = None


def get_redis_client(request: Request):
    """Get Redis client from request app state."""
    return request.app.state.redis


@router.post("/health")
async def check_analysis_api(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Check if DeepAnalyze API is available."""
    try:
        from ..deepanalyze_client import DeepAnalyzeClient

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


@router.post("/analyze", response_model=DataAnalysisResponse)
async def analyze_dataset(
    request: DataAnalysisRequest,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Analyze dataset using DeepAnalyze."""
    task_id = f"analyze_{uuid.uuid4().hex[:8]}"

    try:
        from ..deepanalyze_client import DeepAnalyzeClient

        client = DeepAnalyzeClient(
            base_url=os.getenv("DEEPANALYZE_API_URL", "http://localhost:8200/v1"),
            api_key=os.getenv("DEEPANALYZE_API_KEY")
        )

        if not client.health_check():
            raise HTTPException(
                status_code=503,
                detail="DeepAnalyze API is not available. Please ensure the service is running."
            )

        result = client.analyze_dataset(
            dataset_path=request.dataset_path,
            analysis_type=request.analysis_type
        )

        task_data = build_task_record(
            task_id=task_id,
            task_type="analysis",
            user_id=current_user.user_id,
            registry_status="completed" if "error" not in result else "failed",
            submission={
                "dataset_path": request.dataset_path,
                "analysis_type": request.analysis_type
            },
        )
        task_data.update({
            "result": result if "error" not in result else None,
            "error": result.get("error") if "error" in result else None
        })
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
    except ExternalDependencyError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/report", response_model=ReportResponse)
async def generate_report(
    request: ReportRequest,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Generate a comprehensive data science report using DeepAnalyze."""
    task_id = f"report_{uuid.uuid4().hex[:8]}"

    try:
        from ..deepanalyze_client import DeepAnalyzeClient

        client = DeepAnalyzeClient(
            base_url=os.getenv("DEEPANALYZE_API_URL", "http://localhost:8200/v1"),
            api_key=os.getenv("DEEPANALYZE_API_KEY")
        )

        if not client.health_check():
            raise HTTPException(
                status_code=503,
                detail="DeepAnalyze API is not available. Please ensure the service is running."
            )

        result = client.generate_report(
            data_description=request.data_description,
            analysis_goals=request.analysis_goals
        )

        task_data = build_task_record(
            task_id=task_id,
            task_type="report",
            user_id=current_user.user_id,
            registry_status="completed" if "error" not in result else "failed",
            submission={
                "data_description": request.data_description,
                "analysis_goals": request.analysis_goals,
            },
        )
        task_data.update({
            "result": result if "error" not in result else None,
            "error": result.get("error") if "error" in result else None,
        })
        redis_client = get_redis_client(http_request)
        store_task_in_redis(redis_client, task_data)

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
    except ExternalDependencyError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}"
        )
