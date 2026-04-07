"""Continuous training pipeline routes.

This module holds the pipeline endpoints that were split out of the legacy
_routes_impl.py file to keep the primary training route module smaller.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from ..gateway import check_rate_limit, get_redis_client, verify_internal_api_key

router = APIRouter()

_continous_pipeline_instance = None


def _get_continuous_pipeline():
    global _continous_pipeline_instance
    if _continous_pipeline_instance is None:
        from src.pipeline.continuous_training import ContinuousTrainingPipeline

        redis_client = get_redis_client()
        _continous_pipeline_instance = ContinuousTrainingPipeline(redis_client=redis_client)
    return _continous_pipeline_instance


class ContinuousTrainingRequest(BaseModel):
    model_name: str = Field("yolo11m", description="Base model to fine-tune")
    task_id: str = Field(..., description="Task identifier for the new training run")
    drift_threshold: float = Field(0.05, description="Fractional mAP drop that triggers retrain (0.0-1.0)")
    ab_test_duration_hours: int = Field(24, description="A/B test duration in hours")
    output_dir: str = Field("/home/wangxin/runs", description="Output directory for model artifacts")


class DriftCheckRequest(BaseModel):
    current_map: float = Field(..., description="Current production model mAP (0.0-1.0)")
    historical_avg: float = Field(..., description="Long-running average mAP (0.0-1.0)")
    threshold: float | None = Field(None, description="Override drift threshold")


class ABTestStartRequest(BaseModel):
    model_a: str = Field(..., description="Production (control) model path or name")
    model_b: str = Field(..., description="Candidate model path or name")
    duration_hours: int = Field(24, description="Test duration in hours")
    min_samples: int = Field(100, description="Minimum inference samples before evaluating")


class RollbackRequest(BaseModel):
    current_model: str | None = Field(None, description="Model to replace")
    previous_model: str | None = Field(None, description="Model to restore")


@router.post("/pipeline/continuous/start")
async def start_continuous_training(
    request: ContinuousTrainingRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit),
):
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    pipeline = _get_continuous_pipeline()
    pipeline_id = pipeline.start_retrain_pipeline(
        task_id=request.task_id,
        model_name=request.model_name,
        output_dir=request.output_dir,
    )
    return {
        "pipeline_id": pipeline_id,
        "task_id": request.task_id,
        "model_name": request.model_name,
        "status": "started",
        "message": f"Continuous training pipeline started for model {request.model_name}",
    }


@router.get("/pipeline/continuous/status")
async def get_continuous_training_status(
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit),
):
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return _get_continuous_pipeline().get_status()


@router.post("/pipeline/continuous/drift-check")
async def check_drift(
    request: DriftCheckRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit),
):
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    pipeline = _get_continuous_pipeline()
    decision = pipeline.check_drift_and_decide(
        current_map=request.current_map,
        historical_avg=request.historical_avg,
        threshold=request.threshold,
    )
    return {"action": decision.action, "drift_score": decision.drift_score, "message": decision.message}


@router.post("/pipeline/continuous/ab-test")
async def start_ab_test(
    request: ABTestStartRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit),
):
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    pipeline = _get_continuous_pipeline()
    pipeline._transition_stage(pipeline.STAGE_AB_TESTING)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        pipeline.run_ab_test,
        request.model_a,
        request.model_b,
        request.duration_hours,
        request.min_samples,
    )
    return {
        "status": "started",
        "model_a": request.model_a,
        "model_b": request.model_b,
        "duration_hours": request.duration_hours,
        "message": f"A/B test started: {request.model_a} vs {request.model_b}",
    }


@router.get("/pipeline/continuous/ab-test/result")
async def get_ab_test_result(
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit),
):
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    pipeline = _get_continuous_pipeline()
    results = pipeline._ab_test_results
    if not results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No A/B test results available")
    return results[-1].to_dict()


@router.post("/pipeline/continuous/rollback")
async def rollback_model(
    request: RollbackRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit),
):
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    pipeline = _get_continuous_pipeline()
    success = pipeline.auto_rollback(current_model=request.current_model, previous_model=request.previous_model)
    if success:
        return {"status": "rolled_back", "message": f"Rolled back to {request.previous_model or 'production model'}"}
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Rollback failed")


@router.post("/pipeline/continuous/promote")
async def promote_candidate(
    candidate_model: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit),
):
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    pipeline = _get_continuous_pipeline()
    pipeline.promote_candidate(candidate_model)
    return {"status": "promoted", "candidate_model": candidate_model, "message": f"Model {candidate_model} promoted to production"}


@router.post("/pipeline/continuous/reset")
async def reset_continuous_pipeline(
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit),
):
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    pipeline = _get_continuous_pipeline()
    pipeline.reset()
    return {"status": "reset", "message": "Continuous training pipeline reset to idle"}
