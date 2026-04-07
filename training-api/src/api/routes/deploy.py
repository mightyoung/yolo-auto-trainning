"""Deployment and drift-monitoring routes for the Training API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from ..gateway import check_rate_limit, verify_internal_api_key

router = APIRouter()


class DriftCheckRequest(BaseModel):
    """Data drift check request."""

    model_name: str = Field(..., description="Model name being monitored")
    reference_image_dir: str = Field(..., description="Path to reference (training) image directory")
    current_image_dir: str = Field(..., description="Path to current production image directory")
    metrics_history: list[float] | None = Field(
        None,
        description="Optional historical mAP values for concept drift detection",
    )
    psi_threshold: float = Field(0.2, description="PSI threshold for data drift (default 0.2)")


class DriftResponse(BaseModel):
    """Drift detection response."""

    model_name: str
    data_drift_score: float
    concept_drift_detected: bool
    recommendation: str
    feature_drift: dict[str, float]
    timestamp: str


class EdgeConfigResponse(BaseModel):
    """Edge device inference configuration response."""

    device: str
    model_path: str
    batch_size: int
    stream_count: int
    workspace_mb: int
    recommended_format: str
    fallback_formats: list[str]
    precision: str
    dynamic_batch: bool
    imgsz: int
    export_kwargs: dict[str, Any]
    notes: list[str]


@router.post("/monitor/drift-check", response_model=DriftResponse)
async def check_drift(
    request: DriftCheckRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit),
):
    """Detect data and concept drift for a deployed model."""
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    ref_dir = Path(request.reference_image_dir)
    cur_dir = Path(request.current_image_dir)

    if not ref_dir.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Reference image directory not found: {ref_dir}")
    if not cur_dir.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Current image directory not found: {cur_dir}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ref_images = [str(p) for p in ref_dir.iterdir() if p.suffix.lower() in image_exts]
    cur_images = [str(p) for p in cur_dir.iterdir() if p.suffix.lower() in image_exts]

    if not ref_images:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"No images found in reference directory: {ref_dir}")
    if not cur_images:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"No images found in current directory: {cur_dir}")

    monitoring_dir = Path(__file__).parent.parent / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)

    from src.monitoring.drift_detector import DriftDetector

    detector = DriftDetector(psi_threshold=request.psi_threshold)
    report = detector.check_drift(
        model_name=request.model_name,
        reference_images=ref_images,
        current_images=cur_images,
        metrics_history=request.metrics_history,
    )

    return DriftResponse(
        model_name=request.model_name,
        data_drift_score=report.data_drift_score,
        concept_drift_detected=report.concept_drift_detected,
        recommendation=report.recommendation,
        feature_drift=report.feature_drift,
        timestamp=report.timestamp,
    )


@router.get("/deploy/edge-config/{model_name}", response_model=EdgeConfigResponse)
async def get_edge_config(
    model_name: str,
    device: str = "jetson_orin",
    imgsz: int = 640,
    http_request: Request = None,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit),
):
    """Generate optimal inference configuration for a target edge device."""
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    from src.deployment.edge_config import EdgeProfileGenerator

    generator = EdgeProfileGenerator()
    config = generator.generate_config(
        device=device,
        model_path=model_name,
        imgsz=imgsz,
    )

    return EdgeConfigResponse(**config)


@router.get("/deploy/edge-devices")
async def list_edge_devices(
    http_request: Request = None,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit),
):
    """List all supported edge device profiles."""
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    from src.deployment.edge_config import EdgeProfileGenerator

    generator = EdgeProfileGenerator()
    return {
        "devices": generator.list_devices(),
        "note": "Use GET /deploy/edge-config/{model_name}?device=<device> to get full config",
    }
