"""Task-facing request/response models for the Business API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrainStatusResponse(BaseModel):
    """Training status response."""

    task_id: str
    status: str
    progress: float
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    metrics: Optional[dict] = None
    error: Optional[str] = None
    live_mAP50: Optional[float] = None
    lr_decay_triggered: Optional[bool] = None
    lr_decay_signal: Optional[dict] = None
    augment_boost_active: Optional[bool] = None
    augment_boost_signal: Optional[dict] = None
    data_expansion_requested: Optional[bool] = None
    data_expansion_signal: Optional[dict] = None
    strategies_triggered: Optional[list] = None
    resubmit_count: Optional[int] = None
    last_resubmitted_at: Optional[str] = None
    resubmit_reason: Optional[str] = None


class ExportStatusResponse(BaseModel):
    """Export status response."""

    task_id: str
    status: str
    progress: float = 0.0
    model_path: Optional[str] = None
    platform: Optional[str] = None
    imgsz: Optional[int] = None
    formats: Optional[List[str]] = None
    int8_quantize: Optional[bool] = None
    export_path: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class TaskExecutionSummaryResponse(BaseModel):
    """Stable execution summary for list/detail views."""

    status: str
    progress: Optional[float] = None
    updated_at: Optional[str] = None
    error: Optional[str] = None


class TaskRecordResponse(BaseModel):
    """Normalized business task registry record."""

    task_id: str
    task_type: str
    user_id: str
    created_at: str
    status: str
    registry_status: str
    submission: Dict[str, Any] = Field(default_factory=dict)
    links: Dict[str, Any] = Field(default_factory=dict)
    execution_summary: TaskExecutionSummaryResponse
    execution: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    result_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TaskListResponse(BaseModel):
    """Task list response with user isolation."""

    tasks: List[TaskRecordResponse]
    total: int


class TaskDetailResponse(BaseModel):
    """Task detail response."""

    task: TaskRecordResponse
