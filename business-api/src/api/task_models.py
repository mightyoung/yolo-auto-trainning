"""Task-facing request/response models for the Business API."""

from typing import Any

from pydantic import BaseModel, Field


class TrainStatusResponse(BaseModel):
    """Training status response."""

    task_id: str
    status: str
    progress: float
    current_epoch: int | None = None
    total_epochs: int | None = None
    metrics: dict | None = None
    error: str | None = None
    live_mAP50: float | None = None
    lr_decay_triggered: bool | None = None
    lr_decay_signal: dict | None = None
    augment_boost_active: bool | None = None
    augment_boost_signal: dict | None = None
    data_expansion_requested: bool | None = None
    data_expansion_signal: dict | None = None
    strategies_triggered: list | None = None
    resubmit_count: int | None = None
    last_resubmitted_at: str | None = None
    resubmit_reason: str | None = None
    output_path: str | None = None
    output_offset: int | None = None
    output_summary: str | None = None
    output_capped: bool | None = None


class ExportStatusResponse(BaseModel):
    """Export status response."""

    task_id: str
    status: str
    progress: float = 0.0
    model_path: str | None = None
    platform: str | None = None
    imgsz: int | None = None
    formats: list[str] | None = None
    int8_quantize: bool | None = None
    export_path: str | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    output_path: str | None = None
    output_offset: int | None = None
    output_summary: str | None = None
    output_capped: bool | None = None


class TaskExecutionSummaryResponse(BaseModel):
    """Stable execution summary for list/detail views."""

    status: str
    progress: float | None = None
    updated_at: str | None = None
    error: str | None = None


class TaskRecordResponse(BaseModel):
    """Normalized business task registry record."""

    task_id: str
    task_type: str
    user_id: str
    created_at: str
    status: str
    registry_status: str
    submission: dict[str, Any] = Field(default_factory=dict)
    links: dict[str, Any] = Field(default_factory=dict)
    execution_summary: TaskExecutionSummaryResponse
    execution: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    result_summary: dict[str, Any] | None = None
    strategy_ledger: list[dict[str, Any]] = Field(default_factory=list)
    output_path: str | None = None
    output_offset: int | None = None
    output_summary: str | None = None
    output_capped: bool | None = None
    error: str | None = None


class TaskListResponse(BaseModel):
    """Task list response with user isolation."""

    tasks: list[TaskRecordResponse]
    total: int


class TaskDetailResponse(BaseModel):
    """Task detail response."""

    task: TaskRecordResponse
