"""
Agent Orchestration Routes
Location: business-api/src/api/agent_routes.py

Contains:
- CrewAI Agent workflow endpoints
- Task orchestration endpoints
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, status, Depends, Request
from pydantic import BaseModel, Field
import uuid
from datetime import datetime
import sys
from pathlib import Path

# Import authentication from auth module
from .auth import get_current_user, CurrentUser, check_rate_limit

# Add project root for orchestration import
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ==================== Request/Response Models ====================

class AgentTaskRequest(BaseModel):
    """Agent task request."""
    task: str = Field(..., description="Task description")
    context: Optional[dict] = Field(None, description="Additional context")
    agents: Optional[List[str]] = Field(
        None,
        description="Specific agents to use"
    )
    auto_confirm: bool = Field(
        False,
        description="If true, bypasses HiTL human confirmation gates for fully autonomous training"
    )


class AgentTaskResponse(BaseModel):
    """Agent task response."""
    task_id: str
    status: str
    result: Optional[dict] = None
    message: str


class AgentStatusResponse(BaseModel):
    """Agent status response."""
    task_id: str
    status: str
    current_agent: Optional[str] = None
    progress: float = 0.0
    result: Optional[dict] = None


class TaskRequest(BaseModel):
    task: str


class AgentTaskResponse(BaseModel):
    task_id: str
    status: str
    message: str = ""


class AgentStatusResponse(BaseModel):
    status: str
    progress: float
    current_agent: str = ""
    phase1_result: str = ""
    error: Optional[str] = None


class ConfirmRequest(BaseModel):
    approved: bool = True
    overrides: Optional[dict] = None


# ==================== Create Router ====================

agent_router = APIRouter()


# ==================== Agent Endpoints ====================

@agent_router.post("/task", response_model=AgentTaskResponse)
async def submit_agent_task(
    request: AgentTaskRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Submit a task to the CrewAI agent system.

    The agent will:
    1. Discover relevant datasets
    2. Generate synthetic data (if needed)
    3. Train the model
    4. Export to target platform

    Requires authentication.
    """
    task_id = f"agent_{uuid.uuid4().hex[:8]}"

    try:
        # Import agent module
        from ..agents.orchestration import YOLOTrainingOrchestrator

        # Create orchestrator
        orchestrator = YOLOTrainingOrchestrator()

        # Initialize Redis state
        r = orchestrator._get_redis()
        # Store auto_confirm flag in Redis so background phase can read it
        r.hset(f"agent:{task_id}", mapping={
            "status": "submitted",
            "user_id": current_user.user_id,
            "task_description": request.task,
            "progress": "0.0",
            "created_at": datetime.now().isoformat(),
            "auto_confirm": "true" if request.auto_confirm else "false",
        })

        # Kick off Phase 1 in background
        def _run_phase1():
            try:
                orch = YOLOTrainingOrchestrator()
                orch.run_phase1(request.task, current_user.user_id, task_id)
            except Exception as e:
                r.hset(f"agent:{task_id}", mapping={
                    "status": "failed", "error": str(e),
                })

        background_tasks.add_task(_run_phase1)

        return AgentTaskResponse(
            task_id=task_id,
            status="submitted",
            message="Phase 1 started: Dataset discovery in progress. Use GET /task/{id} to poll status.",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit agent task: {str(e)}"
        )


@agent_router.get("/task/{task_id}", response_model=AgentStatusResponse)
async def get_agent_task_status(
    task_id: str,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Get agent task status.

    Requires authentication.
    """
    from ..agents.orchestration import YOLOTrainingOrchestrator
    orchestrator = YOLOTrainingOrchestrator()
    data = orchestrator.get_status(task_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if data.get("user_id") and data["user_id"] != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not your task")
    return AgentStatusResponse(
        status=data.get("status", "unknown"),
        progress=data.get("progress", 0.0),
        current_agent=data.get("current_agent", ""),
        phase1_result=data.get("phase1_result", ""),
        error=data.get("error"),
    )


@agent_router.post("/task/{task_id}/cancel")
async def cancel_agent_task(
    task_id: str,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Cancel a running agent task.

    Requires authentication.
    """
    return {
        "task_id": task_id,
        "status": "cancelled",
        "message": "Agent task cancelled"
    }


@agent_router.post("/task/{task_id}/confirm", response_model=dict)
async def confirm_task(
    task_id: str,
    request: ConfirmRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Human-in-the-loop confirmation endpoint."""
    from ..agents.orchestration import YOLOTrainingOrchestrator
    orchestrator = YOLOTrainingOrchestrator()
    data = orchestrator.get_status(task_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if data.get("user_id") and data["user_id"] != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not your task")

    # Auto-inject source into overrides when user approves without providing it.
    # This ensures coco_builtin source is passed through to Phase 2.
    overrides = request.overrides or {}
    if not overrides.get("source"):
        phase1_result = data.get("phase1_result", "")
        if "coco_builtin" in phase1_result or "COCO-Person" in phase1_result:
            overrides["source"] = "coco_builtin"
            overrides["dataset_name"] = overrides.get("dataset_name", "COCO-Person-BuiltIn")
            overrides["dataset_path"] = overrides.get("dataset_path", "/home/wangxin/data/coco_person")

    success = orchestrator.confirm(task_id, request.approved, overrides)

    # If approved and was waiting for confirmation, chain to next phase
    if success and data.get("status") == "awaiting_confirmation":
        def _chain_phase2():
            try:
                orc2 = YOLOTrainingOrchestrator()
                orc2.run_phase2(task_id, data.get("user_id", ""))
            except Exception as e:
                r2 = orchestrator._get_redis()
                r2.hset(f"agent:{task_id}", mapping={
                    "status": "failed", "error": f"Phase2 error: {e}",
                })

        background_tasks.add_task(_chain_phase2)

    # If approved and was waiting for training confirmation, chain to Phase 3
    if success and data.get("status") == "awaiting_training_confirmation":
        def _chain_phase3():
            try:
                orc3 = YOLOTrainingOrchestrator()
                orc3.run_phase3(task_id, data.get("user_id", ""))
            except Exception as e:
                r2 = orchestrator._get_redis()
                r2.hset(f"agent:{task_id}", mapping={
                    "status": "failed", "error": f"Phase3 error: {e}",
                })

        background_tasks.add_task(_chain_phase3)

    return {"confirmed": success, "task_id": task_id, "approved": request.approved}


@agent_router.get("/task/{task_id}/pipeline", response_model=dict)
async def get_pipeline_status(
    task_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    from ..agents.orchestration import YOLOTrainingOrchestrator
    orchestrator = YOLOTrainingOrchestrator()
    data = orchestrator.get_status(task_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if data.get("user_id") and data["user_id"] != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not your task")
    return orchestrator.get_pipeline_status(task_id)
