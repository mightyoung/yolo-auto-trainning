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

# Import authentication from auth module
from .auth import get_current_user, CurrentUser, check_rate_limit


# ==================== Request/Response Models ====================

class AgentTaskRequest(BaseModel):
    """Agent task request."""
    task: str = Field(..., description="Task description")
    context: Optional[dict] = Field(None, description="Additional context")
    agents: Optional[List[str]] = Field(
        None,
        description="Specific agents to use"
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
    import uuid

    task_id = f"agent_{uuid.uuid4().hex[:8]}"

    try:
        # Import agent module
        from src.agents.orchestration import YOLOTrainingOrchestrator

        # Create orchestrator
        orchestrator = YOLOTrainingOrchestrator()

        # Start workflow (in background)
        # For now, return task_id
        return AgentTaskResponse(
            task_id=task_id,
            status="submitted",
            message="Agent task submitted successfully"
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
    # TODO: Implement status tracking
    return AgentStatusResponse(
        task_id=task_id,
        status="running",
        progress=0.5
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
