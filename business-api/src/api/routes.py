"""
Business API Routes - Backward Compatibility
============================================

.. deprecated::
    This module is deprecated. Use business-api/src/api/route_handlers/ instead.

This file provides backward compatibility for imports like:
    from src.api import routes
    routes.get_current_user

New code should import directly from the route_handlers package:
    from src.api.route_handlers import train_router, data_router, etc.
"""

# Re-export routers from the new route_handlers package
# Re-export auth and task_registry from their original modules
from .auth import CurrentUser, check_rate_limit, create_access_token, get_current_user
from .route_handlers import (
    analysis_router,
    auth_router,
    callback_router,
    data_router,
    deploy_router,
    queue_router,
    train_router,
)
from .task_models import (
    ExportStatusResponse,
    TaskDetailResponse,
    TaskExecutionSummaryResponse,
    TaskListResponse,
    TaskRecordResponse,
    TrainStatusResponse,
)
from .task_registry import (
    attach_execution_snapshot,
    build_export_status_response,
    build_result_summary,
    build_task_record,
    build_training_status_response,
    delete_task_from_redis,
    get_aggregated_task,
    get_task_from_redis,
    get_user_tasks_from_redis,
    normalize_task_record,
    store_task_in_redis,
    verify_task_ownership,
)

__all__ = [
    "data_router",
    "train_router",
    "deploy_router",
    "analysis_router",
    "auth_router",
    "callback_router",
    "queue_router",
    "get_current_user",
    "CurrentUser",
    "create_access_token",
    "check_rate_limit",
    "normalize_task_record",
    "build_task_record",
    "store_task_in_redis",
    "get_task_from_redis",
    "get_user_tasks_from_redis",
    "verify_task_ownership",
    "delete_task_from_redis",
    "attach_execution_snapshot",
    "get_aggregated_task",
    "build_result_summary",
    "build_training_status_response",
    "build_export_status_response",
    "TaskExecutionSummaryResponse",
    "TaskRecordResponse",
    "TaskListResponse",
    "TaskDetailResponse",
    "TrainStatusResponse",
    "ExportStatusResponse",
]
