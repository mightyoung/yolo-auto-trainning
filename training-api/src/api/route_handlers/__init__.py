"""Training API route handlers package.

This package contains route handlers extracted from routes.py.
For backward compatibility, the original routes.py continues to be used.

Structure:
- models/ - Request and response models (DONE)
- store/ - Task storage (DONE)
- services/ - Business logic services (TODO)
- route_handlers/ - Route handlers (TODO)
"""

# Re-export models for convenience
from ..models import (
    ActiveLearnSelectRequest,
    BenchmarkRunRequest,
    ExportStartRequest,
    HPOStartRequest,
    SemiSupervisedRequest,
    TrainStartRequest,
    TrainStatusResponse,
)

# Re-export task store utilities
from ..store.task_store import (
    _task_del,
    _task_get,
    _task_set,
    clear_cancel_event,
    get_cancel_event,
    get_tasks_cache,
    get_tasks_lock,
    set_cancel_event,
)

__all__ = [
    # Models
    "TrainStartRequest",
    "TrainStatusResponse",
    "HPOStartRequest",
    "ExportStartRequest",
    "BenchmarkRunRequest",
    "ActiveLearnSelectRequest",
    "SemiSupervisedRequest",
    # Store
    "_task_get",
    "_task_set",
    "_task_del",
    "get_tasks_cache",
    "get_tasks_lock",
    "get_cancel_event",
    "set_cancel_event",
    "clear_cancel_event",
]
