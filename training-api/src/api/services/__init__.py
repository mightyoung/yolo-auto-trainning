"""Training API services layer.

Simplified service implementations for potential future extraction.
NOTE: The canonical implementations are in api/routes.py - these are
simplified alternatives for reference/extraction in the future.
"""

from .training_service import (
    DynamicTrainingManager,
    _run_curriculum_sync,
    _run_training_sync,
)

__all__ = [
    "DynamicTrainingManager",
    "_run_training_sync",
    "_run_curriculum_sync",
]
