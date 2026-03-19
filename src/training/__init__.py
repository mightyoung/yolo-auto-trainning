# Training Module

"""
YOLO11 training module with Ray Tune HPO and MLflow experiment tracking.
"""

from .runner import (
    YOLOTrainer,
    KnowledgeDistillationTrainer,
    TransferLearningTrainer,
    TrainingResult,
)
from .config import (
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_SANITY_CHECK_CONFIG,
    DEFAULT_HPO_CONFIG,
    DEFAULT_EXPORT_CONFIG,
    TrainingConfig,
    SanityCheckConfig,
    HPOConfig,
    ExportConfig,
)

# MLflow tracker is optional - only import if mlflow is available
try:
    from .mlflow_tracker import (
        MLflowTracker,
        enable_yolo_mlflow_logging,
    )
    _mlflow_available = True
except ImportError:
    _mlflow_available = False
    MLflowTracker = None
    enable_yolo_mlflow_logging = None

__all__ = [
    "YOLOTrainer",
    "KnowledgeDistillationTrainer",
    "TransferLearningTrainer",
    "TrainingResult",
    "MLflowTracker",
    "enable_yolo_mlflow_logging",
    "DEFAULT_TRAINING_CONFIG",
    "DEFAULT_SANITY_CHECK_CONFIG",
    "DEFAULT_HPO_CONFIG",
    "DEFAULT_EXPORT_CONFIG",
    "TrainingConfig",
    "SanityCheckConfig",
    "HPOConfig",
    "ExportConfig",
]
