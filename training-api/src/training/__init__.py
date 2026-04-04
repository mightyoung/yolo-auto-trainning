"""
Training Module — Stable Public API
==================================

This package provides YOLO training, transfer learning, and curriculum training.

Public API (stable, semantically versioned):
---------------------------------------------

Types:
    TrainingResult          — training outcome container
    DatasetDistributionResult — train/val box distribution analysis
    CurriculumStage         — single curriculum stage definition
    CurriculumConfig        — 3-stage progressive curriculum config
    TrainingCancelled       — exception raised on training cancellation

Trainer classes:
    YOLOTrainer             — YOLO11 training with HPO, AMP, DDP, retry
    TransferLearningTrainer — knowledge distillation (MGD / feature / soft)
    PipelineCurriculumTrainer — 3-stage progressive training with plateau detection

Utility functions:
    validate_dataset_distribution — detect train/val box-size mismatch
    setup_gpu_memory             — configure per-process GPU memory fraction
    cleanup_gpu_memory           — clear CUDA cache after training

Configuration:
    TrainingConfig, SanityCheckConfig, HPOConfig, ExportConfig
    DEFAULT_TRAINING_CONFIG, DEFAULT_SANITY_CHECK_CONFIG,
    DEFAULT_HPO_CONFIG, DEFAULT_EXPORT_CONFIG

Internal sub-modules (may change without notice):
-------------------------------------------------
    training_utils.py    — shared types and helpers
    yolo_trainer.py     — YOLOTrainer implementation
    transfer_trainer.py  — TransferLearningTrainer implementation
    curriculum.py        — PipelineCurriculumTrainer + CurriculumStage/CurriculumConfig
    config.py           — TrainingConfig and related dataclasses
    plateau_manager.py   — plateau detection and breaking logic
    mlflow_tracker.py   — MLflow experiment tracking
    active_learner.py   — active learning pipeline
    semi_supervised.py  — semi-supervised pseudo-labeling
"""

# Import submodules to make them accessible as package attributes
# (required for patch() to work reliably in tests)
from . import training_utils
from . import yolo_trainer
from . import transfer_trainer
from . import curriculum
from . import config
from . import plateau_manager
from . import active_learner
from . import semi_supervised
# Note: mlflow_tracker is NOT imported here because it requires `mlflow`
# which may not be installed. Import it directly where needed.

from .training_utils import (
    TrainingCancelled,
    TrainingResult,
    DatasetDistributionResult,
    setup_gpu_memory,
    cleanup_gpu_memory,
    validate_dataset_distribution,
)

from .yolo_trainer import YOLOTrainer
from .transfer_trainer import TransferLearningTrainer
from .curriculum import (
    CurriculumStage,
    CurriculumConfig,
    PipelineCurriculumTrainer,
)

from .config import (
    TrainingConfig,
    SanityCheckConfig,
    HPOConfig,
    ExportConfig,
    LRSchedulerConfig,
    PlateauBreakingConfig,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_SANITY_CHECK_CONFIG,
    DEFAULT_HPO_CONFIG,
    DEFAULT_EXPORT_CONFIG,
)

__all__ = [
    # Types
    "TrainingResult",
    "DatasetDistributionResult",
    "CurriculumStage",
    "CurriculumConfig",
    "TrainingCancelled",
    # Trainer classes
    "YOLOTrainer",
    "TransferLearningTrainer",
    "PipelineCurriculumTrainer",
    # Utility functions
    "validate_dataset_distribution",
    "setup_gpu_memory",
    "cleanup_gpu_memory",
    # Configuration
    "TrainingConfig",
    "SanityCheckConfig",
    "HPOConfig",
    "ExportConfig",
    "LRSchedulerConfig",
    "PlateauBreakingConfig",
    "DEFAULT_TRAINING_CONFIG",
    "DEFAULT_SANITY_CHECK_CONFIG",
    "DEFAULT_HPO_CONFIG",
    "DEFAULT_EXPORT_CONFIG",
]
