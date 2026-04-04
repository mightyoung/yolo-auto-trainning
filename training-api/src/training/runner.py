"""
YOLO Training Runner — Facade module.

This module is a backward-compatible facade that re-exports all public types
from the split sub-modules. The actual implementations live in:

- training_utils.py    — Shared types: TrainingCancelled, TrainingResult,
                         DatasetDistributionResult, validate_dataset_distribution
- yolo_trainer.py     — YOLOTrainer class
- transfer_trainer.py  — TransferLearningTrainer class
- curriculum.py        — CurriculumStage, CurriculumConfig, PipelineCurriculumTrainer

External consumers should migrate to importing directly from sub-modules:
    from training_api.src.training.yolo_trainer import YOLOTrainer
    from training_api.src.training.curriculum import CurriculumStage, CurriculumConfig

This facade preserves existing imports like:
    from src.training.runner import YOLOTrainer, TrainingCancelled
"""

# Re-export shared utilities and types
from .training_utils import (
    TrainingCancelled,
    TrainingResult,
    DatasetDistributionResult,
    setup_gpu_memory,
    cleanup_gpu_memory,
    validate_dataset_distribution,
)

# Re-export YOLOTrainer
from .yolo_trainer import YOLOTrainer

# Re-export TransferLearningTrainer
from .transfer_trainer import TransferLearningTrainer

# Re-export curriculum classes
from .curriculum import (
    CurriculumStage,
    CurriculumConfig,
    PipelineCurriculumTrainer,
)
