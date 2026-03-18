"""
Inference Module for YOLO models.

Provides real-time inference and batch prediction capabilities.
"""

from .engine import (
    InferenceEngine,
    InferenceResult,
    InferenceConfig,
    ModelCache,
    get_inference_engine,
)

from .batch import (
    BatchPredictor,
    BatchConfig,
    BatchPredictionResult,
    BatchStatus,
    ScheduledBatchProcessor,
    create_batch_prediction_task,
)

__all__ = [
    "InferenceEngine",
    "InferenceResult",
    "InferenceConfig",
    "ModelCache",
    "get_inference_engine",
    "BatchPredictor",
    "BatchConfig",
    "BatchPredictionResult",
    "BatchStatus",
    "ScheduledBatchProcessor",
    "create_batch_prediction_task",
]
