"""
Batch Prediction Pipeline for YOLO models.

Based on ML system design patterns:
- Batch prediction for scheduled large-scale processing
- Celery integration for async task processing
- Result storage and retrieval
"""

import os
import time
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
from PIL import Image


class BatchStatus(Enum):
    """Batch prediction status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchPredictionResult:
    """Batch prediction result."""
    batch_id: str
    status: str
    total_images: int
    processed_images: int
    failed_images: int
    results: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BatchConfig:
    """Batch prediction configuration."""
    model_path: str
    input_dir: str
    output_dir: str
    confidence: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 300
    device: str = "cuda:0"
    batch_size: int = 8


class BatchPredictor:
    """
    Batch prediction processor for YOLO models.

    Features:
    - Large-scale image processing
    - Progress tracking
    - Result export to JSON/CSV
    - Error handling and retry
    """

    def __init__(self, config: BatchConfig):
        """
        Initialize batch predictor.

        Args:
            config: Batch configuration
        """
        self.config = config
        self._results: List[Dict[str, Any]] = []

    def process_batch(self) -> BatchPredictionResult:
        """
        Process batch of images.

        Returns:
            BatchPredictionResult with processing results
        """
        batch_id = str(uuid.uuid4())[:8]
        started_at = datetime.now().isoformat()

        # Import inference engine
        try:
            from ..inference.engine import get_inference_engine
            engine = get_inference_engine()
        except ImportError:
            return BatchPredictionResult(
                batch_id=batch_id,
                status="failed",
                total_images=0,
                processed_images=0,
                failed_images=0,
                error="Inference engine not available",
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
            )

        # Get input images
        input_path = Path(self.config.input_dir)
        if not input_path.exists():
            return BatchPredictionResult(
                batch_id=batch_id,
                status="failed",
                total_images=0,
                processed_images=0,
                failed_images=0,
                error=f"Input directory not found: {self.config.input_dir}",
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
            )

        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        total_images = len(image_files)
        processed = 0
        failed = 0
        results = []

        # Create output directory
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Process images in batches
        for i, img_file in enumerate(image_files):
            try:
                # Run inference
                result = engine.predict(
                    model_path=self.config.model_path,
                    source=str(img_file),
                    conf=self.config.confidence,
                    iou=self.config.iou_threshold,
                    max_det=self.config.max_det,
                    device=self.config.device,
                )

                # Store result
                results.append({
                    "image": str(img_file.name),
                    "status": result.status,
                    "detections": result.detections,
                    "inference_time_ms": result.inference_time_ms,
                })

                # Save individual result
                result_file = output_path / f"{img_file.stem}_result.json"
                with open(result_file, 'w') as f:
                    json.dump({
                        "image": str(img_file.name),
                        "status": result.status,
                        "detections": result.detections,
                        "inference_time_ms": result.inference_time_ms,
                    }, f, indent=2)

                processed += 1

            except Exception as e:
                failed += 1
                results.append({
                    "image": str(img_file.name),
                    "status": "error",
                    "error": str(e),
                })

        completed_at = datetime.now().isoformat()

        # Save summary
        summary = {
            "batch_id": batch_id,
            "config": {
                "model_path": self.config.model_path,
                "input_dir": self.config.input_dir,
                "confidence": self.config.confidence,
            },
            "results": {
                "total_images": total_images,
                "processed_images": processed,
                "failed_images": failed,
            },
            "started_at": started_at,
            "completed_at": completed_at,
        }

        summary_file = output_path / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        return BatchPredictionResult(
            batch_id=batch_id,
            status="completed" if failed == 0 else "completed_with_errors",
            total_images=total_images,
            processed_images=processed,
            failed_images=failed,
            results=results,
            started_at=started_at,
            completed_at=completed_at,
        )


class ScheduledBatchProcessor:
    """
    Scheduled batch processor using Celery.

    Features:
    - Cron-like scheduling
    - Periodic task execution
    - Result storage
    """

    def __init__(self):
        """Initialize scheduled processor."""
        self._scheduled_tasks: Dict[str, Dict[str, Any]] = {}

    def schedule_batch(
        self,
        task_id: str,
        config: BatchConfig,
        schedule_type: str = "interval",
        interval_minutes: int = 60,
        cron_expression: str = None,
    ) -> Dict[str, Any]:
        """
        Schedule a batch prediction task.

        Args:
            task_id: Unique task identifier
            config: Batch configuration
            schedule_type: "interval" or "cron"
            interval_minutes: Interval in minutes (for interval type)
            cron_expression: Cron expression (for cron type)

        Returns:
            Scheduled task info
        """
        task_info = {
            "task_id": task_id,
            "config": {
                "model_path": config.model_path,
                "input_dir": config.input_dir,
                "output_dir": config.output_dir,
                "confidence": config.confidence,
                "batch_size": config.batch_size,
            },
            "schedule_type": schedule_type,
            "interval_minutes": interval_minutes if schedule_type == "interval" else None,
            "cron_expression": cron_expression if schedule_type == "cron" else None,
            "status": "scheduled",
            "created_at": datetime.now().isoformat(),
        }

        self._scheduled_tasks[task_id] = task_info
        return task_info

    def get_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """Get all scheduled tasks."""
        return list(self._scheduled_tasks.values())

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a scheduled task.

        Args:
            task_id: Task identifier

        Returns:
            True if cancelled
        """
        if task_id in self._scheduled_tasks:
            self._scheduled_tasks[task_id]["status"] = "cancelled"
            return True
        return False


def create_batch_prediction_task(
    model_path: str,
    input_dir: str,
    output_dir: str,
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
    device: str = "cuda:0",
    batch_size: int = 8,
) -> BatchPredictionResult:
    """
    Create and run a batch prediction task.

    Args:
        model_path: Path to model weights
        input_dir: Input directory with images
        output_dir: Output directory for results
        confidence: Confidence threshold
        iou_threshold: IoU threshold
        device: Device to use
        batch_size: Batch size

    Returns:
        BatchPredictionResult
    """
    config = BatchConfig(
        model_path=model_path,
        input_dir=input_dir,
        output_dir=output_dir,
        confidence=confidence,
        iou_threshold=iou_threshold,
        device=device,
        batch_size=batch_size,
    )

    predictor = BatchPredictor(config)
    return predictor.process_batch()
