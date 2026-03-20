"""
Real-time Inference Module for YOLO models.

Based on ML system design patterns:
- Real-time inference for immediate results
- Model caching for performance
- Batch processing for efficiency
"""

import os
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from PIL import Image


# Try to import ultralytics, handle gracefully if not available
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not installed. Inference requires ultralytics.")


@dataclass
class InferenceResult:
    """Inference result container."""
    task_id: str
    status: str
    detections: List[Dict[str, Any]]
    inference_time_ms: float
    model_name: str
    image_size: tuple
    timestamp: str


@dataclass
class InferenceConfig:
    """Inference configuration."""
    model_path: str
    confidence: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 300
    device: str = "cuda:0"
    half: bool = False


class ModelCache:
    """
    Model cache with thread-safe loading.

    Based on best practices for model serving:
    - Lazy loading
    - Thread-safe access
    - Configurable cache size
    """

    def __init__(self, max_size: int = 3):
        self._cache: Dict[str, YOLO] = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._access_times: Dict[str, float] = {}

    def get(self, model_path: str) -> Optional[YOLO]:
        """Get model from cache."""
        with self._lock:
            if model_path in self._cache:
                self._access_times[model_path] = time.time()
                return self._cache[model_path]
        return None

    def load(self, model_path: str) -> Optional[YOLO]:
        """Load model into cache."""
        if not ULTRALYTICS_AVAILABLE:
            return None

        with self._lock:
            # Check if already loaded
            if model_path in self._cache:
                self._access_times[model_path] = time.time()
                return self._cache[model_path]

            # Evict oldest if cache is full
            if len(self._cache) >= self._max_size:
                oldest = min(self._access_times, key=self._access_times.get)
                if oldest in self._cache:
                    del self._cache[oldest]
                    del self._access_times[oldest]

            # Load new model
            try:
                model = YOLO(model_path)
                self._cache[model_path] = model
                self._access_times[model_path] = time.time()
                return model
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                return None

    def clear(self):
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()


class InferenceEngine:
    """
    Real-time inference engine for YOLO models.

    Features:
    - Thread-safe inference
    - Model caching
    - Configurable parameters
    - Metrics collection
    """

    # Singleton instance
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, cache_size: int = 3):
        if hasattr(self, '_initialized'):
            return
        self._cache = ModelCache(max_size=cache_size)
        self._inference_count = 0
        self._total_time_ms = 0
        self._initialized = True

    def predict(
        self,
        model_path: str,
        source: Union[str, np.ndarray, Image.Image, List],
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        device: str = "cuda:0",
        half: bool = False,
    ) -> InferenceResult:
        """
        Run inference on input source.

        Args:
            model_path: Path to model weights
            source: Input source (image path, numpy array, PIL Image, or list)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            max_det: Maximum detections
            device: Device to use
            half: Use FP16 inference

        Returns:
            InferenceResult with detections
        """
        task_id = f"inf_{int(time.time() * 1000)}"
        start_time = time.time()

        # Get or load model
        model = self._cache.get(model_path)
        if model is None:
            model = self._cache.load(model_path)
            if model is None:
                return InferenceResult(
                    task_id=task_id,
                    status="error",
                    detections=[],
                    inference_time_ms=0,
                    model_name=model_path,
                    image_size=(0, 0),
                    timestamp=datetime.now().isoformat()
                )

        # Run inference
        try:
            results = model.predict(
                source=source,
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                half=half,
                verbose=False,
            )

            # Parse results
            detections = []
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes

                for i in range(len(boxes)):
                    box = boxes[i]
                    detections.append({
                        "class_id": int(box.cls[0]) if box.cls is not None else 0,
                        "class_name": result.names[int(box.cls[0])] if box.cls is not None and result.names else "unknown",
                        "confidence": float(box.conf[0]) if box.conf is not None else 0.0,
                        "bbox": {
                            "x1": float(box.xyxy[0][0]) if box.xyxy is not None else 0,
                            "y1": float(box.xyxy[0][1]) if box.xyxy is not None else 0,
                            "x2": float(box.xyxy[0][2]) if box.xyxy is not None else 0,
                            "y2": float(box.xyxy[0][3]) if box.xyxy is not None else 0,
                        }
                    })

            # Get image size
            img_size = (0, 0)
            if hasattr(results[0], 'orig_shape') and results[0].orig_shape is not None:
                img_size = tuple(results[0].orig_shape)

            inference_time = (time.time() - start_time) * 1000

            # Update metrics
            self._inference_count += 1
            self._total_time_ms += inference_time

            return InferenceResult(
                task_id=task_id,
                status="success",
                detections=detections,
                inference_time_ms=inference_time,
                model_name=model_path,
                image_size=img_size,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            return InferenceResult(
                task_id=task_id,
                status="error",
                detections=[],
                inference_time_ms=(time.time() - start_time) * 1000,
                model_name=model_path,
                image_size=(0, 0),
                timestamp=datetime.now().isoformat()
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        avg_time = self._total_time_ms / self._inference_count if self._inference_count > 0 else 0
        return {
            "total_inferences": self._inference_count,
            "total_time_ms": self._total_time_ms,
            "average_time_ms": avg_time,
            "cached_models": len(self._cache._cache),
        }

    def clear_cache(self):
        """Clear model cache."""
        self._cache.clear()


# Global inference engine instance
_inference_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Get global inference engine instance."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngine()
    return _inference_engine
