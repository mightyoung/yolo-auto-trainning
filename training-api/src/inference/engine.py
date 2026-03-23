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
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from PIL import Image
import cv2

try:
    import torch
except ImportError:
    torch = None


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
    tta: bool = False
    tta_scales: List[float] = field(default_factory=lambda: [0.83, 1.0, 1.17])
    tta_flips: List[int] = field(default_factory=lambda: [0, 1])


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

    # ─── TTA helper methods ────────────────────────────────────────────────

    def _apply_scale(self, image: np.ndarray, scale: float) -> np.ndarray:
        """
        Apply scaling transform to image.

        Args:
            image: Input image as numpy array (H, W, C).
            scale: Scale factor (e.g. 0.83 shrinks, 1.17 enlarges).

        Returns:
            Scaled image.
        """
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _descale_boxes(self, result, scale: float) -> Any:
        """
        Descale bounding boxes back to original image coordinates.

        Args:
            result: Ultralytics Results object from scaled image inference.
            scale: Scale factor used during TTA augmentation.

        Returns:
            Ultralytics Results object with boxes scaled back.
        """
        if result is None or result.boxes is None:
            return result
        boxes = result.boxes
        if len(boxes) == 0:
            return result

        # xyxy are in scaled image coords; divide by scale to recover original
        if hasattr(boxes, 'xyxyn') and boxes.xyxyn is not None:
            # Normalised coords — multiply by original shape stored in orig_shape
            orig_shape = getattr(result, 'orig_shape', None)
            if orig_shape is not None:
                scale_x = orig_shape[1] / (1.0 / scale)
                scale_y = orig_shape[0] / (1.0 / scale)
                # Use absolute coords instead
                pass

        # Work with absolute xyxy coordinates
        if hasattr(boxes, 'xyxy') and boxes.xyxy is not None:
            xyxy = boxes.xyxy.clone()
            # If coords are in normalised form, denormalise first
            if hasattr(boxes, 'orig_shape') and boxes.orig_shape:
                orig_h, orig_w = boxes.orig_shape
                if xyxy.max() <= 1.0:
                    xyxy[:, [0, 2]] *= orig_w
                    xyxy[:, [1, 3]] *= orig_h
            # Descale from scaled image back to original size
            xyxy /= scale
            boxes.xyxy[:] = xyxy
        return result

    def _flip_boxes_horizontal(self, result, width: int) -> Any:
        """
        Flip bounding boxes horizontally.

        Args:
            result: Ultralytics Results object.
            width: Original image width.

        Returns:
            Ultralytics Results object with boxes flipped.
        """
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return result
        boxes = result.boxes
        if hasattr(boxes, 'xyxy') and boxes.xyxy is not None:
            xyxy = boxes.xyxy.clone()
            x1 = xyxy[:, 0].clone()
            x2 = xyxy[:, 2].clone()
            xyxy[:, 0] = width - x2
            xyxy[:, 2] = width - x1
            boxes.xyxy[:] = xyxy
        return result

    def _merge_tta_predictions(self, results: List) -> Any:
        """
        Merge predictions from multiple TTA augmentations using NMS.

        Args:
            results: List of Ultralytics Results objects.

        Returns:
            Single merged Ultralytics Results object.
        """
        try:
            from ultralytics.utils.ops import non_max_suppression
        except ImportError:
            return results[0] if results else None

        all_xyxy = []
        all_scores = []
        all_classes = []
        all_names = None

        for result in results:
            if result is None or result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes
            if all_names is None and hasattr(result, 'names') and result.names:
                all_names = result.names
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.array(boxes.xyxy)
            conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.array(boxes.conf)
            cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.array(boxes.cls)
            all_xyxy.append(xyxy)
            all_scores.append(conf)
            all_classes.append(cls)

        if not all_xyxy:
            return results[0] if results else None

        all_xyxy = np.concatenate(all_xyxy, axis=0).astype(np.float32)
        all_scores = np.concatenate(all_scores, axis=0).astype(np.float32)
        all_classes = np.concatenate(all_classes, axis=0).astype(np.float32)

        # Apply NMS using ultralytics utility
        if torch is not None:
            keep = non_max_suppression(
                torch.from_numpy(all_xyxy),
                conf_thres=0.0,  # Already filtered per-model
                iou_thres=0.45,
            )
            keep = keep[0].numpy() if isinstance(keep, tuple) else keep.numpy()
        else:
            keep = np.arange(len(all_xyxy))

        final_xyxy = all_xyxy[keep]
        final_scores = all_scores[keep]
        final_classes = all_classes[keep]

        # Build a synthetic result using the first result as template
        if results and results[0] is not None:
            merged = results[0].copy()
            if hasattr(merged.boxes, 'xyxy'):
                merged.boxes.xyxy = torch.from_numpy(final_xyxy)
            if hasattr(merged.boxes, 'conf'):
                merged.boxes.conf = torch.from_numpy(final_scores)
            if hasattr(merged.boxes, 'cls'):
                merged.boxes.cls = torch.from_numpy(final_classes)
            return merged

        return results[0] if results else None

    # ─── Predict ─────────────────────────────────────────────────────────

    def predict(
        self,
        model_path: str,
        source: Union[str, np.ndarray, Image.Image, List],
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        device: str = "cuda:0",
        half: bool = False,
        tta: bool = False,
        tta_scales: List[float] = None,
        tta_flips: List[int] = None,
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
            tta: Enable test-time augmentation
            tta_scales: Scale factors for multi-scale TTA (default: [0.83, 1.0, 1.17])
            tta_flips: Flip modes: 0=none, 1=horizontal (default: [0, 1])

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
            if tta:
                # Test-Time Augmentation: multi-scale + flip
                scales = tta_scales or [0.83, 1.0, 1.17]
                flips = tta_flips or [0, 1]

                # Load image as numpy for transformations
                if isinstance(source, str):
                    img = cv2.imread(source)
                elif isinstance(source, Image.Image):
                    img = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
                elif isinstance(source, np.ndarray):
                    img = source.copy()
                else:
                    # Fallback: regular inference
                    results = model.predict(
                        source=source,
                        conf=conf,
                        iou=iou,
                        max_det=max_det,
                        device=device,
                        half=half,
                        verbose=False,
                    )
                    final_result = results[0] if results else None

                image_h, image_w = img.shape[:2]
                all_results = []

                for scale in scales:
                    for flip in flips:
                        # Apply scale augmentation
                        scaled_img = self._apply_scale(img, scale)

                        # Apply horizontal flip if requested
                        if flip == 1:
                            aug_img = cv2.flip(scaled_img, 1)
                            flip_w = scaled_img.shape[1]
                        else:
                            aug_img = scaled_img
                            flip_w = scaled_img.shape[1]

                        # Predict on augmented image
                        pred = model.predict(
                            source=aug_img,
                            conf=conf,
                            iou=iou,
                            max_det=max_det,
                            device=device,
                            half=half,
                            verbose=False,
                        )
                        if pred:
                            pred_result = pred[0]
                            # Descale boxes back to original image coordinates
                            pred_result = self._descale_boxes(pred_result, scale)
                            # Flip boxes horizontally if flip was applied
                            if flip == 1:
                                pred_result = self._flip_boxes_horizontal(pred_result, image_w)
                            all_results.append(pred_result)

                # Merge all TTA predictions using weighted NMS
                final_result = self._merge_tta_predictions(all_results)
            else:
                # Standard inference
                results = model.predict(
                    source=source,
                    conf=conf,
                    iou=iou,
                    max_det=max_det,
                    device=device,
                    half=half,
                    verbose=False,
                )
                final_result = results[0] if results else None

            # Parse results
            detections = []
            img_size = (0, 0)
            if final_result is not None:
                boxes = final_result.boxes
                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        box = boxes[i]
                        detections.append({
                            "class_id": int(box.cls[0]) if box.cls is not None else 0,
                            "class_name": final_result.names[int(box.cls[0])] if box.cls is not None and final_result.names else "unknown",
                            "confidence": float(box.conf[0]) if box.conf is not None else 0.0,
                            "bbox": {
                                "x1": float(box.xyxy[0][0]) if box.xyxy is not None else 0,
                                "y1": float(box.xyxy[0][1]) if box.xyxy is not None else 0,
                                "x2": float(box.xyxy[0][2]) if box.xyxy is not None else 0,
                                "y2": float(box.xyxy[0][3]) if box.xyxy is not None else 0,
                            }
                        })
                if hasattr(final_result, 'orig_shape') and final_result.orig_shape is not None:
                    img_size = tuple(final_result.orig_shape)

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
