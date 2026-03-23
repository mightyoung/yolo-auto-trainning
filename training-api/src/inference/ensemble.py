"""
Model Ensemble Inference for YOLO Auto-Training.

Combines predictions from multiple YOLO models using weighted NMS
to improve detection robustness and reduce false positives.

Reference: ICCV/CVPR best practices for model ensemble in object detection
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from ultralytics import YOLO
    from ultralytics.utils.ops import non_max_suppression
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None
    non_max_suppression = None


@dataclass
class EnsembleConfig:
    """Ensemble configuration."""
    model_paths: List[str]
    weights: Optional[List[float]] = None  # Per-model weights, defaults to equal
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 300


class ModelEnsemble:
    """
    Multi-model ensemble inference using weighted NMS.

    Process:
    1. Run each model independently
    2. Collect all predictions
    3. Weight predictions by model weights
    4. Apply NMS to merge overlapping detections
    """

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self._load_models()

    def _load_models(self):
        """Lazy load all ensemble models."""
        if not ULTRALYTICS_AVAILABLE:
            return
        for path in self.config.model_paths:
            try:
                self.models[path] = YOLO(path)
            except Exception as e:
                print(f"[ModelEnsemble] Failed to load model {path}: {e}")

    def predict(
        self,
        source,
        conf: float = None,
        iou: float = None,
        device: str = "cuda:0",
    ) -> Dict[str, Any]:
        """
        Run ensemble inference on source.

        Args:
            source: Image path, URL, or numpy array.
            conf: Confidence threshold (overrides config).
            iou: IoU threshold for NMS (overrides config).
            device: Device to run inference on.

        Returns:
            Dictionary with boxes, scores, classes, num_models, num_detections.
        """
        conf = conf if conf is not None else self.config.conf_threshold
        iou = iou if iou is not None else self.config.iou_threshold

        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        all_classes: List[np.ndarray] = []

        raw_weights = self.config.weights or [1.0] * len(self.config.model_paths)
        total_w = sum(raw_weights)
        weights = [w / total_w for w in raw_weights]

        for idx, path in enumerate(self.config.model_paths):
            model = self.models.get(path)
            if model is None:
                continue

            try:
                results = model.predict(
                    source=source,
                    conf=conf,
                    iou=iou,
                    max_det=self.config.max_det,
                    device=device,
                    verbose=False,
                )
            except Exception as e:
                print(f"[ModelEnsemble] Inference failed for {path}: {e}")
                continue

            if not results:
                continue

            r = results[0]
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.array(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.array(boxes.conf)
            cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.array(boxes.cls)

            # Weight scores by model weight
            weighted_confs = confs * weights[idx]

            all_boxes.append(xyxy)
            all_scores.append(weighted_confs)
            all_classes.append(cls_ids)

        if not all_boxes:
            return self._empty_result()

        # Concatenate all predictions
        all_boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
        all_scores = np.concatenate(all_scores, axis=0).astype(np.float32)
        all_classes = np.concatenate(all_classes, axis=0).astype(np.float32)

        if len(all_boxes) == 0:
            return self._empty_result()

        # Apply final NMS using ultralytics utility
        keep: np.ndarray
        if non_max_suppression is not None and torch is not None:
            keep_tensor = non_max_suppression(
                torch.from_numpy(all_boxes),
                conf_thres=conf,
                iou_thres=iou,
            )
            # non_max_suppression returns (keep_indices,) on modern ultralytics
            if isinstance(keep_tensor, tuple):
                keep_tensor = keep_tensor[0]
            keep = keep_tensor.cpu().numpy() if hasattr(keep_tensor, 'cpu') else np.array(keep_tensor)
        else:
            keep = np.arange(len(all_boxes))

        final_boxes = all_boxes[keep]
        final_scores = all_scores[keep]
        final_classes = all_classes[keep]

        return {
            "boxes": final_boxes.tolist(),
            "scores": final_scores.tolist(),
            "classes": final_classes.tolist(),
            "num_models": len(self.models),
            "num_detections": int(len(keep)),
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "boxes": [],
            "scores": [],
            "classes": [],
            "num_models": len(self.models),
            "num_detections": 0,
        }
