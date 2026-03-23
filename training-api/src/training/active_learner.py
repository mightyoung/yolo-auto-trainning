"""
Active Learning Pipeline for YOLO Auto-Training.

Implements uncertainty-based active learning to select the most
informative samples for annotation, reducing labeling costs by 50-70%.

Reference:
- PseCo (ECCV 2022): Pseudo Labeling and Consistency Training for SSL Object Detection
- MDPI Active Learning Survey: uncertainty-based vs distribution-based methods
- ICCV 2025: Active Learning Meets Foundation Models for Remote Sensing

Strategy: Uncertainty sampling (entropy + spatial variance)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class ActiveLearningConfig:
    """Active learning configuration."""
    strategy: str = "entropy"  # entropy / margin / random / density
    top_k: int = 100           # Number of samples to select
    uncertainty_threshold: float = 0.5  # Minimum uncertainty to select
    batch_size: int = 16      # Inference batch size
    device: str = "cuda:0"


class ActiveLearningPipeline:
    """
    Uncertainty-based active learning for object detection.

    The core insight: not all samples are equally valuable for training.
    Samples where the model is uncertain (low confidence, conflicting predictions)
    provide the most information gain when labeled.

    Workflow:
    1. Current model → predict on unlabeled pool
    2. Calculate uncertainty scores per image
    3. Select Top-K highest uncertainty samples
    4. Return sample list for annotation (SAM-assisted or human)
    """

    def __init__(self, config: Optional[ActiveLearningConfig] = None):
        self.config = config or ActiveLearningConfig()

    def select_uncertain_samples(
            self,
            model_path: str,
            image_dir: str,
            top_k: Optional[int] = None,
            strategy: Optional[str] = None,
        ) -> Dict[str, Any]:
        """
        Select most uncertain samples from unlabeled image pool.

        Args:
            model_path: Path to current YOLO model
            image_dir: Directory containing unlabeled images
            top_k: Number of samples to select (overrides config)
            strategy: Selection strategy (overrides config)

        Returns:
            Dict with selected samples, uncertainty scores, and metadata
        """
        top_k = top_k or self.config.top_k
        strategy = strategy or self.config.strategy

        if not YOLO:
            return {"error": "ultralytics not available"}

        # Find all images in directory (recursive)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = [
            p for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in image_extensions
        ]

        if not image_paths:
            return {"error": f"No images found in {image_dir}", "selected": []}

        model = YOLO(model_path)
        uncertainty_scores = []

        logging.info(f"[ACTIVE_LEARN] Evaluating {len(image_paths)} images with strategy: {strategy}")

        for img_path in image_paths:
            try:
                results = model.predict(
                    source=str(img_path),
                    conf=0.1,  # Low conf to see uncertain predictions
                    verbose=False,
                    device=self.config.device,
                )

                if not results or len(results) == 0:
                    uncertainty_scores.append((str(img_path), 0.0, "no_prediction"))
                    continue

                result = results[0]
                boxes = result.boxes

                if boxes is None or len(boxes) == 0:
                    # No detections = high uncertainty (model couldn't find anything)
                    score = 1.0
                    uncertainty_scores.append((str(img_path), score, "no_detections"))
                    continue

                confs = boxes.conf.cpu().numpy()

                if strategy == "entropy":
                    score = self._entropy_score(confs)
                elif strategy == "margin":
                    score = self._margin_score(confs)
                elif strategy == "density":
                    score = self._density_score(boxes, confs)
                else:  # random
                    score = np.random.random()

                uncertainty_scores.append((str(img_path), float(score), "scored"))

            except Exception as e:
                logging.warning(f"[ACTIVE_LEARN] Error processing {img_path}: {e}")
                uncertainty_scores.append((str(img_path), 0.0, f"error: {e}"))

        # Sort by uncertainty score (descending) and take top-k
        uncertainty_scores.sort(key=lambda x: x[1], reverse=True)
        selected = uncertainty_scores[:top_k]

        return {
            "strategy": strategy,
            "total_pool": len(image_paths),
            "selected_count": len(selected),
            "selected": [
                {
                    "path": path,
                    "uncertainty_score": round(score, 4),
                    "reason": reason,
                    "rank": i + 1,
                }
                for i, (path, score, reason) in enumerate(selected)
            ],
            "rejected_count": len(image_paths) - len(selected),
        }

    def _entropy_score(self, confidences: np.ndarray) -> float:
        """
        Calculate entropy-based uncertainty.

        High entropy = model is uncertain across classes.
        Low entropy = model is confident.
        """
        if len(confidences) == 0:
            return 1.0  # No detections = maximum uncertainty

        # Convert confidences to probability distribution
        confs = np.clip(confidences, 1e-10, 1.0)
        probs = confs / confs.sum()

        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs))
        # Normalize: max entropy for N classes = log(N), we have ~2 effective bins
        normalized_entropy = entropy / np.log(2)

        # Also penalize low confidence (mean confidence)
        mean_conf = np.mean(confs)

        # Combined score: high entropy + low confidence = high uncertainty
        uncertainty = normalized_entropy * (1.0 - mean_conf)

        return float(np.clip(uncertainty, 0.0, 1.0))

    def _margin_score(self, confidences: np.ndarray) -> float:
        """
        Margin-based uncertainty: difference between top 2 confidence predictions.

        Small margin = model is uncertain about which class is correct.
        """
        if len(confidences) == 0:
            return 1.0

        sorted_confs = np.sort(confidences)[::-1]
        if len(sorted_confs) == 1:
            return 1.0 - sorted_confs[0]

        margin = sorted_confs[0] - sorted_confs[1] if len(sorted_confs) > 1 else 1.0
        return float(1.0 - margin)

    def _density_score(self, boxes, confidences: np.ndarray) -> float:
        """
        Density-based: many overlapping boxes = high uncertainty.
        """
        if boxes is None or len(boxes) == 0:
            return 0.5

        xyxy = boxes.xyxy.cpu().numpy()
        if len(xyxy) <= 1:
            return float(1.0 - np.mean(confidences))

        # Count overlapping pairs (proxy for crowded/scene complexity)
        n = len(xyxy)
        overlaps = 0
        for i in range(n):
            for j in range(i + 1, n):
                x1 = max(xyxy[i, 0], xyxy[j, 0])
                y1 = max(xyxy[i, 1], xyxy[j, 1])
                x2 = min(xyxy[i, 2], xyxy[j, 2])
                y2 = min(xyxy[i, 3], xyxy[j, 3])
                if x2 > x1 and y2 > y1:
                    overlaps += 1

        density = overlaps / max(n * (n - 1) / 2, 1)
        return float(np.clip(density * 0.5 + (1.0 - np.mean(confidences)) * 0.5, 0.0, 1.0))
