"""Data and concept drift detection for production ML models.

References:
- EvidentlyAI: Population Stability Index (PSI) methodology
- YARD / Drift detection via KS test
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Container for drift detection results."""

    data_drift_score: float  # Overall PSI score
    concept_drift_detected: bool
    feature_drift: Dict[str, float]  # Per-feature PSI scores
    recommendation: str  # "retrain" | "monitor" | "ok"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Optional[Dict[str, Any]] = None


class DriftDetector:
    """Detect data and concept drift in production inference.

    PSI thresholds (Population Stability Index):
      < 0.1  : no drift  (stable)
      0.1-0.2: slight drift (monitor closely)
      > 0.2  : significant drift (retrain recommended)
    """

    def __init__(self, psi_threshold: float = 0.2, concept_threshold: float = 0.05):
        """
        Args:
            psi_threshold: PSI above which data drift is flagged (default 0.2)
            concept_threshold: Relative mAP decline to flag concept drift (default 5%)
        """
        self.psi_threshold = psi_threshold
        self.concept_threshold = concept_threshold

    # ------------------------------------------------------------------
    # PSI (Population Stability Index)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_psi(
        expected: np.ndarray,
        actual: np.ndarray,
        buckets: int = 10,
        epsilon: float = 1e-4,
    ) -> float:
        """Compute Population Stability Index between two distributions.

        PSI < 0.1  : no drift
        0.1 <= PSI < 0.2 : slight drift
        PSI >= 0.2  : significant drift

        Args:
            expected: Reference (baseline) distribution
            actual: Current production distribution
            buckets: Number of bins for distribution comparison
            epsilon: Floor value to avoid log(0)
        """
        # Remove NaN/Inf values
        expected = np.asarray(expected, dtype=np.float64).flatten()
        actual = np.asarray(actual, dtype=np.float64).flatten()
        expected = expected[np.isfinite(expected)]
        actual = actual[np.isfinite(actual)]

        if len(expected) == 0 or len(actual) == 0:
            logger.warning("[PSI] Empty array provided, returning 0.0")
            return 0.0

        # Build buckets using expected distribution as reference
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))

        # Ensure full range coverage
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        # Convert to proportions (with floor to prevent log(0))
        expected_pct = np.maximum(expected_counts / len(expected), epsilon)
        actual_pct = np.maximum(actual_counts / len(actual), epsilon)

        # PSI formula: sum((actual% - expected%) * ln(actual% / expected%))
        psi_value = np.sum(
            (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        )
        return float(round(psi_value, 6))

    # ------------------------------------------------------------------
    # Data Drift
    # ------------------------------------------------------------------
    def detect_data_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        buckets: int = 10,
    ) -> float:
        """Detect feature-level data drift using PSI + KS test.

        Args:
            reference_data: Reference feature array (N, D) or (N,) for single feature
            current_data: Current production feature array
            buckets: Number of PSI buckets

        Returns:
            Overall PSI score (average across features if multi-dimensional)
        """
        reference_data = np.asarray(reference_data)
        current_data = np.asarray(current_data)

        # Multi-dimensional: compute per-feature PSI
        if reference_data.ndim == 2:
            scores: List[float] = []
            for i in range(reference_data.shape[1]):
                feat_ref = reference_data[:, i]
                feat_cur = current_data[:, i] if current_data.ndim == 2 else current_data
                score = self.compute_psi(feat_ref, feat_cur, buckets=buckets)
                scores.append(score)
            overall = float(np.mean(scores))
            logger.info(f"[DriftDetector] Per-feature PSI scores: {scores}  |  Overall: {overall:.4f}")
            return overall

        # Single feature
        overall = self.compute_psi(reference_data, current_data, buckets=buckets)
        logger.info(f"[DriftDetector] Data drift PSI: {overall:.4f}")
        return overall

    # ------------------------------------------------------------------
    # Concept Drift
    # ------------------------------------------------------------------
    def detect_concept_drift(
        self,
        metrics_history: List[float],
        window: int = 10,
    ) -> bool:
        """Detect concept drift via rolling mAP decline.

        Concept drift = model performance degradation in production,
        indicating the underlying data distribution changed.

        Method: Compare rolling average of recent window vs. the full history.
        If the recent window average is significantly lower, flag concept drift.

        Args:
            metrics_history: Ordered list of mAP values (oldest → newest)
            window: Number of recent samples for rolling average

        Returns:
            True if concept drift is detected
        """
        if len(metrics_history) < window * 2:
            logger.warning(
                f"[DriftDetector] Not enough history ({len(metrics_history)}) "
                f"for concept drift detection (need >= {window * 2})"
            )
            return False

        history = np.array(metrics_history)

        # Rolling average (recent window)
        recent_avg = float(np.mean(history[-window:]))

        # Baseline: all history except recent window
        baseline = history[:-window]
        baseline_avg = float(np.mean(baseline)) if len(baseline) > 0 else recent_avg

        if baseline_avg == 0:
            return False

        relative_decline = (baseline_avg - recent_avg) / baseline_avg

        logger.info(
            f"[DriftDetector] Concept drift check — "
            f"baseline_avg={baseline_avg:.4f}, recent_avg={recent_avg:.4f}, "
            f"relative_decline={relative_decline:.4f}"
        )

        return relative_decline >= self.concept_threshold

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def check_drift(
        self,
        model_name: str,
        reference_images: List[str],
        current_images: List[str],
        metrics_history: Optional[List[float]] = None,
    ) -> DriftReport:
        """Full drift check pipeline for a deployed model.

        Extracts image-level statistics (aspect ratio, mean pixel intensity)
        from reference and current image sets, then computes PSI.

        Args:
            model_name: Identifier for the model being monitored
            reference_images: Paths to reference (training) images
            current_images: Paths to current production images
            metrics_history: Optional historical mAP values for concept drift

        Returns:
            DriftReport with scores and recommendation
        """
        logger.info(
            f"[DriftDetector] Drift check — model={model_name}, "
            f"ref={len(reference_images)}, current={len(current_images)}"
        )

        # Extract features from images
        ref_features = self._extract_image_features(reference_images)
        cur_features = self._extract_image_features(current_images)

        # Overall PSI
        overall_psi = self.compute_psi(ref_features, cur_features)

        # Per-feature PSI (aspect_ratio, mean_intensity)
        ref_ar = np.array([f["aspect_ratio"] for f in ref_features])
        cur_ar = np.array([f["aspect_ratio"] for f in cur_features])
        ref_mi = np.array([f["mean_intensity"] for f in ref_features])
        cur_mi = np.array([f["mean_intensity"] for f in cur_features])

        feature_drift: Dict[str, float] = {
            "aspect_ratio_psi": self.compute_psi(ref_ar, cur_ar),
            "mean_intensity_psi": self.compute_psi(ref_mi, cur_mi),
        }

        # Concept drift
        concept_drift = False
        if metrics_history:
            concept_drift = self.detect_concept_drift(metrics_history)

        # Recommendation
        if overall_psi >= self.psi_threshold or concept_drift:
            recommendation = "retrain"
        elif overall_psi >= self.psi_threshold * 0.5:
            recommendation = "monitor"
        else:
            recommendation = "ok"

        logger.info(
            f"[DriftDetector] Recommendation={recommendation}, "
            f"overall_psi={overall_psi:.4f}, concept_drift={concept_drift}"
        )

        return DriftReport(
            data_drift_score=overall_psi,
            concept_drift_detected=concept_drift,
            feature_drift=feature_drift,
            recommendation=recommendation,
            details={
                "model_name": model_name,
                "reference_count": len(reference_images),
                "current_count": len(current_images),
            },
        )

    # ------------------------------------------------------------------
    # Helper: extract lightweight features from images
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_image_features(image_paths: List[str]) -> List[Dict[str, float]]:
        """Extract lightweight statistics from images (no deep features needed).

        Returns a list of dicts with:
          - aspect_ratio: H / W
          - mean_intensity: mean pixel value

        Gracefully skips unreadable images.
        """
        features: List[Dict[str, float]] = []
        for path_str in image_paths:
            path = Path(path_str)
            if not path.exists():
                continue
            try:
                import cv2  # type: ignore

                img = cv2.imread(str(path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                mean_int = float(np.mean(img))
                features.append({
                    "aspect_ratio": h / max(w, 1),
                    "mean_intensity": mean_int,
                })
            except Exception as e:
                logger.debug(f"[DriftDetector] Could not read image {path_str}: {e}")
                continue

        if not features:
            logger.warning("[DriftDetector] No valid images found for feature extraction")
            # Return dummy feature to avoid division-by-zero in PSI
            features.append({"aspect_ratio": 1.0, "mean_intensity": 128.0})

        return features
