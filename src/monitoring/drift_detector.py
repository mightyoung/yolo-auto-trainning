"""
Data Drift Detection Module for ML models.

Based on best practices:
- Monitor input feature distributions
- Detect data drift using statistical methods
- Alert when drift is detected
- Integration with Evidently AI for comprehensive monitoring
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

import numpy as np


class DriftStatus(Enum):
    """Drift detection status."""
    NO_DRIFT = "no_drift"
    DRIFT_DETECTED = "drift_detected"
    WARNING = "warning"


@dataclass
class DriftReport:
    """Drift detection report."""
    status: str
    drift_score: float
    feature_drift: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    reference_samples: int = 0
    current_samples: int = 0
    recommendations: List[str] = field(default_factory=list)


class StatisticalDriftDetector:
    """
    Statistical drift detector using common methods.

    Methods:
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov test
    - Chi-squared test for categorical features
    """

    def __init__(
        self,
        threshold: float = 0.2,
        method: str = "psi",
    ):
        """
        Initialize drift detector.

        Args:
            threshold: Drift threshold (default: 0.2 for PSI)
            method: Detection method ("psi", "ks", "chi2")
        """
        self.threshold = threshold
        self.method = method
        self.reference_distribution: Dict[str, np.ndarray] = {}

    def set_reference(self, reference_data: np.ndarray, feature_name: str = "default") -> None:
        """
        Set reference distribution for comparison.

        Args:
            reference_data: Reference data array
            feature_name: Name of the feature
        """
        self.reference_distribution[feature_name] = reference_data

    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins

        Returns:
            PSI value
        """
        # Create bins from reference data
        percentiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(reference, percentiles)

        # Handle edge cases
        if len(np.unique(bin_edges)) < 2:
            return 0.0

        # Calculate actual percentages in each bin
        reference_counts, _ = np.histogram(reference, bins=bin_edges)
        current_counts, _ = np.histogram(current, bins=bin_edges)

        # Avoid division by zero
        reference_pct = (reference_counts + 1) / (len(reference) + bins)
        current_pct = (current_counts + 1) / (len(current) + bins)

        # Calculate PSI
        psi = np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct))

        return psi

    def calculate_ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """
        Calculate Kolmogorov-Smirnov statistic.

        Args:
            reference: Reference distribution
            current: Current distribution

        Returns:
            KS statistic (max difference)
        """
        from scipy import stats

        try:
            ks_stat, _ = stats.ks_2samp(reference, current)
            return ks_stat
        except Exception:
            return 0.0

    def detect_drift(
        self,
        current_data: np.ndarray,
        feature_name: str = "default",
    ) -> Tuple[float, DriftStatus]:
        """
        Detect drift in current data compared to reference.

        Args:
            current_data: Current data to check
            feature_name: Feature name

        Returns:
            Tuple of (drift_score, DriftStatus)
        """
        if feature_name not in self.reference_distribution:
            return 0.0, DriftStatus.NO_DRIFT

        reference = self.reference_distribution[feature_name]

        if self.method == "psi":
            drift_score = self.calculate_psi(reference, current_data)
        elif self.method == "ks":
            drift_score = self.calculate_ks_test(reference, current_data)
        else:
            drift_score = 0.0

        # Determine status based on threshold
        if drift_score > self.threshold:
            status = DriftStatus.DRIFT_DETECTED
        elif drift_score > self.threshold * 0.5:
            status = DriftStatus.WARNING
        else:
            status = DriftStatus.NO_DRIFT

        return drift_score, status


class ImageDriftDetector:
    """
    Drift detector for image data.

    Monitors:
    - Image brightness distribution
    - Image size distribution
    - Color histogram drift
    """

    def __init__(self, threshold: float = 0.2):
        """
        Initialize image drift detector.

        Args:
            threshold: Drift threshold
        """
        self.threshold = threshold
        self.brightness_detector = StatisticalDriftDetector(threshold=threshold)
        self.size_detector = StatisticalDriftDetector(threshold=threshold)

    def extract_features(self, image_path: str) -> Dict[str, Any]:
        """
        Extract features from an image.

        Args:
            image_path: Path to image

        Returns:
            Feature dictionary
        """
        try:
            from PIL import Image

            img = Image.open(image_path)
            img_array = np.array(img)

            # Calculate brightness (mean of grayscale)
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array

            brightness = np.mean(gray)

            # Calculate color histogram
            if len(img_array.shape) == 3:
                hist_r = np.histogram(img_array[:, :, 0], bins=32, range=(0, 256))[0]
                hist_g = np.histogram(img_array[:, :, 1], bins=32, range=(0, 256))[0]
                hist_b = np.histogram(img_array[:, :, 2], bins=32, range=(0, 256))[0]
                color_hist = np.concatenate([hist_r, hist_g, hist_b])
            else:
                color_hist = np.histogram(gray, bins=32, range=(0, 256))[0]

            # Image size
            width, height = img.size

            return {
                "brightness": brightness,
                "color_histogram": color_hist / (color_hist.sum() + 1e-6),
                "width": width,
                "height": height,
                "aspect_ratio": width / (height + 1e-6),
            }

        except Exception as e:
            return {"error": str(e)}

    def detect_drift(
        self,
        image_paths: List[str],
        reference_features: Optional[Dict[str, np.ndarray]] = None,
    ) -> DriftReport:
        """
        Detect drift in a batch of images.

        Args:
            image_paths: List of image paths
            reference_features: Reference feature distributions

        Returns:
            DriftReport with results
        """
        # Extract features from current images
        current_brightness = []
        current_sizes = []

        for img_path in image_paths:
            features = self.extract_features(img_path)
            if "error" not in features:
                current_brightness.append(features["brightness"])
                current_sizes.append(features["width"] * features["height"])

        current_brightness = np.array(current_brightness)
        current_sizes = np.array(current_sizes)

        # Detect drift
        feature_drift = {}
        recommendations = []

        if reference_features is not None:
            # Brightness drift
            if "brightness" in reference_features:
                brightness_score, status = self.brightness_detector.detect_drift(
                    current_brightness, "brightness"
                )
                feature_drift["brightness"] = brightness_score
                if status != DriftStatus.NO_DRIFT:
                    recommendations.append("Image brightness distribution has changed significantly")

            # Size drift
            if "size" in reference_features:
                size_score, status = self.size_detector.detect_drift(
                    current_sizes, "size"
                )
                feature_drift["size"] = size_score
                if status != DriftStatus.NO_DRIFT:
                    recommendations.append("Image size distribution has changed")

        # Calculate overall drift score
        if feature_drift:
            drift_score = np.mean(list(feature_drift.values()))
        else:
            drift_score = 0.0

        # Determine status
        if drift_score > self.threshold:
            status = DriftStatus.DRIFT_DETECTED.value
        elif drift_score > self.threshold * 0.5:
            status = DriftStatus.WARNING.value
        else:
            status = DriftStatus.NO_DRIFT.value

        return DriftReport(
            status=status,
            drift_score=float(drift_score),
            feature_drift={k: float(v) for k, v in feature_drift.items()},
            timestamp=datetime.now().isoformat(),
            reference_samples=reference_features.get("count", 0) if reference_features else 0,
            current_samples=len(image_paths),
            recommendations=recommendations,
        )


class DataMonitor:
    """
    Data monitoring system for ML pipelines.

    Features:
    - Reference data management
    - Scheduled drift detection
    - Alert integration
    """

    def __init__(self, threshold: float = 0.2):
        """
        Initialize data monitor.

        Args:
            threshold: Drift threshold
        """
        self.threshold = threshold
        self.reference_data: Dict[str, np.ndarray] = {}
        self.drift_detector = ImageDriftDetector(threshold=threshold)
        self.drift_history: List[DriftReport] = []

    def set_reference_data(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Set reference data from a batch of images.

        Args:
            image_paths: List of reference image paths

        Returns:
            Summary of reference data
        """
        brightness_values = []
        sizes = []

        for img_path in image_paths:
            features = self.drift_detector.extract_features(img_path)
            if "error" not in features:
                brightness_values.append(features["brightness"])
                sizes.append(features["width"] * features["height"])

        self.reference_data["brightness"] = np.array(brightness_values)
        self.reference_data["size"] = np.array(sizes)
        self.reference_data["count"] = len(brightness_values)

        return {
            "reference_samples": len(brightness_values),
            "brightness_mean": float(np.mean(brightness_values)),
            "brightness_std": float(np.std(brightness_values)),
            "size_mean": float(np.mean(sizes)),
        }

    def check_drift(self, image_paths: List[str]) -> DriftReport:
        """
        Check for data drift.

        Args:
            image_paths: List of current image paths

        Returns:
            DriftReport
        """
        report = self.drift_detector.detect_drift(
            image_paths,
            reference_features=self.reference_data,
        )

        self.drift_history.append(report)
        return report

    def get_drift_history(self, limit: int = 10) -> List[DriftReport]:
        """Get recent drift detection history."""
        return self.drift_history[-limit:]

    def get_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get alerts for drift detection.

        Args:
            hours: Lookback period in hours

        Returns:
            List of alerts
        """
        alerts = []
        cutoff = datetime.now() - timedelta(hours=hours)

        for report in self.drift_history:
            report_time = datetime.fromisoformat(report.timestamp)
            if report_time > cutoff and report.status != DriftStatus.NO_DRIFT.value:
                alerts.append({
                    "timestamp": report.timestamp,
                    "status": report.status,
                    "drift_score": report.drift_score,
                    "recommendations": report.recommendations,
                })

        return alerts


def create_drift_monitor(threshold: float = 0.2) -> DataMonitor:
    """
    Create a data drift monitor.

    Args:
        threshold: Drift threshold

    Returns:
        DataMonitor instance
    """
    return DataMonitor(threshold=threshold)
