"""
Auto Label Client for Business API
Location: business-api/src/api/auto_label_client.py

This module provides client for calling Auto Label service on GPU server.
"""

import os
import requests
from typing import List, Optional, Dict, Any


class AutoLabelClient:
    """Client for Auto Label service on GPU server."""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None
    ):
        """
        Initialize Auto Label client.

        Args:
            base_url: Base URL for training API
            api_key: API key for authentication
        """
        self.base_url = base_url or os.getenv(
            "TRAINING_API_URL",
            "http://localhost:8001"
        )
        self.api_key = api_key or os.getenv("TRAINING_API_KEY", "default-key")

    def health_check(self) -> bool:
        """Check if Auto Label service is available."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def submit_labeling_job(
        self,
        input_folder: str,
        classes: List[str],
        base_model: str = "grounded_sam",
        conf_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Submit an auto-labeling job.

        Args:
            input_folder: Path to folder containing images
            classes: List of class names to label
            base_model: Base model to use
            conf_threshold: Confidence threshold

        Returns:
            Job submission result with task_id
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/label/submit",
                json={
                    "input_folder": input_folder,
                    "classes": classes,
                    "base_model": base_model,
                    "conf_threshold": conf_threshold
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"error": str(e)}

    def get_labeling_status(
        self,
        task_id: str
    ) -> Dict[str, Any]:
        """
        Get labeling job status.

        Args:
            task_id: Task ID

        Returns:
            Job status
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/label/status/{task_id}",
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"error": str(e)}

    def submit_distillation_job(
        self,
        data_yaml: str,
        target_model: str = "yolov8",
        model_size: str = "n",
        epochs: int = 100
    ) -> Dict[str, Any]:
        """
        Submit a model distillation job.

        Args:
            data_yaml: Path to labeled dataset YAML
            target_model: Target model type
            model_size: Model size
            epochs: Training epochs

        Returns:
            Job submission result
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/train/distill",
                json={
                    "data_yaml": data_yaml,
                    "target_model": target_model,
                    "model_size": model_size,
                    "epochs": epochs
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"error": str(e)}
