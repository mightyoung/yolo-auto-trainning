"""
Training API Client - Communication layer between Business API and Training API

This module provides a client for the training API running on the GPU server.
"""

from typing import Optional, Dict, Any
import httpx
from pydantic import BaseModel


class TrainingAPIClient:
    """Client for communicating with the Training API on GPU server."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 300
    ):
        """
        Initialize the training API client.

        Args:
            base_url: Base URL of the training API (e.g., http://localhost:8001)
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def start_training(
        self,
        task_id: str,
        model: str,
        data_yaml: str,
        epochs: int,
        imgsz: int = 640,
        output_dir: str = "/runs"
    ) -> Dict[str, Any]:
        """
        Submit a training job to the training API.

        Args:
            task_id: Unique task identifier
            model: YOLO model size (n/s/m/l/x)
            data_yaml: Path to dataset YAML
            epochs: Number of training epochs
            imgsz: Input image size
            output_dir: Output directory for training results

        Returns:
            Response containing task_id and status
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/internal/train/start",
                json={
                    "task_id": task_id,
                    "model": model,
                    "data_yaml": data_yaml,
                    "epochs": epochs,
                    "imgsz": imgsz,
                    "output_dir": output_dir
                },
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()

    async def start_hpo(
        self,
        task_id: str,
        model: str,
        data_yaml: str,
        n_trials: int = 50,
        epochs_per_trial: int = 50
    ) -> Dict[str, Any]:
        """
        Submit an HPO job to the training API.

        Args:
            task_id: Unique task identifier
            model: YOLO model size
            data_yaml: Path to dataset YAML
            n_trials: Number of trials
            epochs_per_trial: Epochs per trial

        Returns:
            Response containing task_id and status
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/internal/hpo/start",
                json={
                    "task_id": task_id,
                    "model": model,
                    "data_yaml": data_yaml,
                    "n_trials": n_trials,
                    "epochs_per_trial": epochs_per_trial
                },
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()

    async def start_export(
        self,
        task_id: str,
        model_path: str,
        platform: str = "jetson_orin",
        imgsz: int = 640
    ) -> Dict[str, Any]:
        """
        Submit a model export job.

        Args:
            task_id: Unique task identifier
            model_path: Path to trained model
            platform: Target platform (jetson_nano, jetson_orin, rk3588)
            imgsz: Input image size

        Returns:
            Response containing task_id and status
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/internal/export/start",
                json={
                    "task_id": task_id,
                    "model_path": model_path,
                    "platform": platform,
                    "imgsz": imgsz
                },
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a training task.

        Args:
            task_id: Task identifier

        Returns:
            Task status including progress and metrics
        """
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}/api/v1/internal/train/status/{task_id}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()

    async def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        Cancel a running task.

        Args:
            task_id: Task identifier

        Returns:
            Cancellation confirmation
        """
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/internal/train/cancel/{task_id}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()

    async def health_check(self) -> bool:
        """
        Check if the training API is available.

        Returns:
            True if healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self.base_url}/health",
                    headers=self._get_headers()
                )
                return response.status_code == 200
        except Exception:
            return False


class TaskStatus(BaseModel):
    """Task status model."""
    task_id: str
    status: str  # submitted, running, completed, failed
    progress: float = 0.0
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
