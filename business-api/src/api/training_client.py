"""
Training API Client - Communication layer between Business API and Training API

This module provides a client for the training API running on the GPU server.
"""

from typing import Optional, Dict, Any
import httpx
import json
import uuid
from datetime import datetime
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
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

    async def start_training(
        self,
        task_id: str,
        model: str,
        data_yaml: str,
        epochs: int,
        imgsz: int = 640,
        output_dir: str = "/runs",
        batch: int = 16,
        device: str = "cuda:0",
        augmentation_preset: Optional[str] = None,
        resume_from: Optional[str] = None,
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
            batch: Batch size
            device: Device (e.g., cuda:0, cuda:1)
            augmentation_preset: Augmentation preset (balanced|strong)
            resume_from: Path to best.pt for transfer learning

        Returns:
            Response containing task_id and status
        """
        payload = {
            "task_id": task_id,
            "model": model,
            "data_yaml": data_yaml.replace("\\", "/"),
            "epochs": epochs,
            "imgsz": imgsz,
            "output_dir": output_dir,
            "batch": batch,
            "device": device,
        }
        if augmentation_preset:
            payload["augmentation_preset"] = augmentation_preset
        if resume_from:
            payload["resume_from"] = resume_from

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/internal/train/start",
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()

    def start_training_sync(
        self,
        task_id: str,
        model: str,
        data_yaml: str,
        epochs: int,
        imgsz: int = 640,
        output_dir: str = "/runs",
        batch: int = 16,
        device: str = "cuda:0",
        augmentation_preset: Optional[str] = None,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous version of start_training. Safe to call from background threads."""
        payload = {
            "task_id": task_id,
            "model": model,
            "data_yaml": data_yaml.replace("\\", "/"),
            "epochs": epochs,
            "imgsz": imgsz,
            "output_dir": output_dir,
            "batch": batch,
            "device": device,
        }
        if augmentation_preset:
            payload["augmentation_preset"] = augmentation_preset
        if resume_from:
            payload["resume_from"] = resume_from

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/v1/internal/train/start",
                json=payload,
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
                    "data_yaml": data_yaml.replace("\\", "/"),
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

    def get_task_status_sync(self, task_id: str) -> Dict[str, Any]:
        """
        Synchronous version of get_task_status.
        Safe to call from background threads (no asyncio.run() needed).

        Args:
            task_id: Task identifier

        Returns:
            Task status including progress and metrics
        """
        with httpx.Client(timeout=30) as client:
            response = client.get(
                f"{self.base_url}/api/v1/internal/train/status/{task_id}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()

    def start_export_sync(
        self,
        task_id: str,
        model_path: str,
        platform: str = "jetson_orin",
        formats: Optional[list] = None,
        imgsz: int = 640,
        int8_quantize: bool = False,
    ) -> Dict[str, Any]:
        """
        Synchronous version of start_export. Safe to call from background threads.

        Args:
            task_id: Unique task identifier
            model_path: Path to trained model
            platform: Target platform (jetson_nano, jetson_orin, rk3588)
            formats: List of formats (e.g. ["onnx", "engine"])
            imgsz: Input image size
            int8_quantize: Whether to apply INT8 quantization

        Returns:
            Response containing task_id and status
        """
        payload = {
            "task_id": task_id,
            "model_path": model_path,
            "platform": platform,
            "imgsz": imgsz,
            "int8_quantize": int8_quantize,
        }
        if formats:
            payload["formats"] = formats

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/v1/internal/export/start",
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()

    def get_export_status_sync(self, task_id: str) -> Dict[str, Any]:
        """
        Synchronous version of get export task status.
        Safe to call from background threads.

        Args:
            task_id: Task identifier

        Returns:
            Export task status
        """
        with httpx.Client(timeout=30) as client:
            response = client.get(
                f"{self.base_url}/api/v1/internal/export/status/{task_id}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()

    def submit_deployment(
        self,
        model_path: str,
        platform: str = "jetson_orin",
        imgsz: int = 640,
    ) -> Dict[str, Any]:
        """
        Submit a deployment by fetching the edge device configuration.
        The deployment is "registered" by retrieving the optimal config for the target device.
        This stores the config in Redis keyed by task_id.

        Args:
            model_path: Path to the exported model
            platform: Target edge platform (jetson_orin, rk3588, etc.)
            imgsz: Input image size

        Returns:
            Edge device configuration for deployment
        """
        import os
        r = self._get_redis()
        deploy_id = f"deploy_{uuid.uuid4().hex[:8]}"

        with httpx.Client(timeout=30) as client:
            response = client.get(
                f"{self.base_url}/api/v1/internal/deploy/edge-config/{model_path}",
                params={"device": platform, "imgsz": imgsz},
                headers=self._get_headers()
            )
            response.raise_for_status()
            config = response.json()

        # Store deployment config in Redis
        r.hset(f"deploy:{deploy_id}", mapping={
            "model_path": model_path,
            "platform": platform,
            "imgsz": str(imgsz),
            "config": json.dumps(config),
            "status": "deployed",
            "deployed_at": datetime.now().isoformat(),
        })

        return {
            "deploy_id": deploy_id,
            "model_path": model_path,
            "platform": platform,
            "config": config,
            "status": "deployed",
        }

    def _get_redis(self):
        """Get Redis client for deployment registration."""
        import redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_password = os.getenv("REDIS_PASSWORD", None)
        try:
            if redis_password:
                return redis.from_url(redis_url, password=redis_password, decode_responses=True)
            return redis.from_url(redis_url, decode_responses=True)
        except Exception:
            try:
                return redis.Redis(host="localhost", port=6379, db=0, decode_responses=True, password=redis_password)
            except Exception:
                return None

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

    def start_curriculum_sync(
        self,
        task_id: str,
        data_yaml: str,
        output_dir: str = "/home/wangxin/runs",
        device: str = "cuda:0",
        auto_export: bool = True,
        stage1_min_map: float = 0.50,
        stage2_target_map: float = 0.90,
        stage2_min_for_stage3: float = 0.80,
        stage1_overrides: Optional[dict] = None,
        stage2_overrides: Optional[dict] = None,
        stage3_overrides: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Start a 3-stage curriculum training. Safe to call from background threads."""
        payload = {
            "task_id": task_id,
            "data_yaml": data_yaml.replace("\\", "/"),
            "output_dir": output_dir,
            "device": device,
            "auto_export": auto_export,
            "stage1_min_map": stage1_min_map,
            "stage2_target_map": stage2_target_map,
            "stage2_min_for_stage3": stage2_min_for_stage3,
        }
        if stage1_overrides:
            payload["stage1"] = stage1_overrides
        if stage2_overrides:
            payload["stage2"] = stage2_overrides
        if stage3_overrides:
            payload["stage3"] = stage3_overrides

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/v1/internal/train/curriculum/start",
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()


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
