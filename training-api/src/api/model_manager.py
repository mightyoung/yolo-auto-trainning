"""
Model Storage Manager
Location: training-api/src/api/model_manager.py

Manages model versions, storage, and retrieval.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


class ModelManager:
    """Manages model versions and storage."""

    def __init__(self, storage_dir: str = "/models"):
        """
        Initialize model manager.

        Args:
            storage_dir: Base directory for model storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Metadata directory
        self.metadata_dir = self.storage_dir / ".metadata"
        self.metadata_dir.mkdir(exist_ok=True)

    def save_model(
        self,
        model_path: str,
        task_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a trained model to storage.

        Args:
            model_path: Path to the model file
            task_id: Task identifier
            metadata: Additional metadata

        Returns:
            Model ID
        """
        model_id = f"model_{uuid.uuid4().hex[:12]}"

        # Create model directory
        model_dir = self.storage_dir / task_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy model file
        dest_path = model_dir / "weights.pt"
        if Path(model_path).exists():
            shutil.copy2(model_path, dest_path)

        # Save metadata
        model_metadata = {
            "model_id": model_id,
            "task_id": task_id,
            "created_at": datetime.now().isoformat(),
            "model_path": str(dest_path),
            "metadata": metadata or {}
        }

        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)

        return model_id

    def get_model(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model metadata by task ID.

        Args:
            task_id: Task identifier

        Returns:
            Model metadata or None
        """
        model_dir = self.storage_dir / task_id
        metadata_path = model_dir / "metadata.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def list_models(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all models.

        Args:
            limit: Maximum number of models to return

        Returns:
            List of model metadata
        """
        models = []

        for task_dir in self.storage_dir.iterdir():
            if not task_dir.is_dir() or task_dir.name.startswith('.'):
                continue

            metadata_path = task_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    models.append(json.load(f))

            if len(models) >= limit:
                break

        return models

    def delete_model(self, task_id: str) -> bool:
        """
        Delete a model.

        Args:
            task_id: Task identifier

        Returns:
            True if deleted, False if not found
        """
        model_dir = self.storage_dir / task_id

        if not model_dir.exists():
            return False

        shutil.rmtree(model_dir)
        return True

    def export_model(
        self,
        task_id: str,
        export_path: str,
        format: str = "onnx"
    ) -> str:
        """
        Export model to specified format.

        Args:
            task_id: Task identifier
            export_path: Path to export to
            format: Export format (onnx, tensorrt, etc.)

        Returns:
            Path to exported model
        """
        model_metadata = self.get_model(task_id)
        if not model_metadata:
            raise ValueError(f"Model {task_id} not found")

        # TODO: Implement actual export using Exporter
        return export_path
