"""
MLflow Tracking Module for YOLO Training.

Integrates MLflow experiment tracking with YOLO training pipeline.
Based on Ultralytics official MLflow integration:
- https://docs.ultralytics.com/integrations/mlflow/
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import mlflow
from mlflow.tracking import MlflowClient


class MLflowTracker:
    """
    MLflow experiment tracker for YOLO training.

    Features:
    - Automatic parameter and metric logging
    - Model artifact tracking
    - Experiment versioning
    - Run management
    """

    def __init__(
        self,
        experiment_name: str = "yolo-training",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (None for local)
            artifact_location: Artifact store location
        """
        self.experiment_name = experiment_name

        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

        # Set artifact location
        if artifact_location:
            mlflow.set_artifact_root_directory(artifact_location)

        # Create or get experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if not self.experiment:
                self.experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location or "./mlruns"
                )
            else:
                self.experiment_id = self.experiment.experiment_id
        except Exception as e:
            # Fallback for local tracking
            mlflow.set_tracking_uri("file://./mlruns")
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location or "./mlruns"
            )

        mlflow.set_experiment(experiment_name)

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> mlflow.ActiveRun:
        """
        Start an MLflow run.

        Args:
            run_name: Name for the run
            tags: Tags for the run

        Returns:
            Active MLflow run
        """
        return mlflow.start_run(
            run_name=run_name,
            experiment_id=self.experiment_id,
            tags=tags,
        )

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameters
        """
        # Flatten nested params
        flat_params = self._flatten_dict(params)
        for key, value in flat_params.items():
            if isinstance(value, (int, float, str, bool)):
                try:
                    mlflow.log_param(key, value)
                except Exception:
                    pass  # Skip invalid params

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics
            step: Training step/epoch
        """
        # Flatten nested metrics
        flat_metrics = self._flatten_dict(metrics)
        for key, value in flat_metrics.items():
            if isinstance(value, (int, float)):
                try:
                    mlflow.log_metric(
                        mlflow.entities.Metric(key, float(value), step or 0, 0)
                    )
                except Exception:
                    pass  # Skip invalid metrics

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact (file or directory).

        Args:
            local_path: Path to local file/directory
            artifact_path: Path in artifact store
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            print(f"Warning: Failed to log artifact {local_path}: {e}")

    def log_model(
        self,
        model_path: str,
        name: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a trained model as artifact.

        Args:
            model_path: Path to model weights
            name: Model name
            metadata: Model metadata
        """
        if Path(model_path).exists():
            self.log_artifact(model_path, f"models/{name}")

            # Log model metadata if provided
            if metadata:
                for key, value in metadata.items():
                    try:
                        mlflow.log_param(f"{name}_{key}", value)
                    except Exception:
                        pass

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        mlflow.end_run(status=status)

    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
        """
        Flatten nested dictionary.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowTracker._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


def enable_yolo_mlflow_logging() -> None:
    """
    Enable automatic MLflow logging for YOLO training.

    This uses Ultralytics built-in MLflow integration.
    Based on: https://docs.ultralytics.com/integrations/mlflow/
    """
    # Set environment variables for YOLO MLflow integration
    os.environ["MLFLOW_TRACKING_URI"] = os.getenv(
        "MLFLOW_TRACKING_URI",
        "file:./mlruns"
    )


def get_tracking_uri() -> str:
    """Get current MLflow tracking URI."""
    return mlflow.get_tracking_uri()


def list_experiments() -> list:
    """List all experiments."""
    return mlflow.list_experiments()


def get_run(run_id: str) -> mlflow.entities.Run:
    """Get a run by ID."""
    return mlflow.get_run(run_id)


# ============================================================
# Model Registry Functions
# Based on MLflow best practices:
# https://mlflow.org/docs/latest/ml/model-registry/
# ============================================================

def register_model(
    name: str,
    version: int,
    stage: str = "Staging",
    description: str = "",
) -> Optional[mlflow.entities.ModelVersion]:
    """
    Register a model version to Model Registry.

    Args:
        name: Registered model name
        version: Model version
        stage: Target stage (Staging, Production, Archived)
        description: Model description

    Returns:
        ModelVersion object or None
    """
    try:
        client = MlflowClient()
        model_version = client.get_model_version(name, version)
        if stage:
            client.transition_model_version_stage(name, version, stage)
        return model_version
    except Exception as e:
        print(f"Error registering model: {e}")
        return None


def create_registered_model(
    name: str,
    description: str = "",
    tags: Optional[Dict[str, str]] = None,
) -> Optional[mlflow.entities.RegisteredModel]:
    """
    Create a new registered model.

    Args:
        name: Model name
        description: Model description
        tags: Model tags

    Returns:
        RegisteredModel object or None
    """
    try:
        client = MlflowClient()
        return client.create_registered_model(name, description, tags)
    except Exception as e:
        print(f"Error creating registered model: {e}")
        return None


def list_registered_models() -> list:
    """
    List all registered models.

    Returns:
        List of RegisteredModel objects
    """
    try:
        client = MlflowClient()
        return client.list_registered_models()
    except Exception:
        return []


def get_model_versions(name: str) -> list:
    """
    Get all versions of a registered model.

    Args:
        name: Registered model name

    Returns:
        List of ModelVersion objects
    """
    try:
        client = MlflowClient()
        return client.get_model_version_download_uri(name, 1)
    except Exception:
        return []


def get_model_version(name: str, version: int) -> Optional[mlflow.entities.ModelVersion]:
    """
    Get a specific model version.

    Args:
        name: Registered model name
        version: Model version

    Returns:
        ModelVersion object or None
    """
    try:
        client = MlflowClient()
        return client.get_model_version(name, version)
    except Exception:
        return None


def transition_model_stage(
    name: str,
    version: int,
    stage: str,
) -> Optional[mlflow.entities.ModelVersion]:
    """
    Transition a model version to a different stage.

    Args:
        name: Registered model name
        version: Model version
        stage: Target stage (None, Staging, Production, Archived)

    Returns:
        ModelVersion object or None
    """
    try:
        client = MlflowClient()
        return client.transition_model_version_stage(name, version, stage)
    except Exception as e:
        print(f"Error transitioning model stage: {e}")
        return None


def delete_model_version(name: str, version: int) -> bool:
    """
    Delete a model version.

    Args:
        name: Registered model name
        version: Model version

    Returns:
        True if successful
    """
    try:
        client = MlflowClient()
        client.delete_model_version(name, version)
        return True
    except Exception:
        return False


def delete_registered_model(name: str) -> bool:
    """
    Delete a registered model and all its versions.

    Args:
        name: Registered model name

    Returns:
        True if successful
    """
    try:
        client = MlflowClient()
        client.delete_registered_model(name)
        return True
    except Exception:
        return False


def get_latest_model_versions(name: str, stage: Optional[str] = None) -> list:
    """
    Get latest versions of a model, optionally filtered by stage.

    Args:
        name: Registered model name
        stage: Filter by stage (Staging, Production)

    Returns:
        List of ModelVersion objects
    """
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(name, stage)
        return versions or []
    except Exception:
        return []
