"""
YOLO11 Training Runner with Ray Tune HPO integration.

Based on Ultralytics official best practices:
- https://docs.ultralytics.com/usage/cfg/
- https://docs.ultralytics.com/integrations/ray-tune/
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import logging

import torch
from ultralytics import YOLO

from src.training.mlflow_tracker import MLflowTracker

from .config import (
    TrainingConfig,
    SanityCheckConfig,
    HPOConfig,
    ExportConfig,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_SANITY_CHECK_CONFIG,
    DEFAULT_HPO_CONFIG,
    DEFAULT_EXPORT_CONFIG,
)


@dataclass
class TrainingResult:
    """Training result container."""
    status: str
    model_path: Optional[Path] = None
    metrics: Optional[Dict[str, float]] = None
    best_params: Optional[Dict[str, float]] = None
    error: Optional[str] = None


def setup_gpu_memory() -> None:
    """Setup GPU memory management to prevent OOM errors."""
    if torch.cuda.is_available():
        # Clear cache before training
        torch.cuda.empty_cache()

        # Set memory growth limit to 80% of available GPU memory
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_per_process_memory_fraction(0.8, device=i)
            except Exception:
                pass  # Gracefully handle if not supported


def cleanup_gpu_memory() -> None:
    """Cleanup GPU memory after training."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class YOLOTrainer:
    """YOLO11 Trainer with HPO support."""

    def __init__(
        self,
        model: str = "yolo11m",
        output_dir: Path = None,
    ):
        self.model_name = model
        self.output_dir = Path(output_dir or "./runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def sanity_check(
        self,
        data_yaml: Path,
        config: SanityCheckConfig = None,
    ) -> TrainingResult:
        """
        Run sanity check to verify training feasibility.

        Args:
            data_yaml: Path to dataset YAML
            config: Sanity check configuration

        Returns:
            TrainingResult with status and metrics
        """
        config = config or DEFAULT_SANITY_CHECK_CONFIG
        model = YOLO(f"{self.model_name}.pt")

        results = model.train(
            data=str(data_yaml),
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            patience=config.patience,
            cache=config.cache,
            project=str(self.output_dir),
            name="sanity_check",
            exist_ok=True,
            verbose=False,
        )

        map50 = results.results_dict.get("metrics/mAP50(B)", 0)
        map50_95 = results.results_dict.get("metrics/mAP50-95(B)", 0)
        passed = map50 >= config.min_map50

        return TrainingResult(
            status="passed" if passed else "failed",
            model_path=Path(results.save_dir) / "weights" / "best.pt" if passed else None,
            metrics={
                "mAP50": map50,
                "mAP50-95": map50_95,
            },
        )

    def train(
        self,
        data_yaml: Path,
        epochs: int = None,
        config: TrainingConfig = None,
    ) -> TrainingResult:
        """
        Train YOLO model with given configuration.

        Args:
            data_yaml: Path to dataset YAML
            epochs: Number of epochs
            config: Training configuration

        Returns:
            TrainingResult with trained model
        """
        config = config or DEFAULT_TRAINING_CONFIG
        epochs = epochs or config.epochs

        # Setup GPU memory management
        setup_gpu_memory()

        # Initialize MLflow tracker with graceful degradation
        tracker = None
        mlflow_enabled = True
        try:
            tracker = MLflowTracker(experiment_name="yolo-training")
            tracker.start_run(
                run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        except Exception as e:
            mlflow_enabled = False
            logging.warning(f"MLflow tracking disabled: {e}")

        # Log training parameters
        if tracker and mlflow_enabled:
            try:
                tracker.log_params({
                    "model": self.model_name,
                    "epochs": epochs,
                    "data_yaml": str(data_yaml),
                    "batch_size": config.batch,
                    "image_size": config.imgsz,
                    "lr0": config.lr0,
                    "lrf": config.lrf,
                    "momentum": config.momentum,
                    "weight_decay": config.weight_decay,
                })
            except Exception as e:
                logging.warning(f"Failed to log parameters to MLflow: {e}")

        model = YOLO(f"{self.model_name}.pt")

        try:
            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=config.imgsz,
                batch=config.batch,
                project=str(self.output_dir),
                name="train",
                exist_ok=True,
                **config.to_dict(),
            )

            # Log metrics to MLflow
            if tracker and mlflow_enabled:
                try:
                    if hasattr(results, 'results_dict') and results.results_dict:
                        tracker.log_metrics(results.results_dict)
                except Exception as e:
                    logging.warning(f"Failed to log metrics to MLflow: {e}")

            # Log model artifact
            if tracker and mlflow_enabled:
                try:
                    model_path = Path(results.save_dir) / "weights" / "best.pt"
                    if model_path.exists():
                        tracker.log_artifact(str(model_path))
                except Exception as e:
                    logging.warning(f"Failed to log model artifact to MLflow: {e}")

            # End MLflow run
            if tracker and mlflow_enabled:
                try:
                    tracker.end_run(status="FINISHED")
                except Exception as e:
                    logging.warning(f"Failed to end MLflow run: {e}")

            return TrainingResult(
                status="completed",
                model_path=Path(results.save_dir) / "weights" / "best.pt",
                metrics={
                    "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                    "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
                },
            )
        except Exception as e:
            # End MLflow run with failed status
            if tracker and mlflow_enabled:
                try:
                    tracker.end_run(status="FAILED")
                except Exception:
                    pass

            return TrainingResult(
                status="failed",
                error=str(e),
            )
        finally:
            # Always cleanup GPU memory
            cleanup_gpu_memory()

    def tune(
        self,
        data_yaml: Path,
        config: HPOConfig = None,
    ) -> TrainingResult:
        """
        Run hyperparameter optimization with Ray Tune.

        Args:
            data_yaml: Path to dataset YAML
            config: HPO configuration

        Returns:
            TrainingResult with best parameters
        """
        config = config or DEFAULT_HPO_CONFIG

        from ray import tune

        model = YOLO(f"{self.model_name}.pt")

        # Build search space
        space = {}
        for param, (low, high) in config.param_space.items():
            space[param] = tune.uniform(low, high)

        result_grid = model.tune(
            data=str(data_yaml),
            space=space,
            epochs=config.epochs_per_trial,
            imgsz=config.imgsz,
            use_ray=True,
            grace_period=config.grace_period,
            project=str(self.output_dir / "hpo"),
        )

        best_result = result_grid.best_result
        best_params = {
            "lr0": best_result.config.get("lr0", DEFAULT_TRAINING_CONFIG.lr0),
            "lrf": best_result.config.get("lrf", DEFAULT_TRAINING_CONFIG.lrf),
            "momentum": best_result.config.get("momentum", DEFAULT_TRAINING_CONFIG.momentum),
            "weight_decay": best_result.config.get("weight_decay", DEFAULT_TRAINING_CONFIG.weight_decay),
            "box": best_result.config.get("box", DEFAULT_TRAINING_CONFIG.box),
            "cls": best_result.config.get("cls", DEFAULT_TRAINING_CONFIG.cls),
        }

        return TrainingResult(
            status="completed",
            best_params=best_params,
            metrics={
                "best_mAP50": best_result.metrics.get("metrics/mAP50(B)", 0),
            },
        )

    def export(
        self,
        model_path: Path,
        platform: str = "jetson",
        config: ExportConfig = None,
    ) -> Dict[str, Any]:
        """
        Export model to target format.

        Args:
            model_path: Path to trained model
            platform: Target platform (jetson/tensorrt/cpu)
            config: Export configuration

        Returns:
            Export result with model path and size
        """
        config = config or DEFAULT_EXPORT_CONFIG
        model = YOLO(str(model_path))

        platform_config = config.platform_configs.get(platform, config.platform_configs["jetson"])

        export_path = model.export(
            format=platform_config.get("format", config.format),
            half=platform_config.get("half", config.half),
            dynamic=platform_config.get("dynamic", config.dynamic),
            simplify=config.simplify,
            project=str(self.output_dir / "export"),
            exist_ok=True,
        )

        model_size_mb = Path(export_path).stat().st_size / (1024 * 1024)

        return {
            "model": export_path,
            "size_mb": model_size_mb,
            "platform": platform,
            "fp16": platform_config.get("half", config.half),
        }


class TransferLearningTrainer:
    """Transfer learning trainer using pretrained weights."""

    def __init__(
        self,
        teacher_model: str = "yolo11m",
        freeze_layers: int = 10,
    ):
        self.teacher_model_name = teacher_model
        self.freeze_layers = freeze_layers

    def train(
        self,
        data_yaml: Path,
        epochs: int = 100,
    ) -> TrainingResult:
        """
        Train with transfer learning (frozen backbone).

        Args:
            data_yaml: Path to dataset YAML
            epochs: Number of epochs

        Returns:
            TrainingResult with trained model
        """
        model = YOLO(f"{self.teacher_model_name}.pt")

        try:
            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                freeze=self.freeze_layers,
                project="./runs/transfer",
                name="student",
                verbose=False,
            )

            return TrainingResult(
                status="completed",
                model_path=Path(results.save_dir) / "weights" / "best.pt",
                metrics={
                    "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                },
            )
        except Exception as e:
            return TrainingResult(
                status="failed",
                error=str(e),
            )
