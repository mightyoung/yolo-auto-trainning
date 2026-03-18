"""
YOLO11 Training Runner with Ray Tune HPO integration.

Based on Ultralytics official best practices:
- https://docs.ultralytics.com/usage/cfg/
- https://docs.ultralytics.com/integrations/ray-tune/
- https://docs.ultralytics.com/integrations/mlflow/
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

from ultralytics import YOLO

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


class YOLOTrainer:
    """YOLO11 Trainer with HPO and MLflow support."""

    def __init__(
        self,
        model: str = "yolo11m",
        output_dir: Path = None,
        mlflow_tracker: Any = None,
    ):
        """
        Initialize YOLO Trainer.

        Args:
            model: YOLO model name (e.g., yolo11m, yolo11n)
            output_dir: Output directory for training runs
            mlflow_tracker: Optional MLflowTracker instance for experiment tracking
        """
        self.model_name = model
        self.output_dir = Path(output_dir or "./runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_tracker = mlflow_tracker
        self.current_run = None

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

        # Start MLflow run if tracker is available
        if self.mlflow_tracker:
            self.current_run = self.mlflow_tracker.start_run(
                run_name=f"sanity_check_{self.model_name}"
            )
            # Log parameters
            self.mlflow_tracker.log_params({
                "task_type": "sanity_check",
                "model": self.model_name,
                "data_yaml": str(data_yaml),
                "epochs": config.epochs,
                "imgsz": config.imgsz,
                "batch": config.batch,
            })

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

        # Log metrics to MLflow
        if self.mlflow_tracker:
            self.mlflow_tracker.log_metrics({
                "sanity_check/mAP50": map50,
                "sanity_check/mAP50-95": map50_95,
                "sanity_check/passed": 1 if passed else 0,
            })
            self.mlflow_tracker.log_model(
                str(Path(results.save_dir) / "weights" / "best.pt"),
                name="sanity_check_model"
            )
            self.mlflow_tracker.end_run("FINISHED" if passed else "FAILED")

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

        # Start MLflow run if tracker is available
        if self.mlflow_tracker:
            self.current_run = self.mlflow_tracker.start_run(
                run_name=f"train_{self.model_name}"
            )
            # Log parameters
            params = {
                "task_type": "train",
                "model": self.model_name,
                "data_yaml": str(data_yaml),
                "epochs": epochs,
                "imgsz": config.imgsz,
                "batch": config.batch,
            }
            params.update(config.to_dict())
            self.mlflow_tracker.log_params(params)

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
            if self.mlflow_tracker:
                metrics = {
                    "train/mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                    "train/mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
                }
                self.mlflow_tracker.log_metrics(metrics)
                self.mlflow_tracker.log_model(
                    str(Path(results.save_dir) / "weights" / "best.pt"),
                    name="trained_model"
                )
                self.mlflow_tracker.end_run("FINISHED")

            return TrainingResult(
                status="completed",
                model_path=Path(results.save_dir) / "weights" / "best.pt",
                metrics={
                    "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                    "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
                },
            )
        except Exception as e:
            if self.mlflow_tracker:
                self.mlflow_tracker.log_metrics({"train/error": 1})
                self.mlflow_tracker.end_run("FAILED")
            return TrainingResult(
                status="failed",
                error=str(e),
            )

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

        # Start MLflow run if tracker is available
        if self.mlflow_tracker:
            self.current_run = self.mlflow_tracker.start_run(
                run_name=f"hpo_{self.model_name}"
            )
            # Log HPO parameters
            params = {
                "task_type": "hpo",
                "model": self.model_name,
                "data_yaml": str(data_yaml),
                "epochs_per_trial": config.epochs_per_trial,
                "num_trials": config.num_trials,
            }
            params.update({f"search/{k}": v for k, v in config.param_space.items()})
            self.mlflow_tracker.log_params(params)

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

        # Log best results to MLflow
        if self.mlflow_tracker:
            self.mlflow_tracker.log_metrics({
                "hpo/best_mAP50": best_result.metrics.get("metrics/mAP50(B)", 0),
            })
            self.mlflow_tracker.log_params({f"best/{k}": v for k, v in best_params.items()})
            self.mlflow_tracker.end_run("FINISHED")

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

        # Start MLflow run if tracker is available
        if self.mlflow_tracker:
            self.current_run = self.mlflow_tracker.start_run(
                run_name=f"export_{self.model_name}_{platform}"
            )
            self.mlflow_tracker.log_params({
                "task_type": "export",
                "model": self.model_name,
                "platform": platform,
                "model_path": str(model_path),
            })

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

        # Log export results to MLflow
        if self.mlflow_tracker:
            self.mlflow_tracker.log_metrics({
                "export/size_mb": model_size_mb,
            })
            self.mlflow_tracker.log_artifact(export_path, "exported_models")
            self.mlflow_tracker.end_run("FINISHED")

        return {
            "model": export_path,
            "size_mb": model_size_mb,
            "platform": platform,
            "fp16": platform_config.get("half", config.half),
        }


class KnowledgeDistillationTrainer:
    """Knowledge distillation trainer using official teacher API."""

    def __init__(
        self,
        teacher_model: str = "yolo11m",
        student_model: str = "yolo11n",
    ):
        self.teacher_model_name = teacher_model
        self.student_model_name = student_model

    def train(
        self,
        data_yaml: Path,
        epochs: int = 100,
    ) -> TrainingResult:
        """
        Train with knowledge distillation.

        Args:
            data_yaml: Path to dataset YAML
            epochs: Number of epochs

        Returns:
            TrainingResult with distilled model
        """
        # Load teacher model
        teacher = YOLO(f"{self.teacher_model_name}.pt")
        student = YOLO(f"{self.student_model_name}.pt")

        try:
            # Train student with teacher
            results = student.train(
                data=str(data_yaml),
                epochs=epochs,
                teacher=teacher.model,  # Official teacher parameter
                distillation_loss="cwd",  # Channel-wise distillation
                project="./runs/distill",
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
