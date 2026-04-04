"""
YOLO Trainer with HPO support.
Location: training-api/src/training/yolo_trainer.py

Contains: YOLOTrainer class
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import logging
import os

import torch
import ultralytics

from .training_utils import (
    TrainingCancelled,
    TrainingResult,
    setup_gpu_memory,
    cleanup_gpu_memory,
    validate_dataset_distribution,
)

try:
    from .mlflow_tracker import MLflowTracker
except ImportError:
    MLflowTracker = None  # type: ignore

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


class YOLOTrainer:
    """YOLO11 Trainer with HPO support."""

    def __init__(self, model: str = "yolo11m", output_dir: Path = None):
        self.model_name = model
        self.output_dir = Path(output_dir or "./runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_model_path(self) -> str:
        """Resolve model path, preferring cached model to avoid slow GitHub downloads."""
        model_base = self.model_name
        if model_base.endswith(".pt"):
            model_base = model_base[:-3]
        cache_path = Path(os.path.expanduser("~/.cache/ultralytics")) / f"{model_base}.pt"
        if cache_path.exists():
            return str(cache_path)
        return f"{model_base}.pt"

    def sanity_check(self, data_yaml: Path, config: SanityCheckConfig = None) -> TrainingResult:
        """Run sanity check to verify training feasibility."""
        config = config or DEFAULT_SANITY_CHECK_CONFIG
        model = ultralytics.YOLO(self._resolve_model_path())

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
            metrics={"mAP50": map50, "mAP50-95": map50_95},
        )

    def train(
        self,
        data_yaml: Path,
        epochs: int = None,
        config: TrainingConfig = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        metric_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
    ) -> TrainingResult:
        """Train YOLO model with given configuration."""
        config = config or DEFAULT_TRAINING_CONFIG
        epochs = epochs or config.epochs

        dist_result = validate_dataset_distribution(Path(data_yaml))
        if dist_result.status == "critical":
            logging.error(
                f"[DIST VALIDATION] {dist_result.message}\n"
                f"  Train: {dist_result.train_image_count} images, {dist_result.train_box_count} boxes, "
                f"median box area={dist_result.train_median_area:.4f}\n"
                f"  Val:   {dist_result.val_image_count} images, {dist_result.val_box_count} boxes, "
                f"median box area={dist_result.val_median_area:.4f}\n"
                f"  Recommended action: Use stratified train/val split or the original dataset split."
            )
            return TrainingResult(status="failed", error=f"Dataset distribution validation FAILED: {dist_result.message}")
        elif dist_result.status == "warning":
            logging.warning(
                f"[DIST VALIDATION] {dist_result.message}\n"
                f"  Train: {dist_result.train_image_count} images, median box area={dist_result.train_median_area:.4f}\n"
                f"  Val:   {dist_result.val_image_count} images, median box area={dist_result.val_median_area:.4f}"
            )
        else:
            logging.info(
                f"[DIST VALIDATION] {dist_result.message} "
                f"(train={dist_result.train_median_area:.4f}, val={dist_result.val_median_area:.4f})"
            )

        setup_gpu_memory()

        tracker = None
        mlflow_enabled = True
        try:
            tracker = MLflowTracker(experiment_name="yolo-training")
            tracker.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        except Exception as e:
            mlflow_enabled = False
            logging.warning(f"MLflow tracking disabled: {e}")

        if tracker and mlflow_enabled:
            try:
                tracker.log_params({
                    "model": self.model_name, "epochs": epochs, "data_yaml": str(data_yaml),
                    "batch_size": config.batch, "image_size": config.imgsz,
                    "lr0": config.lr0, "lrf": config.lrf,
                    "momentum": config.momentum, "weight_decay": config.weight_decay,
                })
            except Exception as e:
                logging.warning(f"Failed to log parameters to MLflow: {e}")

        logging.info(f"[MODEL] Loading model: {self.model_name} (supports YOLO11/YOLO26 series)")
        if "yolo26" in self.model_name.lower():
            logging.info(f"[YOLO26] Using YOLO26 architecture with MuSGD optimizer compatibility")

        model = ultralytics.YOLO(self._resolve_model_path())

        if config.amp:
            logging.info("[AMP] Automatic Mixed Precision (AMP) training enabled")

        train_kwargs: Dict[str, Any] = {**config.to_dict()}

        if config.lr_scheduler.type == "cosine":
            train_kwargs["lrf"] = config.lr_scheduler.lrf
            logging.info(f"[LR_SCHEDULER] Cosine annealing: lr0={config.lr0}, min_lr={config.lr0 * train_kwargs['lrf']:.6f}")
        elif config.lr_scheduler.type == "exponential":
            train_kwargs["lrf"] = config.lr_scheduler.lrf
            logging.info(f"[LR_SCHEDULER] Exponential decay: lrf={train_kwargs['lrf']}")
        elif config.lr_scheduler.type == "linear":
            train_kwargs["lrf"] = config.lr_scheduler.lrf
            logging.info(f"[LR_SCHEDULER] Linear decay: lrf={train_kwargs['lrf']}")
        else:
            train_kwargs["lrf"] = 1.0
            logging.info(f"[LR_SCHEDULER] Constant LR: {config.lr0}")

        if config.resume_checkpoint:
            train_kwargs["resume"] = str(config.resume_checkpoint)
            logging.info(f"[RESUME] Resuming training from checkpoint: {config.resume_checkpoint}")

        _early_stopped = False
        _stopped_at_epoch = None

        def _on_fit_end(trainer):
            nonlocal _early_stopped, _stopped_at_epoch
            best_epoch = getattr(trainer, "best_epoch", None)
            total_epochs = getattr(trainer, "epochs", epochs)
            if best_epoch is not None and best_epoch < total_epochs - 1:
                _early_stopped = True
                _stopped_at_epoch = best_epoch
                logging.info(f"[EARLY STOPPING] Training stopped early at epoch {best_epoch} (total={total_epochs})")
                if tracker and mlflow_enabled:
                    try:
                        tracker.log_params({"early_stopped": True, "stopped_at_epoch": best_epoch})
                    except Exception:
                        pass

        model.add_callback("on_fit_end", _on_fit_end)

        if progress_callback or metric_callback:
            def _on_epoch_end(trainer):
                current_epoch = trainer.epoch
                total_epochs = trainer.epochs
                try:
                    if progress_callback:
                        progress_callback(current_epoch, total_epochs)
                    if metric_callback:
                        metrics_dict = {}
                        if hasattr(trainer, "metrics") and trainer.metrics:
                            m = trainer.metrics
                            metrics_dict = {
                                "mAP50": float(m.get("metrics/mAP50(B)", 0)),
                                "mAP50-95": float(m.get("metrics/mAP50-95(B)", 0)),
                                "box_loss": float(m.get("train/box_loss", 0)),
                                "cls_loss": float(m.get("train/cls_loss", 0)),
                                "dfl_loss": float(m.get("train/dfl_loss", 0)),
                                "val_box_loss": float(m.get("val/box_loss", 0)),
                                "val_cls_loss": float(m.get("val/cls_loss", 0)),
                                "val_dfl_loss": float(m.get("val/dfl_loss", 0)),
                            }
                        metric_callback(current_epoch, total_epochs, metrics_dict)
                except Exception as e:
                    if "cancel" in str(e).lower():
                        raise
            model.add_callback("on_train_epoch_end", _on_epoch_end)

        try:
            device_str = config.device
            if getattr(config, "num_gpus", 1) > 1:
                device_str = ",".join(str(i) for i in range(config.num_gpus))
                logging.info(f"[DDP] Multi-GPU training: {config.num_gpus} GPUs, device={device_str}")
            elif device_str == "cuda:0" and getattr(config, "num_gpus", 1) == 0:
                device_str = "cpu"

            results = None
            best_model_path_check = Path(self.output_dir) / "train" / "weights" / "best.pt"
            max_retries = 2
            retry_count = 0
            last_error = None

            while retry_count <= max_retries:
                try:
                    results = model.train(
                        data=str(data_yaml),
                        project=str(self.output_dir),
                        name="train",
                        exist_ok=True,
                        device=device_str,
                        workers=0,
                        **train_kwargs,
                    )
                    break
                except Exception as train_err:
                    last_error = str(train_err)
                    if hasattr(train_err, 'stderr') and train_err.stderr:
                        last_error += "\n" + str(train_err.stderr)
                    if hasattr(train_err, '__cause__') and train_err.__cause__:
                        last_error += "\n" + str(train_err.__cause__)
                    error_type = type(train_err).__name__

                    if best_model_path_check.exists():
                        logging.warning(
                            f"[RETRY] Training likely completed (best.pt exists) but "
                            f"return crashed: {error_type}: {last_error[:200]}. "
                            f"Returning saved checkpoint."
                        )
                        break

                    transient_keywords = [
                        "CUDA out of memory", "OutOfMemoryError", "Out of memory",
                        "NCCL", "timeout", "timeout expired", "ConnectionResetError",
                        "BrokenPipeError", "ProcessExitedException", "Address already in use",
                        "Address not available", "RuntimeError: CUDA error", "ChildFailedError",
                        "local_rank", "torch.OutOfMemoryError", "CUDA error", "NOLOAD",
                    ]
                    is_transient = any(kw in last_error for kw in transient_keywords)

                    if not is_transient or retry_count >= max_retries:
                        logging.error(f"[RETRY] Non-transient or max retries exceeded: {error_type}: {last_error[:200]}")
                        raise

                    retry_count += 1
                    if retry_count <= max_retries:
                        import time
                        wait_sec = retry_count * 30
                        logging.warning(f"[RETRY] Transient error #{retry_count}/{max_retries}: {error_type}: {last_error[:200]}. Waiting {wait_sec}s...")
                        time.sleep(wait_sec)
                        if "out of memory" in last_error.lower():
                            current_batch = train_kwargs.get("batch", config.batch)
                            new_batch = max(1, current_batch // 2)
                            train_kwargs["batch"] = new_batch
                            config.batch = new_batch
                            logging.info(f"[RETRY] Reducing batch from {current_batch} to {new_batch}")
                        train_kwargs.pop("resume", None)
                except KeyboardInterrupt:
                    raise TrainingCancelled("Training cancelled by user")

            if tracker and mlflow_enabled:
                try:
                    if results is not None and hasattr(results, 'results_dict') and results.results_dict:
                        tracker.log_metrics(results.results_dict)
                except Exception as e:
                    logging.warning(f"Failed to log metrics to MLflow: {e}")

            if tracker and mlflow_enabled:
                try:
                    if results is not None and hasattr(results, 'save_dir'):
                        model_path = Path(results.save_dir) / "weights" / "best.pt"
                        if not model_path.exists():
                            model_path = Path(self.output_dir) / "train" / "weights" / "best.pt"
                    else:
                        model_path = Path(self.output_dir) / "train" / "weights" / "best.pt"
                    if model_path.exists():
                        tracker.log_artifact(str(model_path))
                except Exception as e:
                    logging.warning(f"Failed to log model artifact to MLflow: {e}")

            if tracker and mlflow_enabled:
                try:
                    tracker.end_run(status="FINISHED")
                except Exception as e:
                    logging.warning(f"Failed to end MLflow run: {e}")

            train_output_dir = Path(self.output_dir) / "train"
            best_model_path = train_output_dir / "weights" / "best.pt"
            if not best_model_path.exists():
                best_model_path = train_output_dir / "weights" / "last.pt"

            map50 = 0.0
            map50_95 = 0.0
            if results is not None and hasattr(results, 'results_dict') and results.results_dict:
                rd = results.results_dict
                map50 = rd.get("metrics/mAP50(B)", 0) or rd.get("metrics/mAP50(B)", 0.0)
                map50_95 = rd.get("metrics/mAP50-95(B)", 0) or rd.get("metrics/mAP50-95(B)", 0.0)

            if map50 == 0.0 and best_model_path.exists():
                try:
                    import torch
                    ckpt = torch.load(best_model_path, map_location="cpu", weights_only=False)
                    train_metrics = ckpt.get("train_metrics", {})
                    if train_metrics and isinstance(train_metrics, dict) and "metrics/mAP50(B)" in train_metrics:
                        map50 = float(train_metrics.get("metrics/mAP50(B)", 0) or 0)
                        map50_95 = float(train_metrics.get("metrics/mAP50-95(B)", 0) or 0)
                    else:
                        results_dict = ckpt.get("train_results", {}) or ckpt.get("results_dict", {})
                        if results_dict and isinstance(results_dict, dict):
                            val50 = results_dict.get("metrics/mAP50(B)", [])
                            val95 = results_dict.get("metrics/mAP50-95(B)", [])
                            if isinstance(val50, list) and len(val50) > 0:
                                map50 = float(val50[-1])
                            elif isinstance(val50, (int, float)):
                                map50 = float(val50)
                            if isinstance(val95, list) and len(val95) > 0:
                                map50_95 = float(val95[-1])
                            elif isinstance(val95, (int, float)):
                                map50_95 = float(val95)
                    if map50 > 0:
                        logging.info(f"[METRICS] Read from checkpoint: mAP50={map50:.4f}, mAP50-95={map50_95:.4f}")
                except Exception as e:
                    logging.warning(f"[METRICS] Failed to read from checkpoint: {e}")

            return TrainingResult(
                status="completed",
                model_path=best_model_path,
                metrics={"mAP50": map50, "mAP50-95": map50_95},
                early_stopped=_early_stopped,
            )
        except TrainingCancelled:
            if tracker and mlflow_enabled:
                try:
                    tracker.end_run(status="FINISHED")
                except Exception:
                    pass
            return TrainingResult(status="cancelled", error="Training was cancelled")
        except Exception as e:
            if tracker and mlflow_enabled:
                try:
                    tracker.end_run(status="FAILED")
                except Exception:
                    pass
            return TrainingResult(status="failed", error=str(e))
        finally:
            del model
            model = None
            cleanup_gpu_memory()

    def tune(self, data_yaml: Path, config: HPOConfig = None) -> TrainingResult:
        """Run hyperparameter optimization with Ray Tune."""
        config = config or DEFAULT_HPO_CONFIG

        from ray import tune
        model = ultralytics.YOLO(self._resolve_model_path())

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
            metrics={"best_mAP50": best_result.metrics.get("metrics/mAP50(B)", 0)},
        )

    def export(self, model_path: Path, platform: str = "jetson", config: ExportConfig = None) -> Dict[str, Any]:
        """Export model to target format."""
        config = config or DEFAULT_EXPORT_CONFIG
        model = ultralytics.YOLO(str(model_path))

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

    def export_multi(
        self,
        model_path: Path,
        formats: List[str],
        platform: str = "jetson",
        imgsz: int = 640,
        calibration_image_dir: Optional[Path] = None,
        calibration_n: int = 1000,
    ) -> Dict[str, Dict[str, Any]]:
        """Export model to multiple formats in one call."""
        model = ultralytics.YOLO(str(model_path))
        results: Dict[str, Dict[str, Any]] = {}

        fp16_formats = {"onnx", "engine", "engine-fp16", "tensorrt"}
        int8_formats = {"engine-int8"}

        for fmt in formats:
            try:
                is_tflite = (fmt == "tflite")
                is_fp16 = fmt in fp16_formats and fmt not in int8_formats and not is_tflite
                is_int8 = fmt in int8_formats

                export_kwargs: Dict[str, Any] = {
                    "format": fmt,
                    "half": is_fp16 and not is_int8,
                    "imgsz": imgsz,
                    "simplify": True,
                    "project": str(self.output_dir / "export"),
                    "exist_ok": True,
                }

                if is_int8:
                    logging.info(f"[INT8] Starting INT8 calibration with up to {calibration_n} images")
                    export_kwargs["int8"] = True
                    export_kwargs["half"] = False
                    if calibration_image_dir and calibration_image_dir.exists():
                        export_kwargs["data"] = str(calibration_image_dir)
                        logging.info(f"[INT8] Using calibration data from: {calibration_image_dir}")
                    else:
                        logging.warning("[INT8] No calibration directory provided; INT8 export may use default calibration")

                export_path = model.export(**export_kwargs)

                size_mb = Path(export_path).stat().st_size / (1024 * 1024)
                results[fmt] = {
                    "path": export_path,
                    "size_mb": round(size_mb, 2),
                    "fp16": is_fp16,
                    "int8": is_int8,
                }
            except Exception as e:
                logging.warning(f"[export_multi] Failed to export format '{fmt}': {e}")
                results[fmt] = {"path": None, "size_mb": 0.0, "fp16": False, "int8": False, "error": str(e)}

        return results
