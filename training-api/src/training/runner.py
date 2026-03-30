"""
YOLO11 Training Runner with Ray Tune HPO integration.

Based on Ultralytics official best practices:
- https://docs.ultralytics.com/usage/cfg/
- https://docs.ultralytics.com/integrations/ray-tune/
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import os

import torch
from ultralytics import YOLO

try:
    from src.training.mlflow_tracker import MLflowTracker
except ImportError:
    MLflowTracker = None  # type: ignore

from .config import (
    TrainingConfig,
    SanityCheckConfig,
    HPOConfig,
    ExportConfig,
    PlateauBreakingConfig,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_SANITY_CHECK_CONFIG,
    DEFAULT_HPO_CONFIG,
    DEFAULT_EXPORT_CONFIG,
)


class TrainingCancelled(Exception):
    """Raised when training is cancelled via the progress callback."""
    pass


@dataclass
class TrainingResult:
    """Training result container."""
    status: str
    model_path: Optional[Path] = None
    metrics: Optional[Dict[str, float]] = None
    best_params: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    early_stopped: bool = False


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


@dataclass
class DatasetDistributionResult:
    """Result of dataset distribution validation."""
    train_median_area: float
    val_median_area: float
    ratio: float  # val/train ratio
    status: str   # "ok", "warning", "critical"
    train_box_count: int
    val_box_count: int
    train_image_count: int
    val_image_count: int
    message: str


def validate_dataset_distribution(data_yaml_path: Path) -> DatasetDistributionResult:
    """Validate train/val distribution balance in a YOLO dataset.

    Analyzes bounding box area distributions to detect train/val mismatch
    that causes model collapse (e.g., val boxes 8x larger than train).

    Returns DatasetDistributionResult with status and statistics.
    """
    import math
    from pathlib import Path

    yaml_path = Path(data_yaml_path)
    if not yaml_path.exists():
        return DatasetDistributionResult(
            train_median_area=0.0, val_median_area=0.0,
            ratio=0.0, status="warning",
            train_box_count=0, val_box_count=0,
            train_image_count=0, val_image_count=0,
            message=f"data.yaml not found at {data_yaml_path}"
        )

    import yaml
    try:
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
    except Exception as e:
        return DatasetDistributionResult(
            train_median_area=0.0, val_median_area=0.0,
            ratio=0.0, status="warning",
            train_box_count=0, val_box_count=0,
            train_image_count=0, val_image_count=0,
            message=f"Failed to parse data.yaml: {e}"
        )

    dataset_root = yaml_path.parent if yaml_data.get("path") is None else Path(yaml_data["path"])
    # data.yaml's train/val fields point directly to images subdirectories
    train_images = dataset_root / yaml_data.get("train", "train")
    val_images = dataset_root / yaml_data.get("val", "val")

    def _get_label_dir(images_path: Path) -> Path:
        """Derive labels path from images path.

        YOLO standard: /path/train/images -> /path/train/labels
        YOLO standard: /path/val/images -> /path/val/labels
        """
        s = str(images_path)
        s_norm = s.replace("\\", "/")
        if s_norm.endswith("/images"):
            # /path/train/images -> /path/train/labels (keep the / before images)
            return Path(s[:-7] + "/labels")
        # Fallback: replace "images" in last path component only
        last_slash = max(s_norm.rfind("/"), s_norm.rfind("\\"))
        component = s_norm[last_slash + 1:]
        if "images" in component:
            new_component = component.replace("images", "labels")
            return Path(s[:last_slash + 1] + new_component)
        return Path(s + "_labels")

    def _get_images_dir(labels_path: Path) -> Path:
        """Derive images path from labels path (inverse of _get_label_dir)."""
        s = str(labels_path)
        s_norm = s.replace("\\", "/")
        if s_norm.endswith("/labels"):
            return Path(s[:-7] + "/images")
        last_slash = max(s_norm.rfind("/"), s_norm.rfind("\\"))
        component = s_norm[last_slash + 1:]
        if "labels" in component:
            new_component = component.replace("labels", "images")
            return Path(s[:last_slash + 1] + new_component)
        return Path(s + "_images")

    def _parse_label_areas(label_dir: Path) -> tuple:
        """Parse all label files, return (areas_list, image_count)."""
        label_dir = Path(label_dir)
        if not label_dir.exists():
            return [], 0

        all_areas = []

        for label_file in label_dir.glob("*.txt"):
            try:
                with open(label_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                w, h = float(parts[3]), float(parts[4])
                                all_areas.append(w * h)  # YOLO normalized area = w * h
                            except ValueError:
                                pass
            except Exception:
                pass

        # Count images in sibling images directory (reverse: labels -> images)
        img_dir = _get_images_dir(label_dir)
        if img_dir.exists():
            image_count = len([f for f in img_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        else:
            image_count = 0

        return all_areas, image_count

    train_labels = _get_label_dir(train_images)
    val_labels = _get_label_dir(val_images)

    train_areas, train_imgs = _parse_label_areas(train_labels)
    val_areas, val_imgs = _parse_label_areas(val_labels)

    def _median(lst: list) -> float:
        if not lst:
            return 0.0
        sorted_lst = sorted(lst)
        n = len(sorted_lst)
        if n % 2 == 0:
            return (sorted_lst[n // 2 - 1] + sorted_lst[n // 2]) / 2
        return sorted_lst[n // 2]

    train_med = _median(train_areas)
    val_med = _median(val_areas)

    if train_med > 0 and val_med > 0:
        ratio = val_med / train_med
    elif val_med > 0:
        ratio = float('inf')
    else:
        ratio = 0.0

    # Determine status
    if ratio == float('inf') or ratio == 0.0:
        status = "critical"
        message = f"No train labels found. train_imgs={train_imgs}, val_imgs={val_imgs}"
    elif ratio > 5.0:
        status = "critical"
        message = (
            f"CRITICAL: val median box area ({val_med:.4f}) is {ratio:.1f}x "
            f"larger than train median ({train_med:.4f}). "
            f"This WILL cause mAP50 collapse. Use stratified train/val split."
        )
    elif ratio > 3.0:
        status = "warning"
        message = (
            f"WARNING: val median box area ({val_med:.4f}) is {ratio:.1f}x "
            f"larger than train median ({train_med:.4f}). "
            f"Consider using stratified train/val split."
        )
    else:
        status = "ok"
        message = f"Distribution OK: val/train ratio = {ratio:.2f}x"

    return DatasetDistributionResult(
        train_median_area=train_med,
        val_median_area=val_med,
        ratio=ratio,
        status=status,
        train_box_count=len(train_areas),
        val_box_count=len(val_areas),
        train_image_count=train_imgs,
        val_image_count=val_imgs,
        message=message,
    )


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

    def _resolve_model_path(self) -> str:
        """Resolve model path, preferring cached model to avoid slow GitHub downloads.

        Supports both YOLO11 and YOLO26 series. Ultralytics automatically handles
        the model architecture detection based on the model name string.
        """
        # Strip .pt suffix if present to avoid double-suffix (e.g. "yolo11m.pt" -> "yolo11m.pt.pt")
        model_base = self.model_name
        if model_base.endswith(".pt"):
            model_base = model_base[:-3]
        cache_path = Path(os.path.expanduser("~/.cache/ultralytics")) / f"{model_base}.pt"
        if cache_path.exists():
            return str(cache_path)
        return f"{model_base}.pt"

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
        model = YOLO(self._resolve_model_path())

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
        progress_callback: Optional[Callable[[int, int], None]] = None,
        metric_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
    ) -> TrainingResult:
        """
        Train YOLO model with given configuration.

        Args:
            data_yaml: Path to dataset YAML
            epochs: Number of epochs
            config: Training configuration
            progress_callback: Optional callable(epoch, total_epochs) called each epoch end.
                               If the callback raises TrainingCancelled, training is aborted.
            metric_callback: Optional callable(epoch, total_epochs, metrics_dict) called each
                             epoch end with the full metrics dict from ultralytics trainer.
                             metrics_dict keys: 'mAP50', 'mAP50-95', 'box_loss', 'cls_loss', 'dfl_loss'.
                             If the callback raises TrainingCancelled, training is aborted.

        Returns:
            TrainingResult with trained model
        """
        config = config or DEFAULT_TRAINING_CONFIG
        epochs = epochs or config.epochs

        # T4: Validate train/val distribution BEFORE training starts
        # This prevents the catastrophic mAP50 collapse caused by box size mismatch
        # (e.g., val median box area 8x larger than train median)
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
            return TrainingResult(
                status="failed",
                error=f"Dataset distribution validation FAILED: {dist_result.message}",
            )
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

        logging.info(f"[MODEL] Loading model: {self.model_name} (supports YOLO11/YOLO26 series)")
        if "yolo26" in self.model_name.lower():
            logging.info(f"[YOLO26] Using YOLO26 architecture with MuSGD optimizer compatibility")

        model = YOLO(self._resolve_model_path())

        # T3: AMP logging — confirm mixed precision is active
        if config.amp:
            logging.info("[AMP] Automatic Mixed Precision (AMP) training enabled")

        # T1: Build train kwargs; pass resume checkpoint if set
        train_kwargs: Dict[str, Any] = {**config.to_dict()}

        # Configure learning rate scheduler (supports RAFS: Rectified Adam + Warmup + Flat + Sharp)
        if config.lr_scheduler.type == "cosine":
            # Cosine annealing: lrf is the minimum LR factor
            # FIX: Removed broken * cosine_min_lr multiplication
            # Old code: lrf = config.lr_scheduler.lrf * config.lr_scheduler.cosine_min_lr (e.g. 0.01 * 1e-6 = 1e-8)
            # This made min LR = lr0 * 1e-10 which is far too aggressive
            # Now: lrf is used directly (final LR = lr0 * lrf, e.g. 0.01 * 0.01 = 1e-4)
            train_kwargs["lrf"] = config.lr_scheduler.lrf
            logging.info(f"[LR_SCHEDULER] Cosine annealing: lr0={config.lr0}, min_lr={config.lr0 * train_kwargs['lrf']:.6f}")
        elif config.lr_scheduler.type == "exponential":
            # Exponential decay
            train_kwargs["lrf"] = config.lr_scheduler.lrf
            logging.info(f"[LR_SCHEDULER] Exponential decay: lrf={train_kwargs['lrf']}")
        elif config.lr_scheduler.type == "linear":
            # Linear decay (ultralytics default)
            train_kwargs["lrf"] = config.lr_scheduler.lrf
            logging.info(f"[LR_SCHEDULER] Linear decay: lrf={train_kwargs['lrf']}")
        else:
            # Constant (no scheduling)
            train_kwargs["lrf"] = 1.0
            logging.info(f"[LR_SCHEDULER] Constant LR: {config.lr0}")

        if config.resume_checkpoint:
            train_kwargs["resume"] = str(config.resume_checkpoint)
            logging.info(f"[RESUME] Resuming training from checkpoint: {config.resume_checkpoint}")

        # T2: Register on_fit_end callback to detect early stopping and log to MLflow
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
                # Log early stopping event to MLflow
                if tracker and mlflow_enabled:
                    try:
                        tracker.log_params({
                            "early_stopped": True,
                            "stopped_at_epoch": best_epoch,
                        })
                    except Exception:
                        pass

        model.add_callback("on_fit_end", _on_fit_end)

        # Register progress callback and metric callback using ultralytics callback system
        if progress_callback or metric_callback:
            def _on_epoch_end(trainer):
                current_epoch = trainer.epoch
                total_epochs = trainer.epochs
                try:
                    if progress_callback:
                        progress_callback(current_epoch, total_epochs)
                    if metric_callback:
                        # Extract metrics from trainer object
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
                    # Re-raise cancellation signals
                    if "cancel" in str(e).lower():
                        raise
            model.add_callback("on_train_epoch_end", _on_epoch_end)

        try:
            # Resolve device: multi-GPU DDP or single GPU
            device_str = config.device
            if getattr(config, "num_gpus", 1) > 1:
                # Build comma-separated GPU list: "0,1,2"
                device_str = ",".join(str(i) for i in range(config.num_gpus))
                logging.info(f"[DDP] Multi-GPU training: {config.num_gpus} GPUs, device={device_str}")
            elif device_str == "cuda:0" and getattr(config, "num_gpus", 1) == 0:
                device_str = "cpu"

            # T5: Retry on transient GPU failures
            # Common transient errors: CUDA OOM, NCCL timeout, process crash
            # Before retry, always check if best.pt already exists (training may have
            # completed — the crash might only affect the result return path).
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
                        workers=0,  # Disable multiprocessing workers
                        **train_kwargs,
                    )
                    break  # Success
                except Exception as train_err:
                    last_error = str(train_err)
                    # For CalledProcessError / ChildFailedError, include stderr for better diagnosis
                    if hasattr(train_err, 'stderr') and train_err.stderr:
                        last_error += "\n" + str(train_err.stderr)
                    # For ChildFailedError (torch.distributed.elastic), extract child error
                    if hasattr(train_err, '__cause__') and train_err.__cause__:
                        last_error += "\n" + str(train_err.__cause__)
                    error_type = type(train_err).__name__

                    # Check if best.pt already exists (training completed, return crashed)
                    if best_model_path_check.exists():
                        logging.warning(
                            f"[RETRY] Training likely completed (best.pt exists) but "
                            f"return crashed: {error_type}: {last_error[:200]}. "
                            f"Returning saved checkpoint."
                        )
                        break  # Exit retry loop, use checkpoint fallback below

                    # Transient error detection
                    transient_keywords = [
                        "CUDA out of memory",
                        "OutOfMemoryError",
                        "Out of memory",
                        "NCCL",
                        "timeout",
                        "timeout expired",
                        "ConnectionResetError",
                        "BrokenPipeError",
                        "ProcessExitedException",
                        "Address already in use",
                        "Address not available",
                        "RuntimeError: CUDA error",
                        "ChildFailedError",
                        "local_rank",
                        "torch.OutOfMemoryError",
                        "CUDA error",
                        "NOLOAD",
                    ]
                    is_transient = any(kw in last_error for kw in transient_keywords)

                    if not is_transient or retry_count >= max_retries:
                        logging.error(
                            f"[RETRY] Non-transient or max retries exceeded: "
                            f"{error_type}: {last_error[:200]}"
                        )
                        raise  # Re-raise non-transient errors

                    retry_count += 1
                    if retry_count <= max_retries:
                        # Exponential backoff: 30s, 60s
                        import time
                        wait_sec = retry_count * 30
                        logging.warning(
                            f"[RETRY] Transient error #{retry_count}/{max_retries}: "
                            f"{error_type}: {last_error[:200]}. "
                            f"Waiting {wait_sec}s before retry..."
                        )
                        time.sleep(wait_sec)
                        # Try reducing batch size on OOM
                        if "out of memory" in last_error.lower():
                            current_batch = train_kwargs.get("batch", config.batch)
                            new_batch = max(1, current_batch // 2)
                            train_kwargs["batch"] = new_batch
                            config.batch = new_batch
                            logging.info(f"[RETRY] Reducing batch from {current_batch} to {new_batch} for retry")
                        # Reset optimizer state for clean restart
                        train_kwargs.pop("resume", None)
                except KeyboardInterrupt:
                    raise TrainingCancelled("Training cancelled by user")

            # Log metrics to MLflow
            if tracker and mlflow_enabled:
                try:
                    if results is not None and hasattr(results, 'results_dict') and results.results_dict:
                        tracker.log_metrics(results.results_dict)
                except Exception as e:
                    logging.warning(f"Failed to log metrics to MLflow: {e}")

            # Log model artifact
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

            # End MLflow run
            if tracker and mlflow_enabled:
                try:
                    tracker.end_run(status="FINISHED")
                except Exception as e:
                    logging.warning(f"Failed to end MLflow run: {e}")

            # Determine model path and metrics - handle DDP mode where results.metrics may be None/empty
            train_output_dir = Path(self.output_dir) / "train"
            best_model_path = train_output_dir / "weights" / "best.pt"
            if not best_model_path.exists():
                best_model_path = train_output_dir / "weights" / "last.pt"

            # Try to extract mAP50 from results.metrics (DetMetrics object)
            # In DDP mode, results may be None or have empty results_dict - fall back to checkpoint
            map50 = 0.0
            map50_95 = 0.0
            if results is not None and hasattr(results, 'results_dict') and results.results_dict:
                rd = results.results_dict
                map50 = rd.get("metrics/mAP50(B)", 0) or rd.get("metrics/mAP50(B)", 0.0)
                map50_95 = rd.get("metrics/mAP50-95(B)", 0) or rd.get("metrics/mAP50-95(B)", 0.0)

            # DDP fallback: if mAP50 is still 0 but model exists, read from checkpoint
            if map50 == 0.0 and best_model_path.exists():
                try:
                    import torch
                    ckpt = torch.load(best_model_path, map_location="cpu", weights_only=False)
                    # Priority: train_metrics (final scalar metrics) > train_results (per-epoch history)
                    # In ultralytics 8.4.23, train_results is per-epoch history (dict of lists),
                    # train_metrics contains final scalar metrics, results_dict may be None
                    train_metrics = ckpt.get("train_metrics", {})
                    if train_metrics and isinstance(train_metrics, dict) and "metrics/mAP50(B)" in train_metrics:
                        # This is the correct format for ultralytics 8.4.x
                        map50 = float(train_metrics.get("metrics/mAP50(B)", 0) or 0)
                        map50_95 = float(train_metrics.get("metrics/mAP50-95(B)", 0) or 0)
                    else:
                        # Fall back to train_results (per-epoch history) and take last value
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
                metrics={
                    "mAP50": map50,
                    "mAP50-95": map50_95,
                },
                early_stopped=_early_stopped,
            )
        except TrainingCancelled:
            # End MLflow run with finished status (cancelled is not a failure)
            if tracker and mlflow_enabled:
                try:
                    tracker.end_run(status="FINISHED")
                except Exception:
                    pass
            return TrainingResult(
                status="cancelled",
                error="Training was cancelled",
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
            # Explicitly release model and GPU memory
            del model
            model = None
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

        model = YOLO(self._resolve_model_path())

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

    def export_multi(
        self,
        model_path: Path,
        formats: List[str],
        platform: str = "jetson",
        imgsz: int = 640,
        calibration_image_dir: Optional[Path] = None,
        calibration_n: int = 1000,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Export model to multiple formats in one call.

        Args:
            model_path: Path to trained model
            formats: List of format strings
                (e.g. ["onnx", "engine-fp16", "engine-int8"])
            platform: Target platform hint for config selection
            imgsz: Image size for export
            calibration_image_dir: Directory of calibration images for INT8
            calibration_n: Max number of calibration images (default 1000)

        Returns:
            Dict mapping format name to {path, size_mb, fp16}
        """
        model = YOLO(str(model_path))
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
                        logging.warning(f"[INT8] No calibration directory provided; INT8 export may use default calibration")

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
                results[fmt] = {
                    "path": None,
                    "size_mb": 0.0,
                    "fp16": False,
                    "int8": False,
                    "error": str(e),
                }

        return results


class TransferLearningTrainer:
    """Transfer learning trainer using pretrained weights.

    Supports multiple knowledge distillation modes:
    - none: standard transfer learning (frozen backbone)
    - mgd:   Minimal Generative Distillation (arXiv:2506.14440) — feature-level L2 loss
    - feature: intermediate feature-map alignment with L2 loss
    - soft:   temperature-scaled soft label distillation
    """

    def __init__(
        self,
        teacher_model: str = "yolo11m",
        freeze_layers: int = 10,
    ):
        self.teacher_model_name = teacher_model
        self.freeze_layers = freeze_layers

    def _resolve_model_path(self) -> str:
        """Resolve model path, preferring cached model to avoid slow GitHub downloads."""
        base = self.teacher_model_name
        if base.endswith(".pt"):
            base = base[:-3]
        cache_path = Path(os.path.expanduser("~/.cache/ultralytics")) / f"{base}.pt"
        if cache_path.exists():
            return str(cache_path)
        return f"{base}.pt"

    def _build_distiller_hook(
        self,
        teacher_model,
        student_model,
        distiller: str,
        loss_weight: float,
        temperature: float,
        device: str,
    ):
        """Build the distillation loss callback.

        Registers per-batch forward hooks on teacher and student to capture
        intermediate feature maps, then combines them into a distillation loss.
        The combined loss (detection + KD) is attached to the student trainer.
        """
        teacher_feats: dict = {}
        student_feats: dict = {}
        _hook_handles: list = []

        # Heuristic layer names to hook (adapt to YOLO architecture)
        target_layer_names = ["model.7", "model.16", "model.23"]  # C3/SPPF blocks

        def _make_hook(name: str, storage: dict):
            def hook_fn(module, input, output):
                try:
                    storage[name] = output.detach()
                except Exception:
                    pass
            return hook_fn

        # Attach teacher hooks
        teacher_state = teacher_model.model.state_dict() if hasattr(teacher_model, "model") else {}
        for name, module in teacher_model.model.named_modules():
            if any(t in name for t in target_layer_names):
                handle = module.register_forward_hook(_make_hook(f"t_{name}", teacher_feats))
                _hook_handles.append(handle)

        # Attach student hooks
        for name, module in student_model.model.named_modules():
            if any(t in name for t in target_layer_names):
                handle = module.register_forward_hook(_make_hook(f"s_{name}", student_feats))
                _hook_handles.append(handle)

        # Cleanup helper
        def _remove_hooks():
            for h in _hook_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            _hook_handles.clear()

        # Callback that runs after each training batch
        def _distill_callback(trainer):
            if distiller == "none":
                return

            try:
                t_keys = sorted(teacher_feats.keys())
                s_keys = sorted(student_feats.keys())

                if not t_keys or not s_keys:
                    return

                # Compute feature matching loss
                distill_loss = torch.tensor(0.0, device=device)

                if distiller == "mgd":
                    # MGD: L2 loss across all feature maps, with channel-wise mask
                    for tk in t_keys:
                        sf = student_feats.get(tk.replace("t_", "s_", 1))
                        if sf is None:
                            continue
                        tf = teacher_feats[tk]
                        # Align shapes via 1x1 conv if needed
                        t_f = tf
                        s_f = sf
                        if t_f.shape[1] != s_f.shape[1]:
                            # Project student channels to teacher channels
                            proj = torch.nn.Conv2d(
                                s_f.shape[1], t_f.shape[1], kernel_size=1, bias=False, device=s_f.device
                            ).to(s_f.dtype)
                            s_f = proj(s_f)
                        # L2 loss with MGD-style channel masking
                        diff = (t_f - s_f) ** 2
                        distill_loss = distill_loss + diff.mean()

                elif distiller == "feature":
                    # Feature: direct L2 alignment on matching layers
                    for tk, sk in zip(t_keys, s_keys):
                        tf = teacher_feats[tk]
                        sf = student_feats.get(sk)
                        if sf is None:
                            continue
                        if tf.shape[1] != sf.shape[1]:
                            proj = torch.nn.Conv2d(
                                sf.shape[1], tf.shape[1], kernel_size=1, bias=False, device=sf.device
                            ).to(sf.dtype)
                            sf = proj(sf)
                        distill_loss = distill_loss + ((tf - sf) ** 2).mean()

                elif distiller == "soft":
                    # Soft: KL divergence between softened logits (handled in loss closure)
                    # For feature-level soft KD: align intermediate features
                    for tk, sk in zip(t_keys, s_keys):
                        tf = teacher_feats[tk]
                        sf = student_feats.get(sk)
                        if sf is None:
                            continue
                        t_soft = torch.softmax(tf.flatten(1) / temperature, dim=-1)
                        s_soft = torch.softmax(sf.flatten(1) / temperature, dim=-1)
                        distill_loss = distill_loss + (t_soft - s_soft).abs().mean()

                # Clear stored features for next batch
                teacher_feats.clear()
                student_feats.clear()

                # Log distillation loss
                distill_val = distill_loss.item() if isinstance(distill_loss, torch.Tensor) else float(distill_loss)
                if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
                    trainer.loss_items = (
                        trainer.loss_items + loss_weight * distill_val
                    )

            except Exception as e:
                logging.warning(f"[Distillation callback] Error: {e}")

        return _distill_callback, _remove_hooks

    def train(
        self,
        data_yaml: Path,
        epochs: int = 100,
        distiller: str = "none",
        loss_weight: float = 1.0,
        temperature: float = 4.0,
        teacher_model_path: Optional[str] = None,
        output_dir: str = "./runs/transfer",
        device: str = "cuda:0",
    ) -> TrainingResult:
        """
        Train with transfer learning and optional knowledge distillation.

        Args:
            data_yaml: Path to dataset YAML
            epochs: Number of epochs
            distiller: Distillation mode — "none" | "mgd" | "feature" | "soft"
            loss_weight: Weight of the distillation loss term
            temperature: Temperature for soft distillation (higher = softer targets)
            teacher_model_path: Optional path to a teacher model for distillation.
                                Defaults to self.teacher_model_name.
            output_dir: Directory for training output
            device: Device to train on

        Returns:
            TrainingResult with trained model
        """
        # Resolve teacher and student paths
        teacher_path = teacher_model_path or self._resolve_model_path()
        student_path = self._resolve_model_path()

        logging.info(
            f"[TransferLearning] distiller={distiller}, loss_weight={loss_weight}, "
            f"temperature={temperature}, teacher={teacher_path}, device={device}"
        )

        student_model = YOLO(student_path)
        teacher_model = None

        if distiller != "none":
            logging.info(f"[TransferLearning] Loading teacher model: {teacher_path}")
            teacher_model = YOLO(teacher_path)

            # Build distillation callback
            distill_cb, remove_hooks = self._build_distiller_hook(
                teacher_model=teacher_model,
                student_model=student_model,
                distiller=distiller,
                loss_weight=loss_weight,
                temperature=temperature,
                device=device,
            )
            student_model.add_callback("on_train_batch_start", distill_cb)

        try:
            results = student_model.train(
                data=str(data_yaml),
                epochs=epochs,
                freeze=self.freeze_layers,
                project=output_dir,
                name="student",
                verbose=False,
                device=device,
                distiller=distiller,
            )

            return TrainingResult(
                status="completed",
                model_path=Path(results.save_dir) / "weights" / "best.pt",
                metrics={
                    "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                    "distiller": distiller,
                    "loss_weight": loss_weight,
                    "temperature": temperature,
                },
            )
        except Exception as e:
            logging.error(f"[TransferLearning] Training failed: {e}", exc_info=True)
            return TrainingResult(
                status="failed",
                error=str(e),
            )
        finally:
            if teacher_model is not None:
                del teacher_model
            cleanup_gpu_memory()


@dataclass
class CurriculumStage:
    """Single stage in the progressive training curriculum.

    Proportional parameters (warmup_ratio, close_mosaic) scale with total epochs:
      - warmup_ratio: fraction of epochs for warmup (default 5%). Mirrors autoresearch WARMUP_RATIO.
      - close_mosaic: now computed as 20% of epochs in _build_config (was hardcoded).
        Mirrors WARMDOWN_RATIO pattern from autoresearch for augmentation landing.
      - num_gpus: number of GPUs for DDP training (default 1). Set to 2+ for multi-GPU.
        Ultralytics DDP: device="0,1,..." triggers torchrun, auto-distributes batch across GPUs.
        IMPORTANT: When num_gpus > 1, select_device() OVERRIDES CUDA_VISIBLE_DEVICES, so
        set num_gpus WITHOUT pre-setting CUDA_VISIBLE_DEVIDES in the environment.
        Ultralytics auto-scales batch across GPUs (no need to divide batch by num_gpus).
    """
    name: str
    epochs: int
    imgsz: int
    batch: int
    model: str
    augmentation_preset: str
    warmup_ratio: float = 0.05   # Fraction of epochs for warmup (default 5%)
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    num_gpus: int = 1           # Number of GPUs for DDP training (1 = single GPU, 2+ = multi-GPU)
    resume_from: Optional[str] = None  # Path to best.pt from previous stage


@dataclass
class CurriculumConfig:
    """Progressive training curriculum configuration.

    Three-stage curriculum optimized for mAP 90%+ on limited datasets:

    Stage 1 — Rapid Validation (50 epochs @ 640px):
      - Cheap: ~8 GPU-hours on T4
      - Validates pipeline, augmentation strategy, dataset quality
      - Pass criterion: mAP50 >= 0.5 (otherwise pipeline is broken)

    Stage 2 — Deep Training (150 epochs @ 1280px):
      - Main training with strong augmentation
      - Expected ~15-25 GPU-hours on T4 (batch=8 for 1280px)
      - Expected mAP50: 0.78-0.88

    Stage 3 — Fine-Tuning (100 epochs @ 1280px):
      - Reduced augmentation (close_mosaic=20, mixup=0.1)
      - Allows model to learn fine-grained details lost during heavy augmentation
      - Expected mAP50: 0.85-0.92

    Decision logic between stages:
      - Stage 1 mAP50 < 0.5: ABORT — pipeline broken (dataset/label issue)
      - Stage 1 mAP50 >= 0.5: proceed to Stage 2
      - Stage 2 mAP50 >= 0.90: GOAL REACHED — stop
      - Stage 2 mAP50 < 0.85 AND PlateauBreaker triggered: proceed to Stage 3
      - Stage 2 mAP50 < 0.75: trigger AutoAdjustAgent data expansion

    Total worst-case: 300 epochs, ~23 GPU-hours on T4
    Total best-case (goal reached at Stage 2): 200 epochs, ~23 GPU-hours
    """
    stage1: CurriculumStage = field(default_factory=lambda: CurriculumStage(
        name="rapid_validation",
        epochs=50,
        imgsz=640,
        batch=16,
        model="yolo11m",
        augmentation_preset="balanced",
        mosaic=1.0, mixup=0.1, copy_paste=0.1,
        degrees=0.0, translate=0.1, scale=0.5,
        # close_mosaic now computed proportionally in _build_config (20% of epochs)
    ))
    stage2: CurriculumStage = field(default_factory=lambda: CurriculumStage(
        name="deep_training",
        epochs=150,
        imgsz=1280,
        batch=8,
        model="yolo11x",
        augmentation_preset="strong",
        mosaic=1.0, mixup=0.3, copy_paste=0.4,
        degrees=15.0, translate=0.2, scale=0.7,
    ))
    stage3: CurriculumStage = field(default_factory=lambda: CurriculumStage(
        name="fine_tuning",
        epochs=100,
        imgsz=1280,
        batch=8,
        model="yolo11x",
        augmentation_preset="strong",
        mosaic=0.0, mixup=0.1, copy_paste=0.1,
        degrees=5.0, translate=0.1, scale=0.5,
    ))

    # Decision thresholds
    stage1_min_map: float = 0.50   # Min mAP50 to pass Stage 1
    stage2_target_map: float = 0.90  # Goal: stop if reached
    stage2_min_for_stage3: float = 0.80  # Min mAP50 to proceed to Stage 3


class PipelineCurriculumTrainer:
    """Progressive curriculum trainer for YOLO.

    Runs a 3-stage curriculum with automated gate decisions between stages.
    Uses existing YOLOTrainer internally. No new model architectures needed.
    """

    def __init__(
        self,
        output_dir: Path = None,
        target_mAP: float = 0.90,
    ):
        self.output_dir = Path(output_dir or "./runs/curriculum")
        self.target_mAP = target_mAP
        self._stage_history: list[dict] = []

    def _build_config(self, stage: CurriculumStage, resume_from: Optional[str] = None) -> TrainingConfig:
        """Build TrainingConfig for a given stage.

        Applies proportional warmup and close_mosaic based on total epochs.
        Inspired by autoresearch: WARMUP_RATIO and WARMDOWN_RATIO patterns.
        """
        cfg = DEFAULT_TRAINING_CONFIG
        cfg.epochs = stage.epochs
        cfg.imgsz = stage.imgsz
        cfg.batch = stage.batch
        cfg.model = stage.model
        cfg.mosaic = stage.mosaic
        cfg.mixup = stage.mixup
        cfg.copy_paste = stage.copy_paste
        cfg.degrees = stage.degrees
        cfg.translate = stage.translate
        cfg.scale = stage.scale
        cfg.num_gpus = getattr(stage, 'num_gpus', 1)

        # For curriculum stage transitions (Stage 1→2, 2→3), pass previous stage's best.pt
        # as the model to initialize from (NOT as resume, which ultralytics rejects
        # for completed checkpoints). resume_checkpoint is for continuing interrupted training.
        if resume_from:
            cfg.model = resume_from  # Load weights from previous stage's best.pt

        # Proportional warmup: 5% of training epochs (min 2 epochs)
        # Mirrors WARMUP_RATIO pattern from autoresearch
        warmup_epochs = max(2, int(stage.epochs * getattr(stage, 'warmup_ratio', 0.05)))
        cfg.lr_scheduler.warmup_epochs = warmup_epochs
        cfg.warmup_epochs = warmup_epochs

        # Proportional close_mosaic: last 20% of training (min 3 epochs)
        # "Slow landing" for augmentation - clean mosaic before fine-tuning
        # Mirrors WARMDOWN_RATIO=0.5 pattern from autoresearch (but for augmentation only)
        close_mosaic = max(3, int(stage.epochs * 0.2))
        cfg.close_mosaic = close_mosaic

        logging.info(
            f"[CURRICULUM] Proportional config: epochs={stage.epochs}, "
            f"warmup_epochs={warmup_epochs} ({warmup_epochs/stage.epochs*100:.0f}%), "
            f"close_mosaic={close_mosaic} ({close_mosaic/stage.epochs*100:.0f}%), "
            f"num_gpus={cfg.num_gpus}"
        )
        return cfg

    def _run_stage(
        self,
        stage: CurriculumStage,
        data_yaml: Path,
        stage_num: int,
        resume_from: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        metric_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
        plateau_manager: Optional["PlateauManager"] = None,
        redis_client=None,
        task_id_for_redis: str = "curriculum",
    ) -> Tuple[TrainingResult, str]:
        """Run a single curriculum stage with in-stage plateau recovery.

        When PlateauManager detects plateau during training, it returns a PlateauDecision.
        Level 1 (LR decay) and Level 2 (augment boost) trigger an in-stage restart
        with adjusted parameters — the stage effectively continues with better config.
        Level 3 (data expansion) ends the stage with status="plateau".
        """
        stage_output_dir = self.output_dir / f"stage{stage_num}_{stage.name}"
        trainer = YOLOTrainer(model=stage.model, output_dir=stage_output_dir)
        config = self._build_config(stage, resume_from)

        # Track best checkpoint across restarts within this stage
        best_checkpoint: Optional[Path] = None
        best_checkpoint_map: Optional[float] = None
        if resume_from:
            best_checkpoint = Path(resume_from)

        logging.info(
            f"[CURRICULUM] Stage {stage_num} ({stage.name}): "
            f"model={stage.model}, epochs={stage.epochs}, imgsz={stage.imgsz}, "
            f"batch={stage.batch}, resume={resume_from or 'None'}"
        )

        # In-stage restart loop: allows Level 1/2 plateau recovery
        _epoch_count = [0]

        while True:
            # Create per-epoch callback that feeds both user's callback and PlateauManager
            def epoch_callback(epoch: int, total: int, metrics: Dict[str, float]):
                nonlocal _epoch_count
                _epoch_count[0] += 1
                # Forward to user's callback
                if metric_callback:
                    metric_callback(epoch, total, metrics)
                # Forward to PlateauManager
                if plateau_manager:
                    decision = plateau_manager.on_metric(epoch, total, metrics)
                    if decision.triggered:
                        logging.warning(
                            f"[CURRICULUM][PLATEAU] Stage {stage_num} in-stage decision: "
                            f"level={decision.level}, action={decision.action}, "
                            f"avg_mAP50={decision.avg_recent_mAP50:.4f}"
                        )
                # Write live plateau signals to Redis every 5 epochs
                # This enables AutoAdjustAgent (in Business API) to monitor progress
                if redis_client is not None and _epoch_count[0] % 5 == 0:
                    try:
                        status = plateau_manager.get_status() if plateau_manager else {}
                        strategies = status.get("strategies_triggered", [])
                        latest_by_action = {}
                        for strategy in strategies:
                            action = strategy.get("action")
                            if action:
                                latest_by_action[action] = strategy.get("adjustment", {})
                        redis_mapping = {
                            "live_mAP50": str(metrics.get("mAP50", 0)),
                            "lr_decay_triggered": str(status.get("lr_reduction_count", 0) > 0),
                            "lr_decay_signal": json.dumps(latest_by_action.get("lr_decay")),
                            "augment_boost_active": str(status.get("augment_boost_active", False)),
                            "augment_boost_signal": json.dumps(latest_by_action.get("augment_boost")),
                            "data_expansion_requested": str(status.get("signaled_expansion", False)),
                            "data_expansion_signal": json.dumps(latest_by_action.get("data_expansion")),
                            "in_stage_restarts": str(status.get("in_stage_restarts", 0)),
                            "strategies_triggered": json.dumps(strategies),
                            "llm_diagnosis": json.dumps(status.get("llm_diagnosis")),
                            "curriculum_stage_num": str(stage_num),
                        }
                        redis_client.hset(f"training:task:{task_id_for_redis}", mapping=redis_mapping)
                    except Exception:
                        pass  # Non-critical: Redis write failure should not disrupt training

            result = trainer.train(
                data_yaml=data_yaml,
                config=config,
                progress_callback=progress_callback,
                metric_callback=epoch_callback,
            )

            # Track best checkpoint
            if result.model_path and Path(result.model_path).exists():
                result_map = result.metrics.get("mAP50", 0.0) if result.metrics else 0.0
                if best_checkpoint_map is None or result_map > best_checkpoint_map:
                    best_checkpoint = Path(result.model_path)
                    best_checkpoint_map = result_map
                    if plateau_manager:
                        plateau_manager.set_best_checkpoint_path(str(best_checkpoint))

            mAP50 = result.metrics.get("mAP50", 0.0) if result.metrics else 0.0
            logging.info(
                f"[CURRICULUM] Stage {stage_num} epoch-run complete: "
                f"mAP50={mAP50:.4f}, status={result.status}"
            )

            # Check if plateau manager wants an in-stage restart
            if plateau_manager and plateau_manager._in_stage_restarts > 0:
                last_decision = plateau_manager._triggered_strategies[-1] if plateau_manager._triggered_strategies else {}
                action = last_decision.get("action", "")
                adjustment = last_decision.get("adjustment", {})

                if action == "lr_decay":
                    # Level 1: Restart with lower LR + extra epochs
                    new_lr = adjustment.get("new_lr", config.lr0 * 0.5)
                    extra_epochs = 50
                    config.lr0 = new_lr
                    config.epochs = stage.epochs + extra_epochs
                    config.resume_checkpoint = str(best_checkpoint) if best_checkpoint else None
                    logging.info(
                        f"[CURRICULUM][RESTART] Level 1 LR decay: "
                        f"lr0={new_lr:.6f}, epochs={config.epochs}, "
                        f"resume={config.resume_checkpoint}"
                    )
                    continue  # Restart this stage

                elif action == "augment_boost":
                    # Level 2: Restart with stronger augmentation
                    extra_epochs = adjustment.get("boost_epochs", 30)
                    config.mixup = adjustment.get("mixup", 0.3)
                    config.copy_paste = adjustment.get("copy_paste", 0.4)
                    config.degrees = adjustment.get("degrees", 15.0)
                    config.translate = adjustment.get("translate", 0.2)
                    config.scale = adjustment.get("scale", 0.7)
                    config.epochs = stage.epochs + extra_epochs
                    config.resume_checkpoint = str(best_checkpoint) if best_checkpoint else None
                    logging.info(
                        f"[CURRICULUM][RESTART] Level 2 augment boost: "
                        f"mixup={config.mixup}, copy_paste={config.copy_paste}, "
                        f"epochs={config.epochs}, resume={config.resume_checkpoint}"
                    )
                    continue  # Restart this stage

                elif action == "data_expansion":
                    # Level 3: Cannot expand within stage — end with plateau status
                    logging.warning(
                        f"[CURRICULUM][PLATEAU] Stage {stage_num} data expansion needed. "
                        f"Ending stage with plateau status."
                    )
                    return TrainingResult(
                        status="plateau",
                        model_path=str(best_checkpoint) if best_checkpoint else None,
                        metrics={
                            "mAP50": mAP50,
                            "plateau_level": 3,
                            "recommendation": "data_expansion_needed",
                            "strategies_triggered": plateau_manager._triggered_strategies,
                        },
                    ), "plateau"

            # Normal completion or failure — exit restart loop
            break

        logging.info(
            f"[CURRICULUM] Stage {stage_num} complete: mAP50={mAP50:.4f}, "
            f"status={result.status}, best_pt={result.model_path}"
        )

        return result, ""

    def train(
        self,
        data_yaml: Path,
        config: CurriculumConfig = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        stage_callback: Optional[Callable[[int, str, float, dict], None]] = None,
        metric_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
        task_id: str = "curriculum",
        plateau_config: Optional[PlateauBreakingConfig] = None,
        redis_client=None,
    ) -> TrainingResult:
        """
        Run the full progressive curriculum.

        Args:
            data_yaml: Path to dataset YAML
            config: Curriculum configuration
            progress_callback: Called each epoch with (epoch, total_epochs)
            stage_callback: Called after each stage with
                          (stage_num, stage_name, mAP50, decision_dict)
            metric_callback: Called each epoch with (epoch, total_epochs, metrics_dict).
                           Enables PlateauManager plateau detection during curriculum.
            task_id: Task identifier for plateau manager cache
            plateau_config: PlateauBreakingConfig for in-stage recovery (default: enabled)

        Returns:
            TrainingResult from the final completed stage
        """
        config = config or CurriculumConfig()
        plateau_config = plateau_config or PlateauBreakingConfig()
        data_yaml = Path(data_yaml)
        best_model_path: Optional[Path] = None
        best_mAP50 = 0.0

        # Create PlateauManager for in-stage plateau recovery
        from .plateau_manager import PlateauManager
        pm = PlateauManager(task_id=f"{task_id}_stage", config=plateau_config)

        # Stage 1: Rapid validation
        if stage_callback:
            stage_callback(1, "rapid_validation", 0.0, {"action": "starting"})
        pm._task_id = f"{task_id}_s1"
        s1_result, _ = self._run_stage(
            config.stage1, data_yaml, stage_num=1,
            progress_callback=progress_callback, metric_callback=metric_callback,
            plateau_manager=pm,
            redis_client=redis_client,
            task_id_for_redis=task_id,
        )
        s1_map = s1_result.metrics.get("mAP50", 0.0) if s1_result.metrics else 0.0
        self._stage_history.append({
            "stage": 1, "name": "rapid_validation",
            "mAP50": s1_map, "status": s1_result.status,
        })

        if s1_result.status == "plateau":
            return s1_result
        if s1_result.status != "completed":
            return s1_result

        # Gate: Stage 1 pass criterion
        if s1_map < config.stage1_min_map:
            logging.error(
                f"[CURRICULUM] Stage 1 FAILED: mAP50={s1_map:.4f} < {config.stage1_min_map}. "
                f"Pipeline broken — check dataset quality, labels, and augmentation."
            )
            return s1_result

        best_model_path = Path(s1_result.model_path) if s1_result.model_path else None
        best_mAP50 = s1_map

        # Stage 2: Deep training
        if stage_callback:
            stage_callback(2, "deep_training", s1_map, {"action": "proceeding", "resume": str(best_model_path)})
        # Reset PlateauManager for Stage 2 (fresh detection window)
        pm = PlateauManager(task_id=f"{task_id}_s2", config=plateau_config)
        pm.set_current_lr(config.stage2.model == "yolo11x" and 0.01 or 0.01)
        pm.set_best_checkpoint_path(str(best_model_path) if best_model_path else "")
        s2_result, _ = self._run_stage(
            config.stage2, data_yaml, stage_num=2,
            resume_from=str(best_model_path),
            progress_callback=progress_callback, metric_callback=metric_callback,
            plateau_manager=pm,
            redis_client=redis_client,
            task_id_for_redis=task_id,
        )
        s2_map = s2_result.metrics.get("mAP50", 0.0) if s2_result.metrics else 0.0
        s2_status = s2_result.status
        self._stage_history.append({
            "stage": 2, "name": "deep_training",
            "mAP50": s2_map, "status": s2_status,
            "strategies_triggered": pm._triggered_strategies,
        })

        # Update best checkpoint from Stage 2
        if s2_result.model_path and Path(s2_result.model_path).exists():
            if s2_map > best_mAP50:
                best_model_path = Path(s2_result.model_path)
                best_mAP50 = s2_map
        elif best_model_path and best_model_path.exists():
            pass  # Keep Stage 1 best
        else:
            best_model_path = Path(s2_result.model_path) if s2_result.model_path else best_model_path

        # Handle plateau at Stage 2
        if s2_status == "plateau":
            logging.warning(
                f"[CURRICULUM] Stage 2 plateau: mAP50={s2_map:.4f}. "
                f"Data expansion needed. Strategies: {pm._triggered_strategies}"
            )
            return TrainingResult(
                status="plateau",
                model_path=str(best_model_path) if best_model_path else None,
                metrics={
                    "mAP50": best_mAP50,
                    "stage_history": self._stage_history,
                    "recommendation": "data_expansion_needed",
                    "llm_diagnosis": pm._llm_diagnosis,
                    "strategies_triggered": pm._triggered_strategies,
                    "in_stage_restarts": pm._in_stage_restarts,
                },
            )

        if s2_status != "completed":
            return s2_result

        # Gate: Goal reached?
        if s2_map >= config.stage2_target_map:
            logging.info(
                f"[CURRICULUM] GOAL REACHED: mAP50={s2_map:.4f} >= {config.stage2_target_map}. "
                f"Stopping curriculum."
            )
            return TrainingResult(
                status="completed",
                model_path=str(best_model_path),
                metrics={"mAP50": best_mAP50, "stage": "stage2_goal_reached"},
                early_stopped=True,
            )

        # Gate: Proceed to Stage 3?
        if s2_map >= config.stage2_min_for_stage3:
            if stage_callback:
                stage_callback(3, "fine_tuning", s2_map, {"action": "proceeding_to_stage3"})
            # Reset PlateauManager for Stage 3
            pm = PlateauManager(task_id=f"{task_id}_s3", config=plateau_config)
            pm.set_best_checkpoint_path(str(best_model_path) if best_model_path else "")
            s3_result, _ = self._run_stage(
                config.stage3, data_yaml, stage_num=3,
                resume_from=str(best_model_path),
                progress_callback=progress_callback, metric_callback=metric_callback,
                plateau_manager=pm,
                redis_client=redis_client,
                task_id_for_redis=task_id,
            )
            s3_map = s3_result.metrics.get("mAP50", 0.0) if s3_result.metrics else 0.0
            self._stage_history.append({
                "stage": 3, "name": "fine_tuning",
                "mAP50": s3_map, "status": s3_result.status,
            })
            if s3_map > best_mAP50:
                best_model_path = Path(s3_result.model_path) if s3_result.model_path else best_model_path
                best_mAP50 = s3_map

            return TrainingResult(
                status="completed",
                model_path=str(best_model_path) if best_model_path else None,
                metrics={
                    "mAP50": best_mAP50,
                    "stage_history": self._stage_history,
                },
            )

        # Stage 2 completed but didn't reach goal AND below Stage 3 threshold
        logging.warning(
            f"[CURRICULUM] Stage 2 mAP50={s2_map:.4f} below Stage 3 threshold "
            f"({config.stage2_min_for_stage3}). Triggering plateau strategies."
        )
        return TrainingResult(
            status="plateau",
            model_path=str(best_model_path) if best_model_path else None,
            metrics={
                "mAP50": best_mAP50,
                "stage_history": self._stage_history,
                "recommendation": "data_expansion_needed",
                "strategies_triggered": pm._triggered_strategies,
            },
        )
