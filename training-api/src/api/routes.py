"""
Training API Internal Routes
Location: training-api/src/api/routes.py

.. deprecated::
    This module is deprecated and should be split into smaller modules.
    New code should use the routes/ package structure.

Contains internal endpoints for:
- Training task management
- HPO task management
- Model export management

Migration plan:
- models/ - Request/Response models (DONE)
- store/ - Task storage (DONE)
- services/ - DynamicTrainingManager (TODO)
- routes/ - Route handlers (TODO)
"""

import os
import sys
import asyncio
import logging
import threading
import subprocess
from pathlib import Path

# Add training-api/src to sys.path so 'src' package resolves here (not legacy src/)
# This prevents legacy /home/wangxin/yolo-auto-training/src/ from shadowing training-api/src/
# CRITICAL: Must add 'src/' dir (not parent of 'src/') so that 'from src.training.runner'
# in _run_training_sync() resolves to training-api/src/training/runner.py NOT
# yolo-auto-training/src/training/runner.py (legacy package).
_training_api_src_root = Path(__file__).parent.parent  # = training-api/src/
if str(_training_api_src_root) not in sys.path:
    sys.path.insert(0, str(_training_api_src_root))

import json
import os
import uuid
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Header, status, BackgroundTasks, Depends, Request
from pydantic import BaseModel, Field

# Import verify_internal_api_key from gateway for timing-safe comparison
# Use relative import since gateway is in the same package
from .gateway import verify_internal_api_key, check_rate_limit, get_redis_client

# Module-level retry circuit breaker: task_id -> retry count
_retry_counts: Dict[str, int] = {}


# ==================== Request/Response Models ====================

class TrainStartRequest(BaseModel):
    """Internal training start request."""
    task_id: str = Field(..., description="Task identifier")
    model: str = Field("yolo11m", description="Model size")
    data_yaml: str = Field(..., description="Dataset YAML path")
    epochs: int = Field(100, description="Number of epochs")
    imgsz: int = Field(640, description="Image size")
    batch: int = Field(16, description="Batch size")
    output_dir: str = Field("/home/wangxin/runs", description="Output directory")
    device: str = Field("cuda:0", description="Device")
    auto_export: bool = Field(True, description="Automatically trigger ONNX export after training completes")
    augmentation_preset: Optional[str] = Field(None, description="Augmentation preset: fast, balanced, strong")
    # HPO-injected params (optional)
    lr0: Optional[float] = Field(None, description="Initial learning rate (from HPO)")
    lrf: Optional[float] = Field(None, description="Final learning rate factor (from HPO)")
    weight_decay: Optional[float] = Field(None, description="L2 regularization (from HPO)")
    momentum: Optional[float] = Field(None, description="SGD momentum (from HPO)")
    # Best.pt inheritance (optional) - resume training from a trained checkpoint
    resume_from: Optional[str] = Field(None, description="Path to best.pt for transfer learning")


class TrainStatusResponse(BaseModel):
    """Training status response."""
    task_id: str
    status: str  # submitted, running, completed, failed
    progress: float = 0.0
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    metrics: Optional[dict] = None
    error: Optional[str] = None
    early_stopped: Optional[bool] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    # Plateau detection fields (populated when training plateaus)
    live_mAP50: Optional[float] = None
    lr_decay_triggered: Optional[bool] = None
    lr_decay_signal: Optional[dict] = None
    augment_boost_active: Optional[bool] = None
    augment_boost_signal: Optional[dict] = None
    data_expansion_requested: Optional[bool] = None
    data_expansion_signal: Optional[dict] = None
    strategies_triggered: Optional[list] = None
    resubmit_count: Optional[int] = None
    last_resubmitted_at: Optional[str] = None
    resubmit_reason: Optional[str] = None
    # Curriculum training fields (populated during progressive 3-stage training)
    curriculum_stage: Optional[str] = None
    curriculum_stage_num: Optional[int] = None
    curriculum_stage_history: Optional[list] = None
    curriculum_stage_mAP: Optional[float] = None


class HPOStartRequest(BaseModel):
    """HPO start request."""
    task_id: str
    model: str = "yolo11m"
    data_yaml: str
    n_trials: int = 50
    epochs_per_trial: int = 50
    strategy: str = Field("asha", description="HPO strategy: 'asha' (Ray Tune ASHA) or 'bayesian' (GP+EI via scikit-optimize)")


class ExportStartRequest(BaseModel):
    """Export start request."""
    task_id: str
    model_path: str
    platform: str = "jetson_orin"
    imgsz: int = 640
    formats: List[str] = Field(default_factory=lambda: ["onnx"])
    int8_quantize: bool = Field(False, description="Enable INT8 quantization (requires calibration data)")


class BenchmarkRunRequest(BaseModel):
    """Benchmark run request."""
    task_id: str = Field(..., description="Task identifier")
    model_path: str = Field(..., description="Path to exported model file")
    format: str = Field("onnx", description="Model format for reporting")
    imgsz: int = Field(640, description="Image size for benchmark")
    warmup: int = Field(10, description="Number of warmup runs")
    runs: int = Field(100, description="Number of timed runs")


class ActiveLearnSelectRequest(BaseModel):
    """Active learning sample selection request."""
    model_path: str = Field(..., description="Path to current YOLO model")
    image_pool_dir: str = Field(..., description="Directory containing unlabeled images")
    top_k: int = Field(100, description="Number of samples to select")
    strategy: str = Field("entropy", description="Selection strategy: entropy / margin / density / random")


class SemiSupervisedRequest(BaseModel):
    """Semi-supervised learning request."""
    task_id: str
    labeled_data_yaml: str = Field(..., description="YAML for labeled training data")
    unlabeled_image_dir: str = Field(..., description="Directory with unlabeled images")
    method: str = Field("yolo_teacher", description="Pseudo-label method: yolo_teacher / sam / hybrid")
    confidence_threshold: float = Field(0.7, description="Min confidence for pseudo-labels")
    iterations: int = Field(1, description="Number of self-training iterations")
    epochs: int = Field(50, description="Training epochs per iteration")


# ==================== Create Router ====================

router = APIRouter()


# ==================== Task Storage ====================

# Redis-backed task storage with L1 in-memory cache.
# On reads: check local dict first, then Redis, populate cache on miss.
# On writes: write-through to both local dict and Redis.
# Key pattern in Redis: training:task:{task_id}

_redis_client = get_redis_client()
_tasks_cache: dict = {}
_tasks_lock = threading.Lock()


def _task_get(task_id: str) -> Optional[dict]:
    """Read a task. L1 dict cache, then Redis."""
    with _tasks_lock:
        if task_id in _tasks_cache:
            return _tasks_cache[task_id]
    if _redis_client is None:
        return None
    try:
        key = f"training:task:{task_id}"
        raw = _redis_client.get(key)
        if raw:
            task = json.loads(raw)
            with _tasks_lock:
                _tasks_cache[task_id] = task
            return task
    except Exception:
        pass
    return None


def _task_set(task_id: str, task: dict) -> None:
    """Write a task. Write-through to local cache and Redis."""
    with _tasks_lock:
        _tasks_cache[task_id] = task
    if _redis_client is None:
        return
    try:
        key = f"training:task:{task_id}"
        _redis_client.set(key, json.dumps(task))
    except Exception as e:
        # Log but don't fail the request
        print(f"[_task_set] Redis write failed for {task_id}: {e}")


def _task_del(task_id: str) -> None:
    """Delete a task from local cache and Redis."""
    with _tasks_lock:
        _tasks_cache.pop(task_id, None)
    if _redis_client is None:
        return
    try:
        _redis_client.delete(f"training:task:{task_id}")
    except Exception as e:
        print(f"[_task_del] Redis delete failed for {task_id}: {e}")


# Cancellation registry: task_id -> threading.Event
# Stored separately from task records so Event objects aren't JSON-serialised.
_cancel_events: dict[str, threading.Event] = {}
_cancel_lock = threading.Lock()


# ==================== Dynamic Training Manager ====================

class DynamicTrainingManager:
    """Monitors training metrics in real-time and triggers plateau-breaking strategies.

    Tracks mAP50 history with a sliding window. When improvement stalls:
      Level 1: Reduce learning rate by factor (up to 3 times)
      Level 2: Boost augmentation for a burst window, then restore
      Level 3: Log expansion signal to task cache for Business API to trigger ActiveLearning
    """

    def __init__(
        self,
        task_id: str,
        plateau_config: "PlateauBreakingConfig",
        device: str = "cuda:0",
    ):
        self.task_id = task_id
        self.cfg = plateau_config
        self.device = device
        self._map_history: list[tuple[int, float]] = []  # (epoch, mAP50)
        self._lr_reduction_count = 0
        self._augment_boost_active = False
        self._augment_boost_remaining = 0
        self._expansion_round = 0
        self._signaled_expansion = False
        self._original_augment: dict = {}
        self._triggered_strategies: list[dict] = []
        self._last_reported_epoch = -1

    def on_metric(self, epoch: int, total_epochs: int, metrics: dict[str, float]) -> None:
        """Called each epoch with current metrics. Returns dict of adjustments or None."""
        if not self.cfg.enabled:
            return
        if epoch <= self._last_reported_epoch:
            return
        self._last_reported_epoch = epoch

        mAP50 = metrics.get("mAP50", 0.0)
        self._map_history.append((epoch, mAP50))

        # Keep history bounded
        if len(self._map_history) > self.cfg.window * 3:
            self._map_history = self._map_history[-self.cfg.window * 2:]

        # Update cache with live metrics
        with _tasks_lock:
            if self.task_id in _tasks_cache:
                _tasks_cache[self.task_id]["live_metrics"] = metrics
                _tasks_cache[self.task_id]["live_mAP50"] = mAP50
                _tasks_cache[self.task_id]["strategies_triggered"] = self._triggered_strategies

        # Don't trigger before minimum epoch threshold
        if epoch < self.cfg.min_epochs_before_trigger:
            return

        # Handle augmentation boost countdown
        if self._augment_boost_remaining > 0:
            self._augment_boost_remaining -= 1
            if self._augment_boost_remaining == 0:
                self._end_augment_boost()

        # Check for plateau
        strategy = self._check_plateau(epoch)
        if strategy:
            self._trigger_strategy(strategy)

    def _check_plateau(self, current_epoch: int) -> Optional[dict]:
        """Detect plateau using sliding window comparison. Returns strategy dict or None."""
        if len(self._map_history) < self.cfg.window:
            return None

        recent = self._map_history[-self.cfg.window:]
        older = self._map_history[-self.cfg.window * 2:-self.cfg.window]

        if not recent or not older:
            return None

        avg_recent = sum(m for _, m in recent) / len(recent)
        avg_older = sum(m for _, m in older) / len(older)
        improvement = avg_recent - avg_older

        if improvement >= self.cfg.min_improvement:
            return None  # Still improving, not plateau

        # Plateau detected — determine best strategy
        if self._lr_reduction_count < self.cfg.lr_reduction_max_times:
            return {
                "level": 1,
                "action": "lr_decay",
                "improvement": improvement,
                "avg_recent": avg_recent,
            }
        elif not self._augment_boost_active:
            return {
                "level": 2,
                "action": "augment_boost",
                "improvement": improvement,
                "avg_recent": avg_recent,
            }
        elif not self._signaled_expansion:
            target_map = self.cfg.expansion_target_map
            current_best = max(m for _, m in self._map_history) if self._map_history else 0
            if current_best >= target_map - 0.05 and self._expansion_round < self.cfg.max_expansion_rounds:
                return {
                    "level": 3,
                    "action": "data_expansion",
                    "improvement": improvement,
                    "avg_recent": avg_recent,
                    "current_best": current_best,
                }

        return None

    def _trigger_strategy(self, strategy: dict) -> None:
        level = strategy["level"]
        action = strategy["action"]
        logging.warning(
            f"[{self.task_id}][PLATEAU] Level-{level} {action} triggered: "
            f"improvement={strategy['improvement']:.5f}, avg_mAP50={strategy['avg_recent']:.5f}"
        )
        self._triggered_strategies.append({
            "epoch": self._last_reported_epoch,
            "level": level,
            "action": action,
            "mAP50": strategy.get("avg_recent", 0),
        })

        if level == 1:
            self._apply_lr_decay()
        elif level == 2:
            self._start_augment_boost()
        elif level == 3:
            self._signal_data_expansion(strategy)

    def _apply_lr_decay(self) -> None:
        """Signal LR reduction to task cache for Business API to handle."""
        self._lr_reduction_count += 1
        new_lr = max(
            self.cfg.min_lr,
            self.cfg.lr_reduction_factor,
        )
        with _tasks_lock:
            if self.task_id in _tasks_cache:
                _tasks_cache[self.task_id]["lr_decay_triggered"] = True
                _tasks_cache[self.task_id]["lr_decay_count"] = self._lr_reduction_count
                _tasks_cache[self.task_id]["lr_decay_signal"] = {
                    "factor": self.cfg.lr_reduction_factor,
                    "min_lr": self.cfg.min_lr,
                    "epoch": self._last_reported_epoch,
                    "current_mAP50": self._map_history[-1][1] if self._map_history else 0,
                }
        logging.info(
            f"[{self.task_id}][PLATEAU] LR decay #{self._lr_reduction_count} "
            f"signaled to Business API. Cache updated."
        )

    def _start_augment_boost(self) -> None:
        """Signal augmentation boost to task cache."""
        self._augment_boost_active = True
        self._augment_boost_remaining = self.cfg.augmentation_boost_epochs
        with _tasks_lock:
            if self.task_id in _tasks_cache:
                _tasks_cache[self.task_id]["augment_boost_active"] = True
                _tasks_cache[self.task_id]["augment_boost_remaining"] = self._augment_boost_remaining
                _tasks_cache[self.task_id]["augment_boost_signal"] = {
                    "epochs": self.cfg.augmentation_boost_epochs,
                    "mixup": self.cfg.boosted_mixup,
                    "copy_paste": self.cfg.boosted_copy_paste,
                    "degrees": self.cfg.boosted_degrees,
                    "translate": self.cfg.boosted_translate,
                    "scale": self.cfg.boosted_scale,
                    "start_epoch": self._last_reported_epoch,
                }
        logging.info(
            f"[{self.task_id}][PLATEAU] Augmentation boost STARTED for {self.cfg.augmentation_boost_epochs} epochs. "
            f"mixup={self.cfg.boosted_mixup}, copy_paste={self.cfg.boosted_copy_paste}"
        )

    def _end_augment_boost(self) -> None:
        """Signal augmentation boost ended."""
        self._augment_boost_active = False
        with _tasks_lock:
            if self.task_id in _tasks_cache:
                _tasks_cache[self.task_id]["augment_boost_active"] = False
                _tasks_cache[self.task_id]["augment_boost_signal"] = None
        logging.info(f"[{self.task_id}][PLATEAU] Augmentation boost ENDED. Resuming normal augmentation.")

    def _signal_data_expansion(self, strategy: dict) -> None:
        """Signal data expansion request to Business API via cache."""
        self._signaled_expansion = True
        self._expansion_round += 1
        with _tasks_lock:
            if self.task_id in _tasks_cache:
                _tasks_cache[self.task_id]["data_expansion_requested"] = True
                _tasks_cache[self.task_id]["data_expansion_round"] = self._expansion_round
                _tasks_cache[self.task_id]["data_expansion_signal"] = {
                    "round": self._expansion_round,
                    "current_mAP50": strategy.get("avg_recent", 0),
                    "target_mAP50": self.cfg.expansion_target_map,
                    "epoch": self._last_reported_epoch,
                    "recommendation": (
                        f"Use ActiveLearningPipeline to expand dataset. "
                        f"Current best mAP50={strategy.get('current_best', 0):.4f}, "
                        f"target={self.cfg.expansion_target_map:.2f}. "
                        f"Suggest searching HuggingFace/Kaggle for fire+smoke detection datasets."
                    ),
                }
        logging.warning(
            f"[{self.task_id}][PLATEAU] Data expansion signal sent to Business API "
            f"(round {self._expansion_round}). mAP50={strategy.get('avg_recent', 0):.4f}"
        )

    def get_status(self) -> dict:
        """Return current plateau detection status."""
        return {
            "enabled": self.cfg.enabled,
            "lr_reduction_count": self._lr_reduction_count,
            "augment_boost_active": self._augment_boost_active,
            "augment_boost_remaining": self._augment_boost_remaining,
            "expansion_round": self._expansion_round,
            "signaled_expansion": self._signaled_expansion,
            "current_best_mAP50": max((m for _, m in self._map_history), default=0.0),
            "recent_mAP50": self._map_history[-1][1] if self._map_history else 0.0,
            "strategies_triggered": self._triggered_strategies,
        }


# ==================== Training Endpoints ====================

def _run_training_sync(
    task_id: str,
    model: str,
    data_yaml: str,
    epochs: int,
    imgsz: int,
    batch: int,
    output_dir: str,
    device: str,
    auto_export: bool = True,
    resume_checkpoint: Optional[str] = None,
    augmentation_preset: Optional[str] = None,
    _loop: Optional["asyncio.AbstractEventLoop"] = None,
    hpo_params: Optional[Dict[str, Any]] = None,
    resume_from: Optional[str] = None,
) -> None:
    """Run YOLO training synchronously. Called from background task."""
    # Import here to avoid import-time errors on systems without GPU
    from src.training.runner import YOLOTrainer, TrainingCancelled
    from src.training.config import DEFAULT_TRAINING_CONFIG, DEFAULT_PLATEAU_CONFIG, AUGMENTATION_PRESETS

    max_retries = 2
    retry_delay = 180  # 3 minutes
    last_error = None
    for attempt in range(max_retries + 1):
        with _cancel_lock:
            cancel_event = _cancel_events.get(task_id)

        try:
            logging.info(f"[{task_id}] Starting training: model={model}, data={data_yaml}, epochs={epochs}, device={device}, resume={resume_checkpoint}")

            # Update status to running
            _tasks_cache[task_id]["status"] = "running"
            _tasks_cache[task_id]["started_at"] = datetime.now().isoformat()
            _task_set(task_id, _tasks_cache[task_id])

            # --- DVC dataset versioning (graceful degradation) ---
            data_yaml_path = Path(data_yaml).resolve().parent
            version_file = Path(output_dir) / task_id / "dataset_version.json"
            try:
                from src.scripts.dvc_versioning import record_version
                record_version(str(data_yaml_path), str(version_file))
                _tasks_cache[task_id]["dataset_version_file"] = str(version_file)
            except Exception as dvc_err:
                logging.warning(f"[{task_id}] DVC versioning skipped: {dvc_err}")

            # Create runner — use resume_from best.pt if provided (transfer learning)
            model_to_use = resume_from if resume_from else model
            if resume_from and Path(resume_from).exists():
                logging.info(f"[{task_id}] Loading from best.pt checkpoint: {resume_from}")
            runner = YOLOTrainer(
                model=model_to_use,
                output_dir=Path(output_dir),
            )

            # Create training config
            config = DEFAULT_TRAINING_CONFIG
            config.epochs = epochs
            config.imgsz = imgsz
            config.batch = batch
            config.device = device
            config.resume_checkpoint = resume_checkpoint

            # Apply augmentation preset if specified
            if augmentation_preset and augmentation_preset in AUGMENTATION_PRESETS:
                preset = AUGMENTATION_PRESETS[augmentation_preset]
                config.mosaic = preset.mosaic
                config.mixup = preset.mixup
                config.copy_paste = preset.copy_paste
                config.copy_paste_mode = preset.copy_paste_mode
                config.degrees = preset.degrees
                config.translate = preset.translate
                config.scale = preset.scale
                config.shear = preset.shear
                config.perspective = preset.perspective
                config.flipud = preset.flipud
                config.fliplr = preset.fliplr
                config.hsv_h = preset.hsv_h
                config.hsv_s = preset.hsv_s
                config.hsv_v = preset.hsv_v
                logging.info(
                    f"[{task_id}] Applied augmentation preset '{augmentation_preset}': "
                    f"mosaic={preset.mosaic}, mixup={preset.mixup}, copy_paste={preset.copy_paste}, "
                    f"degrees={preset.degrees}, translate={preset.translate}, scale={preset.scale}, "
                    f"hsv_h={preset.hsv_h}, hsv_s={preset.hsv_s}, hsv_v={preset.hsv_v}"
                )

            # Apply HPO params (Agent 2: consume HPO best_params)
            if hpo_params:
                if hpo_params.get("lr0") is not None:
                    config.lr0 = hpo_params["lr0"]
                if hpo_params.get("lrf") is not None:
                    config.lrf = hpo_params["lrf"]
                if hpo_params.get("weight_decay") is not None:
                    config.weight_decay = hpo_params["weight_decay"]
                if hpo_params.get("momentum") is not None:
                    config.momentum = hpo_params["momentum"]
                logging.info(f"[{task_id}] Applied HPO params: lr0={config.lr0}, lrf={config.lrf}, weight_decay={config.weight_decay}, momentum={config.momentum}")

            logging.info(f"[{task_id}] Config device after assignment: {config.device}")
            logging.info(f"[{task_id}] Config to_dict device: {config.to_dict().get('device')}")

            # Dynamic plateau detection manager
            plateau_mgr = DynamicTrainingManager(
                task_id=task_id,
                plateau_config=DEFAULT_PLATEAU_CONFIG,
                device=config.device,
            )
            logging.info(f"[{task_id}] DynamicTrainingManager enabled: window={DEFAULT_PLATEAU_CONFIG.window}, "
                         f"min_improvement={DEFAULT_PLATEAU_CONFIG.min_improvement}")

            # Progress callback: updates tasks cache and checks cancellation each epoch.
            def _on_progress(epoch: int, total: int) -> None:
                try:
                    progress = ((epoch + 1) / total) * 100.0 if total > 0 else 0.0
                    with _tasks_lock:
                        _tasks_cache[task_id]["current_epoch"] = epoch
                        _tasks_cache[task_id]["total_epochs"] = total
                        _tasks_cache[task_id]["progress"] = progress
                    _task_set(task_id, _tasks_cache[task_id])
                    if cancel_event and cancel_event.is_set():
                        raise TrainingCancelled("Training cancelled by user")
                except TrainingCancelled:
                    raise
                except Exception as e:
                    logging.warning(f"[{task_id}] Progress callback error: {e}")

            # Metric callback: feeds live metrics to plateau detector
            def _on_metric(epoch: int, total: int, metrics: dict) -> None:
                try:
                    plateau_mgr.on_metric(epoch, total, metrics)
                except Exception as e:
                    logging.warning(f"[{task_id}] Metric callback error: {e}")

            # Run training with progress + metric tracking
            result = runner.train(
                data_yaml=Path(data_yaml),
                config=config,
                progress_callback=_on_progress,
                metric_callback=_on_metric,
            )

            # Update with results
            if result.status == "completed":
                _tasks_cache[task_id]["status"] = "completed"
                _tasks_cache[task_id]["progress"] = 100.0
                _tasks_cache[task_id]["metrics"] = result.metrics or {}
                _tasks_cache[task_id]["model_path"] = str(result.model_path) if result.model_path else None
                _tasks_cache[task_id]["early_stopped"] = result.early_stopped

                # Record dataset_version in task record
                if version_file.exists():
                    try:
                        _tasks_cache[task_id]["dataset_version"] = json.loads(version_file.read_text())
                    except Exception:
                        pass

                logging.info(f"[{task_id}] Training completed successfully (early_stopped={result.early_stopped})")

                # Auto-trigger ONNX export if enabled
                if auto_export and result.model_path:
                    export_task_id = f"{task_id}_export"
                    model_path_str = str(result.model_path)
                    logging.info(f"[{task_id}] Auto-triggering ONNX export: {export_task_id}")
                    _task_set(export_task_id, {
                        "task_id": export_task_id,
                        "type": "export",
                        "status": "submitted",
                        "model_path": model_path_str,
                        "platform": "jetson_orin",
                        "imgsz": 640,
                        "formats": ["onnx"],
                        "progress": 0.0,
                        "triggered_by": task_id,
                        "created_at": datetime.now().isoformat()
                    })
                    try:
                        # Use the event loop passed from the FastAPI thread (the caller always passes it).
                        # We cannot call asyncio.get_event_loop() or asyncio.get_running_loop() here —
                        # we're inside ThreadPoolExecutor which has no event loop, causing:
                        # RuntimeError: There is no current event loop in thread 'ThreadPoolExecutor-0_X'
                        if _loop is not None:
                            _loop.run_in_executor(
                                None,
                                _run_export_sync,
                                export_task_id,
                                model_path_str,
                                "jetson_orin",
                                640,
                                ["onnx"],
                                False,
                                False,  # int8_quantize
                                None,   # calibration_data_dir
                            )
                            logging.info(f"[{task_id}] ONNX export task {export_task_id} started")
                        else:
                            logging.warning(f"[{task_id}] No event loop available for ONNX export — skipping auto-export (training status still saved to Redis)")
                    except Exception as export_err:
                        # Export failure should NOT affect the training completion status in Redis.
                        # The model is already saved; export can be retried manually.
                        logging.error(f"[{task_id}] Failed to trigger ONNX export: {export_err}", exc_info=True)
            elif result.status == "cancelled":
                _tasks_cache[task_id]["status"] = "cancelled"
                _tasks_cache[task_id]["error"] = "Training cancelled by user"
                logging.info(f"[{task_id}] Training cancelled")
            else:
                _tasks_cache[task_id]["status"] = "failed"
                _tasks_cache[task_id]["error"] = result.error or "Unknown error"
                logging.error(f"[{task_id}] Training failed: {result.error}")

        except TrainingCancelled:
            # User cancelled — do not retry, propagate immediately
            _tasks_cache[task_id]["status"] = "cancelled"
            _tasks_cache[task_id]["error"] = "Training cancelled by user"
            logging.info(f"[{task_id}] Training cancelled")
            _tasks_cache[task_id]["completed_at"] = datetime.now().isoformat()
            _task_set(task_id, _tasks_cache[task_id])
            with _cancel_lock:
                _cancel_events.pop(task_id, None)
            return
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logging.warning(
                    f"[{task_id}] Training failed (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {retry_delay}s: {e}"
                )
                _tasks_cache[task_id]["status"] = "retrying"
                _tasks_cache[task_id]["error"] = str(e)
                _task_set(task_id, _tasks_cache[task_id])
                # Signal the trainer to stop before sleeping
                with _cancel_lock:
                    if task_id in _cancel_events:
                        _cancel_events[task_id].set()
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff: 180 -> 360 -> 720
                continue
            else:
                # Final failure after all retries exhausted
                logging.error(f"[{task_id}] Training failed after {max_retries + 1} attempts: {e}", exc_info=True)
                _tasks_cache[task_id]["status"] = "failed"
                _tasks_cache[task_id]["error"] = f"Failed after {max_retries + 1} attempts: {last_error}"
        finally:
            _tasks_cache[task_id]["completed_at"] = datetime.now().isoformat()
            # Write to Redis BEFORE returning so polling always sees final state
            _task_set(task_id, _tasks_cache[task_id])
            # Clean up cancel event (only if not returning for retry)
            with _cancel_lock:
                _cancel_events.pop(task_id, None)


def _run_export_sync(
    task_id: str,
    model_path: str,
    platform: str,
    imgsz: int,
    formats: Optional[List[str]] = None,
    auto_benchmark: bool = False,
    int8_quantize: bool = False,
    calibration_data_dir: Optional[str] = None,
) -> None:
    """Run model export synchronously. Called from background task."""
    from src.deployment.exporter import ModelExporter

    formats = formats or ["onnx"]
    single_format = len(formats) == 1

    # If INT8 is requested, ensure engine-int8 is in formats
    if int8_quantize and "engine-int8" not in formats:
        formats = list(formats) + ["engine-int8"]
        logging.info(f"[{task_id}] INT8 requested, adding engine-int8 to formats: {formats}")

    try:
        logging.info(f"[{task_id}] Starting export: model={model_path}, platform={platform}, formats={formats}")

        # Update status to running
        _tasks_cache[task_id]["status"] = "running"
        _tasks_cache[task_id]["started_at"] = datetime.now().isoformat()
        _task_set(task_id, _tasks_cache[task_id])

        if single_format:
            # Single-format: use existing ModelExporter
            exporter = ModelExporter(output_dir=Path(model_path).parent)
            result = exporter.export(
                model_path=Path(model_path),
                platform=platform,
                imgsz=imgsz,
            )

            if result.status == "success":
                _tasks_cache[task_id]["status"] = "completed"
                _tasks_cache[task_id]["progress"] = 100.0
                _tasks_cache[task_id]["export_path"] = str(result.model_path)
                _tasks_cache[task_id]["size_mb"] = result.size_mb
                _tasks_cache[task_id]["format"] = result.format
                _tasks_cache[task_id]["formats"] = formats
                logging.info(f"[{task_id}] Export completed: {result.model_path} ({result.size_mb:.1f}MB)")

                # Auto-benchmark after single-format export
                if auto_benchmark:
                    _run_benchmark_sync(task_id, str(result.model_path), result.format, imgsz)
            else:
                _tasks_cache[task_id]["status"] = "failed"
                _tasks_cache[task_id]["error"] = result.error or "Unknown export error"
                logging.error(f"[{task_id}] Export failed: {result.error}")
        else:
            # Multi-format: use YOLOTrainer.export_multi
            from src.training.runner import YOLOTrainer

            runner = YOLOTrainer(output_dir=Path(model_path).parent.parent)
            model_path_obj = Path(model_path)

            # Resolve calibration data directory
            calib_dir: Optional[Path] = None
            if calibration_data_dir:
                calib_dir = Path(calibration_data_dir)
            elif int8_quantize:
                # Try to derive from data.yaml if available in model parent dir
                model_dir = model_path_obj.parent
                data_yaml = model_dir / "data.yaml"
                if data_yaml.exists():
                    calib_dir = model_dir / "train" / "images"
                    if not calib_dir.exists():
                        calib_dir = model_dir / "valid" / "images"
                        if not calib_dir.exists():
                            calib_dir = None

            results = runner.export_multi(
                model_path=model_path_obj,
                formats=formats,
                platform=platform,
                imgsz=imgsz,
                calibration_image_dir=calib_dir,
                calibration_n=1000,
            )

            # Summarise results
            successful = {fmt: r for fmt, r in results.items() if r.get("path")}
            failed = {fmt: r.get("error") for fmt, r in results.items() if not r.get("path")}

            _tasks_cache[task_id]["status"] = "completed"
            _tasks_cache[task_id]["progress"] = 100.0
            _tasks_cache[task_id]["export_results"] = results
            _tasks_cache[task_id]["formats"] = formats
            _tasks_cache[task_id]["failed_formats"] = failed if failed else None
            logging.info(f"[{task_id}] Multi-format export completed: {list(successful.keys())}")

    except Exception as e:
        logging.error(f"[{task_id}] Export exception: {e}", exc_info=True)
        _tasks_cache[task_id]["status"] = "failed"
        _tasks_cache[task_id]["error"] = str(e)
    finally:
        _tasks_cache[task_id]["completed_at"] = datetime.now().isoformat()
        _task_set(task_id, _tasks_cache[task_id])


def _run_benchmark_sync(
    task_id: str,
    model_path: str,
    format: str = "onnx",
    imgsz: int = 640,
    warmup: int = 10,
    runs: int = 100,
) -> None:
    """Run benchmark on an exported model. Called from background thread."""
    try:
        logging.info(f"[{task_id}] Starting benchmark: model={model_path}, format={format}")

        _tasks_cache[task_id]["status"] = "running"
        _tasks_cache[task_id]["started_at"] = datetime.now().isoformat()
        _task_set(task_id, _tasks_cache[task_id])

        from src.benchmark.runner import BenchmarkRunner

        runner = BenchmarkRunner()
        result = runner.run(
            model_path=Path(model_path),
            format=format,
            imgsz=imgsz,
            warmup=warmup,
            runs=runs,
        )

        _tasks_cache[task_id]["status"] = "completed"
        _tasks_cache[task_id]["progress"] = 100.0
        _tasks_cache[task_id]["benchmark"] = result.to_dict()
        logging.info(
            f"[{task_id}] Benchmark completed: FPS={result.fps}, "
            f"params={result.params_m}M, gflops={result.gflops}, "
            f"size={result.size_mb}MB, gpu={result.gpu_available}"
        )
    except Exception as e:
        logging.error(f"[{task_id}] Benchmark exception: {e}", exc_info=True)
        _tasks_cache[task_id]["status"] = "failed"
        _tasks_cache[task_id]["error"] = str(e)
    finally:
        _tasks_cache[task_id]["completed_at"] = datetime.now().isoformat()
        _task_set(task_id, _tasks_cache[task_id])

@router.post("/train/start")
async def start_training(
    request: TrainStartRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Start a training job on the GPU.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    # Create task record
    task_id = request.task_id
    _task_set(task_id, {
        "task_id": task_id,
        "type": "training",
        "status": "submitted",
        "model": request.model,
        "data_yaml": request.data_yaml,
        "epochs": request.epochs,
        "imgsz": request.imgsz,
        "batch": request.batch,
        "output_dir": request.output_dir,
        "device": request.device,
        "progress": 0.0,
        "current_epoch": 0,
        "total_epochs": request.epochs,
        "auto_export": request.auto_export,
        "augmentation_preset": request.augmentation_preset,
        "resume_from": request.resume_from,
        "created_at": datetime.now().isoformat()
    })

    # Create cancel event for this task
    with _cancel_lock:
        _cancel_events[task_id] = threading.Event()

    # Use daemon thread instead of loop.run_in_executor to avoid asyncio/PyTorch conflicts
    t = threading.Thread(
        target=_run_training_sync,
        args=(
            task_id,
            request.model,
            request.data_yaml,
            request.epochs,
            request.imgsz,
            request.batch,
            request.output_dir,
            request.device,
            request.auto_export,
            None,
            request.augmentation_preset,
            None,  # _loop not needed for basic training; export can be done separately
            {"lr0": request.lr0, "lrf": request.lrf, "weight_decay": request.weight_decay, "momentum": request.momentum}
            if any(x is not None for x in [request.lr0, request.lrf, request.weight_decay, request.momentum])
            else None,
            request.resume_from,
        ),
        daemon=True,
    )
    t.start()

    return {
        "task_id": task_id,
        "status": "started",
        "worker_id": f"worker_{uuid.uuid4().hex[:6]}",
        "message": "Training task started"
    }


# ==================== Curriculum Training (3-Stage Progressive) ====================

class CurriculumStageRequest(BaseModel):
    """Single stage in curriculum training request."""
    name: str = "rapid_validation"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    model: str = "yolo11m"
    augmentation_preset: Optional[str] = "balanced"
    num_gpus: int = Field(1, description="Number of GPUs for DDP training (1=single, 2+=multi-GPU on same node)")
    warmup_ratio: float = Field(0.05, description="Fraction of epochs for warmup (default 5%%). Mirrors WARMUP_RATIO from autoresearch.")
    mosaic: float = 1.0
    mixup: float = 0.1
    copy_paste: float = 0.1
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    # close_mosaic now computed proportionally in _build_config (20%% of epochs)


class CurriculumStartRequest(BaseModel):
    """Request to start progressive 3-stage curriculum training."""
    task_id: str = Field(..., description="Task identifier")
    data_yaml: str = Field(..., description="Dataset YAML path")
    output_dir: str = Field("/home/wangxin/runs", description="Output directory")
    device: str = Field("cuda:0", description="Device")
    auto_export: bool = Field(True, description="Auto-export ONNX after completion")
    # Three stages (all optional, use defaults if not specified)
    stage1: Optional[CurriculumStageRequest] = Field(
        None,
        description="Stage 1: Rapid validation (default: 50ep@640px)"
    )
    stage2: Optional[CurriculumStageRequest] = Field(
        None,
        description="Stage 2: Deep training (default: 150ep@1280px)"
    )
    stage3: Optional[CurriculumStageRequest] = Field(
        None,
        description="Stage 3: Fine-tuning (default: 100ep@1280px)"
    )
    # Gate thresholds
    stage1_min_map: float = Field(0.50, description="Min mAP50 to pass Stage 1")
    stage2_target_map: float = Field(0.90, description="Target mAP50 — stop if reached in Stage 2")
    stage2_min_for_stage3: float = Field(0.80, description="Min mAP50 to proceed to Stage 3")


def _run_curriculum_sync(
    task_id: str,
    data_yaml: str,
    output_dir: str,
    device: str,
    auto_export: bool,
    stage1_req: Optional[dict],
    stage2_req: Optional[dict],
    stage3_req: Optional[dict],
    stage1_min_map: float,
    stage2_target_map: float,
    stage2_min_for_stage3: float,
    _loop: Optional["asyncio.AbstractEventLoop"] = None,
) -> None:
    """Run 3-stage curriculum training synchronously in background thread."""
    from src.training.runner import PipelineCurriculumTrainer, CurriculumStage, CurriculumConfig
    from pathlib import Path

    # Track current stage for callbacks — initialized to Stage 1 immediately
    _current_stage = {"num": 1, "name": "rapid_validation"}

    # Immediately persist Stage 1 + running status so polling sees both ASAP
    with _tasks_lock:
        if task_id in _tasks_cache:
            _tasks_cache[task_id]["status"] = "running"
            _tasks_cache[task_id]["curriculum_stage"] = "rapid_validation"
            _tasks_cache[task_id]["curriculum_stage_num"] = 1
            _tasks_cache[task_id]["progress"] = 3.0  # Stage 1 just started
    # Flush to Redis immediately so polling sees it
    _task_set(task_id, _tasks_cache.get(task_id, {}))

    _progress_flush_counter = 0

    def _on_progress(epoch: int, total: int) -> None:
        try:
            with _tasks_lock:
                if task_id in _tasks_cache:
                    _tasks_cache[task_id]["current_epoch"] = epoch
                    _tasks_cache[task_id]["total_epochs"] = total
                    _tasks_cache[task_id]["curriculum_stage"] = _current_stage["name"]
                    _tasks_cache[task_id]["curriculum_stage_num"] = _current_stage["num"]
                    # Estimate progress: stage fraction + epoch fraction within stage
                    stage_weight = {"rapid_validation": 10, "deep_training": 50, "fine_tuning": 40}
                    sw = stage_weight.get(_current_stage["name"], 10)
                    base = (_current_stage["num"] - 1) / 3.0 * 100.0
                    fraction = (epoch / total) * sw * 0.3 if total > 0 else 0
                    _tasks_cache[task_id]["progress"] = min(base + fraction, 99.0)
                    # Check for augment boost signal from DynamicTrainingManager (Level 2)
                    if _tasks_cache[task_id].get("augment_boost_active"):
                        boost_signal = _tasks_cache[task_id].get("augment_boost_signal", {})
                        logging.warning(
                            f"[{task_id}][CURRICULUM] Augment boost ACTIVE: "
                            f"mixup={boost_signal.get('mixup')}, copy_paste={boost_signal.get('copy_paste')}. "
                            f"Restart training with these params to apply."
                        )
                    logging.debug(f"[{task_id}][CURRICULUM] {_current_stage['name']} EP {epoch}/{total}")
                    # Throttle Redis writes: flush every 5 epochs to avoid excessive I/O
                    nonlocal _progress_flush_counter
                    _progress_flush_counter += 1
                    if _progress_flush_counter % 5 == 0:
                        _task_set(task_id, _tasks_cache[task_id])
        except Exception as e:
            logging.warning(f"[{task_id}] Curriculum progress callback error: {e}")

    # Build stages from requests or defaults
    def _make_stage(req_dict: Optional[dict], default: CurriculumStage) -> CurriculumStage:
        if req_dict:
            return CurriculumStage(
                name=req_dict.get("name", default.name),
                epochs=req_dict.get("epochs", default.epochs),
                imgsz=req_dict.get("imgsz", default.imgsz),
                batch=req_dict.get("batch", default.batch),
                model=req_dict.get("model", default.model),
                augmentation_preset=req_dict.get("augmentation_preset", default.augmentation_preset),
                num_gpus=req_dict.get("num_gpus", default.num_gpus),
                warmup_ratio=req_dict.get("warmup_ratio", default.warmup_ratio),
                mosaic=req_dict.get("mosaic", default.mosaic),
                mixup=req_dict.get("mixup", default.mixup),
                copy_paste=req_dict.get("copy_paste", default.copy_paste),
                degrees=req_dict.get("degrees", default.degrees),
                translate=req_dict.get("translate", default.translate),
                scale=req_dict.get("scale", default.scale),
            )
        return default

    default_cfg = CurriculumConfig()
    cfg = CurriculumConfig(
        stage1=_make_stage(stage1_req, default_cfg.stage1),
        stage2=_make_stage(stage2_req, default_cfg.stage2),
        stage3=_make_stage(stage3_req, default_cfg.stage3),
        stage1_min_map=stage1_min_map,
        stage2_target_map=stage2_target_map,
        stage2_min_for_stage3=stage2_min_for_stage3,
    )

    trainer = PipelineCurriculumTrainer(output_dir=Path(output_dir))

    # Stage callback — called at the START of each stage by PipelineCurriculumTrainer.train()
    def _stage_tracker(sn: int, name: str, mAP: float, decision: dict) -> None:
        logging.info(f"[{task_id}][CURRICULUM] Stage {sn} ({name}) callback: mAP50={mAP:.4f}, decision={decision}")

        # ALWAYS update _current_stage (not just on "starting") — this is the authoritative stage tracker
        _current_stage["num"] = sn
        _current_stage["name"] = name
        logging.info(f"[{task_id}][CURRICULUM] Starting stage {sn}: {name}")

        # Persist stage info to cache immediately
        with _tasks_lock:
            if task_id in _tasks_cache:
                _tasks_cache[task_id]["curriculum_stage"] = name
                _tasks_cache[task_id]["curriculum_stage_num"] = sn
                _tasks_cache[task_id]["curriculum_stage_history"] = list(trainer._stage_history)
                if trainer._stage_history:
                    latest = trainer._stage_history[-1]
                    _tasks_cache[task_id]["curriculum_stage_mAP"] = latest.get("mAP50", 0.0)
                    _tasks_cache[task_id]["metrics"] = {"mAP50": latest.get("mAP50", 0.0)}
                stage_weight = {"rapid_validation": 10, "deep_training": 50, "fine_tuning": 40}
                sw = stage_weight.get(name, 10)
                _tasks_cache[task_id]["progress"] = (sn - 1) / 3 * 100 + sw * 0.1
                # Flush to Redis so status endpoint sees the update
                _task_set(task_id, _tasks_cache[task_id])

    result = trainer.train(
        data_yaml=Path(data_yaml),
        config=cfg,
        progress_callback=_on_progress,
        stage_callback=_stage_tracker,
    )

    # Final cache update — called after trainer.train() returns
    def _update_curriculum_stage() -> None:
        with _tasks_lock:
            if task_id in _tasks_cache:
                # Final stage history (includes all completed stages, including failed ones)
                stage_history = list(trainer._stage_history)
                # Get the mAP from the last completed stage
                final_mAP = 0.0
                if stage_history:
                    final_mAP = stage_history[-1].get("mAP50", 0.0)
                _tasks_cache[task_id]["status"] = result.status
                _tasks_cache[task_id]["metrics"] = result.metrics or {"mAP50": final_mAP}
                _tasks_cache[task_id]["model_path"] = str(result.model_path) if result.model_path else None
                _tasks_cache[task_id]["curriculum_stage_history"] = stage_history
                _tasks_cache[task_id]["curriculum_stage_mAP"] = final_mAP
                _tasks_cache[task_id]["curriculum_stage"] = stage_history[-1]["name"] if stage_history else "unknown"
                _tasks_cache[task_id]["progress"] = 100.0
        _task_set(task_id, _tasks_cache.get(task_id, {}))

        if result.status == "completed":
            logging.info(f"[{task_id}][CURRICULUM] All stages completed. Best mAP50={result.metrics.get('mAP50', 0):.4f}")

            # Auto-trigger ONNX export using the best weights from curriculum
            if auto_export and result.model_path:
                export_task_id = f"{task_id}_export"
                model_path_str = str(result.model_path)
                logging.info(f"[{task_id}] Auto-triggering ONNX export after curriculum: {export_task_id}")
                _task_set(export_task_id, {
                    "task_id": export_task_id,
                    "type": "export",
                    "status": "submitted",
                    "model_path": model_path_str,
                    "platform": "jetson_orin",
                    "imgsz": 640,
                    "formats": ["onnx"],
                    "progress": 0.0,
                    "triggered_by": task_id,
                    "created_at": datetime.now().isoformat()
                })
                try:
                    if _loop is not None:
                        _loop.run_in_executor(
                            None,
                            _run_export_sync,
                            export_task_id,
                            model_path_str,
                            "jetson_orin",
                            640,
                            ["onnx"],
                            False,
                            False,  # int8_quantize
                            None,   # calibration_data_dir
                        )
                        logging.info(f"[{task_id}] ONNX export task {export_task_id} started")
                    else:
                        logging.warning(f"[{task_id}] No event loop available for ONNX export — skipping")
                except Exception as export_err:
                    logging.error(f"[{task_id}] Failed to trigger ONNX export: {export_err}", exc_info=True)

    _update_curriculum_stage()


@router.post("/train/curriculum/start")
async def start_curriculum_training(
    request: CurriculumStartRequest,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Start a progressive 3-stage curriculum training job.

    Stage 1 (Rapid Validation): 50 epochs @ 640px — cheap pipeline validation
    Stage 2 (Deep Training): 150 epochs @ 1280px — main training with strong augmentation
    Stage 3 (Fine-Tuning): 100 epochs @ 1280px — reduced augmentation for detail

    Gate decisions:
      - Stage 1 mAP50 < 0.50: ABORT (pipeline broken)
      - Stage 2 mAP50 >= 0.90: STOP (goal reached)
      - Stage 2 mAP50 >= 0.80: proceed to Stage 3
      - Stage 2 mAP50 < 0.80: trigger plateau strategies
    """
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    task_id = request.task_id
    _task_set(task_id, {
        "task_id": task_id,
        "type": "curriculum",
        "status": "submitted",
        "data_yaml": request.data_yaml,
        "progress": 0.0,
        "current_epoch": 0,
        "curriculum_stage": "pending",
        "curriculum_stage_history": [],
        "created_at": datetime.now().isoformat(),
    })

    with _cancel_lock:
        _cancel_events[task_id] = threading.Event()

    # Write task params to cache file for subprocess
    import json
    s1 = request.stage1
    s2 = request.stage2
    s3 = request.stage3
    cache_file = f"/tmp/curriculum_{task_id}.json"
    with open(cache_file, "w") as f:
        json.dump({
            "task_id": task_id,
            "data_yaml": request.data_yaml,
            "output_dir": request.output_dir,
            "device": request.device,
            # Model from stage1 or default
            "model": s1.model if s1 else "yolo11m",
            # Stage 1
            "stage1_epochs": s1.epochs if s1 else 50,
            "stage1_imgsz": s1.imgsz if s1 else 640,
            "stage1_batch": s1.batch if s1 else 16,
            "stage1_num_gpus": s1.num_gpus if s1 else 1,
            "stage1_warmup_ratio": s1.warmup_ratio if s1 else 0.05,
            "stage1_mosaic": s1.mosaic if s1 else 1.0,
            "stage1_mixup": s1.mixup if s1 else 0.1,
            "stage1_copy_paste": s1.copy_paste if s1 else 0.1,
            "stage1_degrees": s1.degrees if s1 else 0.0,
            "stage1_translate": s1.translate if s1 else 0.1,
            "stage1_scale": s1.scale if s1 else 0.5,
            # Stage 2
            "stage2_epochs": s2.epochs if s2 else 150,
            "stage2_imgsz": s2.imgsz if s2 else 1280,
            "stage2_batch": s2.batch if s2 else 8,
            "stage2_num_gpus": s2.num_gpus if s2 else 1,
            "stage2_warmup_ratio": s2.warmup_ratio if s2 else 0.05,
            "stage2_mosaic": s2.mosaic if s2 else 1.0,
            "stage2_mixup": s2.mixup if s2 else 0.1,
            "stage2_copy_paste": s2.copy_paste if s2 else 0.1,
            "stage2_degrees": s2.degrees if s2 else 0.0,
            "stage2_translate": s2.translate if s2 else 0.1,
            "stage2_scale": s2.scale if s2 else 0.5,
            # Stage 3
            "stage3_epochs": s3.epochs if s3 else 100,
            "stage3_imgsz": s3.imgsz if s3 else 1280,
            "stage3_batch": s3.batch if s3 else 4,
            "stage3_num_gpus": s3.num_gpus if s3 else 1,
            "stage3_warmup_ratio": s3.warmup_ratio if s3 else 0.05,
            "stage3_mosaic": s3.mosaic if s3 else 0.0,
            "stage3_mixup": s3.mixup if s3 else 0.0,
            "stage3_copy_paste": s3.copy_paste if s3 else 0.0,
            "stage3_degrees": s3.degrees if s3 else 0.0,
            "stage3_translate": s3.translate if s3 else 0.0,
            "stage3_scale": s3.scale if s3 else 0.0,
            # Gate thresholds
            "stage1_min_map": request.stage1_min_map,
            "stage2_target_map": request.stage2_target_map,
            "stage2_min_for_stage3": request.stage2_min_for_stage3,
        }, f)

    # Run curriculum as a SEPARATE subprocess to avoid asyncio/PyTorch conflicts
    # that cause loop.run_in_executor() and threading to block the uvicorn worker
    training_api_dir = Path(__file__).parent.parent.parent  # .../training-api
    script_path = training_api_dir / "scripts" / "run_curriculum.py"
    venv_python = training_api_dir.parent / "training-venv" / "bin" / "python"

    proc = subprocess.Popen(
        [str(venv_python), str(script_path), task_id],
        cwd=str(training_api_dir),
        env=os.environ,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logging.info(f"[{task_id}] Curriculum subprocess started (PID={proc.pid})")

    return {
        "task_id": task_id,
        "status": "started",
        "worker_id": f"curriculum_{uuid.uuid4().hex[:6]}",
        "message": "Curriculum training started (3 stages)",
        "stages": "rapid_validation → deep_training → fine_tuning",
    }


@router.get("/train/status/{task_id}", response_model=TrainStatusResponse)
async def get_training_status(
    task_id: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Get training job status. Auto-resubmits tasks stuck in "submitted" for >60s
    (e.g. after a server restart that lost the background thread).
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    task = _task_get(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )

    # Nested helper for auto-resubmit — defined BEFORE any calls
    def _do_resubmit(candidate):
        nonlocal task
        # Circuit breaker: max 3 retries per task
        count = _retry_counts.get(task_id, 0)
        if count >= 3:
            logging.error(f"[{task_id}] Auto-resubmit circuit breaker open (3 retries exceeded), giving up")
            return
        _retry_counts[task_id] = count + 1
        # Exponential backoff: 2, 4, 8 seconds (capped at 300s)
        backoff = min(300, 2 ** count)
        if backoff > 0:
            time.sleep(backoff)
        # Ensure cancel event exists
        with _cancel_lock:
            if task_id not in _cancel_events:
                _cancel_events[task_id] = threading.Event()
        # Update Redis + cache to running and restart background executor
        reason = "failed_task" if candidate.get("status") == "failed" else "stuck_submitted"
        resubmit_count = count + 1
        resubmitted_at = datetime.now().isoformat()
        _task_set(task_id, {
            **candidate,
            "status": "running",
            "started_at": resubmitted_at,
            "progress": 0.0,
            "current_epoch": 0,
            "error": None,
            "resubmit_count": resubmit_count,
            "last_resubmitted_at": resubmitted_at,
            "resubmit_reason": reason,
        })
        task["status"] = "running"
        task["started_at"] = resubmitted_at
        task["progress"] = 0.0
        task["current_epoch"] = 0
        task["error"] = None
        task["resubmit_count"] = resubmit_count
        task["last_resubmitted_at"] = resubmitted_at
        task["resubmit_reason"] = reason
        # Fire the background thread using fresh Redis data
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            None,
            _run_training_sync,
            task_id,
            candidate.get("model", "yolo11m"),
            candidate.get("data_yaml"),
            candidate.get("epochs", 100),
            candidate.get("imgsz", 640),
            candidate.get("batch", 16),
            candidate.get("output_dir", "/home/wangxin/runs"),
            candidate.get("device", "cuda:0"),
            candidate.get("auto_export", True),
            None,
            candidate.get("augmentation_preset"),
            None,
            None,
            candidate.get("resume_from"),
        )

    # Auto-resubmit: if task is stuck in "submitted" for >60s in Redis, the background
    # executor was likely lost (e.g. server restart that lost the background thread).
    # Read from Redis directly to bypass stale in-memory cache.
    _redis_client_local = get_redis_client()
    resubmit_candidate = None
    if _redis_client_local is not None:
        try:
            raw = _redis_client_local.get(f"training:task:{task_id}")
            if raw:
                resubmit_candidate = json.loads(raw)
        except Exception:
            pass

    if resubmit_candidate and resubmit_candidate.get("status") == "submitted":
        created_str = resubmit_candidate.get("created_at")
        resubmit = False
        if created_str:
            try:
                created = datetime.fromisoformat(created_str)
                age = (datetime.now() - created).total_seconds()
                resubmit = age > 60
            except Exception:
                resubmit = True  # If we can't parse, assume it's stuck
        else:
            resubmit = True

        if resubmit:
            logging.warning(f"[{task_id}] Task stuck in 'submitted' — auto-resubmitting")
            _do_resubmit(resubmit_candidate)

    # Also auto-resubmit failed tasks (e.g. due to TypeError, OOM, etc.)
    elif resubmit_candidate and resubmit_candidate.get("status") == "failed":
        logging.warning(f"[{task_id}] Task previously failed ({resubmit_candidate.get('error','?')[:80]}) — auto-resubmitting")
        _do_resubmit(resubmit_candidate)

    # Also read plateau/live metrics from Redis (written by curriculum subprocess PlateauManager)
    # This enables AutoAdjustAgent to monitor curriculum training progress
    if _redis_client_local is not None:
        try:
            redis_hdata = _redis_client_local.hgetall(f"training:task:{task_id}")
            if redis_hdata:
                def _safe_json_loads(val):
                    if isinstance(val, str):
                        try:
                            return json.loads(val)
                        except json.JSONDecodeError:
                            return val
                    return val

                _live_mAP = redis_hdata.get("live_mAP50")
                _lr_dec = redis_hdata.get("lr_decay_triggered")
                _aug_boost = redis_hdata.get("augment_boost_active")
                _data_exp = redis_hdata.get("data_expansion_requested")
                _strat = redis_hdata.get("strategies_triggered")
                _llm_diag = redis_hdata.get("llm_diagnosis")
                _lr_signal = redis_hdata.get("lr_decay_signal")
                _aug_signal = redis_hdata.get("augment_boost_signal")
                _data_signal = redis_hdata.get("data_expansion_signal")
                _in_stage = redis_hdata.get("in_stage_restarts")

                if _live_mAP and _live_mAP not in ("", "None"):
                    task["live_mAP50"] = float(_live_mAP)
                if _lr_dec and _lr_dec not in ("", "None"):
                    task["lr_decay_triggered"] = _lr_dec == "True"
                if _aug_boost and _aug_boost not in ("", "None"):
                    task["augment_boost_active"] = _aug_boost == "True"
                if _data_exp and _data_exp not in ("", "None"):
                    task["data_expansion_requested"] = _data_exp == "True"
                if _strat and _strat not in ("", "None"):
                    task["strategies_triggered"] = _safe_json_loads(_strat)
                if _llm_diag and _llm_diag not in ("", "None"):
                    task["llm_diagnosis"] = _safe_json_loads(_llm_diag)
                if _lr_signal and _lr_signal not in ("", "None"):
                    task["lr_decay_signal"] = _safe_json_loads(_lr_signal)
                if _aug_signal and _aug_signal not in ("", "None"):
                    task["augment_boost_signal"] = _safe_json_loads(_aug_signal)
                if _data_signal and _data_signal not in ("", "None"):
                    task["data_expansion_signal"] = _safe_json_loads(_data_signal)
                if _in_stage and _in_stage not in ("", "None"):
                    task["in_stage_restarts"] = int(_in_stage)
        except Exception as e:
            logging.warning(f"[{task_id}] Failed to read plateau signals from Redis: {e}")

    return TrainStatusResponse(
        task_id=task["task_id"],
        status=task.get("status", "unknown"),
        progress=task.get("progress", 0.0),
        current_epoch=task.get("current_epoch"),
        total_epochs=task.get("epochs"),
        metrics=task.get("metrics"),
        error=task.get("error"),
        early_stopped=task.get("early_stopped"),
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at"),
        live_mAP50=task.get("live_mAP50"),
        lr_decay_triggered=task.get("lr_decay_triggered"),
        lr_decay_signal=task.get("lr_decay_signal"),
        augment_boost_active=task.get("augment_boost_active"),
        augment_boost_signal=task.get("augment_boost_signal"),
        data_expansion_requested=task.get("data_expansion_requested"),
        data_expansion_signal=task.get("data_expansion_signal"),
        strategies_triggered=task.get("strategies_triggered"),
        resubmit_count=task.get("resubmit_count"),
        last_resubmitted_at=task.get("last_resubmitted_at"),
        resubmit_reason=task.get("resubmit_reason"),
        curriculum_stage=task.get("curriculum_stage"),
        curriculum_stage_num=task.get("curriculum_stage_num"),
        curriculum_stage_history=task.get("curriculum_stage_history"),
        curriculum_stage_mAP=task.get("curriculum_stage_mAP"),
    )


@router.post("/train/cancel/{task_id}")
async def cancel_training(
    task_id: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Cancel a training job.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    task = _task_get(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )

    with _tasks_lock:
        _tasks_cache[task_id]["status"] = "cancelled"
        _tasks_cache[task_id]["cancelled_at"] = datetime.now().isoformat()
    _task_set(task_id, _tasks_cache[task_id])

    # Signal the training thread to abort at the next epoch boundary
    with _cancel_lock:
        cancel_event = _cancel_events.get(task_id)
    if cancel_event:
        cancel_event.set()

    return {
        "task_id": task_id,
        "status": "cancelled",
        "message": "Training task cancelled"
    }


class TrainResumeRequest(BaseModel):
    """Internal training resume request."""
    task_id: str = Field(..., description="Original task identifier to resume from")
    output_dir: Optional[str] = Field(None, description="Override output directory (default: use original task's output_dir)")
    epochs: Optional[int] = Field(None, description="Override number of epochs (default: use original task's epochs)")


def _find_last_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the most recent last.pt checkpoint under output_dir/train/expN/weights/.
    Searches in reverse order to find the highest exp number.
    Returns the path to last.pt if found, otherwise None.
    """
    train_base = Path(output_dir) / "train"
    if not train_base.exists():
        return None

    # Collect all exp directories
    exp_dirs = [d for d in train_base.iterdir() if d.is_dir() and d.name.startswith("exp")]
    if not exp_dirs:
        return None

    # Sort by name descending (exp10 > exp9 > ... > exp1)
    exp_dirs.sort(key=lambda d: d.name, reverse=True)

    for exp_dir in exp_dirs:
        last_pt = exp_dir / "weights" / "last.pt"
        if last_pt.exists():
            logging.info(f"[_find_last_checkpoint] Found checkpoint: {last_pt}")
            return str(last_pt)

    return None


@router.post("/train/resume")
async def resume_training(
    request: TrainResumeRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Resume a training job from its last checkpoint (last.pt).

    Looks up the original task by task_id, finds the last.pt checkpoint,
    then starts a new training run with resume=True.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    # Look up original task
    original_task = _task_get(request.task_id)
    if original_task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original task {request.task_id} not found"
        )

    if original_task.get("type") != "training":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task {request.task_id} is not a training task"
        )

    # Determine output_dir and find checkpoint
    output_dir = request.output_dir or original_task.get("output_dir", "/home/wangxin/runs")
    last_pt = _find_last_checkpoint(output_dir)
    if last_pt is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No last.pt checkpoint found under {output_dir}/train/"
        )

    # Create resume task_id
    resume_task_id = f"{request.task_id}_resume_{uuid.uuid4().hex[:6]}"

    # Override epochs if provided
    epochs = request.epochs if request.epochs is not None else original_task.get("epochs", 100)

    _task_set(resume_task_id, {
        "task_id": resume_task_id,
        "type": "training",
        "status": "submitted",
        "model": original_task.get("model", "yolo11m"),
        "data_yaml": original_task.get("data_yaml"),
        "epochs": epochs,
        "output_dir": output_dir,
        "device": original_task.get("device", "cuda:0"),
        "auto_export": original_task.get("auto_export", True),
        "resume_checkpoint": last_pt,
        "resumed_from": request.task_id,
        "progress": 0.0,
        "current_epoch": 0,
        "total_epochs": epochs,
        "created_at": datetime.now().isoformat()
    })

    # Create cancel event
    _cancel_events[resume_task_id] = threading.Event()

    # Start training in background thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        _run_training_sync,
        resume_task_id,
        original_task.get("model", "yolo11m"),
        original_task.get("data_yaml"),
        epochs,
        original_task.get("imgsz", 640),
        original_task.get("batch", 16),
        output_dir,
        original_task.get("device", "cuda:0"),
        original_task.get("auto_export", True),
    )

    return {
        "task_id": resume_task_id,
        "status": "started",
        "resumed_from": request.task_id,
        "checkpoint": last_pt,
        "message": f"Training resume task started from {last_pt}"
    }


# ==================== HPO Endpoints ====================

def _run_hpo_sync(
    task_id: str,
    model: str,
    data_yaml: str,
    n_trials: int,
    epochs_per_trial: int,
    strategy: str = "asha",
) -> None:
    """Run HPO synchronously. Called from background thread.

    Args:
        strategy: 'asha' uses Ray Tune ASHA scheduler (default).
                  'bayesian' uses Bayesian Optimization via scikit-optimize.
    """
    from src.training.runner import YOLOTrainer, TrainingResult
    from src.training.config import HPOConfig, DEFAULT_TRAINING_CONFIG

    try:
        logging.info(
            f"[{task_id}] Starting HPO: model={model}, data={data_yaml}, "
            f"n_trials={n_trials}, strategy={strategy}"
        )

        _tasks_cache[task_id]["status"] = "running"
        _tasks_cache[task_id]["started_at"] = datetime.now().isoformat()
        _tasks_cache[task_id]["strategy"] = strategy
        _task_set(task_id, _tasks_cache[task_id])

        # Run HPO
        runner = YOLOTrainer(
            model=model,
            output_dir=Path("/home/wangxin/runs/hpo"),
        )

        if strategy == "bayesian":
            # Bayesian Optimization via scikit-optimize
            from src.hpo.bayesian_optimizer import BayesianHPOptimizer, HAS_SKOPT

            optimizer = BayesianHPOptimizer(n_trials=n_trials)
            _tasks_cache[task_id]["hpo_engine"] = "bayesian"
            _tasks_cache[task_id]["has_skopt"] = HAS_SKOPT
            _task_set(task_id, _tasks_cache[task_id])

            for trial_idx in range(n_trials):
                # Progress update
                _tasks_cache[task_id]["current_trial"] = trial_idx + 1
                _tasks_cache[task_id]["progress"] = ((trial_idx + 1) / n_trials) * 100.0
                _task_set(task_id, _tasks_cache[task_id])

                # Suggest next params
                params = optimizer.suggest()

                # Apply to config
                cfg = DEFAULT_TRAINING_CONFIG
                cfg.epochs = epochs_per_trial
                cfg.lr0 = params.get("lr0", cfg.lr0)
                cfg.lrf = params.get("lrf", cfg.lrf)
                cfg.momentum = params.get("momentum", cfg.momentum)
                cfg.weight_decay = params.get("weight_decay", cfg.weight_decay)
                cfg.box = params.get("box", cfg.box)
                cfg.cls = params.get("cls", cfg.cls)

                # Run trial
                trial_result = runner.train(
                    data_yaml=Path(data_yaml),
                    config=cfg,
                )

                # Extract mAP50 as score
                score = 0.0
                if trial_result.metrics:
                    score = trial_result.metrics.get("metrics/mAP50(B)", 0.0)
                    if score is None:
                        score = trial_result.metrics.get("mAP50", 0.0)

                # Report to optimizer
                optimizer.report(params, score)
                logging.info(
                    f"[{task_id}] Trial {trial_idx + 1}/{n_trials}: "
                    f"score={score:.4f}, params={params}"
                )

            best_params = optimizer.get_best()
            best_score = optimizer._best_score or 0.0
            result = TrainingResult(
                status="completed",
                best_params=best_params,
                metrics={"best_mAP50": best_score},
            )
        else:
            # Default: Ray Tune ASHA
            hpo_config = HPOConfig(
                n_trials=n_trials,
                epochs_per_trial=epochs_per_trial,
            )
            _tasks_cache[task_id]["hpo_engine"] = "ray_tune_asha"
            _task_set(task_id, _tasks_cache[task_id])

            result = runner.tune(
                data_yaml=Path(data_yaml),
                config=hpo_config,
            )

        if result.status == "completed":
            _tasks_cache[task_id]["status"] = "completed"
            _tasks_cache[task_id]["progress"] = 100.0
            _tasks_cache[task_id]["best_params"] = result.best_params or {}
            _tasks_cache[task_id]["metrics"] = result.metrics or {}
            logging.info(f"[{task_id}] HPO completed: best_params={result.best_params}")
        else:
            _tasks_cache[task_id]["status"] = "failed"
            _tasks_cache[task_id]["error"] = result.error or "Unknown error"
            logging.error(f"[{task_id}] HPO failed: {result.error}")

    except Exception as e:
        logging.error(f"[{task_id}] HPO exception: {e}", exc_info=True)
        _tasks_cache[task_id]["status"] = "failed"
        _tasks_cache[task_id]["error"] = str(e)
    finally:
        _tasks_cache[task_id]["completed_at"] = datetime.now().isoformat()
        _task_set(task_id, _tasks_cache[task_id])


@router.get("/gpu/status")
async def get_gpu_status(
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Get GPU resource status for multi-GPU training orchestration.
    Returns memory usage, utilization, and free memory per GPU.
    """
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return {
            "gpus": [],
            "total_slots": 0,
            "free_slots": 0,
            "error": "nvidia-smi unavailable",
        }

    gpus = []
    free_slots = 0
    for line in result.stdout.strip().split("\n"):
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 7:
            continue
        idx, name, mem_used, mem_total, mem_free, util, temp = parts[:7]
        util_int = int(util)
        if util_int < 10:
            free_slots += 1
        gpu_entry = {
            "index": int(idx),
            "name": name,
            "memory_used": int(mem_used),
            "memory_total": int(mem_total),
            "memory_free": int(mem_free),
            "utilization": util_int,
            "temperature": int(temp),
        }

        # Compute process info per GPU
        try:
            proc_result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                 "--format=csv,noheader,nounits", "-i", idx],
                capture_output=True, text=True, timeout=5,
            )
            processes = []
            for pline in proc_result.stdout.strip().split("\n"):
                if not pline.strip():
                    continue
                pparts = [x.strip() for x in pline.split(",")]
                if len(pparts) >= 3:
                    try:
                        processes.append({
                            "pid": pparts[0],
                            "name": pparts[1],
                            "memory_mb": int(pparts[2]),
                        })
                    except (ValueError, IndexError):
                        pass
            gpu_entry["processes"] = processes
        except Exception:
            gpu_entry["processes"] = []

        gpus.append(gpu_entry)

    return {
        "gpus": gpus,
        "total_slots": len(gpus),
        "free_slots": free_slots,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/hpo/start")
async def start_hpo(
    request: HPOStartRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Start an HPO job with Ray Tune.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    task_id = request.task_id
    _task_set(task_id, {
        "task_id": task_id,
        "type": "hpo",
        "status": "submitted",
        "model": request.model,
        "data_yaml": request.data_yaml,
        "n_trials": request.n_trials,
        "epochs_per_trial": request.epochs_per_trial,
        "strategy": request.strategy,
        "progress": 0.0,
        "created_at": datetime.now().isoformat()
    })

    # Start HPO in background thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        _run_hpo_sync,
        task_id,
        request.model,
        request.data_yaml,
        request.n_trials,
        request.epochs_per_trial,
        request.strategy,
    )

    return {
        "task_id": task_id,
        "status": "started",
        "strategy": request.strategy,
        "message": f"HPO task started ({request.n_trials} trials, {request.epochs_per_trial} epochs each, strategy={request.strategy})"
    }


@router.get("/hpo/status/{task_id}")
async def get_hpo_status(
    task_id: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """Get HPO job status."""
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    task = _task_get(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )

    return task


# ==================== Export Endpoints ====================

@router.post("/export/start")
async def start_export(
    request: ExportStartRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Start a model export job.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    task_id = request.task_id
    _task_set(task_id, {
        "task_id": task_id,
        "type": "export",
        "status": "submitted",
        "model_path": request.model_path,
        "platform": request.platform,
        "imgsz": request.imgsz,
        "formats": request.formats,
        "int8_quantize": request.int8_quantize,
        "progress": 0.0,
        "created_at": datetime.now().isoformat()
    })

    # Launch background export thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        _run_export_sync,
        task_id,
        request.model_path,
        request.platform,
        request.imgsz,
        request.formats,
        False,
        request.int8_quantize,
        None,  # calibration_data_dir (auto-derived if needed)
    )

    return {
        "task_id": task_id,
        "status": "started",
        "message": "Export task started"
    }


@router.get("/export/status/{task_id}")
async def get_export_status(
    task_id: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """Get export job status."""
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    task = _task_get(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )

    return task


# ==================== Benchmark Endpoints ====================

@router.post("/benchmark/run")
async def run_benchmark(
    request: BenchmarkRunRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Run a benchmark on an exported model file.

    Measures FPS, parameter count, FLOPs, and file size.
    GPU availability is auto-detected; FPS measurement on CPU is informational only.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    task_id = request.task_id
    _task_set(task_id, {
        "task_id": task_id,
        "type": "benchmark",
        "status": "submitted",
        "model_path": request.model_path,
        "format": request.format,
        "imgsz": request.imgsz,
        "warmup": request.warmup,
        "runs": request.runs,
        "progress": 0.0,
        "created_at": datetime.now().isoformat()
    })

    # Launch benchmark in background thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        _run_benchmark_sync,
        task_id,
        request.model_path,
        request.format,
        request.imgsz,
        request.warmup,
        request.runs,
    )

    return {
        "task_id": task_id,
        "status": "started",
        "message": f"Benchmark task started for {request.model_path}"
    }


@router.get("/benchmark/status/{task_id}")
async def get_benchmark_status(
    task_id: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """Get benchmark job status and results."""
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    task = _task_get(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    if task.get("type") != "benchmark":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task {task_id} is not a benchmark task"
        )

    return task


# ==================== Auto Label Endpoints ====================

class AutoLabelRequest(BaseModel):
    """Auto labeling request."""
    task_id: str
    input_folder: str
    classes: list[str]
    base_model: str = "grounded_sam"
    conf_threshold: float = 0.3


class AutoLabelResponse(BaseModel):
    """Auto labeling response."""
    task_id: str
    status: str
    message: str
    output_folder: Optional[str] = None
    data_yaml_path: Optional[str] = None


@router.post("/label/submit")
async def submit_labeling(
    request: AutoLabelRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Submit an auto-labeling job.

    Uses foundation models (GroundedSAM, etc.) to automatically
    label images in the input folder.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    task_id = request.task_id

    # Store task
    _task_set(task_id, {
        "task_id": task_id,
        "type": "labeling",
        "status": "submitted",
        "input_folder": request.input_folder,
        "classes": request.classes,
        "base_model": request.base_model,
        "progress": 0.0,
        "created_at": datetime.now().isoformat()
    })

    # Note: Actual labeling runs in background
    # For now, return task info

    return {
        "task_id": task_id,
        "status": "submitted",
        "message": f"Labeling job submitted. Base model: {request.base_model}",
        "input_folder": request.input_folder,
        "classes": request.classes
    }


@router.get("/label/status/{task_id}")
async def get_labeling_status(
    task_id: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """Get auto-labeling job status."""
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    task = _task_get(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    if task.get("type") != "labeling":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task {task_id} is not a labeling task"
        )

    return task


class DistillRequest(BaseModel):
    """Model distillation request."""
    task_id: str
    data_yaml: str
    target_model: str = "yolov8"
    model_size: str = "n"
    epochs: int = 100
    device: str = "cuda:0"
    distiller: str = Field("none", description="Distillation mode: none | mgd | feature | soft")
    loss_weight: float = Field(1.0, description="Weight of the distillation loss term")
    temperature: float = Field(4.0, description="Temperature for soft label distillation")
    output_dir: str = Field("/home/wangxin/runs/distill", description="Output directory for distillation run")


def _run_distill_sync(
    task_id: str,
    data_yaml: str,
    target_model: str,
    model_size: str,
    epochs: int,
    device: str,
    distiller: str,
    loss_weight: float,
    temperature: float,
    output_dir: str,
) -> None:
    """Run distillation training synchronously. Called from background thread."""
    from src.training.runner import TransferLearningTrainer

    try:
        logging.info(
            f"[{task_id}] Starting distillation: target={target_model}{model_size}, "
            f"data={data_yaml}, epochs={epochs}, distiller={distiller}, "
            f"loss_weight={loss_weight}, temperature={temperature}, device={device}"
        )

        _tasks_cache[task_id]["status"] = "running"
        _tasks_cache[task_id]["started_at"] = datetime.now().isoformat()
        _task_set(task_id, _tasks_cache[task_id])

        # Map target_model + size to ultralytics model name
        model_map = {
            "yolov8": f"yolov8{model_size}",
            "yolo11": f"yolo11{model_size}",
            "yolov11": f"yolov11{model_size}",
        }
        student_model = model_map.get(target_model.lower(), f"{target_model}{model_size}")

        # Teacher is the same family but larger (use 'm' as default teacher)
        teacher_map = {
            "yolov8": "yolov8m",
            "yolo11": "yolo11m",
            "yolov11": "yolov11m",
        }
        teacher_model = teacher_map.get(target_model.lower(), f"{target_model}m")

        runner = TransferLearningTrainer(teacher_model=teacher_model, freeze_layers=10)

        result = runner.train(
            data_yaml=Path(data_yaml),
            epochs=epochs,
            distiller=distiller,
            loss_weight=loss_weight,
            temperature=temperature,
            teacher_model_path=None,  # use default teacher
            output_dir=output_dir,
            device=device,
        )

        if result.status == "completed":
            _tasks_cache[task_id]["status"] = "completed"
            _tasks_cache[task_id]["progress"] = 100.0
            _tasks_cache[task_id]["model_path"] = str(result.model_path) if result.model_path else None
            _tasks_cache[task_id]["metrics"] = result.metrics or {}
            logging.info(f"[{task_id}] Distillation completed. Model: {result.model_path}")
        else:
            _tasks_cache[task_id]["status"] = "failed"
            _tasks_cache[task_id]["error"] = result.error or "Unknown error"
            logging.error(f"[{task_id}] Distillation failed: {result.error}")

    except Exception as e:
        logging.error(f"[{task_id}] Distillation exception: {e}", exc_info=True)
        _tasks_cache[task_id]["status"] = "failed"
        _tasks_cache[task_id]["error"] = str(e)
    finally:
        _tasks_cache[task_id]["completed_at"] = datetime.now().isoformat()
        _task_set(task_id, _tasks_cache[task_id])


@router.post("/train/distill")
async def start_distillation(
    request: DistillRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Start a model distillation job.

    Supports multiple knowledge distillation modes:
    - none:    standard transfer learning (frozen backbone)
    - mgd:     Minimal Generative Distillation (arXiv:2506.14440) — feature-level L2 loss
    - feature: intermediate feature-map alignment
    - soft:    temperature-scaled soft label distillation
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    task_id = request.task_id
    _task_set(task_id, {
        "task_id": task_id,
        "type": "distillation",
        "status": "submitted",
        "data_yaml": request.data_yaml,
        "target_model": request.target_model,
        "model_size": request.model_size,
        "epochs": request.epochs,
        "device": request.device,
        "distiller": request.distiller,
        "loss_weight": request.loss_weight,
        "temperature": request.temperature,
        "progress": 0.0,
        "created_at": datetime.now().isoformat()
    })

    # Launch distillation in background thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        _run_distill_sync,
        task_id,
        request.data_yaml,
        request.target_model,
        request.model_size,
        request.epochs,
        request.device,
        request.distiller,
        request.loss_weight,
        request.temperature,
        request.output_dir,
    )

    return {
        "task_id": task_id,
        "status": "started",
        "message": f"Distillation job started. Target: {request.target_model}{request.model_size}, mode: {request.distiller}",
        "data_yaml": request.data_yaml,
        "epochs": request.epochs,
        "distiller": request.distiller,
    }


# ==================== Model Registry Endpoints ====================
# Based on MLflow best practices:
# - Version everything: Use Git for code, DVC for data, MLflow for models
# - Use stages: Staging, Production, Archived
# - Enable rollbacks: Easy model version switching
# ============================================================

class ModelRegisterRequest(BaseModel):
    """Model registration request."""
    name: str = Field(..., description="Registered model name")
    version: int = Field(..., description="Model version")
    stage: str = Field("Staging", description="Target stage")
    description: str = Field("", description="Model description")


class ModelCreateRequest(BaseModel):
    """Create registered model request."""
    name: str = Field(..., description="Model name")
    description: str = Field("", description="Model description")
    tags: Optional[dict] = Field(default_factory=dict, description="Model tags")


class ModelStageTransitionRequest(BaseModel):
    """Model stage transition request."""
    name: str = Field(..., description="Registered model name")
    version: int = Field(..., description="Model version")
    stage: str = Field(..., description="Target stage")


@router.get("/models/registry")
async def list_registered_models(
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    List all registered models.

    Returns all models in the MLflow Model Registry.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.training.mlflow_tracker import list_registered_models
        models = list_registered_models()
        return {
            "models": [
                {
                    "name": m.name,
                    "description": m.description,
                    "latest_versions": len(m.latest_versions) if hasattr(m, 'latest_versions') else 0,
                    "created_at": m.creation_timestamp if hasattr(m, 'creation_timestamp') else None,
                }
                for m in models
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.post("/models/registry")
async def create_registered_model(
    request: ModelCreateRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Create a new registered model.

    Creates a new model entry in MLflow Model Registry.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.training.mlflow_tracker import create_registered_model as create_model
        model = create_model(
            name=request.name,
            description=request.description,
            tags=request.tags if request.tags else None
        )
        if model:
            return {
                "name": model.name,
                "description": model.description,
                "status": "created"
            }
        raise HTTPException(status_code=400, detail="Failed to create model")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")


@router.get("/models/registry/{name}")
async def get_model_info(
    name: str,
    http_request: Request,
    stage: Optional[str] = None,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Get model information and versions.

    Args:
        name: Registered model name
        stage: Optional stage filter (Staging, Production)
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.training.mlflow_tracker import get_latest_model_versions
        versions = get_latest_model_versions(name, stage)
        return {
            "name": name,
            "versions": [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "created_at": v.creation_timestamp if hasattr(v, 'creation_timestamp') else None,
                }
                for v in versions
            ] if versions else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.post("/models/registry/{name}/transition")
async def transition_model_stage(
    name: str,
    request: ModelStageTransitionRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Transition a model version to a different stage.

    Stages: Staging -> Production -> Archived
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.training.mlflow_tracker import transition_model_stage as transition
        result = transition(name, request.version, request.stage)
        if result:
            return {
                "name": name,
                "version": request.version,
                "stage": request.stage,
                "status": "success"
            }
        raise HTTPException(status_code=400, detail="Failed to transition stage")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to transition stage: {str(e)}")


@router.delete("/models/registry/{name}/version/{version}")
async def delete_model_version(
    name: str,
    version: int,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Delete a specific model version.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.training.mlflow_tracker import delete_model_version as delete
        success = delete(name, version)
        if success:
            return {"status": "deleted", "name": name, "version": version}
        raise HTTPException(status_code=400, detail="Failed to delete model version")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@router.delete("/models/registry/{name}")
async def delete_registered_model(
    name: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Delete a registered model and all its versions.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.training.mlflow_tracker import delete_registered_model as delete
        success = delete(name)
        if success:
            return {"status": "deleted", "name": name}
        raise HTTPException(status_code=400, detail="Failed to delete model")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


# ==================== Real-time Inference Endpoints ====================
# Based on ML system design patterns:
# - Real-time inference for immediate results
# - Model caching for performance
# - Configurable parameters
# ============================================================

class InferenceRequest(BaseModel):
    """Real-time inference request."""
    model_path: str = Field(..., description="Path to model weights")
    confidence: float = Field(0.25, description="Confidence threshold")
    iou_threshold: float = Field(0.45, description="IoU threshold for NMS")
    max_det: int = Field(300, description="Maximum detections")
    device: str = Field("cuda:0", description="Device to use")
    half: bool = Field(False, description="Use FP16 inference")
    tta: bool = Field(False, description="Enable test-time augmentation")
    tta_scales: Optional[List[float]] = Field(
        None,
        description="TTA scale factors, e.g. [0.83, 1.0, 1.17]"
    )
    tta_flips: Optional[List[int]] = Field(
        None,
        description="TTA flip modes: 0=none, 1=horizontal flip"
    )


class EnsembleRequest(BaseModel):
    """Model ensemble inference request."""
    model_paths: List[str] = Field(..., description="List of model paths for ensemble")
    weights: Optional[List[float]] = Field(None, description="Per-model weights, normalized automatically")
    source: Optional[str] = Field(None, description="Image path or URL")
    conf: float = Field(0.25, description="Confidence threshold")
    iou_threshold: float = Field(0.45, description="IoU threshold for NMS")
    max_det: int = Field(300, description="Maximum detections per model")
    device: str = Field("cuda:0", description="Device")


class InferenceResponse(BaseModel):
    """Inference response."""
    task_id: str
    status: str
    detections: List[dict]
    inference_time_ms: float
    model_name: str
    image_size: tuple
    timestamp: str


@router.post("/inference/predict", response_model=InferenceResponse)
async def predict(
    request: InferenceRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Run real-time inference on an image.

    Supports:
    - Image file upload (multipart/form-data)
    - Image URL
    - Base64 encoded image
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.inference.engine import get_inference_engine

        engine = get_inference_engine()

        result = engine.predict(
            model_path=request.model_path,
            source=request.model_path,  # caller passes path as source; endpoint expects path
            conf=request.confidence,
            iou=request.iou_threshold,
            max_det=request.max_det,
            device=request.device,
            half=request.half,
            tta=request.tta,
            tta_scales=request.tta_scales,
            tta_flips=request.tta_flips,
        )

        return InferenceResponse(
            task_id=result.task_id,
            status=result.status,
            detections=result.detections,
            inference_time_ms=result.inference_time_ms,
            model_name=result.model_name,
            image_size=result.image_size,
            timestamp=result.timestamp,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.post("/inference/predict/image")
async def predict_image(
    model_path: str,
    http_request: Request,
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 300,
    device: str = "cuda:0",
    half: bool = False,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Run inference on uploaded image.

    Upload an image file for real-time inference.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.inference.engine import get_inference_engine

        engine = get_inference_engine()

        return {
            "status": "ready",
            "message": "Image upload endpoint ready",
            "model_path": model_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


class EnsembleResponse(BaseModel):
    """Ensemble inference response."""
    task_id: str
    status: str
    detections: List[dict]
    total_boxes: int
    merged_boxes: int
    per_model_counts: Dict[str, int]
    inference_time_ms: float
    ensemble_weights: List[float]
    timestamp: str


@router.post("/inference/ensemble", response_model=EnsembleResponse)
async def ensemble_predict(
    request: EnsembleRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Run ensemble inference across multiple YOLO models.

    Combines predictions from multiple models using weighted NMS
    to improve detection recall and reduce false positives.
    """
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.inference.ensemble import ModelEnsemble

        ensemble = ModelEnsemble()
        result = ensemble.predict(
            model_paths=request.model_paths,
            source=request.source,
            weights=request.weights,
            conf=request.conf,
            iou_threshold=request.iou_threshold,
            max_det=request.max_det,
            device=request.device,
        )

        return EnsembleResponse(
            task_id=result.task_id,
            status=result.status,
            detections=result.detections,
            total_boxes=result.total_boxes,
            merged_boxes=result.merged_boxes,
            per_model_counts=result.per_model_counts,
            inference_time_ms=result.inference_time_ms,
            ensemble_weights=result.ensemble_weights,
            timestamp=result.timestamp,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ensemble inference failed: {str(e)}")


@router.get("/inference/stats")
async def get_inference_stats(
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Get inference statistics.

    Returns metrics about inference performance.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.inference.engine import get_inference_engine

        engine = get_inference_engine()
        stats = engine.get_stats()

        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/inference/cache/clear")
async def clear_inference_cache(
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Clear the model cache.

    Frees up memory by removing cached models.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.inference.engine import get_inference_engine

        engine = get_inference_engine()
        engine.clear_cache()

        return {"status": "cleared", "message": "Model cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


# ==================== Active Learning Endpoints ====================

@router.post("/active-learn/select")
async def select_active_learning_samples(
    request: ActiveLearnSelectRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """Select most uncertain samples for annotation using active learning."""
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    from src.training.active_learner import ActiveLearningPipeline, ActiveLearningConfig
    pipeline = ActiveLearningPipeline(ActiveLearningConfig(
        strategy=request.strategy,
        top_k=request.top_k,
    ))
    result = pipeline.select_uncertain_samples(
        model_path=request.model_path,
        image_dir=request.image_pool_dir,
    )
    return result


# ==================== Semi-Supervised Learning Endpoints ====================

@router.post("/train/semi-supervised/start")
async def start_semi_supervised(
    request: SemiSupervisedRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """Start semi-supervised training with pseudo-labeling."""
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    task_id = request.task_id

    _task_set(task_id, {
        "task_id": task_id,
        "type": "semi_supervised",
        "status": "submitted",
        "labeled_data_yaml": request.labeled_data_yaml,
        "unlabeled_image_dir": request.unlabeled_image_dir,
        "method": request.method,
        "confidence_threshold": request.confidence_threshold,
        "iterations": request.iterations,
        "epochs": request.epochs,
        "progress": 0.0,
        "created_at": datetime.now().isoformat()
    })

    # Run pseudo-labeling and training in background thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        _run_semi_supervised_sync,
        task_id,
        request.labeled_data_yaml,
        request.unlabeled_image_dir,
        request.method,
        request.confidence_threshold,
        request.iterations,
        request.epochs,
    )

    return {
        "task_id": task_id,
        "status": "started",
        "message": "Semi-supervised training task started"
    }


def _run_semi_supervised_sync(
    task_id: str,
    labeled_data_yaml: str,
    unlabeled_image_dir: str,
    method: str,
    confidence_threshold: float,
    iterations: int,
    epochs: int,
) -> None:
    """Run semi-supervised training synchronously. Called from background thread."""
    from src.training.semi_supervised import SemiSupervisedPipeline
    from src.training.runner import YOLOTrainer
    from src.training.config import DEFAULT_TRAINING_CONFIG
    from pathlib import Path

    try:
        logging.info(f"[{task_id}] Starting semi-supervised training")
        logging.info(f"[{task_id}] Labeled data: {labeled_data_yaml}, unlabeled dir: {unlabeled_image_dir}")
        logging.info(f"[{task_id}] Method: {method}, conf_thresh: {confidence_threshold}, iterations: {iterations}")

        _tasks_cache[task_id]["status"] = "running"
        _tasks_cache[task_id]["started_at"] = datetime.now().isoformat()
        _task_set(task_id, _tasks_cache[task_id])

        # Find all unlabeled images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = [
            str(p) for p in Path(unlabeled_image_dir).iterdir()
            if p.suffix.lower() in image_extensions
        ]

        if not image_paths:
            _tasks_cache[task_id]["status"] = "failed"
            _tasks_cache[task_id]["error"] = f"No images found in {unlabeled_image_dir}"
            _task_set(task_id, _tasks_cache[task_id])
            return

        logging.info(f"[{task_id}] Found {len(image_paths)} unlabeled images")

        # Phase 1: Train teacher model on labeled data (first iteration only)
        if iterations > 0:
            teacher_model_path = Path("/home/wangxin/runs/semi_supervised") / task_id / "teacher.pt"
            teacher_model_path.parent.mkdir(parents=True, exist_ok=True)

            if not teacher_model_path.exists():
                logging.info(f"[{task_id}] Training teacher model on labeled data")
                _tasks_cache[task_id]["phase"] = "train_teacher"
                _task_set(task_id, _tasks_cache[task_id])

                teacher_runner = YOLOTrainer(
                    model="yolo11m",
                    output_dir=teacher_model_path.parent,
                )
                config = DEFAULT_TRAINING_CONFIG
                config.epochs = min(epochs, 30)  # Shorter for teacher
                config.device = "cuda:0"

                teacher_result = teacher_runner.train(
                    data_yaml=Path(labeled_data_yaml),
                    config=config,
                )

                if teacher_result.model_path:
                    import shutil
                    teacher_model_path_fixed = teacher_result.model_path
                    shutil.copy(str(teacher_model_path_fixed), str(teacher_model_path))
                    logging.info(f"[{task_id}] Teacher model saved to {teacher_model_path}")
                else:
                    _tasks_cache[task_id]["status"] = "failed"
                    _tasks_cache[task_id]["error"] = "Teacher model training failed"
                    _task_set(task_id, _tasks_cache[task_id])
                    return

        # Phase 2: Generate pseudo-labels
        _tasks_cache[task_id]["phase"] = "pseudo_labeling"
        _task_set(task_id, _tasks_cache[task_id])

        ssl_pipeline = SemiSupervisedPipeline(
            confidence_threshold=confidence_threshold,
        )

        pseudo_labels = ssl_pipeline.generate_pseudo_labels(
            teacher_model_path=str(teacher_model_path),
            unlabeled_images=image_paths,
            method=method,
        )

        filtered_labels = ssl_pipeline.filter_pseudo_labels(pseudo_labels)
        logging.info(f"[{task_id}] Generated {len(filtered_labels)} filtered pseudo-labels from {len(pseudo_labels)} total")

        # Phase 3: Create pseudo dataset
        _tasks_cache[task_id]["phase"] = "create_dataset"
        _task_set(task_id, _tasks_cache[task_id])

        # Load class names from labeled data YAML
        import yaml
        class_names = []
        try:
            with open(labeled_data_yaml) as f:
                data_cfg = yaml.safe_load(f)
                class_names = [data_cfg.get("names", {})[str(i)] for i in range(data_cfg.get("nc", 0))]
        except Exception as e:
            logging.warning(f"[{task_id}] Could not read class names from {labeled_data_yaml}: {e}")
            class_names = ["class_0"]

        pseudo_output_dir = str(Path("/home/wangxin/runs/semi_supervised") / task_id / "pseudo_dataset")
        pseudo_yaml = ssl_pipeline.create_pseudo_dataset(
            pseudo_labels=filtered_labels,
            output_dir=pseudo_output_dir,
            class_names=class_names,
        )

        logging.info(f"[{task_id}] Pseudo dataset created at {pseudo_yaml}")

        # Phase 4: Self-training iterations
        _tasks_cache[task_id]["phase"] = "self_training"
        _task_set(task_id, _tasks_cache[task_id])

        current_model_path = str(teacher_model_path)

        for iteration in range(iterations):
            logging.info(f"[{task_id}] Self-training iteration {iteration + 1}/{iterations}")

            _tasks_cache[task_id]["current_iteration"] = iteration + 1
            _tasks_cache[task_id]["progress"] = ((iteration + 1) / iterations) * 100.0
            _task_set(task_id, _tasks_cache[task_id])

            # Train student on labeled + pseudo-labeled data
            combined_yaml = pseudo_yaml  # For now, train on pseudo data only
            student_runner = YOLOTrainer(
                model=current_model_path,
                output_dir=Path("/home/wangxin/runs/semi_supervised") / task_id / f"student_iter{iteration + 1}",
            )
            config = DEFAULT_TRAINING_CONFIG
            config.epochs = epochs
            config.device = "cuda:0"

            student_result = student_runner.train(
                data_yaml=Path(combined_yaml),
                config=config,
            )

            if student_result.model_path:
                current_model_path = str(student_result.model_path)
                logging.info(f"[{task_id}] Student model iteration {iteration + 1}: {current_model_path}")
            else:
                logging.warning(f"[{task_id}] Student training iteration {iteration + 1} failed, continuing with previous model")

        # Final result
        _tasks_cache[task_id]["status"] = "completed"
        _tasks_cache[task_id]["progress"] = 100.0
        _tasks_cache[task_id]["model_path"] = current_model_path
        _tasks_cache[task_id]["pseudo_labels_count"] = len(filtered_labels)
        _tasks_cache[task_id]["pseudo_dataset_yaml"] = pseudo_yaml
        logging.info(f"[{task_id}] Semi-supervised training completed. Final model: {current_model_path}")

    except Exception as e:
        logging.error(f"[{task_id}] Semi-supervised training exception: {e}", exc_info=True)
        _tasks_cache[task_id]["status"] = "failed"
        _tasks_cache[task_id]["error"] = str(e)
    finally:
        _tasks_cache[task_id]["completed_at"] = datetime.now().isoformat()
        _task_set(task_id, _tasks_cache[task_id])


@router.get("/semi-supervised/status/{task_id}")
async def get_semi_supervised_status(
    task_id: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """Get semi-supervised training job status."""
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    task = _task_get(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    if task.get("type") != "semi_supervised":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task {task_id} is not a semi-supervised task"
        )

    return task


# ==================== Dataset Quality Filtering ====================

class QualityFilterRequest(BaseModel):
    """Dataset quality filtering request."""
    dataset_path: str = Field(..., description="Path to dataset root or labels directory")
    filter_bbox_size: bool = Field(True, description="Apply bbox area filter")
    min_box_area: float = Field(0.001, description="Min box area as fraction of image (0.001 = 0.1%)")
    max_box_area: float = Field(0.95, description="Max box area as fraction of image (0.95 = 95%)")
    filter_aspect_ratio: bool = Field(True, description="Apply aspect ratio filter")
    min_aspect_ratio: float = Field(0.05, description="Min width/height ratio")
    filter_low_confidence: bool = Field(False, description="Apply low-confidence annotation filter")
    min_confidence: float = Field(0.3, description="Minimum confidence threshold")
    mine_hard_negatives: bool = Field(False, description="Run hard negative mining")
    hard_negative_model: Optional[str] = Field(None, description="Model path for hard negative mining")
    hard_negative_conf: float = Field(0.1, description="Confidence threshold for hard negative detection")
    hard_negative_image_dir: Optional[str] = Field(None, description="Image directory for hard negative mining")
    output_dir: Optional[str] = Field(None, description="Output directory for filtered labels (default: in-place)")


@router.post("/data/filter-quality")
async def filter_dataset_quality(
    request: QualityFilterRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Filter a dataset to remove low-quality annotations.

    Supports filtering by:
    - Bounding box area (too small or too large)
    - Aspect ratio (extreme ratios likely background)
    - Low-confidence annotations
    - Hard negative mining (images with many false positives)

    Returns a summary report with counts and quality scores.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        from src.data.quality_filter import DatasetQualityFilter, QualityReport
        from src.data.evaluator import DatasetEvaluator

        output_dir = request.output_dir or None
        filter_obj = DatasetQualityFilter(output_dir=output_dir)

        # Determine which label dir to use
        dataset_path = Path(request.dataset_path)

        # If dataset_path is a directory with train/val subdirs, find label dirs
        label_dirs = []
        for split in ["train", "val", "valid"]:
            for sub in ["labels", "label"]:
                ld = dataset_path / split / sub
                if ld.exists():
                    label_dirs.append(ld)

        if not label_dirs and dataset_path.exists() and any(dataset_path.glob("*.txt")):
            label_dirs.append(dataset_path)

        if not label_dirs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No label directories found under {request.dataset_path}"
            )

        # Run selected filters
        total_original = 0
        total_kept = 0
        total_removed = 0
        all_issues: List[str] = []

        for ld in label_dirs:
            original_count = 0
            kept_count = 0
            removed_count = 0
            issues: List[str] = []

            # Count original boxes
            for lbl_file in ld.glob("*.txt"):
                try:
                    lines = lbl_file.read_text().strip().splitlines()
                    original_count += len(lines) if lines else 0
                except Exception:
                    pass

            if request.filter_bbox_size:
                r = filter_obj.filter_bbox_size(
                    str(ld),
                    min_area=request.min_box_area,
                    max_area=request.max_box_area,
                )
                kept_count += r.kept_count
                removed_count += r.removed_count
                issues.extend(r.issues)

            if request.filter_aspect_ratio:
                r = filter_obj.filter_aspect_ratio(
                    str(ld),
                    min_ratio=request.min_aspect_ratio,
                )
                kept_count += r.kept_count
                removed_count += r.removed_count
                issues.extend(r.issues)

            if request.filter_low_confidence:
                r = filter_obj.filter_low_confidence(
                    str(ld),
                    min_confidence=request.min_confidence,
                )
                kept_count += r.kept_count
                removed_count += r.removed_count
                issues.extend(r.issues)

            total_original += original_count
            total_kept += kept_count
            total_removed += removed_count
            all_issues.extend(issues)

        quality_score = total_kept / total_original if total_original > 0 else 1.0

        # Hard negative mining
        hard_negatives: List[str] = []
        if request.mine_hard_negatives:
            model_path = request.hard_negative_model
            image_dir = request.hard_negative_image_dir or str(dataset_path / "images")
            if model_path:
                try:
                    hard_negatives = filter_obj.mine_hard_negatives(
                        model_path=model_path,
                        image_dir=image_dir,
                        conf_threshold=request.hard_negative_conf,
                    )
                except Exception as e:
                    logging.warning(f"[QualityFilter] Hard negative mining failed: {e}")
                    all_issues.append(f"Hard negative mining failed: {e}")

        return {
            "status": "ok",
            "original_count": total_original,
            "kept_count": total_kept,
            "removed_count": total_removed,
            "quality_score": round(quality_score, 4),
            "issues": all_issues[:50],
            "hard_negatives": hard_negatives[:100] if hard_negatives else [],
            "hard_negative_count": len(hard_negatives),
            "label_dirs_processed": [str(d) for d in label_dirs],
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[QualityFilter] Filter failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Quality filtering failed: {str(e)}")


# ==================== Dataset Evaluation Endpoint ====================

class DatasetEvaluateRequest(BaseModel):
    """Dataset evaluation request."""
    data_yaml: str = Field(
        "/home/wangxin/data/D-Fire/data/data.yaml",
        description="Path to data.yaml file"
    )


@router.post("/data/evaluate")
async def evaluate_dataset(
    request: DatasetEvaluateRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Evaluate dataset quality.

    Checks annotation quality, class balance, image quality, and label noise.
    Returns a quality score (0-100) and recommendations.
    """
    # Verify API key
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    try:
        evaluator = DatasetEvaluator(request.data_yaml)
        result = evaluator.to_dict()
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logging.error(f"[DatasetEvaluator] Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


# ==================== Drift Detection Endpoints ====================

class DriftCheckRequest(BaseModel):
    """Data drift check request."""
    model_name: str = Field(..., description="Model name being monitored")
    reference_image_dir: str = Field(..., description="Path to reference (training) image directory")
    current_image_dir: str = Field(..., description="Path to current production image directory")
    metrics_history: Optional[List[float]] = Field(
        None,
        description="Optional historical mAP values for concept drift detection"
    )
    psi_threshold: float = Field(0.2, description="PSI threshold for data drift (default 0.2)")


class DriftResponse(BaseModel):
    """Drift detection response."""
    model_name: str
    data_drift_score: float
    concept_drift_detected: bool
    recommendation: str
    feature_drift: Dict[str, float]
    timestamp: str


@router.post("/monitor/drift-check", response_model=DriftResponse)
async def check_drift(
    request: DriftCheckRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Detect data and concept drift for a deployed model.

    Compares statistical distributions between reference (training) images
    and current production images using Population Stability Index (PSI).

    PSI interpretation:
      < 0.1  : no drift  (stable)
      0.1-0.2: slight drift (monitor)
      > 0.2  : significant drift (retrain recommended)

    Concept drift is detected when recent rolling mAP average is
    significantly lower than the historical baseline.
    """
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    # Resolve image directories
    ref_dir = Path(request.reference_image_dir)
    cur_dir = Path(request.current_image_dir)

    if not ref_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference image directory not found: {ref_dir}"
        )
    if not cur_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Current image directory not found: {cur_dir}"
        )

    # Collect image paths
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ref_images = [str(p) for p in ref_dir.iterdir() if p.suffix.lower() in image_exts]
    cur_images = [str(p) for p in cur_dir.iterdir() if p.suffix.lower() in image_exts]

    if not ref_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No images found in reference directory: {ref_dir}"
        )
    if not cur_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No images found in current directory: {cur_dir}"
        )

    # Ensure monitoring module directory exists
    monitoring_dir = Path(__file__).parent.parent / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)

    from src.monitoring.drift_detector import DriftDetector

    detector = DriftDetector(psi_threshold=request.psi_threshold)

    report = detector.check_drift(
        model_name=request.model_name,
        reference_images=ref_images,
        current_images=cur_images,
        metrics_history=request.metrics_history,
    )

    return DriftResponse(
        model_name=request.model_name,
        data_drift_score=report.data_drift_score,
        concept_drift_detected=report.concept_drift_detected,
        recommendation=report.recommendation,
        feature_drift=report.feature_drift,
        timestamp=report.timestamp,
    )


# ==================== Edge Deployment Config Endpoints ====================

class EdgeConfigResponse(BaseModel):
    """Edge device inference configuration response."""
    device: str
    model_path: str
    batch_size: int
    stream_count: int
    workspace_mb: int
    recommended_format: str
    fallback_formats: List[str]
    precision: str
    dynamic_batch: bool
    imgsz: int
    export_kwargs: Dict[str, Any]
    notes: List[str]


@router.get("/deploy/edge-config/{model_name}", response_model=EdgeConfigResponse)
async def get_edge_config(
    model_name: str,
    device: str = "jetson_orin",
    imgsz: int = 640,
    http_request: Request = None,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """
    Generate optimal inference configuration for a target edge device.

    Returns device-specific runtime parameters including:
      - batch_size, stream_count, workspace_mb
      - recommended model format (engine-fp16, engine-int8, onnx, tflite)
      - export kwargs for the recommended format
      - Performance notes for the device

    Supported devices:
      jetson_orin, jetson_orin_nx, jetson_tx2, jetson_nano,
      rk3588, mobile, edge_tpu, generic
    """
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    from src.deployment.edge_config import EdgeProfileGenerator

    generator = EdgeProfileGenerator()
    config = generator.generate_config(
        device=device,
        model_path=model_name,
        imgsz=imgsz,
    )

    return EdgeConfigResponse(**config)


@router.get("/deploy/edge-devices")
async def list_edge_devices(
    http_request: Request = None,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    """List all supported edge device profiles."""
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    from src.deployment.edge_config import EdgeProfileGenerator

    generator = EdgeProfileGenerator()
    return {
        "devices": generator.list_devices(),
        "note": "Use GET /deploy/edge-config/{model_name}?device=<device> to get full config"
    }


# ==================== Continuous Training Pipeline Endpoints ====================

# Singleton pipeline instance shared across requests
_continous_pipeline_instance = None


def _get_continuous_pipeline():
    global _continous_pipeline_instance
    if _continous_pipeline_instance is None:
        from src.pipeline.continuous_training import ContinuousTrainingPipeline
        redis_client = get_redis_client()
        _continous_pipeline_instance = ContinuousTrainingPipeline(redis_client=redis_client)
    return _continous_pipeline_instance


class ContinuousTrainingRequest(BaseModel):
    model_name: str = Field("yolo11m", description="Base model to fine-tune")
    task_id: str = Field(..., description="Task identifier for the new training run")
    drift_threshold: float = Field(0.05, description="Fractional mAP drop that triggers retrain (0.0-1.0)")
    ab_test_duration_hours: int = Field(24, description="A/B test duration in hours")
    output_dir: str = Field("/home/wangxin/runs", description="Output directory for model artifacts")


class DriftCheckRequest(BaseModel):
    current_map: float = Field(..., description="Current production model mAP (0.0-1.0)")
    historical_avg: float = Field(..., description="Long-running average mAP (0.0-1.0)")
    threshold: Optional[float] = Field(None, description="Override drift threshold")


class ABTestStartRequest(BaseModel):
    model_a: str = Field(..., description="Production (control) model path or name")
    model_b: str = Field(..., description="Candidate model path or name")
    duration_hours: int = Field(24, description="Test duration in hours")
    min_samples: int = Field(100, description="Minimum inference samples before evaluating")


class RollbackRequest(BaseModel):
    current_model: Optional[str] = Field(None, description="Model to replace")
    previous_model: Optional[str] = Field(None, description="Model to restore")


@router.post("/pipeline/continuous/start")
async def start_continuous_training(
    request: ContinuousTrainingRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    pipeline = _get_continuous_pipeline()
    pipeline_id = pipeline.start_retrain_pipeline(task_id=request.task_id, model_name=request.model_name, output_dir=request.output_dir)
    return {"pipeline_id": pipeline_id, "task_id": request.task_id, "model_name": request.model_name, "status": "started", "message": f"Continuous training pipeline started for model {request.model_name}"}


@router.get("/pipeline/continuous/status")
async def get_continuous_training_status(
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return _get_continuous_pipeline().get_status()


@router.post("/pipeline/continuous/drift-check")
async def check_drift(
    request: DriftCheckRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    pipeline = _get_continuous_pipeline()
    decision = pipeline.check_drift_and_decide(current_map=request.current_map, historical_avg=request.historical_avg, threshold=request.threshold)
    return {"action": decision.action, "drift_score": decision.drift_score, "message": decision.message}


@router.post("/pipeline/continuous/ab-test")
async def start_ab_test(
    request: ABTestStartRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    pipeline = _get_continuous_pipeline()
    pipeline._transition_stage(pipeline.STAGE_AB_TESTING)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, pipeline.run_ab_test, request.model_a, request.model_b, request.duration_hours, request.min_samples)
    return {"status": "started", "model_a": request.model_a, "model_b": request.model_b, "duration_hours": request.duration_hours, "message": f"A/B test started: {request.model_a} vs {request.model_b}"}


@router.get("/pipeline/continuous/ab-test/result")
async def get_ab_test_result(
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    pipeline = _get_continuous_pipeline()
    results = pipeline._ab_test_results
    if not results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No A/B test results available")
    return results[-1].to_dict()


@router.post("/pipeline/continuous/rollback")
async def rollback_model(
    request: RollbackRequest,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    pipeline = _get_continuous_pipeline()
    success = pipeline.auto_rollback(current_model=request.current_model, previous_model=request.previous_model)
    if success:
        return {"status": "rolled_back", "message": f"Rolled back to {request.previous_model or 'production model'}"}
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Rollback failed")


@router.post("/pipeline/continuous/promote")
async def promote_candidate(
    candidate_model: str,
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    pipeline = _get_continuous_pipeline()
    pipeline.promote_candidate(candidate_model)
    return {"status": "promoted", "candidate_model": candidate_model, "message": f"Model {candidate_model} promoted to production"}


@router.post("/pipeline/continuous/reset")
async def reset_continuous_pipeline(
    http_request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
    _: None = Depends(check_rate_limit)
):
    if not verify_internal_api_key(x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    pipeline = _get_continuous_pipeline()
    pipeline.reset()
    return {"status": "reset", "message": "Continuous training pipeline reset to idle"}
