"""Shared state and background sync functions for all route handlers.

This module is imported by all route sub-modules and provides:
- Task store imports (_tasks_cache, _task_set, etc.)
- Gateway utility imports (verify_internal_api_key, check_rate_limit)
- Module-level retry circuit breaker state
- DynamicTrainingManager class
- All _run_*_sync functions (background task runners)

These are kept in a shared module because:
1. They are tightly coupled to task_store state
2. They are invoked by route handlers via loop.run_in_executor()
3. They are not reusable business logic -- specific to this API's task model
"""

import os
import sys
import asyncio
import logging
import threading
import subprocess
import json
import uuid
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Header, status, BackgroundTasks, Depends, Request
from pydantic import BaseModel, Field

# Add training-api/src to sys.path so 'src' package resolves here (not legacy src/)
# This prevents legacy /home/wangxin/yolo-auto-training/src/ from shadowing training-api/src/
_training_api_src_root = Path(__file__).parent.parent  # = training-api/src/
if str(_training_api_src_root) not in sys.path:
    sys.path.insert(0, str(_training_api_src_root))

# Import verify_internal_api_key from gateway for timing-safe comparison
# NOTE: This import is safe even during gateway.py circular import because
# we only read function references, and Python returns the partially-initialized
# module without blocking.
from ..gateway import verify_internal_api_key, check_rate_limit, get_redis_client

# Import task store
from ..store.task_store import (
    _tasks_cache,
    _tasks_lock,
    _task_get,
    _task_set,
    _task_del,
    _cancel_events,
    _cancel_lock,
)

# Module-level retry circuit breaker: task_id -> retry count
_retry_counts: Dict[str, int] = {}


# ==================== Dynamic Training Manager ====================

class DynamicTrainingManager:
    """Backward-compatible wrapper around PlateauManager.

    Phase 3.1 refactoring: Delegates plateau detection to PlateauManager,
    which handles all plateau detection logic and cache updates internally.
    """

    def __init__(
        self,
        task_id: str,
        plateau_config: "PlateauBreakingConfig",
        device: str = "cuda:0",
    ):
        from src.training.plateau_manager import PlateauManager

        self.task_id = task_id
        self.cfg = plateau_config
        self.device = device

        # PlateauManager handles plateau detection and cache updates
        self._manager = PlateauManager(
            task_id=task_id,
            config=plateau_config,
        )

    def on_metric(self, epoch: int, total_epochs: int, metrics: dict[str, float]) -> None:
        """Called each epoch with current metrics."""
        decision = self._manager.on_metric(epoch, total_epochs, metrics)
        if decision.triggered:
            self._manager.apply_decision(decision)

    def get_status(self) -> dict:
        """Return current plateau detection status."""
        return self._manager.get_status()


# ==================== Training Sync Function ====================

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


# ==================== Export Sync Function ====================

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


# ==================== Benchmark Sync Function ====================

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


# ==================== HPO Sync Function ====================

def _run_hpo_sync(
    task_id: str,
    data_yaml: str,
    model: str,
    search_space: dict,
    epochs: int,
    imgsz: int,
    batch: int,
    output_dir: str,
    device: str,
    num_samples: int = 20,
    max_concurrent: int = 1,
    fixed_params: Optional[dict] = None,
) -> None:
    """Run HPO synchronously using Ray Tune."""
    from src.training.runner import YOLOTrainer
    from src.training.config import HPOConfig, SanityCheckConfig

    logging.info(f"[{task_id}] Starting HPO: model={model}, data={data_yaml}, samples={num_samples}, max_concurrent={max_concurrent}")

    _tasks_cache[task_id]["status"] = "running"
    _tasks_cache[task_id]["started_at"] = datetime.now().isoformat()
    _task_set(task_id, _tasks_cache[task_id])

    try:
        from ray import tune
        from src.training.hpo_ray import run_hpo_tuning

        # Run HPO
        best_result = run_hpo_tuning(
            task_id=task_id,
            data_yaml=data_yaml,
            model=model,
            search_space=search_space,
            num_samples=num_samples,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            output_dir=output_dir,
            device=device,
            max_concurrent=max_concurrent,
            fixed_params=fixed_params,
        )

        # Record best result
        _tasks_cache[task_id]["status"] = "completed"
        _tasks_cache[task_id]["progress"] = 100.0
        _tasks_cache[task_id]["best_trial_id"] = best_result.trial_id
        _tasks_cache[task_id]["best_metrics"] = best_result.metrics
        _tasks_cache[task_id]["best_checkpoint"] = best_result.checkpoint_path
        logging.info(f"[{task_id}] HPO completed: best_trial={best_result.trial_id}, mAP={best_result.metrics.get('mAP50', 0):.4f}")

    except ImportError as e:
        logging.error(f"[{task_id}] HPO requires Ray Tune: {e}")
        _tasks_cache[task_id]["status"] = "failed"
        _tasks_cache[task_id]["error"] = f"Ray Tune not available: {e}"
    except Exception as e:
        logging.error(f"[{task_id}] HPO exception: {e}", exc_info=True)
        _tasks_cache[task_id]["status"] = "failed"
        _tasks_cache[task_id]["error"] = str(e)
    finally:
        _tasks_cache[task_id]["completed_at"] = datetime.now().isoformat()
        _task_set(task_id, _tasks_cache[task_id])


# ==================== Curriculum Training Sync Function ====================

def _run_curriculum_sync(
    task_id: str,
    data_yaml: str,
    output_dir: str,
    model: str = "yolo11m",
    device: str = "cuda:0",
    curriculum_stages: Optional[list] = None,
) -> None:
    """Run 3-stage progressive curriculum training."""
    from src.training.runner import YOLOTrainer
    from src.training.config import DEFAULT_TRAINING_CONFIG

    if curriculum_stages is None:
        curriculum_stages = [
            {"name": "rapid_validation", "epochs_frac": 0.2, "imgsz": 640, "aug": "strong"},
            {"name": "full_training", "epochs_frac": 0.6, "imgsz": 1280, "aug": "medium"},
            {"name": "fine_tuning", "epochs_frac": 0.2, "imgsz": 1280, "aug": "weak"},
        ]

    total_epochs = _tasks_cache[task_id].get("submission", {}).get("epochs", 100)
    start_time = time.time()
    accumulated_time = 0.0

    for stage_idx, stage in enumerate(curriculum_stages):
        stage_name = stage["name"]
        epochs_frac = stage["epochs_frac"]
        stage_epochs = max(1, int(total_epochs * epochs_frac))
        stage_imgsz = stage["imgsz"]
        stage_aug = stage["aug"]

        logging.info(f"[{task_id}] Curriculum Stage {stage_idx + 1}/3: {stage_name}, epochs={stage_epochs}, imgsz={stage_imgsz}, aug={stage_aug}")

        # Update stage info in cache
        _tasks_cache[task_id].update({
            "curriculum_stage": stage_name,
            "curriculum_stage_idx": stage_idx + 1,
            "curriculum_stage_epochs": stage_epochs,
        })
        _task_set(task_id, _tasks_cache[task_id])

        # Check for cancellation
        with _cancel_lock:
            cancel_event = _cancel_events.get(task_id)
        if cancel_event and cancel_event.is_set():
            logging.info(f"[{task_id}] Curriculum training cancelled at stage {stage_name}")
            _tasks_cache[task_id]["status"] = "cancelled"
            _task_set(task_id, _tasks_cache[task_id])
            return

        # Create stage-specific output dir
        stage_output = Path(output_dir) / task_id / "curriculum" / stage_name

        try:
            runner = YOLOTrainer(model=model, output_dir=stage_output)
            config = DEFAULT_TRAINING_CONFIG
            config.epochs = stage_epochs
            config.imgsz = stage_imgsz
            config.batch = 16
            config.device = device

            # Apply augmentation preset
            from src.training.config import AUGMENTATION_PRESETS
            if stage_aug in AUGMENTATION_PRESETS:
                aug_config = AUGMENTATION_PRESETS[stage_aug]
                for key, value in aug_config.items():
                    setattr(config, key, value)

            stage_start = time.time()
            result = runner.train(
                data=data_yaml,
                epochs=stage_epochs,
                imgsz=stage_imgsz,
                batch=config.batch,
                device=device,
            )
            stage_time = time.time() - stage_start
            accumulated_time += stage_time

            # Update metrics
            stage_map = result.get("metrics", {}).get("mAP50", 0.0)
            logging.info(f"[{task_id}] Stage {stage_name} complete: mAP50={stage_map:.4f}, time={stage_time:.1f}s")

            _tasks_cache[task_id].update({
                "curriculum_stage_metrics": {
                    stage_name: {
                        "mAP50": stage_map,
                        "time_seconds": stage_time,
                        "epochs": stage_epochs,
                    }
                }
            })
            _task_set(task_id, _tasks_cache[task_id])

            # Check for early termination in validation stage
            if stage_name == "rapid_validation":
                if stage_map < 0.1:
                    logging.warning(f"[{task_id}] Rapid validation mAP50={stage_map:.4f} is very low — consider checking dataset")
                if stage_map > 0.5:
                    logging.info(f"[{task_id}] Rapid validation mAP50={stage_map:.4f} looks good! Skipping to fine-tuning...")

        except Exception as e:
            logging.error(f"[{task_id}] Curriculum stage {stage_name} failed: {e}")
            _tasks_cache[task_id]["status"] = "failed"
            _tasks_cache[task_id]["error"] = f"Stage {stage_name} failed: {e}"
            _task_set(task_id, _tasks_cache[task_id])
            raise

    # All stages complete
    total_time = time.time() - start_time
    final_metrics = _tasks_cache[task_id].get("curriculum_stage_metrics", {})

    _tasks_cache[task_id].update({
        "status": "completed",
        "progress": 1.0,
        "completed_at": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "final_metrics": final_metrics,
    })
    _task_set(task_id, _tasks_cache[task_id])

    logging.info(f"[{task_id}] Curriculum training complete! Total time: {total_time:.1f}s")


# ==================== Distillation Sync Function ====================

def _run_distill_sync(
    task_id: str,
    data_yaml: str,
    teacher_model: str,
    student_model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    output_dir: str,
    device: str,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> None:
    """Run knowledge distillation synchronously."""
    from src.training.runner import YOLOTrainer

    logging.info(f"[{task_id}] Starting distillation: teacher={teacher_model}, student={student_model}")

    _tasks_cache[task_id]["status"] = "running"
    _tasks_cache[task_id]["started_at"] = datetime.now().isoformat()
    _task_set(task_id, _tasks_cache[task_id])

    try:
        from src.training.distillation import DistillationTrainer

        runner = DistillationTrainer(
            teacher_model=teacher_model,
            student_model=student_model,
            output_dir=Path(output_dir),
        )

        result = runner.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            temperature=temperature,
            alpha=alpha,
        )

        _tasks_cache[task_id]["status"] = "completed"
        _tasks_cache[task_id]["progress"] = 100.0
        _tasks_cache[task_id]["metrics"] = result.metrics or {}
        _tasks_cache[task_id]["student_path"] = str(result.model_path)
        logging.info(f"[{task_id}] Distillation completed: mAP50={result.metrics.get('mAP50', 0):.4f}")

    except ImportError as e:
        logging.error(f"[{task_id}] Distillation requires src.training.distillation: {e}")
        _tasks_cache[task_id]["status"] = "failed"
        _tasks_cache[task_id]["error"] = f"Distillation not available: {e}"
    except Exception as e:
        logging.error(f"[{task_id}] Distillation exception: {e}", exc_info=True)
        _tasks_cache[task_id]["status"] = "failed"
        _tasks_cache[task_id]["error"] = str(e)
    finally:
        _tasks_cache[task_id]["completed_at"] = datetime.now().isoformat()
        _task_set(task_id, _tasks_cache[task_id])


# ==================== Semi-Supervised Sync Function ====================

def _run_semi_supervised_sync(
    task_id: str,
    labeled_data_yaml: str,
    unlabeled_data_yaml: str,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    output_dir: str,
    device: str,
    pseudo_label_threshold: float = 0.9,
    unsupervised_weight: float = 1.0,
) -> None:
    """Run semi-supervised training synchronously."""
    from src.training.runner import YOLOTrainer

    logging.info(f"[{task_id}] Starting semi-supervised training: model={model}")

    _tasks_cache[task_id]["status"] = "running"
    _tasks_cache[task_id]["started_at"] = datetime.now().isoformat()
    _task_set(task_id, _tasks_cache[task_id])

    try:
        from src.training.semi_supervised import SemiSupervisedTrainer

        runner = SemiSupervisedTrainer(
            model=model,
            output_dir=Path(output_dir),
        )

        result = runner.train(
            labeled_data=labeled_data_yaml,
            unlabeled_data=unlabeled_data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            pseudo_label_threshold=pseudo_label_threshold,
            unsupervised_weight=unsupervised_weight,
        )

        _tasks_cache[task_id]["status"] = "completed"
        _tasks_cache[task_id]["progress"] = 100.0
        _tasks_cache[task_id]["metrics"] = result.metrics or {}
        _tasks_cache[task_id]["model_path"] = str(result.model_path)
        logging.info(f"[{task_id}] Semi-supervised training completed: mAP50={result.metrics.get('mAP50', 0):.4f}")

    except ImportError as e:
        logging.error(f"[{task_id}] Semi-supervised requires src.training.semi_supervised: {e}")
        _tasks_cache[task_id]["status"] = "failed"
        _tasks_cache[task_id]["error"] = f"Semi-supervised not available: {e}"
    except Exception as e:
        logging.error(f"[{task_id}] Semi-supervised exception: {e}", exc_info=True)
        _tasks_cache[task_id]["status"] = "failed"
        _tasks_cache[task_id]["error"] = str(e)
    finally:
        _tasks_cache[task_id]["completed_at"] = datetime.now().isoformat()
        _task_set(task_id, _tasks_cache[task_id])
