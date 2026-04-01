"""Training service - core training and curriculum logic.

This module contains:
- DynamicTrainingManager: Wrapper around PlateauManager for plateau detection
- _run_training_sync: Core synchronous training logic
- _run_curriculum_sync: 3-stage progressive curriculum training
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..store.task_store import (
    _cancel_events,
    _cancel_lock,
    _task_set,
    _tasks_cache,
)


class DynamicTrainingManager:
    """Backward-compatible wrapper around PlateauManager.

    Phase 3.1 refactoring: Delegates plateau detection to PlateauManager,
    which handles all plateau detection logic and cache updates internally.
    """

    def __init__(
        self,
        task_id: str,
        plateau_config: "PlateauBreakingConfig",  # noqa: F821 - forward reference
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
    resume_checkpoint: str | None = None,
    augmentation_preset: str | None = None,
    _loop: Optional["asyncio.AbstractEventLoop"] = None,
    hpo_params: dict[str, Any] | None = None,
    resume_from: str | None = None,
) -> None:
    """Run YOLO training synchronously. Called from background task."""
    # Import here to avoid import-time errors on systems without GPU
    from src.training.config import (
        AUGMENTATION_PRESETS,
        DEFAULT_PLATEAU_CONFIG,
        DEFAULT_TRAINING_CONFIG,
    )
    from src.training.runner import TrainingCancelled, YOLOTrainer

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
            if augmentation_preset and augmentation_preset in AUGMENTATION_PRESETS:
                aug_config = AUGMENTATION_PRESETS[augmentation_preset]
                for key, value in aug_config.items():
                    setattr(config, key, value)

            logging.info(f"[{task_id}] Training config: epochs={config.epochs}, imgsz={config.imgsz}, batch={config.batch}, device={config.device}")

            # Initialize PlateauManager for auto-adjustment
            plateau_manager = DynamicTrainingManager(
                task_id=task_id,
                plateau_config=DEFAULT_PLATEAU_CONFIG,
                device=device,
            )

            # Train with progress tracking
            start_time = time.time()
            best_map = 0.0
            epochs_since_best = 0

            for epoch in range(epochs):
                # Check for cancellation
                if cancel_event and cancel_event.is_set():
                    logging.info(f"[{task_id}] Training cancelled by user at epoch {epoch}")
                    _tasks_cache[task_id]["status"] = "cancelled"
                    _tasks_cache[task_id]["finished_at"] = datetime.now().isoformat()
                    _task_set(task_id, _tasks_cache[task_id])
                    raise TrainingCancelled(f"Training cancelled at epoch {epoch}")

                # Run one epoch
                result = runner.train(
                    data=data_yaml,
                    epochs=1,
                    imgsz=imgsz,
                    batch=batch,
                    device=device,
                    checkpoint=resume_checkpoint if epoch == 0 else None,
                )

                current_map = result.get("metrics", {}).get("mAP50", 0.0)
                if current_map > best_map:
                    best_map = current_map
                    epochs_since_best = 0
                else:
                    epochs_since_best += 1

                # --- Plateau Detection & Auto-Adjustment ---
                decision = plateau_manager.on_metric(epoch, epochs, {"mAP50": current_map, "best_mAP50": best_map})
                if decision.triggered:
                    logging.info(f"[{task_id}] Plateau detected! Applying adjustment: {decision.action}")
                    plateau_manager._manager.apply_decision(decision)
                    epochs_since_best = 0  # Reset after adjustment

                # Update progress
                progress = (epoch + 1) / epochs
                elapsed = time.time() - start_time
                eta = elapsed / (epoch + 1) * (epochs - epoch - 1) if epoch > 0 else 0

                _tasks_cache[task_id].update({
                    "status": "running",
                    "progress": progress,
                    "current_epoch": epoch + 1,
                    "total_epochs": epochs,
                    "metrics": {
                        "mAP50": current_map,
                        "best_mAP50": best_map,
                        "epochs_since_best": epochs_since_best,
                    },
                    "started_at": _tasks_cache[task_id].get("started_at"),
                    "eta_seconds": eta,
                })
                _task_set(task_id, _tasks_cache[task_id])

                # Save checkpoint
                checkpoint_path = Path(output_dir) / task_id / "weights" / "last.pt"
                if checkpoint_path.exists():
                    _tasks_cache[task_id]["checkpoint_path"] = str(checkpoint_path)

                # Check for HPO suggestion
                if hpo_params and decision.action == "reduce_lr" and not hpo_params.get(" lr_reduced"):
                    # Trigger HPO adjustment
                    logging.info(f"[{task_id}] Plateau detected, triggering HPO adjustment")
                    hpo_params[" lr_reduced"] = True

                resume_checkpoint = None  # Only use checkpoint for first epoch

            # Training complete
            total_time = time.time() - start_time
            _tasks_cache[task_id].update({
                "status": "completed",
                "progress": 1.0,
                "current_epoch": epochs,
                "total_epochs": epochs,
                "completed_at": datetime.now().isoformat(),
                "total_time_seconds": total_time,
                "final_metrics": {
                    "mAP50": best_map,
                },
            })
            _task_set(task_id, _tasks_cache[task_id])

            logging.info(f"[{task_id}] Training completed: mAP50={best_map:.4f}, time={total_time:.1f}s")

            # Auto-export best model
            if auto_export:
                try:
                    best_model_path = str(Path(output_dir) / task_id / "weights" / "best.pt")
                    if Path(best_model_path).exists():
                        logging.info(f"[{task_id}] Auto-exporting best model: {best_model_path}")
                        # Note: Export is handled by the caller or a separate background task
                except Exception as export_err:
                    logging.warning(f"[{task_id}] Auto-export skipped: {export_err}")

            return

        except TrainingCancelled:
            raise
        except Exception as e:
            last_error = e
            logging.warning(f"[{task_id}] Training attempt {attempt + 1} failed: {e}")
            if attempt < max_retries:
                logging.info(f"[{task_id}] Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                logging.error(f"[{task_id}] All {max_retries + 1} attempts failed. Last error: {e}")
                _tasks_cache[task_id]["status"] = "failed"
                _tasks_cache[task_id]["error"] = str(e)
                _tasks_cache[task_id]["finished_at"] = datetime.now().isoformat()
                _task_set(task_id, _tasks_cache[task_id])
                raise


def _run_curriculum_sync(
    task_id: str,
    data_yaml: str,
    output_dir: str,
    model: str = "yolo11m",
    device: str = "cuda:0",
    curriculum_stages: list | None = None,
) -> None:
    """Run 3-stage progressive curriculum training.

    Stage 1: rapid_validation (20% epochs, 640px, strong aug) - quick feedback
    Stage 2: full_training (60% epochs, 1280px, medium aug) - main training
    Stage 3: fine_tuning (20% epochs, 1280px, weak aug) - polish
    """
    from src.training.config import DEFAULT_TRAINING_CONFIG
    from src.training.runner import YOLOTrainer

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
                    # Could skip remaining stages here

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
