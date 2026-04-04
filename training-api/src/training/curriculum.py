"""
Progressive curriculum training for YOLO.
Location: training-api/src/training/curriculum.py

Contains: CurriculumStage, CurriculumConfig, PipelineCurriculumTrainer
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import json
import logging

from .training_utils import TrainingResult, TrainingCancelled
from .yolo_trainer import YOLOTrainer
from .config import (
    TrainingConfig,
    PlateauBreakingConfig,
    DEFAULT_TRAINING_CONFIG,
)


@dataclass
class CurriculumStage:
    """Single stage in the progressive training curriculum."""
    name: str
    epochs: int
    imgsz: int
    batch: int
    model: str
    augmentation_preset: str
    warmup_ratio: float = 0.05
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    num_gpus: int = 1
    resume_from: Optional[str] = None


@dataclass
class CurriculumConfig:
    """Progressive training curriculum configuration."""
    stage1: CurriculumStage = field(default_factory=lambda: CurriculumStage(
        name="rapid_validation", epochs=50, imgsz=640, batch=16, model="yolo11m",
        augmentation_preset="balanced", mosaic=1.0, mixup=0.1, copy_paste=0.1,
        degrees=0.0, translate=0.1, scale=0.5,
    ))
    stage2: CurriculumStage = field(default_factory=lambda: CurriculumStage(
        name="deep_training", epochs=150, imgsz=1280, batch=8, model="yolo11x",
        augmentation_preset="strong", mosaic=1.0, mixup=0.3, copy_paste=0.4,
        degrees=15.0, translate=0.2, scale=0.7,
    ))
    stage3: CurriculumStage = field(default_factory=lambda: CurriculumStage(
        name="fine_tuning", epochs=100, imgsz=1280, batch=8, model="yolo11x",
        augmentation_preset="strong", mosaic=0.0, mixup=0.1, copy_paste=0.1,
        degrees=5.0, translate=0.1, scale=0.5,
    ))
    stage1_min_map: float = 0.50
    stage2_target_map: float = 0.90
    stage2_min_for_stage3: float = 0.80


class PipelineCurriculumTrainer:
    """Progressive curriculum trainer for YOLO."""

    def __init__(self, output_dir: Path = None, target_mAP: float = 0.90):
        self.output_dir = Path(output_dir or "./runs/curriculum")
        self.target_mAP = target_mAP
        self._stage_history: list[dict] = []

    def _build_config(self, stage: CurriculumStage, resume_from: Optional[str] = None) -> TrainingConfig:
        """Build TrainingConfig for a given stage."""
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

        if resume_from:
            cfg.model = resume_from

        warmup_epochs = max(2, int(stage.epochs * getattr(stage, 'warmup_ratio', 0.05)))
        cfg.lr_scheduler.warmup_epochs = warmup_epochs
        cfg.warmup_epochs = warmup_epochs

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
        """Run a single curriculum stage with in-stage plateau recovery."""
        stage_output_dir = self.output_dir / f"stage{stage_num}_{stage.name}"
        trainer = YOLOTrainer(model=stage.model, output_dir=stage_output_dir)
        config = self._build_config(stage, resume_from)

        best_checkpoint: Optional[Path] = None
        best_checkpoint_map: Optional[float] = None
        if resume_from:
            best_checkpoint = Path(resume_from)

        logging.info(
            f"[CURRICULUM] Stage {stage_num} ({stage.name}): "
            f"model={stage.model}, epochs={stage.epochs}, imgsz={stage.imgsz}, "
            f"batch={stage.batch}, resume={resume_from or 'None'}"
        )

        _epoch_count = [0]

        while True:
            def epoch_callback(epoch: int, total: int, metrics: Dict[str, float]):
                nonlocal _epoch_count
                _epoch_count[0] += 1
                if metric_callback:
                    metric_callback(epoch, total, metrics)
                if plateau_manager:
                    decision = plateau_manager.on_metric(epoch, total, metrics)
                    if decision.triggered:
                        logging.warning(
                            f"[CURRICULUM][PLATEAU] Stage {stage_num} in-stage decision: "
                            f"level={decision.level}, action={decision.action}, "
                            f"avg_mAP50={decision.avg_recent_mAP50:.4f}"
                        )
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
                        pass

            result = trainer.train(
                data_yaml=data_yaml,
                config=config,
                progress_callback=progress_callback,
                metric_callback=epoch_callback,
            )

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

            if plateau_manager and plateau_manager._in_stage_restarts > 0:
                last_decision = plateau_manager._triggered_strategies[-1] if plateau_manager._triggered_strategies else {}
                action = last_decision.get("action", "")
                adjustment = last_decision.get("adjustment", {})

                if action == "lr_decay":
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
                    continue

                elif action == "augment_boost":
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
                    continue

                elif action == "data_expansion":
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
        """Run the full progressive curriculum."""
        config = config or CurriculumConfig()
        plateau_config = plateau_config or PlateauBreakingConfig()
        data_yaml = Path(data_yaml)
        best_model_path: Optional[Path] = None
        best_mAP50 = 0.0

        from .plateau_manager import PlateauManager
        pm = PlateauManager(task_id=f"{task_id}_stage", config=plateau_config)

        # Stage 1
        if stage_callback:
            stage_callback(1, "rapid_validation", 0.0, {"action": "starting"})
        pm._task_id = f"{task_id}_s1"
        s1_result, _ = self._run_stage(
            config.stage1, data_yaml, stage_num=1,
            progress_callback=progress_callback, metric_callback=metric_callback,
            plateau_manager=pm, redis_client=redis_client, task_id_for_redis=task_id,
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

        if s1_map < config.stage1_min_map:
            logging.error(
                f"[CURRICULUM] Stage 1 FAILED: mAP50={s1_map:.4f} < {config.stage1_min_map}. "
                f"Pipeline broken — check dataset quality, labels, and augmentation."
            )
            return s1_result

        best_model_path = Path(s1_result.model_path) if s1_result.model_path else None
        best_mAP50 = s1_map

        # Stage 2
        if stage_callback:
            stage_callback(2, "deep_training", s1_map, {"action": "proceeding", "resume": str(best_model_path)})
        pm = PlateauManager(task_id=f"{task_id}_s2", config=plateau_config)
        pm.set_current_lr(config.stage2.model == "yolo11x" and 0.01 or 0.01)
        pm.set_best_checkpoint_path(str(best_model_path) if best_model_path else "")
        s2_result, _ = self._run_stage(
            config.stage2, data_yaml, stage_num=2,
            resume_from=str(best_model_path),
            progress_callback=progress_callback, metric_callback=metric_callback,
            plateau_manager=pm, redis_client=redis_client, task_id_for_redis=task_id,
        )
        s2_map = s2_result.metrics.get("mAP50", 0.0) if s2_result.metrics else 0.0
        s2_status = s2_result.status
        self._stage_history.append({
            "stage": 2, "name": "deep_training",
            "mAP50": s2_map, "status": s2_status,
            "strategies_triggered": pm._triggered_strategies,
        })

        if s2_result.model_path and Path(s2_result.model_path).exists():
            if s2_map > best_mAP50:
                best_model_path = Path(s2_result.model_path)
                best_mAP50 = s2_map
        elif best_model_path and best_model_path.exists():
            pass
        else:
            best_model_path = Path(s2_result.model_path) if s2_result.model_path else best_model_path

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

        if s2_map >= config.stage2_min_for_stage3:
            if stage_callback:
                stage_callback(3, "fine_tuning", s2_map, {"action": "proceeding_to_stage3"})
            pm = PlateauManager(task_id=f"{task_id}_s3", config=plateau_config)
            pm.set_best_checkpoint_path(str(best_model_path) if best_model_path else "")
            s3_result, _ = self._run_stage(
                config.stage3, data_yaml, stage_num=3,
                resume_from=str(best_model_path),
                progress_callback=progress_callback, metric_callback=metric_callback,
                plateau_manager=pm, redis_client=redis_client, task_id_for_redis=task_id,
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
                metrics={"mAP50": best_mAP50, "stage_history": self._stage_history},
            )

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
