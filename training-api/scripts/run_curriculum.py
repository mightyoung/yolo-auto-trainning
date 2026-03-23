#!/usr/bin/env python3
"""Standalone curriculum training script - runs as a separate process.

Enhanced with:
- PlateauManager: in-stage plateau detection and automatic recovery (LR decay, augment boost)
- PlateauAdvisor: LLM-powered diagnosis when training cannot recover automatically
- Transient GPU error retry: automatic retry on CUDA OOM, NCCL timeout, etc.
"""
import sys, os, json, logging
from pathlib import Path
from typing import Dict

# Setup path
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent  # .../training-api
sys.path.insert(0, str(PROJECT_DIR))
os.chdir(str(PROJECT_DIR))

# Load environment
env_file = PROJECT_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() != "CUDA_VISIBLE_DEVICES":
                os.environ[k.strip()] = v.strip()

# Override CUDA device - ONLY set if not already inherited
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_curriculum.py <task_id>")
        sys.exit(1)
    
    task_id = sys.argv[1]
    logger.info(f"[{task_id}] Starting curriculum training as subprocess (PID={os.getpid()})")
    
    # Load cached task params
    cache_file = f"/tmp/curriculum_{task_id}.json"
    if not Path(cache_file).exists():
        logger.error(f"[{task_id}] Cache file not found: {cache_file}")
        sys.exit(1)
    
    with open(cache_file) as f:
        params = json.load(f)
    
    logger.info(f"[{task_id}] Params: data_yaml={params.get('data_yaml')}, model={params.get('model')}")
    
    # Import and run
    from pathlib import Path as P
    from src.training.runner import PipelineCurriculumTrainer, CurriculumStage, CurriculumConfig
    
    output_dir = params.get("output_dir", "/home/wangxin/runs")
    trainer = PipelineCurriculumTrainer(output_dir=P(output_dir) / task_id)
    
    def make_stage(prefix, name):
        return CurriculumStage(
            name=name,
            epochs=params.get(f"{prefix}_epochs", 50),
            imgsz=params.get(f"{prefix}_imgsz", 640),
            batch=params.get(f"{prefix}_batch", 16),
            model=params.get("model", "yolo11m"),
            augmentation_preset="balanced",
            num_gpus=params.get(f"{prefix}_num_gpus", 1),
            warmup_ratio=params.get(f"{prefix}_warmup_ratio", 0.05),
            mosaic=params.get(f"{prefix}_mosaic", 1.0),
            mixup=params.get(f"{prefix}_mixup", 0.1),
            copy_paste=params.get(f"{prefix}_copy_paste", 0.1),
            degrees=params.get(f"{prefix}_degrees", 0.0),
            translate=params.get(f"{prefix}_translate", 0.1),
            scale=params.get(f"{prefix}_scale", 0.5),
        )
    
    cfg = CurriculumConfig(
        stage1=make_stage("stage1", "rapid_validation"),
        stage2=make_stage("stage2", "deep_training"),
        stage3=make_stage("stage3", "fine_tuning"),
        stage1_min_map=params.get("stage1_min_map", 0.05),
        stage2_target_map=params.get("stage2_target_map", 0.70),
        stage2_min_for_stage3=params.get("stage2_min_for_stage3", 0.80),
    )
    
    # Setup Redis for progress updates
    import redis as redis_lib
    redis_client = redis_lib.Redis(
        host="192.168.11.134", port=6379, db=0, decode_responses=True,
        password="123456"
    )
    _progress_count = [0]

    # --- PlateauAdvisor: LLM diagnosis when plateau is detected ---
    def diagnose_and_log_plateau(stage_num: int, mAP50: float, strategies: list):
        """Call LLM advisor on plateau, write result to Redis."""
        try:
            from src.training.plateau_advisor import PlateauAdvisor
            advisor = PlateauAdvisor()
            if not advisor.enabled:
                logger.info(f"[{task_id}] PlateauAdvisor disabled (no DEEPSEEK_API_KEY)")
                return

            logger.info(f"[{task_id}] Calling LLM PlateauAdvisor for Stage {stage_num} diagnosis...")
            diagnosis = advisor.diagnose(
                mAP50_history=[],
                dataset_info={
                    "train_images": params.get("train_images", 0),
                    "val_images": params.get("val_images", 0),
                    "num_classes": params.get("num_classes", 2),
                },
                augmentation_params={
                    "mixup": params.get("stage2_mixup", 0.3),
                    "copy_paste": params.get("stage2_copy_paste", 0.4),
                },
                current_config={
                    "lr0": params.get("lr0", 0.01),
                    "epochs": params.get("stage2_epochs", 150),
                    "imgsz": params.get("stage2_imgsz", 1280),
                },
                target_mAP=cfg.stage2_target_map,
            )
            diag_dict = diagnosis.to_dict()
            logger.info(
                f"[{task_id}] LLM Diagnosis: {diagnosis.diagnosis} "
                f"(confidence={diagnosis.confidence:.2f}), "
                f"reasoning: {diagnosis.reasoning[:100]}"
            )
            if diagnosis.recommendations:
                for rec in diagnosis.recommendations[:3]:
                    logger.info(f"  → {rec.get('action')}: {rec.get('params')}")

            # Write to Redis
            redis_client.hset(f"training:task:{task_id}", mapping={
                "llm_diagnosis": json.dumps(diag_dict),
                "llm_diagnosis_str": f"{diagnosis.diagnosis}: {diagnosis.reasoning[:200]}",
            })
        except Exception as e:
            logger.warning(f"[{task_id}] PlateauAdvisor call failed: {e}")

    def on_progress(epoch, total):
        _progress_count[0] += 1
        if _progress_count[0] % 5 == 0:
            try:
                redis_client.hset(f"training:task:{task_id}", mapping={
                    "current_epoch": epoch,
                    "total_epochs": total,
                    "progress": min(epoch / total * 33, 32),
                })
            except Exception as e:
                logger.warning(f"[{task_id}] Redis update failed: {e}")

    def on_stage(sn, name, mAP, decision):
        logger.info(f"[{task_id}] Stage {sn} ({name}) started, mAP={mAP:.4f}")
        try:
            redis_client.hset(f"training:task:{task_id}", mapping={
                "curriculum_stage": name,
                "curriculum_stage_num": sn,
                "status": "running",
            })
        except Exception as e:
            logger.warning(f"[{task_id}] Redis update failed: {e}")

    # --- PlateauBreakingConfig for in-stage recovery ---
    from src.training.config import PlateauBreakingConfig
    plateau_cfg = PlateauBreakingConfig(
        enabled=params.get("plateau_detection_enabled", True),
        window=params.get("plateau_window", 10),
        min_improvement=params.get("plateau_min_improvement", 0.002),
        min_epochs_before_trigger=params.get("plateau_min_epochs", 30),
        lr_reduction_factor=params.get("lr_reduction_factor", 0.5),
        lr_reduction_max_times=params.get("lr_reduction_max_times", 3),
        augmentation_boost_epochs=params.get("augment_boost_epochs", 30),
        boosted_mixup=params.get("boosted_mixup", 0.4),
        boosted_copy_paste=params.get("boosted_copy_paste", 0.5),
        auto_expand_data=params.get("auto_expand_data", True),
        expansion_target_map=cfg.stage2_target_map,
        max_expansion_rounds=params.get("max_expansion_rounds", 2),
    )

    # PlateauManager is instantiated internally by PipelineCurriculumTrainer.
    # Redis writing is handled inside _run_stage() via the plateau_manager callback.

    result = trainer.train(
        data_yaml=P(params["data_yaml"]),
        config=cfg,
        progress_callback=on_progress,
        stage_callback=on_stage,
        task_id=task_id,
        plateau_config=plateau_cfg,
        redis_client=redis_client,
    )

    # If plateau, call LLM advisor
    if result.status == "plateau":
        strategies = result.metrics.get("strategies_triggered", []) if result.metrics else []
        diagnose_and_log_plateau(
            stage_num=2,
            mAP50=result.metrics.get("mAP50", 0) if result.metrics else 0,
            strategies=strategies,
        )
    
    # Update final status
    try:
        mapping = {
            "status": result.status,
            "progress": 100.0,
            "mAP50": result.metrics.get("mAP50", 0) if result.metrics else 0,
        }
        if result.metrics:
            mapping["strategies_triggered"] = json.dumps(
                result.metrics.get("strategies_triggered", [])
            )
            mapping["in_stage_restarts"] = str(
                result.metrics.get("in_stage_restarts", 0)
            )
        redis_client.hset(f"training:task:{task_id}", mapping=mapping)
    except Exception as e:
        logger.warning(f"[{task_id}] Final Redis update failed: {e}")

    logger.info(f"[{task_id}] Curriculum complete: status={result.status}, mAP50={result.metrics}")
    print(json.dumps({"status": result.status, "metrics": result.metrics}))

if __name__ == "__main__":
    main()
