# Plateau Detection & Dynamic Training — Findings

## Root Cause: Open-Loop Architecture

The training system was **purely open-loop** — metric data flowed one way only:

```
ultralytics → YOLOTrainer → Training API → Redis → Business API → User
                ↑
            No feedback
```

**Three specific gaps identified:**

1. **`progress_callback` only receives `(epoch, total_epochs)`** — never sees actual mAP/loss values
2. **`_on_progress` in routes.py computes `progress` as `((epoch+1)/total)*100`** — purely epoch-based, no metric comparison
3. **HiTL orchestrator's `_poll_training` polls every 30s but only checks for terminal statuses** — never inspects metrics or triggers any action

## Changes Implemented

### 1. `training-api/src/training/runner.py`
- Added `metric_callback: Callable[[int, int, Dict[str, float]], None]` parameter to `train()`
- `_on_epoch_end` callback now extracts full metrics from `trainer.metrics` each epoch:
  - mAP50, mAP50-95, box_loss, cls_loss, dfl_loss, val_box_loss, val_cls_loss, val_dfl_loss
  - Calls both `progress_callback(epoch, total)` AND `metric_callback(epoch, total, metrics_dict)`

### 2. `training-api/src/training/config.py`
- Added `PlateauBreakingConfig` dataclass with:
  - Plateau detection: `window=10`, `min_improvement=0.002`, `min_epochs_before_trigger=30`
  - Level 1 (LR decay): `lr_reduction_factor=0.5`, `max_times=3`, `min_lr=1e-6`
  - Level 2 (Aug boost): `mixup=0.3`, `copy_paste=0.4`, `degrees=15`, `translate=0.2`, `scale=0.7`
  - Level 3 (Data expansion): `target_mAP=0.90`, `max_rounds=2`

### 3. `training-api/src/api/routes.py`
- Added `DynamicTrainingManager` class (~150 lines):
  - Maintains `mAP50` sliding window history
  - Detects plateau via window comparison (recent avg vs older avg)
  - Level 1: Writes `lr_decay_triggered` + signal to Redis cache
  - Level 2: Writes `augment_boost_active` + boost params to cache
  - Level 3: Writes `data_expansion_requested` + recommendation to cache
  - Updates `live_mAP50` in cache every epoch
- Updated `_run_training_sync` to create `DynamicTrainingManager` and pass `metric_callback`
- Extended `TrainStatusResponse` with plateau fields:
  - `live_mAP50`, `lr_decay_triggered`, `lr_decay_signal`
  - `augment_boost_active`, `augment_boost_signal`
  - `data_expansion_requested`, `data_expansion_signal`, `strategies_triggered`
- Status endpoint now returns all plateau fields

### 4. `business-api/src/agents/orchestration.py`
- Enhanced `_poll_training` to:
  - Read all plateau signals from status response
  - Print `[PLATEAU ALERT]` messages for ops team
  - Call `_trigger_dataset_search()` when `data_expansion_requested=True`
- Added `_trigger_dataset_search()`: Uses `DatasetDiscovery` to search HuggingFace for fire/smoke datasets, stores results in Redis
- Updated `run_phase3` to pass `augmentation_preset="strong"` and `resume_from` params to training

### 5. `business-api/src/api/routes.py`
- Extended `TrainStatusResponse` model with plateau fields
- Updated status endpoint to proxy plateau fields
- Added `AdjustRequest` model and `POST /train/adjust/{task_id}` endpoint:
  - Cancels current training
  - Restarts with adjusted `lr0`, `augmentation_preset`, `resume_from`, `additional_epochs`
  - Links new task to original task in Redis

### 6. `business-api/src/api/training_client.py`
- Added `augmentation_preset` and `resume_from` params to `start_training` (async) and `start_training_sync`

## Expected Behavior After Deployment

```
Training starts
    ↓
Each epoch: DynamicTrainingManager.on_metric() called with mAP50
    ↓
After EP 30: Plateau detection activates
    ↓
If mAP50 plateau for 10 consecutive epochs:
    ├─ Level 1: lr_decay_triggered=True (Business API alerts ops, can call /adjust)
    ├─ Level 2: augment_boost_active=True (auto-boost augmentation)
    └─ Level 3: data_expansion_requested=True (orchestrator searches HuggingFace)
```

## What's NOT Yet Implemented (Next Steps)

~~1. **Mid-training LR injection**~~ — DONE: AutoAdjustAgent cancels + restarts with halved lr0
~~2. **ActiveLearningPipeline integration**~~ — DONE: AutoAdjustAgent Level 3 runs full AL+SS pipeline
~~3. **Business API auto-adjust**~~ — DONE: AutoAdjustAgent spawns as background thread in _poll_training

## Additional Changes Implemented

### `training-api/src/training/config.py`
- **AugmentationPreset expanded** from 4 fields → 13 fields (adds degrees, translate, scale, shear, perspective, flipud, fliplr, hsv_h, hsv_s, hsv_v)
- **"strong" preset upgraded** with values optimized for 90%+ mAP:
  - mosaic=1.0, mixup=0.3, copy_paste=0.4
  - degrees=15.0, translate=0.2, scale=0.7, shear=2.0, perspective=0.0005
  - hsv_h=0.02, hsv_s=0.8, hsv_v=0.5

### `training-api/src/api/routes.py`
- `_run_training_sync` now applies ALL preset fields (not just mosaic/mixup/copy_paste)
- **NEW: `POST /train/curriculum/start`** endpoint for 3-stage progressive training
- **NEW: `_run_curriculum_sync`** background runner
- `_tasks_cache` extended with `curriculum_stage`, `curriculum_stage_mAP`, `curriculum_stage_history`

### `business-api/src/agents/orchestration.py`
- **AutoAdjustAgent class** (~280 lines): Background thread that monitors plateau and auto-adjusts
- **HiTL now uses 3-stage curriculum** instead of single-stage (see below)
- **AutoAdjustAgent spawned** in `_poll_training` right before polling loop starts
- **AutoAdjustAgent stopped** at all 4 exit points (completed/failed/timeout/crash)
- `_poll_training` extended with curriculum stage tracking

### `business-api/src/api/training_client.py`
- **NEW: `start_curriculum_sync()`** method for submitting curriculum training jobs

## AutoAdjustAgent Behavior

```
Training starts
    ↓
AutoAdjustAgent.start() → background thread spawned
    ↓
Every 60s: poll Training API status
    ↓
Level 1 (lr_decay triggered):
    → Cancel current task
    → Submit new task: halved lr0 + resume_from best.pt + extra 50 epochs
    → Record adjustment in Redis
Level 2 (aug_boost active):
    → Already handled by Training API (augment_boost_signal in cache)
Level 3 (data_expansion requested):
    → ActiveLearning: select top 200 uncertain samples from unlabeled dirs
    → SemiSupervised: generate pseudo-labels with YOLO teacher (conf=0.75)
    → Filter: keep 1-50 boxes per image
    → Create expanded dataset (symlinks + YOLO txt)
    → Submit new training task with expanded yaml + resume_from best.pt
```

## 3-Stage Progressive Curriculum (HiTL Default)

### Why NOT 200 epochs @ 1280px straight away

Running YOLO11x at 1280px for 200 epochs immediately wastes GPU resources:

| Config | GPU Hours (T4) | Risk if pipeline broken |
|--------|---------------|----------------------|
| yolo11m@640px × 200ep | ~52h | 52h wasted |
| **Progressive curriculum** | **~23h worst case** | Stage 1 cheap abort |

### Curriculum Stages

```
Stage 1 — Rapid Validation (50 epochs @ 640px, yolo11m, balanced aug)
  Cost: ~8 GPU-hours
  Purpose: Cheap pipeline validation — dataset quality, augmentation strategy
  Gate: mAP50 < 0.50 → ABORT (pipeline is broken)
        mAP50 >= 0.50 → proceed to Stage 2
  Expected: ~55-65% mAP50

Stage 2 — Deep Training (150 epochs @ 1280px, yolo11x, strong aug)
  Cost: ~15 GPU-hours
  Purpose: Main training with resolution and augmentation
  Gate: mAP50 >= 0.90 → GOAL REACHED, stop
        mAP50 >= 0.80 → proceed to Stage 3
        mAP50 < 0.80 → trigger AutoAdjustAgent (LR decay / data expansion)
  Expected: ~78-88% mAP50

Stage 3 — Fine-Tuning (100 epochs @ 1280px, yolo11x, LOW aug, mosaic=0)
  Cost: ~8 GPU-hours
  Purpose: Mosaic/copy-paste disabled → model learns fine-grained details
  Expected: +2-5% mAP50 over Stage 2
  Expected: ~82-92% mAP50

Total worst-case: ~300 epochs, ~31 GPU-hours on T4
```

### Decision Logic

| Condition | Action |
|-----------|--------|
| Stage1 mAP50 < 0.50 | ABORT — dataset/label issue |
| Stage2 mAP50 >= 0.90 | STOP — goal reached |
| Stage2 mAP50 >= 0.80 | Stage 3 fine-tuning |
| Stage2 mAP50 < 0.80 | AutoAdjustAgent (LR decay, data expansion) |
