# 自主训练系统实现计划

## 目标
全部训练任务由 Business API + Training API 自主完成，用户只负责派发训练任务、监控和分析问题。

## 当前状态

### 已实现能力
- ✅ 训练提交 (`start_training` / `start_curriculum_training`)
- ✅ 状态轮询 (`_poll_training` 每30s)
- ✅ AutoAdjustAgent L1 (LR decay)
- ✅ AutoAdjustAgent L2 (增强 boost, 被动日志)
- ✅ 普通训练自动导出 (`_run_training_sync` → `run_in_executor(_run_export_sync)`)
- ✅ Curriculum 3阶段渐进训练 (`PipelineCurriculumTrainer`)
- ✅ GPU 显存/利用率读取

### 缺失能力（按优先级排序）

| 优先级 | 能力 | 严重度 | 文件 |
|--------|------|--------|------|
| P0 | Curriculum 自动导出 | 高 | training-api/src/api/routes.py |
| P0 | `TrainingCancelled` 导入修复 | 高 | training-api/src/api/routes.py |
| P1 | GPU 任务队列 + 自主调度 | 高 | business-api/src/agents/ |
| P1 | 训练完成 → 导出 → 部署链 | 高 | business-api + training-api |
| P2 | 失败自动重试 | 中 | training-api/src/api/routes.py |
| P2 | AutoAdjustAgent L3 激活 | 中 | business-api/src/agents/ |
| P3 | 事件驱动 Webhook | 低 | training-api + business-api |
| P3 | GPU 资源仪表盘 API | 低 | training-api/src/api/routes.py |

## 实现详情

### P0.1: 修复 TrainingCancelled 导入

**问题**: `routes.py:456` 导入 `TrainingCancelled` 失败，`src/training/runner.py` 未导出此类。

**修复**: 检查 `training-api/src/training/runner.py` 是否导出 `TrainingCancelled`，如无则添加。

### P0.2: Curriculum 自动导出

**问题**: `_run_curriculum_sync()` 完成后从不调用导出，训练完成但模型未导出。

**修复**: 在 `_run_curriculum_sync` 的 `CURRICULUM_COMPLETE` 路径添加导出调用：

```python
# _run_curriculum_sync 约 line 1013 后
# 找到 best.pt 路径
best_weights = os.path.join(output_dir, "weights", "best.pt")
# 异步触发导出
loop = asyncio.get_event_loop()
loop.run_in_executor(
    None, _run_export_sync,
    task_id + "_export",
    best_weights,
    ["onnx"],
    "jetson_orin",
    640,
    None, None, None,
)
```

### P1.1: GPU 任务队列

**新增模块**: `business-api/src/agents/gpu_scheduler.py`

**功能**:
- 维护 Redis 队列 `training:queue` (List, 存 task metadata JSON)
- 轮询 Training API 的 `/api/v1/internal/gpu/status` 每 60s
- 检测空闲 GPU 槽位 + 队列非空时，自动提交下一任务
- 任务状态变化时自动出队

**数据结构**:
```python
# Redis: training:queue
# [
#   {"task_id": "q_xxx", "type": "curriculum", "data_yaml": "...", "model": "yolo11m", ...},
#   {"task_id": "q_yyy", "type": "curriculum", "data_yaml": "...", "model": "yolo11n", ...},
# ]
```

**GPU 槽位**: Training API 新增 `/api/v1/internal/gpu/status` 返回:
```json
{
  "gpus": [
    {"index": 0, "name": "Tesla T4", "memory_used": 3827, "memory_total": 15109, "utilization": 0, "tasks": []},
    {"index": 1, "name": "Tesla T4", "memory_used": 0, "memory_total": 16127, "utilization": 0, "tasks": []},
    {"index": 2, "name": "Tesla T4", "memory_used": 6946, "memory_total": 15109, "utilization": 91, "tasks": ["dfire_resume_gpu2"]},
  ],
  "total_slots": 3,
  "free_slots": 1
}
```

### P1.2: 训练完成 → 导出 → 部署链

**修改**: `_poll_training()` 在 `training_completed` 后自动触发：

1. 导出: POST `/api/v1/internal/export`
2. 导出完成后: POST `/api/v1/deploy/edge` (生成 edge 配置)
3. 部署完成后: 写入 `model:deployed` 状态

**需修改**:
- `business-api/src/agents/orchestration.py`: `_poll_training()` 的 `completed` 分支
- 新增 `_auto_export_and_deploy()` 方法

### P2.1: 失败自动重试

**修改**: `_run_training_sync()` 开头增加重试逻辑：

```python
def _run_training_sync(...):
    max_retries = 2
    retry_delay = 180  # 3 minutes
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            # ... 正常训练逻辑 ...
            return
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logging.warning(f"[{task_id}] Training failed (attempt {attempt+1}), retrying in {retry_delay}s: {e}")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                _task_set(task_id, {"status": "failed", "error": str(last_error)})
```

### P2.2: AutoAdjustAgent Level 3 激活

**现状**: `AutoAdjustAgent` 有 L3 代码框架，但 `ActiveLearningPipeline` 和 `SemiSupervisedPipeline` 是 stub 文件。

**需要**: 实现这两个 pipeline 的核心方法：
- `ActiveLearningPipeline.select_uncertain_samples()` → 选出高不确定度样本
- `SemiSupervisedPipeline.generate_pseudo_labels()` → 生成伪标签
- `SemiSupervisedPipeline.create_expanded_yaml()` → 生成扩充数据集 yaml

### P3.1: 事件驱动 Webhook

**Training API → Business API**: 训练状态变化时 POST callback:
```python
# 在 _run_training_sync 完成后
httpx.post(f"{BUSINESS_API_URL}/api/v1/callback/training-event",
    json={"task_id": task_id, "event": "completed", "model_path": model_path})
```

**需修改**: Business API 新增 `/api/v1/callback/training-event` 端点

### P3.2: GPU 资源仪表盘

**新增端点**: `GET /api/v1/internal/gpu/status` (Training API)

**返回**: 所有 GPU 的实时状态 + 当前运行任务列表

## 执行顺序

```
Phase 1 (阻断性修复):
  1. P0.1 修复 TrainingCancelled 导入
  2. P0.2 Curriculum 自动导出
  → 验证: curriculum 任务完成后模型自动导出

Phase 2 (自主调度核心):
  3. P1.2 训练→导出→部署链
  4. P1.1 GPU 任务队列
  → 验证: 多任务自动排队，空闲 GPU 自动调度

Phase 3 (弹性恢复):
  5. P2.1 失败自动重试
  6. P2.2 AutoAdjustAgent L3
  → 验证: 训练失败自动重试，平台期自动数据扩充

Phase 4 (智能优化):
  7. P3.1 Webhook 事件驱动
  8. P3.2 GPU 仪表盘
  → 验证: 事件驱动响应，实时 GPU 监控
```

## 风险与依赖

- **依赖**: GPU 服务器上 `/api/v1/internal/gpu/status` 需要新增端点
- **风险**: 多任务并发时 GPU 槽位判断需准确 (CUDA_VISIBLE_DEVICES 设置)
- **依赖**: Redis 队列需保证原子性 (LPUSH/BRPOP)
