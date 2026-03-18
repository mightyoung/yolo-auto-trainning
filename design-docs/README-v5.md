# v8 设计文档汇总 (最终版)

## 已完成文档

| # | 文档 | 大小 | 描述 |
|---|------|------|------|
| 1 | `1-Overall-Design-v5.md` | 9KB | 整体架构 + 4个Agent |
| 2 | `2-DataDiscovery-Module-v5.md` | 20KB | 多源数据集发现 + 速率限制 + 缓存 |
| 3 | `3-ComfyUI-Workflow-Generator-v5.md` | 20KB | AI工作流生成 |
| 4 | `4-Training-Module-v5.md` | 22KB | YOLO11训练 + HPO (10参数) + MGD官方KD + 官方默认值 |
| 5 | `5-Deployment-Module-v5.md` | 22KB | ONNX + Jetson优化 + MLflow版本管理 |
| 6 | `6-APIService-Module-v5.md` | 16KB | FastAPI + Redis Lua 限流 + 优先级队列 + 死信队列 |
| 7 | `5-AgentOrchestration-Module-v5.md` | 18KB | CrewAI + 层级化编排 (Process.hierarchical) + 多 Crew 协作 |
| 8 | `7-Design-Review-v6.md` | 18KB | 深度审查报告 v6 |
| 9 | `8-Design-Review-v8.md` | 20KB | 深度审查报告 v8 |
| 10 | `9-Final-Review-v9.md` | 15KB | **新增** 最终审查报告 v9 |

---

## v8.0 关键改进

### 训练模块
- **文档一致性**: 架构图与代码同步
- **HPO 参数**: 8个参数 (lr0, lrf, momentum, weight_decay, box, cls, dfl, hsv增强)
- **知识蒸馏**: 使用 Ultralytics 官方 KDDetectionTrainer API
- **官方默认值**: 添加完整的 Ultralytics 默认超参数表

### 部署模块
- **MLflow 集成**: 标准化模型版本管理
- **DVC 数据版本**: 数据集版本控制

### API 服务
- **优先级队列**: 完整队列配置 (high/data/training/deployment/low)
- **死信队列**: 失败任务处理
- **Celery 优化**: 任务路由、优先级、确认机制

### Agent 编排
- **层级化流程**: CrewAI Process.hierarchical
- **管理器 Agent**: 专门的任务协调者
- **多 Crew 协作**: 并行执行支持

---

## v5.1 修复内容

### 修复的问题

| 模块 | 问题 | 修复 |
|------|------|------|
| 训练 | weight_decay 范围 [0.05, 0.3] 错误 | 改为 [0.0001, 0.001] |
| 训练 | Sanity Check 阈值 0.4 | 改为 0.3 + 更合理的建议 |
| 训练 | 知识蒸馏实现 | 改为伪标签方式 |
| 部署 | FP16 量化代码错误 | 修正 convert_float_to_float16 |
| 部署 | TensorRT 构建在 Nano 上 | 添加检查 + 文档说明 |
| 部署 | SSH 密码明文 | 改为环境变量 + Docker 部署 |
| 数据发现 | API 无速率限制 | 添加 RateLimiter |
| 数据发现 | 无缓存机制 | 添加 DatasetCache |

---

## v5 核心新增功能

### 1. 数据集发现 (Dataset Discovery)
- 多源搜索: Roboflow + Kaggle + HuggingFace
- 相关性评分
- 自动下载

### 2. ComfyUI 工作流自动生成
- 任务分类器
- 节点选择器
- LLM 工作流生成

### 3. 4 个 Agent 协作
- Dataset Discovery Agent (新增)
- Data Generator Agent
- Training Agent
- Deployment Agent

### 4. 训练模块增强
- Sanity Check (10 epochs, imgsz=640)
- HPO (Optuna, 25 trials, 10参数)
- 正式训练 (600-800 epochs, patience=50-100)
- MGD 知识蒸馏 (YOLO11m → YOLO11n)
- ONNX 导出 (FP16, opset 13)

### 5. 部署模块
- ONNX 优化 (算子融合)
- TensorRT 构建
- Jetson Nano SSH 部署
- 性能测试 (FPS >= 20)

### 6. API 服务
- FastAPI 网关
- Celery + Redis 任务队列
- API Key 认证
- 多维度限流
- Prometheus 监控

---

## 核心设计决策

| 决策 | 选择 | 依据 |
|------|------|------|
| YOLO 版本 | YOLO11 | YOLO_MODELS_GUIDE.md |
| 数据集策略 | 先搜索后生成 | 减少合成数据依赖 |
| Agent 规则 | 最多 2 条 | CrewAI 最佳实践 |
| 合成数据比例 | ≤ 30% | 保持数据质量 |
| 训练策略 | Sanity Check → HPO → 训练 | 避免资源浪费 |
| 部署优化 | FP16 + TensorRT | 边缘设备最佳性能 |

---

## 完整流水线

```
用户输入任务描述
     │
     ▼
┌────────────────┐
│  Discovery    │ ◄── 搜索 Roboflow/Kaggle/HuggingFace
│    Agent       │     评估相关性分数 > 0.8
└───────┬────────┘
        │ 发现数据集
        ▼
┌────────────────┐
│  Data Gen     │ ◄── ComfyUI 工作流生成
│    Agent       │     VLM 自动标注
└───────┬────────┘     CLIP 过滤 (相关性 > 0.25)
        │ 合成数据
        ▼
┌────────────────┐
│  Training     │ ◄── Sanity Check → HPO → 训练
│    Agent       │     知识蒸馏 (可选)
└───────┬────────┘     ONNX 导出
        │ 训练好的模型
        ▼
┌────────────────┐
│  Deployment   │ ◄── TensorRT 优化
│    Agent       │     SSH 部署到 Jetson
└───────┬────────┘     FPS 测试
        │
        ▼
    部署完成
```

---

## 文档版本

*版本: 8.0*
*最后更新: 2026-03-12*
