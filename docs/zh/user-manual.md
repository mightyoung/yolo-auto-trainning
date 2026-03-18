# YOLO Auto-Training System - 用户手册

**版本**: 1.0
**日期**: 2026-03-14

---

## 1. 系统概述

YOLO Auto-Training System 是一个基于 CrewAI 构建的 AI 驱动自动化 YOLO 模型训练与部署平台。提供从数据集发现到边缘设备部署的端到端工作流程。

### 1.1 核心功能

| 功能 | 描述 |
|------|------|
| **数据集发现** | 从 Roboflow、Kaggle、HuggingFace 搜索数据集 |
| **自动训练** | YOLO11 训练 + Ray Tune 超参数优化 |
| **知识蒸馏** | 从大模型训练小模型 |
| **边缘部署** | 导出为 ONNX/TensorRT 用于 Jetson、RK3588 设备 |
| **多Agent编排** | CrewAI 驱动的智能工作流程 |

### 1.2 支持的模型

| 模型 | 参数量 | 速度 (FPS) | 使用场景 |
|------|--------|-------------|----------|
| YOLO11n | 2.6M | 100+ | 边缘设备 |
| YOLO11s | 9.7M | 60+ | 移动端 |
| YOLO11m | 25.9M | 40+ | 服务器 |
| YOLO11l | 51.5M | 25+ | 高精度 |
| YOLO11x | 97.2M | 15+ | 最高精度 |

---

## 2. 快速开始

### 2.1 安装

```bash
# 克隆并安装
cd yolo-auto-training
pip install -r requirements.txt

# 或作为包安装
pip install -e .
```

### 2.2 环境变量

创建 `.env` 文件：

```bash
# 数据集发现所需
ROBOFLOW_API_KEY=your_roboflow_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
HUGGINGFACE_TOKEN=your_hf_token

# Redis（任务队列）
REDIS_URL=redis://localhost:6379/0

# JWT 密钥
JWT_SECRET_KEY=your_secret_key
```

### 2.3 启动服务

```bash
# 启动 Redis
redis-server

# 启动 API 服务器
uvicorn src.api.gateway:app --host 0.0.0.0 --port 8000

# 启动 Celery worker（独立终端）
celery -A src.api.tasks worker --loglevel=info
```

---

## 3. API 使用指南

### 3.1 健康检查

```bash
GET /health

响应：
{
  "status": "healthy",
  "version": "6.0.0"
}
```

### 3.2 数据集搜索

在多个数据源中搜索数据集。

```bash
POST /api/v1/data/search

请求：
{
  "query": "car detection",
  "max_results": 10
}

响应：
{
  "datasets": [
    {
      "name": "car-detection",
      "source": "roboflow",
      "url": "https://universe.roboflow.com/...",
      "license": "MIT",
      "images": 1000,
      "relevance_score": 0.95
    }
  ],
  "total": 1
}
```

### 3.3 启动训练

提交训练任务。

```bash
POST /api/v1/train/start

请求：
{
  "data_yaml": "/data/car detection.yaml",
  "model": "yolo11m",
  "epochs": 100,
  "imgsz": 640
}

响应：
{
  "task_id": "train_abc123",
  "status": "submitted",
  "message": "Training job submitted successfully"
}
```

### 3.4 查看训练状态

```bash
GET /api/v1/train/status/{task_id}

响应：
{
  "task_id": "train_abc123",
  "status": "running",
  "progress": 0.45
}
```

### 3.5 导出模型

为边缘部署导出训练好的模型。

```bash
POST /api/v1/deploy/export

请求：
{
  "model_path": "/runs/train/weights/best.pt",
  "platform": "jetson_orin",
  "imgsz": 640
}

响应：
{
  "task_id": "export_xyz789",
  "status": "submitted"
}
```

---

## 4. Python SDK 使用

### 4.1 数据集发现

```python
from src.data.discovery import DatasetDiscovery

# 初始化
discovery = DatasetDiscovery(output_dir="./data")

# 搜索数据集
results = discovery.search("car detection", max_results=10)

for ds in results:
    print(f"{ds.name} ({ds.source}) - Score: {ds.relevance_score}")

# 下载数据集
discovery.download(results[0], output_path="./data/car")
```

### 4.2 训练

```python
from src.training.runner import YOLOTrainer

# 初始化训练器
trainer = YOLOTrainer(
    model="yolo11m",
    output_dir="./runs"
)

# 训练模型
result = trainer.train(
    data_yaml="./data/car/data.yaml",
    epochs=100,
)

print(f"状态: {result.status}")
print(f"模型: {result.model_path}")
print(f"mAP50: {result.metrics.get('mAP50', 0):.3f}")
```

### 4.3 超参数优化

```python
from src.training.runner import YOLOTrainer
from src.training.config import HPOConfig

# 配置 HPO
hpo_config = HPOConfig(
    n_trials=50,
    epochs_per_trial=30,
)

# 运行 HPO
trainer = YOLOTrainer(model="yolo11n", output_dir="./runs")
result = trainer.tune(
    data_yaml="./data/car/data.yaml",
    config=hpo_config,
)

print(f"最佳参数: {result.best_params}")
print(f"最佳 mAP50: {result.metrics.get('mAP50', 0):.3f}")
```

### 4.4 模型导出

```python
from src.deployment.exporter import ModelExporter

# 初始化导出器
exporter = ModelExporter(output_dir="./runs/export")

# 导出为 ONNX
result = exporter.export(
    model_path="./runs/train/weights/best.pt",
    platform="jetson_orin",
    imgsz=640,
)

print(f"格式: {result.format}")
print(f"大小: {result.size_mb} MB")
print(f"路径: {result.model_path}")
```

---

## 5. Agent 工作流

### 5.1 CrewAI 集成

```python
from src.agents.orchestration import create_training_crew

# 创建 Crew
crew = create_training_crew()

# 执行任务
result = crew.kickoff(
    inputs={"task_description": "Detect cars on highway"}
)

print(result)
```

### 5.2 决策规则

系统使用智能决策规则：

| Agent | 决策规则 |
|-------|----------|
| **数据发现** | 分数 > 0.8：直接选择；0.5-0.8：带警告包含；< 0.5：生成合成数据 |
| **训练** | < 1000 张图片：激进数据增强；mAP50 < 0.5：尝试更大模型；边缘部署：使用 YOLO11n |
| **部署** | FPS < 20：优化；内存 < 2GB：INT8 量化；失败：回滚 |

---

## 6. 支持的平台

### 6.1 边缘设备

| 平台 | 格式 | 优化 |
|------|------|------|
| Jetson Nano | TensorRT FP16 | 10-30 FPS |
| Jetson Orin | TensorRT FP16 | 50-100 FPS |
| RK3588 | ONNX FP16 | 30-50 FPS |
| 通用 ARM | ONNX FP32 | 10-20 FPS |

### 6.2 云端/服务器

| 平台 | 格式 | 使用场景 |
|------|------|----------|
| 云端 CPU | ONNX FP32 | 批量推理 |
| 云端 GPU | TensorRT FP16 | 实时推理 |
| 服务器 | PyTorch | 研究 |

---

## 7. 故障排除

### 7.1 常见问题

| 问题 | 解决方案 |
|------|----------|
| API 返回 401 | 检查 JWT_SECRET_KEY 和认证头 |
| 限流 (429) | 等待1分钟或调整限流配置 |
| 训练卡住 | 检查 Celery worker 日志 |
| 模型导出失败 | 验证 CUDA 可用性 |

### 7.2 API 密钥设置

```bash
# Roboflow
# 1. 访问 https://app.roboflow.com/settings/api
# 2. 复制您的 API 密钥

# Kaggle
# 1. 访问 https://www.kaggle.com/account
# 2. 创建新的 API 令牌

# HuggingFace
# 1. 访问 https://huggingface.co/settings/tokens
# 2. 创建具有"读取"权限的新令牌
```

---

## 8. 示例

### 8.1 完整训练流程

```python
from src.data.discovery import DatasetDiscovery
from src.training.runner import YOLOTrainer
from src.deployment.exporter import ModelExporter

# 1. 发现数据集
discovery = DatasetDiscovery()
datasets = discovery.search("car detection", max_results=5)
print(f"找到 {len(datasets)} 个数据集")

# 2. 下载最佳数据集
best = max(datasets, key=lambda x: x.relevance_score)
discovery.download(best, output_path="./data/car")

# 3. 训练模型
trainer = YOLOTrainer(model="yolo11m", output_dir="./runs")
result = trainer.train(
    data_yaml="./data/car/data.yaml",
    epochs=100,
)

# 4. 导出到边缘设备
exporter = ModelExporter(output_dir="./export")
export_result = exporter.export(
    model_path=result.model_path,
    platform="jetson_orin",
)

print(f"已导出到: {export_result.model_path}")
```

---

## 9. API 参考

### 9.1 请求/响应模型

完整的 API schema 请参见 `src/api/routes.py`：

- `DatasetSearchRequest` / `DatasetSearchResponse`
- `TrainRequest` / `TrainResponse` / `TrainStatusResponse`
- `ExportRequest` / `ExportResponse`

---

*文档版本：1.0*
*最后更新：2026-03-14*
