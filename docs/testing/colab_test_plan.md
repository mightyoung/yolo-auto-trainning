# YOLO Auto-Training Colab 测试方案

## 1. 方案概述

### 目标
在 Google Colab 上验证 yolo-auto-trainning 项目的核心训练功能

### 架构选择

由于 Colab 是单实例环境，采用**精简架构**：

```
┌─────────────────────────────────────────────────┐
│                 Google Colab                      │
│                                                  │
│  ┌─────────────────┐    ┌──────────────────┐   │
│  │  Business API   │───▶│  Training API    │   │
│  │  (简化版)       │    │  (完整版 - GPU)  │   │
│  │  port: 8000    │    │  port: 8001     │   │
│  └─────────────────┘    └──────────────────┘   │
│           │                      │               │
│           ▼                      ▼               │
│  ┌─────────────────┐    ┌──────────────────┐   │
│  │  In-Memory/     │    │   YOLO Model    │   │
│  │  SQLite Task    │    │   Training      │   │
│  │  Storage        │    │   (GPU)         │   │
│  └─────────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────┘
```

---

## 2. Colab Notebook 结构

### 2.1 环境准备 (Step 1-2)

```python
# ============== Step 1: 环境检查 ==============
# 检查 GPU
import subprocess
result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
print(result.stdout)

# 检查 PyTorch + CUDA
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============== Step 2: 安装依赖 ==============
# 核心依赖
!pip install ultralytics>=8.0.0
!pip install ray[tune]>=2.0.0
!pip install fastapi uvicorn pydantic pydantic-settings
!pip install python-jose cryptography
```

### 2.2 准备测试数据 (Step 3)

```python
# ============== Step 3: 准备测试数据 ==============
# 使用 COCO128 或自定义小数据集进行快速验证
import os
import urllib.request

# 下载示例数据集
!mkdir -p /content/data
%cd /content/data

# 使用 Ultralytics 官方示例数据集 (COCO128)
!yolo detect dataset=coco128 batch=16
# 或手动下载小样本进行测试
```

### 2.3 核心训练测试 (Step 4)

```python
# ============== Step 4: 直接训练测试 ==============
from ultralytics import YOLO
import torch

# 检查 GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载模型
model = YOLO("yolo11m.pt" if torch.cuda.is_available() else "yolo11n.pt")

# 使用公开数据集进行训练测试
# COCO128 是一个128张图像的小数据集，适合快速验证
results = model.train(
    data="coco128.yaml",  # Ultralytics 内置数据集
    epochs=3,  # 快速验证用少量 epoch
    imgsz=320,  # 小尺寸加速
    batch=8,
    device=device,
    project="/content/runs",
    exist_ok=True,
    verbose=True
)

print("训练完成!")
print(f"Results: {results}")
```

### 2.4 Training API 测试 (Step 5)

```python
# ============== Step 5: Training API 服务 ==============
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, "/content/yolo-auto-trainning")

# 启动 Training API
import asyncio
from training_api.src.api.gateway import app
from training_api.src.training.runner import YOLOTrainer

# 配置
os.environ["INTERNAL_API_KEY"] = "test-api-key-12345"
os.environ["REDIS_URL"] = "redis://localhost:6379"  # 或使用内存存储

# 启动服务 (后台)
# uvicorn training_api.src.api.gateway:app --host 0.0.0.0 --port 8001 &
```

### 2.5 端到端测试 (Step 6)

```python
# ============== Step 6: 端到端测试 ==============
import httpx
import json

TRAINING_API_URL = "http://localhost:8001"
API_KEY = "test-api-key-12345"

headers = {"X-API-Key": API_KEY}

# 6.1 健康检查
health = httpx.get(f"{TRAINING_API_URL}/health")
print(f"Health: {health.json()}")

# 6.2 启动训练任务
train_request = {
    "task_id": "colab-test-001",
    "model": "yolo11n",
    "data_yaml": "coco128.yaml",
    "epochs": 3,
    "imgsz": 320,
    "batch": 8,
    "device": "cuda:0"
}

response = httpx.post(
    f"{TRAINING_API_URL}/train/start",
    json=train_request,
    headers=headers,
    timeout=300  # 5分钟超时
)
print(f"Train Start Response: {response.json()}")

# 6.3 检查训练状态
task_id = train_request["task_id"]
status_response = httpx.get(
    f"{TRAINING_API_URL}/train/status/{task_id}",
    headers=headers
)
print(f"Status: {status_response.json()}")
```

---

## 3. 完整 Colab Notebook

### 3.1 Notebook 链接方案

创建以下 notebook：

| Notebook | 用途 | 运行时长 |
|----------|------|----------|
| `colab/yolo_auto_training_colab.ipynb` | 完整测试 (含 GitHub 克隆) | ~20-40 min |

### 3.2 推荐测试流程

```
1. 打开 yolo_auto_training_colab.ipynb
   ↓
2. 选择 T4 GPU 运行时
   ↓
3. 运行所有单元格 (Ctrl+F9)
   ↓
4. 验证每个 Step 的输出
```

---

## 4. 简化版：单 Notebook 方案

对于快速验证，创建 `colab/yolo_auto_training_colab.ipynb`：

```python
# ===========================================
# YOLO Auto-Training Colab 快速测试
# ===========================================

# ===========================================
# Section 1: 环境准备
# ===========================================
# GPU 检查
import subprocess
print("=" * 50)
print("GPU 信息:")
print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout)

# PyTorch 检查
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 50)

# ===========================================
# Section 2: 安装依赖
# ===========================================
!pip install ultralytics>=8.0.0 ray[tune]>=2.0.0 fastapi uvicorn pydantic-settings -q

# ===========================================
# Section 3: 项目导入
# ===========================================
import sys
from pathlib import Path

# 克隆项目 (如果从 GitHub 运行)
# !git clone https://github.com/your-repo/yolo-auto-trainning.git
PROJECT_ROOT = Path("/content/yolo-auto-trainning")
sys.path.insert(0, str(PROJECT_ROOT))

# ===========================================
# Section 4: 训练测试 (核心功能)
# ===========================================
from ultralytics import YOLO

# 使用小型模型进行快速测试
model_name = "yolo11n"  # nano 模型，最快
model = YOLO(f"{model_name}.pt")

# 使用内置 COCO128 数据集快速验证
print("\n" + "=" * 50)
print("开始训练测试...")
print("=" * 50)

results = model.train(
    data="coco128.yaml",  # Ultralytics 内置数据集
    epochs=3,
    imgsz=320,
    batch=16,
    device=0 if torch.cuda.is_available() else "cpu",
    project="/content/runs",
    exist_ok=True,
    verbose=True
)

print("\n" + "=" * 50)
print("训练完成!")
print("=" * 50)

# 提取关键指标
if hasattr(results, 'results_dict'):
    metrics = results.results_dict
    print(f"mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"训练时间: {metrics.get('elapsed_time', 'N/A')}s")

# ===========================================
# Section 5: 模型导出测试 (可选)
# ===========================================
print("\n" + "=" * 50)
print("测试模型导出...")
print("=" * 50)

# 导出为 ONNX
export_path = model.export(format="onnx", imgsz=320)
print(f"导出成功: {export_path}")

# ===========================================
# Section 6: API 服务测试 (可选)
# ===========================================
# 如果需要测试完整 API，取消注释以下代码

# from training_api.src.api.gateway import app
# import uvicorn
# import threading

# def run_api():
#     uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")

# api_thread = threading.Thread(target=run_api, daemon=True)
# api_thread.start()
# print("Training API 已启动: http://localhost:8001")
```

---

## 5. 注意事项

### 5.1 Colab 限制

| 限制 | 说明 | 应对方案 |
|------|------|----------|
| 运行时间 | ~90 分钟 (免费版) | 使用少量 epoch 快速测试 |
| GPU 内存 | ~15 GB (T4) | 使用 yolo11n 或 batch=8 |
| 磁盘空间 | ~100 GB | 训练输出保存到 Google Drive |
| 网络 | 有限制 | 提前下载模型/数据集 |

### 5.2 数据集准备

推荐使用 Ultralytics 内置数据集：
- `coco128.yaml` - 128张 COCO 图像
- `coco8.yaml` - 8张 COCO 图像

或使用 Roboflow 导出的数据集：
```python
# 从 Roboflow 下载
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("project").project("dataset")
dataset = project.version(1).download("yolov8")
```

### 5.3 保存结果

```python
# 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 复制训练结果
import shutil
shutil.copytree("/content/runs", "/content/drive/MyDrive/yolo_training_results")
```

---

## 6. 测试检查清单

| # | 测试项 | 预期结果 | 状态 |
|---|--------|----------|------|
| 1 | GPU 检测 | nvidia-smi 显示 GPU | ☐ |
| 2 | PyTorch CUDA | torch.cuda.is_available() = True | ☐ |
| 3 | Ultralytics 安装 | yolo 命令可用 | ☐ |
| 4 | 模型加载 | yolo11n.pt 下载成功 | ☐ |
| 5 | 数据集下载 | coco128.yaml 可用 | ☐ |
| 6 | 训练启动 | 开始训练无报错 | ☐ |
| 7 | 训练完成 | 3 epoch 完成 | ☐ |
| 8 | 指标记录 | mAP50 > 0 | ☐ |
| 9 | 模型导出 | ONNX 导出成功 | ☐ |
| 10 | API 启动 | /health 返回 200 | ☐ |

---

## 7. 快速验证命令

```bash
# 一键测试命令 (在 Colab cell 中运行)
!python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
results = model.train(data='coco128.yaml', epochs=1, imgsz=160, batch=8, device=0 if torch.cuda.is_available() else 'cpu', verbose=False)
print('训练成功!' if results else '训练失败')
"
```

---

## 8. 下一步

测试成功后，可以进一步测试：

1. **HPO 超参数优化** - 使用 Ray Tune 进行超参搜索
2. **模型导出** - 测试 ONNX/TensorRT 导出
3. **完整 API** - 启动 Business API + Training API
4. **分布式训练** - 多 GPU 或多节点训练
