# Google Colab 测试

本目录包含用于在 Google Colab 上测试 yolo-auto-trainning 项目的文件。

## 快速开始

### 方法 1: 直接上传 Notebook

1. 打开 Google Colab (https://colab.research.google)
2. 上传 `yolo_auto_training_colab.ipynb`
3. 选择 GPU 运行时: Runtime → Change runtime type → **T4 GPU** 或 **A100**
4. 点击 **Run all** (或按 Ctrl+F9 运行所有)

### 方法 2: 从 Google Drive 打开

1. 将本文件夹上传到 Google Drive
2. 右键点击 `.ipynb` 文件
3. 选择 "打开方式" → "Google Colaboratory"

### 方法 3: GitHub 直接打开

1. 打开 https://github.com/muyi-dev/yolo-auto-trainning
2. 进入 `colab/` 目录
3. 点击 `.ipynb` 文件
4. 点击 "Open in Colab" 按钮

## 文件说明

| 文件 | 说明 |
|------|------|
| `yolo_auto_training_colab.ipynb` | 完整的 Colab Notebook (从 GitHub 克隆) |
| `quick_test.sh` | 快速测试命令 |

## Colab 限制注意事项

| 限制 | 免费版 | Pro/Pro+ |
|------|--------|----------|
| GPU | T4 (约 15GB) | A100 (约 40GB) |
| 运行时间 | ~90 分钟 | ~12 小时 |
| 并发 | 1 | 1 |

## 训练配置建议

根据 GPU 类型自动选择配置：

### T4 (免费版)
```python
model = "yolo11n"
epochs = 10
imgsz = 320
batch = 16
```

### A100 (Pro版)
```python
model = "yolo11m"
epochs = 50
imgsz = 640
batch = 32
```

## 测试流程

### 完整测试清单

| 步骤 | 检查项 | 说明 |
|------|--------|------|
| 1 | GPU 检测 | 自动识别 T4/A100 |
| 2 | 依赖安装 | ultralytics, fastapi, 等 |
| 3 | 项目克隆 | 从 GitHub 克隆 |
| 4 | Business API | 数据集发现服务 (port 8000) |
| 5 | Training API | 训练任务服务 (port 8001) |
| 6 | 数据集搜索 | 需要 Roboflow/Kaggle/HF API Keys |
| 7 | 任务提交 | 通过 API 提交训练任务 |
| 8 | 状态监控 | 轮询训练进度 |
| 9 | 结果获取 | 获取训练指标 |
| 10 | 直接训练 | 使用 Ultralytics 直接训练 |
| 11 | 模型导出 | 导出为 ONNX |

## API Keys 配置 (可选)

### 数据集发现 (需要付费 API Keys)

如果需要测试自动数据集发现功能，需要配置以下 API Keys：

```python
# Roboflow
os.environ["ROBOFLOW_API_KEY"] = "your-roboflow-api-key"

# Kaggle
os.environ["KAGGLE_USERNAME"] = "your-kaggle-username"
os.environ["KAGGLE_KEY"] = "your-kaggle-api-key"

# HuggingFace
os.environ["HF_TOKEN"] = "your-huggingface-token"
```

### 配置位置

在 Step 4 (启动 Business API) 之前添加环境变量配置。

## 快速验证命令

在 Colab cell 中运行:

```bash
!python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
results = model.train(data='coco128.yaml', epochs=1, imgsz=160, batch=8, device=0, verbose=False)
print('训练成功!' if results else '失败')
"
```

## 测试检查清单

- [ ] GPU 检测成功
- [ ] 依赖安装成功
- [ ] 模型下载成功
- [ ] Business API 启动成功 (port 8000)
- [ ] Training API 启动成功 (port 8001)
- [ ] 数据集搜索 (需要 API Keys)
- [ ] 训练任务提交成功
- [ ] 训练状态监控正常
- [ ] 直接训练完成
- [ ] mAP50 > 0
- [ ] ONNX 导出成功

## 故障排除

### ModuleNotFoundError

如果遇到模块导入错误，确保在克隆项目后设置了正确的 sys.path：

```python
import sys
sys.path.insert(0, '/content/yolo-auto-trainning')
sys.path.insert(0, '/content/yolo-auto-trainning/training-api/src')
```

### 训练超时

Colab 免费版运行时间有限 (~90分钟)。如果训练超时：
- 减少 epochs 数量
- 使用更小的模型 (yolo11n)
- 使用更小的图像尺寸 (320)

### API 服务停止

如果 API 服务在后台线程中停止，可以使用 Pro/Pro+ 获得更长运行时间。

## 下一步

测试成功后，可以进一步测试：

1. **HPO 超参数优化** - 使用 Ray Tune 进行超参搜索
2. **模型导出** - 测试 ONNX/TensorRT 导出
3. **完整 API** - 启动 Business API + Training API
4. **分布式训练** - 多 GPU 或多节点训练
