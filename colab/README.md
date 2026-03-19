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
| 3 | 项目克隆 | 从 GitHub 克隆，创建 __init__.py |
| 4 | Business API | 数据集发现服务 (port 8000) |
| 5 | Training API | 训练任务服务 (port 8001) |
| 6 | 数据集 API Keys | Kaggle 已配置 |
| 7 | 健康检查 & 数据集发现 | 使用 /api/v1/data/search |
| 8 | 任务提交 | 使用 /api/v1/internal/train/start |
| 9 | 状态监控 | 使用 /api/v1/internal/train/status/{id} |
| 10 | 直接训练 | 使用 Ultralytics 直接训练 |
| 11 | 模型导出 | 导出为 ONNX |

## API Endpoints (已验证)

### Business API (port 8000)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/v1/data/search` | POST | 数据集搜索 |

### Training API (port 8001)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/v1/internal/train/start` | POST | 提交训练任务 |
| `/api/v1/internal/train/status/{task_id}` | GET | 获取训练状态 |
| `/api/v1/internal/train/cancel/{task_id}` | POST | 取消训练任务 |

## API Keys 配置

### Kaggle (已配置)
```python
os.environ["KAGGLE_USERNAME"] = "amurdaddy"
os.environ["KAGGLE_KEY"] = "KGAT_xxxxxxxxxxxxxxxx"
```

### 其他 API Keys (可选)

```python
# Roboflow
os.environ["ROBOFLOW_API_KEY"] = "your-roboflow-api-key"

# HuggingFace
os.environ["HF_TOKEN"] = "your-huggingface-token"
```

## 故障排除

### ModuleNotFoundError: No module named 'api'

**问题**: Colab 克隆代码后 `__init__.py` 文件可能缺失。

**解决方案**: Notebook Step 3 会自动创建 `__init__.py` 文件。如果仍然报错，尝试：

1. 重启 Runtime: Runtime → Restart runtime
2. 重新运行所有 cells: Runtime → Run all

### sys.path 顺序问题

**问题**: 模块导入时找到错误的包。

**解决方案**: Notebook Step 4 会清理并重新设置 sys.path：
```python
# 清理旧的路径
sys.path = [p for p in sys.path if 'yolo-auto-trainning' not in p]
# 设置正确的顺序
sys.path.insert(0, f"{PROJECT_ROOT}/business-api/src")
sys.path.append(f"{PROJECT_ROOT}/training-api/src")
```

### Redis 连接问题

**问题**: Colab 没有 Redis 服务。

**解决方案**: Notebook 会设置 `DISABLE_REDIS=1`，API 会在没有 Redis 的情况下运行（速率限制将被禁用）。

### 训练超时

Colab 免费版运行时间有限 (~90分钟)。如果训练超时：
- 减少 epochs 数量
- 使用更小的模型 (yolo11n)
- 使用更小的图像尺寸 (320)

## 测试检查清单

- [ ] GPU 检测成功
- [ ] 依赖安装成功
- [ ] 项目克隆成功
- [ ] __init__.py 文件创建成功
- [ ] Business API 启动成功 (port 8000)
- [ ] Training API 启动成功 (port 8001)
- [ ] 健康检查通过
- [ ] 数据集搜索成功 (demo 数据)
- [ ] 训练任务提交成功
- [ ] 训练状态查询成功
- [ ] 直接训练完成 (mAP50 > 0)
- [ ] ONNX 导出成功

## 本地测试验证

在推送 Colab 测试之前，已在本地验证以下 API 端点：

```bash
# Business API 测试
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/data/search \
  -H "X-API-Key: test-api-key" \
  -d '{"query": "yolo", "max_results": 5}'

# Training API 测试
curl http://localhost:8001/health
curl -X POST http://localhost:8001/api/v1/internal/train/start \
  -H "X-API-Key: test-training-key" \
  -d '{"task_id": "test-001", "model": "yolo11n", "data_yaml": "coco128.yaml", "epochs": 1, "device": "cpu"}'
```

## 下一步

测试成功后，可以进一步测试：

1. **HPO 超参数优化** - 使用 Ray Tune 进行超参搜索
2. **模型导出** - 测试 ONNX/TensorRT 导出
3. **真实数据集** - 使用 Roboflow/Kaggle/HuggingFace API Keys
4. **分布式训练** - 多 GPU 或多节点训练
