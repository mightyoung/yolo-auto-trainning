# Google Colab 测试

本目录包含用于在 Google Colab 上测试 yolo-auto-trainning 项目的文件。

## 快速开始

### 方法 1: 直接上传 Notebook

1. 打开 Google Colab (https://colab.research.google.com)
2. 上传 `yolo_auto_training_colab.ipynb`
3. 选择 GPU 运行时: Runtime → Change runtime type → **T4 GPU**
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

根据 GPU 类型选择配置：

### T4 (免费版)
```python
epochs = 3
imgsz = 320
batch = 16
model = "yolo11n"  # nano 模型最快
```

### A100 (Pro版)
```python
epochs = 50-100
imgsz = 640
batch = 32-64
model = "yolo11m"  # medium 模型
```

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
- [ ] 数据集可用
- [ ] 训练启动成功
- [ ] 3 epoch 完成
- [ ] mAP50 > 0
- [ ] ONNX 导出成功
