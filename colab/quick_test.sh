#!/bin/bash
# YOLO Auto-Training Colab 快速测试脚本
# 在 Colab cell 中运行: !bash quick_test.sh

echo "=========================================="
echo "YOLO Auto-Training Colab 快速测试"
echo "=========================================="

echo ""
echo "[1/5] 检查 GPU..."
nvidia-smi -L

echo ""
echo "[2/5] 检查 PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "[3/5] 安装依赖..."
pip install ultralytics>=8.0.0 -q
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"

echo ""
echo "[4/5] 运行快速训练测试..."
python -c "
from ultralytics import YOLO
import torch

model = YOLO('yolo11n.pt')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

results = model.train(
    data='coco128.yaml',
    epochs=1,
    imgsz=160,
    batch=8,
    device=device,
    verbose=False
)
print('训练完成!')
"

echo ""
echo "[5/5] 测试完成!"
echo "=========================================="
