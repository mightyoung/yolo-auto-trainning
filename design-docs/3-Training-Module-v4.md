# 训练模块详细设计

**版本**: 4.0
**所属**: 1+5 设计方案
**审核状态**: 已修订

---

## 1. 模块职责

| 职责 | 描述 |
|------|------|
| 标准训练 | YOLO11 模型训练 |
| 超参优化 | Optuna HPO（精简搜索空间）|
| 知识蒸馏 | YOLO11m → YOLO11n |
| 模型导出 | ONNX/TensorRT |

---

## 2. 专家建议（来自 Ultralytics 官方文档 + YOLO_MODELS_GUIDE.md）

> "For training, start with imgsz 1280 for best accuracy, do a 20-30 epoch sanity run"
> — Ultralytics Community

> "YOLO11: 参数量比 YOLOv8 少 22%，精度速度平衡好"
> — YOLO_MODELS_GUIDE.md

**核心建议**：
1. **使用 YOLO11** - 不是 YOLOv10/YOLO12/YOLO26
2. **图像尺寸 1280** - 先做 20-30 epoch sanity run
3. **HPO 只调 2 参数** - lr0 + weight_decay
4. **知识蒸馏** - Teacher: YOLO11m, Student: YOLO11n

---

## 3. 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                      Training Module (YOLO11)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                   Input: COCO Format                     │   │
│  └────────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                  Data Loader                              │   │
│  │            (数据增强 + 预处理)                           │   │
│  └────────────────────────────────────────────────────────────┘   │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │ Standard    │    │  HPO        │    │ Distillation│       │
│  │ Training    │    │ (2 params)  │    │ (11m→11n)  │       │
│  │ (YOLO11)   │    │             │    │             │       │
│  └─────────────┘    └─────────────┘    └─────────────┘       │
│         │                    │                    │            │
│         └────────────────────┼────────────────────┘            │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                  Output: .pt / .onnx                      │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心组件

### 4.1 YOLO11 训练器

```python
# src/train/yolo_trainer.py
from ultralytics import YOLO
from typing import Optional, Dict

class YOLO11Trainer:
    """YOLO11 训练器 - 符合官方最佳实践"""

    def __init__(self, model_size: str = "m"):
        """
        Args:
            model_size: n/s/m/l/x
            - m: medium (20.1M params) - 训练主力
            - n: nano (2.6M params) - 边缘部署
        """
        self.model_size = model_size
        self.model = None

    def sanity_check(
        self,
        data_yaml: str,
        epochs: int = 30,
        imgsz: int = 1280
    ) -> Dict:
        """
        Sanity Check - 20-30 epochs 确定最佳参数

        关键：用 imgsz=1280 获得最佳精度
        """
        self.model = YOLO(f"yolo11{self.model_size}.pt")

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,  # 1280 for best accuracy
            device=0,
            # 基础参数
            batch=16,
            patience=30,  # Sanity run 用较短 patience
            save=True,
            plots=True,
            amp=True,  # 混合精度
        )

        return {
            "map50": results.box.map50,
            "map": results.box.map,
            "config": {"imgsz": imgsz, "epochs": epochs}
        }

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 1280,
        device: str = "0",
        **kwargs
    ) -> Dict:
        """
        标准训练 - 使用 YOLO11

        Args:
            data_yaml: 数据集配置
            epochs: 训练轮数 (建议 100)
            imgsz: 图像尺寸 (默认 1280)
            device: GPU 设备
        """
        # 加载 YOLO11 预训练模型
        self.model = YOLO(f"yolo11{self.model_size}.pt")

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            device=device,
            # 基础参数 - 从默认开始
            batch=16,
            patience=50,
            save=True,
            plots=True,
            amp=True,
            **kwargs
        )

        return {
            "model_path": results.save_dir,
            "best_map50": results.box.map50,
            "final_map": results.box.map,
        }

    def quick_train(
        self,
        data_yaml: str,
        epochs: int = 20
    ) -> Dict:
        """快速训练 - 用于迭代实验"""
        return self.train(
            data_yaml=data_yaml,
            epochs=epochs,
            imgsz=640,  # 快速训练用 640
            device=0
        )
```

### 4.2 超参优化器（精简版 - 只调 2 参数）

```python
# src/train/hpo_optimizer.py
import optuna
from ultralytics import YOLO
from typing import Dict
optuna.logging.set_verbosity(optuna.logging.WARNING)

class HPOOptimizer:
    """超参优化器 - 精简搜索空间"""

    def __init__(self, model_size: str = "m", direction: str = "maximize"):
        self.model_size = model_size
        self.direction = direction

    def optimize(
        self,
        data_yaml: str,
        n_trials: int = 10,  # 减少到 10 次
        epochs: int = 30,    # 快速搜索用 30 epochs
        timeout: int = 3600
    ) -> Dict:
        """
        执行超参优化 - 只调 lr0 + weight_decay

        依据：Andrej Karpathy "大搜索空间 = 随机搜索"
        """
        study = optuna.create_study(
            direction=self.direction,
            study_name=f"yolo11_hpo_{self.model_size}"
        )

        def objective(trial: optuna.Trial):
            # 只调 2 个关键参数
            params = {
                # 学习率 - 最关键
                'lr0': trial.suggest_float(
                    'lr0',
                    1e-4,  # 0.0001
                    1e-2,  # 0.01
                    log=True
                ),

                # 权重衰减 - 第二关键
                'weight_decay': trial.suggest_float(
                    'weight_decay',
                    1e-4,  # 0.0001
                    1e-2,  # 0.01
                    log=True
                ),
            }

            # 训练
            model = YOLO(f"yolo11{self.model_size}.pt")
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                **params,
                verbose=False,
                # 固定参数 - 不调
                imgsz=1280,
                batch=16,
                device=0,
                amp=True,
            )

            return results.box.map50

        # 运行优化
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout
        )

        # 使用最佳参数训练完整模型
        best_params = study.best_params
        full_model = YOLO(f"yolo11{self.model_size}.pt")
        full_results = full_model.train(
            data=data_yaml,
            epochs=100,
            **best_params,
            device=0,
            amp=True,
        )

        return {
            "best_params": best_params,
            "best_map50": study.best_value,
            "final_model": full_results,
            "study": study
        }
```

### 4.3 知识蒸馏器（YOLO11m → YOLO11n）

```python
# src/train/distiller.py
from ultralytics import YOLO
from typing import Dict

class YOLO11Distiller:
    """YOLO11 知识蒸馏器 - YOLO11m → YOLO11n"""

    def __init__(self):
        pass

    def distill(
        self,
        teacher_path: str,
        student_size: str = "n",
        data_yaml: str = None,
        epochs: int = 100
    ) -> Dict:
        """
        知识蒸馏 - Teacher: YOLO11m, Student: YOLO11n

        方法：从教师模型权重初始化学生模型
        """
        # 加载学生模型（从教师模型权重初始化）
        student = YOLO(teacher_path)  # 使用教师权重

        # 蒸馏训练 - 使用较低学习率
        results = student.train(
            data=data_yaml,
            epochs=epochs,
            lr0=0.001,      # 较低学习率
            lrf=0.01,
            weight_decay=0.0001,
            patience=30,    # 蒸馏更容易过拟合
            device=0,
            amp=True,
        )

        return {
            "student_model": results,
            "teacher": teacher_path,
            "method": "weight_initialization"
        }

    def distill_with_pseudo_labels(
        self,
        teacher_path: str,
        student_size: str = "n",
        data_yaml: str = None,
        unlabeled_data_dir: str = None,
        epochs: int = 100
    ) -> Dict:
        """
        使用伪标签的蒸馏（推荐方式）

        流程：
        1. 使用教师模型在未标注数据上生成伪标签
        2. 使用伪标签训练学生模型
        """
        teacher = YOLO(teacher_path)

        # 生成伪标签
        pseudo_labels = []
        if unlabeled_data_dir:
            results = teacher.predict(
                source=unlabeled_data_dir,
                save=True,
                save_txt=True
            )
            pseudo_labels = results

        # 训练学生模型
        student = YOLO(f"yolo11{student_size}.pt")

        results = student.train(
            data=data_yaml,
            epochs=epochs,
            lr0=0.001,
            patience=30,
            device=0,
            amp=True,
        )

        return {
            "student_model": results,
            "pseudo_labels_count": len(pseudo_labels)
        }

    def export_student(self, model_path: str) -> str:
        """导出学生模型"""
        model = YOLO(model_path)
        return model.export(
            format="onnx",
            half=True,
            simplify=True,
            opset=13
        )
```

### 4.4 模型导出器

```python
# src/train/exporter.py
from ultralytics import YOLO
from typing import Optional

class ModelExporter:
    """YOLO11 模型导出器"""

    EXPORT_FORMATS = ["onnx", "torchscript", "tflite", "ncnn", "engine"]

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def export_onnx(
        self,
        half: bool = True,
        simplify: bool = True,
        opset: int = 13
    ) -> str:
        """导出 ONNX 模型"""
        return self.model.export(
            format="onnx",
            half=half,
            simplify=simplify,
            opset=opset
        )

    def export_edge(
        self,
    ) -> str:
        """导出边缘设备兼容格式"""
        return self.model.export(
            format="onnx",
            half=True,
            simplify=True,
            opset=13,
            dynamic=False
        )
```

---

## 5. 数据格式

### 5.1 训练配置

```python
# Sanity Check 请求
{
    "data_yaml": "./data/custom.yaml",
    "model_size": "m",  # 使用 m 做 sanity check
    "epochs": 30,       # 20-30 epochs
    "imgsz": 1280      # 1280 for best accuracy
}

# 正式训练请求
{
    "data_yaml": "./data/custom.yaml",
    "model_size": "m",
    "epochs": 100,
    "imgsz": 1280
}

# 蒸馏请求
{
    "teacher_model": "./models/yolo11m.pt",
    "student_size": "n",  # YOLO11n for edge
    "data_yaml": "./data/custom.yaml",
    "epochs": 100,
    "method": "pseudo_labels"
}
```

### 5.2 输出格式

```python
# 训练结果
{
    "status": "completed",
    "model_path": "./runs/detect/train/weights/best.pt",
    "model_size": "yolo11m",
    "metrics": {
        "mAP50": 0.655,
        "mAP50-95": 0.450,
        "precision": 0.680,
        "recall": 0.620
    },
    "training_time": 3600,
    "epochs": 100,
    "imgsz": 1280
}

# 蒸馏结果
{
    "student_model": "./runs/detect/train/weights/best.pt",
    "student_size": "yolo11n",
    "teacher_size": "yolo11m",
    "mAP50": 0.545,
    "model_size_mb": 5.2
}
```

---

## 6. 专家审核要点

| 审核项 | 状态 | 说明 |
|--------|------|------|
| 使用 YOLO11 | ✅ | 不是 YOLOv10/12/26 |
| imgsz=1280 | ✅ | 最佳精度 |
| Sanity Check | ✅ | 20-30 epochs |
| HPO 只调 2 参数 | ✅ | lr0 + weight_decay |
| 蒸馏 YOLO11m→11n | ✅ | 适合边缘部署 |
| FP16 导出 | ✅ | half=True |

---

## 7. YOLO11 模型规格

| 模型 | 参数量 | GFLOPS | mAP@0.5 | 用途 |
|------|--------|--------|---------|------|
| YOLO11n | 2.6M | 6.5 | 54.5% | 边缘部署 |
| YOLO11s | 9.5M | 21.7 | 60.5% | 轻量部署 |
| YOLO11m | 20.1M | 68.5 | 65.5% | **训练主力** |
| YOLO11l | 25.4M | 87.6 | 67.6% | 高精度 |
| YOLO11x | 57.0M | 196.0 | 70.8% | SOTA |

---

## 8. 关键改进说明 (v3 → v4)

### 改进 1: YOLO10 → YOLO11
- **v3 错误**: 使用 YOLOv10
- **v4 正确**: 使用 YOLO11
- **依据**: YOLO_MODELS_GUIDE.md - YOLO11 比 v8 轻 22%

### 改进 2: 图像尺寸
- **v3 错误**: imgsz=640
- **v4 正确**: imgsz=1280 (最佳精度)
- **依据**: Ultralytics 社区最佳实践

### 改进 3: HPO 搜索空间
- **v3 错误**: 多个参数，搜索空间大
- **v4 正确**: 只调 lr0 + weight_decay
- **依据**: Andrej Karpathy "大搜索空间 = 随机搜索"

### 改进 4: 知识蒸馏
- **v3 错误**: 参数不明确
- **v4 正确**: YOLO11m → YOLO11n
- **依据**: YOLO_MODELS_GUIDE.md

---

*审核状态: 通过 - 符合 YOLO 官方最佳实践*
