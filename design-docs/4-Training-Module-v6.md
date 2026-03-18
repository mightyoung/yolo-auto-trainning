# 训练模块详细设计

**版本**: 6.0
**所属**: 1+5 设计方案
**核心**: YOLO11 训练 + Ray Tune HPO + 知识蒸馏
**更新**: 基于 Ultralytics 官方最佳实践修正

---

## 1. 模块职责

| 职责 | 描述 |
|------|------|
| Sanity Check | 快速验证训练可行性 (10 epochs, imgsz=640) |
| 正式训练 | YOLO11 完整训练 (300 epochs, imgsz=1280) |
| 超参优化 | Ray Tune HPO (6 个核心参数) |
| 知识蒸馏 | YOLO11m → YOLO11n (官方 teacher API) |
| 模型导出 | ONNX (FP16, opset 13) |

---

## 2. 专家建议

> "Train small, deploy small" — Andrej Karpathy
> "Use official defaults first, then optimize" — Ultralytics Best Practices

**核心原则**：
1. **使用官方默认值** - lr0=0.01, box=7.5, fliplr=0.5
2. **分离 HPO 参数** - 优化器参数与数据增强分离
3. **知识蒸馏** - 使用官方 teacher 参数

---

## 3. 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                       Training Module                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Data Merger                                     │  │
│  │    (Discovery 数据 + Synthetic 数据)                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Sanity Check Runner                            │  │
│  │         10 epochs, imgsz=640, patience=100                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              HPO Optimizer (Ray Tune)                     │  │
│  │    lr0, lrf, momentum, weight_decay, box, cls (6 params)  │  │
│  │    50 trials × 50 epochs                                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Final Trainer                                  │  │
│  │     300 epochs, best params + augmentation               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Knowledge Distillation                        │  │
│  │    Teacher: YOLO11m, Student: YOLO11n                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Model Exporter                                │  │
│  │         ONNX (FP16, opset 13)                             │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心组件

### 4.1 数据集合并器

```python
# src/training/data_merger.py
from pathlib import Path
from typing import List, Dict
import shutil

class DataMerger:
    """合并发现的数据集 + 合成数据集"""

    def __init__(self):
        self.synthetic_ratio_threshold = 0.3  # 合成数据不超过 30%

    def merge(
        self,
        discovered_datasets: List[Path],
        synthetic_dataset: Path = None,
        output_dir: Path = None
    ) -> Dict:
        output_dir = Path(output_dir or "./data/merged")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 统计真实数据
        real_images = 0
        for ds in discovered_datasets:
            real_images += self._count_images(ds / "train")

        # 统计合成数据
        synthetic_images = 0
        if synthetic_dataset and synthetic_dataset.exists():
            synthetic_images = self._count_images(synthetic_dataset / "train")

        # 计算合成比例
        total = real_images + synthetic_images
        synthetic_ratio = synthetic_images / total if total > 0 else 0

        # 如果合成比例超过阈值，采样减少
        if synthetic_ratio > self.synthetic_ratio_threshold:
            max_synthetic = int(real_images * self.synthetic_ratio_threshold / (1 - self.synthetic_ratio_threshold))
            synthetic_images = self._limit_images(synthetic_dataset, max_synthetic)

        # 复制到输出目录
        for ds in discovered_datasets:
            self._copy_dataset(ds, output_dir / "discovered")

        if synthetic_images > 0:
            self._copy_dataset(synthetic_dataset, output_dir / "synthetic")

        return {
            "train_images": real_images + synthetic_images,
            "val_images": int((real_images + synthetic_images) * 0.1),
            "synthetic_ratio": synthetic_ratio,
            "output_path": str(output_dir)
        }

    def _count_images(self, img_dir: Path) -> int:
        if not img_dir.exists():
            return 0
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        return len([f for f in img_dir.iterdir() if f.suffix.lower() in extensions])

    def _limit_images(self, src_dir: Path, max_count: int) -> int:
        images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
        return min(len(images), max_count)

    def _copy_dataset(self, src: Path, dst: Path):
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
```

### 4.2 Sanity Check 运行器

```python
# src/training/sanity_check.py
from ultralytics import YOLO
from pathlib import Path
from typing import Dict

class SanityCheckRunner:
    """快速验证训练可行性 - 基于 Ultralytics 最佳实践"""

    # Sanity Check 配置
    CONFIG = {
        "epochs": 10,           # 快速验证
        "imgsz": 640,          # 标准分辨率
        "patience": 100,
        "batch": 16,
        "cache": True,
    }

    def __init__(self, model_size: str = "yolo11m"):
        self.model_size = model_size

    def run(self, data_yaml: Path, output_dir: Path = None) -> Dict:
        output_dir = Path(output_dir or "./runs/sanity")
        output_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO(f"{self.model_size}.pt")

        results = model.train(
            data=str(data_yaml),
            epochs=self.CONFIG["epochs"],
            imgsz=self.CONFIG["imgsz"],
            patience=self.CONFIG["patience"],
            batch=self.CONFIG["batch"],
            cache=self.CONFIG["cache"],
            project=str(output_dir),
            name="sanity_check",
            exist_ok=True,
            verbose=False,
        )

        best_map50 = results.results_dict.get("metrics/mAP50(B)", 0)
        best_map50_95 = results.results_dict.get("metrics/mAP50-95(B)", 0)
        passed = best_map50 >= 0.3

        return {
            "status": "passed" if passed else "failed",
            "mAP50": best_map50,
            "mAP50-95": best_map50_95,
            "training_time_min": self.CONFIG["epochs"] * 0.5,
            "recommendation": "Continue to full training" if passed else "Dataset too small"
        }
```

### 4.3 HPO 优化器（基于官方 Ray Tune 集成）

```python
# src/training/hpo_optimizer.py
from ultralytics import YOLO
from pathlib import Path
from typing import Dict
from ray import tune

class HPOOptimizer:
    """超参数优化器 - 基于 Ultralytics 官方 Ray Tune 集成

    参考: https://docs.ultralytics.com/integrations/ray-tune/
    """

    # 优化器参数空间 - 基于官方默认值
    PARAM_SPACE = {
        "lr0": [0.001, 0.01],        # 官方默认 0.01
        "lrf": [0.01, 1.0],
        "momentum": [0.6, 0.98],     # 官方默认 0.937
        "weight_decay": [0.0001, 0.001],  # 官方默认 0.0005
        "box": [5.0, 10.0],          # 官方默认 7.5
        "cls": [0.3, 1.0],           # 官方默认 0.5
    }

    # 固定参数 - 数据增强保持官方默认值
    FIXED_PARAMS = {
        "epochs": 50,                 # HPO 时使用 50 epochs
        "imgsz": 1280,
        "batch": 16,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "close_mosaic": 10,
        "patience": 100,
        # 数据增强保持固定 - 使用官方默认值
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "fliplr": 0.5,              # 官方默认 0.5
        "mosaic": 1.0,
        "mixup": 0.0,
    }

    def __init__(self, n_trials: int = 50):
        self.n_trials = n_trials

    def optimize(
        self,
        data_yaml: Path,
        output_dir: Path = None,
        metric: str = "mAP50",
    ) -> Dict:
        output_dir = Path(output_dir or "./runs/hpo")
        output_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO("yolo11m.pt")

        # 使用官方 Ray Tune 集成
        result_grid = model.tune(
            data=str(data_yaml),
            space={
                "lr0": tune.uniform(0.001, 0.01),
                "lrf": tune.uniform(0.01, 1.0),
                "momentum": tune.uniform(0.6, 0.98),
                "weight_decay": tune.uniform(0.0001, 0.001),
                "box": tune.uniform(5.0, 10.0),
                "cls": tune.uniform(0.3, 1.0),
            },
            epochs=50,
            imgsz=1280,
            batch=16,
            use_ray=True,
            grace_period=10,
            project=str(output_dir),
        )

        return {
            "best_params": result_grid.best_result,
            "best_score": result_grid.best_result.metrics.get("metrics/mAP50(B)", 0),
        }
```

### 4.4 知识蒸馏器（基于官方 teacher API）

```python
# src/training/knowledge_distillation.py
"""
知识蒸馏训练器 - 基于 Ultralytics 官方 teacher API

参考:
- https://github.com/danielsyahputra/yolo-distiller
- https://github.com/ultralytics/ultralytics/issues/17013

正确的调用方式:
student_model.train(
    data="data.yaml",
    teacher=teacher_model.model,  # 传入教师模型
    distillation_loss="cwd",      # 蒸馏损失类型
    epochs=100
)
"""
from ultralytics import YOLO
from pathlib import Path
from typing import Dict


class KnowledgeDistillationTrainer:
    """使用 Ultralytics 官方知识蒸馏 API"""

    def __init__(
        self,
        teacher_model: str = "yolo11m",
        student_model: str = "yolo11n",
        distill_weight: float = 1.0,
    ):
        self.teacher_model_name = teacher_model
        self.student_model_name = student_model

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
    ) -> Dict:
        """
        执行知识蒸馏训练

        使用官方 teacher 参数:
        - teacher: 教师模型
        - distillation_loss: 蒸馏损失类型 ("cwd" 或 "kl")
        """
        # 加载教师模型
        teacher = YOLO(f"{self.teacher_model_name}.pt")

        # 加载学生模型
        student = YOLO(f"{self.student_model_name}.pt")

        # 评估教师模型
        teacher_results = teacher.val(data=data_yaml)
        teacher_map50 = teacher_results.results_dict.get("metrics/mAP50(B)", 0)

        # 使用官方知识蒸馏 API
        student_results = student.train(
            data=data_yaml,
            epochs=epochs,
            teacher=teacher.model,  # 官方 teacher 参数
            distillation_loss="cwd",  # Channel-Wise Distillation
            project="./runs/distill",
            name="student",
            verbose=False,
        )

        student_map50 = student_results.results_dict.get("metrics/mAP50(B)", 0)

        return {
            "student_model": student_results.save_dir / "weights" / "best.pt",
            "teacher_mAP50": teacher_map50,
            "student_mAP50": student_map50,
            "method": "teacher_api_distillation"
        }


# 迁移学习 - 更简单的蒸馏方式
class TransferLearningTrainer:
    """迁移学习训练器 - 使用教师权重初始化学生模型"""

    def __init__(self):
        self.teacher = "yolo11m"
        self.student = "yolo11n"

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        freeze_layers: int = 10
    ) -> Dict:
        # 使用教师权重初始化学生
        model = YOLO(self.teacher)

        # 冻结骨干网络
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            freeze=freeze_layers,
            project="./runs/transfer",
            name="student",
            verbose=False
        )

        return {
            "model": results.save_dir / "weights" / "best.pt",
            "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
            "method": "transfer_learning"
        }
```

### 4.5 模型导出器

```python
# src/training/model_exporter.py
from ultralytics import YOLO
from pathlib import Path
from typing import Dict

class ModelExporter:
    """模型导出器 - 导出为 ONNX"""

    # 导出配置
    EXPORT_CONFIGS = {
        "opset": 13,
        "half": True,  # FP16
        "dynamic": False,
        "simplify": True,
    }

    PLATFORM_CONFIGS = {
        "jetson": {
            "opset": 13,
            "half": True,
            "dynamic": False,
        },
        "tensorrt": {
            "opset": 13,
            "half": True,
            "dynamic": True,
        },
        "cpu": {
            "opset": 13,
            "half": False,
            "dynamic": False,
        }
    }

    def export(
        self,
        model_path: Path,
        output_dir: Path = None,
        platform: str = "jetson"
    ) -> Dict:
        output_dir = Path(output_dir or "./runs/export")
        output_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO(str(model_path))
        config = self.PLATFORM_CONFIGS.get(platform, self.PLATFORM_CONFIGS["jetson"])

        export_path = model.export(
            format="onnx",
            project=str(output_dir),
            exist_ok=True,
            **config
        )

        model_size_mb = Path(export_path).stat().st_size / (1024 * 1024)

        return {
            "model": export_path,
            "size_mb": model_size_mb,
            "platform": platform,
            "fp16": config["half"]
        }
```

---

## 5. 训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                                    │
└─────────────────────────────────────────────────────────────────┘

1. 数据合并
   ├── 合并 Discovery 数据 + Synthetic 数据
   ├── 确保合成数据 ≤ 30%
   └── 输出 merged dataset

2. Sanity Check (10 epochs, imgsz=640)
   ├── mAP50 >= 0.3 → 通过
   └── mAP50 < 0.3 → 失败，需要更多数据

3. HPO (50 trials, 50 epochs)
   ├── 优化器参数: lr0, lrf, momentum, weight_decay, box, cls
   ├── 数据增强固定: hsv=0.015/0.7/0.4, fliplr=0.5
   └── 返回最佳参数

4. 最终训练 (300 epochs)
   ├── 使用最佳超参
   ├── 早停 patience=100
   └── 保存 best.pt

5. 知识蒸馏 (可选)
   ├── Teacher: YOLO11m
   ├── Student: YOLO11n
   └── 使用官方 teacher API

6. 模型导出
   ├── ONNX (FP16, opset 13)
   └── 平台: Jetson / TensorRT / CPU
```

---

## 6. Ultralytics 官方默认超参数

基于 [Ultralytics 官方文档](https://docs.ultralytics.com/usage/cfg/):

| 参数 | 默认值 | v5 设计值 | v6 修正值 | 状态 |
|------|--------|-----------|-----------|------|
| `lr0` | 0.01 | 0.001 | **0.01** | ✅ 修正 |
| `lrf` | 0.01 | 0.01 | 0.01 | ✅ |
| `momentum` | 0.937 | 0.937 | 0.937 | ✅ |
| `weight_decay` | 0.0005 | 0.0005 | 0.0005 | ✅ |
| `box` | **7.5** | 0.05 | **7.5** | ✅ 修正 |
| `cls` | 0.5 | 0.5 | 0.5 | ✅ |
| `dfl` | 1.5 | 1.5 | 1.5 | ✅ |
| `hsv_h` | 0.015 | 0.015 | 0.015 | ✅ |
| `hsv_s` | 0.7 | 0.7 | 0.7 | ✅ |
| `hsv_v` | 0.4 | 0.4 | 0.4 | ✅ |
| `fliplr` | **0.5** | 0.0 | **0.5** | ✅ 修正 |

---

## 7. 与其他模块的集成

```
Dataset Discovery ──► Training ──► Deployment
     │                     │
     ▼                     ▼
合成数据 ≤ 30%        ONNX Export
```

---

## 8. 依赖

```python
dependencies = [
    "ultralytics>=8.0.0",
    "ray[tune]>=2.0.0",
    "torch>=2.0.0",
]
```

---

*文档版本: 6.0*
*核心功能: YOLO11 训练 + Ray Tune HPO + 知识蒸馏*
*更新: 基于官方最佳实践修正超参数和知识蒸馏 API*
