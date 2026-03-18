"""
Training configuration based on Ultralytics official best practices.

Reference: https://docs.ultralytics.com/usage/cfg/
"""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Training configuration based on Ultralytics official defaults."""

    # Model settings
    model: str = "yolo11m"
    task: str = "detect"

    # Official default hyperparameters
    # Reference: https://docs.ultralytics.com/usage/cfg/
    lr0: float = 0.01          # Initial learning rate (SGD)
    lrf: float = 0.01           # Final learning rate factor
    momentum: float = 0.937      # SGD momentum
    weight_decay: float = 0.0005  # L2 regularization
    box: float = 7.5            # Box loss weight
    cls: float = 0.5            # Classification loss weight
    dfl: float = 1.5            # DFL loss weight

    # Data augmentation (official defaults)
    hsv_h: float = 0.015        # Hue augmentation
    hsv_s: float = 0.7          # Saturation augmentation
    hsv_v: float = 0.4          # Brightness augmentation
    degrees: float = 0.0        # Rotation
    translate: float = 0.1       # Translation
    scale: float = 0.5           # Scale
    shear: float = 0.0           # Shear
    perspective: float = 0.0     # Perspective
    flipud: float = 0.0          # Vertical flip
    fliplr: float = 0.5         # Horizontal flip (official default)
    mosaic: float = 1.0          # Mosaic
    mixup: float = 0.0          # Mixup
    copy_paste: float = 0.0     # Copy-paste

    # Training settings
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    patience: int = 100         # Early stopping
    warmup_epochs: float = 3.0  # Warmup
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    close_mosaic: int = 10      # Disable mosaic in last N epochs

    # Optimization
    optimizer: str = "SGD"
    amp: bool = True            # Automatic Mixed Precision
    cache: bool = False         # Cache images

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YOLO training."""
        return {
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "box": self.box,
            "cls": self.cls,
            "dfl": self.dfl,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "patience": self.patience,
            "warmup_epochs": self.warmup_epochs,
            "warmup_momentum": self.warmup_momentum,
            "warmup_bias_lr": self.warmup_bias_lr,
            "close_mosaic": self.close_mosaic,
            "optimizer": self.optimizer,
            "amp": self.amp,
            "cache": self.cache,
        }


@dataclass
class SanityCheckConfig:
    """Sanity check configuration for quick validation."""

    epochs: int = 10
    imgsz: int = 640
    batch: int = 16
    patience: int = 100
    cache: bool = True
    min_map50: float = 0.3


@dataclass
class HPOConfig:
    """Hyperparameter optimization configuration."""

    # Parameters to optimize (6 core optimizer parameters)
    # Separation of augmentation and optimization is key
    param_space: Dict[str, Any] = field(default_factory=lambda: {
        "lr0": (0.001, 0.01),        # Official default: 0.01
        "lrf": (0.01, 1.0),
        "momentum": (0.6, 0.98),     # Official default: 0.937
        "weight_decay": (0.0001, 0.001),  # Official default: 0.0005
        "box": (5.0, 10.0),          # Official default: 7.5
        "cls": (0.3, 1.0),           # Official default: 0.5
    })

    # Fixed parameters (data augmentation)
    fixed_params: Dict[str, Any] = field(default_factory=lambda: {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "fliplr": 0.5,              # Official default
        "mosaic": 1.0,
        "mixup": 0.0,
    })

    n_trials: int = 50
    epochs_per_trial: int = 50
    imgsz: int = 1280
    grace_period: int = 10  # ASHA early stopping


@dataclass
class ExportConfig:
    """Model export configuration."""

    format: str = "onnx"
    opset: int = 13
    half: bool = True        # FP16
    dynamic: bool = False
    simplify: bool = True

    # Platform-specific configs
    platform_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "jetson": {
            "format": "engine",
            "half": True,
            "dynamic": True,
        },
        "tensorrt": {
            "format": "engine",
            "half": True,
            "dynamic": True,
        },
        "cpu": {
            "format": "onnx",
            "half": False,
            "dynamic": False,
        }
    })


# Default configurations
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_SANITY_CHECK_CONFIG = SanityCheckConfig()
DEFAULT_HPO_CONFIG = HPOConfig()
DEFAULT_EXPORT_CONFIG = ExportConfig()
