"""
Training configuration based on Ultralytics official best practices.

Reference: https://docs.ultralytics.com/usage/cfg/
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


# NOTE: LRSchedulerConfig MUST be defined BEFORE TrainingConfig to avoid
# forward-reference NameError at class-definition time (lambda: LRSchedulerConfig()
# in TrainingConfig.lr_scheduler would otherwise fail since LRSchedulerConfig
# hasn't been defined yet).
@dataclass
class LRSchedulerConfig:
    """Learning rate scheduler configuration.

    Supports multiple scheduling strategies:
    - linear: Linear decay from lr0 to lr0*lrf
    - cosine: Cosine annealing from lr0 to lr0*lrf
    - exponential: Exponential decay
    - constant: No scheduling (fixed lr0)

    RAFS (Rectified Adam + Warmup + Flat + Sharp):
    - Warmup: lr builds up over warmup_epochs (or warmup_ratio * total_epochs)
    - Flat: Constant lr during middle epochs
    - Sharp: Sharp decay to lr0*lrf at end

    Reference: Ultralytics official docs - lr0, lrf, warmup_epochs
    Inspired by: autoresearch WARMUP_RATIO/WARMDOWN_RATIO patterns
    """
    type: str = "linear"           # Scheduler type
    lrf: float = 0.01              # Final LR factor (final_lr = lr0 * lrf)
    warmup_epochs: float = 3.0     # Warmup epochs (overridden by CurriculumStage.warmup_ratio)
    warmup_momentum: float = 0.8  # Warmup momentum
    warmup_bias_lr: float = 0.1   # Warmup bias LR

    # Cosine annealing specific
    # NOTE: cosine_min_lr was removed from computation in runner.py
    # lrf is now used directly (final LR = lr0 * lrf, e.g. 0.01 * 0.01 = 1e-4)
    # This prevents the broken case where min_lr = lr0 * 1e-10
    cosine_min_lr: float = 1e-6    # Deprecated: kept for compat, no longer used in lrf calc

    # RAFS (Sharp) specific
    sharp_epochs: int = 10         # Number of sharp-decay epochs at end


@dataclass
class TrainingConfig:
    """Training configuration based on Ultralytics official defaults."""

    # Model settings
    model: str = "yolo11m"  # 支持 yolo26n/s/m/l/x 和 yolo11n/s/m/l/x
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
    mixup: float = 0.1          # Mixup
    copy_paste: float = 0.1     # Copy-paste
    copy_paste_mode: str = "flip"  # Copy-paste mode

    # Training settings
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "cuda:0"      # GPU device(s) for training: "cuda:0", "0,1", "0,1,2"
    num_gpus: int = 1           # Number of GPUs for multi-GPU DDP training.
                                  # Ultralytics auto-distributes batch across GPUs.
                                  # Use 0 for CPU, 1 for single GPU, 2+ for multi-GPU DDP.
    patience: int = 100         # Early stopping
    warmup_epochs: float = 3.0  # Warmup (supports RAFS: warmup + flat + sharp decay)
    warmup_momentum: float = 0.8  # Warmup momentum for RAFS
    warmup_bias_lr: float = 0.1  # Warmup bias LR for RAFS
    close_mosaic: int = 10      # Disable mosaic in last N epochs

    # Optimization
    optimizer: str = "SGD"
    amp: bool = True            # AMP: Automatic Mixed Precision（已启用，GPU显存占用降低约50%）
    cache: bool = False         # Cache images
    resume_checkpoint: Optional[str] = None  # Checkpoint path for resuming training (last.pt)
    # lr_scheduler must be initialized in __post_init__ to avoid forward-reference NameError
    # (LRSchedulerConfig is defined before this class so it's safe to instantiate here)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YOLO training."""
        return {
            "lr0": self.lr0,
            "lrf": self.lr_scheduler.lrf,
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
            "warmup_epochs": self.lr_scheduler.warmup_epochs,
            "warmup_momentum": self.lr_scheduler.warmup_momentum,
            "warmup_bias_lr": self.lr_scheduler.warmup_bias_lr,
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
    formats: List[str] = field(default_factory=lambda: ["onnx"])
    opset: int = 13
    half: bool = True        # FP16
    dynamic: bool = False
    simplify: bool = True
    int8_calibration_images: int = 1000  # Number of images for INT8 calibration

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
        },
        "tflite": {
            "format": "tflite",
            "half": False,
            "dynamic": False,
        }
    })


# NOTE: LRSchedulerConfig is defined at the TOP of this file (before TrainingConfig)
# to avoid forward-reference NameError. Do NOT define it here again.


@dataclass
class AugmentationPreset:
    """Augmentation preset configuration with all parameters for mAP 90%+ target."""

    name: str
    mosaic: float
    mixup: float
    copy_paste: float
    copy_paste_mode: str = "flip"
    # Geometric augmentation
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    # Color augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4


# Preset templates following Ultralytics best practices
AUGMENTATION_PRESETS: Dict[str, AugmentationPreset] = {
    # "fast": Minimal augmentation for quick sanity checks
    "fast": AugmentationPreset(
        name="fast",
        mosaic=1.0, mixup=0.0, copy_paste=0.0,
        degrees=0.0, translate=0.1, scale=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    ),
    # "balanced": Default for moderate mAP targets
    "balanced": AugmentationPreset(
        name="balanced",
        mosaic=1.0, mixup=0.1, copy_paste=0.1,
        degrees=0.0, translate=0.1, scale=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    ),
    # "strong": Optimized for mAP 90%+ — mosaic/mixup/copy-paste + geometric + color jitter
    "strong": AugmentationPreset(
        name="strong",
        mosaic=1.0, mixup=0.3, copy_paste=0.4,
        degrees=15.0, translate=0.2, scale=0.7,
        shear=2.0, perspective=0.0005,
        flipud=0.05, fliplr=0.5,
        hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
    ),
}


@dataclass
class PlateauBreakingConfig:
    """Configuration for dynamic plateau detection and breaking during training.

    When mAP50 stagnates for N consecutive epochs, the system automatically
    applies progressively aggressive strategies:
      Level 1: Reduce LR by factor, reset optimizer momentum
      Level 2: Boost augmentation (mixup, copy_paste, degrees)
      Level 3: Trigger data expansion via ActiveLearningPipeline
    """

    # Plateau detection
    enabled: bool = True
    window: int = 10          # Number of epochs to look back for trend
    min_improvement: float = 0.002  # Minimum mAP50 improvement per window to NOT be plateau
    min_epochs_before_trigger: int = 30  # Don't trigger before this epoch

    # Level 1: LR decay
    lr_reduction_factor: float = 0.5   # Multiply LR by this factor
    lr_reduction_max_times: int = 3    # Max LR reductions
    min_lr: float = 1e-6

    # Level 2: Augmentation boost
    augmentation_boost_epochs: int = 15  # Epochs to run with boosted augmentation
    boosted_mixup: float = 0.3          # Up from default 0.1
    boosted_copy_paste: float = 0.4      # Up from default 0.1
    boosted_degrees: float = 15.0         # Add rotation augmentation
    boosted_translate: float = 0.2       # Up from default 0.1
    boosted_scale: float = 0.7           # Up from default 0.5

    # Level 3: Auto data expansion (triggers ActiveLearningPipeline)
    auto_expand_data: bool = True
    expansion_target_map: float = 0.90   # Only expand if target mAP >= this
    max_expansion_rounds: int = 2        # Max ActiveLearning iterations


# Default configurations
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_SANITY_CHECK_CONFIG = SanityCheckConfig()
DEFAULT_HPO_CONFIG = HPOConfig()
DEFAULT_EXPORT_CONFIG = ExportConfig()
DEFAULT_PLATEAU_CONFIG = PlateauBreakingConfig()
