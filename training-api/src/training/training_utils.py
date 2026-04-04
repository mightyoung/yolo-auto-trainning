"""
Shared training utilities, types, and helper functions.
Location: training-api/src/training/training_utils.py

Contains:
- TrainingCancelled exception
- TrainingResult, DatasetDistributionResult dataclasses
- setup_gpu_memory, cleanup_gpu_memory
- validate_dataset_distribution
"""

from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
import os

import torch


class TrainingCancelled(Exception):
    """Raised when training is cancelled via the progress callback."""
    pass


@dataclass
class TrainingResult:
    """Training result container."""
    status: str
    model_path: Optional[Path] = None
    metrics: Optional[Dict[str, float]] = None
    best_params: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    early_stopped: bool = False


@dataclass
class DatasetDistributionResult:
    """Result of dataset distribution validation."""
    train_median_area: float
    val_median_area: float
    ratio: float  # val/train ratio
    status: str   # "ok", "warning", "critical"
    train_box_count: int
    val_box_count: int
    train_image_count: int
    val_image_count: int
    message: str


def setup_gpu_memory() -> None:
    """Setup GPU memory management to prevent OOM errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_per_process_memory_fraction(0.8, device=i)
            except Exception:
                pass  # Gracefully handle if not supported


def cleanup_gpu_memory() -> None:
    """Cleanup GPU memory after training."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def validate_dataset_distribution(data_yaml_path: Path) -> DatasetDistributionResult:
    """Validate train/val distribution balance in a YOLO dataset.

    Analyzes bounding box area distributions to detect train/val mismatch
    that causes model collapse (e.g., val boxes 8x larger than train).
    """
    yaml_path = Path(data_yaml_path)
    if not yaml_path.exists():
        return DatasetDistributionResult(
            train_median_area=0.0, val_median_area=0.0,
            ratio=0.0, status="warning",
            train_box_count=0, val_box_count=0,
            train_image_count=0, val_image_count=0,
            message=f"data.yaml not found at {data_yaml_path}"
        )

    import yaml
    try:
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
    except Exception as e:
        return DatasetDistributionResult(
            train_median_area=0.0, val_median_area=0.0,
            ratio=0.0, status="warning",
            train_box_count=0, val_box_count=0,
            train_image_count=0, val_image_count=0,
            message=f"Failed to parse data.yaml: {e}"
        )

    # Resolve path relative to yaml file's directory (not cwd)
    # path: . means "the directory containing this yaml file"
    if yaml_data.get("path") is None:
        dataset_root = yaml_path.parent
    else:
        dataset_root = (yaml_path.parent / yaml_data["path"]).resolve()
    train_images = dataset_root / yaml_data.get("train", "train")
    val_images = dataset_root / yaml_data.get("val", "val")

    def _get_label_dir(images_path: Path) -> Path:
        s = str(images_path)
        s_norm = s.replace("\\", "/")
        if s_norm.endswith("/images"):
            return Path(s[:-7] + "/labels")
        last_slash = max(s_norm.rfind("/"), s_norm.rfind("\\"))
        component = s_norm[last_slash + 1:]
        if "images" in component:
            new_component = component.replace("images", "labels")
            return Path(s[:last_slash + 1] + new_component)
        return Path(s + "_labels")

    def _get_images_dir(labels_path: Path) -> Path:
        s = str(labels_path)
        s_norm = s.replace("\\", "/")
        if s_norm.endswith("/labels"):
            return Path(s[:-7] + "/images")
        last_slash = max(s_norm.rfind("/"), s_norm.rfind("\\"))
        component = s_norm[last_slash + 1:]
        if "labels" in component:
            new_component = component.replace("labels", "images")
            return Path(s[:last_slash + 1] + new_component)
        return Path(s + "_images")

    def _parse_label_areas(label_dir: Path) -> tuple:
        label_dir = Path(label_dir)
        if not label_dir.exists():
            return [], 0
        all_areas = []
        for label_file in label_dir.glob("*.txt"):
            try:
                with open(label_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                w, h = float(parts[3]), float(parts[4])
                                all_areas.append(w * h)
                            except ValueError:
                                pass
            except Exception:
                pass
        img_dir = _get_images_dir(label_dir)
        if img_dir.exists():
            image_count = len([f for f in img_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        else:
            image_count = 0
        return all_areas, image_count

    def _median(lst: list) -> float:
        if not lst:
            return 0.0
        sorted_lst = sorted(lst)
        n = len(sorted_lst)
        if n % 2 == 0:
            return (sorted_lst[n // 2 - 1] + sorted_lst[n // 2]) / 2
        return sorted_lst[n // 2]

    train_labels = _get_label_dir(train_images)
    val_labels = _get_label_dir(val_images)
    train_areas, train_imgs = _parse_label_areas(train_labels)
    val_areas, val_imgs = _parse_label_areas(val_labels)

    train_med = _median(train_areas)
    val_med = _median(val_areas)

    if train_med > 0 and val_med > 0:
        ratio = val_med / train_med
    elif val_med > 0:
        ratio = float('inf')
    else:
        ratio = 0.0

    if ratio == float('inf') or ratio == 0.0:
        status = "critical"
        message = f"No train labels found. train_imgs={train_imgs}, val_imgs={val_imgs}"
    elif ratio > 5.0:
        status = "critical"
        message = (
            f"CRITICAL: val median box area ({val_med:.4f}) is {ratio:.1f}x "
            f"larger than train median ({train_med:.4f}). "
            f"This WILL cause mAP50 collapse. Use stratified train/val split."
        )
    elif ratio > 3.0:
        status = "warning"
        message = (
            f"WARNING: val median box area ({val_med:.4f}) is {ratio:.1f}x "
            f"larger than train median ({train_med:.4f}). "
            f"Consider using stratified train/val split."
        )
    else:
        status = "ok"
        message = f"Distribution OK: val/train ratio = {ratio:.2f}x"

    return DatasetDistributionResult(
        train_median_area=train_med,
        val_median_area=val_med,
        ratio=ratio,
        status=status,
        train_box_count=len(train_areas),
        val_box_count=len(val_areas),
        train_image_count=train_imgs,
        val_image_count=val_imgs,
        message=message,
    )
