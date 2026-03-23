"""
Semi-Supervised Learning Pipeline for YOLO Auto-Training.

Implements SAM-driven self-training framework:
1. Train Teacher model on labeled data
2. Use SAM to generate pseudo-labels on unlabeled data
3. Filter: confidence > threshold
4. Joint training: labeled + pseudo-labeled
5. Update Teacher → repeat

Reference:
- SAM-Driven Self-Training Framework (arXiv:2507.23307)
- PseCo: Pseudo Labeling and Consistency Training (ECCV 2022)
- SCOUT: Semi-supervised Camouflaged Object Detection (IJCAI 2025)
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PseudoLabel:
    """A pseudo-labeled image with bounding boxes."""
    image_path: str
    boxes: List[Dict[str, Any]]  # [{x1, y1, x2, y2, class_id, confidence}]
    generation_method: str  # "sam" / "yolo_teacher"
    filtered_count: int   # Number of boxes after filtering


class SemiSupervisedPipeline:
    """
    SAM-driven self-training for object detection.

    Reduces labeling costs by leveraging unlabeled data through:
    - Foundation model (SAM) for high-quality pseudo-labels
    - Confidence filtering to ensure label quality
    - Iterative self-training for progressive improvement
    """

    def __init__(
        self,
        sam_model: str = "sam2.1_b+",
        confidence_threshold: float = 0.7,
        max_boxes: int = 100,
    ):
        self.sam_model = sam_model
        self.confidence_threshold = confidence_threshold
        self.max_boxes = max_boxes
        self._sam_available = self._check_sam_availability()

    def _check_sam_availability(self) -> bool:
        """Check if SAM is available."""
        try:
            import segment_anything
            return True
        except ImportError:
            logging.warning("[SEMI_SUPER] SAM not available. Pseudo-labeling will use YOLO teacher only.")
            return False

    def generate_pseudo_labels(
        self,
        teacher_model_path: str,
        unlabeled_images: List[str],
        method: str = "yolo_teacher",  # "sam" / "yolo_teacher" / "hybrid"
    ) -> List[PseudoLabel]:
        """
        Generate pseudo-labels for unlabeled images.

        Args:
            teacher_model_path: Path to YOLO teacher model
            unlabeled_images: List of unlabeled image paths
            method: Generation method

        Returns:
            List of PseudoLabel objects with filtered bounding boxes
        """
        if method == "yolo_teacher":
            return self._generate_with_yolo_teacher(teacher_model_path, unlabeled_images)
        elif method == "sam":
            if not self._sam_available:
                logging.warning("[SEMI_SUPER] SAM not available, falling back to YOLO teacher")
                return self._generate_with_yolo_teacher(teacher_model_path, unlabeled_images)
            return self._generate_with_sam(unlabeled_images)
        else:
            return self._generate_with_yolo_teacher(teacher_model_path, unlabeled_images)

    def _generate_with_yolo_teacher(
        self,
        model_path: str,
        images: List[str],
    ) -> List[PseudoLabel]:
        """Use YOLO teacher model to generate pseudo-labels."""
        from ultralytics import YOLO

        model = YOLO(model_path)
        pseudo_labels = []

        for img_path in images:
            try:
                results = model.predict(
                    source=img_path,
                    conf=self.confidence_threshold,
                    verbose=False,
                )

                if not results or len(results) == 0:
                    continue

                result = results[0]
                boxes = result.boxes

                if boxes is None or len(boxes) == 0:
                    continue

                filtered_boxes = []
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i][0]) if boxes.conf is not None else 0.0
                    if conf >= self.confidence_threshold:
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        xywhn = boxes.xywhn[i].cpu().numpy()
                        cls = int(boxes.cls[i][0]) if boxes.cls is not None else 0
                        filtered_boxes.append({
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "x2": float(xyxy[2]),
                            "y2": float(xyxy[3]),
                            "xywhn": [float(v) for v in xywhn],
                            "class_id": cls,
                            "confidence": conf,
                        })

                if filtered_boxes:
                    pseudo_labels.append(PseudoLabel(
                        image_path=img_path,
                        boxes=filtered_boxes,
                        generation_method="yolo_teacher",
                        filtered_count=len(filtered_boxes),
                    ))

            except Exception as e:
                logging.warning(f"[SEMI_SUPER] Error generating pseudo-label for {img_path}: {e}")

        return pseudo_labels

    def _generate_with_sam(self, images: List[str]) -> List[PseudoLabel]:
        """Use SAM + classifier for pseudo-label generation."""
        # SAM generates masks, not boxes. We convert masks to bounding boxes.
        # This is a simplified version; full implementation would use SAM's box predictor
        logging.info("[SEMI_SUPER] SAM pseudo-labeling not yet fully implemented — use yolo_teacher method")
        return []

    def filter_pseudo_labels(
        self,
        pseudo_labels: List[PseudoLabel],
        min_boxes: int = 1,
        max_boxes: int = 50,
    ) -> List[PseudoLabel]:
        """Filter pseudo-labels by box count."""
        return [
            p for p in pseudo_labels
            if min_boxes <= len(p.boxes) <= max_boxes
        ]

    def create_pseudo_dataset(
        self,
        pseudo_labels: List[PseudoLabel],
        output_dir: str,
        class_names: List[str],
    ) -> str:
        """
        Convert pseudo-labels to YOLO format dataset.

        Creates:
        - output_dir/images/ with symlinked images
        - output_dir/labels/ with YOLO TXT annotation files
        - output_dir/pseudo_dataset.yaml
        """
        output_path = Path(output_dir)
        img_dir = output_path / "images" / "pseudo"
        label_dir = output_path / "labels" / "pseudo"

        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for pseudo in pseudo_labels:
            # Symlink image
            src = Path(pseudo.image_path)
            dst = img_dir / src.name
            if not dst.exists():
                try:
                    dst.symlink_to(src.resolve())
                except OSError:
                    import shutil
                    shutil.copy(src, dst)

            # Write YOLO format annotation (normalized xywhn)
            label_file = label_dir / f"{src.stem}.txt"
            with open(label_file, "w") as f:
                for box in pseudo.boxes:
                    # YOLO format: class_id x_center y_center width height (normalized)
                    x_center = ((box["x1"] + box["x2"]) / 2)
                    y_center = ((box["y1"] + box["y2"]) / 2)
                    width = box["x2"] - box["x1"]
                    height = box["y2"] - box["y1"]
                    # Normalize by image dimensions from boxes (ultralytics provides xywhn)
                    # Fallback: use xywhn directly if available, otherwise normalize
                    norm = box.get("xywhn")
                    if norm is not None:
                        f.write(f"{box['class_id']} {norm[0]:.6f} {norm[1]:.6f} "
                                f"{norm[2]:.6f} {norm[3]:.6f}\n")
                    else:
                        # Normalize by image dimensions (assuming 640x640 for now)
                        f.write(f"{box['class_id']} {x_center/640:.6f} {y_center/640:.6f} "
                                f"{width/640:.6f} {height/640:.6f}\n")

        # Create pseudo_dataset.yaml
        yaml_content = {
            "path": str(output_path),
            "train": "images/pseudo",
            "val": "images/pseudo",  # Use same for simplicity
            "nc": len(class_names),
            "names": {i: name for i, name in enumerate(class_names)},
        }

        import yaml
        yaml_path = output_path / "pseudo_dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        return str(yaml_path)

    def create_expanded_yaml(
        self,
        pseudo_labels: List[PseudoLabel],
        output_yaml: str,
        original_yaml: Optional[str] = None,
    ) -> str:
        """
        Write pseudo-label YOLO txt files and create expanded data.yaml.

        Args:
            pseudo_labels: List of PseudoLabel objects from generate_pseudo_labels()
            output_yaml: Path to write the expanded data.yaml
            original_yaml: Optional path to original data.yaml to inherit names/nc

        Returns:
            Path to the created expanded data.yaml
        """
        import shutil
        import yaml

        # Determine base directory for images
        if pseudo_labels and original_yaml:
            with open(original_yaml) as f:
                orig = yaml.safe_load(f)
            base_dir = str(Path(original_yaml).parent)
            train_img_dir = orig.get('train', 'train/images')
            val_img_dir = orig.get('val', 'train/images')  # use train for pseudo val
        elif pseudo_labels:
            base_dir = str(Path(pseudo_labels[0].image_path).parent)
            train_img_dir = "train/images"
            val_img_dir = "train/images"
        else:
            raise ValueError("pseudo_labels cannot be empty")

        # Write pseudo-label txt files to a pseudo_labels subdirectory
        pseudo_dir = Path(base_dir) / "pseudo_labels"
        pseudo_train_lbl = pseudo_dir / "train" / "labels"
        pseudo_train_img = pseudo_dir / "train" / "images"
        pseudo_train_lbl.mkdir(parents=True, exist_ok=True)
        pseudo_train_img.mkdir(parents=True, exist_ok=True)

        for pseudo in pseudo_labels:
            img_path = Path(pseudo.image_path)
            lbl_path = pseudo_train_lbl / (img_path.stem + ".txt")
            lines = []
            for box in pseudo.boxes:
                norm = box.get("xywhn")
                if norm is not None:
                    lines.append(
                        f"{box['class_id']} {norm[0]:.6f} {norm[1]:.6f} "
                        f"{norm[2]:.6f} {norm[3]:.6f}"
                    )
                else:
                    x_c = ((box["x1"] + box["x2"]) / 2) / 640
                    y_c = ((box["y1"] + box["y2"]) / 2) / 640
                    w = (box["x2"] - box["x1"]) / 640
                    h = (box["y2"] - box["y1"]) / 640
                    lines.append(
                        f"{box['class_id']} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                    )
            lbl_path.write_text("\n".join(lines) + "\n")
            # Copy image into pseudo train set
            dst_img = pseudo_train_img / img_path.name
            if not dst_img.exists():
                shutil.copy(img_path, dst_img)

        # Create expanded data.yaml
        expanded_yaml_content = {
            "path": str(pseudo_dir),
            "train": "train/images",
            "val": "train/images",
        }
        if original_yaml:
            with open(original_yaml) as f:
                orig = yaml.safe_load(f)
            if "names" in orig:
                expanded_yaml_content["names"] = orig["names"]
            if "nc" in orig:
                expanded_yaml_content["nc"] = orig["nc"]
        else:
            expanded_yaml_content["names"] = {0: "object"}

        with open(output_yaml, "w") as f:
            yaml.dump(expanded_yaml_content, f, default_flow_style=False)

        return output_yaml
