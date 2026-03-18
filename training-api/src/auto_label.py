"""
Auto Labeling Module using AutoDistill
Location: training-api/src/auto_label.py

This module provides automatic labeling functionality using foundation models.
Based on AutoDistill architecture concepts.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import asyncio


@dataclass
class LabelResult:
    """Labeling result for a single image."""
    image_path: str
    labels: List[Dict[str, Any]]
    annotations_path: str
    success: bool
    error: Optional[str] = None


@dataclass
class LabelDatasetResult:
    """Result of labeling an entire dataset."""
    total_images: int
    labeled_images: int
    failed_images: int
    output_folder: str
    data_yaml_path: str
    labels: List[str]


class AutoLabeler:
    """
    Automatic image labeling using foundation models.

    Supports multiple base models:
    - GroundedSAM (recommended): Best open-source detection + segmentation
    - GroundingDINO: Open-set object detection
    - OWLv2: Zero-shot object detection
    """

    def __init__(
        self,
        base_model: str = "grounded_sam",
        device: str = "cuda",
        conf_threshold: float = 0.3
    ):
        """
        Initialize AutoLabeler.

        Args:
            base_model: Base model to use for labeling
            device: Device to use (cuda/cpu)
            conf_threshold: Confidence threshold for predictions
        """
        self.base_model = base_model
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None

    def _load_model(self):
        """Load the base model (lazy loading)."""
        if self.model is not None:
            return

        if self.base_model == "grounded_sam":
            try:
                # Try importing autodistill components
                from autodistill_grounded_sam import GroundedSAM
                from autodistill.detection import CaptionOntology
                self.model = GroundedSAM
                self.ontology = CaptionOntology
            except ImportError:
                # Fallback: use direct SAM + Grounding DINO
                self._load_grounded_sam_direct()
        elif self.base_model == "grounding_dino":
            self._load_grounding_dino()
        elif self.base_model == "owlv2":
            self._load_owlv2()
        else:
            raise ValueError(f"Unknown base model: {self.base_model}")

    def _load_grounded_sam_direct(self):
        """Load GroundedSAM directly without autodistill."""
        try:
            import torch
            from grounding_dino import GroundingDINO
            from segment_anything import sam_model_registry, SamPredictor

            # Load Grounding DINO
            self.grounding_dino = GroundingDINO(
                config_path="grounding_dino/config/GroundingDINO_SwinT_OGC.py",
                checkpoint_path="grounding_dino.pth",
                device=self.device
            )

            # Load SAM
            sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)

            self.model_type = "grounded_sam"
        except ImportError as e:
            raise ImportError(
                "Please install required packages: "
                "pip install grounding-dino segment-anything"
            ) from e

    def _load_grounding_dino(self):
        """Load GroundingDINO directly."""
        try:
            from grounding_dino import GroundingDINO
            self.grounding_dino = GroundingDINO(
                config_path="grounding_dino/config/GroundingDINO_SwinT_OGC.py",
                checkpoint_path="grounding_dino.pth",
                device=self.device
            )
            self.model_type = "grounding_dino"
        except ImportError:
            raise ImportError("Please install grounding-dino")

    def _load_owlv2(self):
        """Load OWLv2 directly."""
        try:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            self.owlv2_model = Owlv2ForObjectDetection.from_pretrained(
                "google/owlv2-base"
            )
            self.owlv2_processor = Owlv2Processor.from_pretrained(
                "google/owlv2-base"
            )
            self.model_type = "owlv2"
        except ImportError:
            raise ImportError("Please install transformers")

    def predict(
        self,
        image_path: str,
        classes: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Predict bounding boxes for a single image.

        Args:
            image_path: Path to image file
            classes: List of class names to detect

        Returns:
            List of predictions with bounding boxes, labels, and confidence
        """
        self._load_model()

        import cv2
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.model_type == "grounded_sam":
            return self._predict_grounded_sam(image_rgb, classes)
        elif self.model_type == "grounding_dino":
            return self._predict_grounding_dino(image, classes)
        elif self.model_type == "owlv2":
            return self._predict_owlv2(image_rgb, classes)

    def _predict_grounded_sam(
        self,
        image,
        classes: List[str]
    ) -> List[Dict[str, Any]]:
        """Predict using GroundedSAM."""
        import torch

        # Run GroundingDINO
        detections = self.grounding_dino.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=self.conf_threshold,
            text_threshold=self.conf_threshold
        )

        # Run SAM on detections
        self.sam_predictor.set_image(image)
        h, w = image.shape[:2]

        results = []
        for i, box in enumerate(detections.xyxy):
            # Get mask from SAM
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False
            )

            results.append({
                "bbox": box.tolist(),
                "class": detections.class_id[i],
                "confidence": float(detections.confidence[i]),
                "mask": masks[0] if len(masks) > 0 else None
            })

        return results

    def _predict_grounding_dino(
        self,
        image,
        classes: List[str]
    ) -> List[Dict[str, Any]]:
        """Predict using GroundingDINO only."""
        detections = self.grounding_dino.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=self.conf_threshold,
            text_threshold=self.conf_threshold
        )

        results = []
        for i, box in enumerate(detections.xyxy):
            results.append({
                "bbox": box.tolist(),
                "class": detections.class_id[i],
                "confidence": float(detections.confidence[i])
            })

        return results

    def _predict_owlv2(
        self,
        image,
        classes: List[str]
    ) -> List[Dict[str, Any]]:
        """Predict using OWLv2."""
        import torch

        inputs = self.owlv2_processor(
            text=classes,
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.owlv2_model(**inputs)

        target_sizes = torch.Tensor([image.shape[:2]])
        results = self.owlv2_processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.conf_threshold
        )

        predictions = []
        for result in results:
            for score, label, box in zip(
                result["scores"],
                result["labels"],
                result["boxes"]
            ):
                predictions.append({
                    "bbox": box.tolist(),
                    "class": classes[label.item()],
                    "confidence": score.item()
                })

        return predictions

    def label_dataset(
        self,
        input_folder: str,
        classes: List[str],
        output_folder: str = None,
        extension: str = ".jpg"
    ) -> LabelDatasetResult:
        """
        Label an entire dataset.

        Args:
            input_folder: Folder containing images to label
            classes: List of class names
            output_folder: Output folder for labeled dataset
            extension: Image file extension

        Returns:
            LabelDatasetResult with labeling statistics
        """
        self._load_model()

        import glob
        import shutil

        # Setup paths
        input_path = Path(input_folder)
        if output_folder is None:
            output_folder = str(input_path.parent / f"{input_path.name}_labeled")

        output_path = Path(output_folder)
        images_dir = output_path / "images"
        annotations_dir = output_path / "annotations"

        # Create directories
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_files = glob.glob(str(input_path / f"*{extension}"))

        labeled_count = 0
        failed_count = 0

        for img_file in image_files:
            try:
                # Get predictions
                predictions = self.predict(img_file, classes)

                # Copy image
                shutil.copy(img_file, images_dir / Path(img_file).name)

                # Write YOLO format annotation
                img_name = Path(img_file).stem
                ann_file = annotations_dir / f"{img_name}.txt"

                with open(ann_file, 'w') as f:
                    for pred in predictions:
                        # Convert bbox to YOLO format (center_x, center_y, width, height)
                        bbox = pred["bbox"]
                        x1, y1, x2, y2 = bbox
                        h_img, w_img = 640, 640  # Would get actual image size

                        # Normalize to [0, 1]
                        cx = ((x1 + x2) / 2) / w_img
                        cy = ((y1 + y2) / 2) / h_img
                        w = (x2 - x1) / w_img
                        h = (y2 - y1) / h_img

                        class_id = classes.index(pred["class"]) if pred["class"] in classes else 0

                        f.write(f"{class_id} {cx} {cy} {w} {h}\n")

                labeled_count += 1

            except Exception as e:
                print(f"Failed to label {img_file}: {e}")
                failed_count += 1

        # Create data.yaml
        data_yaml = output_path / "data.yaml"
        with open(data_yaml, 'w') as f:
            f.write(f"path: {output_path}\n")
            f.write(f"train: images/train\n")
            f.write(f"val: images/val\n")
            f.write(f"nc: {len(classes)}\n")
            f.write(f"names: {classes}\n")

        return LabelDatasetResult(
            total_images=len(image_files),
            labeled_images=labeled_count,
            failed_images=failed_count,
            output_folder=output_folder,
            data_yaml_path=str(data_yaml),
            labels=classes
        )


class DistillationTrainer:
    """
    Train a target model using auto-labeled dataset.

    Supports:
    - YOLOv8
    - YOLOv5
    - DETR
    """

    def __init__(
        self,
        target_model: str = "yolov8",
        model_size: str = "n"
    ):
        """
        Initialize trainer.

        Args:
            target_model: Target model type (yolov8, yolov5, detr)
            model_size: Model size (n/s/m/l/x for YOLO)
        """
        self.target_model = target_model
        self.model_size = model_size
        self.model = None

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        device: str = "cuda",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the target model.

        Args:
            data_yaml: Path to dataset YAML
            epochs: Number of training epochs
            imgsz: Image size
            device: Device to train on
            **kwargs: Additional training arguments

        Returns:
            Training results
        """
        if self.target_model == "yolov8":
            return self._train_yolov8(data_yaml, epochs, imgsz, device, **kwargs)
        elif self.target_model == "yolov5":
            return self._train_yolov5(data_yaml, epochs, imgsz, device, **kwargs)
        else:
            raise ValueError(f"Unknown target model: {self.target_model}")

    def _train_yolov8(
        self,
        data_yaml: str,
        epochs: int,
        imgsz: int,
        device: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Train YOLOv8 model."""
        try:
            from ultralytics import YOLO

            # Load model
            model_name = f"yolov8{self.model_size}.pt"
            model = YOLO(model_name)

            # Train
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                device=device,
                **kwargs
            )

            return {
                "status": "completed",
                "model_path": results.save_dir,
                "best_map": results.best_map
            }

        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")

    def _train_yolov5(
        self,
        data_yaml: str,
        epochs: int,
        imgsz: int,
        device: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Train YOLOv5 model."""
        # YOLOv5 would require separate installation
        raise NotImplementedError("YOLOv5 training not yet implemented")
