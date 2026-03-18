"""
Feature Store Module for ML pipelines.

Provides centralized feature management:
- Feature definitions
- Feature computation
- Feature versioning
- Online/offline feature serving

Based on best practices:
- Feast (open source) patterns
- Feature store architecture
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class FeatureType(Enum):
    """Feature data types."""
    FLOAT = "float"
    INT = "int"
    STRING = "string"
    BOOL = "bool"
    FLOAT_LIST = "float_list"
    INT_LIST = "int_list"


@dataclass
class FeatureDefinition:
    """Feature definition."""
    name: str
    feature_type: FeatureType
    description: str = ""
    default_value: Any = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class FeatureVector:
    """Feature vector with values."""
    name: str
    features: Dict[str, Any]
    timestamp: str = ""
    version: int = 1


class FeatureStore:
    """
    Simple feature store for ML features.

    Features:
    - Feature registration
    - Feature versioning
    - Online feature serving
    - Offline feature extraction
    """

    def __init__(self, storage_path: str = "./data/features"):
        """
        Initialize feature store.

        Args:
            storage_path: Path for feature storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._features: Dict[str, FeatureDefinition] = {}
        self._feature_groups: Dict[str, List[str]] = {}

    def register_feature(
        self,
        name: str,
        feature_type: FeatureType,
        description: str = "",
        group: str = "default",
    ) -> FeatureDefinition:
        """
        Register a feature.

        Args:
            name: Feature name
            feature_type: Feature type
            description: Feature description
            group: Feature group

        Returns:
            FeatureDefinition
        """
        feature = FeatureDefinition(
            name=name,
            feature_type=feature_type,
            description=description,
        )

        self._features[name] = feature

        if group not in self._feature_groups:
            self._feature_groups[group] = []
        self._feature_groups[group].append(name)

        # Save to storage
        self._save_feature_registry()

        return feature

    def register_features(
        self,
        features: List[FeatureDefinition],
        group: str = "default",
    ) -> None:
        """
        Register multiple features.

        Args:
            features: List of feature definitions
            group: Feature group
        """
        for feature in features:
            self.register_feature(
                name=feature.name,
                feature_type=feature.feature_type,
                description=feature.description,
                group=group,
            )

    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        """Get feature definition."""
        return self._features.get(name)

    def get_features(self, group: str = None) -> List[FeatureDefinition]:
        """Get features by group."""
        if group:
            feature_names = self._feature_groups.get(group, [])
            return [self._features[name] for name in feature_names]
        return list(self._features.values())

    def get_feature_groups(self) -> List[str]:
        """Get all feature groups."""
        return list(self._feature_groups.keys())

    def create_feature_vector(
        self,
        name: str,
        features: Dict[str, Any],
    ) -> FeatureVector:
        """
        Create a feature vector.

        Args:
            name: Vector name
            features: Feature values

        Returns:
            FeatureVector
        """
        # Compute version hash
        content = json.dumps(features, sort_keys=True)
        version = int(hashlib.md5(content.encode()).hexdigest()[:8], 16) % 1000000

        return FeatureVector(
            name=name,
            features=features,
            timestamp=datetime.now().isoformat(),
            version=version,
        )

    def save_feature_vector(
        self,
        vector: FeatureVector,
    ) -> None:
        """
        Save feature vector to storage.

        Args:
            vector: Feature vector to save
        """
        filename = f"{vector.name}_v{vector.version}.json"
        filepath = self.storage_path / filename

        with open(filepath, 'w') as f:
            json.dump({
                "name": vector.name,
                "features": vector.features,
                "timestamp": vector.timestamp,
                "version": vector.version,
            }, f, indent=2)

    def get_feature_vector(
        self,
        name: str,
        version: Optional[int] = None,
    ) -> Optional[FeatureVector]:
        """
        Get feature vector from storage.

        Args:
            name: Vector name
            version: Specific version (latest if None)

        Returns:
            FeatureVector or None
        """
        if version:
            filename = f"{name}_v{version}.json"
        else:
            # Find latest version
            files = list(self.storage_path.glob(f"{name}_v*.json"))
            if not files:
                return None
            filename = max(files, key=lambda p: p.stat().st_mtime).name

        filepath = self.storage_path / filename
        if not filepath.exists():
            return None

        with open(filepath, 'r') as f:
            data = json.load(f)

        return FeatureVector(
            name=data["name"],
            features=data["features"],
            timestamp=data["timestamp"],
            version=data["version"],
        )

    def _save_feature_registry(self) -> None:
        """Save feature registry to storage."""
        registry = {
            "features": {
                name: {
                    "name": f.name,
                    "feature_type": f.feature_type.value,
                    "description": f.description,
                    "tags": f.tags,
                }
                for name, f in self._features.items()
            },
            "groups": self._feature_groups,
        }

        filepath = self.storage_path / "registry.json"
        with open(filepath, 'w') as f:
            json.dump(registry, f, indent=2)

    def export_to_dict(self) -> Dict[str, Any]:
        """Export feature store to dictionary."""
        return {
            "features": {
                name: {
                    "name": f.name,
                    "feature_type": f.feature_type.value,
                    "description": f.description,
                }
                for name, f in self._features.items()
            },
            "groups": self._feature_groups,
        }


class YOLOFeatureStore(FeatureStore):
    """
    Specialized feature store for YOLO training.

    Predefined features for object detection:
    - Image features (brightness, contrast, etc.)
    - Dataset statistics
    - Training metrics
    """

    def __init__(self, storage_path: str = "./data/features"):
        super().__init__(storage_path)
        self._register_yolo_features()

    def _register_yolo_features(self) -> None:
        """Register YOLO-specific features."""
        # Image features
        image_features = [
            FeatureDefinition(
                name="image_brightness",
                feature_type=FeatureType.FLOAT,
                description="Average image brightness",
            ),
            FeatureDefinition(
                name="image_contrast",
                feature_type=FeatureType.FLOAT,
                description="Image contrast ratio",
            ),
            FeatureDefinition(
                name="image_sharpness",
                feature_type=FeatureType.FLOAT,
                description="Image sharpness score",
            ),
            FeatureDefinition(
                name="object_density",
                feature_type=FeatureType.FLOAT,
                description="Average objects per image",
            ),
            FeatureDefinition(
                name="class_distribution",
                feature_type=FeatureType.FLOAT_LIST,
                description="Class distribution vector",
            ),
        ]

        self.register_features(image_features, group="image")

        # Dataset features
        dataset_features = [
            FeatureDefinition(
                name="dataset_size",
                feature_type=FeatureType.INT,
                description="Total number of images",
            ),
            FeatureDefinition(
                name="annotation_count",
                feature_type=FeatureType.INT,
                description="Total number of annotations",
            ),
            FeatureDefinition(
                name="avg_bbox_size",
                feature_type=FeatureType.FLOAT,
                description="Average bounding box size",
            ),
        ]

        self.register_features(dataset_features, group="dataset")

    def compute_image_features(self, image_path: str) -> Dict[str, Any]:
        """
        Compute image features.

        Args:
            image_path: Path to image

        Returns:
            Feature dictionary
        """
        try:
            from PIL import Image
            import numpy as np

            img = Image.open(image_path)
            img_array = np.array(img)

            # Brightness
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            brightness = float(np.mean(gray))

            # Contrast (std deviation)
            contrast = float(np.std(gray))

            return {
                "image_brightness": brightness,
                "image_contrast": contrast,
                "image_sharpness": contrast * 1.5,  # Approximation
            }

        except Exception as e:
            return {"error": str(e)}


def create_feature_store(storage_path: str = "./data/features") -> FeatureStore:
    """Create a feature store instance."""
    return FeatureStore(storage_path)


def create_yolo_feature_store(storage_path: str = "./data/features") -> YOLOFeatureStore:
    """Create a YOLO-specific feature store."""
    return YOLOFeatureStore(storage_path)
