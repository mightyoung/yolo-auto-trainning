"""Auto-generate optimal inference configs for edge devices.

Generates device-specific runtime configurations (batch size, precision,
stream count, recommended format) for efficient edge inference.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class EdgeProfileGenerator:
    """Generate device-specific inference configurations."""

    # Known device profiles with recommended runtime parameters
    DEVICE_PROFILES: Dict[str, Dict[str, Any]] = {
        "jetson_orin": {
            "batch_size": 4,
            "stream_count": 4,
            "workspace_mb": 1024,
            "recommended_format": "engine-fp16",
            "fallback_formats": ["engine-int8", "onnx"],
            "precision": "fp16",
            "dynamic_batch": True,
            "coreml_compatible": False,
            "max_workspace_gb": 8,
            "dla_core": 0,  # DLA core 0 on Orin
        },
        "jetson_orin_nx": {
            "batch_size": 4,
            "stream_count": 4,
            "workspace_mb": 1024,
            "recommended_format": "engine-fp16",
            "fallback_formats": ["engine-int8", "onnx"],
            "precision": "fp16",
            "dynamic_batch": True,
            "coreml_compatible": False,
            "max_workspace_gb": 8,
        },
        "jetson_tx2": {
            "batch_size": 2,
            "stream_count": 2,
            "workspace_mb": 512,
            "recommended_format": "engine-fp16",
            "fallback_formats": ["onnx"],
            "precision": "fp16",
            "dynamic_batch": False,
            "coreml_compatible": False,
            "max_workspace_gb": 4,
            "dla_core": 0,
        },
        "jetson_nano": {
            "batch_size": 1,
            "stream_count": 1,
            "workspace_mb": 256,
            "recommended_format": "engine-fp16",
            "fallback_formats": ["onnx"],
            "precision": "fp16",
            "dynamic_batch": False,
            "coreml_compatible": False,
            "max_workspace_gb": 2,
        },
        "rk3588": {
            "batch_size": 4,
            "stream_count": 2,
            "workspace_mb": 512,
            "recommended_format": "onnx",
            "fallback_formats": ["engine-fp16", "engine-int8"],
            "precision": "fp16",
            "dynamic_batch": True,
            "coreml_compatible": False,
            "max_workspace_gb": 4,
            "note": "RKNPU accelerator — use ONNX with rknn-toolkit for NPU inference",
        },
        "mobile": {
            "batch_size": 1,
            "stream_count": 1,
            "workspace_mb": 128,
            "recommended_format": "tflite",
            "fallback_formats": ["onnx"],
            "precision": "int8",
            "dynamic_batch": False,
            "coreml_compatible": True,
            "max_workspace_gb": 1,
        },
        "edge_tpu": {
            "batch_size": 1,
            "stream_count": 1,
            "workspace_mb": 256,
            "recommended_format": "tflite",
            "fallback_formats": ["onnx"],
            "precision": "int8",
            "dynamic_batch": False,
            "coreml_compatible": False,
            "max_workspace_gb": 1,
            "note": "Use edgetpu_compiler to compile .tflite for Coral TPU",
        },
        "generic": {
            "batch_size": 1,
            "stream_count": 1,
            "workspace_mb": 256,
            "recommended_format": "onnx",
            "fallback_formats": [],
            "precision": "fp32",
            "dynamic_batch": False,
            "coreml_compatible": False,
            "max_workspace_gb": 2,
        },
    }

    # Inference config templates per format
    FORMAT_CONFIGS: Dict[str, Dict[str, Any]] = {
        "engine-fp16": {
            "half": True,
            "int8": False,
            "dynamic": True,
            "workspace_mb": 1024,
        },
        "engine-int8": {
            "half": False,
            "int8": True,
            "dynamic": True,
            "workspace_mb": 1024,
            "calibration_images": 1000,
        },
        "onnx": {
            "half": False,
            "int8": False,
            "dynamic": False,
            "opset": 13,
        },
        "tflite": {
            "half": False,
            "int8": False,
            "optimize": True,
        },
    }

    def get_profile(self, device: str) -> Dict[str, Any]:
        """Get the base profile for a device, falling back to 'generic'."""
        profile = self.DEVICE_PROFILES.get(device, self.DEVICE_PROFILES["generic"])
        if device not in self.DEVICE_PROFILES:
            logger.warning(f"[EdgeProfileGenerator] Unknown device '{device}', using generic profile")
        return profile

    def generate_config(
        self,
        device: str,
        model_path: str,
        imgsz: int = 640,
    ) -> Dict[str, Any]:
        """Generate optimal inference configuration for target device.

        Args:
            device: Target device name (e.g. 'jetson_orin', 'rk3588', 'mobile')
            model_path: Path to the model file (used for format inference)
            imgsz: Target image size (default 640)

        Returns:
            Complete device-specific inference configuration
        """
        profile = self.get_profile(device)
        recommended_fmt = profile["recommended_format"]
        fmt_config = self.FORMAT_CONFIGS.get(
            recommended_fmt, self.FORMAT_CONFIGS["onnx"]
        )

        # Infer model format from path if possible
        model_ext = ""
        if model_path:
            model_ext = model_path.rsplit(".", 1)[-1].lower() if "." in model_path else ""

        config: Dict[str, Any] = {
            "device": device,
            "model_path": model_path,
            "model_format_detected": model_ext or "unknown",
            # Runtime parameters
            "batch_size": profile["batch_size"],
            "stream_count": profile["stream_count"],
            "workspace_mb": profile["workspace_mb"],
            "max_workspace_gb": profile.get("max_workspace_gb", 2),
            # Format parameters
            "recommended_format": recommended_fmt,
            "fallback_formats": profile.get("fallback_formats", []),
            "precision": profile.get("precision", "fp32"),
            "dynamic_batch": profile.get("dynamic_batch", False),
            "coreml_compatible": profile.get("coreml_compatible", False),
            # Image size
            "imgsz": imgsz,
            # Export kwargs for this format
            "export_kwargs": fmt_config,
            # Performance notes
            "notes": [],
        }

        # Add DLA core if applicable
        if "dla_core" in profile:
            config["dla_core"] = profile["dla_core"]
            config["notes"].append(
                "Enable DLA core with --dla flag in TensorRT runtime"
            )

        # Add device-specific note if present
        if "note" in profile:
            config["notes"].append(profile["note"])

        # Performance tip for INT8
        if recommended_fmt == "engine-int8":
            config["notes"].append(
                "INT8 requires calibration dataset (1000 images recommended). "
                "Use /export/start with int8_quantize=True."
            )

        logger.info(
            f"[EdgeProfileGenerator] Generated config for device={device}, "
            f"format={recommended_fmt}, batch={profile['batch_size']}"
        )

        return config

    def list_devices(self) -> list[str]:
        """Return list of all supported devices."""
        return list(self.DEVICE_PROFILES.keys())

    def compare_formats(
        self,
        device: str,
        model_path: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare all available formats for a device.

        Args:
            device: Target device name
            model_path: Path to model file

        Returns:
            Dict mapping format name to config dict
        """
        profile = self.get_profile(device)
        results: Dict[str, Dict[str, Any]] = {}

        all_formats = [profile["recommended_format"]] + profile.get("fallback_formats", [])

        for fmt in all_formats:
            fmt_config = self.FORMAT_CONFIGS.get(fmt, self.FORMAT_CONFIGS["onnx"])
            results[fmt] = {
                "format": fmt,
                "export_kwargs": fmt_config,
                "precision": profile.get("precision", "fp32"),
                "recommended": (fmt == profile["recommended_format"]),
            }

        return results
