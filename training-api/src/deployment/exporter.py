"""
Model Deployment Module - Export and optimize models for edge deployment.

Supported formats:
- ONNX: Cross-platform inference
- TensorRT: NVIDIA Jetson optimization
- TorchScript: PyTorch native

Reference: https://docs.ultralytics.com/integrations/tensorrt/
"""

from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ultralytics import YOLO


@dataclass
class ExportResult:
    """Export result container."""
    status: str
    model_path: Optional[Path] = None
    size_mb: float = 0.0
    format: str = ""
    platform: str = ""
    error: Optional[str] = None


class ModelExporter:
    """Model exporter for various platforms."""

    # Platform-specific configurations
    PLATFORM_CONFIGS = {
        "jetson_orin": {
            "format": "engine",
            "half": True,
            "dynamic": True,
            "workspace": 4,
            "int8": False,
        },
        "jetson_nano": {
            "format": "engine",
            "half": True,
            "dynamic": False,
            "workspace": 2,
            "int8": False,
        },
        "tensorrt": {
            "format": "engine",
            "half": True,
            "dynamic": True,
            "workspace": 4,
            "int8": False,
        },
        "onnx": {
            "format": "onnx",
            "half": False,
            "dynamic": False,
            "simplify": True,
            "opset": 13,
        },
        "onnx_fp16": {
            "format": "onnx",
            "half": True,
            "dynamic": False,
            "simplify": True,
            "opset": 13,
        },
        "cpu": {
            "format": "onnx",
            "half": False,
            "dynamic": False,
        },
    }

    def __init__(self, output_dir: Path = None):
        self.output_dir = Path(output_dir or "./runs/export")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        model_path: Path,
        platform: str = "jetson_orin",
        imgsz: int = 640,
    ) -> ExportResult:
        """
        Export model to target platform.

        Args:
            model_path: Path to trained model
            platform: Target platform
            imgsz: Input image size

        Returns:
            ExportResult with exported model info
        """
        config = self.PLATFORM_CONFIGS.get(platform, self.PLATFORM_CONFIGS["onnx"])

        model = YOLO(str(model_path))

        try:
            # Build export kwargs
            export_kwargs = {
                "format": config["format"],
                "half": config.get("half", False),
                "dynamic": config.get("dynamic", False),
                "project": str(self.output_dir),
                "exist_ok": True,
            }

            if "opset" in config:
                export_kwargs["opset"] = config["opset"]
            if "simplify" in config:
                export_kwargs["simplify"] = config["simplify"]
            if "workspace" in config:
                export_kwargs["workspace"] = config["workspace"]
            if "int8" in config and config["int8"]:
                export_kwargs["int8"] = True

            export_path = model.export(
                imgsz=imgsz,
                **export_kwargs,
            )

            model_size_mb = Path(export_path).stat().st_size / (1024 * 1024)

            return ExportResult(
                status="success",
                model_path=Path(export_path),
                size_mb=model_size_mb,
                format=config["format"],
                platform=platform,
            )

        except Exception as e:
            return ExportResult(
                status="failed",
                error=str(e),
                platform=platform,
            )

    def export_int8_calibration(
        self,
        model_path: Path,
        calibration_data: Path,
        platform: str = "jetson_orin",
    ) -> ExportResult:
        """
        Export model with INT8 quantization (requires calibration).

        Args:
            model_path: Path to trained model
            calibration_data: Path to calibration dataset
            platform: Target platform

        Returns:
            ExportResult with exported model
        """
        config = self.PLATFORM_CONFIGS.get(platform, self.PLATFORM_CONFIGS["jetson_orin"]).copy()
        config["int8"] = True

        model = YOLO(str(model_path))

        try:
            export_path = model.export(
                format=config["format"],
                half=False,
                dynamic=config.get("dynamic", True),
                int8=True,
                data=str(calibration_data),
                project=str(self.output_dir),
                exist_ok=True,
            )

            model_size_mb = Path(export_path).stat().st_size / (1024 * 1024)

            return ExportResult(
                status="success",
                model_path=Path(export_path),
                size_mb=model_size_mb,
                format=config["format"],
                platform=f"{platform}_int8",
            )

        except Exception as e:
            return ExportResult(
                status="failed",
                error=str(e),
                platform=platform,
            )


class EdgeDeployer:
    """Deploy models to edge devices."""

    def __init__(self):
        self.ssh_key_path = None

    def deploy_to_jetson(
        self,
        model_path: Path,
        jetson_ip: str,
        jetson_user: str = "nvidia",
    ) -> Dict[str, Any]:
        """
        Deploy model to Jetson device via SCP.

        Args:
            model_path: Path to exported model
            jetson_ip: Jetson IP address
            jetson_user: Jetson username

        Returns:
            Deployment result
        """
        # Placeholder - requires paramiko or subprocess
        return {
            "status": "pending",
            "device": f"{jetson_user}@{jetson_ip}",
            "model": str(model_path),
        }

    def test_inference(
        self,
        model_path: Path,
        test_image: Path,
    ) -> Dict[str, Any]:
        """
        Test inference on exported model.

        Args:
            model_path: Path to exported model
            test_image: Path to test image

        Returns:
            Inference result
        """
        model = YOLO(str(model_path))

        results = model.predict(
            source=str(test_image),
            verbose=False,
        )

        return {
            "status": "success",
            "detections": len(results[0].boxes),
            "classes": results[0].boxes.cls.cpu().tolist() if len(results[0].boxes) > 0 else [],
        }


# Performance benchmarks for Jetson
JETSON_BENCHMARKS = {
    "jetson_orin_nx_16gb": {
        "fp32": {"fps": 163, "latency_ms": 6.11, "map50": 0.37},
        "fp16": {"fps": 314, "latency_ms": 3.18, "map50": 0.37},
        "int8": {"fps": 434, "latency_ms": 2.30, "map50": 0.32},
    },
    "jetson_nano": {
        "fp32": {"fps": 25, "latency_ms": 40.0, "map50": 0.30},
        "fp16": {"fps": 50, "latency_ms": 20.0, "map50": 0.30},
    },
}


def get_benchmark(platform: str, precision: str = "fp16") -> Dict[str, float]:
    """Get performance benchmark for platform."""
    return JETSON_BENCHMARKS.get(platform, {}).get(precision, {})
