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

from training_api.src.deployment.validator import ModelValidator

# Lazy imports for optional dependencies
_paramiko = None
_scp = None


def _get_paramiko():
    """Lazy load paramiko."""
    global _paramiko
    if _paramiko is None:
        import paramiko
        _paramiko = paramiko
    return _paramiko


def _get_scp():
    """Lazy load scp."""
    global _scp
    if _scp is None:
        from scp import SCPClient
        _scp = SCPClient
    return _scp


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

            model_path = Path(export_path)

            # Validate exported model
            validation = ModelValidator.validate_model_file(model_path)
            if not validation["valid"]:
                return ExportResult(
                    status="failed",
                    error=f"Model validation failed: {validation['error']}",
                    platform=platform,
                )

            model_size_mb = model_path.stat().st_size / (1024 * 1024)

            return ExportResult(
                status="success",
                model_path=model_path,
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

            model_path = Path(export_path)

            # Validate exported model
            validation = ModelValidator.validate_model_file(model_path)
            if not validation["valid"]:
                return ExportResult(
                    status="failed",
                    error=f"Model validation failed: {validation['error']}",
                    platform=platform,
                )

            model_size_mb = model_path.stat().st_size / (1024 * 1024)

            return ExportResult(
                status="success",
                model_path=model_path,
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

    def __init__(self, ssh_key_path: Path = None):
        self.ssh_key_path = ssh_key_path

    def deploy_to_jetson(
        self,
        model_path: Path,
        jetson_ip: str,
        jetson_user: str = "nvidia",
        jetson_password: str = None,
        remote_model_dir: str = "/home/nvidia/models",
    ) -> Dict[str, Any]:
        """
        Deploy model to Jetson device via SSH/SCP.

        Args:
            model_path: Path to exported model file
            jetson_ip: IP address of the Jetson device
            jetson_user: Username for SSH login (default: "nvidia")
            jetson_password: Password (if not using SSH key)
            remote_model_dir: Remote directory to store models

        Returns:
            Dict with deployment status and details
        """
        paramiko = _get_paramiko()
        SCPClient = _get_scp()

        model_path = Path(model_path)
        if not model_path.exists():
            return {
                "status": "failed",
                "error": f"Model file not found: {model_path}",
            }

        ssh = None
        try:
            # Create SSH client
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connection parameters
            connect_kwargs = {
                "hostname": jetson_ip,
                "username": jetson_user,
                "timeout": 10,
            }

            # Use SSH key if provided, otherwise try password
            if self.ssh_key_path and Path(self.ssh_key_path).exists():
                connect_kwargs["key_filename"] = str(self.ssh_key_path)
            elif jetson_password:
                connect_kwargs["password"] = jetson_password
            # Otherwise rely on SSH agent/default keys

            ssh.connect(**connect_kwargs)

            # Create remote directory if needed
            stdin, stdout, stderr = ssh.exec_command(
                f"mkdir -p {remote_model_dir}"
            )
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                error = stderr.read().decode() if stderr else "Unknown error"
                ssh.close()
                return {
                    "status": "failed",
                    "error": f"Failed to create remote directory: {error}",
                }

            # Use SCP to transfer the model file
            with SCPClient(ssh.get_transport()) as scp_client:
                scp_client.put(str(model_path), remote_model_dir)

            # Verify deployment
            model_name = model_path.name
            stdin, stdout, stderr = ssh.exec_command(
                f"ls -la {remote_model_dir}/{model_name}"
            )
            exit_status = stdout.channel.recv_exit_status()

            ssh.close()

            if exit_status == 0:
                return {
                    "status": "deployed",
                    "device": f"{jetson_user}@{jetson_ip}",
                    "model": f"{remote_model_dir}/{model_name}",
                    "model_size": model_path.stat().st_size,
                }
            else:
                return {
                    "status": "verification_failed",
                    "error": "Model file not found on remote device after transfer",
                }

        except paramiko.AuthenticationException:
            if ssh:
                ssh.close()
            return {
                "status": "failed",
                "error": "Authentication failed. Check username/password or SSH key.",
            }
        except paramiko.SSHException as e:
            if ssh:
                ssh.close()
            return {
                "status": "failed",
                "error": f"SSH error: {str(e)}",
            }
        except FileNotFoundError as e:
            if ssh:
                ssh.close()
            return {
                "status": "failed",
                "error": f"SSH key not found: {str(e)}",
            }
        except Exception as e:
            if ssh:
                ssh.close()
            return {
                "status": "failed",
                "error": f"Deployment failed: {str(e)}",
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
