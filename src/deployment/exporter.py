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
    """
    Deploy models to edge devices.

    Supports:
    - Jetson Nano/Orin (via SSH/SCP)
    - Raspberry Pi (via SSH/SCP)
    - Generic edge devices with SSH access

    Based on best practices for edge deployment:
    - Automated deployment scripts
    - Health checks after deployment
    - Rollback capability
    """

    def __init__(self, ssh_key_path: str = None):
        """
        Initialize EdgeDeployer.

        Args:
            ssh_key_path: Path to SSH private key
        """
        self.ssh_key_path = ssh_key_path
        self.deployment_history: list = []

    def _run_ssh_command(
        self,
        host: str,
        user: str,
        command: str,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Run SSH command on remote device.

        Args:
            host: Target host IP
            user: SSH username
            command: Command to execute
            timeout: Command timeout

        Returns:
            Result dict with stdout, stderr, returncode
        """
        import subprocess

        # Build SSH command
        ssh_cmd = ["ssh"]
        if self.ssh_key_path:
            ssh_cmd.extend(["-i", str(self.ssh_key_path)])
        ssh_cmd.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            f"{user}@{host}",
            command
        ])

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timeout",
                "returncode": -1,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
            }

    def _scp_copy(
        self,
        local_path: Path,
        remote_host: str,
        remote_user: str,
        remote_path: str,
    ) -> Dict[str, Any]:
        """
        Copy file to remote device via SCP.

        Args:
            local_path: Local file path
            remote_host: Remote host IP
            remote_user: Remote username
            remote_path: Remote destination path

        Returns:
            Result dict
        """
        import subprocess

        scp_cmd = ["scp"]
        if self.ssh_key_path:
            scp_cmd.extend(["-i", str(self.ssh_key_path)])
        scp_cmd.extend([
            "-o", "StrictHostKeyChecking=no",
            str(local_path),
            f"{remote_user}@{remote_host}:{remote_path}"
        ])

        try:
            result = subprocess.run(
                scp_cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout for file transfer
            )
            return {
                "success": result.returncode == 0,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except Exception as e:
            return {
                "success": False,
                "stderr": str(e),
                "returncode": -1,
            }

    def deploy_to_jetson(
        self,
        model_path: Path,
        jetson_ip: str,
        jetson_user: str = "nvidia",
        remote_model_dir: str = "/home/nvidia/models",
    ) -> Dict[str, Any]:
        """
        Deploy model to Jetson device via SCP.

        Args:
            model_path: Path to exported model
            jetson_ip: Jetson IP address
            jetson_user: Jetson username
            remote_model_dir: Remote model directory

        Returns:
            Deployment result
        """
        # Validate model exists
        if not Path(model_path).exists():
            return {
                "status": "failed",
                "error": f"Model file not found: {model_path}",
                "device": f"{jetson_user}@{jetson_ip}",
            }

        # Create remote directory
        mkdir_result = self._run_ssh_command(
            jetson_ip,
            jetson_user,
            f"mkdir -p {remote_model_dir}",
        )

        if not mkdir_result.get("success"):
            return {
                "status": "failed",
                "error": f"Failed to create remote directory: {mkdir_result.get('stderr')}",
                "device": f"{jetson_user}@{jetson_ip}",
            }

        # Copy model file
        model_filename = Path(model_path).name
        remote_path = f"{remote_model_dir}/{model_filename}"

        scp_result = self._scp_copy(
            model_path,
            jetson_ip,
            jetson_user,
            remote_path,
        )

        if not scp_result.get("success"):
            return {
                "status": "failed",
                "error": f"Failed to copy model: {scp_result.get('stderr')}",
                "device": f"{jetson_user}@{jetson_ip}",
            }

        # Record deployment
        deployment_record = {
            "timestamp": str(Path(model_path).stat().st_mtime),
            "model": str(model_path),
            "device": f"{jetson_user}@{jetson_ip}",
            "remote_path": remote_path,
        }
        self.deployment_history.append(deployment_record)

        return {
            "status": "success",
            "device": f"{jetson_user}@{jetson_ip}",
            "model": str(model_path),
            "remote_path": remote_path,
            "deployment_id": len(self.deployment_history),
        }

    def deploy_to_raspberry_pi(
        self,
        model_path: Path,
        pi_ip: str,
        pi_user: str = "pi",
        remote_model_dir: str = "/home/pi/models",
    ) -> Dict[str, Any]:
        """
        Deploy model to Raspberry Pi.

        Args:
            model_path: Path to exported model (ONNX format recommended)
            pi_ip: Raspberry Pi IP address
            pi_user: Pi username
            remote_model_dir: Remote model directory

        Returns:
            Deployment result
        """
        return self.deploy_to_jetson(
            model_path=model_path,
            jetson_ip=pi_ip,
            jetson_user=pi_user,
            remote_model_dir=remote_model_dir,
        )

    def check_device_health(
        self,
        host: str,
        user: str,
    ) -> Dict[str, Any]:
        """
        Check edge device health.

        Args:
            host: Device IP address
            user: SSH username

        Returns:
            Health status
        """
        # Check SSH connectivity
        ssh_result = self._run_ssh_command(
            host, user, "echo 'ok'", timeout=10
        )

        if not ssh_result.get("success"):
            return {
                "status": "unhealthy",
                "ssh": "failed",
                "error": ssh_result.get("stderr"),
            }

        # Get device info
        info_result = self._run_ssh_command(
            host, user, "uname -a && free -h && df -h", timeout=10
        )

        return {
            "status": "healthy",
            "ssh": "ok",
            "device_info": info_result.get("stdout", ""),
        }

    def get_deployment_history(self) -> List[Dict[str, Any]]:
        """Get deployment history."""
        return self.deployment_history

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
