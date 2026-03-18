"""
Model Validation Module - Validate exported model files.

Validates:
- File existence and basic integrity
- Format-specific headers (ONNX, TensorRT, TFLite)
- File size sanity checks
"""

from pathlib import Path
from typing import Dict, Any, List, Optional


class ModelValidator:
    """Validator for exported model files."""

    # Minimum reasonable file sizes (in bytes)
    MIN_FILE_SIZES = {
        ".onnx": 1000,      # 1KB
        ".engine": 10000,   # 10KB
        ".plan": 10000,     # 10KB
        ".tflite": 1000,    # 1KB
        ".pt": 1000,        # 1KB
        ".torchscript": 1000,
    }

    # Format magic numbers / headers
    FORMAT_HEADERS = {
        ".onnx": b"ONNX",
        ".tflite": b"TFL3",
    }

    @staticmethod
    def validate_model_file(
        model_path: Path,
        expected_formats: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Validate exported model file.

        Args:
            model_path: Path to the exported model file
            expected_formats: Optional list of expected formats to validate against

        Returns:
            Dict with validation result containing:
            - valid: bool
            - error: Optional[str]
            - file_size: Optional[int]
            - format: Optional[str]
        """
        # Check file exists
        if not model_path.exists():
            return {
                "valid": False,
                "error": f"Model file not found: {model_path}",
            }

        # Check file is not empty
        file_size = model_path.stat().st_size
        if file_size == 0:
            return {
                "valid": False,
                "error": "Model file is empty",
            }

        # Get file extension
        suffix = model_path.suffix.lower()

        # Check minimum size
        min_size = ModelValidator.MIN_FILE_SIZES.get(suffix, 1000)
        if file_size < min_size:
            return {
                "valid": False,
                "error": f"Model file too small: {file_size} bytes (minimum: {min_size})",
            }

        # Format-specific validation
        if suffix == ".onnx":
            return ModelValidator._validate_onnx(model_path, file_size)
        elif suffix in [".engine", ".plan"]:
            return ModelValidator._validate_tensorrt(model_path, file_size)
        elif suffix == ".tflite":
            return ModelValidator._validate_tflite(model_path, file_size)
        elif suffix == ".pt":
            return ModelValidator._validate_torchscript(model_path, file_size)

        # Unknown format - just check size
        return {
            "valid": True,
            "file_size": file_size,
            "format": suffix.lstrip("."),
        }

    @staticmethod
    def _validate_onnx(model_path: Path, file_size: int) -> Dict[str, Any]:
        """Validate ONNX model format by checking header."""
        try:
            with open(model_path, "rb") as f:
                header = f.read(8)
                # ONNX files start with ONNX magic number
                if not header.startswith(b"ONNX"):
                    return {
                        "valid": False,
                        "error": "Invalid ONNX header - file does not start with 'ONNX'",
                    }
            return {
                "valid": True,
                "file_size": file_size,
                "format": "onnx",
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"ONNX validation failed: {e}",
            }

    @staticmethod
    def _validate_tensorrt(model_path: Path, file_size: int) -> Dict[str, Any]:
        """
        Validate TensorRT engine file.

        Note: TensorRT .engine files don't have a standard magic number.
        We validate based on minimum file size (already checked) and
        check for calibration cache presence for INT8 models.
        """
        # Check for INT8 calibration cache (optional file)
        model_dir = model_path.parent
        model_stem = model_path.stem
        calibration_cache = model_dir / f"{model_stem}.cache"

        result = {
            "valid": True,
            "file_size": file_size,
            "format": "tensorrt",
        }

        if calibration_cache.exists():
            result["calibration_cache"] = str(calibration_cache)

        return result

    @staticmethod
    def _validate_tflite(model_path: Path, file_size: int) -> Dict[str, Any]:
        """Validate TensorFlow Lite model format by checking header."""
        try:
            with open(model_path, "rb") as f:
                header = f.read(8)
                # TFLite files start with TFL3 magic number
                if not header.startswith(b"TFL3"):
                    return {
                        "valid": False,
                        "error": "Invalid TFLite header - file does not start with 'TFL3'",
                    }
            return {
                "valid": True,
                "file_size": file_size,
                "format": "tflite",
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"TFLite validation failed: {e}",
            }

    @staticmethod
    def _validate_torchscript(model_path: Path, file_size: int) -> Dict[str, Any]:
        """
        Validate TorchScript model format.

        TorchScript files start with 'torch' prefix in serialized format.
        """
        try:
            with open(model_path, "rb") as f:
                header = f.read(7)
                # TorchScript serialized format starts with 'torch.'
                if not header.startswith(b"torch."):
                    return {
                        "valid": False,
                        "error": "Invalid TorchScript header - file does not start with 'torch.'",
                    }
            return {
                "valid": True,
                "file_size": file_size,
                "format": "torchscript",
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"TorchScript validation failed: {e}",
            }
