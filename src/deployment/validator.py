"""Stub for validator module.

This is a stub to satisfy imports from training-api/src/deployment/exporter.py.
The actual validator implementation is in training-api/src/deployment/validator.py.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelValidator:
    """Stub ModelValidator for testing purposes."""

    MIN_FILE_SIZES = {
        ".onnx": 1000,
        ".engine": 10000,
        ".plan": 10000,
        ".tflite": 1000,
        ".pt": 1000,
        ".torchscript": 1000,
    }

    @staticmethod
    def validate_model_file(
        model_path: Path,
        expected_formats: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Stub validation - always returns valid for testing."""
        return {"valid": True}
