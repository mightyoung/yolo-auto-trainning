"""Stub for validator module.

This is a stub to satisfy imports from training-api/src/deployment/exporter.py.
The actual validator implementation is in training-api/src/deployment/validator.py.
"""
from typing import Any, Dict, Optional


class ModelValidator:
    """Stub ModelValidator."""

    def __init__(self, model_path: str):
        self.model_path = model_path

    def validate(self) -> Dict[str, Any]:
        return {"valid": True, "model_path": self.model_path}
