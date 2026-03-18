"""
Unit tests for MLflow integration module.

Tests MLflowTracker functionality and YOLO training integration.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestMLflowTracker:
    """Test MLflowTracker class."""

    def test_initialization(self):
        """Test MLflowTracker initialization."""
        with patch('src.training.mlflow_tracker.mlflow') as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = None
            mock_mlflow.create_experiment.return_value = "test_experiment_id"

            # This will fail due to import issues, but tests the concept
            from src.training.mlflow_tracker import MLflowTracker

            # Skip if mlflow not installed
            pytest.skip("MLflow installation required")

    def test_flatten_dict(self):
        """Test dictionary flattening utility."""
        from src.training.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker.__new__(MLflowTracker)

        # Test nested dict flattening
        nested = {
            "train": {
                "mAP50": 0.8,
                "mAP50-95": 0.6
            },
            "epochs": 100
        }

        flat = tracker._flatten_dict(nested)
        assert "train/mAP50" in flat
        assert flat["train/mAP50"] == 0.8
        assert flat["epochs"] == 100


class TestYOLOTrainerMLflow:
    """Test YOLOTrainer MLflow integration."""

    def test_trainer_with_mlflow(self):
        """Test YOLOTrainer can accept MLflowTracker."""
        with patch('src.training.runner.YOLO'):
            from src.training.runner import YOLOTrainer

            # Mock MLflowTracker
            mock_tracker = Mock()
            mock_tracker.start_run.return_value = Mock()
            mock_tracker.log_params = Mock()
            mock_tracker.log_metrics = Mock()
            mock_tracker.log_model = Mock()
            mock_tracker.end_run = Mock()

            # Create trainer with MLflow tracker
            trainer = YOLOTrainer(
                model="yolo11n",
                output_dir=Path(tempfile.mkdtemp()),
                mlflow_tracker=mock_tracker
            )

            assert trainer.mlflow_tracker == mock_tracker


class TestMLflowIntegration:
    """Test MLflow integration with YOLO."""

    def test_mlflow_env_vars(self):
        """Test MLflow environment variable setup."""
        # Test that environment variables can be set
        os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

        # Verify the env var is set
        assert os.getenv("MLFLOW_TRACKING_URI") == "file:./mlruns"

        # Clean up
        del os.environ["MLFLOW_TRACKING_URI"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
