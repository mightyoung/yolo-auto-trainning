# Unit Tests - YOLO Training Runner

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import sys
import tempfile

# Setup mock for ultralytics BEFORE importing
mock_ultralytics = Mock()
mock_yolo_class = Mock()
mock_ultralytics.YOLO = mock_yolo_class
sys.modules['ultralytics'] = mock_ultralytics

# Add training-api to path
test_dir = Path(__file__).parent
project_root = test_dir.parent.parent
training_api_path = project_root / "training-api"
src_path = training_api_path / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(training_api_path) not in sys.path:
    sys.path.insert(0, str(training_api_path))

# Force reimport with mocks
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith('training'):
        del sys.modules[mod_name]

from training.runner import (
    YOLOTrainer,
    TransferLearningTrainer,
    TrainingResult,
)
from training.config import (
    TrainingConfig,
    SanityCheckConfig,
    HPOConfig,
    ExportConfig,
)


# ==================== Fixtures ====================

@pytest.fixture(autouse=True)
def reset_mock():
    """Reset mock before each test."""
    mock_yolo_class.reset_mock()
    yield


# ==================== Test TrainingResult ====================

class TestTrainingResult:
    """Test TrainingResult dataclass."""

    def test_training_result_status_only(self):
        """TrainingResult with only status field."""
        result = TrainingResult(status="pending")
        assert result.status == "pending"
        assert result.model_path is None
        assert result.metrics is None
        assert result.best_params is None
        assert result.error is None

    def test_training_result_with_metrics(self):
        """TrainingResult with metrics."""
        result = TrainingResult(
            status="completed",
            model_path=Path("/runs/train/weights/best.pt"),
            metrics={"mAP50": 0.85, "mAP50-95": 0.65},
        )
        assert result.status == "completed"
        assert result.model_path == Path("/runs/train/weights/best.pt")
        assert result.metrics["mAP50"] == 0.85
        assert result.metrics["mAP50-95"] == 0.65

    def test_training_result_with_error(self):
        """TrainingResult with error."""
        result = TrainingResult(
            status="failed",
            error="CUDA out of memory",
        )
        assert result.status == "failed"
        assert result.error == "CUDA out of memory"
        assert result.model_path is None
        assert result.metrics is None


# ==================== Test YOLOTrainer Init ====================

class TestYOLOTrainerInit:
    """Test YOLOTrainer initialization."""

    def test_initialization_with_model(self, temp_dir):
        """Initialization with custom model."""
        trainer = YOLOTrainer(model="yolo11n", output_dir=temp_dir)
        assert trainer.model_name == "yolo11n"
        assert trainer.output_dir == temp_dir

    def test_initialization_with_output_dir(self, temp_dir):
        """Initialization with custom output directory."""
        trainer = YOLOTrainer(output_dir=temp_dir)
        assert trainer.output_dir == temp_dir

    def test_initialization_creates_output_dir(self, temp_dir):
        """Initialization creates output directory if it doesn't exist."""
        new_output_dir = temp_dir / "new_runs" / "nested"
        assert not new_output_dir.exists()

        trainer = YOLOTrainer(output_dir=new_output_dir)
        assert trainer.output_dir.exists()
        assert trainer.output_dir.is_dir()

    def test_default_initialization(self, temp_dir):
        """Default initialization uses yolo11m and ./runs."""
        with patch("training.runner.Path.mkdir"):
            trainer = YOLOTrainer()
            assert trainer.model_name == "yolo11m"


# ==================== Test Config ====================

class TestConfig:
    """Test configuration classes."""

    def test_training_config_defaults(self):
        """TrainingConfig has correct defaults."""
        config = TrainingConfig()
        assert config.model == "yolo11m"
        assert config.epochs == 100
        assert config.lr0 == 0.01
        assert config.optimizer == "SGD"

    def test_sanity_check_config_defaults(self):
        """SanityCheckConfig has correct defaults."""
        config = SanityCheckConfig()
        assert config.epochs == 10
        assert config.min_map50 == 0.3
        assert config.cache is True

    def test_hpo_config_defaults(self):
        """HPOConfig has correct defaults."""
        config = HPOConfig()
        assert config.n_trials == 50
        assert config.epochs_per_trial == 50
        assert "lr0" in config.param_space

    def test_export_config_defaults(self):
        """ExportConfig has correct defaults."""
        config = ExportConfig()
        assert config.format == "onnx"
        assert config.half is True
        assert "jetson" in config.platform_configs
        assert config.platform_configs["jetson"]["half"] is True


# ==================== Test TransferLearningTrainer ====================

class TestTransferLearningTrainer:
    """Test TransferLearningTrainer class."""

    def test_initialization(self):
        """TransferLearningTrainer initializes correctly."""
        trainer = TransferLearningTrainer(
            teacher_model="yolo11m",
            freeze_layers=10,
        )
        assert trainer.teacher_model_name == "yolo11m"
        assert trainer.freeze_layers == 10

    def test_default_initialization(self):
        """Default initialization uses correct defaults."""
        trainer = TransferLearningTrainer()
        assert trainer.teacher_model_name == "yolo11m"
        assert trainer.freeze_layers == 10


# ==================== Test Config Validation ====================

class TestConfigValidation:
    """Test configuration validation."""

    def test_training_config_to_dict(self):
        """TrainingConfig.to_dict() returns expected keys."""
        config = TrainingConfig()
        d = config.to_dict()

        assert "lr0" in d
        assert "epochs" in d
        assert "batch" in d
        assert "optimizer" in d
        assert d["optimizer"] == "SGD"
        assert d["epochs"] == 100

    def test_training_config_custom_values(self):
        """TrainingConfig accepts custom values."""
        config = TrainingConfig(
            lr0=0.001,
            epochs=50,
            batch=32,
            optimizer="Adam",
        )
        assert config.lr0 == 0.001
        assert config.epochs == 50
        assert config.batch == 32
        assert config.optimizer == "Adam"

    def test_hpo_config_search_space(self):
        """HPOConfig defines valid search space."""
        config = HPOConfig()
        assert len(config.param_space) == 6
        assert "lr0" in config.param_space
        assert "momentum" in config.param_space

        # Check bounds are valid (low < high)
        for param, (low, high) in config.param_space.items():
            assert low < high, f"{param} has invalid bounds"

    def test_export_config_platforms(self):
        """ExportConfig defines all required platforms."""
        config = ExportConfig()

        assert "jetson" in config.platform_configs
        assert "tensorrt" in config.platform_configs
        assert "cpu" in config.platform_configs

        # Jetson should use FP16
        assert config.platform_configs["jetson"]["half"] is True
        assert config.platform_configs["jetson"]["format"] == "engine"

        # CPU should use FP32
        assert config.platform_configs["cpu"]["half"] is False

    def test_sanity_check_config_min_map(self):
        """SanityCheckConfig validates min_map threshold."""
        config = SanityCheckConfig(min_map50=0.4)
        assert config.min_map50 == 0.4
