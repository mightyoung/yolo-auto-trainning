# Unit Tests - Training Configuration Module

import pytest
from pathlib import Path
import sys

# Add src to path - handle both direct and package execution
test_dir = Path(__file__).parent
project_root = test_dir.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from training.config import (
    TrainingConfig,
    SanityCheckConfig,
    HPOConfig,
    ExportConfig,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_SANITY_CHECK_CONFIG,
    DEFAULT_HPO_CONFIG,
    DEFAULT_EXPORT_CONFIG,
)


# ==================== Test TrainingConfig ====================

class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_values_match_ultralytics(self):
        """Default values should match Ultralytics official defaults."""
        config = TrainingConfig()

        # Critical parameters from official docs
        assert config.lr0 == 0.01, "lr0 should be 0.01"
        assert config.box == 7.5, "box should be 7.5"
        assert config.fliplr == 0.5, "fliplr should be 0.5"
        assert config.momentum == 0.937, "momentum should be 0.937"
        assert config.weight_decay == 0.0005, "weight_decay should be 0.0005"

    def test_to_dict_contains_all_params(self):
        """to_dict should contain all training parameters."""
        config = TrainingConfig()
        params = config.to_dict()

        # Check critical parameters
        assert "lr0" in params
        assert "box" in params
        assert "fliplr" in params
        assert "epochs" in params
        assert "imgsz" in params
        assert "batch" in params

    def test_to_dict_values_match_config(self):
        """to_dict values match config attributes."""
        config = TrainingConfig()
        params = config.to_dict()

        assert params["lr0"] == config.lr0
        assert params["box"] == config.box
        assert params["epochs"] == config.epochs

    def test_custom_values(self):
        """Custom values are stored correctly."""
        config = TrainingConfig(
            model="yolo11n",
            lr0=0.001,
            epochs=50,
            batch=8,
        )

        assert config.model == "yolo11n"
        assert config.lr0 == 0.001
        assert config.epochs == 50
        assert config.batch == 8


# ==================== Test SanityCheckConfig ====================

class TestSanityCheckConfig:
    """Test SanityCheckConfig dataclass."""

    def test_default_values(self):
        """Default values are sensible for sanity check."""
        config = SanityCheckConfig()

        assert config.epochs == 10  # Quick sanity check
        assert config.imgsz == 640
        assert config.min_map50 == 0.3
        assert config.cache is True  # Cache for speed

    def test_custom_values(self):
        """Custom values are stored correctly."""
        config = SanityCheckConfig(
            epochs=5,
            imgsz=320,
            min_map50=0.2,
        )

        assert config.epochs == 5
        assert config.imgsz == 320
        assert config.min_map50 == 0.2


# ==================== Test HPOConfig ====================

class TestHPOConfig:
    """Test HPOConfig dataclass."""

    def test_default_search_space(self):
        """Default search space has 6 parameters."""
        config = HPOConfig()

        # Should have 6 optimizer parameters (not augmentation)
        assert len(config.param_space) == 6

    def test_search_space_bounds_valid(self):
        """Search space bounds are valid (low < high)."""
        config = HPOConfig()

        for param, (low, high) in config.param_space.items():
            assert low < high, f"{param} bounds invalid: {low} >= {high}"

    def test_hpo_epochs_reasonable(self):
        """HPO trial epochs are reasonable."""
        config = HPOConfig()

        assert config.epochs_per_trial >= 30, "Too few epochs for HPO"
        assert config.epochs_per_trial <= 100, "Too many epochs for HPO"

    def test_grace_period_valid(self):
        """Grace period is valid for ASHA."""
        config = HPOConfig()

        assert config.grace_period >= 10, "Grace period too short"
        assert config.grace_period < config.epochs_per_trial

    def test_n_trials_reasonable(self):
        """Number of trials is reasonable."""
        config = HPOConfig()

        assert config.n_trials >= 20, "Too few trials"
        assert config.n_trials <= 100, "Too many trials"

    def test_fixed_params_are_set(self):
        """Fixed augmentation parameters are set."""
        config = HPOConfig()

        assert "fliplr" in config.fixed_params
        assert "mosaic" in config.fixed_params
        assert config.fixed_params["fliplr"] == 0.5


# ==================== Test ExportConfig ====================

class TestExportConfig:
    """Test ExportConfig dataclass."""

    def test_default_values(self):
        """Default export config is sensible."""
        config = ExportConfig()

        assert config.format == "onnx"
        assert config.opset == 13
        assert config.half is True  # FP16
        assert config.simplify is True

    def test_platform_configs_defined(self):
        """Platform-specific configs are defined."""
        config = ExportConfig()

        assert "jetson" in config.platform_configs
        assert "tensorrt" in config.platform_configs
        assert "cpu" in config.platform_configs

    def test_jetson_config_uses_fp16(self):
        """Jetson config uses FP16."""
        config = ExportConfig()

        assert config.platform_configs["jetson"]["half"] is True

    def test_cpu_config_uses_fp32(self):
        """CPU config uses FP32."""
        config = ExportConfig()

        assert config.platform_configs["cpu"]["half"] is False


# ==================== Test Default Configs ====================

class TestDefaultConfigs:
    """Test default configuration singletons."""

    def test_default_training_config_exists(self):
        """Default training config exists."""
        assert DEFAULT_TRAINING_CONFIG is not None
        assert isinstance(DEFAULT_TRAINING_CONFIG, TrainingConfig)

    def test_default_sanity_check_config_exists(self):
        """Default sanity check config exists."""
        assert DEFAULT_SANITY_CHECK_CONFIG is not None
        assert isinstance(DEFAULT_SANITY_CHECK_CONFIG, SanityCheckConfig)

    def test_default_hpo_config_exists(self):
        """Default HPO config exists."""
        assert DEFAULT_HPO_CONFIG is not None
        assert isinstance(DEFAULT_HPO_CONFIG, HPOConfig)

    def test_default_export_config_exists(self):
        """Default export config exists."""
        assert DEFAULT_EXPORT_CONFIG is not None
        assert isinstance(DEFAULT_EXPORT_CONFIG, ExportConfig)
