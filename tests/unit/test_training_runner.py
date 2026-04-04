# Unit Tests - YOLO Training Runner

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import sys
import tempfile

# ==================== Fixtures ====================

# Cache for lazily-loaded training API imports
_training_api_imports = {}


@pytest.fixture(autouse=True)
def setup_training_api_imports():
    """Set up training-api paths and imports for tests that need them.

    This runs BEFORE each test, ensuring sys.path is modified only during
    test execution (not during pytest collection phase).
    """
    global _training_api_imports

    # If already imported, just set in globals and yield
    if _training_api_imports:
        globals().update(_training_api_imports)
        yield _training_api_imports
        return

    # Training API path setup - only during test execution
    project_root = Path(__file__).parent.parent.parent
    training_api_path = project_root / "training-api"
    training_api_src_path = training_api_path / "src"

    # Save original sys.path
    original_sys_path = sys.path.copy()

    # Insert training-api paths at front for imports
    for p in list(sys.path):
        if 'training-api' in p:
            sys.path.remove(p)
    sys.path.insert(0, str(training_api_src_path))
    sys.path.insert(0, str(training_api_path))

    # Clear any cached 'training' modules from sys.modules to avoid
    # conflicts with src/training that may have been loaded by other tests
    modules_to_remove = [k for k in sys.modules.keys()
                         if k == 'training' or k.startswith('training.')]
    for mod in modules_to_remove:
        del sys.modules[mod]

    # Import the modules
    from training.runner import (
        YOLOTrainer,
        TransferLearningTrainer,
        TrainingResult,
        PipelineCurriculumTrainer,
        CurriculumStage,
    )
    from training.config import (
        TrainingConfig,
        SanityCheckConfig,
        HPOConfig,
        ExportConfig,
    )

    _training_api_imports = {
        'YOLOTrainer': YOLOTrainer,
        'TransferLearningTrainer': TransferLearningTrainer,
        'TrainingResult': TrainingResult,
        'PipelineCurriculumTrainer': PipelineCurriculumTrainer,
        'CurriculumStage': CurriculumStage,
        'TrainingConfig': TrainingConfig,
        'SanityCheckConfig': SanityCheckConfig,
        'HPOConfig': HPOConfig,
        'ExportConfig': ExportConfig,
    }

    # Set in module globals so tests can access them directly
    globals().update(_training_api_imports)

    yield _training_api_imports

    # Restore sys.path after test
    sys.path[:] = original_sys_path


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


class TestPipelineCurriculumTrainer:
    """Regression tests for curriculum plateau handling."""

    def test_run_stage_writes_plateau_signals_to_redis(self, temp_dir):
        trainer = PipelineCurriculumTrainer(output_dir=temp_dir)
        stage = CurriculumStage(
            name="stage2",
            epochs=10,
            imgsz=640,
            batch=16,
            model="yolo11m",
            augmentation_preset="strong",
        )
        redis_client = MagicMock()
        plateau_manager = MagicMock()
        plateau_manager.get_status.return_value = {
            "lr_reduction_count": 1,
            "augment_boost_active": True,
            "signaled_expansion": True,
            "in_stage_restarts": 2,
            "strategies_triggered": [
                {"action": "lr_decay", "adjustment": {"new_lr": 0.005}},
                {"action": "augment_boost", "adjustment": {"mixup": 0.3}},
                {"action": "data_expansion", "adjustment": {"target": 0.9}},
            ],
            "llm_diagnosis": {"root_cause": "small_dataset"},
        }
        plateau_manager.on_metric.return_value = MagicMock(triggered=False)
        plateau_manager._in_stage_restarts = 0

        train_result = TrainingResult(
            status="completed",
            model_path=temp_dir / "best.pt",
            metrics={"mAP50": 0.6},
        )
        train_result.model_path.write_text("weights")

        with patch.object(YOLOTrainer, "train", return_value=train_result) as mock_train:
            trainer._run_stage(
                stage=stage,
                data_yaml=temp_dir / "data.yaml",
                stage_num=2,
                plateau_manager=plateau_manager,
                redis_client=redis_client,
                task_id_for_redis="task123",
            )

        epoch_callback = mock_train.call_args.kwargs["metric_callback"]
        for epoch in range(1, 6):
            epoch_callback(epoch, 10, {"mAP50": 0.5 + epoch * 0.01})

        redis_client.hset.assert_called_once()
        mapping = redis_client.hset.call_args.kwargs["mapping"]
        assert mapping["lr_decay_triggered"] == "True"
        assert mapping["lr_decay_signal"] == '{"new_lr": 0.005}'
        assert mapping["augment_boost_signal"] == '{"mixup": 0.3}'
        assert mapping["data_expansion_signal"] == '{"target": 0.9}'
        assert mapping["llm_diagnosis"] == '{"root_cause": "small_dataset"}'

    def test_run_stage_prefers_higher_map_over_checkpoint_size(self, temp_dir):
        trainer = PipelineCurriculumTrainer(output_dir=temp_dir)
        stage = CurriculumStage(
            name="stage2",
            epochs=10,
            imgsz=640,
            batch=16,
            model="yolo11m",
            augmentation_preset="strong",
        )
        resume_path = temp_dir / "resume.pt"
        resume_path.write_text("x" * 5000)
        better_path = temp_dir / "better.pt"
        better_path.write_text("small")
        plateau_manager = MagicMock()
        plateau_manager.on_metric.return_value = MagicMock(triggered=False)
        plateau_manager._in_stage_restarts = 0

        train_result = TrainingResult(
            status="completed",
            model_path=better_path,
            metrics={"mAP50": 0.72},
        )

        with patch.object(YOLOTrainer, "train", return_value=train_result):
            result, _ = trainer._run_stage(
                stage=stage,
                data_yaml=temp_dir / "data.yaml",
                stage_num=2,
                resume_from=str(resume_path),
                plateau_manager=plateau_manager,
            )

        assert result.model_path == better_path
        plateau_manager.set_best_checkpoint_path.assert_called_once_with(str(better_path))
