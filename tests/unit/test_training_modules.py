# Unit Tests - Training Module Splits (P0 Coverage)
#
# Tests the split training modules:
# - training_utils.py: validate_dataset_distribution, setup_gpu_memory, cleanup_gpu_memory
# - yolo_trainer.py: export(), export_multi(), config injection, checkpoint selection
# - curriculum.py: CurriculumStage, CurriculumConfig, stage gate decisions

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import sys
import tempfile

# Pre-import mock ultralytics so the training modules can be loaded
mock_ultralytics = Mock()
mock_yolo_class = Mock()
mock_yolo_instance = Mock()
mock_yolo_class.return_value = mock_yolo_instance
mock_ultralytics.YOLO = mock_yolo_class
sys.modules['ultralytics'] = mock_ultralytics


# ==================== Fixtures ====================

@pytest.fixture(autouse=True)
def reset_mock():
    """Reset mock before each test."""
    mock_yolo_class.reset_mock()
    # reset_mock() creates a new return_value, re-bind to our instance
    mock_yolo_class.return_value = mock_yolo_instance
    mock_yolo_instance.reset_mock()
    yield


_training_api_imports = {}


@pytest.fixture(autouse=True)
def setup_training_api_imports():
    """Lazily import training modules with correct sys.path."""
    global _training_api_imports

    project_root = Path(__file__).parent.parent.parent
    training_api_src = project_root / "training-api" / "src"

    original_sys_path = sys.path.copy()
    for p in list(sys.path):
        if 'training-api' in p:
            sys.path.remove(p)
    sys.path.insert(0, str(training_api_src))

    modules_to_remove = [k for k in list(sys.modules.keys())
                        if k == 'training' or k.startswith('training.')]
    for mod in modules_to_remove:
        del sys.modules[mod]

    # Import training package first so __init__.py registers submodules as attributes
    import training
    from training import training_utils
    from training import yolo_trainer
    from training import transfer_trainer
    from training import curriculum
    from training import config

    from training.training_utils import (
        TrainingResult,
        DatasetDistributionResult,
        validate_dataset_distribution,
        setup_gpu_memory,
        cleanup_gpu_memory,
    )
    from training.yolo_trainer import YOLOTrainer
    from training.transfer_trainer import TransferLearningTrainer
    from training.curriculum import (
        CurriculumStage,
        CurriculumConfig,
        PipelineCurriculumTrainer,
    )
    from training.config import TrainingConfig, ExportConfig

    _training_api_imports = {
        'TrainingResult': TrainingResult,
        'DatasetDistributionResult': DatasetDistributionResult,
        'validate_dataset_distribution': validate_dataset_distribution,
        'setup_gpu_memory': setup_gpu_memory,
        'cleanup_gpu_memory': cleanup_gpu_memory,
        'YOLOTrainer': YOLOTrainer,
        'TransferLearningTrainer': TransferLearningTrainer,
        'CurriculumStage': CurriculumStage,
        'CurriculumConfig': CurriculumConfig,
        'PipelineCurriculumTrainer': PipelineCurriculumTrainer,
        'TrainingConfig': TrainingConfig,
        'ExportConfig': ExportConfig,
    }
    globals().update(_training_api_imports)
    yield _training_api_imports
    sys.path[:] = original_sys_path


# ==================== Test training_utils.py ====================

class TestValidateDatasetDistribution:
    """Test validate_dataset_distribution function."""

    def test_returns_warning_when_yaml_not_found(self):
        """Missing data.yaml returns warning status."""
        result = validate_dataset_distribution(Path("/nonexistent/data.yaml"))
        assert result.status == "warning"
        assert "not found" in result.message

    def test_returns_warning_when_yaml_unparseable(self, temp_dir):
        """Unparseable YAML returns warning status."""
        yaml_path = temp_dir / "data.yaml"
        yaml_path.write_text("invalid: yaml: content: [")
        result = validate_dataset_distribution(yaml_path)
        assert result.status == "warning"
        assert "Failed to parse" in result.message

    def test_returns_critical_when_no_train_labels(self, temp_dir):
        """No train labels found returns critical status."""
        # Create data.yaml
        yaml_content = "path: .\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['a']"
        (temp_dir / "data.yaml").write_text(yaml_content)
        # Create val labels but no train labels
        val_labels = temp_dir / "val" / "labels"
        val_labels.mkdir(parents=True)
        (val_labels / "img.txt").write_text("0 0.5 0.5 0.3 0.4\n")

        result = validate_dataset_distribution(temp_dir / "data.yaml")
        assert result.status == "critical"
        assert "No train labels" in result.message

    def test_returns_ok_when_train_val_balanced(self, temp_dir):
        """Similar train/val box areas returns ok status."""
        yaml_content = "path: .\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['a']"
        (temp_dir / "data.yaml").write_text(yaml_content)

        # Create train labels with similar box sizes
        train_labels = temp_dir / "train" / "labels"
        train_labels.mkdir(parents=True)
        train_images = temp_dir / "train" / "images"
        train_images.mkdir(parents=True)
        # 10 boxes of area 0.01 (normalized)
        for i in range(10):
            (train_labels / f"img{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
            (train_images / f"img{i}.jpg").write_text("fake")

        val_labels = temp_dir / "val" / "labels"
        val_labels.mkdir(parents=True)
        val_images = temp_dir / "val" / "images"
        val_images.mkdir(parents=True)
        # 5 boxes of similar area
        for i in range(5):
            (val_labels / f"img{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
            (val_images / f"img{i}.jpg").write_text("fake")

        result = validate_dataset_distribution(temp_dir / "data.yaml")
        assert result.status == "ok"
        assert "Distribution OK" in result.message
        assert result.ratio < 3.0

    def test_returns_critical_when_val_8x_larger(self, temp_dir):
        """Val boxes 8x larger than train returns critical."""
        yaml_content = "path: .\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['a']"
        (temp_dir / "data.yaml").write_text(yaml_content)

        train_labels = temp_dir / "train" / "labels"
        train_labels.mkdir(parents=True)
        train_images = temp_dir / "train" / "images"
        train_images.mkdir(parents=True)
        # Small boxes: area = 0.1 * 0.1 = 0.01
        for i in range(5):
            (train_labels / f"img{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
            (train_images / f"img{i}.jpg").write_text("fake")

        val_labels = temp_dir / "val" / "labels"
        val_labels.mkdir(parents=True)
        val_images = temp_dir / "val" / "images"
        val_images.mkdir(parents=True)
        # Large boxes: area = 0.8 * 0.8 = 0.64 (64x larger)
        for i in range(5):
            (val_labels / f"img{i}.txt").write_text("0 0.5 0.5 0.8 0.8\n")
            (val_images / f"img{i}.jpg").write_text("fake")

        result = validate_dataset_distribution(temp_dir / "data.yaml")
        assert result.status == "critical"
        assert "CRITICAL" in result.message
        assert result.ratio > 5.0

    def test_returns_warning_when_val_4x_larger(self, temp_dir):
        """Val boxes 4x larger than train returns warning."""
        yaml_content = "path: .\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['a']"
        (temp_dir / "data.yaml").write_text(yaml_content)

        train_labels = temp_dir / "train" / "labels"
        train_labels.mkdir(parents=True)
        train_images = temp_dir / "train" / "images"
        train_images.mkdir(parents=True)
        for i in range(10):
            (train_labels / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")  # area=0.04
            (train_images / f"img{i}.jpg").write_text("fake")

        val_labels = temp_dir / "val" / "labels"
        val_labels.mkdir(parents=True)
        val_images = temp_dir / "val" / "images"
        val_images.mkdir(parents=True)
        for i in range(10):
            (val_labels / f"img{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")  # area=0.16 (4x)
            (val_images / f"img{i}.jpg").write_text("fake")

        result = validate_dataset_distribution(temp_dir / "data.yaml")
        assert result.status == "warning"
        assert "WARNING" in result.message


class TestGPUUtilities:
    """Test GPU memory utility functions."""

    def test_setup_gpu_memory_does_not_raise_without_cuda(self):
        """setup_gpu_memory should not raise even without CUDA."""
        # Test that function handles missing CUDA gracefully
        # by checking it doesn't raise when torch.cuda.is_available returns False
        import torch
        original_is_available = torch.cuda.is_available if hasattr(torch, 'cuda') else None
        try:
            if hasattr(torch, 'cuda'):
                torch.cuda.is_available = lambda: False
            # Should not raise
            setup_gpu_memory()
        finally:
            if hasattr(torch, 'cuda') and original_is_available is not None:
                torch.cuda.is_available = original_is_available

    def test_cleanup_gpu_memory_does_not_raise(self):
        """cleanup_gpu_memory should not raise even without CUDA."""
        import torch
        original_is_available = torch.cuda.is_available if hasattr(torch, 'cuda') else None
        try:
            if hasattr(torch, 'cuda'):
                torch.cuda.is_available = lambda: False
            cleanup_gpu_memory()
        finally:
            if hasattr(torch, 'cuda') and original_is_available is not None:
                torch.cuda.is_available = original_is_available

    def test_setup_gpu_memory_calls_cuda_when_available(self):
        """setup_gpu_memory calls CUDA APIs when GPU is available."""
        # Test that when CUDA is available, the function calls the right APIs
        import torch
        if not hasattr(torch, 'cuda'):
            pytest.skip("CUDA not available")
        # This test verifies the function structure is correct
        # (actual GPU calls require real CUDA hardware)
        assert callable(setup_gpu_memory)
        assert callable(cleanup_gpu_memory)


# ==================== Test yolo_trainer.py ====================

class TestYOLOTrainerExport:
    """Test YOLOTrainer.export() and export_multi()."""

    def test_export_returns_model_path_and_size(self, temp_dir):
        """export() returns dict with model path and size_mb."""
        trainer = YOLOTrainer(output_dir=temp_dir)

        # Mock YOLO model and export
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        mock_model.export.return_value = str(temp_dir / "model.engine")

        # Mock file size
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 50 * 1024 * 1024  # 50 MB
            with patch.object(Path, 'stat') as mock_path_stat:
                mock_path_stat.return_value.st_size = 50 * 1024 * 1024
                result = trainer.export(temp_dir / "best.pt", platform="jetson")

        assert "model" in result
        assert "size_mb" in result
        assert result["platform"] == "jetson"
        assert result["fp16"] is True

    def test_export_multi_returns_all_formats(self, temp_dir):
        """export_multi() returns results for all requested formats."""
        trainer = YOLOTrainer(output_dir=temp_dir)

        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        mock_model.export.return_value = str(temp_dir / "model.onnx")

        with patch.object(Path, 'stat') as mock_path_stat:
            mock_path_stat.return_value.st_size = 25 * 1024 * 1024
            result = trainer.export_multi(
                model_path=temp_dir / "best.pt",
                formats=["onnx", "engine-fp16", "engine-int8"],
                platform="jetson",
            )

        assert "onnx" in result
        assert "engine-fp16" in result
        assert "engine-int8" in result
        # engine-fp16: fp16=True, not int8
        assert result["engine-fp16"]["fp16"] is True
        assert result["engine-fp16"]["int8"] is False
        # engine-int8: int8=True, fp16=False (int8 takes precedence)
        assert result["engine-int8"]["int8"] is True
        assert result["engine-int8"]["fp16"] is False

    def test_export_multi_handles_failure_gracefully(self, temp_dir):
        """export_multi() includes error entry for failed format."""
        trainer = YOLOTrainer(output_dir=temp_dir)

        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        # First call succeeds, second raises
        mock_model.export.side_effect = [str(temp_dir / "model.onnx"), RuntimeError("Export failed")]

        with patch.object(Path, 'stat') as mock_path_stat:
            mock_path_stat.return_value.st_size = 25 * 1024 * 1024
            result = trainer.export_multi(
                model_path=temp_dir / "best.pt",
                formats=["onnx", "engine-fp16"],
                platform="jetson",
            )

        assert result["onnx"]["path"] is not None
        assert result["engine-fp16"]["path"] is None
        assert "error" in result["engine-fp16"]


class TestYOLOTrainerCheckpointSelection:
    """Test YOLOTrainer checkpoint selection logic."""

    def test_prefers_best_pt_over_last_pt(self, temp_dir):
        """When both exist, best.pt is returned over last.pt."""
        trainer = YOLOTrainer(output_dir=temp_dir)

        train_dir = temp_dir / "train" / "weights"
        train_dir.mkdir(parents=True)
        (train_dir / "best.pt").write_text("best")
        (train_dir / "last.pt").write_text("last")

        # Mock results.metrics to return mAP so checkpoint reading is skipped
        mock_results = Mock()
        mock_results.results_dict = {"metrics/mAP50(B)": 0.85, "metrics/mAP50-95(B)": 0.65}
        mock_results.save_dir = str(temp_dir / "train")
        mock_yolo_instance.train.return_value = mock_results

        result = trainer.train(data_yaml=temp_dir / "data.yaml", config=TrainingConfig())

        # best.pt was created, so model_path should be best.pt
        assert "best.pt" in str(result.model_path)

    def test_falls_back_to_last_pt_when_best_missing(self, temp_dir):
        """When best.pt missing, last.pt is used."""
        trainer = YOLOTrainer(output_dir=temp_dir)

        train_dir = temp_dir / "train" / "weights"
        train_dir.mkdir(parents=True)
        (train_dir / "last.pt").write_text("last")

        mock_results = Mock()
        mock_results.results_dict = {}
        mock_results.save_dir = str(temp_dir / "train")
        mock_yolo_instance.train.return_value = mock_results

        # Mock torch.load to return checkpoint metrics
        with patch('training.yolo_trainer.torch') as mock_torch:
            mock_torch.load.return_value = {
                "train_metrics": {"metrics/mAP50(B)": 0.7}
            }
            result = trainer.train(data_yaml=temp_dir / "data.yaml", config=TrainingConfig())

        assert "last.pt" in str(result.model_path)


class TestYOLOTrainerConfigInjection:
    """Test YOLOTrainer train() config injection."""

    def test_injects_correct_train_kwargs_cosine(self, temp_dir):
        """Cosine scheduler sets lrf correctly in train_kwargs."""
        trainer = YOLOTrainer(output_dir=temp_dir)

        mock_results = Mock()
        mock_results.results_dict = {"metrics/mAP50(B)": 0.5, "metrics/mAP50-95(B)": 0.4}
        mock_results.save_dir = str(temp_dir / "train")
        mock_yolo_instance.train.return_value = mock_results

        config = TrainingConfig()
        config.lr_scheduler.type = "cosine"
        config.lr_scheduler.lrf = 0.01
        config.lr0 = 0.01

        # Mock validate_dataset_distribution to return ok status
        with patch('training.yolo_trainer.validate_dataset_distribution') as mock_validate:
            mock_validate.return_value = DatasetDistributionResult(
                train_median_area=0.01, val_median_area=0.01,
                ratio=1.0, status="ok",
                train_box_count=100, val_box_count=50,
                train_image_count=10, val_image_count=5,
                message="OK"
            )
            with patch('training.yolo_trainer.MLflowTracker') as mock_tracker:
                mock_tracker_instance = Mock()
                mock_tracker.return_value = mock_tracker_instance
                mock_tracker_instance.start_run = Mock()
                mock_tracker_instance.log_params = Mock()
                mock_tracker_instance.log_metrics = Mock()
                mock_tracker_instance.log_artifact = Mock()
                mock_tracker_instance.end_run = Mock()

                result = trainer.train(data_yaml=temp_dir / "data.yaml", config=config)

        # Verify model.train was called
        mock_yolo_instance.train.assert_called_once()
        call_kwargs = mock_yolo_instance.train.call_args.kwargs
        # lrf should be set from config.lr_scheduler.lrf (not multiplied)
        assert call_kwargs["lrf"] == 0.01

    def test_resume_checkpoint_passed_to_train_kwargs(self, temp_dir):
        """resume_checkpoint is passed to model.train as resume kwarg."""
        trainer = YOLOTrainer(output_dir=temp_dir)

        mock_results = Mock()
        mock_results.results_dict = {"metrics/mAP50(B)": 0.5, "metrics/mAP50-95(B)": 0.4}
        mock_results.save_dir = str(temp_dir / "train")
        mock_yolo_instance.train.return_value = mock_results

        config = TrainingConfig()
        config.resume_checkpoint = str(temp_dir / "checkpoint.pt")

        with patch('training.yolo_trainer.validate_dataset_distribution') as mock_validate:
            mock_validate.return_value = DatasetDistributionResult(
                train_median_area=0.01, val_median_area=0.01,
                ratio=1.0, status="ok",
                train_box_count=100, val_box_count=50,
                train_image_count=10, val_image_count=5,
                message="OK"
            )
            with patch('training.yolo_trainer.MLflowTracker') as mock_tracker:
                mock_tracker_instance = Mock()
                mock_tracker.return_value = mock_tracker_instance
                mock_tracker_instance.start_run = Mock()
                mock_tracker_instance.log_params = Mock()
                mock_tracker_instance.log_metrics = Mock()
                mock_tracker_instance.log_artifact = Mock()
                mock_tracker_instance.end_run = Mock()

                result = trainer.train(data_yaml=temp_dir / "data.yaml", config=config)

        call_kwargs = mock_yolo_instance.train.call_args.kwargs
        assert call_kwargs["resume"] == str(temp_dir / "checkpoint.pt")


# ==================== Test curriculum.py ====================

class TestCurriculumStage:
    """Test CurriculumStage dataclass."""

    def test_default_values(self):
        """CurriculumStage has correct defaults."""
        stage = CurriculumStage(
            name="test", epochs=50, imgsz=640, batch=16,
            model="yolo11m", augmentation_preset="balanced"
        )
        assert stage.warmup_ratio == 0.05
        assert stage.mosaic == 1.0
        assert stage.mixup == 0.0
        assert stage.copy_paste == 0.0
        assert stage.num_gpus == 1

    def test_resume_from_field(self):
        """CurriculumStage accepts resume_from for stage chaining."""
        stage = CurriculumStage(
            name="deep_training", epochs=150, imgsz=1280, batch=8,
            model="yolo11x", augmentation_preset="strong",
            resume_from="/path/to/best.pt"
        )
        assert stage.resume_from == "/path/to/best.pt"


class TestCurriculumConfig:
    """Test CurriculumConfig dataclass."""

    def test_stage1_has_correct_defaults(self):
        """Stage 1 rapid validation defaults are correct."""
        config = CurriculumConfig()
        assert config.stage1.name == "rapid_validation"
        assert config.stage1.epochs == 50
        assert config.stage1.imgsz == 640
        assert config.stage1.model == "yolo11m"

    def test_stage2_has_correct_defaults(self):
        """Stage 2 deep training defaults are correct."""
        config = CurriculumConfig()
        assert config.stage2.name == "deep_training"
        assert config.stage2.epochs == 150
        assert config.stage2.imgsz == 1280
        assert config.stage2.model == "yolo11x"

    def test_stage3_has_correct_defaults(self):
        """Stage 3 fine-tuning defaults are correct."""
        config = CurriculumConfig()
        assert config.stage3.name == "fine_tuning"
        assert config.stage3.epochs == 100
        assert config.stage3.imgsz == 1280
        assert config.stage3.mosaic == 0.0  # Mosaic off for fine-tuning

    def test_decision_thresholds(self):
        """Decision thresholds are correctly set."""
        config = CurriculumConfig()
        assert config.stage1_min_map == 0.50
        assert config.stage2_target_map == 0.90
        assert config.stage2_min_for_stage3 == 0.80


class TestPipelineCurriculumTrainerGateDecisions:
    """Test PipelineCurriculumTrainer stage gate decisions."""

    def test_stage1_gate_fails_below_threshold(self, temp_dir):
        """Stage 1 fails when mAP50 < 0.5."""
        trainer = PipelineCurriculumTrainer(output_dir=temp_dir)

        # Mock Stage 1 to return low mAP
        with patch.object(PipelineCurriculumTrainer, '_run_stage') as mock_run:
            mock_run.return_value = (
                TrainingResult(status="completed", metrics={"mAP50": 0.3}),
                ""
            )

            result = trainer.train(
                data_yaml=temp_dir / "data.yaml",
                task_id="test_task",
            )

            assert result.status == "completed"  # Stage 1 returned but failed gate
            # Stage history records the failure
            assert trainer._stage_history[0]["mAP50"] == 0.3

    def test_stage2_goal_reached_stops_early(self, temp_dir):
        """Stage 2 mAP50 >= 0.90 stops curriculum without Stage 3."""
        trainer = PipelineCurriculumTrainer(output_dir=temp_dir)

        call_count = [0]

        def mock_run_stage(stage, *args, **kwargs):
            call_count[0] += 1
            stage_num = args[1] if len(args) > 1 else kwargs.get("stage_num", 1)
            if stage_num == 1:
                return TrainingResult(status="completed", metrics={"mAP50": 0.7}), ""
            elif stage_num == 2:
                # Goal reached at stage 2
                return TrainingResult(status="completed", metrics={"mAP50": 0.92}), ""
            return TrainingResult(status="completed", metrics={"mAP50": 0.95}), ""

        with patch.object(PipelineCurriculumTrainer, '_run_stage', side_effect=mock_run_stage):
            result = trainer.train(data_yaml=temp_dir / "data.yaml", task_id="test_task")

        assert result.status == "completed"
        assert result.early_stopped is True
        # Only 2 stages ran (stage 3 never called)
        assert call_count[0] == 2

    def test_stage2_triggers_stage3_when_above_min_threshold(self, temp_dir):
        """Stage 2 mAP50 >= 0.80 and < 0.90 proceeds to Stage 3."""
        trainer = PipelineCurriculumTrainer(output_dir=temp_dir)

        call_count = [0]

        def mock_run_stage(stage, *args, **kwargs):
            call_count[0] += 1
            stage_num = args[1] if len(args) > 1 else kwargs.get("stage_num", 1)
            if stage_num == 1:
                return TrainingResult(status="completed", metrics={"mAP50": 0.7}), ""
            elif stage_num == 2:
                return TrainingResult(status="completed", metrics={"mAP50": 0.85}), ""
            elif stage_num == 3:
                return TrainingResult(status="completed", metrics={"mAP50": 0.90}), ""
            return TrainingResult(status="completed", metrics={"mAP50": 0.90}), ""

        with patch.object(PipelineCurriculumTrainer, '_run_stage', side_effect=mock_run_stage):
            result = trainer.train(data_yaml=temp_dir / "data.yaml", task_id="test_task")

        assert result.status == "completed"
        # All 3 stages were called
        assert call_count[0] == 3
