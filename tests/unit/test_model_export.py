# Unit Tests - Model Exporter Module

import pytest
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Mock ultralytics before importing the module
sys.modules['ultralytics'] = MagicMock()

# Add src to path - handle both direct and package execution
test_dir = Path(__file__).parent
project_root = test_dir.parent.parent
training_api_path = project_root / "training-api"
src_path = training_api_path / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from deployment.exporter import ModelExporter, ExportResult, EdgeDeployer, get_benchmark


# ==================== Test ModelExporter Initialization ====================

class TestModelExporterInit:
    """Test ModelExporter initialization."""

    def test_initialization_sets_format(self):
        """Test initialization sets default format."""
        exporter = ModelExporter()
        assert hasattr(exporter, 'output_dir')
        assert exporter.output_dir == Path("./runs/export")

    def test_initialization_sets_output_dir(self, temp_dir):
        """Test initialization with custom output directory."""
        exporter = ModelExporter(output_dir=temp_dir)
        assert exporter.output_dir == temp_dir

    def test_output_dir_created_if_not_exists(self, temp_dir):
        """Test that output directory is created."""
        new_dir = temp_dir / "new_export"
        exporter = ModelExporter(output_dir=new_dir)
        assert new_dir.exists()

    def test_platform_configs_defined(self):
        """Test that platform configs are defined."""
        exporter = ModelExporter()
        assert hasattr(exporter, 'PLATFORM_CONFIGS')
        assert len(exporter.PLATFORM_CONFIGS) > 0


# ==================== Test Export Formats ====================

class TestExportFormats:
    """Test different export format methods."""

    @pytest.fixture
    def exporter(self, temp_dir):
        """Create ModelExporter instance."""
        return ModelExporter(output_dir=temp_dir)

    @pytest.fixture
    def mock_model_file(self, temp_dir):
        """Create a mock model file."""
        model_path = temp_dir / "model.pt"
        model_path.write_text("mock model content")
        return model_path

    def test_export_to_onnx(self, exporter, mock_model_file, temp_dir):
        """Test export to ONNX format."""
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.return_value = str(temp_dir / "model.onnx")

        # Mock the file stat to return a size
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value = MagicMock(st_size=1024 * 1024)  # 1MB

            with patch('deployment.exporter.YOLO') as mock_yolo:
                mock_yolo.return_value = mock_yolo_instance

                result = exporter.export(
                    model_path=mock_model_file,
                    platform="onnx",
                    imgsz=640,
                )

                assert result.status == "success"
                assert result.format == "onnx"
                assert result.platform == "onnx"

    def test_export_to_tensorrt(self, exporter, mock_model_file, temp_dir):
        """Test export to TensorRT format."""
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.return_value = str(temp_dir / "model.engine")

        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value = MagicMock(st_size=2 * 1024 * 1024)  # 2MB

            with patch('deployment.exporter.YOLO') as mock_yolo:
                mock_yolo.return_value = mock_yolo_instance

                result = exporter.export(
                    model_path=mock_model_file,
                    platform="tensorrt",
                    imgsz=640,
                )

                assert result.status == "success"
                assert result.format == "engine"
                assert result.platform == "tensorrt"

    def test_export_to_tflite(self, exporter, mock_model_file, temp_dir):
        """Test export to TFLite format."""
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.return_value = str(temp_dir / "model.tflite")

        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value = MagicMock(st_size=512 * 1024)  # 512KB

            with patch('deployment.exporter.YOLO') as mock_yolo:
                mock_yolo.return_value = mock_yolo_instance

                result = exporter.export(
                    model_path=mock_model_file,
                    platform="tflite",
                    imgsz=640,
                )

                # TFLite not in PLATFORM_CONFIGS, defaults to onnx
                assert result.status == "success"

    def test_export_to_rknn(self, exporter, mock_model_file, temp_dir):
        """Test export to RKNN format."""
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.return_value = str(temp_dir / "model.rknn")

        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value = MagicMock(st_size=768 * 1024)  # 768KB

            with patch('deployment.exporter.YOLO') as mock_yolo:
                mock_yolo.return_value = mock_yolo_instance

                result = exporter.export(
                    model_path=mock_model_file,
                    platform="rknn",
                    imgsz=640,
                )

                # RKNN not in PLATFORM_CONFIGS, defaults to onnx
                assert result.status == "success"

    def test_export_jetson_orin(self, exporter, mock_model_file, temp_dir):
        """Test export to Jetson Orin platform."""
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.return_value = str(temp_dir / "model.engine")

        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value = MagicMock(st_size=3 * 1024 * 1024)

            with patch('deployment.exporter.YOLO') as mock_yolo:
                mock_yolo.return_value = mock_yolo_instance

                result = exporter.export(
                    model_path=mock_model_file,
                    platform="jetson_orin",
                    imgsz=640,
                )

                assert result.status == "success"
                assert result.format == "engine"
                assert result.platform == "jetson_orin"

    def test_export_jetson_nano(self, exporter, mock_model_file, temp_dir):
        """Test export to Jetson Nano platform."""
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.return_value = str(temp_dir / "model.engine")

        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value = MagicMock(st_size=2 * 1024 * 1024)

            with patch('deployment.exporter.YOLO') as mock_yolo:
                mock_yolo.return_value = mock_yolo_instance

                result = exporter.export(
                    model_path=mock_model_file,
                    platform="jetson_nano",
                    imgsz=640,
                )

                assert result.status == "success"
                assert result.format == "engine"


# ==================== Test Export Validation ====================

class TestExportValidation:
    """Test export validation logic."""

    @pytest.fixture
    def exporter(self, temp_dir):
        """Create ModelExporter instance."""
        return ModelExporter(output_dir=temp_dir)

    def test_validate_model_path_valid(self, exporter, temp_dir):
        """Test validation of valid model path."""
        model_path = temp_dir / "model.pt"
        model_path.touch()

        # Should not raise - the export method handles this
        exporter.export(model_path=model_path, platform="onnx", imgsz=640)

    def test_validate_model_path_invalid(self, exporter, temp_dir):
        """Test validation of invalid model path."""
        model_path = temp_dir / "nonexistent.pt"

        # Should handle gracefully (the YOLO constructor will fail)
        result = exporter.export(model_path=model_path, platform="onnx", imgsz=640)
        assert result.status == "failed"
        assert result.error is not None

    def test_validate_platform_supported(self, exporter, mock_model_file, temp_dir):
        """Test validation of supported platform."""
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.return_value = str(temp_dir / "model.onnx")

        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value = MagicMock(st_size=1024 * 1024)

            with patch('deployment.exporter.YOLO') as mock_yolo:
                mock_yolo.return_value = mock_yolo_instance

                result = exporter.export(
                    model_path=mock_model_file,
                    platform="onnx",
                    imgsz=640,
                )

                assert result.platform == "onnx"

    def test_validate_platform_unsupported(self, exporter, mock_model_file, temp_dir):
        """Test validation of unsupported platform defaults to onnx."""
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.return_value = str(temp_dir / "model.onnx")

        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value = MagicMock(st_size=1024 * 1024)

            with patch('deployment.exporter.YOLO') as mock_yolo:
                mock_yolo.return_value = mock_yolo_instance

                result = exporter.export(
                    model_path=mock_model_file,
                    platform="unsupported_platform",
                    imgsz=640,
                )

                # Should default to onnx
                assert result.format == "onnx"

    def test_export_error_handling(self, exporter, mock_model_file):
        """Test export error handling during export (not during model loading)."""
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.side_effect = Exception("Export failed")

        # Create the model file to avoid file not found error
        mock_model_file.touch()

        with patch('deployment.exporter.YOLO') as mock_yolo:
            mock_yolo.return_value = mock_yolo_instance

            result = exporter.export(
                model_path=mock_model_file,
                platform="onnx",
                imgsz=640,
            )

            assert result.status == "failed"
            assert "Export failed" in result.error


# ==================== Test Export Config ====================

class TestExportConfig:
    """Test export configuration."""

    def test_default_export_config(self):
        """Test default export configuration."""
        exporter = ModelExporter()

        # Check default platform config
        assert "onnx" in exporter.PLATFORM_CONFIGS
        onnx_config = exporter.PLATFORM_CONFIGS["onnx"]
        assert onnx_config["format"] == "onnx"
        assert onnx_config["half"] is False
        assert onnx_config["opset"] == 13

    def test_custom_export_config(self, temp_dir):
        """Test custom export configuration."""
        exporter = ModelExporter(output_dir=temp_dir)
        assert exporter.output_dir == temp_dir

    def test_platform_specific_config(self):
        """Test platform-specific configurations."""
        exporter = ModelExporter()

        # Jetson Orin config
        jetson_orin = exporter.PLATFORM_CONFIGS["jetson_orin"]
        assert jetson_orin["format"] == "engine"
        assert jetson_orin["half"] is True
        assert jetson_orin["dynamic"] is True
        assert jetson_orin["workspace"] == 4

        # Jetson Nano config
        jetson_nano = exporter.PLATFORM_CONFIGS["jetson_nano"]
        assert jetson_nano["format"] == "engine"
        assert jetson_nano["half"] is True
        assert jetson_nano["dynamic"] is False
        assert jetson_nano["workspace"] == 2

        # TensorRT config
        tensorrt = exporter.PLATFORM_CONFIGS["tensorrt"]
        assert tensorrt["format"] == "engine"
        assert tensorrt["half"] is True

        # ONNX FP16 config
        onnx_fp16 = exporter.PLATFORM_CONFIGS["onnx_fp16"]
        assert onnx_fp16["format"] == "onnx"
        assert onnx_fp16["half"] is True

        # CPU config
        cpu = exporter.PLATFORM_CONFIGS["cpu"]
        assert cpu["format"] == "onnx"
        assert cpu["half"] is False

    def test_all_platforms_have_format(self):
        """Test all platforms have format defined."""
        exporter = ModelExporter()
        for platform, config in exporter.PLATFORM_CONFIGS.items():
            assert "format" in config, f"{platform} missing format"


# ==================== Test Export Result ====================

class TestExportResult:
    """Test ExportResult dataclass."""

    def test_export_result_contains_path(self):
        """Test ExportResult contains model path."""
        result = ExportResult(
            status="success",
            model_path=Path("/path/to/model.onnx"),
            size_mb=10.5,
            format="onnx",
            platform="onnx",
        )
        assert result.model_path == Path("/path/to/model.onnx")
        assert result.status == "success"

    def test_export_result_contains_size(self):
        """Test ExportResult contains size."""
        result = ExportResult(
            status="success",
            model_path=Path("/path/to/model.onnx"),
            size_mb=15.3,
            format="onnx",
            platform="onnx",
        )
        assert result.size_mb == 15.3

    def test_export_result_contains_platform(self):
        """Test ExportResult contains platform."""
        result = ExportResult(
            status="success",
            model_path=Path("/path/to/model.engine"),
            size_mb=20.0,
            format="engine",
            platform="jetson_orin",
        )
        assert result.platform == "jetson_orin"

    def test_export_result_default_values(self):
        """Test ExportResult default values."""
        result = ExportResult(status="failed", error="Test error")
        assert result.status == "failed"
        assert result.error == "Test error"
        assert result.model_path is None
        assert result.size_mb == 0.0
        assert result.format == ""
        assert result.platform == ""

    def test_export_result_contains_format(self):
        """Test ExportResult contains format."""
        result = ExportResult(
            status="success",
            model_path=Path("/path/to/model.engine"),
            size_mb=20.0,
            format="engine",
            platform="tensorrt",
        )
        assert result.format == "engine"


# ==================== Test INT8 Calibration Export ====================

class TestExportInt8Calibration:
    """Test INT8 calibration export."""

    @pytest.fixture
    def exporter(self, temp_dir):
        """Create ModelExporter instance."""
        return ModelExporter(output_dir=temp_dir)

    @pytest.fixture
    def mock_model_file(self, temp_dir):
        """Create a mock model file."""
        model_path = temp_dir / "model.pt"
        model_path.write_text("mock model content")
        return model_path

    @pytest.fixture
    def mock_calibration_data(self, temp_dir):
        """Create mock calibration data."""
        calib_dir = temp_dir / "calibration"
        calib_dir.mkdir()
        (calib_dir / "images").mkdir()
        return calib_dir

    def test_export_int8_calibration(self, exporter, mock_model_file, mock_calibration_data, temp_dir):
        """Test INT8 calibration export."""
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.return_value = str(temp_dir / "model.engine")

        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value = MagicMock(st_size=1024 * 1024)

            with patch('deployment.exporter.YOLO') as mock_yolo:
                mock_yolo.return_value = mock_yolo_instance

                result = exporter.export_int8_calibration(
                    model_path=mock_model_file,
                    calibration_data=mock_calibration_data,
                    platform="jetson_orin",
                )

                assert result.status == "success"
                assert "_int8" in result.platform


# ==================== Test Edge Deployer ====================

class TestEdgeDeployer:
    """Test EdgeDeployer class."""

    def test_initialization(self):
        """Test EdgeDeployer initialization."""
        deployer = EdgeDeployer()
        assert deployer.ssh_key_path is None

    def test_deploy_to_jetson(self, temp_dir):
        """Test deploy to Jetson device."""
        deployer = EdgeDeployer()
        model_path = temp_dir / "model.onnx"

        result = deployer.deploy_to_jetson(
            model_path=model_path,
            jetson_ip="192.168.1.100",
            jetson_user="nvidia",
        )

        assert result["status"] == "pending"
        assert result["device"] == "nvidia@192.168.1.100"
        assert str(model_path) in result["model"]


# ==================== Test Benchmark Functions ====================

class TestBenchmarkFunctions:
    """Test benchmark helper functions."""

    def test_get_benchmark_jetson_orin_nx(self):
        """Test getting Jetson Orin NX benchmark."""
        benchmark = get_benchmark("jetson_orin_nx_16gb", "fp16")
        assert "fps" in benchmark
        assert "latency_ms" in benchmark

    def test_get_benchmark_jetson_nano(self):
        """Test getting Jetson Nano benchmark."""
        benchmark = get_benchmark("jetson_nano", "fp16")
        assert "fps" in benchmark

    def test_get_benchmark_unknown_platform(self):
        """Test getting benchmark for unknown platform."""
        benchmark = get_benchmark("unknown_platform", "fp16")
        assert benchmark == {}

    def test_get_benchmark_default_precision(self):
        """Test getting benchmark with default precision."""
        benchmark = get_benchmark("jetson_orin_nx_16gb")
        assert "fps" in benchmark


# ==================== Test Fixtures ====================

@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for tests."""
    return tmp_path


@pytest.fixture
def mock_model_file(temp_dir):
    """Create a mock model file."""
    model_path = temp_dir / "model.pt"
    model_path.write_text("mock model content")
    return model_path
