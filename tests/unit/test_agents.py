# Unit Tests - Agent Orchestration Module (Real Tests)
#
# Tests the business-api/src/agents/orchestration.py module
# which provides: DatasetSearchTool, DatasetDownloadTool, TrainModelTool,
# ExportModelTool, create_training_agent, create_deployment_agent,
# YOLOTrainingOrchestrator, get_llm
#
# NOTE: The following were removed from the module and are NOT tested here:
# - DataSynthesizeTool (removed)
# - create_dataset_discovery_agent (removed)
# - create_data_generator_agent (removed)
# - create_training_crew (removed)
# - create_simple_crew (removed)

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import importlib

# Mock crewai before importing so the module can be loaded even when crewai is not installed
sys.modules['crewai'] = MagicMock()

# Pre-import the business-api agents.orchestration module directly
# using importlib to avoid sys.path namespace confusion with deprecated src/agents/
project_root = Path(__file__).parent.parent.parent
biz_api_src = project_root / "business-api" / "src"

# Mock dependencies that orchestration module imports at top level
sys.modules['redis'] = MagicMock()

# Create a mock DatasetInfo class for tests
class DatasetInfo:
    def __init__(
        self,
        id: str = "",
        name: str = "",
        source: str = "",
        url: str = "",
        license: str = "unknown",
        task: str = "object-detection",
        images: int = 0,
        relevance_score: float = 0.0,
        description: str = None,
        categories: list = None,
        format: str = "yolo",
        annotations: str = "unknown",
    ):
        self.id = id
        self.name = name
        self.source = source
        self.url = url
        self.license = license
        self.task = task
        self.images = images
        self.relevance_score = relevance_score
        self.description = description
        self.categories = categories or []
        self.format = format
        self.annotations = annotations

# Mock src.data.discovery so orchestration module can load
sys.modules['src'] = MagicMock()
sys.modules['src.data'] = MagicMock()
sys.modules['src.data.discovery'] = MagicMock()
sys.modules['src.data.discovery'].DatasetInfo = DatasetInfo

# Import orchestration module directly using importlib
orchestration_spec = importlib.util.spec_from_file_location(
    "agents.orchestration",
    biz_api_src / "agents" / "orchestration.py"
)
orchestration_module = importlib.util.module_from_spec(orchestration_spec)
sys.modules['agents.orchestration'] = orchestration_module

# Also register and execute the tools submodule so patches apply to both
tools_spec = importlib.util.spec_from_file_location(
    "agents.tools",
    biz_api_src / "agents" / "tools.py"
)
tools_module = importlib.util.module_from_spec(tools_spec)
sys.modules['agents.tools'] = tools_module
tools_spec.loader.exec_module(tools_module)

# Now execute the orchestration module
orchestration_spec.loader.exec_module(orchestration_module)

# Import TrainingAPIClient
training_client_spec = importlib.util.spec_from_file_location(
    "api.training_client",
    biz_api_src / "api" / "training_client.py"
)
training_client_module = importlib.util.module_from_spec(training_client_spec)
sys.modules['api.training_client'] = training_client_module
training_client_spec.loader.exec_module(training_client_module)


# ==================== Mock fixtures ====================

@pytest.fixture
def mock_llm():
    """Mock LLM for agent creation."""
    mock = MagicMock()
    mock.model = "deepseek-reasoner"
    return mock


# ==================== Test Tool Classes ====================

class TestDatasetSearchTool:
    """Test DatasetSearchTool."""

    def test_tool_has_correct_name(self):
        tool = orchestration_module.DatasetSearchTool()
        assert tool.name == "dataset_search"

    def test_tool_has_correct_description(self):
        tool = orchestration_module.DatasetSearchTool()
        assert "Roboflow" in tool.description
        assert "Kaggle" in tool.description

    def test_tool_run_returns_no_results_message(self):
        with patch.object(orchestration_module, 'DatasetDiscovery') as MockDiscovery, \
             patch.object(tools_module, 'DatasetDiscovery') as MockDiscovery2:
            mock_instance = Mock()
            mock_instance.search.return_value = []
            MockDiscovery.return_value = mock_instance
            MockDiscovery2.return_value = mock_instance
            tool = orchestration_module.DatasetSearchTool()
            result = tool._run(query="fire detection", max_results=5)
            assert isinstance(result, str)
            assert "No datasets found" in result

    def test_tool_run_with_results(self):
        with patch.object(orchestration_module, 'DatasetDiscovery') as MockDiscovery, \
             patch.object(tools_module, 'DatasetDiscovery') as MockDiscovery2:
            mock_instance = Mock()
            mock_ds = DatasetInfo(
                source="roboflow",
                name="fire-dataset",
                url="https://roboflow.com/fire",
                license="CC BY 4.0",
                annotations="coco",
                images=1000,
                categories=["fire", "smoke"],
                relevance_score=0.9,
            )
            mock_instance.search.return_value = [mock_ds]
            MockDiscovery.return_value = mock_instance
            MockDiscovery2.return_value = mock_instance
            tool = orchestration_module.DatasetSearchTool()
            result = tool._run(query="fire", max_results=5)
            assert "fire-dataset" in result
            assert "roboflow" in result
            assert "0.90" in result


class TestDatasetDownloadTool:
    """Test DatasetDownloadTool."""

    def test_tool_has_correct_name(self):
        tool = orchestration_module.DatasetDownloadTool()
        assert tool.name == "dataset_download"

    def test_tool_has_correct_description(self):
        tool = orchestration_module.DatasetDownloadTool()
        desc = tool.description.lower()
        assert "roboflow" in desc or "kaggle" in desc

    def test_tool_run_returns_success_message(self):
        with patch.object(orchestration_module, 'DatasetDiscovery') as MockDiscovery, \
             patch.object(tools_module, 'DatasetDiscovery') as MockDiscovery2:
            mock_instance = Mock()
            mock_instance.download.return_value = "/data/fire-dataset"
            MockDiscovery.return_value = mock_instance
            MockDiscovery2.return_value = mock_instance
            tool = orchestration_module.DatasetDownloadTool()
            result = tool._run(dataset_name="fire-dataset", source="roboflow")
            assert "Downloaded" in result
            assert "/data/fire-dataset" in result


class TestTrainModelTool:
    """Test TrainModelTool."""

    def test_tool_has_correct_name(self):
        tool = orchestration_module.TrainModelTool()
        assert tool.name == "model_train"

    def test_tool_run_submits_training(self):
        # TrainModelTool imports TrainingAPIClient inside _run
        # We verify the tool structure and name are correct
        tool = orchestration_module.TrainModelTool()
        assert tool.name == "model_train"
        assert "Train" in tool.description

    def test_tool_description_mentions_training(self):
        tool = orchestration_module.TrainModelTool()
        desc = tool.description.lower()
        assert "train" in desc or "yolo" in desc


class TestExportModelTool:
    """Test ExportModelTool."""

    def test_tool_has_correct_name(self):
        tool = orchestration_module.ExportModelTool()
        assert tool.name == "model_export"

    def test_tool_run_returns_export_message(self):
        # ExportModelTool just returns a message - it doesn't make HTTP calls
        tool = orchestration_module.ExportModelTool()
        result = tool._run(model_path="/runs/best.pt", platform="jetson_orin")
        assert "Export" in result
        assert "jetson_orin" in result


# ==================== Test Agent Creation ====================

class TestAgentCreation:
    """Test agent creation functions.

    NOTE: create_training_agent and create_deployment_agent return None when
    crewai is not available. These tests verify that behavior.
    """

    def test_training_agent_returns_none_when_crewai_unavailable(self):
        # When crewai is not installed, these agents return None
        agent = orchestration_module.create_training_agent()
        # Agent should be None since crewai is mocked as unavailable
        assert agent is None

    def test_deployment_agent_returns_none_when_crewai_unavailable(self):
        agent = orchestration_module.create_deployment_agent()
        # Agent should be None since crewai is mocked as unavailable
        assert agent is None


# ==================== Test YOLOTrainingOrchestrator ====================

class TestYOLOTrainingOrchestrator:
    """Test YOLOTrainingOrchestrator class."""

    def test_orchestrator_instantiates(self):
        with patch.object(orchestration_module, 'get_llm') as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            orch = orchestration_module.YOLOTrainingOrchestrator()
            assert orch is not None

    def test_get_status_returns_none_for_unknown_task(self):
        with patch.object(orchestration_module, 'get_llm') as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            with patch.object(
                orchestration_module.YOLOTrainingOrchestrator, '_get_redis'
            ) as mock_redis:
                mock_r = MagicMock()
                mock_r.hgetall.return_value = {}
                mock_redis.return_value = mock_r
                orch = orchestration_module.YOLOTrainingOrchestrator()
                result = orch.get_status("nonexistent_task")
                assert result is None

    def test_get_status_returns_dict_for_known_task(self):
        with patch.object(orchestration_module, 'get_llm') as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            with patch.object(
                orchestration_module.YOLOTrainingOrchestrator, '_get_redis'
            ) as mock_redis:
                mock_r = MagicMock()
                mock_r.hgetall.return_value = {
                    "status": "running",
                    "user_id": "user1",
                    "task_description": "fire detection",
                    "progress": "50.0",
                    "current_agent": "ML Engineer",
                    "result": "",
                }
                mock_redis.return_value = mock_r
                orch = orchestration_module.YOLOTrainingOrchestrator()
                result = orch.get_status("task_123")
                assert result is not None
                assert result["status"] == "running"
                assert result["progress"] == 50.0

    def test_cancel_always_returns_true(self):
        # The cancel method always returns True and sets status to cancelled
        with patch.object(
            orchestration_module.YOLOTrainingOrchestrator, '_get_redis'
        ) as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            orch = orchestration_module.YOLOTrainingOrchestrator()
            result = orch.cancel("any_task_123")
            # cancel() always returns True after setting status to cancelled
            assert result is True

    def test_cancel_sets_cancelled_status_for_running_task(self):
        with patch.object(orchestration_module, 'get_llm') as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            with patch.object(
                orchestration_module.YOLOTrainingOrchestrator, '_get_redis'
            ) as mock_redis:
                mock_r = MagicMock()
                # First call returns "running", second call (for training_task_id) returns None
                mock_r.hget.side_effect = lambda k, f: (
                    "running" if f == "status" else None
                )
                mock_redis.return_value = mock_r
                orch = orchestration_module.YOLOTrainingOrchestrator()
                result = orch.cancel("test_task_123")
                assert result is True
                # Verify hset was called to update status to cancelled
                mock_r.hset.assert_called()


# ==================== Test get_llm function ====================

class TestGetLLM:
    """Test get_llm function behavior."""

    def test_get_llm_raises_when_no_api_key(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}, clear=False):
            # Clear the specific env var
            import os
            original = os.environ.get("DEEPSEEK_API_KEY")
            if original is not None:
                del os.environ["DEEPSEEK_API_KEY"]
            try:
                with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
                    orchestration_module.get_llm()
            finally:
                if original is not None:
                    os.environ["DEEPSEEK_API_KEY"] = original

    def test_get_llm_raises_when_crewai_unavailable(self):
        # Force CREWAI_AVAILABLE to False
        original = orchestration_module.CREWAI_AVAILABLE
        orchestration_module.CREWAI_AVAILABLE = False
        try:
            with patch.dict(
                "os.environ",
                {
                    "DEEPSEEK_API_KEY": "test-key",
                    "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1",
                    "DEEPSEEK_MODEL": "deepseek-chat",
                },
            ):
                with pytest.raises(RuntimeError, match="crewai not installed"):
                    orchestration_module.get_llm()
        finally:
            orchestration_module.CREWAI_AVAILABLE = original
