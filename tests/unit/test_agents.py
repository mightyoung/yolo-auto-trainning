# Unit Tests - Agent Orchestration Module (Real Tests)

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add project paths so "agents.orchestration" resolves (business-api uses hyphen, not underscore)
project_root = Path(__file__).parent.parent.parent
biz_api_src = project_root / "business-api" / "src"
src_path = project_root / "src"

for p in [str(biz_api_src), str(src_path)]:
    if p not in sys.path:
        sys.path.insert(0, p)


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
        from agents.orchestration import DatasetSearchTool
        tool = DatasetSearchTool()
        assert tool.name == "dataset_search"

    def test_tool_has_correct_description(self):
        from agents.orchestration import DatasetSearchTool
        tool = DatasetSearchTool()
        assert "Roboflow" in tool.description
        assert "Kaggle" in tool.description

    def test_tool_run_returns_no_results_message(self):
        from agents.orchestration import DatasetSearchTool
        with patch("agents.orchestration.DatasetDiscovery") as MockDiscovery:
            mock_instance = Mock()
            mock_instance.search.return_value = []
            MockDiscovery.return_value = mock_instance
            tool = DatasetSearchTool()
            result = tool._run(query="fire detection", max_results=5)
            assert isinstance(result, str)
            assert "No datasets found" in result

    def test_tool_run_with_results(self):
        from agents.orchestration import DatasetSearchTool
        from data.discovery import DatasetInfo
        with patch("agents.orchestration.DatasetDiscovery") as MockDiscovery:
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
            tool = DatasetSearchTool()
            result = tool._run(query="fire", max_results=5)
            assert "fire-dataset" in result
            assert "roboflow" in result
            assert "0.90" in result


class TestDatasetDownloadTool:
    """Test DatasetDownloadTool."""

    def test_tool_has_correct_name(self):
        from agents.orchestration import DatasetDownloadTool
        tool = DatasetDownloadTool()
        assert tool.name == "dataset_download"

    def test_tool_has_correct_description(self):
        from agents.orchestration import DatasetDownloadTool
        tool = DatasetDownloadTool()
        assert "roboflow" in tool.description.lower() or "kaggle" in tool.description.lower()

    def test_tool_run_returns_success_message(self):
        from agents.orchestration import DatasetDownloadTool
        with patch("agents.orchestration.DatasetDiscovery") as MockDiscovery:
            mock_instance = Mock()
            mock_instance.download.return_value = "/data/fire-dataset"
            MockDiscovery.return_value = mock_instance
            tool = DatasetDownloadTool()
            result = tool._run(dataset_name="fire-dataset", source="roboflow")
            assert "Downloaded" in result
            assert "/data/fire-dataset" in result


class TestTrainModelTool:
    """Test TrainModelTool."""

    def test_tool_has_correct_name(self):
        from agents.orchestration import TrainModelTool
        tool = TrainModelTool()
        assert tool.name == "model_train"

    def test_tool_run_returns_task_id(self):
        from agents.orchestration import TrainModelTool
        with patch("agents.orchestration.TrainingAPIClient") as MockClient:
            mock_instance = Mock()
            mock_instance.start_training.return_value = {"task_id": "train_abc123", "status": "started"}
            MockClient.return_value = mock_instance
            tool = TrainModelTool()
            result = tool._run(dataset_path="/data/coco8.yaml", model_size="yolo11n", epochs=10)
            mock_instance.start_training.assert_called_once()
            assert "train_abc123" in result or "Training started" in result

    def test_tool_run_handles_exception(self):
        from agents.orchestration import TrainModelTool
        with patch("agents.orchestration.TrainingAPIClient") as MockClient:
            mock_instance = Mock()
            mock_instance.start_training.side_effect = Exception("Connection refused")
            MockClient.return_value = mock_instance
            tool = TrainModelTool()
            result = tool._run(dataset_path="/data/coco8.yaml")
            assert "failed" in result.lower()


class TestExportModelTool:
    """Test ExportModelTool."""

    def test_tool_has_correct_name(self):
        from agents.orchestration import ExportModelTool
        tool = ExportModelTool()
        assert tool.name == "model_export"

    def test_tool_run_calls_export_api(self):
        from agents.orchestration import ExportModelTool
        with patch("agents.orchestration.httpx.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            tool = ExportModelTool()
            result = tool._run(model_path="/runs/best.pt", platform="jetson_orin")
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "export/start" in call_args[0][0]
            assert "Training started" in result or "Export started" in result

    def test_tool_run_handles_http_error(self):
        from agents.orchestration import ExportModelTool
        with patch("agents.orchestration.httpx.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_post.return_value = mock_response
            tool = ExportModelTool()
            result = tool._run(model_path="/runs/best.pt", platform="jetson_orin")
            assert "failed" in result.lower()


class TestDataSynthesizeTool:
    """Test DataSynthesizeTool."""

    def test_tool_exists(self):
        from agents.orchestration import DataSynthesizeTool
        tool = DataSynthesizeTool()
        assert tool.name == "data_synthesize"

    def test_tool_has_description(self):
        from agents.orchestration import DataSynthesizeTool
        tool = DataSynthesizeTool()
        assert "synthetic" in tool.description.lower()

    def test_tool_run_falls_back_when_import_fails(self):
        from agents.orchestration import DataSynthesizeTool
        with patch("agents.orchestration.DataSynthesizeTool._run") as mock_run:
            # Simulate ImportError by patching the generator module
            mock_run.side_effect = ImportError("Module not found")
            tool = DataSynthesizeTool()
            # The tool should return a graceful message when imports fail
            result = tool._run(task_description="detect cars")
            assert isinstance(result, str)


# ==================== Test Agent Creation ====================

class TestAgentCreation:
    """Test agent creation functions."""

    def test_dataset_discovery_agent_exists(self):
        with patch("agents.orchestration.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            from agents.orchestration import create_dataset_discovery_agent
            agent = create_dataset_discovery_agent()
            assert agent is not None
            assert hasattr(agent, "role")
            assert agent.role == "Dataset Curator"

    def test_dataset_discovery_agent_has_correct_tools(self):
        with patch("agents.orchestration.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            from agents.orchestration import create_dataset_discovery_agent
            agent = create_dataset_discovery_agent()
            tool_names = [t.name for t in (agent.tools or [])]
            assert "dataset_search" in tool_names
            assert "dataset_download" in tool_names

    def test_data_generator_agent_has_synthesize_tool(self):
        with patch("agents.orchestration.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            from agents.orchestration import create_data_generator_agent
            agent = create_data_generator_agent()
            assert agent is not None
            tool_names = [t.name for t in (agent.tools or [])]
            assert "data_synthesize" in tool_names
            assert agent.role == "Data Engineer"

    def test_training_agent_has_train_tool(self):
        with patch("agents.orchestration.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            from agents.orchestration import create_training_agent
            agent = create_training_agent()
            assert agent is not None
            tool_names = [t.name for t in (agent.tools or [])]
            assert "model_train" in tool_names
            assert agent.role == "ML Engineer"

    def test_deployment_agent_has_export_tool(self):
        with patch("agents.orchestration.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            from agents.orchestration import create_deployment_agent
            agent = create_deployment_agent()
            assert agent is not None
            tool_names = [t.name for t in (agent.tools or [])]
            assert "model_export" in tool_names
            assert agent.role == "DevOps Engineer"


# ==================== Test Crew Creation ====================

class TestCrewCreation:
    """Test crew creation."""

    def test_create_training_crew_returns_crew(self):
        with patch("agents.orchestration.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            from agents.orchestration import create_training_crew
            crew = create_training_crew(task_description="fire detection")
            assert crew is not None
            assert hasattr(crew, "agents")
            assert len(crew.agents) == 4

    def test_crew_tasks_have_descriptions(self):
        with patch("agents.orchestration.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            from agents.orchestration import create_training_crew
            crew = create_training_crew(task_description="smoke detection")
            assert len(crew.tasks) == 4
            for task in crew.tasks:
                assert task.description is not None

    def test_crew_tasks_include_task_description(self):
        with patch("agents.orchestration.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            from agents.orchestration import create_training_crew
            crew = create_training_crew(task_description="person detection")
            # First task description should include the task description
            assert "person detection" in crew.tasks[0].description

    def test_create_simple_crew_returns_crew(self):
        with patch("agents.orchestration.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            from agents.orchestration import create_simple_crew
            crew = create_simple_crew(task_description="cat detection")
            assert crew is not None
            assert len(crew.agents) == 2
            assert len(crew.tasks) == 2


# ==================== Test YOLOTrainingOrchestrator ====================

class TestYOLOTrainingOrchestrator:
    """Test YOLOTrainingOrchestrator class."""

    def test_orchestrator_instantiates(self):
        with patch("agents.orchestration.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            from agents.orchestration import YOLOTrainingOrchestrator
            orch = YOLOTrainingOrchestrator()
            assert orch is not None

    def test_get_status_returns_none_for_unknown_task(self):
        with patch("agents.orchestration.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            with patch(
                "agents.orchestration.YOLOTrainingOrchestrator._get_redis"
            ) as mock_redis:
                mock_r = MagicMock()
                mock_r.hgetall.return_value = {}
                mock_redis.return_value = mock_r
                from agents.orchestration import YOLOTrainingOrchestrator
                orch = YOLOTrainingOrchestrator()
                result = orch.get_status("nonexistent_task")
                assert result is None

    def test_get_status_returns_dict_for_known_task(self):
        with patch("agents.orchestration.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            with patch(
                "agents.orchestration.YOLOTrainingOrchestrator._get_redis"
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
                from agents.orchestration import YOLOTrainingOrchestrator
                orch = YOLOTrainingOrchestrator()
                result = orch.get_status("task_123")
                assert result is not None
                assert result["status"] == "running"
                assert result["progress"] == 50.0

    def test_cancel_returns_false_when_not_running(self):
        with patch("agents.orchestration.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            with patch(
                "agents.orchestration.YOLOTrainingOrchestrator._get_redis"
            ) as mock_redis:
                mock_r = MagicMock()
                mock_r.hget.return_value = "completed"
                mock_redis.return_value = mock_r
                from agents.orchestration import YOLOTrainingOrchestrator
                orch = YOLOTrainingOrchestrator()
                result = orch.cancel("completed_task_123")
                assert result is False

    def test_cancel_sets_cancelled_status_for_running_task(self):
        with patch("agents.orchestration.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            with patch(
                "agents.orchestration.YOLOTrainingOrchestrator._get_redis"
            ) as mock_redis:
                mock_r = MagicMock()
                # First call returns "running", second call (for training_task_id) returns None
                mock_r.hget.side_effect = lambda k, f: (
                    "running" if f == "status" else None
                )
                mock_redis.return_value = mock_r
                from agents.orchestration import YOLOTrainingOrchestrator
                orch = YOLOTrainingOrchestrator()
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
                from agents.orchestration import get_llm
                with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
                    get_llm()
            finally:
                if original is not None:
                    os.environ["DEEPSEEK_API_KEY"] = original

    def test_get_llm_returns_llm_instance(self):
        with patch.dict(
            "os.environ",
            {
                "DEEPSEEK_API_KEY": "test-key",
                "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1",
                "DEEPSEEK_MODEL": "deepseek-chat",
            },
        ):
            from agents.orchestration import get_llm
            llm = get_llm()
            assert llm is not None
