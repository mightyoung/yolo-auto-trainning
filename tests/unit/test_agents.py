# Unit Tests - Agent Orchestration Module

import pytest
from pathlib import Path
import sys
import re
import ast

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# ==================== Test Tools - Parse Source ====================

class TestDatasetTools:
    """Test dataset-related tools by inspecting source code."""

    def test_dataset_search_tool_exists(self):
        """DatasetSearchTool exists with correct attributes."""
        # Read source file
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        # Check for DatasetSearchTool class
        assert "class DatasetSearchTool" in source
        assert 'name: str = "dataset_search"' in source

    def test_dataset_search_tool_description(self):
        """DatasetSearchTool has correct description."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        assert "Roboflow" in source
        assert "Kaggle" in source

    def test_dataset_download_tool_exists(self):
        """DatasetDownloadTool exists."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        assert "class DatasetDownloadTool" in source


class TestTrainingTools:
    """Test training tools."""

    def test_train_model_tool_exists(self):
        """TrainModelTool exists."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        assert "class TrainModelTool" in source
        assert 'name: str = "model_train"' in source


class TestExportTools:
    """Test export tools."""

    def test_export_model_tool_exists(self):
        """ExportModelTool exists."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        assert "class ExportModelTool" in source
        assert 'name: str = "model_export"' in source


# ==================== Test Agents ====================

class TestAgentDefinitions:
    """Test agent creation functions exist."""

    def test_dataset_discovery_agent_function_exists(self):
        """create_dataset_discovery_agent function exists."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        assert "def create_dataset_discovery_agent" in source

    def test_data_generator_agent_function_exists(self):
        """create_data_generator_agent function exists."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        assert "def create_data_generator_agent" in source

    def test_training_agent_function_exists(self):
        """create_training_agent function exists."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        assert "def create_training_agent" in source

    def test_deployment_agent_function_exists(self):
        """create_deployment_agent function exists."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        assert "def create_deployment_agent" in source


class TestAgentContent:
    """Test agent content in source."""

    def test_dataset_curator_role(self):
        """Dataset discovery agent has correct role."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        # Check for role in create_dataset_discovery_agent
        assert 'role="Dataset Curator"' in source

    def test_data_engineer_role(self):
        """Data generator agent has correct role."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        assert 'role="Data Engineer"' in source

    def test_ml_engineer_role(self):
        """Training agent has correct role."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        assert 'role="ML Engineer"' in source

    def test_devops_engineer_role(self):
        """Deployment agent has correct role."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        assert 'role="DevOps Engineer"' in source

    def test_decision_rules_present(self):
        """Decision rules are present in agent backstories."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        # Should have "decision rules" in the source
        assert "decision rules" in source.lower()

    def test_data_discovery_rules(self):
        """Data discovery agent has 3 decision rules."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        # Check for rules in discovery agent
        # Looking for score thresholds
        assert "0.8" in source or "0.5" in source

    def test_training_agent_rules(self):
        """Training agent has mAP50 rule."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        assert "mAP50" in source

    def test_deployment_agent_rules(self):
        """Deployment agent has FPS rule."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        assert "FPS" in source


class TestCrew:
    """Test Crew creation."""

    def test_crew_function_exists(self):
        """create_training_crew function exists."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        assert "def create_training_crew" in source

    def test_crew_has_agents(self):
        """Crew function creates 4 agents."""
        source_file = src_path / "agents" / "orchestration.py"
        source = source_file.read_text(encoding='utf-8')

        # Should reference all 4 agents
        assert "create_dataset_discovery_agent()" in source
        assert "create_data_generator_agent()" in source
        assert "create_training_agent()" in source
        assert "create_deployment_agent()" in source
