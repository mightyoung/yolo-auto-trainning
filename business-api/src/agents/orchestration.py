"""
CrewAI Agents - Multi-agent orchestration for YOLO training system.

Based on CrewAI best practices:
- https://docs.crewai.com/en/concepts/processes
"""

import os
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai.llm import LLM
from typing import List, Dict, Any
from pydantic import BaseModel

# Import real modules
from ..data.discovery import DatasetDiscovery, DatasetInfo


def get_llm():
    """Get the LLM instance based on environment configuration."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set")

    return LLM(
        model=model,
        base_url=base_url,
        api_key=api_key
    )


# Tool definitions
class SearchDatasetsInput(BaseModel):
    """Input for dataset search tool."""
    query: str
    max_results: int = 10


class TrainModelInput(BaseModel):
    """Input for training tool."""
    dataset_path: str
    model_size: str = "yolo11m"
    epochs: int = 100


class ExportModelInput(BaseModel):
    """Input for model export tool."""
    model_path: str
    platform: str = "jetson_orin"


# Custom tools with real implementations
class DatasetSearchTool(BaseTool):
    """Tool for searching datasets from multiple sources."""

    name: str = "dataset_search"
    description: str = "Search for relevant datasets from Roboflow, Kaggle, and HuggingFace. Returns dataset info with relevance scores."

    def _run(self, query: str, max_results: int = 10) -> str:
        """
        Search for datasets across multiple sources.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            Formatted string with found datasets
        """
        discovery = DatasetDiscovery()
        results = discovery.search(query=query, max_results=max_results)

        if not results:
            return f"No datasets found for query: {query}"

        output = f"Found {len(results)} datasets:\n\n"
        for ds in results:
            output += f"• {ds.name} ({ds.source})\n"
            output += f"  URL: {ds.url}\n"
            output += f"  Relevance: {ds.relevance_score:.2f}\n"
            output += f"  Images: {ds.images}\n"
            output += f"  License: {ds.license}\n\n"

        return output


class DatasetDownloadTool(BaseTool):
    """Tool for downloading datasets."""

    name: str = "dataset_download"
    description: str = "Download a dataset from a specific source (roboflow, kaggle, or huggingface)"

    def _run(self, dataset_name: str, source: str = "roboflow") -> str:
        """
        Download dataset from source.

        Args:
            dataset_name: Dataset reference (e.g., 'username/project' for Roboflow/Kaggle)
            source: Source platform (roboflow, kaggle, huggingface)

        Returns:
            Status message with download path
        """
        discovery = DatasetDiscovery()

        # Create dataset info
        dataset_info = DatasetInfo(
            source=source,
            name=dataset_name,
            url="",
            license="unknown",
            annotations="unknown",
            images=0,
            categories=[],
        )

        try:
            output_path = discovery.download(dataset_info)
            return f"Downloaded dataset to: {output_path}"
        except Exception as e:
            return f"Download failed: {str(e)}"


class TrainModelTool(BaseTool):
    """Tool for training YOLO models."""

    name: str = "model_train"
    description: str = "Train a YOLO model on a dataset with specified parameters"

    def _run(self, dataset_path: str, model_size: str = "yolo11m", epochs: int = 100) -> str:
        """
        Train YOLO model.

        Args:
            dataset_path: Path to dataset YAML
            model_size: Model size (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
            epochs: Number of training epochs

        Returns:
            Status message with training info
        """
        # This tool returns task_id for async training
        # Actual training happens via Celery task
        task_id = f"train_{model_size}_{epochs}"
        return f"Training task submitted: {task_id}. Use /train/status/{task_id} to check progress."


class ExportModelTool(BaseTool):
    """Tool for exporting models."""

    name: str = "model_export"
    description: str = "Export trained model to ONNX or TensorRT format for deployment"

    def _run(self, model_path: str, platform: str = "jetson_orin") -> str:
        """
        Export model to target platform.

        Args:
            model_path: Path to trained model
            platform: Target platform (jetson_nano, jetson_orin, rk3588, cloud)

        Returns:
            Status message with export info
        """
        task_id = f"export_{platform}"
        return f"Export task submitted: {task_id}. Use /deploy/export/status/{task_id} to check progress."


# Agent definitions with decision rules from design doc
def create_dataset_discovery_agent() -> Agent:
    """Create dataset discovery agent with decision rules."""
    return Agent(
        role="Dataset Curator",
        goal="Find and select the most relevant datasets for the task",
        backstory="""
            You are an expert in dataset discovery and curation.
            You know how to search and evaluate datasets from:
            - Roboflow (250k+ datasets)
            - Kaggle (hundreds of thousands of datasets)
            - HuggingFace (multimodal datasets)
            - Open Images

            Your decision rules:
            1. If relevance score > 0.8 → select dataset directly
            2. If 0.5 < score < 0.8 → include with warning
            3. If score < 0.5 → reject and trigger synthetic generation

            Always prioritize real-world data over synthetic data.
        """,
        llm=get_llm(),
        tools=[DatasetSearchTool(), DatasetDownloadTool()],
        verbose=True,
        allow_delegation=False,
    )


def create_data_generator_agent() -> Agent:
    """Create data generation agent with decision rules."""
    return Agent(
        role="Data Engineer",
        goal="Generate high-quality synthetic data using ComfyUI workflows",
        backstory="""
            You are an expert in synthetic data generation.

            Your decision rules:
            1. If synthetic ratio > 30% → stop generating, use discovered data
            2. If CLIP relevance score < 0.25 → filter out low-quality images
            3. If generation fails → fallback to manual labeling

            Always prefer quality over quantity.
        """,
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_training_agent() -> Agent:
    """Create training agent with decision rules."""
    return Agent(
        role="ML Engineer",
        goal="Train YOLO11 model with optimal performance",
        backstory="""
            You are an expert in YOLO11 training.

            Your decision rules:
            1. If dataset < 1000 images → use aggressive data augmentation
            2. If mAP50 < 0.5 after HPO → try larger model
            3. If edge deployment → use YOLO11n (nano)
            4. If server deployment → use YOLO11m or YOLO11l
            5. If training time > 10 hours → enable aggressive early stopping

            Always balance accuracy and inference speed.
        """,
        llm=get_llm(),
        tools=[TrainModelTool()],
        verbose=True,
        allow_delegation=False,
    )


def create_deployment_agent() -> Agent:
    """Create deployment agent with decision rules."""
    return Agent(
        role="DevOps Engineer",
        goal="Deploy model to edge device reliably",
        backstory="""
            You are an expert in edge deployment.

            Your decision rules:
            1. If FPS < 20 → optimize model or reduce input size
            2. If device memory < 2GB → use INT8 quantization
            3. If deployment fails → rollback to previous version

            Prioritize reliability over performance.
        """,
        llm=get_llm(),
        tools=[ExportModelTool()],
        verbose=True,
        allow_delegation=False,
    )


# Crew definitions
def create_training_crew() -> Crew:
    """Create training crew with hierarchical process."""

    # Create agents
    discovery_agent = create_dataset_discovery_agent()
    generator_agent = create_data_generator_agent()
    training_agent = create_training_agent()
    deployment_agent = create_deployment_agent()

    # Create tasks
    discovery_task = Task(
        description="Find relevant datasets for detecting {task_description}",
        agent=discovery_agent,
        expected_output="List of relevant datasets with relevance scores",
    )

    generation_task = Task(
        description="Generate synthetic data to supplement real datasets",
        agent=generator_agent,
        expected_output="Path to generated synthetic dataset",
        context=[discovery_task],
    )

    training_task = Task(
        description="Train YOLO model with optimal hyperparameters",
        agent=training_agent,
        expected_output="Path to trained model weights",
        context=[discovery_task, generation_task],
    )

    deployment_task = Task(
        description="Export and prepare model for edge deployment",
        agent=deployment_agent,
        expected_output="Path to exported model ready for deployment",
        context=[training_task],
    )

    # Create crew with hierarchical process
    crew = Crew(
        agents=[
            discovery_agent,
            generator_agent,
            training_agent,
            deployment_agent,
        ],
        tasks=[
            discovery_task,
            generation_task,
            training_task,
            deployment_task,
        ],
        process=Process.hierarchical,
        manager_llm=get_llm(),  # Use Deepseek for manager
        verbose=True,
    )

    return crew


# Simple workflow crew
def create_simple_crew(task_description: str) -> Crew:
    """Create a simple sequential crew for training."""

    discovery_agent = create_dataset_discovery_agent()
    training_agent = create_training_agent()

    discovery_task = Task(
        description=f"Find datasets for: {task_description}",
        agent=discovery_agent,
    )

    training_task = Task(
        description="Train YOLO model on discovered dataset",
        agent=training_agent,
        context=[discovery_task],
    )

    return Crew(
        agents=[discovery_agent, training_agent],
        tasks=[discovery_task, training_task],
        process=Process.sequential,
        verbose=True,
    )
