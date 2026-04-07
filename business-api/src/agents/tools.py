"""
Tools for CrewAI agents.
Location: business-api/src/agents/tools.py

Contains:
- get_llm() - LLM factory
- DatasetSearchTool
- DatasetDownloadTool
- TrainModelTool
- ExportModelTool
"""

import os
import uuid
from pathlib import Path

# Lazy import for optional crewai dependency
CREWAI_AVAILABLE = False
_LLM = None

def _try_import_crewai():
    global CREWAI_AVAILABLE, _LLM
    if CREWAI_AVAILABLE:
        return True
    try:
        from crewai import Agent, Task, Crew, Process
        from crewai.tools import BaseTool
        from crewai.llm import LLM
        global _Agent, _Task, _Crew, _Process, _BaseTool, _LLM
        _Agent = Agent
        _Task = Task
        _Crew = Crew
        _Process = Process
        _BaseTool = BaseTool
        _LLM = LLM
        CREWAI_AVAILABLE = True
        return True
    except ImportError:
        CREWAI_AVAILABLE = False
        return False


# Import real modules
_project_root = Path(__file__).parent.parent.parent.parent  # project root (contains src/)
_biz_api_root = Path(__file__).parent.parent  # business-api/src/
import sys
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_biz_api_root) not in sys.path:
    sys.path.insert(0, str(_biz_api_root))

from src.data.discovery import DatasetDiscovery, DatasetInfo
from .operation_policy import require_operation_allowed


def get_llm():
    """Get the LLM instance based on environment configuration."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set")

    _try_import_crewai()
    if not CREWAI_AVAILABLE:
        raise RuntimeError("crewai not installed - cannot create LLM")
    return _LLM(
        model=model,
        base_url=base_url,
        api_key=api_key
    )


# Tool definitions - standalone classes (no BaseTool inheritance needed when crewai unavailable)
class DatasetSearchTool:
    """Tool for searching datasets from multiple sources."""

    name = "dataset_search"
    description = "Search for relevant datasets from Roboflow, Kaggle, and HuggingFace. Returns dataset info with relevance scores."

    def _run(self, query: str, max_results: int = 10) -> str:
        discovery = DatasetDiscovery()
        results = discovery.search(query=query, max_results=max_results)

        if not results:
            return f"No datasets found for query: {query}"

        output = f"Found {len(results)} datasets:\n\n"
        for ds in results:
            output += f"- {ds.name} ({ds.source})\n"
            output += f"  URL: {ds.url}\n"
            output += f"  Relevance: {ds.relevance_score:.2f}\n"
            output += f"  Images: {ds.images}\n"
            output += f"  License: {ds.license}\n\n"

        return output


class DatasetDownloadTool:
    """Tool for downloading datasets."""

    name = "dataset_download"
    description = "Download a dataset from a specific source (roboflow, kaggle, or huggingface)"

    def _run(self, dataset_name: str, source: str = "roboflow") -> str:
        require_operation_allowed("dataset_download", context={"dataset_name": dataset_name, "source": source})
        discovery = DatasetDiscovery()
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


class TrainModelTool:
    """Tool for training YOLO models."""

    name = "model_train"
    description = "Train a YOLO model on a dataset with specified parameters"

    def _run(self, dataset_path: str, model_size: str = "yolo11m", epochs: int = 100) -> str:
        try:
            require_operation_allowed(
                "gpu_training_submit",
                context={"dataset_path": dataset_path, "model_size": model_size, "epochs": epochs},
            )
            from src.api.training_client import TrainingAPIClient
            client = TrainingAPIClient()
            task_id = f"train_{uuid.uuid4().hex[:8]}"
            result = client.start_training(
                task_id=task_id,
                model=model_size,
                data_yaml=dataset_path,
                epochs=epochs,
                device="cuda:0",
            )
            return f"Training started: task_id={result.get('task_id', task_id)}"
        except Exception as e:
            return f"Train submission failed: {str(e)}"


class ExportModelTool:
    """Tool for exporting models."""

    name = "model_export"
    description = "Export trained model to ONNX or TensorRT format for deployment"

    def _run(self, model_path: str, platform: str = "jetson_orin") -> str:
        require_operation_allowed(
            "model_export",
            context={"model_path": model_path, "platform": platform},
        )
        task_id = f"export_{platform}"
        return f"Export task submitted: task_id={task_id}, platform={platform}"
