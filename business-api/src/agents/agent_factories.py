"""
CrewAI agent factory functions.
Location: business-api/src/agents/agent_factories.py

Contains:
- create_dataset_discovery_agent()
- create_training_agent()
- create_deployment_agent()
"""

from .tools import get_llm, DatasetSearchTool, DatasetDownloadTool, TrainModelTool, ExportModelTool

# Lazy import for optional crewai dependency
CREWAI_AVAILABLE = False
_Agent = None

def _try_import_crewai():
    global CREWAI_AVAILABLE, _Agent
    if CREWAI_AVAILABLE:
        return True
    try:
        from crewai import Agent, Task, Crew, Process
        from crewai.tools import BaseTool
        from crewai.llm import LLM
        global _Agent, _Task, _Crew, _Process, _BaseTool
        _Agent = Agent
        _Task = Task
        _Crew = Crew
        _Process = Process
        _BaseTool = BaseTool
        CREWAI_AVAILABLE = True
        return True
    except ImportError:
        CREWAI_AVAILABLE = False
        return False


# CrewAI-backed agent factories (only called when crewai is available)
def create_dataset_discovery_agent():
    """Create dataset discovery agent with decision rules."""
    _try_import_crewai()
    if not CREWAI_AVAILABLE:
        return None
    return _Agent(
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
            1. If relevance score > 0.8 -> select dataset directly
            2. If 0.5 < score < 0.8 -> include with warning
            3. If score < 0.5 -> reject and trigger synthetic generation

            Always prioritize real-world data over synthetic data.
        """,
        llm=get_llm(),
        tools=[DatasetSearchTool(), DatasetDownloadTool()],
        verbose=True,
        allow_delegation=False,
    )


def create_training_agent():
    """Create training agent with decision rules."""
    _try_import_crewai()
    if not CREWAI_AVAILABLE:
        return None
    return _Agent(
        role="ML Engineer",
        goal="Train YOLO11 model with optimal performance",
        backstory="""
            You are an expert in YOLO11 training.

            Your decision rules:
            1. If dataset < 1000 images -> use aggressive data augmentation
            2. If mAP50 < 0.5 after HPO -> try larger model
            3. If edge deployment -> use YOLO11n (nano)
            4. If server deployment -> use YOLO11m or YOLO11l
            5. If training time > 10 hours -> enable aggressive early stopping

            Always balance accuracy and inference speed.
        """,
        llm=get_llm(),
        tools=[TrainModelTool()],
        verbose=True,
        allow_delegation=False,
    )


def create_deployment_agent():
    """Create deployment agent with decision rules."""
    _try_import_crewai()
    if not CREWAI_AVAILABLE:
        return None
    return _Agent(
        role="DevOps Engineer",
        goal="Deploy model to edge device reliably",
        backstory="""
            You are an expert in edge deployment.

            Your decision rules:
            1. If FPS < 20 -> optimize model or reduce input size
            2. If device memory < 2GB -> use INT8 quantization
            3. If deployment fails -> rollback to previous version

            Prioritize reliability over performance.
        """,
        llm=get_llm(),
        tools=[ExportModelTool()],
        verbose=True,
        allow_delegation=False,
    )
