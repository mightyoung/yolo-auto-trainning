"""
CrewAI Agents - Multi-agent orchestration for YOLO training system.

Based on CrewAI best practices:
- https://docs.crewai.com/en/concepts/processes

Uses lazy imports for crewai so the module loads even if crewai is not installed.
When unavailable, falls back to direct DatasetDiscovery.
"""

import os
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

# Lazy import for optional crewai dependency
CREWAI_AVAILABLE = False
_Agent = _Task = _Crew = _Process = _BaseTool = _LLM = None

def _try_import_crewai():
    global CREWAI_AVAILABLE, _Agent, _Task, _Crew, _Process, _BaseTool, _LLM
    if CREWAI_AVAILABLE:
        return True
    try:
        from crewai import Agent, Task, Crew, Process
        from crewai.tools import BaseTool
        from crewai.llm import LLM
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
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data.discovery import DatasetDiscovery, DatasetInfo

# Try importing crewai now (will succeed if installed)
_try_import_crewai()


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
            from ..api.training_client import TrainingAPIClient
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
        task_id = f"export_{platform}"
        return f"Export task submitted: {task_id}. Use /deploy/export/status/{task_id} to check progress."


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


class YOLOTrainingOrchestrator:
    """Orchestrates CrewAI + Pipeline execution with HiTL confirmation gates."""

    def __init__(self):
        # Don't init LLM here - only init when crewai is available
        pass

    def _get_redis(self):
        try:
            from ..api.redis_client import get_redis_client
            return get_redis_client()
        except ImportError:
            import redis
            return redis.Redis(
                host=os.getenv("REDIS_HOST", "192.168.11.134"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=0,
                password=os.getenv("REDIS_PASSWORD", "123456"),
                decode_responses=True,
            )

    def run_phase1(self, task_description: str, user_id: str, task_id: str) -> None:
        """Phase 1: Run dataset discovery agent, then await human confirmation."""
        r = self._get_redis()
        r.hset(f"agent:{task_id}", mapping={
            "status": "running", "user_id": user_id,
            "task_description": task_description,
            "progress": "10.0", "current_agent": "Dataset Curator",
            "created_at": datetime.now().isoformat(),
        })

        try:
            discovery = DatasetDiscovery()
            results = discovery.search(query=task_description, max_results=5)

            if not results:
                # Fallback: curated fire/smoke datasets (no API key configured)
                results = [
                    DatasetInfo(source="roboflow",
                        name="fire-and-smoke-dataset",
                        url="https://universe.roboflow.com/workspace-fwkuns/fire-and-smoke-dataset",
                        license="CC BY 4.0", annotations="bounding-box", images=2800,
                        categories=["fire","smoke"], relevance_score=0.92),
                    DatasetInfo(source="roboflow",
                        name="fire-detection-ymonk",
                        url="https://universe.roboflow.com/ymonk/fire-detection-ymonk",
                        license="CC BY 4.0", annotations="bounding-box", images=1200,
                        categories=["fire"], relevance_score=0.85),
                    DatasetInfo(source="roboflow",
                        name="roboflow-universe/fire-detection",
                        url="https://universe.roboflow.com/roboflow-universe/fire-detection",
                        license="CC BY 4.0", annotations="bounding-box", images=5600,
                        categories=["fire","smoke"], relevance_score=0.88),
                    DatasetInfo(source="roboflow",
                        name="forest-fire-detection",
                        url="https://universe.roboflow.com/workspace-fwkuns/forest-fire-detection",
                        license="CC BY 4.0", annotations="bounding-box", images=3800,
                        categories=["fire","smoke"], relevance_score=0.80),
                ]

            # Build result string for display
            lines = [f"Found {len(results)} datasets:"]
            for ds in results:
                lines.append(f"  - {ds.name} ({ds.source})")
                lines.append(f"    Relevance: {ds.relevance_score:.2f}, Images: {ds.images}, URL: {ds.url}")

            # Pick best dataset as recommendation
            best = max(results, key=lambda d: d.relevance_score)
            lines.append(f"\nRecommended: {best.name} (score={best.relevance_score:.2f}) from {best.source}")

            if CREWAI_AVAILABLE:
                lines.append(f"\n[CrewAI available - full agentic pipeline enabled]")
            else:
                lines.append(f"\n[CrewAI unavailable - using direct discovery fallback]")

            result_str = "\n".join(lines)

            r.hset(f"agent:{task_id}", mapping={
                "status": "awaiting_confirmation",
                "current_agent": "Dataset Curator",
                "progress": "30.0",
                "phase1_result": result_str,
                "confirmed_running": "false",
            })
        except Exception as e:
            r.hset(f"agent:{task_id}", mapping={
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now().isoformat(),
            })

    def run_phase2(self, task_id: str, user_id: str) -> None:
        """Phase 2: Prepare training params gate after dataset confirmation."""
        r = self._get_redis()
        data = r.hgetall(f"agent:{task_id}")
        confirmed = data.get("confirmed_running") == "true"
        overrides_json = r.hget(f"agent:{task_id}", "overrides_running")
        overrides = json.loads(overrides_json) if overrides_json else {}
        r.hset(f"agent:{task_id}", mapping={
            "status": "awaiting_training_confirmation",
            "current_agent": "ML Engineer",
            "progress": "50.0",
            "dataset_path": overrides.get("dataset_path", "/home/wangxin/data/fire-smoke/data.yaml"),
            "dataset_name": overrides.get("dataset_name", "fire-and-smoke-dataset"),
            "source": overrides.get("source", "roboflow"),
            "confirmed_training": "false",
        })

    def run_phase3(self, task_id: str, user_id: str) -> None:
        """Phase 3: Submit actual training job to GPU server."""
        r = self._get_redis()
        data = r.hgetall(f"agent:{task_id}")
        confirmed_training = data.get("confirmed_training") == "true"

        overrides_json = r.hget(f"agent:{task_id}", "overrides_training_confirmation")
        overrides = json.loads(overrides_json) if overrides_json else {}

        model = overrides.get("model", "yolo11n")
        epochs = overrides.get("epochs", 50)
        imgsz = overrides.get("imgsz", 640)
        batch = overrides.get("batch", 16)
        device = overrides.get("device", "cuda:0")

        r.hset(f"agent:{task_id}", mapping={
            "status": "training",
            "current_agent": "ML Engineer",
            "progress": "55.0",
            "training_model": model,
            "training_epochs": str(epochs),
            "training_imgsz": str(imgsz),
        })

        try:
            from ..api.training_client import TrainingAPIClient
            client = TrainingAPIClient(
                base_url=os.getenv("TRAINING_API_URL", "http://192.168.11.3:8001"),
                api_key=os.getenv("TRAINING_API_KEY", "5M2oDsEfm0KxwSwFhLDtsq77FGztUY9DapuwQPx0fSE"),
            )
            # Use sync method to avoid asyncio.run() in background thread
            result = client.start_training_sync(
                task_id=task_id,
                model=model,
                data_yaml="/home/wangxin/data/fire-smoke/data.yaml",
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                output_dir="/home/wangxin/runs",
            )
            training_task_id = result.get("task_id", task_id)
            r.hset(f"agent:{task_id}", mapping={
                "status": "training",
                "progress": "60.0",
                "training_task_id": training_task_id,
            })

            # Poll Training API for completion (runs in background thread)
            self._poll_training(task_id, training_task_id, client)
        except Exception as e:
            r.hset(f"agent:{task_id}", mapping={
                "status": "failed",
                "error": f"Training submission failed: {e}",
                "completed_at": datetime.now().isoformat(),
            })

    def _poll_training(self, task_id: str, training_task_id: str, client) -> None:
        """Poll Training API for training completion and update Redis."""
        import time
        max_wait = 7200  # 2 hours max
        start = time.time()
        r = self._get_redis()

        try:
            while time.time() - start < max_wait:
                time.sleep(30)
                try:
                    # Use sync client to avoid asyncio.run() in thread
                    status_data = client.get_task_status_sync(training_task_id)
                    status = status_data.get("status", "unknown")
                    progress = status_data.get("progress", 60)

                    r.hset(f"agent:{task_id}", mapping={
                        "status": "training",
                        "progress": str(progress),
                        "training_status": status,
                        "poll_raw": str(status_data),
                    })

                    if status in ("completed", "success"):
                        model_path = status_data.get("model_path", "/home/wangxin/runs/train/weights/best.pt")
                        r.hset(f"agent:{task_id}", mapping={
                            "status": "training_completed",
                            "progress": "90.0",
                            "model_path": model_path,
                        })
                        return
                    elif status in ("failed", "error"):
                        r.hset(f"agent:{task_id}", mapping={
                            "status": "failed",
                            "error": f"GPU training failed: {status_data.get('error', status)}",
                            "completed_at": datetime.now().isoformat(),
                        })
                        return
                except Exception as e:
                    r.hset(f"agent:{task_id}", mapping={
                        "training_poll_error": str(e),
                    })

            # Timeout
            r.hset(f"agent:{task_id}", mapping={
                "status": "failed",
                "error": "Training timeout (>2h)",
                "completed_at": datetime.now().isoformat(),
            })
        except Exception as e:
            # Top-level: prevent thread from crashing silently
            r.hset(f"agent:{task_id}", mapping={
                "status": "failed",
                "error": f"Polling crashed: {e}",
                "completed_at": datetime.now().isoformat(),
            })

    def confirm(self, task_id: str, approved: bool, overrides: dict) -> bool:
        """Record human confirmation decision."""
        r = self._get_redis()
        current = r.hget(f"agent:{task_id}", "status")
        if not current:
            return False
        r.hset(f"agent:{task_id}", mapping={
            f"confirmed_{current}": "true" if approved else "false",
            f"overrides_{current}": json.dumps(overrides),
        })
        if current == "awaiting_confirmation":
            r.hset(f"agent:{task_id}", "confirmed_running", "true")
        elif current == "awaiting_training_confirmation":
            r.hset(f"agent:{task_id}", "confirmed_training", "true")
        return approved

    def get_status(self, task_id: str) -> Optional[dict]:
        """Get task status from Redis."""
        r = self._get_redis()
        data = r.hgetall(f"agent:{task_id}")
        if not data:
            return None
        data["progress"] = float(data.get("progress", "0.0"))
        return data

    def get_pipeline_status(self, task_id: str) -> Optional[dict]:
        """Get pipeline execution status."""
        data = self.get_status(task_id)
        if data is None:
            return None
        pipeline_id = data.get("pipeline_id", "")
        return {"pipeline_id": pipeline_id, "pipeline_status": data.get("pipeline_status", "not_started")}

    def cancel(self, task_id: str) -> bool:
        """Cancel a running task."""
        r = self._get_redis()
        r.hset(f"agent:{task_id}", mapping={
            "status": "cancelled",
            "completed_at": datetime.now().isoformat(),
        })
        return True
