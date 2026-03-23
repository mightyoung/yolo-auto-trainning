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
import threading
import time
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
_biz_api_root = Path(__file__).parent.parent  # business-api/
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_biz_api_root) not in sys.path:
    sys.path.insert(0, str(_biz_api_root))

from src.data.discovery import DatasetDiscovery, DatasetInfo
from .gpu_scheduler import start_scheduler

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
            from src.api.redis_client import get_redis_client
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
                    # COCO Person built-in fallback: no API key needed
                    DatasetInfo(source="coco_builtin",
                        name="COCO-Person-BuiltIn",
                        url="https://cocodataset.org",
                        license="CC BY 4.0",
                        annotations="coco",
                        images=0,
                        categories=["person"],
                        relevance_score=0.95),
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

            # Autonomous mode: bypass human gate and auto-confirm Phase 1
            r_check = self._get_redis()
            if r_check.hget(f"agent:{task_id}", "auto_confirm") == "true":
                # Set Phase 1 confirmation overrides so run_phase2 reads the right dataset
                # FIX: Use canonical path for coco_builtin instead of generating from name
                if best.source == "coco_builtin":
                    dataset_path = "/home/wangxin/data/coco_person"
                else:
                    dataset_path = f"/home/wangxin/data/{best.name.replace('/', '_').replace('.', '_')}"
                default_overrides = {
                    "dataset_name": best.name,
                    "dataset_path": dataset_path,
                    "source": best.source,
                    "user_id": user_id,
                }
                r.hset(f"agent:{task_id}", mapping={
                    "overrides_awaiting_confirmation": json.dumps(default_overrides),
                    "status": "running",
                    "confirmed_running": "true",
                    "progress": "32.0",
                })
                # Chain to phase2 in background thread
                def _auto_phase2():
                    try:
                        import time
                        time.sleep(1)
                        orch2 = YOLOTrainingOrchestrator()
                        orch2.run_phase2(task_id, user_id)
                    except Exception as e:
                        r.hset(f"agent:{task_id}", mapping={
                            "status": "failed", "error": f"Auto Phase2 error: {e}",
                        })
                t = threading.Thread(target=_auto_phase2, daemon=True)
                t.start()
        except Exception as e:
            r.hset(f"agent:{task_id}", mapping={
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now().isoformat(),
            })

    def run_phase2(self, task_id: str, user_id: str) -> None:
        """Phase 2: Download dataset + generate data.yaml, then await training param confirmation."""
        r = self._get_redis()

        # Read Phase 1 confirmation overrides (dataset choice)
        overrides_json = r.hget(f"agent:{task_id}", "overrides_awaiting_confirmation")
        overrides = json.loads(overrides_json) if overrides_json else {}

        dataset_name = overrides.get("dataset_name", "workspace-fwkuns/fire-and-smoke-dataset")
        dataset_path = overrides.get("dataset_path", "/home/wangxin/data/fire-smoke")
        source = overrides.get("source", "roboflow")

        # Update status: downloading
        r.hset(f"agent:{task_id}", mapping={
            "status": "downloading_dataset",
            "current_agent": "Data Engineer",
            "progress": "35.0",
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "source": source,
            "confirmed_training": "false",
        })

        try:
            # Check if dataset already exists with images
            skip_download = self._check_dataset_exists(dataset_path, source)

            if not skip_download:
                # Download dataset to GPU server via SSH
                if source == "coco_builtin":
                    self._download_coco_builtin_ssh(dataset_path)
                else:
                    self._download_dataset_ssh(dataset_name, dataset_path, source)

            # Generate data.yaml on GPU server (if missing or stale)
            self._generate_data_yaml_ssh(dataset_path)

            r.hset(f"agent:{task_id}", mapping={
                "status": "awaiting_training_confirmation",
                "current_agent": "ML Engineer",
                "progress": "50.0",
                "download_status": "completed" if not skip_download else "skipped_existing",
                "confirmed_training": "false",
            })

            # Autonomous mode: bypass training confirmation gate and auto-chain to Phase 3
            r_check = self._get_redis()
            if r_check.hget(f"agent:{task_id}", "auto_confirm") == "true":
                r.hset(f"agent:{task_id}", mapping={
                    "status": "running",
                    "confirmed_training": "true",
                    "progress": "52.0",
                })
                def _auto_phase3():
                    try:
                        import time
                        time.sleep(1)
                        orch3 = YOLOTrainingOrchestrator()
                        orch3.run_phase3(task_id, "auto")
                    except Exception as e:
                        r.hset(f"agent:{task_id}", mapping={
                            "status": "failed", "error": f"Auto Phase3 error: {e}",
                        })
                t = threading.Thread(target=_auto_phase3, daemon=True)
                t.start()
        except Exception as e:
            r.hset(f"agent:{task_id}", mapping={
                "status": "failed",
                "error": f"Dataset download failed: {e}",
                "completed_at": datetime.now().isoformat(),
            })

    def _check_dataset_exists(self, dataset_path: str, source: str = "roboflow") -> bool:
        """Check if dataset already exists at the given path with train images.

        For coco_builtin source, tries multiple known path variants to find existing data.
        """
        import paramiko
        ssh_host = os.getenv("GPU_SERVER_HOST", "192.168.11.3")
        ssh_user = os.getenv("GPU_SERVER_USER", "wangxin")
        ssh_pass = os.getenv("GPU_SERVER_PASS", "123123")

        # For coco_builtin, try multiple known path variants
        paths_to_try = [dataset_path]
        if source == "coco_builtin":
            paths_to_try.extend([
                "/home/wangxin/data/coco_person",
                "/home/wangxin/data/COCO_Person_BuiltIn",
                "/home/wangxin/data/COCO-Person-BuiltIn",
            ])

        for path in paths_to_try:
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(ssh_host, username=ssh_user, password=ssh_pass, timeout=10)
                stdin, stdout, stderr = client.exec_command(
                    '/home/wangxin/yolo-auto-training/training-venv/bin/python -c "'
                    'import pathlib; p=pathlib.Path(r\\\"' + path + '\\\"); '
                    'train=list((p/\\\"train\\\"/\\\"images\\\").glob(\\\"*.jpg\\\"))+list((p/\\\"train\\\"/\\\"images\\\").glob(\\\"*.png\\\")); '
                    'print(len(train))'
                    '"',
                    timeout=15
                )
                stdout.channel.recv_exit_status()
                count = int(stdout.read().decode().strip())
                print(f"[Dataset check] Found {count} train images at {path}")
                client.close()
                if count > 0:
                    return True
            except Exception as e:
                print(f"[Dataset check] Error for {path}: {e}")
        return False

    def _download_dataset_ssh(self, dataset_name: str, dataset_path: str, source: str) -> None:
        """Download Roboflow dataset to GPU server via SSH."""
        import paramiko

        ssh_host = os.getenv("GPU_SERVER_HOST", "192.168.11.3")
        ssh_user = os.getenv("GPU_SERVER_USER", "wangxin")
        ssh_pass = os.getenv("GPU_SERVER_PASS", "123123")

        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY not set in environment")

        script = (
            "import urllib.request, urllib.error, json, zipfile, os, sys\n"
            "from pathlib import Path\n\n"
            "output_path = Path(r'" + dataset_path + "')\n"
            "output_path.mkdir(parents=True, exist_ok=True)\n\n"
            "api_key = '" + api_key + "'\n"
            "name = '" + dataset_name + "'\n\n"
            "parts = name.split('/')\n"
            "workspace = parts[0] if len(parts) > 0 else None\n"
            "project = parts[1] if len(parts) > 1 else parts[0]\n"
            "version = parts[2] if len(parts) > 2 else None\n\n"
            "if not version:\n"
            "    try:\n"
            "        meta_url = 'https://api.roboflow.com/' + workspace + '/' + project + '/info?api_key=' + api_key\n"
            "        req = urllib.request.Request(meta_url)\n"
            "        with urllib.request.urlopen(req, timeout=30) as resp:\n"
            "            meta = json.loads(resp.read())\n"
            "        versions = meta.get('versions', [])\n"
            "        if versions:\n"
            "            version = versions[-1]['id']\n"
            "            print('Latest version: ' + version)\n"
            "    except Exception as e:\n"
            "        print('Could not get version: ' + str(e))\n"
            "        raise RuntimeError('Cannot determine dataset version for ' + name)\n\n"
            "if not version:\n"
            "    raise RuntimeError('No version found for ' + name)\n\n"
            "download_url = 'https://app.roboflow.com/' + workspace + '/' + project + '/' + version + '/download?api_key=' + api_key + '&format=yolov8'\n"
            "print('Downloading ' + workspace + '/' + project + '/' + version + ' to ' + str(output_path) + '...')\n"
            "req = urllib.request.Request(download_url)\n"
            "with urllib.request.urlopen(req, timeout=600) as resp:\n"
            "    data = resp.read()\n\n"
            "zip_path = output_path / 'dataset.zip'\n"
            "with open(zip_path, 'wb') as f:\n"
            "    f.write(data)\n\n"
            "print('Extracting...')\n"
            "with zipfile.ZipFile(zip_path, 'r') as z:\n"
            "    z.extractall(output_path)\n"
            "zip_path.unlink()\n\n"
            "items = list(output_path.iterdir())\n"
            "print('Contents: ' + str([i.name for i in items]))\n"
            "for item in items:\n"
            "    if item.is_dir():\n"
            "        subdirs = [s.name for s in item.iterdir()]\n"
            "        print('Subdir ' + item.name + ' contains: ' + str(subdirs))\n\n"
            "print('Download complete!')\n"
        )

        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(ssh_host, username=ssh_user, password=ssh_pass, timeout=15)

            sftp = client.open_sftp()
            f = sftp.file('/tmp/dl_roboflow.py', 'wb', -1)
            f.write(script.encode())
            f.close()
            sftp.close()

            stdin, stdout, stderr = client.exec_command(
                '/home/wangxin/yolo-auto-training/training-venv/bin/python /tmp/dl_roboflow.py 2>&1',
                timeout=700
            )
            output = stdout.read().decode(errors='replace')
            error = stderr.read().decode(errors='replace')
            client.close()

            print(f"[Download] stdout: {output}")
            if error:
                print(f"[Download] stderr: {error}")

            if 'Download complete!' not in output:
                raise RuntimeError(f"Download script failed. Output: {output[:500]}")
        except Exception as e:
            raise RuntimeError(f"SSH download failed: {e}")

    def _download_coco_builtin_ssh(self, dataset_path: str) -> None:
        """
        Download COCO val2017, filter to person class, and convert to YOLO format
        on the GPU server.  No API keys needed.
        """
        import paramiko

        ssh_host = os.getenv("GPU_SERVER_HOST", "192.168.11.3")
        ssh_user = os.getenv("GPU_SERVER_USER", "wangxin")
        ssh_pass = os.getenv("GPU_SERVER_PASS", "123123")

        script = (
            "import urllib.request, zipfile, json, shutil, os\n"
            "from pathlib import Path\n\n"
            "output_path = Path(r'" + dataset_path + "')\n"
            "output_path.mkdir(parents=True, exist_ok=True)\n\n"
            "coco_cache = Path.home() / '.cache' / 'ultralytics' / 'coco'\n"
            "coco_cache.mkdir(parents=True, exist_ok=True)\n\n"
            "val_img_dir = coco_cache / 'images' / 'val2017'\n"
            "ann_dir = coco_cache / 'annotations'\n\n"
            "# Download COCO annotations (trainval2017, ~11 MB)\n"
            "ann_zip = coco_cache / 'annotations_trainval2017.zip'\n"
            "if not (ann_dir / 'instances_val2017.json').exists():\n"
            "    print('[COCO] Downloading annotations...')\n"
            "    if not ann_zip.exists():\n"
            "        urllib.request.urlretrieve(\n"
            "            'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',\n"
            "            ann_zip)\n"
            "    print('[COCO] Extracting annotations...')\n"
            "    with zipfile.ZipFile(ann_zip, 'r') as z:\n"
            "        z.extractall(coco_cache)\n"
            "    ann_zip.unlink()\n\n"
            "# Download COCO val images (~300 MB)\n"
            "val_zip = coco_cache / 'val2017.zip'\n"
            "if not val_img_dir.exists() or len(list(val_img_dir.glob('*.jpg'))) < 100:\n"
            "    print('[COCO] Downloading val images (~300 MB)...')\n"
            "    urllib.request.urlretrieve(\n"
            "        'http://images.cocodataset.org/zips/val2017.zip',\n"
            "        val_zip)\n"
            "    print('[COCO] Extracting val images...')\n"
            "    with zipfile.ZipFile(val_zip, 'r') as z:\n"
            "        z.extractall(coco_cache)\n"
            "    val_zip.unlink()\n\n"
            "# Parse annotations, filter to person (cat_id=1)\n"
            "ann_file = ann_dir / 'instances_val2017.json'\n"
            "with open(ann_file) as f:\n"
            "    coco = json.load(f)\n\n"
            "img_map = {img['id']: img for img in coco['images']}\n"
            "person_bboxes = {}\n"
            "for ann in coco['annotations']:\n"
            "    if ann['category_id'] == 1 and ann.get('bbox'):\n"
            "        img_id = ann['image_id']\n"
            "        person_bboxes.setdefault(img_id, []).append(ann['bbox'])\n\n"
            "valid_ids = sorted(person_bboxes.keys())\n"
            "n = len(valid_ids)\n"
            "n_train = int(n * 0.8)\n"
            "train_ids = set(valid_ids[:n_train])\n"
            "val_ids = set(valid_ids[n_train:])\n\n"
            "train_img_d = output_path / 'train' / 'images'\n"
            "train_lbl_d = output_path / 'train' / 'labels'\n"
            "val_img_d = output_path / 'val' / 'images'\n"
            "val_lbl_d = output_path / 'val' / 'labels'\n"
            "for d in [train_img_d, train_lbl_d, val_img_d, val_lbl_d]:\n"
            "    d.mkdir(parents=True, exist_ok=True)\n\n"
            "copied = {'train': 0, 'val': 0}\n"
            "for img_id, bboxes in person_bboxes.items():\n"
            "    img_meta = img_map[img_id]\n"
            "    src = val_img_dir / img_meta['file_name']\n"
            "    if not src.exists():\n"
            "        continue\n"
            "    split = 'train' if img_id in train_ids else 'val'\n"
            "    img_d = train_img_d if split == 'train' else val_img_d\n"
            "    lbl_d = train_lbl_d if split == 'train' else val_lbl_d\n"
            "    dst_img = img_d / img_meta['file_name']\n"
            "    dst_lbl = (lbl_d / img_meta['file_name']).with_suffix('.txt')\n"
            "    shutil.copy2(src, dst_img)\n"
            "    W, H = img_meta['width'], img_meta['height']\n"
            "    lines = []\n"
            "    for x, y, w, h in bboxes:\n"
            "        xc = max(0.0, min(1.0, (x + w / 2) / W))\n"
            "        yc = max(0.0, min(1.0, (y + h / 2) / H))\n"
            "        nw = max(0.0, min(1.0, w / W))\n"
            "        nh = max(0.0, min(1.0, h / H))\n"
            "        lines.append(f'0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}')\n"
            "    with open(dst_lbl, 'w') as f:\n"
            "        f.write('\\n'.join(lines))\n"
            "    copied[split] += 1\n\n"
            "print(f'[COCO] Wrote YOLO dataset: {copied[\"train\"]} train / {copied[\"val\"]} val images')\n\n"
            "# Write data.yaml\n"
            "yaml_content = (\n"
            "    f'# COCO Person Detection (auto-generated)\\n'\n"
            "    f'# Source: http://cocodataset.org  |  License: CC BY 4.0\\n\\n'\n"
            "    f'path: {output_path.resolve()}\\n'\n"
            "    f'train: train/images\\n'\n"
            "    f'val: val/images\\n\\n'\n"
            "    f'nc: 1\\n'\n"
            "    f'names: [person]\\n'\n"
            ")\n"
            "with open(output_path / 'data.yaml', 'w') as f:\n"
            "    f.write(yaml_content)\n"
            "print('[COCO] data.yaml written.')\n"
            "print('[COCO] Download complete!')\n"
        )

        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(ssh_host, username=ssh_user, password=ssh_pass, timeout=15)

            sftp = client.open_sftp()
            f = sftp.file('/tmp/dl_coco_person.py', 'wb', -1)
            f.write(script.encode())
            f.close()
            sftp.close()

            stdin, stdout, stderr = client.exec_command(
                '/home/wangxin/yolo-auto-training/training-venv/bin/python /tmp/dl_coco_person.py 2>&1',
                timeout=900
            )
            output = stdout.read().decode(errors='replace')
            error = stderr.read().decode(errors='replace')
            client.close()

            print(f"[COCO Download] stdout: {output}")
            if error:
                print(f"[COCO Download] stderr: {error}")

            if 'Download complete!' not in output:
                raise RuntimeError(f"COCO download script failed. Output: {output[:500]}")
        except Exception as e:
            raise RuntimeError(f"SSH COCO download failed: {e}")

    def _generate_data_yaml_ssh(self, dataset_path: str) -> None:
        """Generate data.yaml on GPU server based on actual dataset structure."""
        import paramiko

        ssh_host = os.getenv("GPU_SERVER_HOST", "192.168.11.3")
        ssh_user = os.getenv("GPU_SERVER_USER", "wangxin")
        ssh_pass = os.getenv("GPU_SERVER_PASS", "123123")

        script = (
            "import os, json\n"
            "from pathlib import Path\n\n"
            "base = Path(r'" + dataset_path + "')\n\n"
            "# Find actual dataset root\n"
            "dataset_root = base\n"
            "for item in base.iterdir():\n"
            "    if item.is_dir():\n"
            "        if (item / 'train').exists() or (item / 'data.yaml').exists():\n"
            "            dataset_root = item\n"
            "            break\n\n"
            "train_dir = dataset_root / 'train' / 'images'\n"
            "val_dir = dataset_root / 'val' / 'images'\n"
            "if not val_dir.exists():\n"
            "    val_dir = dataset_root / 'valid' / 'images'\n\n"
            "yaml_path = dataset_root / 'data.yaml'\n"
            "if yaml_path.exists():\n"
            "    print('data.yaml already exists, skipping generation.')\n"
            "else:\n"
            "    # Detect classes from label files\n"
            "    import glob\n"
            "    label_files = glob.glob(str(dataset_root / 'train' / 'labels' / '*.txt'))\n"
            "    if not label_files:\n"
            "        label_files = glob.glob(str(dataset_root / 'train' / 'labels' / '*.txt'))\n\n"
            "    class_ids = set()\n"
            "    for lf in label_files[:500]:\n"
            "        with open(lf) as f:\n"
            "            for line in f:\n"
            "                parts = line.strip().split()\n"
            "                if parts:\n"
            "                    class_ids.add(int(parts[0]))\n\n"
            "    num_classes = max(class_ids) + 1 if class_ids else 1\n\n"
            "    # Generate default class names\n"
            "    names = {i: f'class_{i}' for i in range(num_classes)}\n\n"
            "    yaml_content = (\n"
            "        f'path: {dataset_root.resolve()}\\n'\n"
            "        f'train: train/images\\n'\n"
            "        f'val: val/images\\n'\n"
            "        f'nc: {num_classes}\\n'\n"
            "        f'names: {names}\\n'\n"
            "    )\n"
            "    with open(yaml_path, 'w') as f:\n"
            "        f.write(yaml_content)\n"
            "    print('Generated data.yaml: ' + yaml_content)\n"
        )

        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(ssh_host, username=ssh_user, password=ssh_pass, timeout=15)

            sftp = client.open_sftp()
            f = sftp.file('/tmp/gen_yaml.py', 'wb', -1)
            f.write(script.encode())
            f.close()
            sftp.close()

            stdin, stdout, stderr = client.exec_command(
                '/home/wangxin/yolo-auto-training/training-venv/bin/python /tmp/gen_yaml.py 2>&1',
                timeout=60
            )
            output = stdout.read().decode(errors='replace')
            error = stderr.read().decode(errors='replace')
            client.close()

            print(f"[YAML] stdout: {output}")
            if error:
                print(f"[YAML] stderr: {error}")
        except Exception as e:
            print(f"[YAML] Warning: {e}")

    def run_phase3(self, task_id: str, user_id: str) -> None:
        """Phase 3: Submit actual training job to GPU server."""
        r = self._get_redis()
        data = r.hgetall(f"agent:{task_id}")
        confirmed_training = data.get("confirmed_training") == "true"

        overrides_json = r.hget(f"agent:{task_id}", "overrides_training_confirmation")
        overrides = json.loads(overrides_json) if overrides_json else {}

        model = overrides.get("model", "yolo11x")     # Upgraded: yolo11n → yolo11x for mAP target
        epochs = overrides.get("epochs", 200)          # Upgraded: 50 → 200 for mAP target
        imgsz = overrides.get("imgsz", 1280)          # Upgraded: 640 → 1280 for small object detection
        batch = overrides.get("batch", 8)            # Halved from 16 due to 1280px images
        device = overrides.get("device", "cuda:0")
        augmentation_preset = overrides.get("augmentation_preset", "strong")  # Strong for 90%+ mAP
        resume_from = overrides.get("resume_from", None)

        r.hset(f"agent:{task_id}", mapping={
            "status": "training",
            "current_agent": "ML Engineer",
            "progress": "55.0",
            "training_model": model,
            "training_epochs": str(epochs),
            "training_imgsz": str(imgsz),
        })

        try:
            from src.api.training_client import TrainingAPIClient
            client = TrainingAPIClient(
                base_url=os.getenv("TRAINING_API_URL", "http://192.168.11.3:8001"),
                api_key=os.getenv("TRAINING_API_KEY", "5M2oDsEfm0KxwSwFhLDtsq77FGztUY9DapuwQPx0fSE"),
            )

            # Use curriculum (3-stage progressive) training for HiTL
            # Stage 1: rapid_validation (50ep @ 640px) — validates pipeline cheaply
            # Stage 2: deep_training (150ep @ 1280px) — main training
            # Stage 3: fine_tuning (100ep @ 1280px) — reduced augmentation for detail
            #
            # Decision gates:
            #   Stage1 mAP50 < 0.50 → ABORT (pipeline broken)
            #   Stage2 mAP50 >= 0.90 → GOAL REACHED, stop
            #   Stage2 mAP50 >= 0.80 → proceed to Stage 3
            #   Stage2 mAP50 < 0.80 → trigger plateau strategies (LR decay / data expansion)
            stage1 = {
                "name": "rapid_validation",
                "epochs": 50,
                "imgsz": 640,
                "batch": 16,
                "model": "yolo11m",
                "augmentation_preset": "balanced",
            }
            stage2 = {
                "name": "deep_training",
                "epochs": 150,
                "imgsz": 1280,
                "batch": 8,
                "model": "yolo11x",
                "augmentation_preset": "strong",
                "mosaic": 1.0,
                "mixup": 0.3,
                "copy_paste": 0.4,
                "degrees": 15.0,
                "translate": 0.2,
                "scale": 0.7,
                "close_mosaic": 15,
            }
            stage3 = {
                "name": "fine_tuning",
                "epochs": 100,
                "imgsz": 1280,
                "batch": 8,
                "model": "yolo11x",
                "augmentation_preset": "strong",
                "mosaic": 0.0,       # Disable mosaic for fine detail
                "mixup": 0.1,
                "copy_paste": 0.1,
                "degrees": 5.0,
                "translate": 0.1,
                "scale": 0.5,
                "close_mosaic": 100,  # Mosaic disabled entirely
            }

            # dataset_path is a directory; curriculum needs the actual YAML file path
            # Use string join to avoid Windows backslash path issues
            import posixpath
            _dataset_dir = data.get("dataset_path", "/home/wangxin/data/fire-smoke")
            _yaml_path = posixpath.join(_dataset_dir, "data.yaml")
            result = client.start_curriculum_sync(
                task_id=task_id,
                data_yaml=_yaml_path,
                output_dir="/home/wangxin/runs",
                device=device,
                auto_export=True,
                stage1_min_map=0.50,
                stage2_target_map=0.90,
                stage2_min_for_stage3=0.80,
                stage1_overrides=stage1,
                stage2_overrides=stage2,
                stage3_overrides=stage3,
            )
            training_task_id = result.get("task_id", task_id)
            r.hset(f"agent:{task_id}", mapping={
                "status": "training",
                "training_type": "curriculum",
                "progress": "10.0",  # Stage 1 just started
                "training_task_id": training_task_id,
            })

            # Poll Training API for completion (runs in background thread)
            self._poll_training(task_id, training_task_id, client)

            # Start the GPU task queue scheduler for autonomous dispatch
            start_scheduler()
        except Exception as e:
            r.hset(f"agent:{task_id}", mapping={
                "status": "failed",
                "error": f"Training submission failed: {e}",
                "completed_at": datetime.now().isoformat(),
            })

    def _poll_training(self, task_id: str, training_task_id: str, client) -> None:
        """Poll Training API for training completion and update Redis.

        Dynamically reacts to plateau signals from the Training API:
        - lr_decay_triggered: Logs LR reduction signal (action: Business API can cancel & restart with adjusted lr0)
        - augment_boost_active: Logs augmentation boost activation
        - data_expansion_requested: Triggers dataset search via DatasetDiscovery agent
        """
        import time
        max_wait = 7200  # 2 hours max
        start = time.time()
        r = self._get_redis()
        _expansion_notified = False  # Track if we've already triggered dataset search for this request

        # Spawn AutoAdjustAgent for autonomous plateau-breaking
        auto_adjust_agent = AutoAdjustAgent(task_id=task_id, training_task_id=training_task_id)
        auto_adjust_agent.start()

        try:
            while time.time() - start < max_wait:
                time.sleep(30)
                try:
                    # Use sync client to avoid asyncio.run() in thread
                    status_data = client.get_task_status_sync(training_task_id)
                    status = status_data.get("status", "unknown")
                    progress = status_data.get("progress", 60)

                    # Extract plateau signals
                    live_mAP50 = status_data.get("live_mAP50")
                    lr_decay_triggered = status_data.get("lr_decay_triggered", False)
                    lr_decay_signal = status_data.get("lr_decay_signal") or {}
                    augment_boost_active = status_data.get("augment_boost_active", False)
                    augment_boost_signal = status_data.get("augment_boost_signal")
                    data_expansion_requested = status_data.get("data_expansion_requested", False)
                    data_expansion_signal = status_data.get("data_expansion_signal") or {}
                    strategies_triggered = status_data.get("strategies_triggered") or []
                    # Curriculum-specific fields
                    curriculum_stage = status_data.get("curriculum_stage", "")
                    curriculum_stage_mAP = status_data.get("curriculum_stage_mAP")
                    curriculum_history = status_data.get("curriculum_stage_history") or []

                    # Build mapping with plateau info + curriculum info
                    redis_mapping = {
                        "status": "training",
                        "progress": str(progress),
                        "training_status": status,
                        "poll_raw": str(status_data),
                        "live_mAP50": str(live_mAP50) if live_mAP50 is not None else "",
                        "lr_decay_triggered": str(lr_decay_triggered),
                        "augment_boost_active": str(augment_boost_active),
                        "data_expansion_requested": str(data_expansion_requested),
                        "strategies_triggered": str(strategies_triggered),
                        "curriculum_stage": curriculum_stage,
                        "curriculum_stage_mAP": str(curriculum_stage_mAP) if curriculum_stage_mAP is not None else "",
                        "curriculum_stage_history": str(curriculum_history),
                    }

                    # React to LR decay signal: log prominently for ops team
                    if lr_decay_triggered and lr_decay_signal:
                        count = lr_decay_signal.get("lr_decay_count", "?")
                        factor = lr_decay_signal.get("factor", 0.5)
                        epoch = lr_decay_signal.get("epoch", "?")
                        mAP = lr_decay_signal.get("current_mAP50", 0)
                        print(f"[PLATEAU ALERT] Task {task_id}: LR decay #{count} triggered "
                              f"at epoch {epoch}, mAP50={mAP:.4f}, factor={factor}")

                    # React to data expansion request: trigger DatasetDiscovery agent
                    if data_expansion_requested and not _expansion_notified:
                        _expansion_notified = True
                        rec = data_expansion_signal.get("recommendation", "")
                        print(f"[PLATEAU ALERT] Task {task_id}: Data expansion triggered! "
                              f"mAP50={data_expansion_signal.get('current_mAP50', 0):.4f}, "
                              f"target={data_expansion_signal.get('target_mAP50', 0):.2f}")
                        print(f"[PLATEAU] Recommendation: {rec}")

                        # Trigger dataset search for more fire/smoke data
                        self._trigger_dataset_search(task_id, data_expansion_signal)

                        redis_mapping["plateau_action"] = "dataset_search_triggered"
                        redis_mapping["expansion_recommendation"] = rec

                    # Log augmentation boost
                    if augment_boost_active and augment_boost_signal:
                        ep = augment_boost_signal.get("start_epoch", "?")
                        rem = status_data.get("augment_boost_remaining", "?")
                        mix = augment_boost_signal.get("mixup", "?")
                        cp = augment_boost_signal.get("copy_paste", "?")
                        print(f"[PLATEAU ALERT] Task {task_id}: Augmentation boost active "
                              f"at EP {ep}, remaining={rem}, mixup={mix}, copy_paste={cp}")

                    # Log curriculum stage progress
                    if curriculum_stage:
                        stage_mAP_str = f", mAP50={curriculum_stage_mAP:.4f}" if curriculum_stage_mAP else ""
                        print(f"[CURRICULUM] Task {task_id}: {curriculum_stage}{stage_mAP_str}, progress={progress:.1f}%")

                    r.hset(f"agent:{task_id}", mapping=redis_mapping)

                    if status in ("completed", "success"):
                        auto_adjust_agent.stop()
                        model_path = status_data.get("model_path", "/home/wangxin/runs/train/weights/best.pt")
                        project_name = data.get("project_name", task_id)
                        r.hset(f"agent:{task_id}", mapping={
                            "status": "training_completed",
                            "progress": "90.0",
                            "model_path": model_path,
                        })
                        # Trigger auto export + deploy
                        self._auto_export_and_deploy(task_id, model_path, project_name)
                        return
                    elif status in ("failed", "error"):
                        auto_adjust_agent.stop()
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
            auto_adjust_agent.stop()
            r.hset(f"agent:{task_id}", mapping={
                "status": "failed",
                "error": "Training timeout (>2h)",
                "completed_at": datetime.now().isoformat(),
            })
        except Exception as e:
            # Top-level: prevent thread from crashing silently
            auto_adjust_agent.stop()
            r.hset(f"agent:{task_id}", mapping={
                "status": "failed",
                "error": f"Polling crashed: {e}",
                "completed_at": datetime.now().isoformat(),
            })

    def _auto_export_and_deploy(self, task_id: str, model_path: str, project_name: str) -> None:
        """Automatically export and deploy after training completes.

        Spawns a background thread that:
          1. Submits an ONNX export job to the Training API
          2. Polls export status for up to 5 minutes
          3. On success, registers deployment via submit_deployment()
          4. Updates Redis with export/deploy status
        """
        def _do_export_deploy():
            try:
                from src.api.training_client import TrainingAPIClient
                client = TrainingAPIClient(
                    base_url=os.getenv("TRAINING_API_URL", "http://192.168.11.3:8001"),
                    api_key=os.getenv("TRAINING_API_KEY", "5M2oDsEfm0KxwSwFhLDtsq77FGztUY9DapuwQPx0fSE"),
                )
                r = self._get_redis()

                export_task_id = f"{task_id}_export"
                print(f"[{task_id}] Auto-export triggered: model={model_path}")

                # Step 1: Submit export
                export_resp = client.start_export_sync(
                    task_id=export_task_id,
                    model_path=model_path,
                    platform="jetson_orin",
                    formats=["onnx"],
                    imgsz=640,
                )
                print(f"[{task_id}] Export submitted: {export_resp}")
                r.hset(f"agent:{task_id}", mapping={
                    "export_status": "submitted",
                    "export_task_id": export_task_id,
                })

                # Step 2: Poll export status (up to 5 minutes, 10s intervals)
                export_done = False
                for attempt in range(30):
                    time.sleep(10)
                    try:
                        status_resp = client.get_export_status_sync(export_task_id)
                        export_status = status_resp.get("status", "unknown")
                        print(f"[{task_id}] Export status check {attempt+1}/30: {export_status}")
                        if export_status in ("completed", "exported", "done", "success"):
                            export_done = True
                            break
                        if export_status in ("failed", "error"):
                            print(f"[{task_id}] Export failed: {status_resp.get('error', export_status)}")
                            r.hset(f"agent:{task_id}", mapping={
                                "export_status": "failed",
                                "export_error": str(status_resp.get("error", export_status)),
                            })
                            return
                    except Exception as poll_err:
                        print(f"[{task_id}] Export poll error: {poll_err}")

                if not export_done:
                    print(f"[{task_id}] Export timed out after 5 minutes")
                    r.hset(f"agent:{task_id}", mapping={"export_status": "timeout"})
                    return

                print(f"[{task_id}] Export completed successfully")
                r.hset(f"agent:{task_id}", mapping={
                    "export_status": "completed",
                    "exported_model_path": status_resp.get("export_path", model_path),
                })

                # Step 3: Deploy
                try:
                    deploy_resp = client.submit_deployment(
                        model_path=model_path,
                        platform="jetson_orin",
                        imgsz=640,
                    )
                    print(f"[{task_id}] Auto-deploy submitted: {deploy_resp}")
                    r.hset(f"agent:{task_id}", mapping={
                        "deployment_status": "deployed",
                        "deploy_id": deploy_resp.get("deploy_id", ""),
                        "deploy_platform": deploy_resp.get("platform", "jetson_orin"),
                        "deployment_config": json.dumps(deploy_resp.get("config", {})),
                    })
                except Exception as deploy_err:
                    print(f"[{task_id}] Auto-deploy failed: {deploy_err}")
                    r.hset(f"agent:{task_id}", mapping={
                        "deployment_status": "failed",
                        "deployment_error": str(deploy_err),
                    })

            except Exception as e:
                print(f"[{task_id}] Auto export/deploy chain failed: {e}")
                try:
                    r = self._get_redis()
                    r.hset(f"agent:{task_id}", mapping={
                        "export_status": "failed",
                        "export_error": str(e),
                    })
                except Exception:
                    pass

        thread = threading.Thread(target=_do_export_deploy, daemon=True, name=f"AutoExport-{task_id[:8]}")
        thread.start()
        print(f"[{task_id}] Auto-export-and-deploy thread started")

    def _trigger_dataset_search(self, task_id: str, expansion_signal: dict) -> None:
        """Trigger dataset search via DatasetDiscovery when training plateaus.

        Searches HuggingFace and other sources for additional fire/smoke datasets
        to expand the training set and break the plateau.
        """
        try:
            from src.data.discovery import DatasetDiscovery
            discovery = DatasetDiscovery()

            # Search for fire/smoke datasets on HuggingFace
            search_results = discovery.search("fire smoke detection", top_k=10, source_filter="huggingface")

            if search_results:
                datasets_found = []
                for ds in search_results[:5]:
                    datasets_found.append({
                        "name": ds.get("name", "unknown"),
                        "source": ds.get("source", "unknown"),
                        "url": ds.get("url", ""),
                        "size": ds.get("size", "unknown"),
                        "license": ds.get("license", "unknown"),
                    })

                r = self._get_redis()
                r.hset(f"agent:{task_id}", mapping={
                    "expansion_datasets_found": str(datasets_found),
                    "expansion_search_complete": "true",
                })
                print(f"[PLATEAU] Dataset search found {len(datasets_found)} candidates for expansion")
                for ds in datasets_found:
                    print(f"  - {ds['name']} ({ds['source']}, {ds['size']})")
            else:
                print(f"[PLATEAU] No additional datasets found on HuggingFace")
                r = self._get_redis()
                r.hset(f"agent:{task_id}", mapping={
                    "expansion_datasets_found": "[]",
                    "expansion_search_complete": "true",
                    "expansion_search_error": "No datasets found matching fire/smoke detection",
                })
        except Exception as e:
            print(f"[PLATEAU] Dataset search failed: {e}")
            r = self._get_redis()
            r.hset(f"agent:{task_id}", mapping={
                "expansion_search_complete": "true",
                "expansion_search_error": str(e),
            })

    def confirm(self, task_id: str, approved: bool, overrides: dict) -> bool:
        """Record human confirmation decision."""
        r = self._get_redis()
        current = r.hget(f"agent:{task_id}", "status")
        if not current:
            return False
        # Use overrides_awaiting_confirmation key for Phase 1 (dataset confirmation)
        override_key = "overrides_awaiting_confirmation" if current == "awaiting_confirmation" else f"overrides_{current}"
        r.hset(f"agent:{task_id}", mapping={
            f"confirmed_{current}": "true" if approved else "false",
            override_key: json.dumps(overrides),
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


class AutoAdjustAgent:
    """Background agent that monitors training plateau and auto-triggers adjustments.

    Runs in a background thread, polling the Training API every 60s. When plateau
    signals are detected (lr_decay, augment_boost, data_expansion), it autonomously:
      Level 1: Cancel current task + restart with halved lr0 + resume_from best.pt
      Level 2: Log augmentation boost (already handled by Training API)
      Level 3: Run ActiveLearning + SemiSupervised pipeline to expand dataset

    Usage:
        agent = AutoAdjustAgent(task_id, training_task_id)
        agent.start()   # Spawns background thread
        agent.stop()    # Terminates gracefully
    """

    def __init__(self, task_id: str, training_task_id: str):
        self.task_id = task_id
        self.training_task_id = training_task_id
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._adjustments_triggered: list[dict] = []
        self._lr0_history: list[float] = [0.01]  # Start with default

    def start(self) -> None:
        """Start the auto-adjustment background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"AutoAdjust-{self.task_id[:8]}")
        self._thread.start()
        print(f"[AutoAdjust] Started for task {self.task_id}")

    def stop(self) -> None:
        """Stop the background thread gracefully."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        print(f"[AutoAdjust] Stopped for task {self.task_id}")

    def _run(self) -> None:
        """Main polling loop."""
        import time
        import httpx

        r = _get_redis_for_adjust()
        base_url = os.getenv("TRAINING_API_URL", "http://192.168.11.3:8001")
        api_key = os.getenv("TRAINING_API_KEY", "5M2oDsEfm0KxwSwFhLDtsq77FGztUY9DapuwQPx0fSE")

        headers = {"X-API-Key": api_key}

        while not self._stop_event.is_set():
            try:
                # Poll Training API status
                with httpx.Client(timeout=15.0) as client:
                    resp = client.get(
                        f"{base_url}/api/v1/internal/train/status/{self.training_task_id}",
                        headers=headers,
                    )
                    if resp.status_code != 200:
                        time.sleep(60)
                        continue
                    status_data = resp.json()

                status = status_data.get("status", "unknown")
                live_mAP50 = status_data.get("live_mAP50")
                lr_decay_triggered = status_data.get("lr_decay_triggered", False)
                lr_decay_signal = status_data.get("lr_decay_signal") or {}
                data_expansion_requested = status_data.get("data_expansion_requested", False)
                data_expansion_signal = status_data.get("data_expansion_signal") or {}
                augment_boost_active = status_data.get("augment_boost_active", False)
                strategies_triggered = status_data.get("strategies_triggered") or []

                # Update Redis with current state
                r.hset(f"autoadjust:{self.task_id}", mapping={
                    "status": status,
                    "live_mAP50": str(live_mAP50) if live_mAP50 is not None else "",
                    "lr_decay_triggered": str(lr_decay_triggered),
                    "data_expansion_requested": str(data_expansion_requested),
                    "augment_boost_active": str(augment_boost_active),
                    "strategies_triggered": str(strategies_triggered),
                    "last_checked": datetime.now().isoformat(),
                })

                # Don't act if training already finished
                if status in ("completed", "failed", "cancelled"):
                    break

                # Level 1: LR decay — auto-restart with halved lr0 and resume
                if lr_decay_triggered and lr_decay_signal:
                    decay_count = lr_decay_signal.get("lr_decay_count", 1)
                    # Check if we already triggered this round
                    already_triggered = any(
                        a.get("level") == 1 and a.get("decay_count") == decay_count
                        for a in self._adjustments_triggered
                    )
                    if not already_triggered and decay_count <= 3:
                        print(f"[AutoAdjust] Level 1: LR decay #{decay_count} — triggering auto-adjust")
                        self._trigger_lr_adjustment(
                            lr_decay_signal=lr_decay_signal,
                            decay_count=decay_count,
                            base_url=base_url,
                            headers=headers,
                            r=r,
                        )

                # Level 3: Data expansion — run ActiveLearning + SemiSupervised pipeline
                if data_expansion_requested and data_expansion_signal:
                    already_expanded = any(a.get("level") == 3 for a in self._adjustments_triggered)
                    if not already_expanded:
                        print(f"[AutoAdjust] Level 3: Data expansion triggered")
                        self._trigger_data_expansion(
                            data_expansion_signal=data_expansion_signal,
                            base_url=base_url,
                            headers=headers,
                            r=r,
                        )

            except Exception as e:
                print(f"[AutoAdjust] Error in polling loop: {e}")

            # Poll every 60 seconds
            self._stop_event.wait(timeout=60)

    def _trigger_lr_adjustment(
        self,
        lr_decay_signal: dict,
        decay_count: int,
        base_url: str,
        headers: dict,
        r,
    ) -> None:
        """Cancel current task and restart with halved lr0 + resume_from best.pt."""
        try:
            current_lr = self._lr0_history[-1]
            new_lr = max(current_lr * 0.5, 1e-6)
            self._lr0_history.append(new_lr)

            # Get current task params from Redis
            task_data = r.hgetall(f"agent:{self.task_id}")
            params = json.loads(task_data.get("params", "{}")) if task_data.get("params") else {}

            model = params.get("model", "yolo11m")
            data_yaml = params.get("data_yaml", "/home/wangxin/data/fire-smoke/data.yaml")
            original_epochs = int(params.get("epochs", 100))
            imgsz = int(params.get("imgsz", 640))
            device = params.get("device", "cuda:0")

            # Find the best.pt from current run
            current_output_dir = task_data.get("output_dir", f"/home/wangxin/runs/{self.task_id}")
            best_pt = f"{current_output_dir}/weights/best.pt"

            # Generate new task ID
            new_task_id = f"train_{uuid.uuid4().hex[:8]}"

            # Cancel current training via Training API
            with httpx.Client(timeout=15.0) as client:
                client.post(
                    f"{base_url}/api/v1/internal/train/cancel/{self.training_task_id}",
                    headers=headers,
                )

            # Start new training with halved lr0 + resume
            hpo_params = {"lr0": new_lr}

            with httpx.Client(timeout=15.0) as client:
                resp = client.post(
                    f"{base_url}/api/v1/internal/train/start",
                    json={
                        "task_id": new_task_id,
                        "model": model,
                        "data_yaml": data_yaml,
                        "epochs": original_epochs + 50,  # Add extra epochs
                        "imgsz": imgsz,
                        "batch": 16,
                        "device": device,
                        "output_dir": f"/home/wangxin/runs/{new_task_id}",
                        "augmentation_preset": "strong",
                        "resume_from": best_pt,
                        "lr0": new_lr,
                    },
                    headers=headers,
                )
                resp.raise_for_status()
                result = resp.json()

            new_training_task_id = result.get("task_id", new_task_id)

            # Update Redis
            r.hset(f"agent:{self.task_id}", mapping={
                "status": "auto_adjusting",
                "adjusted_task_id": new_task_id,
                "adjusted_training_id": new_training_task_id,
                "lr_adjustment": str({
                    "old_lr": current_lr,
                    "new_lr": new_lr,
                    "decay_count": decay_count,
                    "resume_from": best_pt,
                    "timestamp": datetime.now().isoformat(),
                }),
            })

            self._adjustments_triggered.append({
                "level": 1,
                "decay_count": decay_count,
                "old_lr": current_lr,
                "new_lr": new_lr,
                "new_task_id": new_task_id,
                "timestamp": datetime.now().isoformat(),
            })

            print(f"[AutoAdjust] Level 1 complete: new task={new_task_id}, lr={new_lr:.6f}, resume={best_pt}")

        except Exception as e:
            print(f"[AutoAdjust] Level 1 failed: {e}")
            r.hset(f"agent:{self.task_id}", mapping={
                "lr_adjustment_error": str(e),
            })

    def _trigger_data_expansion(
        self,
        data_expansion_signal: dict,
        base_url: str,
        headers: dict,
        r,
    ) -> None:
        """Run ActiveLearning + SemiSupervised pipeline to expand dataset."""
        try:
            target_mAP = data_expansion_signal.get("target_mAP50", 0.90)
            current_mAP = data_expansion_signal.get("current_mAP50", 0)
            expansion_round = self._adjustments_triggered.count(
                lambda a: a.get("level") == 3
            ) + 1

            print(f"[AutoAdjust] Data expansion round {expansion_round}: "
                  f"mAP50={current_mAP:.4f} → target={target_mAP:.2f}")

            # Get model path
            task_data = r.hgetall(f"agent:{self.task_id}")
            model_path = task_data.get("model_path")
            if not model_path:
                # Try to find best.pt in output dir
                output_dir = task_data.get("output_dir", f"/home/wangxin/runs/{self.task_id}")
                model_path = f"{output_dir}/weights/best.pt"

            if not model_path:
                print("[AutoAdjust] No model path found, skipping data expansion")
                return

            # Step 1: ActiveLearning — select most uncertain samples via SSH on GPU server
            # NOTE: Business API server cannot check GPU server filesystem directly.
            # Use paramiko SSH to enumerate unlabeled image directories on the GPU server.
            unlabeled_dirs = [
                "/home/wangxin/data/unlabeled_fire",
                "/home/wangxin/data/raw",
                "/home/wangxin/data/fire-smoke/unlabeled",
            ]

            selected_samples = []
            # SSH bridge: check GPU server filesystem via paramiko
            try:
                import paramiko
                ssh_host = os.getenv("GPU_SERVER_HOST", "192.168.11.3")
                ssh_user = os.getenv("GPU_SERVER_USER", "wangxin")
                ssh_pass = os.getenv("GPU_SERVER_PASS", "123123")
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(ssh_host, username=ssh_user, password=ssh_pass, timeout=10)

                for img_dir in unlabeled_dirs:
                    # Check via SSH that the directory exists AND has image files
                    check_script = (
                        f"import sys; from pathlib import Path; p=Path(r'{img_dir}'); "
                        f"imgs=list(p.glob('*.jpg'))+list(p.glob('*.png'))+list(p.glob('*.jpeg'))+list(p.glob('*.bmp')); "
                        f"print(f'EXISTS {{len(imgs)}}')"
                    )
                    stdin, stdout, stderr = ssh.exec_command(
                        f'/home/wangxin/yolo-auto-training/training-venv/bin/python -c "{check_script}"',
                        timeout=15
                    )
                    stdout.channel.recv_exit_status()
                    line = stdout.read().decode(errors='replace').strip()
                    if not line.startswith("EXISTS "):
                        continue
                    count = int(line.split()[1])
                    if count == 0:
                        continue
                    print(f"[AutoAdjust] Found {count} unlabeled images in {img_dir}")

                    # Run ActiveLearning on GPU server via SSH
                    al_script = (
                        "import sys, json\n"
                        "sys.path.insert(0, '/home/wangxin/yolo-auto-training/training-api/src')\n"
                        "from training.active_learner import ActiveLearningPipeline, ActiveLearningConfig\n"
                        f"al = ActiveLearningPipeline(config=ActiveLearningConfig(strategy='entropy', top_k=200, batch_size=16))\n"
                        f"result = al.select_uncertain_samples(model_path='{model_path}', image_dir='{img_dir}')\n"
                        "print(json.dumps(result, default=str))"
                    )
                    stdin2, stdout2, stderr2 = ssh.exec_command(
                        f'/home/wangxin/yolo-auto-training/training-venv/bin/python -c "{al_script}"',
                        timeout=120
                    )
                    stdout2.channel.recv_exit_status()
                    output = stdout2.read().decode(errors='replace')
                    if output.strip():
                        try:
                            result = json.loads(output)
                            if result.get("selected"):
                                selected_samples.extend(result["selected"])
                                print(f"[AutoAdjust] AL: found {len(result['selected'])} uncertain samples in {img_dir}")
                        except json.JSONDecodeError:
                            print(f"[AutoAdjust] AL parse error for {img_dir}: {output[:200]}")
                ssh.close()
            except Exception as e:
                print(f"[AutoAdjust] SSH-based AL failed: {e}")

            if not selected_samples:
                print("[AutoAdjust] No unlabeled images found — data expansion not possible")
                r.hset(f"agent:{self.task_id}", mapping={
                    "expansion_status": "no_unlabeled_data",
                    "expansion_note": "No unlabeled image directories found on GPU server",
                })
                self._adjustments_triggered.append({
                    "level": 3,
                    "round": expansion_round,
                    "status": "no_data",
                    "timestamp": datetime.now().isoformat(),
                })
                return

            # Step 2: SemiSupervised — generate pseudo-labels via SSH on GPU server
            pseudo_labels = []
            sample_paths = [s["path"] for s in selected_samples[:500]]
            if sample_paths:
                try:
                    import paramiko
                    ssh_host = os.getenv("GPU_SERVER_HOST", "192.168.11.3")
                    ssh_user = os.getenv("GPU_SERVER_USER", "wangxin")
                    ssh_pass = os.getenv("GPU_SERVER_PASS", "123123")
                    ssh = paramiko.SSHClient()
                    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    ssh.connect(ssh_host, username=ssh_user, password=ssh_pass, timeout=10)

                    # Build Python list string for sample paths
                    paths_repr = repr(sample_paths)
                    ss_script = (
                        "import sys, json\n"
                        "sys.path.insert(0, '/home/wangxin/yolo-auto-training/training-api/src')\n"
                        "from training.semi_supervised import SemiSupervisedPipeline\n"
                        f"ss = SemiSupervisedPipeline(confidence_threshold=0.75)\n"
                        f"pseudo_labels = ss.generate_pseudo_labels(\n"
                        f"    teacher_model_path='{model_path}',\n"
                        f"    unlabeled_images={paths_repr},\n"
                        f"    method='yolo_teacher',\n"
                        ")\n"
                        "print(json.dumps([{'path': p.path, 'boxes': p.boxes, 'confidence': p.confidence} for p in pseudo_labels], default=str))"
                    )
                    stdin, stdout, stderr = ssh.exec_command(
                        f'/home/wangxin/yolo-auto-training/training-venv/bin/python -c "{ss_script}"',
                        timeout=600
                    )
                    stdout.channel.recv_exit_status()
                    output = stdout.read().decode(errors='replace')
                    stderr_err = stderr.read().decode(errors='replace')
                    if stderr_err:
                        print(f"[AutoAdjust] SS stderr: {stderr_err[:300]}")
                    if output.strip():
                        try:
                            pseudo_labels = json.loads(output)
                            print(f"[AutoAdjust] SS: generated {len(pseudo_labels)} pseudo-labels")
                        except json.JSONDecodeError:
                            print(f"[AutoAdjust] SS parse error: {output[:200]}")
                    ssh.close()
                except Exception as e:
                    print(f"[AutoAdjust] SSH-based SS failed: {e}")

            if not pseudo_labels:
                print("[AutoAdjust] No pseudo-labels generated")
                return

            # Step 3: Filter and create expanded dataset via SSH
            filtered = []
            expanded_yaml = None
            if pseudo_labels:
                try:
                    import paramiko
                    ssh_host = os.getenv("GPU_SERVER_HOST", "192.168.11.3")
                    ssh_user = os.getenv("GPU_SERVER_USER", "wangxin")
                    ssh_pass = os.getenv("GPU_SERVER_PASS", "123123")
                    ssh = paramiko.SSHClient()
                    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    ssh.connect(ssh_host, username=ssh_user, password=ssh_pass, timeout=10)

                    pseudo_repr = repr(pseudo_labels[:500])
                    filter_script = (
                        "import sys, json\n"
                        "sys.path.insert(0, '/home/wangxin/yolo-auto-training/training-api/src')\n"
                        "from training.semi_supervised import SemiSupervisedPipeline, PseudoLabel\n"
                        f"ss = SemiSupervisedPipeline(confidence_threshold=0.75)\n"
                        f"pseudo_labels = [PseudoLabel(**p) for p in {pseudo_repr}]\n"
                        f"filtered = ss.filter_pseudo_labels(pseudo_labels, min_boxes=1, max_boxes=50)\n"
                        f"expanded_dir = '/home/wangxin/data/expanded_{self.task_id}'\n"
                        f"expanded_yaml = ss.create_pseudo_dataset(filtered, output_dir=expanded_dir, class_names=['fire', 'smoke'])\n"
                        "print(json.dumps({'filtered': len(filtered), 'yaml': expanded_yaml}))"
                    )
                    stdin, stdout, stderr = ssh.exec_command(
                        f'/home/wangxin/yolo-auto-training/training-venv/bin/python -c "{filter_script}"',
                        timeout=300
                    )
                    stdout.channel.recv_exit_status()
                    output = stdout.read().decode(errors='replace')
                    if output.strip():
                        try:
                            result = json.loads(output)
                            filtered_count = result.get("filtered", 0)
                            expanded_yaml = result.get("yaml")
                            print(f"[AutoAdjust] Filtered to {filtered_count} quality pseudo-labels")
                            if expanded_yaml:
                                print(f"[AutoAdjust] Expanded dataset created: {expanded_yaml}")
                        except json.JSONDecodeError:
                            print(f"[AutoAdjust] Filter/create error: {output[:200]}")
                    ssh.close()
                except Exception as e:
                    print(f"[AutoAdjust] SSH-based filter/create failed: {e}")

            if not filtered or not expanded_yaml:
                return

            # Step 4: Submit new training with expanded dataset
            task_data = r.hgetall(f"agent:{self.task_id}")
            params = json.loads(task_data.get("params", "{}")) if task_data.get("params") else {}
            model = params.get("model", "yolo11m")
            imgsz = int(params.get("imgsz", 640))
            device = params.get("device", "cuda:0")

            new_task_id = f"train_{uuid.uuid4().hex[:8]}"

            with httpx.Client(timeout=15.0) as client:
                resp = client.post(
                    f"{base_url}/api/v1/internal/train/start",
                    json={
                        "task_id": new_task_id,
                        "model": model,
                        "data_yaml": expanded_yaml,
                        "epochs": 100,
                        "imgsz": imgsz,
                        "batch": 16,
                        "device": device,
                        "output_dir": f"/home/wangxin/runs/{new_task_id}",
                        "augmentation_preset": "strong",
                        "resume_from": model_path,
                    },
                    headers=headers,
                )
                resp.raise_for_status()
                result = resp.json()

            new_training_task_id = result.get("task_id", new_task_id)

            r.hset(f"agent:{self.task_id}", mapping={
                "status": "expanding_data",
                "expansion_task_id": new_task_id,
                "expansion_training_id": new_training_task_id,
                "expansion_pseudo_labels": str(len(filtered)),
                "expansion_dataset": expanded_yaml,
            })

            self._adjustments_triggered.append({
                "level": 3,
                "round": expansion_round,
                "status": "success",
                "pseudo_labels": len(filtered),
                "new_task_id": new_task_id,
                "timestamp": datetime.now().isoformat(),
            })

            print(f"[AutoAdjust] Level 3 complete: new task={new_task_id}, "
                  f"pseudo_labels={len(filtered)}, yaml={expanded_yaml}")

        except Exception as e:
            print(f"[AutoAdjust] Level 3 failed: {e}")
            r.hset(f"agent:{self.task_id}", mapping={
                "expansion_status": "failed",
                "expansion_error": str(e),
            })
            self._adjustments_triggered.append({
                "level": 3,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })


def _get_redis_for_adjust():
    """Get Redis client for AutoAdjustAgent."""
    import redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_password = os.getenv("REDIS_PASSWORD", None)
    try:
        if redis_password:
            return redis.from_url(redis_url, password=redis_password, decode_responses=True)
        return redis.from_url(redis_url, decode_responses=True)
    except Exception:
        # Fallback: try direct connection
        try:
            return redis.Redis(host="localhost", port=6379, db=0, decode_responses=True, password=redis_password)
        except Exception:
            return None
