# YOLO Auto-Training System - User Manual

**Version**: 1.0
**Date**: 2026-03-14

---

## 1. System Overview

YOLO Auto-Training System is an AI-driven automated YOLO model training and deployment platform built on CrewAI. It provides end-to-end workflow from dataset discovery to model deployment on edge devices.

### 1.1 Key Features

| Feature | Description |
|---------|-------------|
| **Dataset Discovery** | Search datasets from Roboflow, Kaggle, HuggingFace |
| **Auto Training** | YOLO11 training with Ray Tune hyperparameter optimization |
| **Knowledge Distillation** | Train smaller models from larger ones |
| **Edge Deployment** | Export to ONNX/TensorRT for Jetson, RK3588 devices |
| **Multi-Agent Orchestration** | CrewAI-powered intelligent workflow |

### 1.2 Supported Models

| Model | Parameters | Speed (FPS) | Use Case |
|-------|------------|-------------|----------|
| YOLO11n | 2.6M | 100+ | Edge devices |
| YOLO11s | 9.7M | 60+ | Mobile |
| YOLO11m | 25.9M | 40+ | Server |
| YOLO11l | 51.5M | 25+ | High accuracy |
| YOLO11x | 97.2M | 15+ | Maximum accuracy |

---

## 2. Quick Start

### 2.1 Installation

```bash
# Clone and install
cd yolo-auto-training
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### 2.2 Environment Variables

Create `.env` file:

```bash
# Required for dataset discovery
ROBOFLOW_API_KEY=your_roboflow_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
HUGGINGFACE_TOKEN=your_hf_token

# Redis (for task queue)
REDIS_URL=redis://localhost:6379/0

# JWT Secret
JWT_SECRET_KEY=your_secret_key
```

### 2.3 Start Services

```bash
# Start Redis
redis-server

# Start API server
uvicorn src.api.gateway:app --host 0.0.0.0 --port 8000

# Start Celery worker (separate terminal)
celery -A src.api.tasks worker --loglevel=info
```

---

## 3. API Usage Guide

### 3.1 Health Check

```bash
GET /health

Response:
{
  "status": "healthy",
  "version": "6.0.0"
}
```

### 3.2 Dataset Search

Search for datasets across multiple sources.

```bash
POST /api/v1/data/search

Request:
{
  "query": "car detection",
  "max_results": 10
}

Response:
{
  "datasets": [
    {
      "name": "car-detection",
      "source": "roboflow",
      "url": "https://universe.roboflow.com/...",
      "license": "MIT",
      "images": 1000,
      "relevance_score": 0.95
    }
  ],
  "total": 1
}
```

### 3.3 Start Training

Submit a training job.

```bash
POST /api/v1/train/start

Request:
{
  "data_yaml": "/data/car detection.yaml",
  "model": "yolo11m",
  "epochs": 100,
  "imgsz": 640
}

Response:
{
  "task_id": "train_abc123",
  "status": "submitted",
  "message": "Training job submitted successfully"
}
```

### 3.4 Check Training Status

```bash
GET /api/v1/train/status/{task_id}

Response:
{
  "task_id": "train_abc123",
  "status": "running",
  "progress": 0.45
}
```

### 3.5 Export Model

Export trained model for edge deployment.

```bash
POST /api/v1/deploy/export

Request:
{
  "model_path": "/runs/train/weights/best.pt",
  "platform": "jetson_orin",
  "imgsz": 640
}

Response:
{
  "task_id": "export_xyz789",
  "status": "submitted"
}
```

---

## 4. Python SDK Usage

### 4.1 Dataset Discovery

```python
from src.data.discovery import DatasetDiscovery

# Initialize
discovery = DatasetDiscovery(output_dir="./data")

# Search datasets
results = discovery.search("car detection", max_results=10)

for ds in results:
    print(f"{ds.name} ({ds.source}) - Score: {ds.relevance_score}")

# Download dataset
discovery.download(results[0], output_path="./data/car")
```

### 4.2 Training

```python
from src.training.runner import YOLOTrainer

# Initialize trainer
trainer = YOLOTrainer(
    model="yolo11m",
    output_dir="./runs"
)

# Train model
result = trainer.train(
    data_yaml="./data/car/data.yaml",
    epochs=100,
)

print(f"Status: {result.status}")
print(f"Model: {result.model_path}")
print(f"mAP50: {result.metrics.get('mAP50', 0):.3f}")
```

### 4.3 Hyperparameter Optimization

```python
from src.training.runner import YOLOTrainer
from src.training.config import HPOConfig

# Configure HPO
hpo_config = HPOConfig(
    n_trials=50,
    epochs_per_trial=30,
)

# Run HPO
trainer = YOLOTrainer(model="yolo11n", output_dir="./runs")
result = trainer.tune(
    data_yaml="./data/car/data.yaml",
    config=hpo_config,
)

print(f"Best params: {result.best_params}")
print(f"Best mAP50: {result.metrics.get('mAP50', 0):.3f}")
```

### 4.4 Model Export

```python
from src.deployment.exporter import ModelExporter

# Initialize exporter
exporter = ModelExporter(output_dir="./runs/export")

# Export to ONNX
result = exporter.export(
    model_path="./runs/train/weights/best.pt",
    platform="jetson_orin",
    imgsz=640,
)

print(f"Format: {result.format}")
print(f"Size: {result.size_mb} MB")
print(f"Path: {result.model_path}")
```

---

## 5. Agent Workflow

### 5.1 CrewAI Integration

```python
from src.agents.orchestration import create_training_crew

# Create crew
crew = create_training_crew()

# Execute with task description
result = crew.kickoff(
    inputs={"task_description": "Detect cars on highway"}
)

print(result)
```

### 5.2 Decision Rules

The system uses intelligent decision rules:

| Agent | Decision Rules |
|-------|---------------|
| **Data Discovery** | Score > 0.8: Select directly; 0.5-0.8: Include with warning; < 0.5: Generate synthetic |
| **Training** | < 1000 images: Aggressive augmentation; mAP50 < 0.5: Try larger model; Edge: Use YOLO11n |
| **Deployment** | FPS < 20: Optimize; Memory < 2GB: INT8 quantization; Fail: Rollback |

---

## 6. Supported Platforms

### 6.1 Edge Devices

| Platform | Format | Optimization |
|----------|--------|--------------|
| Jetson Nano | TensorRT FP16 | 10-30 FPS |
| Jetson Orin | TensorRT FP16 | 50-100 FPS |
| RK3588 | ONNX FP16 | 30-50 FPS |
| Generic ARM | ONNX FP32 | 10-20 FPS |

### 6.2 Cloud/Server

| Platform | Format | Use Case |
|----------|--------|----------|
| Cloud CPU | ONNX FP32 | Batch inference |
| Cloud GPU | TensorRT FP16 | Real-time |
| Server | PyTorch | Research |

---

## 7. Troubleshooting

### 7.1 Common Issues

| Issue | Solution |
|-------|----------|
| API returns 401 | Check JWT_SECRET_KEY and authentication headers |
| Rate limited (429) | Wait 1 minute or adjust rate limit config |
| Training stuck | Check Celery worker logs |
| Model export fails | Verify CUDA availability |

### 7.2 API Key Setup

```bash
# Roboflow
# 1. Go to https://app.roboflow.com/settings/api
# 2. Copy your API key

# Kaggle
# 1. Go to https://www.kaggle.com/account
# 2. Create new API token

# HuggingFace
# 1. Go to https://huggingface.co/settings/tokens
# 2. Create new token with "read" permission
```

---

## 8. Examples

### 8.1 Complete Training Pipeline

```python
from src.data.discovery import DatasetDiscovery
from src.training.runner import YOLOTrainer
from src.deployment.exporter import ModelExporter

# 1. Discover dataset
discovery = DatasetDiscovery()
datasets = discovery.search("car detection", max_results=5)
print(f"Found {len(datasets)} datasets")

# 2. Download best dataset
best = max(datasets, key=lambda x: x.relevance_score)
discovery.download(best, output_path="./data/car")

# 3. Train model
trainer = YOLOTrainer(model="yolo11m", output_dir="./runs")
result = trainer.train(
    data_yaml="./data/car/data.yaml",
    epochs=100,
)

# 4. Export for edge
exporter = ModelExporter(output_dir="./export")
export_result = exporter.export(
    model_path=result.model_path,
    platform="jetson_orin",
)

print(f"Exported to: {export_result.model_path}")
```

---

## 9. API Reference

### 9.1 Request/Response Models

See `src/api/routes.py` for complete API schemas:

- `DatasetSearchRequest` / `DatasetSearchResponse`
- `TrainRequest` / `TrainResponse` / `TrainStatusResponse`
- `ExportRequest` / `ExportResponse`

---

*Document Version: 1.0*
*Last Updated: 2026-03-14*
