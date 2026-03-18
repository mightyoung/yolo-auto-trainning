# YOLO Auto-Training System

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/mightyoung/yolo-auto-trainning/actions/workflows/ci-cd.yml/badge.svg)
![Stage](https://img.shields.io/badge/Stage-Beta-orange)

An AI-driven end-to-end YOLO model training and deployment platform. From dataset discovery to edge inference — fully automated with CrewAI multi-agent orchestration, Ray Tune hyperparameter optimization, and one-click export to NVIDIA Jetson / Rockchip RK3588.

---

## Key Features

| Feature | Description |
|---|---|
| **Dataset Discovery** | Multi-source search across Roboflow, Kaggle, and HuggingFace with relevance scoring |
| **Auto Training** | YOLO11 training with Ray Tune HPO, MLflow experiment tracking |
| **Knowledge Distillation** | Train compact student models from large teacher models |
| **Edge Deployment** | One-click export to ONNX / TensorRT for Jetson Nano, Jetson Orin, RK3588 |
| **Multi-Agent** | CrewAI orchestration — Data Discovery, Generation, Training, Deployment agents |
| **Async Pipeline** | Celery + Redis task queue for GPU-intensive background jobs |
| **MLOps Observability** | Prometheus metrics, Grafana dashboards, structured logging (ELK stack) |
| **Auto Labeling** | Semi-automated annotation pipeline using Grounded SAM |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         Web UI (Next.js)                     │
│               Discovery │ Training │ Labeling │ Analysis     │
└──────────────────────────┬─────────────────────────────────┘
                           │ HTTP / REST
┌──────────────────────────▼─────────────────────────────────┐
│              Business API  (FastAPI, port 8000)            │
│  Auth │ Dataset Discovery │ Agent Orchestration │ Routing    │
└──────────┬──────────────────────────────────────────────────┘
           │ Internal HTTP
┌──────────▼──────────────────────────────────────────────────┐
│              Training API  (FastAPI, port 8001)             │
│     YOLO Training │ Ray Tune HPO │ Model Export │ MLflow    │
└──────────┬──────────────────────────────────────────────────┘
           │
    ┌──────▼──────┐     ┌─────────────┐     ┌─────────────┐
    │  GPU Server  │     │ Redis 7     │     │ MLflow      │
    │  (CUDA 12.1)│     │ (Broker)    │     │ (Tracking)  │
    └─────────────┘     └─────────────┘     └─────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- (GPU training) NVIDIA GPU with CUDA 12.1 support

### 1. Clone & Configure

```bash
git clone https://github.com/mightyoung/yolo-auto-trainning.git
cd yolo-auto-training
cp .env.example .env
# Edit .env with your API keys (Roboflow, Kaggle, HuggingFace, etc.)
```

### 2. Start Services (Docker Compose)

```bash
# All-in-one: Redis + Business API + Celery worker + GPU training
docker-compose up -d --build

# With full MLOps stack (Prometheus + Grafana + ELK)
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

### 3. Access

| Service | URL |
|---|---|
| Business API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Training API | http://localhost:8001 |
| Training API Docs | http://localhost:8001/docs |
| Grafana | http://localhost:3000 |
| Kibana | http://localhost:5601 |

---

## Development Setup

### Python Environment

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v
```

### Start APIs Manually

```bash
# Business API
uvicorn business-api.src.api.gateway:app --host 0.0.0.0 --port 8000 --reload

# Training API (GPU node)
uvicorn training-api.src.api.gateway:app --host 0.0.0.0 --port 8001 --reload

# Celery worker
celery -A business-api.src.api.tasks worker --loglevel=info
```

---

## API Reference

### Dataset Discovery

```bash
# Search datasets across Roboflow, Kaggle, HuggingFace
curl -X POST http://localhost:8000/api/v1/data/search \
  -H "Content-Type: application/json" \
  -d '{"query": "vehicle detection", "max_results": 10}'
```

### Submit Training Job

```bash
# Start YOLO11 training
curl -X POST http://localhost:8000/api/v1/train/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo11m",
    "data_yaml": "/data/my_dataset.yaml",
    "epochs": 100,
    "imgsz": 640
  }'
```

### Hyperparameter Optimization

```bash
# Start Ray Tune HPO
curl -X POST http://localhost:8000/api/v1/train/hpo/start \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo11m",
    "data_yaml": "/data/my_dataset.yaml",
    "n_trials": 50,
    "epochs_per_trial": 50
  }'
```

### Export to Edge Platform

```bash
# Export for NVIDIA Jetson Orin
curl -X POST http://localhost:8000/api/v1/deploy/export \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/runs/train/exp/weights/best.pt",
    "platform": "jetson_orin",
    "imgsz": 640
  }'

# Export for Rockchip RK3588
curl -X POST http://localhost:8000/api/v1/deploy/export \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/runs/train/exp/weights/best.pt",
    "platform": "rk3588",
    "imgsz": 640
  }'
```

### Authentication

```bash
# Obtain JWT token
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your-password"}'

# Use token in requests
curl -X POST http://localhost:8000/api/v1/train/submit \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"model": "yolo11n", "data_yaml": "/data/dataset.yaml", "epochs": 50}'
```

---

## Project Structure

```
yolo-auto-training/
├── business-api/           # Business orchestration API (port 8000)
│   └── src/api/
│       ├── gateway.py      # FastAPI app + JWT auth
│       ├── routes.py       # Data, training, export, analysis endpoints
│       ├── training_client.py   # HTTP client → Training API
│       ├── agent_routes.py     # CrewAI agent endpoints
│       └── agents/
│           └── orchestration.py  # Multi-agent workflow definitions
├── training-api/           # GPU training API (port 8001)
│   └── src/
│       ├── training/
│       │   ├── runner.py   # YOLO trainer (ultralytics wrapper)
│       │   └── mlflow_tracker.py  # MLflow integration
│       └── deployment/
│           └── exporter.py # ONNX / TensorRT / TFLite export
├── web-ui-react/           # Next.js frontend
│   └── src/app/
│       ├── discovery/      # Dataset discovery UI
│       ├── training/       # Training management UI
│       ├── labeling/       # Auto-labeling UI
│       └── analysis/       # Analysis dashboard UI
├── src/                    # Monolithic core (legacy)
│   ├── api/               # FastAPI routes + Celery tasks
│   ├── agents/            # CrewAI orchestration
│   ├── data/              # Dataset discovery + quality filter
│   ├── training/          # YOLO training + Ray Tune HPO
│   ├── deployment/        # Model exporter
│   ├── inference/         # Inference engine
│   ├── monitoring/        # Drift detection
│   ├── pipeline/          # End-to-end orchestrator
│   └── features/         # Feature store
├── tests/                 # Unit & integration tests
├── docs/                  # Documentation (en/zh)
│   ├── en/               # English docs
│   └── zh/               # Chinese docs
├── docker-compose.yml     # Core stack (Redis + APIs + Celery)
├── docker-compose.monitoring.yml  # Prometheus + Grafana
├── docker-compose.logging.yml    # ELK stack
└── pyproject.toml        # Project metadata + dependencies
```

---

## Configuration

All sensitive configuration is managed via environment variables. Copy `.env.example` to `.env` and configure:

```bash
# Core
JWT_SECRET_KEY=<generate-with: python -c "import secrets; print(secrets.token_urlsafe(32))">
REDIS_URL=redis://localhost:6379/0

# Training API URL (intra-network address of GPU server)
TRAINING_API_URL=http://localhost:8001
TRAINING_API_KEY=<your-api-key>

# Dataset Sources
ROBOFLOW_API_KEY=<your-roboflow-key>
KAGGLE_USERNAME=<your-kaggle-username>
KAGGLE_KEY=<your-kaggle-key>
HF_TOKEN=<your-huggingface-token>

# AI Providers
DEEPSEEK_API_KEY=<your-deepseek-key>

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## Edge Deployment Targets

| Platform | Format | Toolkit | Max Batch Size |
|---|---|---|---|
| **Jetson Nano** | TensorRT FP16 | `tensorrt` | 8 |
| **Jetson Orin** | TensorRT FP16/INT8 | `tensorrt` | 32 |
| **RK3588** | ONNX + RKNN | `rknn` | 16 |
| **x86 Server** | ONNX | `onnx` | 64 |

---

## Monitoring & Alerting

### Prometheus Metrics

| Metric | Description |
|---|---|
| `yolo_training_jobs_total` | Total training jobs submitted |
| `yolo_training_duration_seconds` | Training job duration histogram |
| `yolo_api_requests_total` | API request counter by endpoint |
| `yolo_gpu_memory_usage` | GPU memory usage gauge |
| `yolo_export_jobs_total` | Model export job counter |

### Alert Rules

| Alert | Condition | Severity |
|---|---|---|
| `HighErrorRate` | Error rate > 5% in 5 min | warning |
| `APIDown` | Business API unreachable > 1 min | critical |
| `GPUMemoryHigh` | GPU memory > 90% for 5 min | warning |
| `TrainingJobFailed` | 3+ consecutive failures | critical |

---

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO implementation
- [CrewAI](https://github.com/crewAI/crewAI) — Multi-agent framework
- [Ray Tune](https://github.com/ray-project/ray) — Hyperparameter optimization
- [Roboflow](https://roboflow.com) — Dataset platform
- [HuggingFace](https://huggingface.co) — Model hub
