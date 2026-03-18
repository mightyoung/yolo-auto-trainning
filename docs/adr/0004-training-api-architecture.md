# ADR 0004: Training API Architecture

## Status
Accepted

## Context
We need to separate the training workload from the business logic API. Training is GPU-intensive and should not block API responses. The system should support multiple training jobs and async task processing.

## Decision
Use a two-API architecture with internal HTTP communication:

### Architecture Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    Business API (port 8000)                 │
│  - Auth (JWT)                                              │
│  - Data Discovery (Roboflow, Kaggle, HuggingFace)          │
│  - CrewAI Agents                                           │
│  - Task Scheduling → Training API                          │
└─────────────────────────┬───────────────────────────────────┘
                        │ HTTP + API Key
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Training API (port 8001)                 │
│  - YOLO Training (GPU)                                     │
│  - Ray Tune HPO                                            │
│  - Model Export                                            │
│  - Internal endpoints only (X-API-Key auth)                 │
└─────────────────────────────────────────────────────────────┘
```

### Service Communication
- **Protocol**: HTTP REST
- **Authentication**: `X-API-Key` header with `TRAINING_API_KEY`
- **Client**: `TrainingAPIClient` class in Business API

### Task Processing
- **Queue**: Redis for task queuing (future: Celery integration)
- **Async**: Training jobs return immediately with job ID
- **Status**: Polling endpoint for job status updates

### Containerization
- Business API: Runs on CPU server/local machine
- Training API: Runs on GPU server with CUDA
- Both can run in Docker Compose with NVIDIA runtime

## Gaps (NOT YET IMPLEMENTED)

### Current State
- Training API exists but async job queue not fully wired
- Celery worker configured in docker-compose but not used for training
- No job persistence (in-memory job status)

### What Needs to Be Done
1. **Celery Integration**: Wire up Celery tasks for async training
2. **Job Persistence**: Store job status in Redis or database
3. **Webhooks**: Add callbacks for job completion
4. **Queue Management**: Priority queues, job cancellation
5. **GPU Scheduling**: Handle multiple GPU training jobs

## Consequences

### Easier
- Clear separation of concerns
- Independent scaling of APIs
- GPU resources isolated from API server
- Can deploy APIs on different machines

### More Difficult
- Network latency between services
- Need to manage two API deployments
- Internal communication security
- Distributed debugging across services
