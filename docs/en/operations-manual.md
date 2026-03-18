# YOLO Auto-Training System - Operations Manual

**Version**: 1.0
**Date**: 2026-03-14

---

## 1. System Architecture

### 1.1 Overview

```
+-----------------------------------------------------------------+
|                        FastAPI Gateway                          |
|                    (Port 8000, Health Check)                  |
+----------------------------+----------------------------------+
                             |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
+---------------+    +---------------+    +---------------+
| Data Service |    |Train Service |    |Deploy Service |
| /api/v1/data|    |/api/v1/train|    |/api/v1/deploy|
+-------+------+    +-------+------+    +-------+------+
        |                      |                      |
        +----------------------+----------------------+
                             |
                             v
                    +---------------------+
                    |    Celery Task     |
                    |       Queue        |
                    +---------+----------+
                             |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
+--------------+    +--------------+    +--------------+
|   Training  |    |     HPO      |    |    Export   |
|   Worker    |    |    Worker    |    |    Worker   |
|  (GPU/CPU)  |    |   (CPU)      |    |   (CPU)     |
+--------------+    +--------------+    +--------------+
```

### 1.2 Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| API Gateway | FastAPI | REST API, Auth, Rate Limiting |
| Task Queue | Celery + Redis | Async job processing |
| Training | Ultralytics + Ray Tune | YOLO training & HPO |
| Storage | Local filesystem | Models, datasets, logs |
| Auth | JWT + Redis | API Key management |

---

## 2. Deployment

### 2.1 Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
docker-compose logs -f celery-worker
```

### 2.2 Manual Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis
redis-server

# Start API (Terminal 1)
uvicorn src.api.gateway:app --host 0.0.0.0 --port 8000

# Start Celery worker (Terminal 2)
celery -A src.api.tasks worker --loglevel=info --concurrency=2

# Start Celery beat (Terminal 3)
celery -A src.api.tasks beat --loglevel=info
```

---

## 3. Configuration

### 3.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | redis://localhost:6379/0 | Redis connection |
| `CELERY_BROKER_URL` | redis://localhost:6379/0 | Celery broker |
| `CELERY_RESULT_BACKEND` | redis://localhost:6379/0 | Result backend |
| `JWT_SECRET_KEY` | auto-generated | JWT signing key |
| `ALLOWED_ORIGINS` | localhost:3000,localhost:8080 | CORS origins |
| `ROBOFLOW_API_KEY` | - | Roboflow API key |
| `KAGGLE_USERNAME` | - | Kaggle username |
| `KAGGLE_KEY` | - | Kaggle API key |

### 3.2 Rate Limiting

| Endpoint | Limit |
|----------|-------|
| `/api/v1/train/*` | 10 req/min |
| `/api/v1/deploy/*` | 10 req/min |
| `/api/v1/data/*` | 60 req/min |

### 3.3 Training Resources

| Resource | Default | Description |
|----------|---------|-------------|
| Worker Memory | 8GB | Max memory per worker |
| Worker CPU | 4 cores | CPU allocation |
| Training Timeout | 10 hours | Max training time |
| HPO Trials | 50 | Max HPO trials |

---

## 4. Monitoring

### 4.1 Health Checks

```bash
# API health
curl http://localhost:8000/health

# Redis health
redis-cli ping

# Celery worker
celery -A src.api.tasks inspect active
```

### 4.2 Logs

| Service | Log Location |
|---------|-------------|
| API | stdout (uvicorn) |
| Celery | stdout |
| Training | ./runs/ |

### 4.3 Metrics

```bash
# Check training progress
curl http://localhost:8000/api/v1/train/status/{task_id}

# Check export progress
curl http://localhost:8000/api/v1/deploy/export/status/{task_id}
```

---

## 5. Backup & Recovery

### 5.1 Backup

```bash
# Backup models
tar -czf models_backup.tar.gz ./runs/

# Backup datasets
tar -czf data_backup.tar.gz ./data/

# Backup Redis (if persistent)
redis-cli SAVE
```

### 5.2 Recovery

```bash
# Restore models
tar -xzf models_backup.tar.gz

# Restore data
tar -xzf data_backup.tar.gz
```

---

## 6. Security

### 6.1 Authentication

The system supports two authentication methods:

1. **API Key** (recommended for production)
   - Header: `X-API-Key: yolo_xxx`
   - Keys stored in Redis with 30-day expiry

2. **JWT Token** (recommended for web apps)
   - Header: `Authorization: Bearer <token>`
   - Access token: 30 minutes
   - Refresh token: 7 days

### 6.2 API Key Management

```bash
# Generate API key (via API)
POST /api/v1/auth/api-key

# Or generate programmatically
from src.api.gateway import generate_api_key, store_api_key_in_redis
key = generate_api_key()
store_api_key_in_redis(key, "user_id")
```

### 6.3 CORS Configuration

Edit `ALLOWED_ORIGINS` in environment:

```bash
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

---

## 7. Scaling

### 7.1 Horizontal Scaling

```bash
# Add more Celery workers
celery -A src.api.tasks worker --concurrency=4

# Or use Docker
docker-compose up --scale celery-worker=3
```

### 7.2 Vertical Scaling

| Resource | Development | Production |
|----------|-------------|------------|
| API RAM | 2GB | 4GB |
| Worker RAM | 4GB | 8GB+ |
| Worker CPU | 2 cores | 4+ cores |
| GPU | Optional | NVIDIA GPU |

### 7.3 GPU Training

```bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Or in docker-compose
environment:
  - CUDA_VISIBLE_DEVICES=0
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## 8. Maintenance

### 8.1 Log Rotation

Configure in `docker-compose.yml`:

```yaml
logging:
  options:
    max-size: "10m"
    max-file: "3"
```

### 8.2 Cleanup

```bash
# Clean old training runs
find ./runs -type d -mtime +30 -exec rm -rf {} \;

# Clean old datasets
find ./data -type d -mtime +90 -exec rm -rf {} \;

# Clean Celery results
celery -A src.api.tasks purge
```

### 8.3 Updates

```bash
# Pull latest code
git pull

# Rebuild containers
docker-compose build

# Restart services
docker-compose up -d
```

---

## 9. Troubleshooting

### 9.1 Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| API returns 401 | Invalid/missing auth | Check API key or token |
| API returns 429 | Rate limit exceeded | Wait 1 minute |
| Training hangs | Worker crashed | Check Celery logs |
| Export fails | CUDA not available | Verify GPU access |
| Slow response | Worker overloaded | Scale workers |

### 9.2 Debug Commands

```bash
# Check Celery tasks
celery -A src.api.tasks inspect active
celery -A src.api.tasks inspect scheduled

# Check Redis
redis-cli INFO
redis-cli KEYS "celery*"

# Check API logs
docker-compose logs api --tail=100

# Check worker logs
docker-compose logs celery-worker --tail=100
```

### 9.3 Emergency Recovery

```bash
# Stop all services
docker-compose down

# Clear Redis
redis-cli FLUSHALL

# Restart services
docker-compose up -d
```

---

## 10. Performance Tuning

### 10.1 Training Optimization

| Parameter | Default | Optimization |
|-----------|---------|--------------|
| Batch Size | 16 | Increase if GPU memory allows |
| Workers | 2 | Increase for parallel training |
| Cache | True | Enable for faster epochs |

### 10.2 API Optimization

| Parameter | Default | Optimization |
|-----------|---------|--------------|
| Worker Threads | 4 | Increase for high traffic |
| Keep-alive | 5s | Adjust for connection patterns |
| Timeout | 30s | Increase for long tasks |

---

## 11. Disaster Recovery

### 11.1 Failure Scenarios

| Scenario | Recovery Time | Data Loss |
|----------|---------------|-----------|
| Redis crash | < 1 min | Task queue lost |
| Worker crash | < 5 min | Current task |
| API crash | < 1 min | None |
| Storage full | N/A | Preventable |

### 11.2 DR Checklist

- [ ] Daily backups of models and data
- [ ] Redis persistence enabled (AOF)
- [ ] Health checks configured
- [ ] Alerting set up
- [ ] Runbook documented

---

## 12. Support

### 12.1 Log Locations

| Issue | Log |
|-------|-----|
| API errors | API stdout |
| Task failures | Celery worker stdout |
| Training issues | `./runs/` directory |
| System errors | System journal |

### 12.2 Getting Help

1. Check logs first
2. Review this operations manual
3. Check GitHub issues
4. Contact support with:
   - Error message
   - Steps to reproduce
   - Log snippets

---

*Document Version: 1.0*
*Last Updated: 2026-03-14*
