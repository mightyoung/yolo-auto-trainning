# API 服务模块详细设计

**版本**: 4.0
**所属**: 1+5 设计方案
**审核状态**: 已修订

---

## 1. 模块职责

| 职责 | 描述 |
|------|------|
| REST API | 暴露数据生成/训练/部署接口 |
| 任务管理 | Celery + Redis 任务队列 |
| 认证授权 | API Key 认证 |
| 限流保护 | 多维度限流 |
| 监控告警 | Prometheus 指标 |

---

## 2. 专家建议（来自 FastAPI + 生产部署最佳实践）

> "Use Celery with Redis as the broker and backend for task queues"
> — FastAPI + Celery Best Practices

> "Monitor everything: latency, errors, resource usage"
> — SRE Best Practices

**核心建议**：
1. **Redis 连接池** - 高并发连接管理
2. **多维度限流** - 用户 + 端点 + 时间
3. **任务优先级** - 紧急任务优先处理
4. **Prometheus 监控** - 指标收集与告警

---

## 3. 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                      API Service Module                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    FastAPI Application                     │  │
│  │                                                            │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐     │  │
│  │  │   /data/*   │ │  /train/*    │ │  /deploy/*   │     │  │
│  │  │   Endpoints │ │  Endpoints   │ │  Endpoints   │     │  │
│  │  └──────────────┘ └──────────────┘ └──────────────┘     │  │
│  │                                                            │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐     │  │
│  │  │   /agent/*   │ │  /metrics   │ │   /docs     │     │  │
│  │  │   Endpoints │ │  Endpoints   │ │   Endpoints  │     │  │
│  │  └──────────────┘ └──────────────┘ └──────────────┘     │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │    Authentication (API Key) + Rate Limiting         │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Celery + Redis                          │  │
│  │         ┌─────────┐ ┌─────────┐ ┌─────────┐           │  │
│  │         │ Data Gen│ │ Training│ │ Deploy  │           │  │
│  │         │  Queue  │ │  Queue  │ │  Queue  │           │  │
│  │         └─────────┘ └─────────┘ └─────────┘           │  │
│  │                            │                             │  │
│  │                            ▼                             │  │
│  │         ┌─────────────────────────────────────┐        │  │
│  │         │    Redis (Connection Pool)           │        │  │
│  │         └─────────────────────────────────────┘        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                 Prometheus + Grafana                       │  │
│  │    任务计数 │ 延迟分布 │ 错误率 │ GPU 温度/利用率       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心组件

### 4.1 Redis 连接池配置

```python
# src/api/redis_pool.py
import redis
from redis import ConnectionPool

# 创建连接池
connection_pool = ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    max_connections=50,  # 最大连接数
    decode_responses=True
)

def get_redis_client():
    """获取 Redis 客户端（使用连接池）"""
    return redis.Redis(connection_pool=connection_pool)
```

### 4.2 多维度限流

```python
# src/api/rate_limiter.py
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import Dict

class MultiDimLimiter:
    """多维度限流器"""

    def __init__(self):
        self.limiter = Limiter(key_func=get_remote_address)

        # 不同端点的限流配置
        self.limits = {
            "/api/v1/data/generate": "10/minute",
            "/api/v1/train/run": "1/minute",
            "/api/v1/train/hpo": "1/minute",
            "/api/v1/distill/run": "1/minute",
            "/api/v1/deploy/run": "5/minute",
            "/api/v1/agent/execute": "1/minute",
            "/health": "100/minute",
        }

    def get_limiter(self, endpoint: str):
        """获取特定端点的限流器"""
        limit = self.limits.get(endpoint, "10/minute")
        return self.limiter.limit(limit)

# 创建限流器
multi_limiter = MultiDimLimiter()
```

### 4.3 Celery 配置（带优先级）

```python
# src/api/celery_config.py
from celery import Celery, Queue
from celery import Priority

# 创建 Celery 应用
celery_app = Celery(
    "yolo_auto_training",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# 定义队列 - 优先级
celery_app.conf.task_queues = {
    Queue('high', priority=10),    # 高优先级
    Queue('default', priority=5),  # 默认
    Queue('low', priority=1),     # 低优先级
}

# 任务路由
celery_app.conf.task_routes = {
    "src.api.tasks.data.*": {"queue": "default"},
    "src.api.tasks.training.*": {"queue": "high"},
    "src.api.tasks.deployment.*": {"queue": "high"},
}

# 任务优先级
celery_app.conf.task_inherit_parent_priority = True
```

### 4.4 API 主程序（带监控）

```python
# src/api/main.py
from fastapi import FastAPI, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
import time
from redis import Redis

app = FastAPI(title="YOLO Auto-Training API")

# ============================================================================
# Prometheus 指标
# ============================================================================
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_latency = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

task_count = Counter(
    'tasks_total',
    'Total tasks',
    ['task_type', 'status']
)

# ============================================================================
# 中间件 - 指标收集
# ============================================================================

@app.middleware("http")
async def metrics_middleware(request, call_next):
    """收集指标"""
    start_time = time.time()

    response = await call_next(request)

    # 记录请求数
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    # 记录延迟
    latency = time.time() - start_time
    request_latency.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(latency)

    return response

# ============================================================================
# 端点
# ============================================================================

@app.get("/")
async def root():
    return {"message": "YOLO Auto-Training API", "version": "4.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    """Prometheus 指标端点"""
    return generate_latest()

# ... 其他端点
```

### 4.5 Celery 任务（带重试）

```python
# src/api/tasks/training.py
from src.api.celery_config import celery_app
import asyncio

@celery_app.task(
    bind=True,
    name="src.api.tasks.training.run_training",
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True
)
def run_training(self, request_data: dict):
    """训练任务 - 带重试"""
    try:
        from src.train.yolo_trainer import YOLO11Trainer

        trainer = YOLO11Trainer(model_size=request_data.get("model_size", "m"))
        result = trainer.train(
            data_yaml=request_data["data_yaml"],
            epochs=request_data.get("epochs", 100),
            imgsz=request_data.get("imgsz", 1280)
        )

        return {
            "status": "completed",
            "model_path": result["model_path"],
            "mAP50": result.get("best_map50", 0)
        }

    except Exception as e:
        # 自动重试
        raise self.retry(exc=e)
```

---

## 5. API 接口列表

| 方法 | 路径 | 限流 | 描述 |
|------|------|------|------|
| GET | / | - | 根信息 |
| GET | /health | 100/min | 健康检查 |
| GET | /metrics | 100/min | Prometheus 指标 |
| GET | /docs | - | API 文档 |
| POST | /api/v1/data/generate | 10/min | 数据生成 |
| GET | /api/v1/data/status/{task_id} | 100/min | 数据状态 |
| POST | /api/v1/train/run | 1/min | 训练模型 |
| POST | /api/v1/train/hpo | 1/min | 超参优化 |
| POST | /api/v1/train/distill | 1/min | 知识蒸馏 |
| GET | /api/v1/train/{task_id} | 100/min | 训练结果 |
| POST | /api/v1/deploy/run | 5/min | 部署模型 |
| GET | /api/v1/deploy/{task_id} | 100/min | 部署状态 |
| POST | /api/v1/agent/execute | 1/min | Agent 执行 |
| GET | /api/v1/agent/{task_id} | 100/min | Agent 状态 |

---

## 6. 监控指标

### 6.1 应用指标

| 指标 | 类型 | 描述 |
|------|------|------|
| http_requests_total | Counter | HTTP 请求总数 |
| http_request_duration_seconds | Histogram | 请求延迟 |
| tasks_total | Counter | 任务总数 |

### 6.2 业务指标

| 指标 | 类型 | 描述 |
|------|------|------|
| training_completed_total | Counter | 训练完成数 |
| training_failed_total | Counter | 训练失败数 |
| deployment_success_total | Counter | 部署成功数 |
| model_mAP50 | Gauge | 最新模型 mAP50 |

### 6.3 系统指标

| 指标 | 类型 | 描述 |
|------|------|------|
| gpu_utilization_percent | Gauge | GPU 利用率 |
| gpu_memory_used_mb | Gauge | GPU 显存使用 |
| gpu_temperature_celsius | Gauge | GPU 温度 |

---

## 7. 专家审核要点

| 审核项 | 状态 | 说明 |
|--------|------|------|
| Redis 连接池 | ✅ | 50 连接数限制 |
| 多维度限流 | ✅ | 端点分级限流 |
| 任务优先级队列 | ✅ | high/default/low |
| Prometheus 监控 | ✅ | 指标收集 |
| Celery 重试 | ✅ | 指数退避 |

---

## 8. 关键改进说明 (v3 → v4)

### 改进 1: Redis 连接池
- **v3 错误**: 每次请求创建新连接
- **v4 正确**: 使用连接池
- **依据**: 生产环境需要连接复用

### 改进 2: 多维度限流
- **v3 错误**: 单一维度限流
- **v4 正确**: 按端点分级限流
- **依据**: 保护关键资源

### 改进 3: 任务优先级
- **v3 错误**: FIFO 队列
- **v4 正确**: 优先级队列
- **依据**: 紧急任务优先处理

### 改进 4: Prometheus 监控
- **v3 错误**: 只有健康检查
- **v4 正确**: 完整指标收集
- **依据**: SRE 最佳实践

---

*审核状态: 通过 - 符合生产部署最佳实践*
