# API 服务模块详细设计

**版本**: 8.0
**所属**: 1+5 设计方案
**核心**: FastAPI + Redis + 任务队列 + 监控

---

## 1. 模块职责

| 职责 | 描述 |
|------|------|
| REST API | 数据发现、生成、训练、部署接口 |
| 任务队列 | Celery + Redis 异步任务 |
| 认证授权 | API Key + JWT |
| 限流保护 | 多维度限流 |
| 监控告警 | Prometheus + Grafana |

---

## 2. 专家建议

> "APIs should be predictable, consistent, and well-documented" — RESTful API Design Best Practices
> "Rate limiting protects your API from abuse and ensures fair usage" — API Design Handbook

**核心原则**：
1. **RESTful 设计** - 资源导向，标准化响应
2. **异步优先** - 长时间任务入队列
3. **安全优先** - 认证、限流、审计

---

## 3. 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                       API Service Module                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              API Gateway (FastAPI)                        │  │
│  │    /data/*  /train/*  /deploy/*  /agent/*              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│    ┌─────────────────────────┼─────────────────────────┐       │
│    ▼                         ▼                         ▼       │
│  ┌────────────┐      ┌────────────┐           ┌────────────┐  │
│  │   Auth    │      │  Rate     │           │  Logging   │  │
│  │  Middle   │      │  Limiter  │           │ Middleware │  │
│  └────────────┘      └────────────┘           └────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Task Queue (Celery + Redis)                  │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │  │
│  │  │Discover│ │Generate│ │ Train  │ │ Deploy │        │  │
│  │  │ Worker │ │ Worker │ │ Worker │ │ Worker │        │  │
│  │  └────────┘ └────────┘ └────────┘ └────────┘        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Monitoring (Prometheus + Grafana)            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心组件

### 4.1 API 网关

```python
# src/api/gateway.py
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import time

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print("Starting API Gateway...")
    yield
    # 关闭时
    print("Shutting down API Gateway...")

def create_app() -> FastAPI:
    """创建 API 应用"""

    app = FastAPI(
        title="AutoML API",
        description="AI-driven YOLO Training & Deployment API",
        version="5.0.0",
        lifespan=lifespan
    )

    # CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 日志中间件
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    return app

# 创建应用实例
app = create_app()

# 注册路由
from src.api.routes import data_router, train_router, deploy_router

app.include_router(data_router, prefix="/api/v1/data", tags=["Data"])
app.include_router(train_router, prefix="/api/v1/train", tags=["Training"])
app.include_router(deploy_router, prefix="/api/v1/deploy", tags=["Deployment"])
```

### 4.2 认证中间件

```python
# src/api/auth.py
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKey
from typing import Optional
import hashlib
import time

# API Key 头
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

class AuthManager:
    """认证管理器"""

    def __init__(self):
        self.api_keys: dict = {}  # {key: {user, expires, rate_limit}}

    def create_api_key(
        self,
        user_id: str,
        rate_limit: int = 100,
        expires_days: int = 30
    ) -> str:
        """创建 API Key"""
        # 生成随机 key
        raw_key = f"{user_id}:{time.time()}"
        api_key = hashlib.sha256(raw_key.encode()).hexdigest()[:32]

        # 存储
        self.api_keys[api_key] = {
            "user_id": user_id,
            "expires": time.time() + expires_days * 86400,
            "rate_limit": rate_limit,
            "created_at": time.time()
        }

        return api_key

    async def verify_api_key(
        self,
        api_key: Optional[str] = Security(API_KEY_HEADER)
    ) -> dict:
        """验证 API Key"""
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API Key"
            )

        if api_key not in self.api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key"
            )

        key_data = self.api_keys[api_key]

        # 检查过期
        if key_data["expires"] < time.time():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API Key expired"
            )

        return key_data

# 全局实例
auth_manager = AuthManager()
```

### 4.3 限流中间件

```python
# src/api/rate_limiter.py
"""
分布式限流器 - 基于 Redis + Lua 脚本

基于 FastAPI 生产环境最佳实践:
- 使用 Token Bucket 算法
- Redis + Lua 确保原子性
- 支持分布式环境

参考: https://python.plainenglish.io/building-a-production-ready-distributed-rate-limiter-with-fastapi-redis-and-lua-a20816198f86
"""
from fastapi import HTTPException, Request, status
from typing import Dict
import time
import redis
import os

class DistributedRateLimiter:
    """分布式限流器 - 使用 Redis + Lua"""

    # Token Bucket Lua 脚本 - 原子操作
    LUA_SCRIPT = """
    local key = KEYS[1]
    local limit = tonumber(ARGV[1])
    local window = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])

    local current = redis.call('GET', key)
    if current and tonumber(current) >= limit then
        return 0
    end

    redis.call('ZREMRANGEBYSCORE', key, 0, now - window * 1000)
    local count = redis.call('ZCARD', key)
    if count >= limit then
        return 0
    end

    redis.call('ZADD', key, now, now .. math.random())
    redis.call('EXPIRE', key, window)
    return 1
    """

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis = redis.from_url(self.redis_url)
        self.script = self.redis.register_script(self.LUA_SCRIPT)

    async def check_rate_limit(
        self,
        key: str,
        limit: int = 100,
        window: int = 60
    ) -> bool:
        """
        检查限流 - 原子操作

        Args:
            key: 限流 key (user_id / ip)
            limit: 窗口内最大请求数
            window: 窗口大小 (秒)

        Returns:
            True: 允许请求
            False: 超出限制
        """
        now_ms = int(time.time() * 1000)

        result = self.script(
            keys=[f"rate_limit:{key}"],
            args=[limit, window, now_ms]
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded. Max {limit} requests per {window}s",
                    "retry_after": window
                }
            )

        return True

# 全局实例
rate_limiter = DistributedRateLimiter()

# 限流配置 - 基于 API 等级
RATE_LIMITS = {
    "default": {"limit": 100, "window": 60},
    "premium": {"limit": 500, "window": 60},
    "train": {"limit": 10, "window": 3600},  # 训练任务限制
    "deploy": {"limit": 10, "window": 3600},
}
```

### 4.4 任务队列

```python
# src/api/tasks.py
from celery import Celery
from typing import Optional, Dict
import json
from pathlib import Path

# Celery 配置 - 基于生产环境最佳实践
CELERY_BROKER_URL = "redis://localhost:6379/0"
CELERY_RESULT_BACKEND = "redis://localhost:6379/1"

celery_app = Celery(
    "automl_tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

# Celery 配置 - 完整生产环境配置
from kombu import Queue, Exchange

# 定义队列和优先级
task_queues = [
    Queue('high_priority', Exchange('high'), routing_key='high', priority=1),
    Queue('data_discovery', Exchange('medium'), routing_key='data', priority=3),
    Queue('training', Exchange('medium'), routing_key='train', priority=5),
    Queue('deployment', Exchange('low'), routing_key='deploy', priority=7),
    Queue('low_priority', Exchange('low'), routing_key='low', priority=10),
    Queue('dead_letter', Exchange('dlx'), routing_key='dead'),  # 死信队列
]

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1小时超时
    task_soft_time_limit=3300,

    # 队列配置
    task_queues=task_queues,
    task_routes={
        'data.discover': {'queue': 'data_discovery', 'priority': 3},
        'data.generate': {'queue': 'data_discovery', 'priority': 3},
        'train.run': {'queue': 'training', 'priority': 5},
        'train.hpo': {'queue': 'high_priority', 'priority': 2},
        'deploy.run': {'queue': 'deployment', 'priority': 7},
    },

    # 死信队列配置
    task_dead_letter_exchange='dlx',
    task_dead_letter_routing_key='dead',

    # 任务确认
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

@celery_app.task(bind=True, name="data.discover")
def discover_datasets(
    self,
    task_description: str,
    min_images: int = 1000,
    sources: list = None
) -> Dict:
    """数据集发现任务"""
    from src.data_discovery import DatasetDiscovery

    discovery = DatasetDiscovery()
    result = discovery.search(task_description, min_images, sources or ["roboflow", "kaggle"])

    return {
        "task_id": self.request.id,
        "status": "completed",
        "result": result
    }

@celery_app.task(bind=True, name="data.generate")
def generate_data(
    self,
    task_description: str,
    num_images: int = 100,
    workflow_type: str = "basic"
) -> Dict:
    """数据生成任务"""
    from src.comfy_workflow import WorkflowGenerator

    generator = WorkflowGenerator()
    result = generator.generate(task_description, num_images, workflow_type)

    return {
        "task_id": self.request.id,
        "status": "completed",
        "result": result
    }

@celery_app.task(bind=True, name="train.run")
def train_model(
    self,
    data_yaml: str,
    model_size: str = "yolo11m",
    epochs: int = 100,
    enable_hpo: bool = False
) -> Dict:
    """训练任务"""
    from src.training import TrainingPipeline

    pipeline = TrainingPipeline()
    result = pipeline.run(
        data_yaml=data_yaml,
        model_size=model_size,
        epochs=epochs,
        enable_hpo=enable_hpo
    )

    return {
        "task_id": self.request.id,
        "status": "completed",
        "result": result
    }

@celery_app.task(bind=True, name="deploy.run")
def deploy_model(
    self,
    model_path: str,
    device_ip: str,
    device_type: str = "jetson-nano"
) -> Dict:
    """部署任务"""
    from src.deployment import EdgeDeployer

    deployer = EdgeDeployer(host=device_ip)
    result = deployer.deploy(model_path, device_type)

    return {
        "task_id": self.request.id,
        "status": "completed",
        "result": result
    }
```

### 4.5 API 路由

```python
# src/api/routes/data_routes.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from celery.result import AsyncResult

from src.api.auth import auth_manager, verify_api_key
from src.api.rate_limiter import rate_limiter
from src.api.tasks import (
    discover_datasets,
    generate_data,
    train_model,
    deploy_model,
    celery_app
)

# 路由
data_router = APIRouter()
train_router = APIRouter()
deploy_router = APIRouter()

# ========== 数据发现 API ==========

class DiscoverRequest(BaseModel):
    """数据集发现请求"""
    task_description: str
    min_images: int = 1000
    sources: List[str] = ["roboflow", "kaggle", "huggingface"]

class DiscoverResponse(BaseModel):
    """数据集发现响应"""
    task_id: str
    status: str
    status_url: str

@data_router.post("/discover", response_model=DiscoverResponse)
async def discover_datasets_endpoint(
    request: DiscoverRequest,
    user: dict = Depends(verify_api_key)
):
    """数据集发现"""
    # 限流检查
    await rate_limiter.check_rate_limit(
        None, user["user_id"],
        RATE_LIMITS["default"]["limit"]
    )

    # 提交任务
    task = discover_datasets.apply_async(
        args=[request.task_description, request.min_images, request.sources]
    )

    return DiscoverResponse(
        task_id=task.id,
        status="pending",
        status_url=f"/api/v1/tasks/{task.id}"
    )

@data_router.get("/discover/{task_id}")
async def get_discover_result(task_id: str):
    """获取发现结果"""
    result = AsyncResult(task_id, app=celery_app)

    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None
    }

# ========== 数据生成 API ==========

class GenerateRequest(BaseModel):
    """数据生成请求"""
    task_description: str
    num_images: int = 100
    workflow_type: str = "basic"
    width: int = 1024
    height: int = 1024

@data_router.post("/generate", response_model=DiscoverResponse)
async def generate_data_endpoint(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(verify_api_key)
):
    """数据生成"""
    task = generate_data.apply_async(
        args=[request.task_description, request.num_images, request.workflow_type]
    )

    return DiscoverResponse(
        task_id=task.id,
        status="pending",
        status_url=f"/api/v1/tasks/{task.id}"
    )

# ========== 训练 API ==========

class TrainRequest(BaseModel):
    """训练请求"""
    data_yaml: str
    model_size: str = "yolo11m"
    epochs: int = 100
    enable_hpo: bool = False
    enable_distillation: bool = False
    target_platform: str = "jetson"

@train_router.post("/run", response_model=DiscoverResponse)
async def train_model_endpoint(
    request: TrainRequest,
    user: dict = Depends(verify_api_key)
):
    """模型训练"""
    # 限流检查
    await rate_limiter.check_rate_limit(
        None, user["user_id"],
        RATE_LIMITS["train"]["limit"]
    )

    task = train_model.apply_async(
        args=[
            request.data_yaml,
            request.model_size,
            request.epochs,
            request.enable_hpo
        ]
    )

    return DiscoverResponse(
        task_id=task.id,
        status="pending",
        status_url=f"/api/v1/tasks/{task.id}"
    )

@train_router.get("/status/{task_id}")
async def get_training_status(task_id: str):
    """训练状态"""
    result = AsyncResult(task_id, app=celery_app)

    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
        "info": result.info if result.info else None
    }

# ========== 部署 API ==========

class DeployRequest(BaseModel):
    """部署请求"""
    model_path: str
    device_ip: str
    device_type: str = "jetson-nano"
    device_username: str = "nvidia"
    device_password: str = None
    precision: str = "fp16"

@deploy_router.post("/run", response_model=DiscoverResponse)
async def deploy_model_endpoint(
    request: DeployRequest,
    user: dict = Depends(verify_api_key)
):
    """模型部署"""
    task = deploy_model.apply_async(
        args=[
            request.model_path,
            request.device_ip,
            request.device_type
        ]
    )

    return DiscoverResponse(
        task_id=task.id,
        status="pending",
        status_url=f"/api/v1/tasks/{task.id}"
    )

@deploy_router.get("/status/{task_id}")
async def get_deployment_status(task_id: str):
    """部署状态"""
    result = AsyncResult(task_id, app=celery_app)

    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None
    }
```

### 4.6 监控指标

```python
# src/api/monitoring.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import APIRouter, Response
import time

# 创建指标
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "endpoint"]
)

ACTIVE_TASKS = Gauge(
    "celery_active_tasks",
    "Number of active Celery tasks",
    ["task_name"]
)

TASK_DURATION = Histogram(
    "celery_task_duration_seconds",
    "Celery task duration in seconds",
    ["task_name"]
)

# 监控路由
monitoring_router = APIRouter()

@monitoring_router.get("/metrics")
async def metrics():
    """Prometheus 指标"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

@monitoring_router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "version": "5.0.0",
        "timestamp": time.time()
    }
```

---

## 5. API 端点汇总

### 5.1 数据接口

| 方法 | 端点 | 描述 | 认证 |
|------|------|------|------|
| POST | /api/v1/data/discover | 数据集发现 | API Key |
| GET | /api/v1/data/discover/{task_id} | 获取发现结果 | API Key |
| POST | /api/v1/data/generate | 生成合成数据 | API Key |
| GET | /api/v1/data/generate/{task_id} | 获取生成结果 | API Key |

### 5.2 训练接口

| 方法 | 端点 | 描述 | 认证 |
|------|------|------|------|
| POST | /api/v1/train/run | 启动训练 | API Key |
| GET | /api/v1/train/status/{task_id} | 训练状态 | API Key |
| GET | /api/v1/train/metrics/{task_id} | 训练指标 | API Key |
| POST | /api/v1/train/cancel | 取消训练 | API Key |

### 5.3 部署接口

| 方法 | 端点 | 描述 | 认证 |
|------|------|------|------|
| POST | /api/v1/deploy/run | 启动部署 | API Key |
| GET | /api/v1/deploy/status/{task_id} | 部署状态 | API Key |
| GET | /api/v1/deploy/health/{device_ip} | 设备健康 | API Key |

### 5.4 系统接口

| 方法 | 端点 | 描述 | 认证 |
|------|------|------|------|
| GET | /api/v1/metrics | Prometheus 指标 | 公开 |
| GET | /api/v1/health | 健康检查 | 公开 |
| POST | /api/v1/auth/key | 创建 API Key | Basic |

---

## 6. 响应格式

### 6.1 成功响应

```python
{
    "success": true,
    "task_id": "abc123",
    "status_url": "/api/v1/tasks/abc123",
    "message": "Task started successfully"
}
```

### 6.2 错误响应

```python
{
    "success": false,
    "error": {
        "code": "TASK_NOT_FOUND",
        "message": "Task with ID abc123 not found",
        "details": {}
    }
}
```

### 6.3 任务结果

```python
{
    "task_id": "abc123",
    "status": "completed",
    "result": {
        "datasets": [...],
        "model_path": "./runs/train/weights/best.pt",
        "mAP50": 0.78
    }
}
```

---

## 7. 专家审核要点

| 审核项 | 状态 | 说明 |
|--------|------|------|
| RESTful 设计 | ✅ | 资源导向 |
| 认证授权 | ✅ | API Key |
| 限流保护 | ✅ | 多维度 |
| 异步任务 | ✅ | Celery + Redis |
| 监控指标 | ✅ | Prometheus |

---

## 8. 依赖

```python
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "celery>=5.3.0",
    "redis>=5.0.0",
    "python-jose[cryptography]>=3.3.0",
    "prometheus-client>=0.19.0",
    "httpx>=0.26.0",
    "pydantic>=2.5.0",
]
```

---

## 9. 与其他模块的集成

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────►│  API Gateway │────►│  Celery     │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    ▼                          ▼                          ▼
             ┌────────────┐            ┌────────────┐            ┌────────────┐
             │Discovery   │            │  Training   │            │ Deployment │
             │  Worker    │            │   Worker    │            │   Worker   │
             └────────────┘            └────────────┘            └────────────┘
```

---

*文档版本: 8.0*
*核心功能: FastAPI + Celery + Redis + 监控*
