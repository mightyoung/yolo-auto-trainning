# API 服务模块详细设计

**版本**: 6.0
**所属**: 1+5 设计方案
**核心**: FastAPI + Redis + 任务队列 + 监控
**更新**: 基于官方最佳实践修正限流、CORS 和安全问题

---

## 1. 模块职责

| 职责 | 描述 |
|------|------|
| REST API | 数据发现、生成、训练、部署接口 |
| 任务队列 | Celery + Redis 异步任务 |
| 认证授权 | API Key + JWT (Redis 持久化) |
| 限流保护 | 多维度限流 (修正 Lua 脚本) |
| 监控告警 | Prometheus + Grafana |

---

## 2. 专家建议

> "APIs should be predictable, consistent, and well-documented" — RESTful API Design Best Practices
> "Rate limiting protects your API from abuse and ensures fair usage" — API Design Handbook
> "Never use allow_origins=['*'] with credentials in production" — FastAPI Security

**核心原则**：
1. **RESTful 设计** - 资源导向，标准化响应
2. **异步优先** - 长时间任务入队列
3. **安全优先** - 认证、限流、审计、生产环境 CORS 配置

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
│  │ Middle   │      │  Limiter  │           │ Middleware │  │
│  │ (Redis)  │      │  (修正)   │           │            │  │
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
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting API Gateway...")
    yield
    print("Shutting down API Gateway...")

def create_app() -> FastAPI:
    app = FastAPI(
        title="AutoML API",
        description="AI-driven YOLO Training & Deployment API",
        version="6.0.0",
        lifespan=lifespan
    )

    # CORS 配置 - 生产环境安全设置
    # ⚠️ v6 修正: 不再使用 allow_origins=["*"]
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
    if not allowed_origins or allowed_origins == [""]:
        allowed_origins = ["http://localhost:3000", "http://localhost:8080"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,  # 生产环境指定域名
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    return app

app = create_app()

from src.api.routes import data_router, train_router, deploy_router
app.include_router(data_router, prefix="/api/v1/data", tags=["Data"])
app.include_router(train_router, prefix="/api/v1/train", tags=["Training"])
app.include_router(deploy_router, prefix="/api/v1/deploy", tags=["Deployment"])
```

### 4.2 认证中间件（Redis 持久化）

```python
# src/api/auth.py
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from typing import Optional
import redis
import json
import os

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

class AuthManager:
    """认证管理器 - 使用 Redis 持久化存储"""

    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis = redis.from_url(redis_url)

    def create_api_key(
        self,
        user_id: str,
        rate_limit: int = 100,
        expires_days: int = 30
    ) -> str:
        """创建 API Key - 存储到 Redis"""
        import hashlib
        import time

        raw_key = f"{user_id}:{time.time()}"
        api_key = hashlib.sha256(raw_key.encode()).hexdigest()[:32]

        key_data = {
            "user_id": user_id,
            "expires": time.time() + expires_days * 86400,
            "rate_limit": rate_limit,
            "created_at": time.time()
        }

        # 存储到 Redis
        self.redis.setex(
            f"api_key:{api_key}",
            expires_days * 86400,
            json.dumps(key_data)
        )

        return api_key

    async def verify_api_key(
        self,
        api_key: Optional[str] = Security(API_KEY_HEADER)
    ) -> dict:
        """验证 API Key - 从 Redis 读取"""
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API Key"
            )

        key_data = self.redis.get(f"api_key:{api_key}")

        if not key_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key"
            )

        data = json.loads(key_data)

        if data["expires"] < time.time():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API Key expired"
            )

        return data

auth_manager = AuthManager()
```

### 4.3 限流中间件（修正后的 Lua 脚本）

```python
# src/api/rate_limiter.py
"""
分布式限流器 - 基于 Redis + Lua 脚本 (修正版)

v6 修正:
- 修复了原版 Lua 脚本的逻辑错误
- 使用正确的滑动窗口算法
"""
from fastapi import HTTPException, Request, status
import time
import redis
import os

class DistributedRateLimiter:
    """分布式限流器 - 修正后的滑动窗口实现"""

    # 修正后的滑动窗口 Lua 脚本
    LUA_SCRIPT = """
    local key = KEYS[1]
    local limit = tonumber(ARGV[1])
    local window = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])

    -- 删除窗口外的请求记录
    redis.call('ZREMRANGEBYSCORE', key, 0, now - window * 1000)

    -- 统计当前窗口内的请求数
    local count = redis.call('ZCARD', key)

    -- 如果超过限制，拒绝请求
    if count >= limit then
        return 0
    end

    -- 添加新请求记录
    redis.call('ZADD', key, now, now .. math.random())
    redis.call('EXPIRE', key, window)

    return 1
    """

    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis = redis.from_url(redis_url)
        self.script = self.redis.register_script(self.LUA_SCRIPT)

    async def check_rate_limit(
        self,
        key: str,
        limit: int = 100,
        window: int = 60
    ) -> bool:
        """检查限流 - 滑动窗口算法"""
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

rate_limiter = DistributedRateLimiter()

# 限流配置
RATE_LIMITS = {
    "default": {"limit": 100, "window": 60},
    "premium": {"limit": 500, "window": 60},
    "train": {"limit": 10, "window": 3600},
    "deploy": {"limit": 10, "window": 3600},
}
```

### 4.4 任务队列

```python
# src/api/tasks.py
from celery import Celery
from typing import Dict
import os

# Celery 配置
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery(
    "automl_tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

from kombu import Queue, Exchange

# 队列和优先级配置
task_queues = [
    Queue('high_priority', Exchange('high'), routing_key='high', priority=1),
    Queue('data_discovery', Exchange('medium'), routing_key='data', priority=3),
    Queue('training', Exchange('medium'), routing_key='train', priority=5),
    Queue('deployment', Exchange('low'), routing_key='deploy', priority=7),
    Queue('low_priority', Exchange('low'), routing_key='low', priority=10),
    Queue('dead_letter', Exchange('dlx'), routing_key='dead'),
]

# v6 修正: 增加任务超时配置
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,

    # 任务超时配置 - v6 修正
    task_time_limit=7200,       # 2小时 (原 3600 不足)
    task_soft_time_limit=6600,

    # 队列配置
    task_queues=task_queues,
    task_routes={
        'data.discover': {'queue': 'data_discovery', 'priority': 3},
        'data.generate': {'queue': 'data_discovery', 'priority': 3},
        'train.run': {'queue': 'training', 'priority': 5},
        'train.hpo': {'queue': 'high_priority', 'priority': 2},
        'deploy.run': {'queue': 'deployment', 'priority': 7},
    },

    # 死信队列
    task_dead_letter_exchange='dlx',
    task_dead_letter_routing_key='dead',

    # 任务确认
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)
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

## 6. 安全配置对比

| 配置项 | v5 问题 | v6 修正 |
|--------|----------|----------|
| CORS | `allow_origins=["*"]` | 环境变量指定域名 |
| API Key 存储 | 内存字典，重启丢失 | Redis 持久化 |
| 限流 Lua 脚本 | 算法错误 | 修正为正确滑动窗口 |
| 任务超时 | 1小时不足 | 调整为 2 小时 |

---

## 7. 依赖

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

*文档版本: 6.0*
*核心功能: FastAPI + Celery + Redis + 监控*
*更新: 修正限流 Lua 脚本、CORS 配置、API Key 存储*
