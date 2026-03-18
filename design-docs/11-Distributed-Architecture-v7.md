# YOLO自动训练系统 - 分布式架构设计方案

**版本**: 7.0
**日期**: 2026-03-14
**状态**: 基于分布式架构的最佳实践设计
**基于**: 方案A1 (双API模式)

---

## 一、设计目标

| 目标 | 描述 |
|------|------|
| **GPU资源优化** | 训练任务在GPU服务器，业务逻辑在终端 |
| **低延迟响应** | 业务API快速响应，不被训练阻塞 |
| **高可用性** | 单点故障不影响整体服务 |
| **易于扩展** | 可独立扩展训练或业务服务 |

---

## 二、系统架构

### 2.1 整体拓扑

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              终端/本地环境                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        业务API (Business API)                          │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐   │ │
│  │  │  数据发现服务  │  │  Agent编排服务  │  │   任务调度服务       │   │ │
│  │  │ DataDiscovery  │  │    CrewAI      │  │   TaskScheduler      │   │ │
│  │  │  - Roboflow   │  │  - Discovery   │  │   - 任务管理        │   │ │
│  │  │  - Kaggle     │  │  - Generator   │  │   - 状态追踪        │   │ │
│  │  │  - HuggingFace│  │  - Training    │  │   - 结果聚合        │   │ │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘   │ │
│  │                              │                                          │ │
│  │                              ▼                                          │ │
│  │  ┌────────────────────────────────────────────────────────────────┐   │ │
│  │  │              本地Redis (状态缓存 + 会话管理)                     │   │ │
│  │  └────────────────────────────────────────────────────────────────┘   │ │
│  │                              │                                          │ │
│  │                              │ HTTP/REST                               │ │
│  │                              ▼                                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                         │                                    │
│                                    网络通信 (LAN)                            │
│                                         │                                    │
└─────────────────────────────────────────┼────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           GPU服务器 (192.168.11.2)                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                       训练API (Training API)                           │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐   │ │
│  │  │   YOLO训练   │  │   超参数优化   │  │     模型导出         │   │ │
│  │  │   Service    │  │   Ray Tune     │  │   Exporter Service   │   │ │
│  │  │              │  │                │  │   - ONNX            │   │ │
│  │  │  - 完整性检查│  │  - ASHA       │  │   - TensorRT        │   │ │
│  │  │  - 正式训练  │  │  - 50 trials  │  │   - FP16优化        │   │ │
│  │  │  - 知识蒸馏  │  │                │  │                     │   │ │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘   │ │
│  │                              │                                          │ │
│  │                              ▼                                          │ │
│  │  ┌────────────────────────────────────────────────────────────────┐   │ │
│  │  │                    GPU (NVIDIA L20)                             │   │ │
│  │  │              CUDA计算 + 显存管理                                │   │ │
│  │  └────────────────────────────────────────────────────────────────┘   │ │
│  │                              │                                          │ │
│  │                              ▼                                          │ │
│  │  ┌────────────────────────────────────────────────────────────────┐   │ │
│  │  │           模型存储 (/runs, /models)                           │   │ │
│  │  │     - 训练权重 (.pt)  - 导出模型 (.onnx, .engine)           │   │ │
│  │  └────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、模块职责划分

### 3.1 业务API (终端/本地)

| 模块 | 功能 | GPU需求 | 依赖 |
|------|------|---------|------|
| **DataDiscoveryService** | 数据集搜索、相关性评分 | 无 | Roboflow API, Kaggle API, HuggingFace API |
| **AgentOrchestrationService** | CrewAI Agent工作流 | 无 | LLM API (DeepSeek) |
| **DataGenerationService** | ComfyUI工作流生成、VLM标注 | 可选 | Stability AI, ComfyUI |
| **TaskScheduler** | 任务分发、状态追踪、结果聚合 | 无 | 本地Redis |
| **WebUI** | 用户界面（可选） | 无 | - |

### 3.2 训练API (GPU服务器)

| 模块 | 功能 | GPU需求 | 依赖 |
|------|------|---------|------|
| **TrainingService** | YOLO11模型训练 | 强GPU | CUDA, cuDNN |
| **HPOService** | Ray Tune超参数优化 | 强GPU | Ray, Tune |
| **ExportService** | 模型导出、TensorRT优化 | 强GPU | TensorRT, ONNX |
| **ModelStorage** | 模型版本管理 | 中等 | 本地存储/S3 |

---

## 四、API接口设计

### 4.1 业务API接口

#### 4.1.1 数据发现

```python
# 端点: POST /api/v1/data/search
# 功能: 搜索数据集
# 位置: 业务API (终端)

Request:
{
    "query": "汽车检测",           # 搜索关键词
    "max_results": 10,             # 最大返回数量
    "sources": ["roboflow", "kaggle", "huggingface"],  # 数据源筛选
    "min_images": 100,            # 最小图片数量
    "license": "MIT"              # 许可证筛选
}

Response:
{
    "datasets": [
        {
            "id": "roboflow/car-detection",
            "name": "car-detection",
            "source": "roboflow",
            "url": "https://universe.roboflow.com/...",
            "images": 1500,
            "license": "MIT",
            "relevance_score": 0.95,
            "categories": ["car", "vehicle"]
        }
    ],
    "total": 1,
    "query_time_ms": 1250
}
```

#### 4.1.2 任务提交

```python
# 端点: POST /api/v1/train/submit
# 功能: 提交训练任务到GPU服务器
# 位置: 业务API (终端)

Request:
{
    "task_type": "training",           # "training" | "hpo" | "export"
    "config": {
        "model": "yolo11m",           # 模型大小
        "data_yaml": "/data/car.yaml", # 数据集路径
        "epochs": 100,                 # 训练轮数
        "imgsz": 640,                 # 输入尺寸
        "batch": 16,                  # 批大小
        "device": "cuda:0"            # 设备
    },
    "callback_url": "http://local:8000/api/v1/train/callback"  # 回调地址
}

Response:
{
    "task_id": "train_abc123",
    "status": "submitted",
    "submitted_at": "2026-03-14T10:00:00Z",
    "gpu_server": "192.168.11.2:8001",
    "estimated_time_minutes": 45
}
```

#### 4.1.3 任务状态查询

```python
# 端点: GET /api/v1/train/status/{task_id}
# 功能: 查询训练任务状态

Response:
{
    "task_id": "train_abc123",
    "status": "running",           # submitted | running | completed | failed
    "progress": 0.65,             # 进度 0-1
    "current_epoch": 65,
    "total_epochs": 100,
    "metrics": {
        "mAP50": 0.72,
        "mAP50-95": 0.48,
        "loss": 0.23
    },
    "logs": "epoch 65/100, mAP50=0.72...",
    "started_at": "2026-03-14T10:00:00Z",
    "updated_at": "2026-03-14T10:45:00Z"
}
```

### 4.2 训练API接口 (GPU服务器)

```python
# 端点: POST /api/v1/internal/train/start
# 功能: 内部训练启动 (仅业务API可调用)
# 认证: API Key

Request:
{
    "task_id": "train_abc123",
    "model": "yolo11m",
    "data_yaml": "s3://bucket/data/car.yaml",
    "epochs": 100,
    "imgsz": 640,
    "output_dir": "/runs/train_abc123"
}

Response:
{
    "task_id": "train_abc123",
    "status": "started",
    "worker_id": "worker_001"
}
```

---

## 五、通信协议设计

### 5.1 REST API通信

```
┌──────────────┐     HTTP/JSON      ┌──────────────┐
│  业务API     │ ◄────────────────► │  训练API     │
│  (终端)      │                    │ (GPU服务器)   │
└──────────────┘                    └──────────────┘
       │                                    │
       │ GET /api/v1/train/submit          │
       │ ─────────────────────────────────► │
       │                                    │
       │ {"task_id": "xxx", "status": ...} │
       │ ◄──────────────────────────────── │
       │                                    │
       │ GET /api/v1/train/status/xxx      │
       │ ─────────────────────────────────► │
       │                                    │
       │ {"status": "running", "progress":..}│
       │ ◄──────────────────────────────── │
```

### 5.2 认证机制

| 方式 | 描述 | 适用场景 |
|------|------|----------|
| **API Key** | 静态密钥，业务API存储在GPU服务器 | 简单场景 |
| **JWT Token** | 动态令牌，可设置过期时间 | 生产环境 |
| **双向TLS** | 证书认证，最安全 | 高安全要求 |

**推荐**: JWT Token + API Key双重认证

### 5.3 数据传输格式

```python
# 请求格式
Content-Type: application/json
Authorization: Bearer <jwt_token>
X-Client-Id: <client_id>

# 响应格式
{
    "success": true,
    "data": {...},
    "error": null,
    "timestamp": "2026-03-14T10:00:00Z"
}
```

---

## 六、部署配置

### 6.1 业务API部署 (终端/本地)

```yaml
# docker-compose.business.yml
version: '3.8'

services:
  # 业务API
  business-api:
    build: ./business-api
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - TRAINING_API_URL=http://192.168.11.2:8001
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - ROBOFLOW_API_KEY=${ROBOFLOW_API_KEY}
    depends_on:
      - redis

  # 本地Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

networks:
  default:
    name: yolo-business-network
```

### 6.2 训练API部署 (GPU服务器)

```yaml
# docker-compose.training.yml
version: '3.8'

services:
  # 训练API
  training-api:
    build: ./training-api
    ports:
      - "8001:8000"
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./runs:/runs
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7-alpine

networks:
  default:
    name: yolo-training-network
```

---

## 七、最佳实践参考

### 7.1 来自Google ML Pipelines的设计原则

| 原则 | 描述 | 应用 |
|------|------|------|
| **模块化** | 每个服务专注单一职责 | 业务/训练完全分离 |
| **可观测性** | 日志、指标、追踪 | 添加Prometheus指标 |
| **幂等性** | 重复请求不产生副作用 | 任务ID去重 |
| **优雅降级** | 部分服务故障不影响整体 | 训练API不可用时本地缓存 |

### 7.2 来自Uber Michelangelo的设计模式

| 模式 | 描述 | 实现 |
|------|------|------|
| **异步训练** | 训练任务后台执行 | Celery + Redis |
| **模型版本管理** | 每次训练独立版本 | UUID命名 |
| **资源隔离** | 训练任务独立资源 | Docker容器 |

### 7.3 来自OpenAI的工程实践

| 实践 | 描述 | 实现 |
|------|------|------|
| **早期停止** | 指标不提升时停止训练 | Ray Tune EarlyStopping |
| **检查点保存** | 定期保存训练状态 | YOLO checkpoint |
| **资源调度** | 按需分配GPU | Celery队列优先级 |

---

## 八、数据流设计

### 8.1 完整训练流程

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              业务API (终端)                                      │
│                                                                                  │
│  1. 用户提交任务                                                                  │
│     POST /api/v1/train/submit                                                    │
│     {task_type: "training", model: "yolo11m", ...}                             │
│                              │                                                   │
│                              ▼                                                   │
│  2. 任务验证 + 状态初始化                                                        │
│     - 验证参数                                                                  │
│     - 生成task_id                                                               │
│     - 保存到本地Redis (status: pending)                                         │
│                              │                                                   │
│                              ▼                                                   │
│  3. 转发到GPU服务器                                                            │
│     POST http://192.168.11.2:8001/api/v1/internal/train/start                  │
│                              │                                                   │
│                              ▼                                                   │
│  4. 返回task_id给用户                                                         │
└──────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           │ HTTP请求
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           训练API (GPU服务器)                                    │
│                                                                                  │
│  5. 接收任务                                                                   │
│     - 验证API Key                                                              │
│     - 创建训练任务                                                             │
│                                                                                  │
│                              │                                                   │
│                              ▼                                                   │
│  6. YOLO训练执行                                                              │
│     - 下载数据集                                                               │
│     - 训练 (10 epochs sanity check → 100 epochs training)                     │
│     - 保存权重到 /runs/{task_id}/weights/best.pt                              │
│                                                                                  │
│                              │                                                   │
│                              ▼                                                   │
│  7. 训练完成                                                                  │
│     - 更新状态 (completed)                                                    │
│     - 回调业务API (可选)                                                       │
└──────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           │ 轮询/回调
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              业务API (终端)                                      │
│                                                                                  │
│  8. 用户查询结果                                                               │
│     GET /api/v1/train/status/{task_id}                                        │
│     ← {status: "completed", metrics: {...}, model_path: "/runs/..."}         │
│                                                                                  │
│                              │                                                   │
│                              ▼                                                   │
│  9. 下载模型 (可选)                                                            │
│     GET /api/v1/train/download/{task_id}                                      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 九、安全考虑

### 9.1 网络隔离

| 层级 | 措施 |
|------|------|
| **业务API** | 仅暴露8000端口给用户 |
| **训练API** | 仅内网暴露8001端口，禁止公网访问 |
| **Redis** | 仅业务API和训练API内网访问 |

### 9.2 认证流程

```
用户 → 业务API (JWT认证)
     → 验证token
     → 转发请求到训练API (API Key)
         → 验证API Key
         → 执行任务
```

### 9.3 数据安全

| 数据类型 | 保护措施 |
|----------|----------|
| **API密钥** | 环境变量，不提交到代码库 |
| **模型权重** | 存储在受控目录 |
| **传输数据** | HTTPS (生产环境) |

---

## 十、监控与运维

### 10.1 监控指标

| 指标 | 描述 | 采集方式 |
|------|------|----------|
| **API响应时间** | /health, /train/* 延迟 | FastAPI middleware |
| **GPU利用率** | nvidia-smi | Prometheus node_exporter |
| **训练进度** | epoch, mAP | YOLO callback |
| **队列长度** | Redis待处理任务 | Redis info |

### 10.2 健康检查

```python
# 业务API
GET /health
Response: {"status": "healthy", "redis": "connected", "training_api": "available"}

# 训练API
GET /health
Response: {"status": "healthy", "gpu": "available", "cuda_version": "12.1"}
```

---

## 十一、实施路线图

### Phase 1: 核心架构 (1周)
- [ ] 拆分业务API和训练API代码
- [ ] 实现HTTP通信层
- [ ] 基本认证机制

### Phase 2: 数据流通 (1周)
- [ ] 数据集上传/下载流程
- [ ] 模型结果回传
- [ ] 任务状态同步

### Phase 3: 生产就绪 (1周)
- [ ] 监控告警
- [ ] 日志收集
- [ ] 高可用配置

### Phase 4: 优化 (持续)
- [ ] 性能调优
- [ ] 资源调度优化
- [ ] 新功能集成

---

## 十二、附录

### A. 环境变量配置

**业务API (.env)**:
```bash
# 业务配置
BUSINESS_API_PORT=8000
REDIS_URL=redis://localhost:6379/0

# 训练API配置
TRAINING_API_URL=http://192.168.11.2:8001
TRAINING_API_KEY=your-training-api-key

# 数据源API
ROBOFLOW_API_KEY=xxx
KAGGLE_USERNAME=xxx
KAGGLE_KEY=xxx

# LLM
DEEPSEEK_API_KEY=xxx
```

**训练API (.env)**:
```bash
# 训练配置
TRAINING_API_PORT=8001
REDIS_URL=redis://localhost:6379/0

# 安全
API_KEY=your-api-key

# GPU
CUDA_VISIBLE_DEVICES=0
```

### B. Docker构建文件位置

| 文件 | 位置 |
|------|------|
| 业务API Dockerfile | `business-api/Dockerfile` |
| 训练API Dockerfile | `training-api/Dockerfile` |
| 业务docker-compose | `docker-compose.business.yml` |
| 训练docker-compose | `docker-compose.training.yml` |

---

*文档版本: 7.0*
*最后更新: 2026-03-14*
*基于: 分布式ML系统最佳实践 + 项目现状*
