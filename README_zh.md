# YOLO 自动训练系统

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/mightyoung/yolo-auto-trainning/actions/workflows/ci-cd.yml/badge.svg)
![Stage](https://img.shields.io/badge/%E9%98%B6%E6%AE%B5-Beta-orange)

端到端 AI 驱动的 YOLO 模型训练与部署平台。从数据集发现到边缘推理 — 全流程自动化，基于 CrewAI 多智能体编排、Ray Tune 超参数优化，一键导出至 NVIDIA Jetson / Rockchip RK3588 等边缘设备。

---

## 核心特性

| 特性 | 说明 |
|---|---|
| **数据集发现** | 跨 Roboflow、Kaggle、HuggingFace 多源搜索，含相关性评分 |
| **自动训练** | YOLO11 训练 + Ray Tune HPO + MLflow 实验追踪 |
| **知识蒸馏** | 用大型教师模型蒸馏小型学生模型 |
| **边缘部署** | 一键导出 ONNX / TensorRT，支持 Jetson Nano、Jetson Orin、RK3588 |
| **多智能体编排** | CrewAI 驱动的 Data Discovery、Generation、Training、Deployment 四大智能体 |
| **异步任务管道** | Celery + Redis 任务队列，处理 GPU 密集型后台作业 |
| **MLOps 可观测性** | Prometheus 指标、Grafana 仪表盘、结构化日志（ELK 技术栈） |
| **自动标注** | 基于 Grounded SAM 的半自动化标注流水线 |

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                        Web UI (Next.js)                       │
│               数据集发现 │ 训练管理 │ 标注 │ 分析              │
└──────────────────────────┬─────────────────────────────────┘
                           │ HTTP / REST
┌──────────────────────────▼─────────────────────────────────┐
│              Business API  (FastAPI, 端口 8000)               │
│  认证 │ 数据集发现 │ 智能体编排 │ 请求路由                     │
└──────────┬──────────────────────────────────────────────────┘
           │ 内部 HTTP 调用
┌──────────▼──────────────────────────────────────────────────┐
│              Training API  (FastAPI, 端口 8001)                │
│     YOLO 训练 │ Ray Tune HPO │ 模型导出 │ MLflow 追踪         │
└──────────┬──────────────────────────────────────────────────┘
           │
    ┌──────▼──────┐     ┌─────────────┐     ┌─────────────┐
    │  GPU 服务器   │     │  Redis 7    │     │  MLflow     │
    │ (CUDA 12.1) │     │  (消息队列)  │     │  (实验追踪)  │
    └─────────────┘     └─────────────┘     └─────────────┘
```

---

## 快速开始

### 环境要求

- Python 3.10+
- Docker 和 Docker Compose
- （GPU 训练）支持 CUDA 12.1 的 NVIDIA GPU

### 1. 克隆并配置

```bash
git clone https://github.com/mightyoung/yolo-auto-trainning.git
cd yolo-auto-training
cp .env.example .env
# 编辑 .env 填入你的 API 密钥（Roboflow、Kaggle、HuggingFace 等）
```

### 2. 启动服务（Docker Compose）

```bash
# 一键启动：Redis + Business API + Celery worker + GPU 训练
docker-compose up -d --build

# 完整 MLOps 技术栈（Prometheus + Grafana + ELK）
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

### 3. 服务访问

| 服务 | 地址 |
|---|---|
| Business API | http://localhost:8000 |
| API 文档 | http://localhost:8000/docs |
| Training API | http://localhost:8001 |
| Training API 文档 | http://localhost:8001/docs |
| Grafana | http://localhost:3000 |
| Kibana | http://localhost:5601 |

---

## 本地开发

### Python 环境

```bash
# 创建虚拟环境
python3.11 -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 运行测试
pytest tests/ -v
```

### 手动启动各服务

```bash
# Business API
uvicorn business-api.src.api.gateway:app --host 0.0.0.0 --port 8000 --reload

# Training API（GPU 节点）
uvicorn training-api.src.api.gateway:app --host 0.0.0.0 --port 8001 --reload

# Celery Worker
celery -A business-api.src.api.tasks worker --loglevel=info
```

---

## API 参考

### 数据集发现

```bash
# 在 Roboflow、Kaggle、HuggingFace 中搜索数据集
curl -X POST http://localhost:8000/api/v1/data/search \
  -H "Content-Type: application/json" \
  -d '{"query": "vehicle detection", "max_results": 10}'
```

### 提交训练任务

```bash
# 启动 YOLO11 训练
curl -X POST http://localhost:8000/api/v1/train/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo11m",
    "data_yaml": "/data/my_dataset.yaml",
    "epochs": 100,
    "imgsz": 640
  }'
```

### 超参数优化

```bash
# 启动 Ray Tune HPO
curl -X POST http://localhost:8000/api/v1/train/hpo/start \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo11m",
    "data_yaml": "/data/my_dataset.yaml",
    "n_trials": 50,
    "epochs_per_trial": 50
  }'
```

### 导出到边缘设备

```bash
# 导出为 NVIDIA Jetson Orin 格式（TensorRT）
curl -X POST http://localhost:8000/api/v1/deploy/export \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/runs/train/exp/weights/best.pt",
    "platform": "jetson_orin",
    "imgsz": 640
  }'

# 导出为 Rockchip RK3588 格式（ONNX + RKNN）
curl -X POST http://localhost:8000/api/v1/deploy/export \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/runs/train/exp/weights/best.pt",
    "platform": "rk3588",
    "imgsz": 640
  }'
```

### 认证

```bash
# 获取 JWT 令牌
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your-password"}'

# 在请求中携带令牌
curl -X POST http://localhost:8000/api/v1/train/submit \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"model": "yolo11n", "data_yaml": "/data/dataset.yaml", "epochs": 50}'
```

---

## 项目结构

```
yolo-auto-training/
├── business-api/           # 业务编排 API（端口 8000）
│   └── src/api/
│       ├── gateway.py      # FastAPI 应用 + JWT 认证
│       ├── routes.py       # 数据、训练、导出、分析接口
│       ├── training_client.py   # → Training API 的 HTTP 客户端
│       ├── agent_routes.py     # CrewAI 智能体接口
│       └── agents/
│           └── orchestration.py  # 多智能体工作流定义
├── training-api/           # GPU 训练 API（端口 8001）
│   └── src/
│       ├── training/
│       │   ├── runner.py   # YOLO 训练器（ultralytics 封装）
│       │   └── mlflow_tracker.py  # MLflow 集成
│       └── deployment/
│           └── exporter.py # ONNX / TensorRT / TFLite 导出器
├── web-ui-react/           # Next.js 前端
│   └── src/app/
│       ├── discovery/      # 数据集发现界面
│       ├── training/      # 训练管理界面
│       ├── labeling/      # 自动标注界面
│       └── analysis/      # 分析仪表盘界面
├── src/                    # 单体核心（遗留）
│   ├── api/               # FastAPI 路由 + Celery 任务
│   ├── agents/            # CrewAI 编排
│   ├── data/              # 数据集发现 + 质量过滤
│   ├── training/          # YOLO 训练 + Ray Tune HPO
│   ├── deployment/        # 模型导出
│   ├── inference/         # 推理引擎
│   ├── monitoring/        # 漂移检测
│   ├── pipeline/          # 端到端编排器
│   └── features/         # 特征存储
├── tests/                 # 单元测试和集成测试
├── docs/                  # 文档（英文/中文）
│   ├── en/               # 英文文档
│   └── zh/               # 中文文档
├── docker-compose.yml     # 核心技术栈（Redis + API + Celery）
├── docker-compose.monitoring.yml  # Prometheus + Grafana
├── docker-compose.logging.yml    # ELK 技术栈
└── pyproject.toml        # 项目元数据 + 依赖
```

---

## 配置说明

所有敏感配置通过环境变量管理。将 `.env.example` 复制为 `.env` 后进行配置：

```bash
# 核心配置
JWT_SECRET_KEY=<生成：python -c "import secrets; print(secrets.token_urlsafe(32))">
REDIS_URL=redis://localhost:6379/0

# 训练 API 地址（GPU 服务器的内网地址）
TRAINING_API_URL=http://localhost:8001
TRAINING_API_KEY=<your-api-key>

# 数据集来源
ROBOFLOW_API_KEY=<your-roboflow-key>
KAGGLE_USERNAME=<your-kaggle-username>
KAGGLE_KEY=<your-kaggle-key>
HF_TOKEN=<your-huggingface-token>

# AI 提供商
DEEPSEEK_API_KEY=<your-deepseek-key>

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## 边缘部署目标平台

| 平台 | 格式 | 工具链 | 最大批量 |
|---|---|---|---|
| **Jetson Nano** | TensorRT FP16 | `tensorrt` | 8 |
| **Jetson Orin** | TensorRT FP16/INT8 | `tensorrt` | 32 |
| **RK3588** | ONNX + RKNN | `rknn` | 16 |
| **x86 服务器** | ONNX | `onnx` | 64 |

---

## 监控与告警

### Prometheus 指标

| 指标 | 说明 |
|---|---|
| `yolo_training_jobs_total` | 已提交训练任务总数 |
| `yolo_training_duration_seconds` | 训练时长分布直方图 |
| `yolo_api_requests_total` | 各端点 API 请求计数器 |
| `yolo_gpu_memory_usage` | GPU 显存使用量仪表盘 |
| `yolo_export_jobs_total` | 模型导出任务计数器 |

### 告警规则

| 告警 | 触发条件 | 严重程度 |
|---|---|---|
| `HighErrorRate` | 5 分钟内错误率 > 5% | 警告 |
| `APIDown` | Business API 不可达 > 1 分钟 | 紧急 |
| `GPUMemoryHigh` | GPU 显存 > 90% 持续 5 分钟 | 警告 |
| `TrainingJobFailed` | 连续 3 次训练失败 | 紧急 |

---

## 贡献指南

欢迎贡献代码！提交 Pull Request 前请阅读贡献指南。

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交改动 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

---

## 开源许可

本项目基于 MIT 许可证开源。详见 [LICENSE](LICENSE) 文件。

---

## 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO 实现
- [CrewAI](https://github.com/crewAI/crewAI) — 多智能体框架
- [Ray Tune](https://github.com/ray-project/ray) — 超参数优化
- [Roboflow](https://roboflow.com) — 数据集平台
- [HuggingFace](https://huggingface.co) — 模型中心
