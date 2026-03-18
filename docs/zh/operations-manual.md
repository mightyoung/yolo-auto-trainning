# YOLO Auto-Training System - 运维手册

**版本**: 1.0
**日期**: 2026-03-14

---

## 1. 系统架构

### 1.1 概览

```
+-----------------------------------------------------------------+
|                        FastAPI 网关                             |
|                    (端口 8000, 健康检查)                        |
+----------------------------+----------------------------------+
                             |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
+---------------+    +---------------+    +---------------+
| 数据服务     |    | 训练服务     |    | 部署服务     |
| /api/v1/data|    |/api/v1/train|    |/api/v1/deploy|
+-------+------+    +-------+------+    +-------+------+
        |                      |                      |
        +----------------------+----------------------+
                             |
                             v
                    +---------------------+
                    |    Celery 任务     |
                    |       队列         |
                    +---------+----------+
                             |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
+--------------+    +--------------+    +--------------+
|   训练      |    |     HPO      |    |    导出     |
|   Worker    |    |    Worker    |    |    Worker   |
|  (GPU/CPU) |    |   (CPU)      |    |   (CPU)     |
+--------------+    +--------------+    +--------------+
```

### 1.2 组件

| 组件 | 技术 | 用途 |
|------|------|------|
| API 网关 | FastAPI | REST API、认证、限流 |
| 任务队列 | Celery + Redis | 异步任务处理 |
| 训练 | Ultralytics + Ray Tune | YOLO 训练与 HPO |
| 存储 | 本地文件系统 | 模型、数据集、日志 |
| 认证 | JWT + Redis | API 密钥管理 |

---

## 2. 部署

### 2.1 Docker 部署

```bash
# 构建并启动所有服务
docker-compose up -d

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f api
docker-compose logs -f celery-worker
```

### 2.2 手动部署

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Redis
redis-server

# 启动 API（终端 1）
uvicorn src.api.gateway:app --host 0.0.0.0 --port 8000

# 启动 Celery worker（终端 2）
celery -A src.api.tasks worker --loglevel=info --concurrency=2

# 启动 Celery beat（终端 3）
celery -A src.api.tasks beat --loglevel=info
```

---

## 3. 配置

### 3.1 环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `REDIS_URL` | redis://localhost:6379/0 | Redis 连接 |
| `CELERY_BROKER_URL` | redis://localhost:6379/0 | Celery 消息代理 |
| `CELERY_RESULT_BACKEND` | redis://localhost:6379/0 | 结果后端 |
| `JWT_SECRET_KEY` | 自动生成 | JWT 签名密钥 |
| `ALLOWED_ORIGINS` | localhost:3000,localhost:8080 | CORS 来源 |
| `ROBOFLOW_API_KEY` | - | Roboflow API 密钥 |
| `KAGGLE_USERNAME` | - | Kaggle 用户名 |
| `KAGGLE_KEY` | - | Kaggle API 密钥 |

### 3.2 限流

| 端点 | 限制 |
|------|------|
| `/api/v1/train/*` | 10 次/分钟 |
| `/api/v1/deploy/*` | 10 次/分钟 |
| `/api/v1/data/*` | 60 次/分钟 |

### 3.3 训练资源

| 资源 | 默认值 | 描述 |
|------|--------|------|
| Worker 内存 | 8GB | 每个 Worker 最大内存 |
| Worker CPU | 4 核 | CPU 分配 |
| 训练超时 | 10 小时 | 最大训练时间 |
| HPO 试验 | 50 | 最大 HPO 试验次数 |

---

## 4. 监控

### 4.1 健康检查

```bash
# API 健康
curl http://localhost:8000/health

# Redis 健康
redis-cli ping

# Celery Worker
celery -A src.api.tasks inspect active
```

### 4.2 日志

| 服务 | 日志位置 |
|------|----------|
| API | stdout (uvicorn) |
| Celery | stdout |
| 训练 | ./runs/ |

### 4.3 指标

```bash
# 查看训练进度
curl http://localhost:8000/api/v1/train/status/{task_id}

# 查看导出进度
curl http://localhost:8000/api/v1/deploy/export/status/{task_id}
```

---

## 5. 备份与恢复

### 5.1 备份

```bash
# 备份模型
tar -czf models_backup.tar.gz ./runs/

# 备份数据集
tar -czf data_backup.tar.gz ./data/

# 备份 Redis（如已持久化）
redis-cli SAVE
```

### 5.2 恢复

```bash
# 恢复模型
tar -xzf models_backup.tar.gz

# 恢复数据
tar -xzf data_backup.tar.gz
```

---

## 6. 安全

### 6.1 认证

系统支持两种认证方式：

1. **API 密钥**（推荐用于生产环境）
   - 请求头：`X-API-Key: yolo_xxx`
   - 密钥存储在 Redis 中，有效期 30 天

2. **JWT 令牌**（推荐用于 Web 应用）
   - 请求头：`Authorization: Bearer <token>`
   - 访问令牌：30 分钟
   - 刷新令牌：7 天

### 6.2 API 密钥管理

```bash
# 通过 API 生成 API 密钥
POST /api/v1/auth/api-key

# 或通过代码生成
from src.api.gateway import generate_api_key, store_api_key_in_redis
key = generate_api_key()
store_api_key_in_redis(key, "user_id")
```

### 6.3 CORS 配置

在环境中编辑 `ALLOWED_ORIGINS`：

```bash
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

---

## 7. 扩展

### 7.1 水平扩展

```bash
# 添加更多 Celery Worker
celery -A src.api.tasks worker --concurrency=4

# 或使用 Docker
docker-compose up --scale celery-worker=3
```

### 7.2 垂直扩展

| 资源 | 开发环境 | 生产环境 |
|------|----------|----------|
| API 内存 | 2GB | 4GB |
| Worker 内存 | 4GB | 8GB+ |
| Worker CPU | 2 核 | 4+ 核 |
| GPU | 可选 | NVIDIA GPU |

### 7.3 GPU 训练

```bash
# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# 或在 docker-compose 中
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

## 8. 维护

### 8.1 日志轮转

在 `docker-compose.yml` 中配置：

```yaml
logging:
  options:
    max-size: "10m"
    max-file: "3"
```

### 8.2 清理

```bash
# 清理旧的训练运行
find ./runs -type d -mtime +30 -exec rm -rf {} \;

# 清理旧的数据集
find ./data -type d -mtime +90 -exec rm -rf {} \;

# 清理 Celery 结果
celery -A src.api.tasks purge
```

### 8.3 更新

```bash
# 拉取最新代码
git pull

# 重新构建容器
docker-compose build

# 重启服务
docker-compose up -d
```

---

## 9. 故障排除

### 9.1 常见问题

| 症状 | 原因 | 解决方案 |
|------|------|----------|
| API 返回 401 | 认证无效/缺失 | 检查 API 密钥或令牌 |
| API 返回 429 | 超出限流 | 等待1分钟 |
| 训练卡住 | Worker 崩溃 | 检查 Celery 日志 |
| 导出失败 | CUDA 不可用 | 验证 GPU 访问 |
| 响应慢 | Worker 过载 | 扩展 Worker |

### 9.2 调试命令

```bash
# 检查 Celery 任务
celery -A src.api.tasks inspect active
celery -A src.api.tasks inspect scheduled

# 检查 Redis
redis-cli INFO
redis-cli KEYS "celery*"

# 查看 API 日志
docker-compose logs api --tail=100

# 查看 Worker 日志
docker-compose logs celery-worker --tail=100
```

### 9.3 紧急恢复

```bash
# 停止所有服务
docker-compose down

# 清除 Redis
redis-cli FLUSHALL

# 重启服务
docker-compose up -d
```

---

## 10. 性能调优

### 10.1 训练优化

| 参数 | 默认值 | 优化建议 |
|------|--------|----------|
| 批量大小 | 16 | 如 GPU 内存允许则增加 |
| Worker 数量 | 2 | 增加以并行训练 |
| 缓存 | 启用 | 加快 epoch 速度 |

### 10.2 API 优化

| 参数 | 默认值 | 优化建议 |
|------|--------|----------|
| Worker 线程 | 4 | 高流量时增加 |
| Keep-alive | 5秒 | 调整连接模式 |
| 超时 | 30秒 | 长时间任务增加 |

---

## 11. 灾难恢复

### 11.1 故障场景

| 场景 | 恢复时间 | 数据丢失 |
|------|----------|----------|
| Redis 崩溃 | < 1 分钟 | 任务队列丢失 |
| Worker 崩溃 | < 5 分钟 | 当前任务 |
| API 崩溃 | < 1 分钟 | 无 |
| 存储满 | 不可用 | 可预防 |

### 11.2 灾难恢复清单

- [ ] 每日备份模型和数据
- [ ] 启用 Redis 持久化（AOF）
- [ ] 配置健康检查
- [ ] 设置告警
- [ ] 记录运维手册

---

## 12. 支持

### 12.1 日志位置

| 问题 | 日志 |
|------|------|
| API 错误 | API stdout |
| 任务失败 | Celery Worker stdout |
| 训练问题 | ./runs/ 目录 |
| 系统错误 | 系统日志 |

### 12.2 获取帮助

1. 首先检查日志
2. 查看本运维手册
3. 检查 GitHub issues
4. 联系支持时提供：
   - 错误信息
   - 复现步骤
   - 日志片段

---

*文档版本：1.0*
*最后更新：2026-03-14*
