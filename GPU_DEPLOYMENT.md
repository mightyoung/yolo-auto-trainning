# GPU服务器Docker部署指南

## 概述
本文档说明如何在GPU服务器上部署YOLO自动训练系统。

## 前提条件
- NVIDIA GPU (支持CUDA 12.1)
- Docker + nvidia-docker2
- NVIDIA驱动

## 部署步骤

### 方式一：在线部署（推荐）

如果服务器可以访问互联网：

```bash
# 1. 克隆项目
git clone <repository-url> yolo-auto-training
cd yolo-auto-training

# 2. 创建环境变量文件
cp .env.example .env
# 编辑 .env 文件，设置必要的环境变量

# 3. 构建并启动所有服务
docker-compose up -d --build

# 4. 查看服务状态
docker-compose ps

# 5. 查看日志
docker-compose logs -f api
docker-compose logs -f celery-worker
```

### 方式二：离线部署（网络受限）

如果服务器无法访问互联网，需要先在本地构建镜像：

```bash
# 1. 在本地机器上构建镜像
docker build -t yolo-auto-training:latest .

# 2. 导出镜像为tar文件
docker save -o yolo-auto-training.tar yolo-auto-training:latest

# 3. 将tar文件复制到GPU服务器
scp yolo-auto-training.tar <YOUR_SERVER_USER>@<YOUR_GPU_SERVER_IP>:~/

# 4. 在GPU服务器上加载镜像
docker load -i yolo-auto-training.tar

# 5. 复制项目文件到服务器
scp -r yolo-auto-training <YOUR_SERVER_USER>@<YOUR_GPU_SERVER_IP>:~/

# 6. 在GPU服务器上启动服务
cd yolo-auto-training
docker-compose up -d
```

## 验证部署

### 检查GPU访问
```bash
# 检查nvidia-docker是否正常工作
docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi
```

### 检查服务健康
```bash
# API健康检查
curl http://<YOUR_GPU_SERVER_IP>:8000/health

# 检查Celery worker
docker-compose logs celery-worker | grep -i ready
```

## 服务端口
- API: 8000
- Redis: 6379

## 常用命令
```bash
# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 查看实时日志
docker-compose logs -f

# 进入容器
docker-compose exec api bash
docker-compose exec celery-worker bash
```

## GPU训练测试
```bash
# 测试GPU训练
curl -X POST http://<YOUR_GPU_SERVER_IP>:8000/api/v1/train/start \
  -H "Content-Type: application/json" \
  -d '{
    "data_yaml": "/data/dataset.yaml",
    "model": "yolo11n",
    "epochs": 10
  }'
```
