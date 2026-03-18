#!/bin/bash
# GPU服务器离线部署脚本
# 使用方法: ./deploy_gpu.sh

set -e

SERVER_USER="<YOUR_SERVER_USER>"
SERVER_IP="<YOUR_GPU_SERVER_IP>"
PROJECT_DIR="yolo-auto-training"

echo "=== YOLO Auto-Training GPU 部署脚本 ==="
echo ""

# 检查本地Docker是否可用
if ! command -v docker &> /dev/null; then
    echo "错误: 本地未安装Docker"
    exit 1
fi

echo "步骤1: 构建Docker镜像..."
docker build -t yolo-auto-training:latest .
echo "✓ 镜像构建完成"

echo ""
echo "步骤2: 导出镜像为tar文件..."
docker save -o ${PROJECT_DIR}.tar yolo-auto-training:latest
echo "✓ 镜像导出完成: ${PROJECT_DIR}.tar"

echo ""
echo "步骤3: 复制到GPU服务器..."
scp ${PROJECT_DIR}.tar ${SERVER_USER}@${SERVER_IP}:~/
echo "✓ 文件复制完成"

echo ""
echo "步骤4: 复制项目文件..."
scp -r . ${SERVER_USER}@${SERVER_IP}:~/yolo-auto-training/ --exclude '.git' --exclude '*.pyc' --exclude '__pycache__' --exclude '.pytest_cache' --exclude 'htmlcov' --exclude '*.tar'
echo "✓ 项目文件复制完成"

echo ""
echo "步骤5: 在服务器上执行部署..."
ssh ${SERVER_USER}@${SERVER_IP} << 'EOF'
    cd ~/yolo-auto-training

    # 加载镜像
    docker load -i ${PROJECT_DIR}.tar

    # 创建环境变量文件
    if [ ! -f .env ]; then
        echo "创建环境变量文件..."
        cat > .env << 'ENVEOF'
# JWT配置
JWT_SECRET_KEY=your-secret-key-change-me

# Redis配置
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# CORS配置
ALLOWED_ORIGINS=http://localhost:3000
ENVEOF
    fi

    # 启动服务
    docker-compose up -d

    echo ""
    echo "=== 部署完成 ==="
    echo "API地址: http://${SERVER_IP}:8000"
    echo "健康检查: curl http://${SERVER_IP}:8000/health"
    echo ""
    echo "查看日志: docker-compose logs -f"
EOF

echo ""
echo "=== 部署完成 ==="
echo "请SSH到服务器验证服务状态"
