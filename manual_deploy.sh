#!/bin/bash
# 手动部署脚本 - 在GPU服务器上运行
# 使用方法: ./manual_deploy.sh

set -e

echo "=== YOLO Auto-Training 手动部署脚本 ==="
echo ""

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo "错误: 未安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "错误: 未安装docker-compose"
    exit 1
fi

echo "步骤1: 创建环境变量文件..."
if [ ! -f .env ]; then
    cat > .env << 'EOF'
# JWT配置 (请修改为安全密钥)
JWT_SECRET_KEY=change-this-to-random-secret-key

# Redis配置
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# CORS配置
ALLOWED_ORIGINS=http://localhost:3000
EOF
    echo "✓ .env 文件已创建"
else
    echo "✓ .env 文件已存在"
fi

echo ""
echo "步骤2: 构建并启动服务..."
docker-compose up -d --build

echo ""
echo "步骤3: 等待服务启动..."
sleep 30

echo ""
echo "=== 部署完成 ==="
echo ""
echo "服务状态:"
docker-compose ps
echo ""
echo "API地址: http://localhost:8000"
echo "健康检查: curl http://localhost:8000/health"
