#!/bin/bash
# GPU服务器在线部署脚本（服务器可访问互联网）
# 使用方法: ./deploy_online.sh

set -e

SERVER_USER="<YOUR_SERVER_USER>"
SERVER_IP="<YOUR_GPU_SERVER_IP>"
PROJECT_DIR="yolo-auto-training"

echo "=== YOLO Auto-Training GPU 在线部署脚本 ==="
echo ""

echo "步骤1: 创建排除文件列表..."
cat > /tmp/exclude_list.txt << 'EOF'
.git
.gitignore
.venv
venv
__pycache__
*.pyc
*.pyo
.pytest_cache
.coverage
htmlcov
*.egg-info
.eggs
node_modules
.npm
.bun
.DS_Store
*.tar
*.tar.gz
.env
.vscode
.idea
design-docs
docs
README.md
CLAUDE.md
GPU_SERVER_SETUP.md
GPU_DEPLOYMENT.md
deploy_gpu.sh
deploy_online.sh
EOF
echo "✓ 排除列表创建完成"

echo "步骤2: 打包项目文件..."
tar -czvf ${PROJECT_DIR}.tar.gz -X /tmp/exclude_list.txt .
echo "✓ 项目打包完成 ($(du -h ${PROJECT_DIR}.tar.gz | cut -f1))"

echo ""
echo "步骤3: 复制到GPU服务器..."
scp ${PROJECT_DIR}.tar.gz ${SERVER_USER}@${SERVER_IP}:~/
echo "✓ 文件复制完成"

echo ""
echo "步骤4: 在服务器上执行部署..."
ssh ${SERVER_USER}@${SERVER_IP} << 'EOF'
    cd ~

    # 解压项目
    tar -xzvf yolo-auto-training.tar.gz

    cd yolo-auto-training

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

    # 构建并启动服务
    docker-compose up -d --build

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
