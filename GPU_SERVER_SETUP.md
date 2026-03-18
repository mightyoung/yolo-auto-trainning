# GPU服务器配置指南

## 服务器信息
- **IP**: `<YOUR_GPU_SERVER_IP>`
- **用户**: `<YOUR_SERVER_USER>`
- **GPU**: L20

## 方式一：手动配置（推荐）

在GPU服务器上执行以下步骤：

### 1. 安装依赖
```bash
# 安装Python和Docker
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv docker.io docker-compose git

# 安装SSH（如果未安装）
sudo apt-get install -y openssh-client
```

### 2. 启动Redis
```bash
# 使用Docker启动Redis
docker run -d --name yolo-redis -p 6379:6379 redis:7-alpine
```

### 3. 配置项目
```bash
# 创建项目目录
mkdir -p ~/yolo-auto-training
cd ~/yolo-auto-training

# 复制项目文件到服务器
# 可以使用scp或从git clone

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量（复制.env内容）
```

### 4. 启动服务
```bash
# 启动API
uvicorn src.api.gateway:app --host 0.0.0.0 --port 8000

# 或使用Docker Compose
docker-compose up -d
```

---

## 方式二：使用配置脚本

1. 将 `setup_gpu_server.sh` 复制到GPU服务器
2. 执行脚本：
```bash
chmod +x setup_gpu_server.sh
./setup_gpu_server.sh
```

---

## 需要我帮您做什么？

请告诉我：
1. **能否SSH到服务器？** - 如果可以，您手动执行上述命令
2. **需要我尝试自动连接吗？** - 如果您提供正确的密码，我可以尝试
