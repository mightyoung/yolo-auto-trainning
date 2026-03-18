# GPU服务器手动部署指南

## 服务器信息
- **IP**: `<YOUR_GPU_SERVER_IP>`
- **用户**: `<YOUR_SERVER_USER>`
- **密码**: `<YOUR_SERVER_PASSWORD>`
- **GPU**: NVIDIA L20 (46GB)

## 问题
SSH无法使用sudo命令（需要交互式终端），需要您手动在服务器上执行以下命令：

## 部署步骤

### 1. SSH连接到服务器
```bash
ssh <YOUR_SERVER_USER>@<YOUR_GPU_SERVER_IP>
# 密码: <YOUR_SERVER_PASSWORD>
```

### 2. 添加用户到docker组（一次性）
```bash
sudo usermod -aG docker $USER
# 退出SSH重新登录使权限生效
```

或者，如果您不想添加用户到docker组，可以在所有docker命令前加`sudo`。

### 3. 进入项目目录
```bash
cd ~/yolo-auto-training
```

如果没有该目录，解压部署包：
```bash
cd ~
tar -xzvf yolo-auto-training.tar.gz
cd yolo-auto-training
```

### 4. 启动服务
```bash
# 使用sudo运行docker-compose
sudo docker-compose up -d --build
```

### 5. 验证部署
```bash
# 查看服务状态
sudo docker-compose ps

# 健康检查
curl http://localhost:8000/health
```

## 服务地址
- API: http://<YOUR_GPU_SERVER_IP>:8000
- 健康检查: http://<YOUR_GPU_SERVER_IP>:8000/health

## 常用命令
```bash
# 查看日志
sudo docker-compose logs -f

# 停止服务
sudo docker-compose down

# 重启服务
sudo docker-compose restart
```
