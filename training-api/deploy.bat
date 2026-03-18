@echo off
REM Training API Deployment Script for Windows
REM Run this in Docker-enabled environment

echo Building Docker image...
cd /d %~dp0training-api
docker build -t yolo-training-api:latest .

echo.
echo Saving image to tar...
docker save -o yolo-training-api.tar yolo-training-api:latest

echo.
echo Build complete!
echo.
echo To transfer to GPU server:
echo   scp yolo-training-api.tar user@<YOUR_GPU_SERVER_IP>:/path/
echo.
echo On GPU server, run:
echo   docker load -i yolo-training-api.tar
echo   docker run -d --gpus all --name yolo-training-api -p 8001:8001 yolo-training-api:latest
