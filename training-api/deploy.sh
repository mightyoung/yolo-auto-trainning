# Training API Deployment Script
# Run this on a machine with Docker installed

# 1. Build the Docker image
cd training-api
docker build -t yolo-training-api:latest .

# 2. Save the image to a tar file
docker save -o yolo-training-api.tar yolo-training-api:latest

# 3. Transfer to GPU server (replace with your server details)
# Using scp (Linux/Mac):
scp yolo-training-api.tar user@192.168.11.2:/path/to/destination/

# Or using pscp (Windows PuTTY):
# pscp yolo-training-api.tar user@192.168.11.2:/path/to/destination/

# 4. On GPU server, load the image:
docker load -i yolo-training-api.tar

# 5. Run the container:
docker run -d \
  --gpus all \
  --name yolo-training-api \
  -p 8001:8001 \
  -v /path/to/data:/runs \
  -v /path/to/models:/models \
  yolo-training-api:latest
