# 模块详细设计文档

本文档提供各核心模块的详细实现设计。

---

## 1. 数据生成模块 (Data Generator)

### 1.1 核心代码设计

```python
# src/data_generator.py
from typing import Dict, List, Optional
from dataclasses import dataclass
import asyncio

@dataclass
class GenerationRequest:
    """数据生成请求"""
    num_images: int
    class_prompts: Dict[str, str]  # class_name -> prompt template
    output_dir: str = "./data/synthetic"
    quality_threshold: float = 0.7

@dataclass
class GenerationResult:
    """生成结果"""
    images: List[str]  # image paths
    annotations: List[dict]  # COCO format
    clip_scores: List[float]
    rejected: int

class SyntheticDataGenerator:
    """合成数据生成器"""

    def __init__(self, llm_provider: str = "qwen", comfy_host: str = "localhost:8188"):
        self.llm = LLMClient(provider=llm_provider)
        self.comfy = ComfyClient(host=comfy_host)
        self.vlm = VLMClient(provider=llm_provider)
        self.quality_checker = CLIPQualityChecker()

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """执行合成数据生成流程"""

        # 1. LLM 生成多样化 prompt
        prompts = await self.llm.generate_prompts(
            templates=request.class_prompts,
            num=request.num_images,
            variations=["indoor", "outdoor", "closeup", "far", "different angles"]
        )

        # 2. ComfyUI 生成图像 (批量)
        images = []
        for prompt in prompts:
            img_path = await self.comfy.generate(
                prompt=prompt,
                workflow="sdxl_base"
            )
            images.append(img_path)

        # 3. VLM 自动标注 (Qwen2-VL)
        annotations = []
        for img_path in images:
            bboxes = await self.vlm.detect_objects(img_path)
            annotations.append({
": img                "image_path,
                "bboxes": bboxes
            })

        # 4. 质量过滤 (CLIP Score)
        filtered_images = []
        filtered_annotations = []
        clip_scores = []
        rejected = 0

        for img, ann in zip(images, annotations):
            score = await self.quality_checker.score(img, ann)
            clip_scores.append(score)
            if score >= request.quality_threshold:
                filtered_images.append(img)
                filtered_annotations.append(ann)
            else:
                rejected += 1

        return GenerationResult(
            images=filtered_images,
            annotations=filtered_annotations,
            clip_scores=clip_scores,
            rejected=rejected
        )
```

### 1.2 ComfyUI 集成

```python
# src/integrations/comfy.py
import aiohttp
import asyncio
import json

class ComfyClient:
    """ComfyUI API 客户端"""

    def __init__(self, host: str = "localhost:8188"):
        self.host = host
        self.base_url = f"http://{host}"

    async def generate(self, prompt: str, workflow: str = "sdxl") -> str:
        """触发图像生成"""

        # 1. 获取工作流 JSON
        workflow_json = await self._load_workflow(workflow)

        # 2. 修改 prompt
        workflow_json["6"]["inputs"]["text"] = prompt

        # 3. 提交任务
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow_json}
            ) as resp:
                result = await resp.json()
                prompt_id = result["prompt_id"]

            # 4. 等待完成
            image_path = await self._wait_for_result(prompt_id)

        return image_path

    async def _wait_for_result(self, prompt_id: str, timeout: int = 120) -> str:
        """等待生成完成"""
        while True:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/history/{prompt_id}"
                ) as resp:
                    data = await resp.json()
                    if prompt_id in data:
                        output = data[prompt_id]["outputs"]
                        # 提取图像路径
                        return output["10"]["images"][0]["filename"]
                    await asyncio.sleep(2)
```

### 1.3 VLM 自动标注

```python
# src/integrations/vlm.py
from typing import List, Dict

class VLMClient:
    """VLM 客户端 for 自动标注"""

    def __init__(self, provider: str = "qwen"):
        self.provider = provider

    async def detect_objects(self, image_path: str) -> List[Dict]:
        """使用 VLM 检测物体并返回 bounding boxes"""

        if self.provider == "qwen":
            return await self._qwen_detect(image_path)
        elif self.provider == "openai":
            return await self._openai_detect(image_path)

    async def _qwen_detect(self, image_path: str) -> List[Dict]:
        """Qwen2-VL 物体检测"""
        # 调用 DashScope API
        response = await dashscope.MultiModalConversation.call(
            model='qwen2-vl-max',
            messages=[{
                'role': 'user',
                'content': [
                    {'image': image_path},
                    {'text': 'Detect all objects in this image. Return bounding boxes in format: class_name, x_min, y_min, x_max, y_max'}
                ]
            }]
        )

        # 解析响应
        return self._parse_qwen_response(response)

    def _parse_qwen_response(self, response) -> List[Dict]:
        """解析 Qwen 响应为标准格式"""
        # 解析 JSON 格式的 bboxes
        results = []
        text = response.output.choices[0].message.content[0]['text']

        for line in text.split('\n'):
            if ',' in line:
                parts = line.split(',')
                results.append({
                    "class": parts[0].strip(),
                    "bbox": [float(x) for x in parts[1:]]
                })

        return results
```

---

## 2. 训练模块 (Trainer)

### 2.1 YOLO 训练器

```python
# src/trainer.py
from ultralytics import YOLO
import optuna
from typing import Optional, Dict
import yaml

class YOLOTrainer:
    """YOLO 训练器 with HPO 支持"""

    def __init__(self, model_size: str = "n"):
        """
        model_size: n/m/s/l/x
        n = nano (最轻量)
        x = xlarge (最大)
        """
        self.model_size = model_size
        self.model = None

    def train(self, data_yaml: str, epochs: int = 100, **kwargs):
        """标准训练"""
        self.model = YOLO(f"yolov10{self.model_size}.pt")

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            **kwargs
        )

        return results

    def train_with_hpo(self, data_yaml: str, trials: int = 20) -> Dict:
        """使用 Optuna 超参优化"""

        def objective(trial: optuna.Trial):
            params = {
                'lr0': trial.suggest_float('lr0', 1e-4, 1e-2, log=True),
                'lrf': trial.suggest_float('lrf', 0.01, 1.0),
                'momentum': trial.suggest_float('momentum', 0.9, 0.999),
                'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
                'warmup_epochs': trial.suggest_int('warmup_epochs', 0, 10),
                'box': trial.suggest_float('box', 0.1, 10.0),
                'cls': trial.suggest_float('cls', 0.1, 10.0),
                'hsv_h': trial.suggest_float('hsv_h', 0.0, 0.1),
                'hsv_s': trial.suggest_float('hsv_s', 0.0, 0.9),
                'hsv_v': trial.suggest_float('hsv_v', 0.0, 0.9),
            }

            results = self.model.train(
                data=data_yaml,
                epochs=30,  # 快速搜索用较少 epoch
                **params,
                verbose=False
            )

            return results.box.map50  # mAP@0.5

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials)

        # 使用最佳参数完整训练
        best_params = study.best_params
        self.model = YOLO(f"yolov10{self.model_size}.pt")
        final_results = self.model.train(
            data=data_yaml,
            epochs=100,
            **best_params
        )

        return {
            "best_params": best_params,
            "best_map": study.best_value,
            "final_model": final_results
        }

    def export_onnx(self, output_dir: str = "./exports") -> str:
        """导出 ONNX 模型"""
        return self.model.export(
            format="onnx",
            half=True,  # FP16 量化
            simplify=True,
            opset=13
        )
```

### 2.2 知识蒸馏

```python
# src/distiller.py
from ultralytics import YOLO
import torch
import torch.nn as nn

class YOLODistiller:
    """YOLO 知识蒸馏器"""

    def __init__(self, teacher_path: str, student_size: str = "n"):
        self.teacher = YOLO(teacher_path)
        self.student = YOLO(f"yolov10{student_size}.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def distill(
        self,
        data_yaml: str,
        epochs: int = 100,
        alpha: float = 0.7,
        temperature: float = 4.0
    ):
        """
        知识蒸馏训练

        alpha: 损失权重平衡 (alpha * hard + (1-alpha) * soft)
        temperature: 知识蒸馏温度
        """

        # 配置蒸馏损失
        distill_config = {
            "alpha": alpha,
            "temperature": temperature,
            "feature_distill": True,  # 特征级蒸馏
            "response_distill": True,  # 响应级蒸馏
        }

        # 训练学生模型
        results = self.student.train(
            data=data_yaml,
            epochs=epochs,
            distill=distill_config,
            teacher=self.teacher,
            device=self.device
        )

        return results

    def export_distilled(self, output_path: str) -> str:
        """导出蒸馏后的模型"""
        return self.student.export(
            format="onnx",
            half=True,
            simplify=True
        )
```

---

## 3. 部署模块 (Deployer)

### 3.1 边缘部署器

```python
# src/deployer.py
import subprocess
import paramiko
from pathlib import Path

class EdgeDeployer:
    """边缘设备部署器"""

    def __init__(self, device_config: dict):
        self.device_type = device_config.get("type", "jetson_nano")
        self.device_ip = device_config.get("ip")
        self.device_user = device_config.get("user", "nvidia")
        self.ssh_key = device_config.get("ssh_key")

    def deploy(self, model_path: str, port: int = 8000) -> bool:
        """部署模型到边缘设备"""

        # 1. 转换模型格式
        onnx_path = self._convert_to_onnx(model_path)

        # 2. 传输模型
        self._upload_model(onnx_path)

        # 3. 启动推理服务
        self._start_inference_service(port)

        return True

    def _convert_to_onnx(self, model_path: str) -> str:
        """转换为 ONNX 格式"""
        from ultralytics import YOLO

        model = YOLO(model_path)
        output = model.export(format="onnx", half=True, simplify=True)

        return output

    def _upload_model(self, local_path: str):
        """通过 SSH 上传模型"""
        with paramiko.SSHClient() as ssh:
            ssh.connect(
                self.device_ip,
                username=self.device_user,
                key_filename=self.ssh_key
            )

            sftp = ssh.open_sftp()
            sftp.put(local_path, f"/home/{self.device_user}/model.onnx")
            sftp.close()

    def _start_inference_service(self, port: int):
        """启动推理服务"""

        # 使用 usls 启动服务
        cmd = [
            "usls", "serve",
            "--model", "/home/nvidia/model.onnx",
            "--port", str(port)
        ]

        with paramiko.SSHClient() as ssh:
            ssh.connect(
                self.device_ip,
                username=self.device_user,
                key_filename=self.ssh_key
            )

            # 启动服务 (后台运行)
            ssh.exec_command(f"{' '.join(cmd)} &")

    def test_inference(self, test_image: str) -> dict:
        """测试推理"""

        import requests

        with open(test_image, "rb") as f:
            response = requests.post(
                f"http://{self.device_ip}:8000/predict",
                files={"image": f}
            )

        return response.json()
```

### 3.2 GitHub Actions CI/CD 配置

```yaml
# .github/workflows/edge-deploy.yml
name: Deploy to Edge Device

on:
  workflow_dispatch:
    inputs:
      model_path:
        description: 'Model path to deploy'
        required: true
        default: 'runs/detect/train/weights/best.pt'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup SSH
        uses: webfactory/ssh-agent@v0.8.0
        with:
          ssh-private-key: ${{ secrets.SSH_PRIVATE_KEY }}

      - name: Convert to ONNX
        run: |
          python -c "
          from ultralytics import YOLO
          model = YOLO('${{ github.event.inputs.model_path }}')
          model.export(format='onnx', half=True)
          "

      - name: Deploy to Jetson
        run: |
          scp runs/detect/train/weights/best.onnx ${{ secrets.JETSON_USER }}@${{ secrets.JETSON_IP }}:/home/nvidia/

          ssh ${{ secrets.JETSON_USER }}@${{ secrets.JETSON_IP }} \
            "usls serve --model /home/nvidia/best.onnx --port 8000"
```

---

## 4. Agent 编排模块

### 4.1 CrewAI 编排器

```python
# src/orchestrator.py
from crewai import Agent, Task, Crew, Process
from langchain.tools import tool

class AutoTrainingOrchestrator:
    """自动化训练编排器"""

    def __init__(self):
        self.data_agent = self._create_data_agent()
        self.training_agent = self._create_training_agent()
        self.deployment_agent = self._create_deployment_agent()

    def _create_data_agent(self) -> Agent:
        """创建数据生成 Agent"""
        return Agent(
            role="Data Scientist",
            goal="Generate high-quality training data efficiently",
            backstory="""
                You are an expert in synthetic data generation.
                You know how to use diffusion models to create
                training data for object detection tasks.
            """,
            tools=[
                generate_synthetic_data,
                validate_data_quality,
                augment_real_data
            ],
            verbose=True
        )

    def _create_training_agent(self) -> Agent:
        """创建训练 Agent"""
        return Agent(
            role="ML Engineer",
            goal="Train optimal YOLO model with best performance",
            backstory="""
                You are an expert in YOLO training and hyperparameter
                optimization. You know how to use Optuna for HPO
                and knowledge distillation for model compression.
            """,
            tools=[
                train_yolo,
                run_hpo,
                perform_distillation,
                evaluate_model
            ],
            verbose=True
        )

    def _create_deployment_agent(self) -> Agent:
        """创建部署 Agent"""
        return Agent(
            role="DevOps Engineer",
            goal="Deploy models to edge devices reliably",
            backstory="""
                You are an expert in edge deployment.
                You know how to convert models to ONNX and
                deploy them using usls runtime.
            """,
            tools=[
                convert_onnx,
                deploy_to_device,
                test_inference
            ],
            verbose=True
        )

    def run_full_pipeline(self, task_description: str) -> dict:
        """运行完整训练流水线"""

        # 1. 数据生成任务
        data_task = Task(
            description=f"""
                Generate training data for: {task_description}
                Create at least 1000 synthetic images with auto-labeling.
            """,
            agent=self.data_agent,
            expected_output="Generated dataset with annotations"
        )

        # 2. 模型训练任务
        training_task = Task(
            description="""
                Train YOLO model with:
                - Use Optuna for 20 trials of HPO
                - Knowledge distillation from yolov10m to yolov10n
                - Export to ONNX format
            """,
            agent=self.training_agent,
            expected_output="Trained model with evaluation metrics",
            context=[data_task]
        )

        # 3. 部署任务
        deployment_task = Task(
            description="""
                Deploy the trained model to Jetson Nano:
                - Convert to ONNX with FP16
                - Deploy using usls
                - Test inference on sample images
            """,
            agent=self.deployment_agent,
            expected_output="Deployed model with test results",
            context=[training_task]
        )

        # 4. 创建 Crew
        crew = Crew(
            agents=[self.data_agent, self.training_agent, self.deployment_agent],
            tasks=[data_task, training_task, deployment_task],
            process=Process.sequential,  # 顺序执行
            verbose=True
        )

        # 5. 执行
        result = crew.kickoff()

        return {
            "status": "completed",
            "result": result
        }
```

### 4.2 工具定义

```python
# src/tools.py
from crewai.tools import tool
import subprocess

@tool("generate_synthetic_data")
def generate_synthetic_data(num_images: int, class_prompts: dict) -> str:
    """Generate synthetic training data using diffusion models"""
    # 实现见 data_generator.py
    pass

@tool("train_yolo")
def train_yolo(data_yaml: str, model_size: str = "n", epochs: int = 100) -> str:
    """Train YOLO model"""
    # 实现见 trainer.py
    pass

@tool("run_hpo")
def run_hpo(data_yaml: str, trials: int = 20) -> dict:
    """Run hyperparameter optimization with Optuna"""
    pass

@tool("perform_distillation")
def perform_distillation(teacher_path: str, student_path: str, data_yaml: str) -> str:
    """Perform knowledge distillation"""
    pass

@tool("convert_onnx")
def convert_onnx(model_path: str, half: bool = True) -> str:
    """Convert model to ONNX format"""
    pass

@tool("deploy_to_device")
def deploy_to_device(model_path: str, device_ip: str, device_user: str) -> str:
    """Deploy model to edge device"""
    pass
```

---

## 5. API 服务扩展

### 5.1 扩展 server.py

```python
# server_v2.py (扩展版本)
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="YOLO Auto-Training API")

# ========== Data Generation ==========

class DataGenerationRequest(BaseModel):
    num_images: int
    class_prompts: dict
    output_dir: str = "./data/synthetic"

@app.post("/api/v1/data/generate")
async def generate_data(req: DataGenerationRequest, background_tasks: BackgroundTasks):
    task_id = generate_task_id()

    # 启动后台任务
    background_tasks.add_task(
        run_data_generation,
        task_id,
        req
    )

    return {"task_id": task_id, "status": "started"}

@app.get("/api/v1/data/{task_id}")
async def get_data_status(task_id: str):
    return get_task_status(task_id)

# ========== Distillation ==========

class DistillationRequest(BaseModel):
    teacher_model: str
    student_size: str = "n"
    data_yaml: str

@app.post("/api/v1/distill/run")
async def run_distillation(req: DistillationRequest, background_tasks: BackgroundTasks):
    task_id = generate_task_id()

    background_tasks.add_task(
        run_distillation_task,
        task_id,
        req
    )

    return {"task_id": task_id, "status": "started"}

# ========== Deployment ==========

class DeploymentRequest(BaseModel):
    model_path: str
    device_ip: str
    device_user: str = "nvidia"

@app.post("/api/v1/deploy/run")
async def run_deployment(req: DeploymentRequest, background_tasks: BackgroundTasks):
    task_id = generate_task_id()

    background_tasks.add_task(
        run_deployment_task,
        task_id,
        req
    )

    return {"task_id": task_id, "status": "started"}

# ========== Agent Execution ==========

class AgentExecutionRequest(BaseModel):
    task_description: str
    max_iterations: int = 10

@app.post("/api/v1/agent/execute")
async def execute_agent(req: AgentExecutionRequest, background_tasks: BackgroundTasks):
    task_id = generate_task_id()

    # 启动 CrewAI 执行
    background_tasks.add_task(
        run_agent_pipeline,
        task_id,
        req.task_description
    )

    return {"task_id": task_id, "status": "started"}
```

---

## 6. 配置示例

### 6.1 环境配置

```bash
# .env
# LLM Provider
OPENAI_API_KEY=sk-...
QWEN_API_KEY=sk-...

# ComfyUI
COMFYUI_HOST=localhost:8188

# Edge Device
JETSON_IP=<YOUR_EDGE_DEVICE_IP>
JETSON_USER=nvidia
SSH_KEY_PATH=~/.ssh/jetson_rsa

# Training
DEFAULT_DATA_YAML=./data/coco.yaml
DEFAULT_MODEL_SIZE=n
```

### 6.2 数据集配置

```yaml
# data/custom.yaml
path: ./data/custom
train: images/train
val: images/val

nc: 3
names:
  0: person
  1: car
  2: dog
```

---

*文档版本: 1.0*
*最后更新: 2026-03-11*
