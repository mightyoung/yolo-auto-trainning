# Agent 编排模块详细设计

**版本**: 3.0
**所属**: 1+5 设计方案
**审核状态**: 已基于业界最佳实践修订

---

## 1. 模块职责

| 职责 | 描述 |
|------|------|
| 多 Agent 协调 | CrewAI 角色化 Agent |
| 决策逻辑 | 明确的决策边界 + 阈值规则 |
| 任务编排 | 顺序/并行执行 |
| 工具暴露 | 为 Agent 提供执行工具 |

---

## 2. 专家建议（来自 CrewAI 官方文档）

> "Crafting Effective Agents: Define clear roles, set specific goals, and equip agents with appropriate tools"
> — [CrewAI 官方文档](https://docs.crewai.com/en/guides/agents/crafting-effective-agents)

> "Task performance: Agents with clear roles and goals execute tasks more effectively"
> — CrewAI Documentation

**核心原则**：
1. **明确的角色和目标** - 每个 Agent 有清晰定义的职责
2. **明确的决策边界** - 不让 Agent 随意决定，需要阈值规则
3. **工具即服务** - Agent 调用工具，不直接处理数据

---

## 3. 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                  Agent Orchestration Module                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    Crew Orchestrator                       │  │
│  │                                                              │  │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │  │
│  │  │ Data Agent   │   │Training Agent│   │Deploy Agent  │  │  │
│  │  │              │   │              │   │              │  │  │
│  │  │ Role: Data   │   │ Role: ML Eng │   │Role: DevOps  │  │  │
│  │  │ Goal: 明确   │   │ Goal: 明确   │   │Goal: 明确   │  │  │
│  │  │ 决策边界     │   │ 决策边界     │   │ 决策边界     │  │  │
│  │  └──────────────┘   └──────────────┘   └──────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                            │                                      │
│                            ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    Tool Layer                              │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────┐  │  │
│  │  │Data Gen   │ │Train/HPO  │ │Distill    │ │Deploy  │  │  │
│  │  │Tool       │ │Tool       │ │Tool       │ │Tool    │  │  │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心组件

### 4.1 Agent 定义（明确的决策边界）

```python
# src/agent/agents.py
from crewai import Agent
from crewai.tools import tool

class DataAgent:
    """数据 Agent - 负责数据相关决策"""

    @staticmethod
    def create():
        return Agent(
            role="Data Scientist",
            goal="Generate high-quality training data efficiently within budget constraints",
            backstory="""
                You are an expert in synthetic data generation for computer vision.
                You have 10 years of experience in data augmentation and generation.

                Your DECISION RULES (you MUST follow these exactly):
                1. If dataset has < 1000 images → generate more data
                2. If synthetic ratio > 30% → stop generating
                3. If CLIP relevance score < 0.25 → reject image
                4. If human review approval rate < 80% → regenerate

                You NEVER directly generate data - you use tools for that.
            """,
            verbose=True,
            allow_delegation=False,
        )


class TrainingAgent:
    """训练 Agent - 负责模型训练决策"""

    @staticmethod
    def create():
        return Agent(
            role="ML Engineer",
            goal="Train optimal YOLO model based on data quality and deployment requirements",
            backstory="""
                You are an expert in YOLO training and hyperparameter optimization.
                You have trained 100+ YOLO models for various computer vision tasks.

                Your DECISION RULES (you MUST follow these exactly):
                1. If dataset size < 1000 → use default parameters first
                2. If mAP50 < 0.5 after training → try HPO
                3. If deploying to edge → use nano size + distill if needed
                4. If mAP50 > 0.8 → stop optimizing, deploy

                You NEVER directly train models - you use tools for that.
                You let Optuna tune hyperparameters, not yourself.
            """,
            verbose=True,
            allow_delegation=False,
        )


class DeploymentAgent:
    """部署 Agent - 负责部署决策"""

    @staticmethod
    def create():
        return Agent(
            role="DevOps Engineer",
            goal="Deploy models to edge devices reliably with minimal latency",
            backstory="""
                You are an expert in edge deployment and model optimization.
                You have deployed 50+ models to NVIDIA Jetson devices.

                Your DECISION RULES (you MUST follow these exactly):
                1. Always use FP16 quantization
                2. Use simplify=True for ONNX export
                3. If FPS < 30 → use smaller model
                4. If deployment fails 3 times → rollback

                You NEVER directly deploy - you use tools for that.
            """,
            verbose=True,
            allow_delegation=False,
        )
```

### 4.2 工具定义（带有明确的输入验证）

```python
# src/agent/tools.py
from crewai.tools import tool
from typing import Dict, List, Optional
import json

# ============================================================================
# Data Generation Tools
# ============================================================================

@tool("generate_synthetic_data")
def generate_synthetic_data(
    num_images: int,
    class_prompts: Dict[str, str],
    output_dir: str = "./data/synthetic",
    quality_threshold: float = 0.7
) -> str:
    """
    Generate synthetic training data using diffusion models.

    Args:
        num_images: Number of images to generate (1-10000)
        class_prompts: {"class_name": "prompt template"}
        output_dir: Output directory
        quality_threshold: CLIP score threshold (0-1)

    Returns:
        Generation summary with stats
    """
    from src.data.generator import SyntheticDataGenerator

    # 输入验证
    if not 1 <= num_images <= 10000:
        raise ValueError("num_images must be between 1 and 10000")
    if not 0 <= quality_threshold <= 1:
        raise ValueError("quality_threshold must be between 0 and 1")

    generator = SyntheticDataGenerator()
    result = generator.generate(
        num_images=num_images,
        class_prompts=class_prompts,
        output_dir=output_dir,
        quality_threshold=quality_threshold
    )

    return json.dumps({
        "generated": len(result.images),
        "rejected": result.rejected,
        "avg_clip_score": sum(result.clip_scores) / len(result.clip_scores),
        "output_dir": output_dir
    })


@tool("evaluate_data_quality")
def evaluate_data_quality(data_yaml: str) -> str:
    """
    Evaluate dataset quality and provide recommendations.

    Returns:
        Quality report with recommendations
    """
    from src.data.quality_evaluator import DataQualityEvaluator

    evaluator = DataQualityEvaluator()
    report = evaluator.evaluate(data_yaml)

    return json.dumps(report)


# ============================================================================
# Training Tools
# ============================================================================

@tool("train_yolo_model")
def train_yolo_model(
    data_yaml: str,
    model_size: str = "n",
    epochs: int = 100
) -> str:
    """
    Train YOLO model with standard parameters.

    Args:
        data_yaml: Dataset config path
        model_size: n/s/m/l/x (must be one of these)
        epochs: Training epochs (1-1000)

    Returns:
        Training results summary
    """
    from src.train.yolo_trainer import YOLOTrainer

    # 输入验证
    if model_size not in ["n", "s", "m", "l", "x"]:
        raise ValueError("model_size must be one of: n, s, m, l, x")
    if not 1 <= epochs <= 1000:
        raise ValueError("epochs must be between 1 and 1000")

    trainer = YOLOTrainer(model_size=model_size)
    result = trainer.train(
        data_yaml=data_yaml,
        epochs=epochs
    )

    return json.dumps({
        "status": "completed",
        "model_path": result["model_path"],
        "mAP50": result.get("best_map", 0)
    })


@tool("run_hyperparameter_optimization")
def run_hyperparameter_optimization(
    data_yaml: str,
    model_size: str = "n",
    n_trials: int = 20
) -> str:
    """
    Run hyperparameter optimization using Optuna.

    Args:
        data_yaml: Dataset config
        model_size: Model size (n/s/m/l/x)
        n_trials: Number of trials (1-100)

    Returns:
        Best parameters and score
    """
    from src.train.hpo_optimizer import HPOOptimizer

    optimizer = HPOOptimizer(model_size=model_size)
    result = optimizer.optimize(
        data_yaml=data_yaml,
        n_trials=n_trials
    )

    return json.dumps({
        "best_params": result["best_params"],
        "best_score": result["best_trial_score"]
    })


@tool("perform_knowledge_distillation")
def perform_knowledge_distillation(
    teacher_model: str,
    student_size: str = "n",
    data_yaml: str
) -> str:
    """
    Perform knowledge distillation from teacher to student model.

    Args:
        teacher_model: Path to teacher model (.pt)
        student_size: Student model size (n/s/m/l/x)
        data_yaml: Dataset config

    Returns:
        Distillation results
    """
    from src.train.distiller import YOLODistiller

    distiller = YOLODistiller()
    result = distiller.distill(
        teacher_path=teacher_model,
        student_size=student_size,
        data_yaml=data_yaml
    )

    return json.dumps({
        "status": "completed",
        "student_model": result["student_model"]
    })


# ============================================================================
# Deployment Tools
# ============================================================================

@tool("convert_to_onnx")
def convert_to_onnx(
    model_path: str,
    half: bool = True
) -> str:
    """
    Convert model to ONNX format with FP16 quantization.

    Args:
        model_path: Path to .pt model
        half: Use FP16 quantization (default: True)

    Returns:
        ONNX model path
    """
    from src.deploy.converter import ModelConverter

    converter = ModelConverter(model_path)
    onnx_path = converter.to_onnx(half=half, simplify=True)

    return onnx_path


@tool("deploy_to_edge_device")
def deploy_to_edge_device(
    model_path: str,
    device_ip: str,
    device_user: str = "nvidia"
) -> str:
    """
    Deploy model to edge device (Jetson).

    Args:
        model_path: Path to ONNX model
        device_ip: Device IP address (format: xxx.xxx.xxx.xxx)
        device_user: SSH username

    Returns:
        Deployment status
    """
    import os
    from src.deploy.edge_deployer import EdgeDeployer

    # 从环境变量获取 SSH 密钥
    ssh_key_path = os.getenv("JETSON_SSH_KEY")
    if not ssh_key_path:
        raise ValueError("JETSON_SSH_KEY environment variable not set")

    deployer = EdgeDeployer({
        "ip": device_ip,
        "user": device_user,
        "ssh_key_path": ssh_key_path
    })

    deployer.connect()
    deployer.deploy_model(model_path)
    deployer.start_inference_service(model_path)
    deployer.close()

    return json.dumps({
        "status": "deployed",
        "endpoint": f"http://{device_ip}:8000/predict"
    })


@tool("test_inference")
def test_inference(
    device_ip: str,
    test_image: str
) -> str:
    """
    Test inference on deployed model.

    Args:
        device_ip: Device IP address
        test_image: Path to test image

    Returns:
        Inference results
    """
    import requests

    with open(test_image, "rb") as f:
        response = requests.post(
            f"http://{device_ip}:8000/predict",
            files={"image": f}
        )

    return response.json()
```

### 4.3 任务编排（明确的预期输出）

```python
# src/agent/orchestrator.py
from crewai import Agent, Task, Crew, Process
from .agents import DataAgent, TrainingAgent, DeploymentAgent
from .tools import *

class AutoTrainingOrchestrator:
    """自动化训练编排器"""

    def __init__(self):
        self.data_agent = DataAgent.create()
        self.training_agent = TrainingAgent.create()
        self.deployment_agent = DeploymentAgent.create()

    def run_full_pipeline(
        self,
        task_description: str,
        dataset_path: str = None,
        device_ip: str = None
    ) -> dict:
        """
        运行完整流水线 - 明确的决策边界
        """
        # 1. 数据生成/评估任务 - 明确的决策规则
        data_task = Task(
            description=f"""
                Task: {task_description}

                Evaluate and prepare dataset at {dataset_path}

                Your DECISION RULES:
                1. First evaluate current dataset quality
                2. If dataset has < 1000 images → generate synthetic data to reach 1000
                3. Keep synthetic/real ratio ≤ 30%
                4. If CLIP relevance < 0.25 → reject
                5. Submit 10% samples for human review

                Output: Dataset ready for training with metadata
            """,
            agent=self.data_agent,
            expected_output="Dataset quality report with: image_count, synthetic_ratio, quality_scores"
        )

        # 2. 训练任务 - 明确的决策规则
        training_task = Task(
            description=f"""
                Task: Train YOLO model for {task_description}

                Your DECISION RULES:
                1. First try training with default parameters (100 epochs)
                2. If mAP50 < 0.5 → run HPO (20 trials)
                3. If deploying to edge:
                   - Use YOLOv10-Nano
                   - Optionally distill from larger model
                4. Export to ONNX with FP16

                Output: Trained model with metrics
            """,
            agent=self.training_agent,
            expected_output="Training results with: model_path, mAP50, export_path",
            context=[data_task]
        )

        # 3. 部署任务 - 明确的决策规则
        deployment_task = Task(
            description=f"""
                Task: Deploy model to edge device at {device_ip}

                Your DECISION RULES:
                1. Always use ONNX + FP16
                2. Use simplify=True for ONNX export
                3. Verify inference works
                4. If FPS < 30 → report issue

                Output: Deployed model with test results
            """,
            agent=self.deployment_agent,
            expected_output="Deployment confirmation with: endpoint, health_status, test_results",
            context=[training_task]
        )

        # 4. 创建 Crew
        crew = Crew(
            agents=[
                self.data_agent,
                self.training_agent,
                self.deployment_agent
            ],
            tasks=[data_task, training_task, deployment_task],
            process=Process.sequential,
            verbose=True
        )

        # 5. 执行
        result = crew.kickoff()

        return {
            "status": "completed",
            "result": result
        }

    def run_data_only(self, task_description: str, dataset_path: str) -> dict:
        """仅运行数据相关任务"""
        task = Task(
            description=f"""
                Evaluate and enhance dataset at {dataset_path} for task: {task_description}

                Your DECISION RULES:
                1. If < 1000 images → generate more
                2. Keep synthetic ratio ≤ 30%
                3. Submit for human review
            """,
            agent=self.data_agent,
            expected_output="Data enhancement report with: current_count, actions_taken"
        )

        crew = Crew(agents=[self.data_agent], tasks=[task])
        return crew.kickoff()
```

---

## 5. 数据格式

### 5.1 Agent 执行请求

```python
# 完整流水线请求
{
    "task_description": "Detect cars and pedestrians on road",
    "dataset_path": "./data/road.yaml",
    "device_ip": "192.168.1.100"
}

# 仅数据请求
{
    "task_description": "Improve car detection",
    "dataset_path": "./data/cars.yaml"
}
```

### 5.2 Agent 执行响应

```python
{
    "status": "completed",
    "stages": {
        "data": {
            "status": "completed",
            "dataset_size": 1500,
            "synthetic_ratio": 0.25,
            "human_review_approval_rate": 0.85,
            "actions": ["generated_500_images", "filtered_by_clip"]
        },
        "training": {
            "status": "completed",
            "default_params_mAP50": 0.65,
            "hpo_run": False,
            "model_path": "./runs/train/weights/best.pt",
            "export_path": "./exports/yolov10n.onnx"
        },
        "deployment": {
            "status": "completed",
            "endpoint": "http://192.168.1.100:8000/predict",
            "test_results": {"detections": [...], "fps": 45}
        }
    }
}
```

---

## 6. 专家审核要点

| 审核项 | 状态 | 说明 |
|--------|------|------|
| 明确的决策规则 | ✅ | 每个 Agent 有 DECISION RULES |
| 决策边界限制 | ✅ | 具体的阈值和条件 |
| 工具输入验证 | ✅ | 参数范围验证 |
| 明确的预期输出 | ✅ | Task 包含 expected_output |
| 不混用编排框架 | ✅ | 仅使用 CrewAI |

---

## 7. 依赖

```python
dependencies = [
    "crewai>=0.50.0",
]
```

---

## 8. 关键改进说明 (v2 → v3)

### 改进 1: 明确的决策规则
- **v2 错误**: Agent 权限过大，可以随意决定
- **v3 正确**: 每个 Agent 有明确的 DECISION RULES
- **依据**: [CrewAI 官方最佳实践](https://docs.crewai.com/en/guides/agents/crafting-effective-agents)

### 改进 2: 输入验证
- **v2 错误**: 工具参数无验证
- **v3 正确**: 工具函数包含输入验证

### 改进 3: 明确的预期输出
- **v2 错误**: Task 没有明确预期输出
- **v3 正确**: 每个 Task 包含 expected_output

### 改进 4: 移除 LangGraph
- **v2 错误**: CrewAI + LangGraph 混用
- **v3 正确**: 仅使用 CrewAI

---

*审核状态: 已基于业界最佳实践修订*
