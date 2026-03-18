# Agent 编排模块详细设计

**版本**: 8.0
**所属**: 1+5 设计方案
**新增功能**: 数据集发现 Agent

---

## 1. 模块职责

| 职责 | 描述 |
|------|------|
| 数据集发现 | 搜索和评估合适的数据集 |
| 多 Agent 协调 | CrewAI 角色化 Agent |
| 决策逻辑 | 精简的决策规则（最多 2 条）|
| Human-in-Loop | 关键节点人工确认 |

---

## 2. 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                  Agent Orchestration Module                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                    Crew Orchestrator                      │    │
│  │                                                         │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐│   │
│  │  │Discovery   │ │ Data Gen    │ │Training    ││   │
│  │  │  Agent     │ │   Agent    │ │  Agent     ││   │
│  │  │             │ │            │ │            ││   │
│  │  │ 规则: 1 条 │ │ 规则: 1 条 │ │ 规则: 2 条 ││   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘│   │
│  │          │                │                │           │    │
│  │          └────────────────┼────────────────┘           │    │
│  │                           ▼                            │    │
│  │                  ┌──────────────┐                    │    │
│  │                  │ Deployment   │                    │    │
│  │                  │   Agent     │                    │    │
│  │                  │ 规则: 1 条 │                    │    │
│  │                  └──────────────┘                    │    │
│  └────────────────────────────────────────────────────────┘    │
│                            │                                   │
│                            ▼                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Human-in-the-Loop                          │    │
│  │      数据集确认 │ 训练前确认 │ 部署前确认           │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Agent 定义

### 3.1 数据集发现 Agent (新增)

```python
# src/agent/agents.py

class DatasetDiscoveryAgent:
    """数据集发现 Agent"""

    @staticmethod
    def create():
        return Agent(
            role="Dataset Curator",
            goal="Find and select the most relevant datasets for the task",
            backstory="""
                You are an expert in dataset discovery and curation.
                You know how to search and evaluate datasets from:
                - Roboflow (250k+ datasets)
                - Kaggle (hundreds of thousands of datasets)
                - HuggingFace (multimodal datasets)
                - Open Images

                Your ONLY decision rule:
                1. If relevant dataset found with score > 0.8 → select it

                You use tools to search and download datasets.
            """,
            verbose=True,
            allow_delegation=False,
        )
```

### 3.2 数据生成 Agent

```python
class DataGeneratorAgent:
    """数据生成 Agent"""

    @staticmethod
    def create():
        return Agent(
            role="Data Engineer",
            goal="Generate high-quality synthetic data using ComfyUI workflows",
            backstory="""
                You are an expert in synthetic data generation.

                Your ONLY decision rule:
                1. If synthetic ratio > 30% → stop generating

                You use tools to generate data with ComfyUI.
            """,
            verbose=True,
            allow_delegation=False,
        )
```

### 3.3 训练 Agent

```python
class TrainingAgent:
    """训练 Agent"""

    @staticmethod
    def create():
        return Agent(
            role="ML Engineer",
            goal="Train YOLO11 model with optimal performance",
            backstory="""
                You are an expert in YOLO11 training.

                Your decision rules (max 2):
                1. If mAP50 < 0.5 → run HPO
                2. If edge deployment → use YOLO11n

                You use tools to train, you don't train directly.
            """,
            verbose=True,
            allow_delegation=False,
        )
```

### 3.4 部署 Agent

```python
class DeploymentAgent:
    """部署 Agent"""

    @staticmethod
    def create():
        return Agent(
            role="DevOps Engineer",
            goal="Deploy model to edge device reliably",
            backstory="""
                You are an expert in edge deployment.

                Your ONLY decision rule:
                1. If FPS < 20 → report issue

                You use tools to deploy.
            """,
            verbose=True,
            allow_delegation=False,
        )
```

---

## 4. 任务编排

```python
# src/agent/orchestrator.py

class AutoTrainingOrchestrator:
    """自动化训练编排器 - 4 个 Agent"""

    def __init__(self):
        self.discovery_agent = DatasetDiscoveryAgent.create()
        self.data_agent = DataGeneratorAgent.create()
        self.training_agent = TrainingAgent.create()
        self.deployment_agent = DeploymentAgent.create()

    def run_full_pipeline(
        self,
        task_description: str,
        device_ip: str = None
    ) -> dict:
        """
        运行完整流水线 - 4 步
        """

        # 1. 数据集发现任务
        discovery_task = Task(
            description=f"""
                Task: Find relevant datasets for: {task_description}

                Steps:
                1. Analyze the task description
                2. Search Roboflow, Kaggle, HuggingFace
                3. Score datasets by relevance
                4. Select top 3 datasets

                Output: List of candidate datasets with scores
            """,
            agent=self.discovery_agent,
            expected_output="Dataset list with relevance scores"
        )

        # 2. 数据生成任务
        generation_task = Task(
            description=f"""
                Task: Generate synthetic data for: {task_description}

                Steps:
                1. Analyze discovered datasets
                2. Identify missing classes
                3. Generate ComfyUI workflows
                4. Generate images with VLM auto-labeling
                5. Filter by CLIP relevance > 0.25

                Output: Generated dataset with quality report
            """,
            agent=self.data_agent,
            expected_output="Generated dataset ready for training",
            context=[discovery_task]
        )

        # 3. 训练任务
        training_task = Task(
            description=f"""
                Task: Train YOLO11 model for: {task_description}

                Steps:
                1. Merge discovered + synthetic data
                2. Sanity check (30 epochs, imgsz=1280)
                3. Train final model (100 epochs)
                4. Export to ONNX with FP16

                Output: Trained model with metrics
            """,
            agent=self.training_agent,
            expected_output="Trained YOLO11 model",
            context=[generation_task]
        )

        # 4. 部署任务
        deployment_task = Task(
            description=f"""
                Task: Deploy to edge device at: {device_ip}

                Steps:
                1. Convert to ONNX with FP16
                2. Deploy to Jetson Nano via SSH
                3. Test inference
                4. Report FPS

                Output: Deployed model with test results
            """,
            agent=self.deployment_agent,
            expected_output="Deployment confirmation",
            context=[training_task]
        )

        # 创建 Crew - 带 Human-in-the-Loop 支持
        crew = Crew(
            agents=[
                self.discovery_agent,
                self.data_agent,
                self.training_agent,
                self.deployment_agent
            ],
            tasks=[
                discovery_task,
                generation_task,
                training_task,
                deployment_task
            ],
            process=Process.sequential,
            verbose=True,
            human_in_the_loop=True,  # 启用 HITL 模式
            webhook_url=os.getenv("CREW_WEBHOOK_URL"),  # 回调通知
            step_callback=self._step_callback  # 步骤回调
        )

        return crew.kickoff()

    def _step_callback(self, step_output):
        """
        步骤回调 - 用于监控和干预

        基于 CrewAI 最佳实践:
        - 记录中间输出用于调试
        - 允许在关键节点进行人工干预
        """
        print(f"Step completed: {step_output}")
        return step_output

    def run_with_approval(
        self,
        task_description: str,
        approval_tasks: list = None
    ) -> dict:
        """
        带审批的工作流 - 仅在关键节点暂停等待人工确认

        适用场景:
        - 数据集选择
        - 模型部署
        """
        approval_tasks = approval_tasks or ["discovery_task", "deployment_task"]

        crew = Crew(
            agents=[
                self.discovery_agent,
                self.data_agent,
                self.training_agent,
                self.deployment_agent
            ],
            tasks=[
                discovery_task,
                generation_task,
                training_task,
                deployment_task
            ],
            process=Process.sequential,
            verbose=True,
            human_in_the_loop=True,
            approval_required=approval_tasks  # 需要审批的任务
        )

        return crew.kickoff()
```

---

## 4.5 层级化编排（基于 CrewAI 官方 Process.hierarchical）

```python
# src/agents/hierarchical_orchestrator.py
"""
层级化 Agent 编排器 - 基于 CrewAI 官方最佳实践

参考:
- https://docs.crewai.com/en/learn/hierarchical-process
- https://github.com/crewAIInc/crewAI/discussions/1220

核心设计:
- 使用管理器 Agent (Manager Agent) 协调工作 Agent
- 管理器负责任务分配和结果审核
- 工作 Agent 专注于具体任务执行
"""

from crewai import Agent, Crew, Task, Process
from crewai.tools import BaseTool


class HierarchicalOrchestrator:
    """层级化编排器 - 管理者 + 工作 Agent 模式

    与顺序流程的区别:
    - 顺序流程: 任务按固定顺序执行
    - 层级流程: 管理器 Agent 动态分配任务，可以并行处理
    """

    def __init__(self):
        # 1. 创建工作 Agent（专家角色）
        self.data_expert = Agent(
            role="Data Expert",
            goal="Find and prepare the best datasets efficiently",
            backstory="""
                You are an expert in dataset discovery and preparation.
                You have deep knowledge of Roboflow, Kaggle, and HuggingFace.
                You always prioritize data quality over quantity.
            """,
            tools=[search_dataset_tool, download_dataset_tool],
            verbose=True
        )

        self.training_expert = Agent(
            role="Training Expert",
            goal="Train optimal YOLO models with best performance",
            backstory="""
                You are an expert in YOLO training and hyperparameter optimization.
                You know how to leverage HPO and knowledge distillation effectively.
                You always aim for the best mAP with minimal compute.
            """,
            tools=[train_model_tool, hpo_tool],
            verbose=True
        )

        self.deployment_expert = Agent(
            role="Deployment Expert",
            goal="Deploy models to edge devices with optimal performance",
            backstory="""
                You are an expert in edge deployment and optimization.
                You specialize in TensorRT, ONNX optimization, and Jetson devices.
                You ensure models meet FPS requirements on target hardware.
            """,
            tools=[deploy_tool, monitor_tool],
            verbose=True
        )

        # 2. 创建管理器 Agent（关键！）
        self.manager = Agent(
            role="Project Manager",
            goal="Coordinate the team to deliver the best results efficiently",
            backstory="""
                You are an experienced project manager specializing in AI/ML projects.
                You coordinate specialized experts to achieve project goals.
                You delegate tasks appropriately and review outputs critically.
                You ensure quality while maintaining efficient workflow.
            """,
            allow_delegation=True,  # 允许委派任务
            verbose=True
        )

    def run(self, task_description: str) -> dict:
        """运行层级化流程"""

        # 定义任务
        data_task = Task(
            description=f"Find and prepare datasets for: {task_description}",
            agent=self.data_expert,
            expected_output="Curated dataset list with quality scores"
        )

        training_task = Task(
            description="Train YOLO11 model with optimal performance using HPO",
            agent=self.training_expert,
            expected_output="Trained model with mAP metrics",
            context=[data_task]  # 依赖数据任务
        )

        deployment_task = Task(
            description="Deploy to edge device with FPS test",
            agent=self.deployment_expert,
            expected_output="Deployed model with FPS test results",
            context=[training_task]
        )

        # 创建层级化 Crew - 关键设置
        crew = Crew(
            agents=[
                self.manager,           # 管理者负责协调
                self.data_expert,
                self.training_expert,
                self.deployment_expert
            ],
            tasks=[data_task, training_task, deployment_task],
            process=Process.hierarchical,  # 关键: 层级化流程
            manager_agent=self.manager,    # 指定管理器
            verbose=True
        )

        return crew.kickoff()

    def run_parallel(self, task_description: str) -> dict:
        """运行并行层级化流程 - 适合独立任务"""

        # 并行数据发现和数据生成
        data_task = Task(
            description=f"Discover datasets for: {task_description}",
            agent=self.data_expert,
            expected_output="Dataset list with relevance scores"
        )

        training_task = Task(
            description="Train model with optimal hyperparameters",
            agent=self.training_expert,
            expected_output="Model checkpoint with metrics"
        )

        # 并行执行
        crew = Crew(
            agents=[self.manager, self.data_expert, self.training_expert],
            tasks=[data_task, training_task],
            process=Process.hierarchical,
            manager_agent=self.manager,
            verbose=True
        )

        return crew.kickoff()


# 多 Crew 协作编排器
class MultiCrewOrchestrator:
    """多 Crew 协作编排器 - 适合复杂流水线

    适用于:
    - 大型项目需要多个独立团队
    - 需要并行执行多个子流水线
    - 不同阶段需要不同的 Agent 配置
    """

    def __init__(self):
        # 数据发现 Crew
        self.discovery_crew = self._create_discovery_crew()

        # 训练 Crew
        self.training_crew = self._create_training_crew()

        # 部署 Crew
        self.deployment_crew = self._create_deployment_crew()

    def _create_discovery_crew(self) -> Crew:
        """创建数据发现 Crew"""
        data_expert = Agent(
            role="Data Expert",
            goal="Find optimal datasets",
            backstory="Dataset discovery specialist",
            tools=[search_dataset_tool]
        )

        task = Task(
            description="Find datasets for the task",
            agent=data_expert,
            expected_output="Curated dataset list"
        )

        return Crew(agents=[data_expert], tasks=[task], process=Process.sequential)

    def _create_training_crew(self) -> Crew:
        """创建训练 Crew"""
        trainer = Agent(
            role="Trainer",
            goal="Train best model",
            backstory="YOLO training specialist",
            tools=[train_model_tool]
        )

        task = Task(
            description="Train YOLO model",
            agent=trainer,
            expected_output="Trained model"
        )

        return Crew(agents=[trainer], tasks=[task], process=Process.sequential)

    def _create_deployment_crew(self) -> Crew:
        """创建部署 Crew"""
        deployer = Agent(
            role="Deployer",
            goal="Deploy to edge",
            backstory="Edge deployment specialist",
            tools=[deploy_tool]
        )

        task = Task(
            description="Deploy model to device",
            agent=deployer,
            expected_output="Deployment confirmation"
        )

        return Crew(agents=[deployer], tasks=[task], process=Process.sequential)

    def run(self, task_description: str) -> dict:
        """运行完整流水线"""

        # 1. 数据发现
        discovery_result = self.discovery_crew.kickoff()

        # 2. 训练 (依赖发现结果)
        training_result = self.training_crew.kickoff(
            inputs={"datasets": discovery_result}
        )

        # 3. 部署 (依赖训练结果)
        deployment_result = self.deployment_crew.kickoff(
            inputs={"model": training_result}
        )

        return {
            "discovery": discovery_result,
            "training": training_result,
            "deployment": deployment_result
        }
```

---

## 5. 工作流

```
用户输入任务
     │
     ▼
┌────────────────┐
│  Discovery    │ ◄── 搜索数据集
│    Agent       │
└───────┬────────┘
        │ 发现数据集
        ▼
┌────────────────┐
│  Data Gen     │ ◄── 生成/补充数据
│    Agent       │
└───────┬────────┘
        │ 训练数据
        ▼
┌────────────────┐
│  Training     │ ◄── 训练 YOLO11
│    Agent       │
└───────┬────────┘
        │ 训练好的模型
        ▼
┌────────────────┐
│  Deployment   │ ◄── 部署到边缘
│    Agent       │
└───────┬────────┘
        │
        ▼
    部署完成
```

---

## 6. 专家审核要点

| 审核项 | 状态 | 说明 |
|--------|------|------|
| 4 个 Agent | ✅ | 发现/生成/训练/部署 |
| 规则精简 | ✅ | 最多 2 条规则 |
| Human-in-Loop | ✅ | 关键节点确认 |
| 顺序执行 | ✅ | 数据流清晰 |

---

## 7. 关键改进说明

### 改进: 新增 Discovery Agent
- **v4 错误**: 直接生成数据，没有利用现有数据集
- **v5 正确**: 先搜索合适的数据集，再决定是否需要生成
- **依据**: 优先利用现有数据，减少合成数据依赖

---

*文档版本: 8.0*
