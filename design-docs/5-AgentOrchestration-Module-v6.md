# Agent 编排模块详细设计

**版本**: 6.0
**所属**: 1+5 设计方案
**更新**: 增强 Agent 决策规则 + 基于 CrewAI 官方最佳实践

---

## 1. 模块职责

| 职责 | 描述 |
|------|------|
| 数据集发现 | 搜索和评估合适的数据集 |
| 多 Agent 协调 | CrewAI 角色化 Agent |
| 决策逻辑 | 增强的决策规则（5-6 条）|
| Human-in-the-Loop | 关键节点人工确认 |

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
│  │  │ 规则: 3 条 │ │ 规则: 2 条 │ │ 规则: 5 条 ││   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘│   │
│  │          │                │                │           │    │
│  │          └────────────────┼────────────────┘           │    │
│  │                           ▼                            │    │
│  │                  ┌──────────────┐                    │    │
│  │                  │ Deployment   │                    │    │
│  │                  │   Agent     │                    │    │
│  │                  │ 规则: 3 条 │                    │    │
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

### 3.1 数据集发现 Agent

```python
class DatasetDiscoveryAgent:
    """数据集发现 Agent - 增强决策规则"""

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

                Your decision rules:
                1. If relevance score > 0.8 → select dataset directly
                2. If 0.5 < score < 0.8 → include with warning
                3. If score < 0.5 → reject and trigger synthetic generation

                Always prioritize real-world data over synthetic data.
            """,
            verbose=True,
            allow_delegation=False,
        )
```

### 3.2 数据生成 Agent

```python
class DataGeneratorAgent:
    """数据生成 Agent - 增强决策规则"""

    @staticmethod
    def create():
        return Agent(
            role="Data Engineer",
            goal="Generate high-quality synthetic data using ComfyUI workflows",
            backstory="""
                You are an expert in synthetic data generation.

                Your decision rules:
                1. If synthetic ratio > 30% → stop generating, use discovered data
                2. If CLIP relevance score < 0.25 → filter out low-quality images
                3. If generation fails → fallback to manual labeling

                Always prefer quality over quantity.
            """,
            verbose=True,
            allow_delegation=False,
        )
```

### 3.3 训练 Agent

```python
class TrainingAgent:
    """训练 Agent - 增强决策规则"""

    @staticmethod
    def create():
        return Agent(
            role="ML Engineer",
            goal="Train YOLO11 model with optimal performance",
            backstory="""
                You are an expert in YOLO11 training.

                Your decision rules:
                1. If dataset < 1000 images → use aggressive data augmentation
                2. If mAP50 < 0.5 after HPO → try larger model
                3. If edge deployment → use YOLO11n (nano)
                4. If server deployment → use YOLO11m or YOLO11l
                5. If training time > 10 hours → enable aggressive early stopping

                Always balance accuracy and inference speed.
            """,
            verbose=True,
            allow_delegation=False,
        )
```

### 3.4 部署 Agent

```python
class DeploymentAgent:
    """部署 Agent - 增强决策规则"""

    @staticmethod
    def create():
        return Agent(
            role="DevOps Engineer",
            goal="Deploy model to edge device reliably",
            backstory="""
                You are an expert in edge deployment.

                Your decision rules:
                1. If FPS < 20 → optimize model or reduce input size
                2. If device memory < 2GB → use INT8 quantization
                3. If deployment fails → rollback to previous version

                Prioritize reliability over performance.
            """,
            verbose=True,
            allow_delegation=False,
        )
```

---

## 4. 层级化编排（基于 CrewAI 官方最佳实践）

```python
class HierarchicalOrchestrator:
    """层级化编排器 - 基于 CrewAI Process.hierarchical"""

    def __init__(self):
        # 1. 创建工作 Agent
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

        # 2. 创建管理器 Agent - 必须指定
        self.manager = Agent(
            role="Project Manager",
            goal="Coordinate the team to deliver the best results efficiently",
            backstory="""
                You are an experienced project manager specializing in AI/ML projects.
                You coordinate specialized experts to achieve project goals.
                You delegate tasks appropriately and review outputs critically.
                You ensure quality while maintaining efficient workflow.
            """,
            allow_delegation=True,
            verbose=True
        )

    def run(self, task_description: str) -> dict:
        """运行层级化流程"""

        data_task = Task(
            description=f"Find and prepare datasets for: {task_description}",
            agent=self.data_expert,
            expected_output="Curated dataset list with quality scores"
        )

        training_task = Task(
            description="Train YOLO11 model with optimal performance using HPO",
            agent=self.training_expert,
            expected_output="Trained model with mAP metrics",
            context=[data_task]
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
                self.manager,
                self.data_expert,
                self.training_expert,
                self.deployment_expert
            ],
            tasks=[data_task, training_task, deployment_task],
            process=Process.hierarchical,  # 层级化流程
            manager_agent=self.manager,       # 必须指定管理器
            verbose=True
        )

        return crew.kickoff()
```

---

## 5. Agent 决策规则对比

| Agent | v5 规则数 | v6 规则数 | v6 规则 |
|-------|-----------|-----------|---------|
| Discovery | 1 | 3 | score>0.8选, 0.5-0.8警告, <0.5拒绝 |
| Generator | 1 | 2 | >30%停止, CLIP<0.25过滤 |
| Training | 2 | 5 | 数据增强, mAP<0.5换模型, 边缘/服务器选择 |
| Deployment | 1 | 3 | FPS<20优化, 内存<2GB用INT8, 失败回滚 |

---

## 6. 参考来源

- [CrewAI Hierarchical Process](https://docs.crewai.com/en/concepts/processes)

---

*文档版本: 6.0*
*更新: 增强 Agent 决策规则*
