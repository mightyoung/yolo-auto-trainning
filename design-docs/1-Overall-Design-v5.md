# AI 驱动自动化 YOLO 训练与部署系统 - 完整设计方案

**版本**: 8.0
**日期**: 2026-03-11
**状态**: 已修复关键问题 + 速率限制 + 安全加固

---

## 一、系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI Agent Orchestration Layer                           │
│                    (CrewAI 纯 Crew 模式 + 数据发现 Agent)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Dataset   │  │   Data      │  │  Training   │  │ Deployment │    │
│  │  Discovery │  │  Generator  │  │  Agent      │  │  Agent     │    │
│  │  Agent     │  │  Agent      │  │             │  │            │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
    ┌───────────────────────────────────┼───────────────────────────────────┐
    │                                   ▼                                   │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │                      REST API Layer (FastAPI)                    │  │
    │  │  /data/discover  /data/generate  /train/*  /deploy/*  /agent/*│  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │                                   │                                   │
    ▼                                   ▼                                   │
┌─────────────┐              ┌─────────────────────┐              ┌──────────┐
│   Dataset   │              │   Training Core     │              │ Deploy   │
│  Discovery  │              │                     │              │ Core     │
├─────────────┤              ├─────────────────────┤              ├──────────┤
│ Roboflow  │◄────────────►│  YOLO11            │◄────────────►│ ONNX     │
│ Kaggle    │              │  + Optuna HPO      │              │ usls     │
│ HuggingFace              │                     │              │          │
│             │              │  Teacher: YOLO11m  │              │          │
│ ComfyUI    │              │  Student: YOLO11n  │              │          │
│ Workflow   │              │                     │              │          │
│ Generator  │              │                     │              │          │
└─────────────┘              └─────────────────────┘              └──────────┘
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │  Celery Workers    │
                              │  + Redis           │
                              └─────────────────────┘
```

---

## 二、核心新增功能

### 2.1 数据集发现 (Dataset Discovery)

**功能**: AI Agent 根据场景描述自动搜索合适的数据集

| 数据源 | 描述 |
|--------|------|
| Roboflow | 25万+ 数据集，支持搜索和筛选 |
| Kaggle | 数十万数据集，搜索能力强大 |
| HuggingFace | 多模态数据集丰富 |
| Open Images | Google 大规模标注数据集 |

### 2.2 ComfyUI 工作流自动生成

**功能**: AI Agent 根据任务描述自动生成 ComfyUI 工作流

| 工具 | 描述 |
|------|------|
| ComfyUI-LLM-API | 官方 API 封装 |
| comfyui-workflow-generator | AST 代码生成 |
| ComfyUI-Copilot | AI 工作流助手 |

---

## 三、设计原则

| 专家视角 | 建议 | 来源 |
|---------|------|------|
| **Andrej Karpathy** | "Data quality > Data quantity" | 公开演讲 |
| **Gartner** | "By 2030, synthetic data will surpass real data" | 2025 预测 |
| **Ultralytics** | "Use YOLO11 for balance of speed and accuracy" | YOLO 官方 |
| **Roboflow** | "Dataset quality is the moat" | Roboflow 博客 |

---

## 四、模块设计文档索引

| 文档 | 描述 | 版本 |
|------|------|------|
| `1-Overall-Design-v5.md` | 本文档 - 整体架构 | v5.0 |
| `2-DataDiscovery-Module-v5.md` | **新增** 数据集发现模块 | v5.0 |
| `3-DataGeneration-Module-v5.md` | 数据生成模块 (ComfyUI) | v5.0 |
| `4-Training-Module-v5.md` | 训练模块 - YOLO11 | v5.0 |
| `5-Deployment-Module-v5.md` | 部署模块 | v5.0 |
| `6-AgentOrchestration-Module-v5.md` | Agent 编排模块 | v5.0 |
| `7-APIService-Module-v5.md` | API 服务模块 | v5.0 |

---

## 五、技术选型汇总

| 层级 | 组件 | 版本/规格 |
|------|------|----------|
| **数据集发现** | Roboflow API + Kaggle API + HuggingFace | latest |
| **工作流生成** | ComfyUI-LLM-API + comfyui-workflow-generator | latest |
| **YOLO 模型** | YOLO11 (m/n) | latest |
| **Agent 编排** | CrewAI | latest |
| **任务队列** | Celery + Redis | latest |
| **模型导出** | ONNX | opset 13 |
| **边缘推理** | usls | latest |

---

## 六、关键设计决策

### 6.1 数据集发现策略

```
任务描述 ──► 关键词提取 ──► 多源搜索 ──► 相关性排序 ──► 自动下载
                              │
                              ▼
                        1. Roboflow (首选)
                        2. Kaggle
                        3. HuggingFace
                        4. Open Images (兜底)
```

### 6.2 ComfyUI 工作流生成策略

```
任务描述 ──► 任务分类 ──► 节点选择 ──► 参数配置 ──► 工作流 JSON
                    │
                    ▼
           ┌────────┬────────┬────────┐
           │人物    │物体    │场景    │
           ├────────┼────────┼────────┤
           │SDXL    │SDXL    │SDXL    │
           │FaceDet │Canny   │ControlNet│
           └────────┴────────┴────────┘
```

### 6.3 合成数据策略

| 原则 | 说明 |
|------|------|
| 合成 ≤ 30% | 真实数据为主 |
| CLIP 相关性 | 过滤不相关图像 |
| 人工抽检 | 10% 样本文检 |
| 域适应 | 确保与目标场景匹配 |

---

## 七、工作流程

### 7.1 完整自动化流程

```
1. 用户输入任务描述
   e.g., "检测工业零件缺陷"

2. Dataset Discovery Agent
   ├── 搜索 Roboflow/Kaggle/HuggingFace
   ├── 评估数据集相关性
   └── 返回候选数据集列表

3. Data Generator Agent (ComfyUI)
   ├── 分析任务类型
   ├── 生成/选择 ComfyUI 工作流
   ├── 调用 SDXL 生成图像
   ├── VLM 自动标注
   └── 质量过滤

4. Training Agent
   ├── Sanity Check (30 epochs, imgsz=1280)
   ├── 正式训练 (100 epochs)
   └── HPO (可选, 10 trials)

5. Deployment Agent
   ├── 导出 ONNX (FP16)
   ├── 部署到 Jetson Nano
   └── 测试推理

6. Human-in-the-Loop
   ├── 训练前确认
   └── 部署前确认
```

---

## 八、风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| 数据集搜索无结果 | 流程中断 | 自动切换到合成数据生成 |
| ComfyUI 工作流失败 | 图像生成失败 | 备用工作流模板 |
| 合成数据质量差 | 模型泛化差 | 限制 30% 比例 |
| Agent 决策错误 | 资源浪费 | Human-in-Loop |

---

*文档版本: 8.0*
*新增功能: 数据集发现 + ComfyUI 工作流自动生成*
