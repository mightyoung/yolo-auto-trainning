# AI 驱动自动化 YOLO 训练与部署系统 - 完整设计方案

**版本**: 6.0
**日期**: 2026-03-13
**状态**: 基于官方最佳实践修正 + 所有问题已修复

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
│ Kaggle    │              │  + Ray Tune HPO    │              │ usls     │
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
| `1-Overall-Design-v6.md` | 本文档 - 整体架构 | v6.0 |
| `2-DataDiscovery-Module-v6.md` | 数据集发现模块 | v6.0 |
| `3-ComfyUI-Workflow-Generator-v6.md` | 数据生成模块 (ComfyUI) | v6.0 |
| `4-Training-Module-v6.md` | 训练模块 - YOLO11 | v6.0 |
| `5-Deployment-Module-v6.md` | 部署模块 | v6.0 |
| `6-AgentOrchestration-Module-v6.md` | Agent 编排模块 | v6.0 |
| `7-APIService-Module-v6.md` | API 服务模块 | v6.0 |

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

## 七、训练配置（基于 Ultralytics 官方最佳实践）

### 7.1 官方默认超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr0` | **0.01** | 初始学习率 (SGD) |
| `lrf` | 0.01 | 最终学习率因子 |
| `momentum` | 0.937 | SGD 动量 |
| `weight_decay` | 0.0005 | 权重衰减 |
| `box` | **7.5** | 边界框损失权重 |
| `cls` | 0.5 | 分类损失权重 |
| `dfl` | 1.5 | DFL 损失权重 |
| `hsv_h` | 0.015 | 色调增强 |
| `hsv_s` | 0.7 | 饱和度增强 |
| `hsv_v` | 0.4 | 明度增强 |
| `fliplr` | **0.5** | 水平翻转概率 |

> ⚠️ **v5 版本错误**: 文档中 lr0=0.001, box=0.05, fliplr=0.0 均为错误值
> ✅ **v6 版本修正**: 已按官方默认值修正

### 7.2 训练流程配置

| 阶段 | Epochs | Image Size | 说明 |
|------|--------|------------|------|
| Sanity Check | **10** | 640 | 快速验证训练可行性 |
| HPO 优化 | **50** | 1280 | Ray Tune 超参搜索 |
| 最终训练 | **300** | 1280 | 标准训练轮数 |

> ⚠️ **v5 版本问题**: epochs 配置混乱 (10/30/100/300/600-800)
> ✅ **v6 版本修正**: 统一为 10 → 50 → 300 的标准流程

### 7.3 HPO 参数空间（修正后）

```python
# 优化器参数 - 6 个核心参数
PARAM_SPACE_OPTIMIZER = {
    "lr0": [0.001, 0.01],        # 官方默认 0.01
    "lrf": [0.01, 1.0],
    "momentum": [0.6, 0.98],      # 官方默认 0.937
    "weight_decay": [0.0001, 0.001],  # 官方默认 0.0005
    "box": [5.0, 10.0],           # 官方默认 7.5
    "cls": [0.3, 1.0],            # 官方默认 0.5
}

# 数据增强 - 保持固定
AUGMENTATION_FIXED = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "fliplr": 0.5,                # 官方默认 0.5
    "mosaic": 1.0,
    "mixup": 0.0,
}

# Ray Tune 试验次数
N_TRIALS = 50  # 原 25 次不足
```

> ⚠️ **v5 版本问题**: 优化 11 个参数（包括 hsv_*, fliplr），只做 25 trials
> ✅ **v6 版本修正**: 分离优化器参数和数据增强，试验次数增加到 50

---

## 八、工作流程

### 8.1 完整自动化流程

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
   ├── Sanity Check (10 epochs, imgsz=640)
   ├── HPO 优化 (50 trials, 50 epochs)
   └── 最终训练 (300 epochs)

5. Deployment Agent
   ├── 导出 ONNX (FP16)
   ├── 部署到 Jetson Nano
   └── 测试推理

6. Human-in-the-Loop
   ├── 训练前确认
   └── 部署前确认
```

---

## 九、风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| 数据集搜索无结果 | 流程中断 | 自动切换到合成数据生成 |
| ComfyUI 工作流失败 | 图像生成失败 | 备用工作流模板 |
| 合成数据质量差 | 模型泛化差 | 限制 30% 比例 |
| Agent 决策错误 | 资源浪费 | Human-in-Loop |

---

## 十、参考来源

| 来源 | 验证内容 |
|------|----------|
| [Ultralytics 官方超参数](https://docs.ultralytics.com/usage/cfg/) | lr0=0.01, box=7.5, fliplr=0.5 |
| [YOLO Ray Tune 集成](https://docs.ultralytics.com/integrations/ray-tune/) | model.tune(use_ray=True) |
| [知识蒸馏实现](https://github.com/danielsyahputra/yolo-distiller) | teacher 参数 + distillation_loss |
| [CrewAI 层级化流程](https://docs.crewai.com/en/concepts/processes) | Process.hierarchical |
| [FastAPI CORS](https://fastapi.tiangolo.com/tutorial/cors/) | 生产环境指定域名 |

---

*文档版本: 6.0*
*更新日期: 2026-03-13*
*基于审查报告修正所有问题*
