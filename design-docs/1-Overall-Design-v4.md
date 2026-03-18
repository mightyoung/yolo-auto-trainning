# AI 驱动自动化 YOLO 训练与部署系统 - 完整设计方案

**版本**: 4.0
**日期**: 2026-03-11
**状态**: 基于专家审核 + YOLO 版本选择指南修订

---

## 一、系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI Agent Orchestration Layer                          │
│                           (CrewAI 纯 Crew 模式)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │
│  │  Data       │  │  Training   │  │  Deployment │                     │
│  │  Agent      │  │  Agent      │  │  Agent      │                     │
│  │  (决策)     │  │  (决策)     │  │  (决策)     │                     │
│  └─────────────┘  └─────────────┘  └─────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
    ┌───────────────────────────────────┼───────────────────────────────────┐
    │                                   ▼                                   │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │                      REST API Layer (FastAPI)                   │  │
    │  │  /data/*    /train/*    /distill/*    /deploy/*    /agent/*   │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │                                   │                                   │
    ▼                                   ▼                                   │
┌─────────────┐              ┌─────────────────────┐              ┌──────────┐
│   Data      │              │   Training Core     │              │ Deploy   │
│   Generation│              │                     │              │ Core     │
├─────────────┤              ├─────────────────────┤              ├──────────┤
│ ComfyUI    │◄────────────►│  YOLO11            │◄────────────►│ ONNX     │
│ (SDXL)     │              │  + Optuna HPO      │              │ usls     │
│             │              │                     │              │          │
│ Qwen2-VL   │              │  Teacher: YOLO11m  │              │          │
│ (标注+验证) │              │  Student: YOLO11n  │              │          │
└─────────────┘              └─────────────────────┘              └──────────┘
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │  Celery Workers    │
                              │  (任务执行器)      │
                              └─────────────────────┘
```

---

## 二、核心设计决策

### 2.1 YOLO 版本选择（关键修订）

根据 `YOLO_MODELS_GUIDE.md` 和业务需求：

| 用途 | 模型 | 参数量 | mAP@0.5 | 理由 |
|------|------|--------|---------|------|
| **Teacher** | YOLO11m | 20.1M | 65.5% | 精度速度平衡最佳 |
| **Student** | YOLO11n | 2.6M | 54.5% | 边缘部署首选 |
| 备选 Teacher | YOLO11l | 25.4M | 67.6% | 更高精度需求时 |

**不选择其他版本的理由**：
- **YOLO26**: 生态建设中，不适合生产环境
- **YOLO12**: 计算量高，训练不稳定
- **YOLOv10**: 已不是最优选择

### 2.2 图像尺寸修订

根据 Ultralytics 社区最佳实践：
- **Sanity Run**: 20-30 epochs, imgsz=1280
- **正式训练**: imgsz=1280 (精度优先) 或 imgsz=640 (速度优先)

### 2.3 HPO 策略修订

- 只调 2 个参数：`lr0` + `weight_decay`
- 减少 trials 到 10 次
- 使用 log=True 连续搜索

---

## 三、设计原则（来自专家审核 + YOLO 官方指南）

| 专家视角 | 建议 | 来源 |
|---------|------|------|
| **Ultralytics 官方** | "imgsz 1280 for best accuracy, 20-30 epoch sanity run" | YOLO 社区 |
| **Andrej Karpathy** | "Data quality > Data quantity" | 公开演讲 |
| **NVIDIA Jetson 团队** | "Use FP16 quantization for 50% latency reduction" | Jetson 文档 |
| **CrewAI 团队** | "Role-based agents with clear responsibilities" | CrewAI 文档 |

---

## 四、模块设计文档索引

| 文档 | 描述 | 版本 |
|------|------|------|
| `1-Overall-Design-v4.md` | 本文档 - 整体架构与设计原则 | v4.0 |
| `2-DataGeneration-Module-v4.md` | 数据生成模块 (修订) | v4.0 |
| `3-Training-Module-v4.md` | 训练模块 - **YOLO11** | v4.0 |
| `4-Deployment-Module-v4.md` | 部署模块 (修订) | v4.0 |
| `5-AgentOrchestration-Module-v4.md` | Agent 编排模块 (修订) | v4.0 |
| `6-APIService-Module-v4.md` | API 服务模块 (修订) | v4.0 |

---

## 五、技术选型汇总

| 层级 | 组件 | 版本/规格 | 说明 |
|------|------|----------|------|
| **YOLO 模型** | YOLO11 | m/n/s/m/l/x | Teacher: YOLO11m, Student: YOLO11n |
| **Agent 编排** | CrewAI | latest | 纯 Crew 模式 |
| **任务队列** | Celery + Redis | latest | 持久化任务状态 |
| **图像生成** | ComfyUI + SDXL | latest | API 队列模式 |
| **VLM 标注** | Qwen2-VL-Max | latest | 二次验证 + 人工抽检 |
| **超参搜索** | Optuna | latest | 有限搜索空间 |
| **模型导出** | ONNX | opset 13 | FP16 量化 |
| **边缘推理** | usls / ONNX Runtime | latest | 高性能推理引擎 |

---

## 六、关键改进点 (v3 → v4)

| # | v3 问题 | v4 改进 | 依据 |
|---|---------|---------|------|
| 1 | 使用 YOLOv10 | 改用 **YOLO11m/n** | YOLO_MODELS_GUIDE.md |
| 2 | imgsz=640 | **imgsz=1280** (sanity run) | Ultralytics 社区 |
| 3 | HPO 搜索空间过大 | **只调 2 参数** | Andrej Karpathy |
| 4 | Agent 规则过多 | **精简到 2 条** | CrewAI 最佳实践 |
| 5 | 缺少监控 | **添加 Prometheus** | 生产需求 |
| 6 | 无断点续训 | **Checkpoint 持久化** | 资源保护 |

---

## 七、硬件需求

| 环境 | 规格 | 用途 |
|------|------|------|
| 训练服务器 | NVIDIA L20 (48GB) | YOLO11 训练 + 数据生成 |
| Redis 服务器 | 2+ CPU, 4GB+ RAM | 任务状态存储 + Celery Broker |
| 边缘设备 | Jetson Nano (4GB) / Orin Nano | YOLO11n 推理 |

---

## 八、风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| 合成数据质量不佳 | 模型泛化差 | 限制 30% 比例 + 人工抽检验证 |
| ONNX 算子不兼容 | 部署失败 | 使用 simplify=True + 基础算子 |
| Agent 决策错误 | 训练效果差 | 明确决策边界 + 阈值规则 |
| Jetson Nano 性能不足 | 推理慢 | 使用 YOLO11n + FP16 |
| 任务状态丢失 | 服务不可恢复 | Redis 持久化 |
| YOLO11n 精度不足 | 检测效果差 | 使用知识蒸馏提升 |

---

## 九、YOLO 版本选择决策树

```
任务需求是什么？
│
├─ 边缘部署 (Jetson Nano)
│   └─ 使用 YOLO11n (2.6M, ~20-25 FPS)
│
├─ 服务器部署
│   ├─ 追求精度 → YOLO11m/l
│   └─ 追求速度 → YOLO11s
│
├─ 知识蒸馏
│   ├─ Teacher: YOLO11m/l
│   └─ Student: YOLO11n
│
└─ 生产环境
    └─ 使用 YOLO11 (不是 YOLO26!)
```

---

*审核状态: 基于 YOLO_MODELS_GUIDE.md 和专家审核*
*文档版本: 4.0*
