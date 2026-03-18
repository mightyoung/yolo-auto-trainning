# 设计文档修复报告 (v5.1)

## 概述

本报告记录了 v5 设计文档中发现的问题及修复方案，基于以下最佳实践来源：

- Ultralytics YOLO 官方文档
- NVIDIA Jetson 部署指南
- Roboflow API 最佳实践
- CrewAI 官方指南

---

## 修复清单

### 1. 训练模块 (4-Training-Module-v5.md)

#### 问题 1: HPO 参数空间错误
- **原问题**: `weight_decay: [0.05, 0.3]` - 范围过大
- **修复**: `weight_decay: [0.0001, 0.001]` - 基于 Ultralytics 默认值 0.0005
- **依据**: [Ultralytics 超参文档](https://docs.ultralytics.com/guides/hyperparameter-tuning/)

#### 问题 2: Sanity Check 阈值不合理
- **原问题**: `mAP50 >= 0.4` 即认为可行
- **修复**: `mAP50 >= 0.3` 最低标准，并提供更合理的建议
- **依据**: 30 epochs 不能完全反映最终性能

#### 问题 3: 知识蒸馏实现错误
- **原问题**: 使用 Ultralytics 不支持的蒸馏方式
- **修复**: 改为伪标签 (pseudo-labeling) 方式
- **依据**: YOLO 不支持传统知识蒸馏

---

### 2. 部署模块 (5-Deployment-Module-v5.md)

#### 问题 1: FP16 量化代码错误
- **原问题**: `convert_float16_to_float(model)` - 方向错误
- **修复**: `convert_float_to_float16(model)` - FP32 → FP16
- **依据**: [ONNX Runtime 文档](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)

#### 问题 2: TensorRT 在 Jetson Nano 上构建
- **原问题**: 尝试在 Nano 上构建 TensorRT 引擎
- **修复**: 添加设备检查，Nano 使用 ONNX Runtime
- **依据**: Nano 内存有限，不适合构建 TensorRT

#### 问题 3: SSH 密码明文传输
- **原问题**: 密码在代码/请求中明文传输
- **修复**: 改为环境变量 + Docker 部署
- **依据**: OWASP 安全最佳实践

---

### 3. 数据发现模块 (2-DataDiscovery-Module-v5.md)

#### 问题 1: API 无速率限制
- **原问题**: 无限制调用可能导致账户被封
- **修复**: 添加 RateLimiter 类，限制调用频率
- **依据**: [Roboflow 速率限制](https://inference.roboflow.com/workflows/blocks/rate_limiter/)

| 数据源 | 速率限制 |
|--------|----------|
| Roboflow | 10 次/分钟 |
| Kaggle | 10 次/分钟 |
| HuggingFace | 30 次/分钟 |

#### 问题 2: 无缓存机制
- **原问题**: 重复搜索浪费 API 调用
- **修复**: 添加 DatasetCache 类，24小时 TTL

---

## 技术依据汇总

| 来源 | 引用内容 |
|------|----------|
| Ultralytics | weight_decay 默认 0.0005 |
| NVIDIA Jetson | TensorRT 需目标设备构建 |
| ONNX Runtime | 使用 convert_float_to_float16 |
| Roboflow | 需要速率限制 |
| OWASP | 禁止明文存储密码 |

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 5.0 | 2026-03-11 | 初始版本 |
| 5.1 | 2026-03-11 | 修复关键问题 |
