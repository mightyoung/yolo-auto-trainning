# Refactoring Progress - 2026-03-31 (Updated)

## 目标
将yolo-auto-trainning从大型单体架构重构为模块化微服务架构。

## 当前状态

### 测试状态
- **通过**: 98 tests (core tests)
- **失败**: 52 tests (due to import/package structure issues)
- **Collection errors**: 3 test files when running full suite

### 核心测试通过 (98 tests)
- tests/state_machine/ - 13 tests
- tests/contract/ - 9 tests
- tests/unit/test_training_runner.py - 36 tests
- tests/unit/test_exceptions.py - 11 tests
- tests/unit/test_training_config.py - 16 tests
- tests/unit/test_pipeline.py - 19 tests (when run individually)
- 其他...

## 已完成的重构

### 1. Task Storage统一 ✅ (Earlier session)
**变更**: routes.py之前有64行重复的Task Storage定义
**结果**: 从`store/task_store.py`导入统一访问器

### 2. Models统一 ✅ (Earlier session)
**变更**: routes.py之前有103行重复的Request/Response模型定义
**结果**: 从`models/`导入所有模型

### 3. DynamicTrainingManager简化 ✅ (Earlier session)
**变更**: 从226行实现简化为28行包装器
**结果**: 委托给`plateau_manager.py`的PlateauManager

### 4. validator.py stub创建 ✅
**变更**: 创建`src/deployment/validator.py`作为stub
**原因**: training-api的exporter导入`from src.deployment.validator`
**结果**: 满足import需求，test_pipeline.py可以运行

### 5. src/__init__.py创建
**变更**: 创建空的`src/__init__.py`文件
**原因**: 试图修复collection errors
**结果**: 未解决collection errors，但无害

## 架构障碍

### 循环导入问题 🔴
```
routes.py ──────→ store.task_store
    ↑                    │
    │                    ↓
    └────── gateway.py ←┘
```
**原因**:
- `routes.py` 导入 `store.task_store`
- `store.task_store` 导入 `gateway` (get_redis_client)
- `gateway` 导入 `routes` (router)

**影响**: 阻止进一步拆分routes.py到route_handlers/

### Package命名问题 🔴
**问题**: `business-api` (hyphen) 不能作为Python模块名
**影响**: 测试导入`business_api` (underscore) 失败

**涉及文件** (19个):
- tests/unit/test_agents.py
- tests/unit/test_authentication.py
- tests/unit/test_mlflow.py
- 等

### sys.path hack 🔴
**代码** (routes.py lines 29-36):
```python
_training_api_src_root = Path(__file__).parent.parent
if str(_training_api_src_root) not in sys.path:
    sys.path.insert(0, str(_training_api_src_root))
```
**原因**: 防止`from src.training.runner`解析到legacy `src/training/runner.py`
**影响**: 丑陋但功能正常

### pytest Collection冲突 🔴
**问题**: 当运行完整测试套件时,3个测试文件collection失败
- tests/unit/test_data_discovery.py
- tests/unit/test_model_export.py
- tests/unit/test_pipeline.py

**现象**:
- 单独运行: 成功
- 作为完整套件一部分: ImportError

**原因**: conftest.py路径设置冲突

## Phase 1.4 & 1.5 互锁

Phase 1.4 (拆分routes.py) 和 Phase 1.5 (修复sys.path) 互相阻塞：
1. **拆分需要sys.path hack工作** - 提取的模块需要相同的hack
2. **sys.path hack需要架构重构** - 当前结构导致必须使用绝对`src.xxx`导入

## 建议

鉴于当前架构复杂度,建议:
1. **接受当前状态** - 98个核心测试通过
2. **标记Phase 1.4/1.5为pending** - 需要架构重构
3. **关注其他改进** - 如文档、代码质量

## 统计数据

| 指标 | 状态 |
|------|------|
| 核心测试通过 | 98 ✅ |
| 失败测试 | 52 (import issues) |
| Collection errors | 3 test files |
| 循环导入 | 存在 |
| sys.path hack | 存在 |
| Package命名问题 | 存在 |
