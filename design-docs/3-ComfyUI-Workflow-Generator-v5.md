# ComfyUI 工作流自动生成模块详细设计

**版本**: 8.0
**所属**: 1+5 设计方案
**新增功能**: AI Agent 自动生成 ComfyUI 工作流

---

## 1. 模块职责

| 职责 | 描述 |
|------|------|
| 任务分析 | 分析生成任务类型 |
| 节点选择 | 根据任务选择合适的节点 |
| 工作流生成 | 自动生成 ComfyUI 工作流 JSON |
| 执行管理 | 调用 ComfyUI API 执行 |

---

## 2. 专家建议

> "Generate ComfyUI workflows from natural language descriptions using LLMs"
> — ComfyUI-WorkflowGenerator GitHub

> "ComfyUI-Copilot provides comprehensive support for workflow building"
> — AIDC-AI GitHub

**核心原则**：
1. **任务分类** - 根据任务类型选择工作流
2. **节点组合** - 合理组合节点实现功能
3. **参数优化** - 根据需求调整参数

---

## 3. 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│              ComfyUI Workflow Generator Module                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Task Classifier                                 │  │
│  │         (人物/物体/场景/工业/医疗)                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Node Selector                                   │  │
│  │    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │  │
│  │    │SDXL    │ │Control │ │Face    │ │Canny   │        │  │
│  │    │        │ │Net     │ │Detailer│ │        │        │  │
│  │    └────────┘ └────────┘ └────────┘ └────────┘        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Workflow Builder                               │  │
│  │         (构建节点连接，生成 JSON)                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              ComfyUI Executor                              │  │
│  │         (调用 API 执行，返回结果)                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心组件

### 4.1 任务分类器

```python
# src/comfy_workflow/task_classifier.py
from enum import Enum
from typing import Dict, List

class TaskType(Enum):
    """任务类型"""
    PERSON = "person"           # 人物
    OBJECT = "object"         # 物体检测
    SCENE = "scene"           # 场景
    INDUSTRIAL = "industrial" # 工业
    MEDICAL = "medical"       # 医疗
    PRODUCT = "product"       # 产品

class TaskClassifier:
    """任务分类器"""

    # 任务类型关键词映射
    TASK_KEYWORDS = {
        TaskType.PERSON: ["person", "human", "face", "people", "portrait"],
        TaskType.OBJECT: ["object", "car", "dog", "cat", "item"],
        TaskType.SCENE: ["scene", "building", "street", "landscape"],
        TaskType.INDUSTRIAL: ["industrial", "manufacturing", "defect", "product", "part"],
        TaskType.MEDICAL: ["medical", "xray", "ct", "disease", "health"],
        TaskType.PRODUCT: ["product", "shopping", "retail", "checkout"],
    }

    # 任务类型对应的节点配置
    TASK_NODE_CONFIG = {
        TaskType.PERSON: {
            "model": "SDXL",
            "refiner": "SDXLRefiner",
            "preprocess": ["LoadImage"],
            "postprocess": ["FaceDetailer", "SaveImage"],
        },
        TaskType.OBJECT: {
            "model": "SDXL",
            "refiner": None,
            "preprocess": ["LoadImage"],
            "postprocess": ["SaveImage"],
        },
        TaskType.SCENE: {
            "model": "SDXL",
            "refiner": "SDXLRefiner",
            "preprocess": ["LoadImage", "ControlNet"],
            "postprocess": ["SaveImage"],
        },
        TaskType.INDUSTRIAL: {
            "model": "SDXL",
            "refiner": None,
            "preprocess": ["LoadImage"],
            "postprocess": ["SaveImage"],
            "quality": "high",
        },
        TaskType.MEDICAL: {
            "model": "SDXL",
            "refiner": None,
            "preprocess": ["LoadImage"],
            "postprocess": ["SaveImage"],
            "quality": "medical",
        },
        TaskType.PRODUCT: {
            "model": "SDXL",
            "refiner": "SDXLRefiner",
            "preprocess": ["LoadImage"],
            "postprocess": ["SaveImage"],
            "quality": "product",
        },
    }

    def classify(self, task_description: str) -> TaskType:
        """
        根据任务描述分类

        Args:
            task_description: 任务描述

        Returns:
            TaskType: 任务类型
        """
        text = task_description.lower()

        # 匹配关键词
        for task_type, keywords in self.TASK_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return task_type

        # 默认物体检测
        return TaskType.OBJECT

    def get_node_config(self, task_type: TaskType) -> Dict:
        """获取任务类型对应的节点配置"""
        return self.TASK_NODE_CONFIG.get(task_type, self.TASK_NODE_CONFIG[TaskType.OBJECT])
```

### 4.2 工作流构建器

```python
# src/comfy_workflow/workflow_builder.py
from typing import Dict, List, Optional

class WorkflowBuilder:
    """ComfyUI 工作流构建器"""

    # 基础节点模板
    NODE_TEMPLATES = {
        "LoadImage": {
            "class_type": "LoadImage",
            "inputs": {"image": ""}
        },
        "CLIPTextEncode": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["clip", 0]}
        },
        "SDXL": {
            "class_type": "SDXLEmbedder",
            "inputs": {
                "width": 1024,
                "height": 1024,
                "positive": ["positive", 0],
                "negative": ["negative", 0],
                "seed": 42
            }
        },
        "SaveImage": {
            "class_type": "SaveImage",
            "inputs": {"images": ["vae", 0], "filename_prefix": "synthetic"}
        },
    }

    def build(
        self,
        task_type: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        seed: int = None
    ) -> Dict:
        """
        构建工作流 JSON

        Args:
            task_type: 任务类型
            prompt: 正向提示词
            negative_prompt: 负向提示词
            width: 图像宽度
            height: 图像高度
            seed: 随机种子

        Returns:
            ComfyUI 工作流 JSON
        """
        workflow = {}
        node_id = 1

        # 1. Load Image (如果需要预处理)
        load_image = {
            str(node_id): {
                "class_type": "LoadImage",
                "inputs": {"image": ""}
            }
        }
        workflow.update(load_image)
        load_image_id = str(node_id)
        node_id += 1

        # 2. CLIP Text Encode (正向)
        clip_positive = {
            str(node_id): {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["4", 0]}
            }
        }
        workflow.update(clip_positive)
        clip_positive_id = str(node_id)
        node_id += 1

        # 3. CLIP Text Encode (负向)
        clip_negative = {
            str(node_id): {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt, "clip": ["4", 0]}
            }
        }
        workflow.update(clip_negative)
        clip_negative_id = str(node_id)
        node_id += 1

        # 4. SDXL Model
        sdxl_model = {
            str(node_id): {
                "class_type": "SDXLEmbedder",
                "inputs": {
                    "width": width,
                    "height": height,
                    "positive": [clip_positive_id, 0],
                    "negative": [clip_negative_id, 0],
                    "seed": seed or 42
                }
            }
        }
        workflow.update(sdxl_model)
        sdxl_model_id = str(node_id)
        node_id += 1

        # 5. VAE Decode (如果需要)
        vae_decode = {
            str(node_id): {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": [sdxl_model_id, 0],
                    "vae": ["5", 0]
                }
            }
        }
        workflow.update(vae_decode)
        vae_decode_id = str(node_id)
        node_id += 1

        # 6. Save Image
        save_image = {
            str(node_id): {
                "class_type": "SaveImage",
                "inputs": {
                    "images": [vae_decode_id, 0],
                    "filename_prefix": f"synthetic_{task_type}"
                }
            }
        }
        workflow.update(save_image)

        return workflow

    def build_with_controlnet(
        self,
        prompt: str,
        controlnet_image: str,
        controlnet_type: str = "canny",
        strength: float = 1.0
    ) -> Dict:
        """构建带 ControlNet 的工作流"""
        # 实现带 ControlNet 的工作流
        workflow = self.build("scene", prompt)
        # 添加 ControlNet 节点
        # ...

        return workflow
```

### 4.3 ComfyUI 执行器

```python
# src/comfy_workflow/executor.py
import aiohttp
import asyncio
from typing import Dict, Optional

class ComfyUIExecutor:
    """ComfyUI 执行器"""

    def __init__(self, host: str = "localhost:8188"):
        self.host = host
        self.base_url = f"http://{host}"
        self.timeout = aiohttp.ClientTimeout(total=300)

    async def execute(
        self,
        workflow: Dict,
        callback_url: Optional[str] = None
    ) -> str:
        """
        执行工作流

        Args:
            workflow: 工作流 JSON
            callback_url: 回调 URL (可选)

        Returns:
            prompt_id: 任务 ID
        """
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # 提交工作流
            payload = {"prompt": workflow}
            if callback_url:
                payload["callback_url"] = callback_url

            async with session.post(
                f"{self.base_url}/prompt",
                json=payload
            ) as resp:
                result = await resp.json()
                return result["prompt_id"]

    async def wait_for_completion(
        self,
        prompt_id: str,
        poll_interval: int = 2
    ) -> Dict:
        """
        等待工作流完成

        Args:
            prompt_id: 任务 ID
            poll_interval: 轮询间隔 (秒)

        Returns:
            执行结果
        """
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.get(
                    f"{self.base_url}/history/{prompt_id}"
                ) as resp:
                    data = await resp.json()

                    if prompt_id in data:
                        status = data[prompt_id].get("status", {})

                        if status.get("completed"):
                            return data[prompt_id]

                        if status.get("errored"):
                            raise RuntimeError(f"Execution failed: {status.get('error_msg')}")

                await asyncio.sleep(poll_interval)

    async def get_output_images(
        self,
        prompt_id: str
    ) -> List[str]:
        """获取输出图像列表"""
        result = await self.wait_for_completion(prompt_id)
        outputs = result.get("outputs", {})

        images = []
        for node_id, node_data in outputs.items():
            if "images" in node_data:
                for img in node_data["images"]:
                    images.append(img["filename"])

        return images
```

### 4.4 LLM 工作流生成器

```python
# src/comfy_workflow/llm_workflow_generator.py
from typing import Dict
import json

class LLMWorkflowGenerator:
    """使用 LLM 生成工作流"""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def generate(
        self,
        task_description: str,
        class_prompts: Dict[str, str]
    ) -> Dict:
        """
        使用 LLM 生成工作流配置

        Args:
            task_description: 任务描述
            class_prompts: 类别提示词

        Returns:
            工作流配置
        """
        # 1. 分析任务
        prompt = self._build_analysis_prompt(task_description, class_prompts)

        # 2. 调用 LLM
        response = await self.llm.chat(prompt)

        # 3. 解析响应
        return self._parse_response(response)

    def _build_analysis_prompt(
        self,
        task_description: str,
        class_prompts: Dict[str, str]
    ) -> str:
        """构建分析提示词"""
        return f"""
你是一个 ComfyUI 工作流生成专家。

任务描述: {task_description}

类别提示词:
{json.dumps(class_prompts, indent=2)}

请根据以上信息，生成 ComfyUI 工作流配置：

返回 JSON 格式：
{{
    "task_type": "person|object|scene|industrial|medical|product",
    "positive_prompt_template": "...",
    "negative_prompt_template": "...",
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "cfg_scale": 7.0,
    "sampler": "euler_ancestral",
    "controlnet": null 或 {{"type": "canny", "strength": 0.8}}
}}

只返回 JSON，不要其他内容。
"""

    def _parse_response(self, response: str) -> Dict:
        """解析 LLM 响应"""
        # 提取 JSON
        try:
            # 尝试直接解析
            return json.loads(response)
        except:
            # 尝试提取代码块
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())

        raise ValueError("Failed to parse LLM response")
```

---

## 5. 数据格式

### 5.1 工作流生成请求

```python
{
    "task_description": "检测工业零件缺陷",
    "class_prompts": {
        "defect": "A photo of industrial product defect",
        "normal": "A photo of normal industrial product"
    },
    "num_images": 100,
    "width": 1024,
    "height": 1024,
    "quality": "high"
}
```

### 5.2 工作流响应

```python
{
    "workflow_id": "wf_abc123",
    "status": "completed",
    "images": [
        "synthetic_defect_001.png",
        "synthetic_defect_002.png"
    ],
    "total_generated": 100,
    "time_elapsed": 300
}
```

---

## 6. 预定义工作流模板

### 6.1 基础物体检测

```python
BASIC_OBJECT_WORKFLOW = {
    "1": {"class_type": "LoadImage", "inputs": {"image": ""}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 0]}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 0]}},
    "4": {"class_type": "SDXL", "inputs": {
        "width": 1024, "height": 1024,
        "positive": ["2", 0], "negative": ["3", 0],
        "seed": 42
    }},
    "5": {"class_type": "VAEDecode", "inputs": {
        "samples": ["4", 0], "vae": ["4", 1]
    }},
    "6": {"class_type": "SaveImage", "inputs": {
        "images": ["5", 0], "filename_prefix": "synthetic"
    }}
}
```

### 6.2 带 ControlNet

```python
CONTROLNET_WORKFLOW = {
    # ... 基础节点
    "10": {"class_type": "LoadImage", "inputs": {"image": ""}},
    "11": {"class_type": "CannyEdgePreprocessor", "inputs": {"image": ["10", 0]}},
    "12": {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["4", 0],
        "control_net": ["13", 0],
        "image": ["11", 0]
    }},
    # ...
}
```

---

## 7. 专家审核要点

| 审核项 | 状态 | 说明 |
|--------|------|------|
| 任务分类 | ✅ | 支持多种任务类型 |
| 节点选择 | ✅ | 根据任务类型选择 |
| LLM 生成 | ✅ | 支持自然语言生成 |
| API 执行 | ✅ | 异步执行 + 等待完成 |

---

## 8. 依赖

```python
dependencies = [
    "aiohttp>=3.9.0",
    "Pillow>=10.0.0",
]
```

---

## 9. 与其他模块的集成

```
Dataset Discovery ──► Workflow Generator ──► VLM Labeler ──► Quality Filter
      │                    │
      │                    ▼
      │              ComfyUI API
      │                    │
      ▼                    ▼
用户输入              生成的图像
```

---

*文档版本: 8.0*
*新增功能: ComfyUI 工作流自动生成*
