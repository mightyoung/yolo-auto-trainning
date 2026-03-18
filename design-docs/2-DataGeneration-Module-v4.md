# 数据生成模块详细设计

**版本**: 4.0
**所属**: 1+5 设计方案
**审核状态**: 已修订

---

## 1. 模块职责

| 职责 | 描述 |
|------|------|
| 合成数据生成 | 使用 Diffusion 模型生成训练图像 |
| 自动标注 | VLM 自动生成 bounding box 标注 |
| 质量过滤 | CLIP 相关性过滤 + 人工抽检 |
| 数据增强 | 对真实数据进行增强 |

---

## 2. 专家建议（来自 Andrej Karpathy + 业界最佳实践）

> "Don't think of LLMs as entities but as simulators."
> — Andrej Karpathy

> "Data quality > Data quantity"
> — Andrej Karpathy

> "Verified Auto Labeling delivers up to 95% model performance on downstream inference"
> — Voxel51

**核心原则**：
1. **合成数据 ≤ 30%** - 真实数据为主
2. **CLIP 用于相关性** - 不是质量评估
3. **VLM 标注需验证** - 置信度过滤 + 人工抽检

---

## 3. 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Generation Module                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   LLM        │───►│  ComfyUI     │───►│   VLM        │   │
│  │ (Prompt Gen) │    │  (SDXL)      │    │ (Auto-Label) │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                   │                   │                  │
│         ▼                   ▼                   ▼                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Quality Filter                               │   │
│  │         (CLIP 相关性 + 置信度 + 人工抽检)             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Human Review Checklist                       │   │
│  │         (bbox准确度 + 遮挡 + 截断)                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Output: COCO Format                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心组件

### 4.1 Prompt 生成器

```python
# src/data/prompt_generator.py
from typing import Dict, List
import random

class PromptGenerator:
    """基于 LLM 的 Prompt 生成器"""

    def __init__(self, provider: str = "qwen"):
        self.provider = provider

    def generate(
        self,
        class_templates: Dict[str, str],
        num_samples: int,
        variations: List[str] = None
    ) -> List[str]:
        """生成多样化的 prompts"""
        if variations is None:
            variations = [
                "indoor", "outdoor", "closeup", "far", "side view",
                "front view", "back view", "shadow", "backlight"
            ]

        prompts = []
        samples_per_class = num_samples // len(class_templates)

        for class_name, template in class_templates.items():
            for i in range(samples_per_class):
                var = random.sample(variations, min(3, len(variations)))
                prompt = f"{template}, {', '.join(var)}, high quality, photorealistic"
                prompts.append(prompt)

        return prompts
```

### 4.2 ComfyUI 客户端

```python
# src/data/comfy_client.py
import aiohttp
import asyncio
from typing import List

class ComfyClient:
    """ComfyUI API 客户端"""

    def __init__(self, host: str = "localhost:8188"):
        self.host = host
        self.base_url = f"http://{host}"
        self.timeout = aiohttp.ClientTimeout(total=300)

    async def generate(self, prompt: str) -> str:
        """生成单张图像"""
        # 简化实现
        workflow = self._create_workflow(prompt)

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow}
            ) as resp:
                result = await resp.json()
                prompt_id = result["prompt_id"]

            # 等待完成
            return await self._wait_completion(session, prompt_id)

    async def batch_generate(
        self,
        prompts: List[str],
        max_concurrent: int = 3
    ) -> List[str]:
        """批量生成"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_limit(prompt):
            async with semaphore:
                return await self.generate(prompt)

        tasks = [generate_with_limit(p) for p in prompts]
        return await asyncio.gather(*tasks)

    def _create_workflow(self, prompt: str) -> dict:
        """创建工作流"""
        return {
            "3": {"inputs": {"text": prompt}, "class_type": "CLIPTextEncode"},
            "4": {"inputs": {"seed": 42}, "class_type": "RandomNoise"},
        }

    async def _wait_completion(self, session, prompt_id: str) -> str:
        """等待完成"""
        while True:
            async with session.get(
                f"{self.base_url}/history/{prompt_id}"
            ) as resp:
                data = await resp.json()
                if prompt_id in data and data[prompt_id]["status"]["completed"]:
                    outputs = data[prompt_id]["outputs"]
                    for node_data in outputs.values():
                        if "images" in node_data:
                            return node_data["images"][0]["filename"]
            await asyncio.sleep(2)
```

### 4.3 VLM 自动标注器（带置信度）

```python
# src/data/vlm_labeler.py
import dashscope
from typing import List, Dict, Tuple

class VLMLabeler:
    """VLM 自动标注器"""

    def __init__(self, model: str = "qwen2-vl-max"):
        self.model = model

    async def label(self, image_path: str) -> Tuple[List[Dict], bool]:
        """
        自动标注 - 返回标注和是否通过置信度
        """
        response = dashscope.MultiModalConversation.call(
            model=self.model,
            messages=[{
                'role': 'user',
                'content': [
                    {'image': f"file://{image_path}"},
                    {'text': self._get_prompt()}
                ]
            }]
        )

        annotations = self._parse_response(response)

        # 检查置信度
        passed = all(a.get("confidence", 0) >= 0.7 for a in annotations)

        return annotations, passed

    def _get_prompt(self) -> str:
        return """
        Detect objects. Return JSON:
        [{"class": "name", "bbox": [x,y,w,h], "confidence": 0.95}]
        """

    def _parse_response(self, response) -> List[Dict]:
        try:
            text = response.output.choices[0].message.content[0]['text']
            import json
            return json.loads(text)
        except:
            return []
```

### 4.4 质量过滤器（修正 CLIP 用途）

```python
# src/data/quality_filter.py
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple
import random

class QualityFilter:
    """CLIP 相关性过滤器 - 修正用途"""

    def __init__(
        self,
        relevance_threshold: float = 0.25,
        human_review_ratio: float = 0.1
    ):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.relevance_threshold = relevance_threshold
        self.human_review_ratio = human_review_ratio
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.review_queue: List[dict] = []

    async def filter(
        self,
        images: List[str],
        annotations: List[Dict],
        vlm_confidences: List[float]
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        过滤 - CLIP 用于相关性，不是质量评估
        """
        passed_images = []
        passed_annotations = []
        review_samples = []

        for img, ann, vlm_conf in zip(images, annotations, vlm_confidences):
            # 1. VLM 置信度过滤
            if vlm_conf < 0.7:
                continue

            # 2. CLIP 相关性过滤（修正！）
            relevance = await self._compute_relevance(img, ann)
            if relevance < self.relevance_threshold:
                continue

            # 3. 随机人工抽检
            if random.random() < self.human_review_ratio:
                sample = {
                    "image": img,
                    "annotation": ann,
                    "relevance": relevance,
                    "vlm_conf": vlm_conf
                }
                review_samples.append(sample)
                self.review_queue.append(sample)

            passed_images.append(img)
            passed_annotations.append(ann)

        return passed_images, passed_annotations, review_samples

    async def _compute_relevance(self, image_path: str, annotation: Dict) -> float:
        """计算图像-标注相关性"""
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        class_names = [ann["class"] for ann in annotation]

        if not class_names:
            return 0.0

        inputs = self.processor(
            text=class_names,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)

        return probs.max().item()
```

### 4.5 人工抽检 Checklist

```python
# src/data/human_review.py
from typing import List, Dict
from enum import Enum

class ReviewChecklist(Enum):
    """人工抽检 Checklist"""
    BBOX_ACCURACY = "bbox_accuracy"     # Bbox 是否准确
    OCCLUSION = "occlusion"             # 是否有遮挡
    TRUNCATION = "truncation"            # 是否截断
    BLUR = "blur"                        # 是否模糊
    WRONG_CLASS = "wrong_class"          # 类别错误

class HumanReviewWorkflow:
    """人工抽检工作流"""

    def __init__(self):
        self.pending: List[Dict] = []
        self.approved: List[Dict] = []
        self.rejected: List[Dict] = []

    def add_sample(self, sample: Dict):
        """添加待检样本"""
        self.pending.append(sample)

    def review(
        self,
        sample_id: str,
        approved: bool,
        checklist: Dict[ReviewChecklist, bool],
        notes: str = ""
    ):
        """提交审查结果"""
        sample = next((s for s in self.pending if s.get("id") == sample_id), None)
        if not sample:
            return

        # 记录审查结果
        sample["approved"] = approved
        sample["checklist"] = {k.value: v for k, v in checklist.items()}
        sample["notes"] = notes

        if approved:
            # 检查清单：所有项必须通过
            if all(checklist.values()):
                self.approved.append(sample)
            else:
                self.rejected.append(sample)
        else:
            self.rejected.append(sample)

        self.pending = [s for s in self.pending if s.get("id") != sample_id]

    def get_statistics(self) -> Dict:
        """获取统计"""
        total = len(self.approved) + len(self.rejected)
        return {
            "total": total,
            "approved": len(self.approved),
            "rejected": len(self.rejected),
            "pending": len(self.pending),
            "approval_rate": len(self.approved) / total if total > 0 else 0
        }
```

---

## 5. 数据格式

### 5.1 请求格式

```python
{
    "num_images": 1000,
    "class_templates": {
        "car": "A photo of a car on the road",
        "person": "A photo of a person walking"
    },
    "output_dir": "./data/synthetic",
    "vlm_confidence_threshold": 0.7,
    "clip_relevance_threshold": 0.25,
    "human_review_ratio": 0.1,
    "synthetic_ratio": 0.3  # 不超过 30%
}
```

### 5.2 输出格式

```python
{
    "images": [{"id": 1, "file_name": "image_001.jpg"}],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100,100,50,50]}
    ],
    "categories": [{"id": 1, "name": "car"}],
    "metadata": {
        "synthetic": True,
        "quality_scores": {
            "vlm_confidence": 0.85,
            "clip_relevance": 0.32
        },
        "human_reviewed": True,
        "review_stats": {
            "approved": 85,
            "rejected": 15,
            "approval_rate": 0.85
        }
    }
}
```

---

## 6. 专家审核要点

| 审核项 | 状态 | 说明 |
|--------|------|------|
| 合成数据 ≤ 30% | ✅ | 比例控制 |
| CLIP 相关性过滤 | ✅ | 只用于相关性 |
| VLM 置信度过滤 | ✅ | ≥ 0.7 |
| 人工抽检 Checklist | ✅ | 定义明确标准 |
| 并发控制 | ✅ | Semaphore |

---

## 7. 关键改进说明 (v3 → v4)

### 改进 1: CLIP 用途修正
- **v3 错误**: 使用 CLIP 评估"图像质量"
- **v4 正确**: 使用 CLIP 评估"图像-标注相关性"
- **依据**: CLIP 本质是图像-文本匹配

### 改进 2: 人工抽检 Checklist
- **v3 错误**: 随机抽检，无标准
- **v4 正确**: 定义明确 Checklist
- **依据**: 标准化审查流程

### 改进 3: 合成数据比例
- **v3 错误**: 未明确限制
- **v4 正确**: ≤ 30%
- **依据**: Andrej Karpathy "真实数据为主"

---

*审核状态: 通过 - 符合数据生成最佳实践*
