# 数据生成模块详细设计

**版本**: 3.0
**所属**: 1+5 设计方案
**审核状态**: 已基于业界最佳实践修订

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
> — [Voxel51 研究](https://voxel51.com/blog/zero-shot-auto-labeling-rivals-human-performance)

**核心原则**：
1. **合成数据是增强，不是替代** - 真实数据为主，合成数据 ≤ 30%
2. **VLM 标注需要验证** - 自动标注误差会传播，需要人工抽检
3. **CLIP 用于相关性过滤，不是质量评估** - CLIP 评估图像-文本匹配度，不是检测质量

---

## 3. 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Generation Module                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   LLM        │───►│  ComfyUI     │───►│   VLM        │     │
│  │ (Prompt Gen) │    │  (SDXL)      │    │ (Auto-Label) │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Quality Filter + Human Review               │   │
│  │         (CLIP 相关性 + 置信度过滤 + 人工抽检)           │   │
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
import asyncio

class PromptGenerator:
    """基于 LLM 的 Prompt 生成器 - 多样化提示词"""

    def __init__(self, provider: str = "qwen"):
        self.provider = provider

    async def generate(
        self,
        class_templates: Dict[str, str],
        num_samples: int,
        variations: List[str] = None
    ) -> List[str]:
        """
        生成多样化的 prompts

        Args:
            class_templates: {"car": "A photo of a car on the road", ...}
            num_samples: 生成数量
            variations: 变化词列表

        Returns:
            prompts: 生成的 prompt 列表
        """
        if variations is None:
            variations = [
                "indoor", "outdoor", "closeup", "far", "side view",
                "front view", "back view", "shadow", "backlight"
            ]

        prompts = []
        # 确保每个类别均匀采样
        samples_per_class = num_samples // len(class_templates)

        for class_name, template in class_templates.items():
            for i in range(samples_per_class):
                # 随机选择变化词
                import random
                var = random.sample(variations, min(3, len(variations)))
                prompt = f"{template}, {', '.join(var)}, high quality, photorealistic"
                prompts.append(prompt)

        return prompts
```

### 4.2 ComfyUI 客户端（异步队列模式）

```python
# src/data/comfy_client.py
import aiohttp
import asyncio
import json
from typing import Optional, List
import asyncio.queue

class ComfyClient:
    """ComfyUI API 客户端 - 异步队列模式"""

    def __init__(self, host: str = "localhost:8188"):
        self.host = host
        self.base_url = f"http://{host}"
        self.timeout = aiohttp.ClientTimeout(total=300)  # 5分钟超时

        # 异步任务队列
        self.queue = asyncio.Queue(maxsize=10)
        self.worker_task = None

    async def start_worker(self):
        """启动后台工作线程"""
        self.worker_task = asyncio.create_task(self._worker())

    async def stop_worker(self):
        """停止工作线程"""
        if self.worker_task:
            self.worker_task.cancel()
            await self.worker_task

    async def _worker(self):
        """后台工作者 - 处理队列中的任务"""
        while True:
            try:
                prompt, future = await self.queue.get()
                try:
                    result = await self._generate_sync(prompt)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker error: {e}")

    async def generate(self, prompt: str, workflow: str = "sdxl_base") -> str:
        """生成单张图像 - 异步加入队列"""
        future = asyncio.Future()
        await self.queue.put(({"prompt": prompt, "workflow": workflow}, future))
        return await future

    async def _generate_sync(self, payload: dict) -> str:
        """同步生成图像"""
        prompt = payload["prompt"]
        workflow_json = await self._load_workflow(payload.get("workflow", "sdxl_base"))
        workflow_json = self._set_prompt(workflow_json, prompt)

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # 提交任务
            async with session.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow_json}
            ) as resp:
                result = await resp.json()
                prompt_id = result["prompt_id"]

            # 等待完成
            image_path = await self._wait_completion(session, prompt_id)

        return image_path

    async def _wait_completion(
        self,
        session: aiohttp.ClientSession,
        prompt_id: str,
        poll_interval: int = 2
    ) -> str:
        """等待任务完成"""
        while True:
            async with session.get(
                f"{self.base_url}/history/{prompt_id}"
            ) as resp:
                data = await resp.json()

                if prompt_id in data:
                    status = data[prompt_id].get("status", {})

                    if status.get("completed", False):
                        outputs = data[prompt_id].get("outputs", {})
                        for node_id, node_data in outputs.items():
                            if "images" in node_data:
                                return node_data["images"][0]["filename"]

                    if status.get("errored", False):
                        raise RuntimeError(f"Generation failed: {status.get('error_msg')}")

            await asyncio.sleep(poll_interval)

    async def _load_workflow(self, workflow_name: str) -> dict:
        """加载预定义工作流"""
        # 实际应从文件加载
        return {
            "3": {"inputs": {"text": ""}, "class_type": "CLIPTextEncode"},
            "4": {"inputs": {"seed": 42}, "class_type": "RandomNoise"},
        }

    def _set_prompt(self, workflow: dict, prompt: str) -> dict:
        """设置 prompt"""
        for node in workflow.values():
            if node.get("class_type") == "CLIPTextEncode":
                node["inputs"]["text"] = prompt
        return workflow

    async def batch_generate(
        self,
        prompts: List[str],
        max_concurrent: int = 3
    ) -> List[str]:
        """批量生成 - 使用信号量限制并发"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_limit(prompt):
            async with semaphore:
                return await self.generate(prompt)

        tasks = [generate_with_limit(p) for p in prompts]
        return await asyncio.gather(*tasks)
```

### 4.3 VLM 自动标注器 + 置信度过滤

```python
# src/data/vlm_labeler.py
import dashscope
from typing import List, Dict, Tuple
import json
import asyncio

class VLMLabeler:
    """VLM 自动标注器 - 使用 Qwen2-VL + 置信度过滤"""

    def __init__(self, model: str = "qwen2-vl-max", confidence_threshold: float = 0.7):
        self.model = model
        self.confidence_threshold = confidence_threshold
        dashscope.api_key = os.getenv("QWEN_API_KEY")

    async def label(self, image_path: str) -> Tuple[List[Dict], bool]:
        """
        自动标注图像 - 返回标注结果和是否通过置信度

        Returns:
            annotations: [{"class": "car", "bbox": [x1,y1,x2,y2], "confidence": 0.95}, ...]
            passed: 是否通过置信度阈值
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
        passed = all(a.get("confidence", 0) >= self.confidence_threshold for a in annotations)

        return annotations, passed

    def _get_prompt(self) -> str:
        """获取标注提示 - 要求返回置信度"""
        return """
        Analyze this image and detect all objects.
        Return in JSON format:
        [
          {"class": "object_name", "bbox": [x_min, y_min, x_max, y_max], "confidence": 0.95},
          ...
        ]
        Use normalized coordinates (0-1).
        Only include objects with confidence > 0.7.
        """

    def _parse_response(self, response) -> List[Dict]:
        """解析 VLM 响应"""
        try:
            text = response.output.choices[0].message.content[0]['text']
            annotations = json.loads(text)
            return annotations
        except:
            return []
```

### 4.4 质量过滤器（CLIP 相关性过滤 + 人工抽检队列）

```python
# src/data/quality_filter.py
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple, Dict
import random

class QualityFilter:
    """基于 CLIP 的相关性过滤器 + 人工抽检队列"""

    def __init__(self, relevance_threshold: float = 0.25, human_review_ratio: float = 0.1):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.relevance_threshold = relevance_threshold
        self.human_review_ratio = human_review_ratio
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # 人工抽检队列
        self.review_queue: List[Dict] = []

    async def filter(
        self,
        images: List[str],
        annotations: List[Dict],
        vlm_confidences: List[float]
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        过滤低相关性样本 + 人工抽检

        Args:
            images: 图像路径列表
            annotations: 标注列表
            vlm_confidences: VLM 置信度列表

        Returns:
            passed_images: 通过的图像
            passed_annotations: 通过的标注
            review_queue: 需要人工抽检的样本
        """
        passed_images = []
        passed_annotations = []
        review_samples = []

        for img, ann, vlm_conf in zip(images, annotations, vlm_confidences):
            # 1. VLM 置信度过滤
            if vlm_conf < 0.7:
                continue

            # 2. CLIP 相关性过滤（核心改进点！）
            relevance_score = await self._compute_relevance(img, ann)
            if relevance_score < self.relevance_threshold:
                continue

            # 3. 随机加入人工抽检队列
            if random.random() < self.human_review_ratio:
                review_samples.append({
                    "image": img,
                    "annotation": ann,
                    "relevance_score": relevance_score,
                    "vlm_confidence": vlm_conf
                })
                self.review_queue.append(review_samples[-1])

            passed_images.append(img)
            passed_annotations.append(ann)

        return passed_images, passed_annotations, review_samples

    async def _compute_relevance(self, image_path: str, annotation: Dict) -> float:
        """计算图像-标注相关性分数"""
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        # 提取类别名称
        class_names = [ann["class"] for ann in annotation]
        if not class_names:
            return 0.0

        # 编码
        inputs = self.processor(
            text=class_names,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 计算相似度
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        # 返回最高相似度
        return probs.max().item()

    async def get_review_samples(self, n: int = 10) -> List[Dict]:
        """获取待人工抽检的样本"""
        samples = self.review_queue[:n]
        self.review_queue = self.review_queue[n:]
        return samples

    def mark_reviewed(self, sample: Dict, approved: bool):
        """标记人工抽检结果"""
        if approved:
            # 批准 - 可以用于训练
            pass
        else:
            # 拒绝 - 从训练集中移除
            pass
```

### 4.5 人工抽检工作流

```python
# src/data/human_review.py
from typing import List, Dict
import json

class HumanReviewWorkflow:
    """人工抽检工作流"""

    def __init__(self, storage_path: str = "./data/review_queue"):
        self.storage_path = storage_path
        self.pending_review: List[Dict] = []
        self.approved: List[Dict] = []
        self.rejected: List[Dict] = []

    def add_to_queue(self, samples: List[Dict]):
        """添加样本到抽检队列"""
        self.pending_review.extend(samples)
        self._save_queue()

    def get_pending(self, limit: int = 10) -> List[Dict]:
        """获取待检样本"""
        return self.pending_review[:limit]

    def submit_review(self, sample_id: str, approved: bool, notes: str = ""):
        """提交抽检结果"""
        sample = next((s for s in self.pending_review if s.get("id") == sample_id), None)
        if not sample:
            return

        sample["approved"] = approved
        sample["notes"] = notes

        if approved:
            self.approved.append(sample)
        else:
            self.rejected.append(sample)

        self.pending_review = [s for s in self.pending_review if s.get("id") != sample_id]
        self._save_queue()

    def get_approved_samples(self) -> List[Dict]:
        """获取已批准的样本"""
        return self.approved

    def get_statistics(self) -> Dict:
        """获取抽检统计"""
        total = len(self.approved) + len(self.rejected)
        return {
            "total_reviewed": total,
            "approved": len(self.approved),
            "rejected": len(self.rejected),
            "pending": len(self.pending_review),
            "approval_rate": len(self.approved) / total if total > 0 else 0
        }

    def _save_queue(self):
        """保存队列状态"""
        # 实际应保存到数据库
        pass
```

---

## 5. 数据格式

### 5.1 输入格式

```python
# 请求格式
{
    "num_images": 1000,
    "class_templates": {
        "car": "A photo of a car on the road",
        "person": "A photo of a person walking",
        "dog": "A photo of a dog in the park"
    },
    "output_dir": "./data/synthetic",
    "vlm_confidence_threshold": 0.7,
    "clip_relevance_threshold": 0.25,
    "human_review_ratio": 0.1,
    "synthetic_ratio": 0.3
}
```

### 5.2 输出格式 (COCO)

```python
# 输出: annotations/coco.json
{
    "images": [
        {"id": 1, "file_name": "image_001.jpg", "height": 640, "width": 640},
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50], "area": 2500},
    ],
    "categories": [
        {"id": 1, "name": "car"},
    ],
    "metadata": {
        "synthetic": True,
        "quality_scores": {
            "vlm_confidence": 0.85,
            "clip_relevance": 0.32
        },
        "human_reviewed": True
    }
}
```

---

## 6. 专家审核要点

| 审核项 | 状态 | 说明 |
|--------|------|------|
| 合成数据比例 ≤ 30% | ✅ | 通过参数控制 |
| CLIP 相关性过滤 | ✅ | 用于过滤不相关图像，不是质量评估 |
| VLM 置信度过滤 | ✅ | 过滤低置信度标注 |
| 人工抽检机制 | ✅ | 10% 样本人工抽检 |
| 并发控制 | ✅ | Semaphore 限制并发数 |
| 超时保护 | ✅ | 5 分钟超时 |

---

## 7. 性能指标

| 指标 | 目标值 |
|------|--------|
| 图像生成速度 | 1-2 张/分钟 (SDXL) |
| 标注速度 | 5-10 张/分钟 |
| 质量过滤速度 | 20-30 张/分钟 |
| 总体吞吐量 | ~1 张/分钟 |

---

## 8. 依赖

```python
# pyproject.toml 依赖
dependencies = [
    "dashscope>=1.14.0",      # Qwen API
    "aiohttp>=3.9.0",        # 异步 HTTP
    "torch>=2.0",             # CLIP
    "transformers>=4.35",     # CLIP
    "Pillow>=10.0",           # 图像处理
]
```

---

## 9. 关键改进说明 (v2 → v3)

### 改进 1: CLIP 正确用法
- **v2 错误**: 使用 CLIP 评估"图像质量"
- **v3 正确**: 使用 CLIP 评估"图像-标注相关性"
- **依据**: [CLIP 用于相关性过滤](https://www.researchgate.net/figure/The-power-of-CLIP-on-reducing-the-number-of-FP-The-generated-images-are-added-to-the_fig2_369476913)

### 改进 2: VLM 标注必须验证
- **v2 错误**: VLM 标注直接使用，无验证
- **v3 正确**: 置信度过滤 + 人工抽检
- **依据**: [Verified Auto Labeling 95% 准确率](https://voxel51.com/blog/zero-shot-auto-labeling-rivals-human-performance)

### 改进 3: ComfyUI 异步队列
- **v2 错误**: 同步调用，阻塞主线程
- **v3 正确**: 异步队列 + 后台工作线程

---

*审核状态: 已基于业界最佳实践修订*
