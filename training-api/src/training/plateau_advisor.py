"""
Plateau Advisor — LLM-driven diagnosis for training plateau situations.

When training plateaus (mAP50 stops improving), this module collects diagnostic
information and queries a LLM (DeepSeek) to determine:
  1. Root cause: overfitting / insufficient_data / augmentation_issue / lr_misconfigured / unknown
  2. Recommended actions with specific parameter adjustments
  3. Estimated mAP50 improvement range

Usage:
    advisor = PlateauAdvisor()
    diagnosis = advisor.diagnose(
        mAP50_history=[...],           # List of mAP50 values (last 50 epochs)
        loss_history={"train_box": [...], "val_box": [...]},
        dataset_info={"train_images": 500, "val_images": 100, "num_classes": 2},
        augmentation_params={"mixup": 0.3, "copy_paste": 0.4},
        current_config={"lr0": 0.01, "epochs": 150, "imgsz": 1280},
    )
    # diagnosis["diagnosis"] — root cause string
    # diagnosis["recommendations"] — list of action dicts
"""
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import httpx

logger = logging.getLogger(__name__)


@dataclass
class DiagnosisResult:
    """Structured result from LLM diagnosis."""
    diagnosis: str          # "overfitting" | "insufficient_data" | "augmentation_weak"
                          # | "lr_too_high" | "lr_too_low" | "data_quality_issue" | "unknown"
    confidence: float = 0.0
    reasoning: str = ""
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    # Each recommendation: {"action": str, "params": dict, "priority": int}
    estimated_gain: str = ""   # e.g. "+0.02~0.05"
    raw_response: str = ""     # Raw LLM text for debugging

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "recommendations": self.recommendations,
            "estimated_gain": self.estimated_gain,
        }


class PlateauAdvisor:
    """LLM-powered plateau diagnosis for YOLO training.

    Collects training telemetry, queries DeepSeek API, returns actionable recommendations.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._enabled = bool(self.api_key)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def diagnose(
        self,
        mAP50_history: Optional[List[float]] = None,
        loss_history: Optional[Dict[str, List[float]]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        augmentation_params: Optional[Dict[str, Any]] = None,
        current_config: Optional[Dict[str, Any]] = None,
        task_description: str = "fire smoke detection",
        target_mAP: float = 0.90,
    ) -> DiagnosisResult:
        """
        Diagnose training plateau and return actionable recommendations.

        Args:
            mAP50_history: Last N mAP50 validation scores
            loss_history: {"train_box": [...], "train_cls": [...], "val_box": [...], "val_cls": [...]}
            dataset_info: {"train_images": int, "val_images": int, "num_classes": int, "total_boxes": int}
            augmentation_params: Current augmentation settings
            current_config: {"lr0": float, "epochs": int, "imgsz": int, "batch": int, "optimizer": str}
            task_description: What the model is detecting
            target_mAP: Target mAP50

        Returns:
            DiagnosisResult with root cause and recommendations
        """
        if not self._enabled:
            logger.warning("[PlateauAdvisor] Disabled: no DEEPSEEK_API_KEY configured")
            return DiagnosisResult(
                diagnosis="unknown",
                reasoning="PlateauAdvisor disabled (no API key)",
            )

        prompt = self._build_prompt(
            mAP50_history=mAP50_history or [],
            loss_history=loss_history or {},
            dataset_info=dataset_info or {},
            augmentation_params=augmentation_params or {},
            current_config=current_config or {},
            task_description=task_description,
            target_mAP=target_mAP,
        )

        try:
            raw = self._call_llm(prompt)
            return self._parse_response(raw)
        except Exception as e:
            logger.error(f"[PlateauAdvisor] LLM call failed: {e}")
            return DiagnosisResult(
                diagnosis="unknown",
                reasoning=f"LLM call failed: {e}",
            )

    def _build_prompt(
        self,
        mAP50_history: List[float],
        loss_history: Dict[str, List[float]],
        dataset_info: Dict[str, Any],
        augmentation_params: Dict[str, Any],
        current_config: Dict[str, Any],
        task_description: str,
        target_mAP: float,
    ) -> str:
        # Compute summary statistics
        if mAP50_history:
            recent_10 = mAP50_history[-10:]
            early_10 = mAP50_history[:10] if len(mAP50_history) >= 10 else mAP50_history
            map_trend = "improving" if sum(recent_10) > sum(early_10) else "plateaued_or_declining"
            map_delta = sum(recent_10) / len(recent_10) - sum(early_10) / len(early_10)
            current_map = mAP50_history[-1]
            best_map = max(mAP50_history)
            epoch_best = mAP50_history.index(best_map) + 1
        else:
            map_trend = "unknown"
            map_delta = 0.0
            current_map = 0.0
            best_map = 0.0
            epoch_best = 0

        # Overfitting detection: val_loss rising while train_loss falling
        overfitting_signal = ""
        if loss_history:
            train_box = loss_history.get("train_box_loss", [])
            val_box = loss_history.get("val_box_loss", [])
            if len(train_box) >= 10 and len(val_box) >= 10:
                train_trend = train_box[-5:].count(min(train_box[-5:])) if train_box[-5:] else 0
                val_trend = val_box[-5:] > [x * 1.05 for x in val_box[-10:-5]] if len(val_box) >= 10 else False
                if val_trend:
                    overfitting_signal = (
                        f"Possible overfitting: val_box_loss rising in recent 5 epochs "
                        f"(last={val_box[-1]:.4f}) while train_box_loss={train_box[-1]:.4f}. "
                    )

        # Dataset scale assessment
        train_imgs = dataset_info.get("train_images", 0)
        num_classes = dataset_info.get("num_classes", 2)
        images_per_class = train_imgs / max(num_classes, 1)
        dataset_scale = "small" if images_per_class < 500 else "medium" if images_per_class < 2000 else "large"

        prompt = f"""You are an expert YOLO training advisor. A YOLO object detection model training for "{task_description}" has plateaued.

## Current Status
- Target mAP50: {target_mAP}
- Current mAP50: {current_map:.4f}
- Best mAP50 so far: {best_map:.4f} (epoch {epoch_best})
- Gap to target: {target_mAP - current_map:.4f}
- mAP50 trend: {map_trend} (delta over full training: {map_delta:+.4f})
- Last 20 mAP50 values: {[f'{v:.4f}' for v in mAP50_history[-20:]]}

## Dataset Info
- Training images: {train_imgs}
- Validation images: {dataset_info.get('val_images', 'unknown')}
- Number of classes: {num_classes}
- Images per class: {images_per_class:.0f}
- Dataset scale assessment: {dataset_scale}
- Total boxes: {dataset_info.get('total_boxes', 'unknown')}

## Current Training Config
- Learning rate (lr0): {current_config.get('lr0', 'unknown')}
- Optimizer: {current_config.get('optimizer', 'SGD')}
- Epochs: {current_config.get('epochs', 'unknown')}
- Image size: {current_config.get('imgsz', 'unknown')}
- Batch size: {current_config.get('batch', 'unknown')}

## Augmentation Settings
{json.dumps(augmentation_params or {}, indent=2)}

## Loss History (last 10 epochs)
{json.dumps({k: [round(v, 4) for v in vals[-10:]] for k, vals in loss_history.items()}, indent=2) if loss_history else "Not available"}

{overfitting_signal}

## Your Task
Analyze the training plateau and provide a structured diagnosis. Output your response as valid JSON with this exact schema:

{{
  "diagnosis": "one_of: overfitting | insufficient_data | augmentation_weak | augmentation_strong | lr_too_high | lr_too_low | data_quality_issue | unknown",
  "confidence": 0.0_to_1.0,
  "reasoning": "2-3 sentence explanation of why you reached this diagnosis",
  "recommendations": [
    {{
      "action": "string: one_of [reduce_mixup, increase_mixup, add_copy_paste, reduce_augmentation, lower_lr, raise_lr, add_data, improve_labels, use_ema, early_stop, change_optimizer, increase_epochs]",
      "params": {{"key": "value pairs of specific parameter changes"}},
      "priority": 1_to_3 (1=highest),
      "expected_gain": "+0.01~0.03 description"
    }}
  ],
  "estimated_gain": "+0.02~0.05 (realistic mAP50 improvement from following recommendations)"
}}

Important:
- Output ONLY valid JSON, no markdown, no explanation outside the JSON
- Be specific about parameter values in recommendations
- Consider that this is for fire/smoke detection (small objects, harsh conditions)
- If dataset is small ({dataset_scale}), data expansion recommendations should be high priority
- If val loss is rising while train loss falls, overfitting is the diagnosis
"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call DeepSeek API with retry."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,  # Low temperature for factual diagnosis
            "max_tokens": 1024,
        }

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                elif resp.status_code == 429:
                    logger.warning("[PlateauAdvisor] Rate limited, waiting 30s...")
                    time.sleep(30)
                    continue
                else:
                    logger.error(f"[PlateauAdvisor] API error {resp.status_code}: {resp.text[:200]}")
                    break
            except httpx.TimeoutException:
                logger.warning(f"[PlateauAdvisor] Timeout (attempt {attempt + 1}/{max_retries + 1})")
                if attempt < max_retries:
                    time.sleep(5)
                    continue
                raise

        raise RuntimeError(f"PlateauAdvisor LLM call failed after {max_retries + 1} attempts")

    def _parse_response(self, raw: str) -> DiagnosisResult:
        """Parse LLM JSON response into DiagnosisResult."""
        # Try to extract JSON from response (might be wrapped in ```json)
        raw = raw.strip()
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

        # Validate required fields
        diagnosis = data.get("diagnosis", "unknown")
        if diagnosis not in (
            "overfitting", "insufficient_data", "augmentation_weak",
            "augmentation_strong", "lr_too_high", "lr_too_low",
            "data_quality_issue", "unknown"
        ):
            diagnosis = "unknown"

        return DiagnosisResult(
            diagnosis=diagnosis,
            confidence=float(data.get("confidence", 0.0)),
            reasoning=str(data.get("reasoning", "")),
            recommendations=data.get("recommendations", []),
            estimated_gain=str(data.get("estimated_gain", "")),
            raw_response=raw,
        )


# Module-level singleton for convenience
_default_advisor: Optional[PlateauAdvisor] = None


def get_advisor() -> PlateauAdvisor:
    """Get or create the default PlateauAdvisor singleton."""
    global _default_advisor
    if _default_advisor is None:
        _default_advisor = PlateauAdvisor()
    return _default_advisor


def diagnose_plateau(
    mAP50_history: Optional[List[float]] = None,
    loss_history: Optional[Dict[str, List[float]]] = None,
    dataset_info: Optional[Dict[str, Any]] = None,
    augmentation_params: Optional[Dict[str, Any]] = None,
    current_config: Optional[Dict[str, Any]] = None,
) -> DiagnosisResult:
    """Convenience function for plateau diagnosis."""
    return get_advisor().diagnose(
        mAP50_history=mAP50_history,
        loss_history=loss_history,
        dataset_info=dataset_info,
        augmentation_params=augmentation_params,
        current_config=current_config,
    )
