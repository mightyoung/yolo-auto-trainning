"""
Transfer Learning Trainer with knowledge distillation support.
Location: training-api/src/training/transfer_trainer.py

Contains: TransferLearningTrainer class
"""

from pathlib import Path
from typing import Optional
import logging
import os

import torch
from ultralytics import YOLO

from .training_utils import TrainingResult, cleanup_gpu_memory


class TransferLearningTrainer:
    """Transfer learning trainer using pretrained weights.

    Supports multiple knowledge distillation modes:
    - none: standard transfer learning (frozen backbone)
    - mgd:   Minimal Generative Distillation (arXiv:2506.14440) — feature-level L2 loss
    - feature: intermediate feature-map alignment with L2 loss
    - soft:   temperature-scaled soft label distillation
    """

    def __init__(self, teacher_model: str = "yolo11m", freeze_layers: int = 10):
        self.teacher_model_name = teacher_model
        self.freeze_layers = freeze_layers

    def _resolve_model_path(self) -> str:
        """Resolve model path, preferring cached model to avoid slow GitHub downloads."""
        base = self.teacher_model_name
        if base.endswith(".pt"):
            base = base[:-3]
        cache_path = Path(os.path.expanduser("~/.cache/ultralytics")) / f"{base}.pt"
        if cache_path.exists():
            return str(cache_path)
        return f"{base}.pt"

    def _build_distiller_hook(
        self,
        teacher_model,
        student_model,
        distiller: str,
        loss_weight: float,
        temperature: float,
        device: str,
    ):
        """Build the distillation loss callback."""
        teacher_feats: dict = {}
        student_feats: dict = {}
        _hook_handles: list = []

        target_layer_names = ["model.7", "model.16", "model.23"]

        def _make_hook(name: str, storage: dict):
            def hook_fn(module, input, output):
                try:
                    storage[name] = output.detach()
                except Exception:
                    pass
            return hook_fn

        teacher_state = teacher_model.model.state_dict() if hasattr(teacher_model, "model") else {}
        for name, module in teacher_model.model.named_modules():
            if any(t in name for t in target_layer_names):
                handle = module.register_forward_hook(_make_hook(f"t_{name}", teacher_feats))
                _hook_handles.append(handle)

        for name, module in student_model.model.named_modules():
            if any(t in name for t in target_layer_names):
                handle = module.register_forward_hook(_make_hook(f"s_{name}", student_feats))
                _hook_handles.append(handle)

        def _remove_hooks():
            for h in _hook_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            _hook_handles.clear()

        def _distill_callback(trainer):
            if distiller == "none":
                return

            try:
                t_keys = sorted(teacher_feats.keys())
                s_keys = sorted(student_feats.keys())

                if not t_keys or not s_keys:
                    return

                distill_loss = torch.tensor(0.0, device=device)

                if distiller == "mgd":
                    for tk in t_keys:
                        sf = student_feats.get(tk.replace("t_", "s_", 1))
                        if sf is None:
                            continue
                        tf = teacher_feats[tk]
                        t_f = tf
                        s_f = sf
                        if t_f.shape[1] != s_f.shape[1]:
                            proj = torch.nn.Conv2d(
                                s_f.shape[1], t_f.shape[1], kernel_size=1, bias=False, device=s_f.device
                            ).to(s_f.dtype)
                            s_f = proj(s_f)
                        diff = (t_f - s_f) ** 2
                        distill_loss = distill_loss + diff.mean()

                elif distiller == "feature":
                    for tk, sk in zip(t_keys, s_keys):
                        tf = teacher_feats[tk]
                        sf = student_feats.get(sk)
                        if sf is None:
                            continue
                        if tf.shape[1] != sf.shape[1]:
                            proj = torch.nn.Conv2d(
                                sf.shape[1], tf.shape[1], kernel_size=1, bias=False, device=sf.device
                            ).to(sf.dtype)
                            sf = proj(sf)
                        distill_loss = distill_loss + ((tf - sf) ** 2).mean()

                elif distiller == "soft":
                    for tk, sk in zip(t_keys, s_keys):
                        tf = teacher_feats[tk]
                        sf = student_feats.get(sk)
                        if sf is None:
                            continue
                        t_soft = torch.softmax(tf.flatten(1) / temperature, dim=-1)
                        s_soft = torch.softmax(sf.flatten(1) / temperature, dim=-1)
                        distill_loss = distill_loss + (t_soft - s_soft).abs().mean()

                teacher_feats.clear()
                student_feats.clear()

                distill_val = distill_loss.item() if isinstance(distill_loss, torch.Tensor) else float(distill_loss)
                if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
                    trainer.loss_items = trainer.loss_items + loss_weight * distill_val

            except Exception as e:
                logging.warning(f"[Distillation callback] Error: {e}")

        return _distill_callback, _remove_hooks

    def train(
        self,
        data_yaml: Path,
        epochs: int = 100,
        distiller: str = "none",
        loss_weight: float = 1.0,
        temperature: float = 4.0,
        teacher_model_path: Optional[str] = None,
        output_dir: str = "./runs/transfer",
        device: str = "cuda:0",
    ) -> TrainingResult:
        """Train with transfer learning and optional knowledge distillation."""
        teacher_path = teacher_model_path or self._resolve_model_path()
        student_path = self._resolve_model_path()

        logging.info(
            f"[TransferLearning] distiller={distiller}, loss_weight={loss_weight}, "
            f"temperature={temperature}, teacher={teacher_path}, device={device}"
        )

        student_model = YOLO(student_path)
        teacher_model = None

        if distiller != "none":
            logging.info(f"[TransferLearning] Loading teacher model: {teacher_path}")
            teacher_model = YOLO(teacher_path)

            distill_cb, remove_hooks = self._build_distiller_hook(
                teacher_model=teacher_model,
                student_model=student_model,
                distiller=distiller,
                loss_weight=loss_weight,
                temperature=temperature,
                device=device,
            )
            student_model.add_callback("on_train_batch_start", distill_cb)

        try:
            results = student_model.train(
                data=str(data_yaml),
                epochs=epochs,
                freeze=self.freeze_layers,
                project=output_dir,
                name="student",
                verbose=False,
                device=device,
                distiller=distiller,
            )

            return TrainingResult(
                status="completed",
                model_path=Path(results.save_dir) / "weights" / "best.pt",
                metrics={
                    "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                    "distiller": distiller,
                    "loss_weight": loss_weight,
                    "temperature": temperature,
                },
            )
        except Exception as e:
            logging.error(f"[TransferLearning] Training failed: {e}", exc_info=True)
            return TrainingResult(status="failed", error=str(e))
        finally:
            if teacher_model is not None:
                del teacher_model
            cleanup_gpu_memory()
