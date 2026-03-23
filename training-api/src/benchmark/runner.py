"""
Model benchmarking for YOLO exports.

Measures FPS, parameter count, FLOPs, and file size for each exported format.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import logging
import time

import torch
from ultralytics import YOLO

try:
    from thop import profile as thop_profile
    _THOP_AVAILABLE = True
except ImportError:
    _THOP_AVAILABLE = False
    thop_profile = None


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark result container."""
    format: str
    fps: float
    params_m: float          # million parameters
    gflops: Optional[float]  # billion FLOPs (None if unavailable)
    size_mb: float
    inference_time_ms: float
    gpu_available: bool

    def to_dict(self):
        return asdict(self)


class BenchmarkRunner:
    """Run performance benchmarks on YOLO model exports."""

    def run(
        self,
        model_path: Path,
        format: str = "onnx",
        imgsz: int = 640,
        warmup: int = 10,
        runs: int = 100,
    ) -> BenchmarkResult:
        """
        Run benchmark on a model file.

        Args:
            model_path: Path to the model file (e.g. .onnx, .engine)
            format: Format string for reporting (e.g. "onnx", "engine", "tflite")
            imgsz: Input image size
            warmup: Number of warmup inference runs
            runs: Number of timed inference runs

        Returns:
            BenchmarkResult with FPS, params, FLOPs, size
        """
        gpu_available = torch.cuda.is_available()
        device = "0" if gpu_available else "cpu"

        # Load model
        try:
            model = YOLO(str(model_path))
            # Attempt to move to device; YOLO handles this internally
            if gpu_available:
                try:
                    model.to("cuda:0")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[BenchmarkRunner] Failed to load model: {e}")
            return BenchmarkResult(
                format=format,
                fps=0.0,
                params_m=0.0,
                gflops=None,
                size_mb=0.0,
                inference_time_ms=0.0,
                gpu_available=gpu_available,
            )

        # File size
        size_mb = model_path.stat().st_size / (1024 * 1024)

        # Parameter count
        params_m = self._count_params(model)

        # FLOPs (optional, requires thop)
        gflops = self._compute_flops(model, imgsz)

        # Warmup runs
        dummy_input = torch.zeros(1, 3, imgsz, imgsz)
        if gpu_available:
            dummy_input = dummy_input.to("cuda:0")

        for _ in range(warmup):
            try:
                model(dummy_input, verbose=False, imgsz=imgsz)
            except Exception:
                pass

        # Timed runs
        if gpu_available:
            if hasattr(torch.cuda, "synchronize"):
                torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            try:
                model(dummy_input, verbose=False, imgsz=imgsz)
            except Exception as e:
                logger.warning(f"[BenchmarkRunner] Inference error: {e}")
                break
            if gpu_available:
                if hasattr(torch.cuda, "synchronize"):
                    torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        if times:
            avg_time = sum(times) / len(times)
            fps = 1.0 / avg_time if avg_time > 0 else 0.0
            inference_time_ms = avg_time * 1000.0
        else:
            fps = 0.0
            inference_time_ms = 0.0

        return BenchmarkResult(
            format=format,
            fps=round(fps, 2),
            params_m=round(params_m, 2),
            gflops=round(gflops, 2) if gflops is not None else None,
            size_mb=round(size_mb, 2),
            inference_time_ms=round(inference_time_ms, 2),
            gpu_available=gpu_available,
        )

    def _count_params(self, model: YOLO) -> float:
        """Count total parameters in millions."""
        try:
            # Try to get the underlying model
            if hasattr(model, "model"):
                pytorch_model = model.model
            else:
                pytorch_model = model

            total = sum(p.numel() for p in pytorch_model.parameters())
            return total / 1e6
        except Exception as e:
            logger.warning(f"[BenchmarkRunner] Parameter count failed: {e}")
            return 0.0

    def _compute_flops(self, model: YOLO, imgsz: int) -> Optional[float]:
        """Compute FLOPs using thop library if available."""
        if not _THOP_AVAILABLE or thop_profile is None:
            return None

        try:
            if hasattr(model, "model"):
                pytorch_model = model.model
            else:
                pytorch_model = model

            dummy = torch.zeros(1, 3, imgsz, imgsz)
            flops, _ = thop_profile(pytorch_model, inputs=(dummy,), verbose=False)
            return flops / 1e9  # Convert to billions
        except Exception as e:
            logger.warning(f"[BenchmarkRunner] FLOPs computation failed (thop): {e}")
            return None
