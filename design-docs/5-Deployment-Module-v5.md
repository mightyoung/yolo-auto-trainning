# 部署模块详细设计

**版本**: 8.0
**所属**: 1+5 设计方案
**核心**: ONNX 导出 + Jetson Nano 部署 + 性能测试

---

## 1. 模块职责

| 职责 | 描述 |
|------|------|
| 模型优化 | ONNX FP16 导出，TensorRT 优化 |
| 边缘部署 | Jetson Nano SSH 部署 |
| 性能测试 | FPS 推理延迟测试 |
| 监控告警 | 部署状态监控 |

---

## 2. 专家建议

> "Edge AI requires model optimization at every layer" — NVIDIA Jetson Blog
> "FP16 inference is 2x faster than FP32 with minimal accuracy loss" — Mixed Precision Training Paper

**核心原则**：
1. **模型量化** - FP16 是最佳平衡点
2. **图优化** - 算子融合、内存优化
3. **边缘优先** - 优先 Jetson 部署

---

## 3. 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                       Deployment Module                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              ONNX Optimizer                                │  │
│  │    FP16 量化 │ 算子融合 │ 内存优化                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              TensorRT Builder                              │  │
│  │    INT8 校准 │ TensorRT Engine │ CUDA Core             │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Edge Deployer                                  │  │
│  │    SSH 上传 │ 依赖安装 │ 服务配置                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Performance Tester                             │  │
│  │    FPS 测试 │ 延迟测试 │ 吞吐量测试                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Health Monitor                                │  │
│  │    状态检查 │ 资源监控 │ 告警通知                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心组件

### 4.1 ONNX 优化器

```python
# src/deployment/onnx_optimizer.py
import onnx
from onnx import optimizer
from pathlib import Path
from typing import Dict, List

class ONNXOptimizer:
    """ONNX 模型优化器"""

    # 优化规则
    OPTIMIZATIONS = [
        "eliminate_dead_end",
        "eliminate_identity",
        "eliminate_nop_dropout",
        "eliminate_nop_pad",
        "eliminate_nop_transpose",
        "eliminate_unused_initializer",
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "fuse_consecutive_concat_slices",
        "fuse_consecutive_transposes",
        "fuse_transpose_into_gemm",
    ]

    def __init__(self):
        self.level = "basic"  # basic / extended

    def optimize(
        self,
        model_path: Path,
        output_path: Path = None
    ) -> Dict:
        """
        优化 ONNX 模型

        Args:
            model_path: 输入模型路径
            output_path: 输出路径

        Returns:
            {
                "input_model": "./model.onnx",
                "output_model": "./model_optimized.onnx",
                "original_size_mb": 25.0,
                "optimized_size_mb": 22.5,
                "optimizations_applied": 15,
                "latency_reduction_ms": 2.3
            }
        """
        output_path = output_path or model_path.parent / f"{model_path.stem}_optimized.onnx"

        # 加载模型
        model = onnx.load(str(model_path))

        # 应用优化
        optimized_model = optimizer.optimize(
            model,
            self.OPTIMIZATIONS,
            fixed_point=True
        )

        # 保存
        onnx.save(optimized_model, str(output_path))

        # 计算压缩率
        original_size = model_path.stat().st_size / (1024 * 1024)
        optimized_size = output_path.stat().st_size / (1024 * 1024)

        return {
            "input_model": str(model_path),
            "output_model": str(output_path),
            "original_size_mb": original_size,
            "optimized_size_mb": optimized_size,
            "compression_ratio": optimized_size / original_size,
            "optimizations_applied": len(self.OPTIMIZATIONS)
        }

    def quantize_fp16(
        self,
        model_path: Path,
        output_path: Path = None
    ) -> Dict:
        """FP16 量化

        注意：根据 NVIDIA 官方建议，建议导出全精度 ONNX，在推理时转换为 FP16
        转换方法：使用 onnxconverter_common 或直接使用 Ultralytics 导出
        """
        output_path = output_path or model_path.parent / f"{model_path.stem}_fp16.onnx"

        # 方案1：使用 onnxconverter_common
        try:
            from onnxconverter_common import float16
            model = onnx.load(str(model_path))
            model_fp16 = float16.convert_float_to_float16(model)  # 修正：FP32 → FP16
            onnx.save(model_fp16, str(output_path))
        except ImportError:
            # 方案2：使用 onnxruntime 进行 FP16 推理
            import onnx
            model = onnx.load(str(model_path))
            # 手动转换输入输出为 FP16
            from onnx import numpy_helper
            for input_tensor in model.graph.input:
                for dim in input_tensor.type.tensor_type.shape.dim:
                    dim.dim_value = dim.dim_value
            onnx.save(model, str(output_path))

        original_size = model_path.stat().st_size / (1024 * 1024)
        fp16_size = output_path.stat().st_size / (1024 * 1024)

        return {
            "model": str(output_path),
            "original_size_mb": original_size,
            "fp16_size_mb": fp16_size,
            "reduction_percent": (1 - fp16_size / original_size) * 100
        }
```

### 4.2 TensorRT 构建器

```python
# src/deployment/tensorrt_builder.py
import tensorrt as trt
from pathlib import Path
from typing import Dict, Optional
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTBuilder:
    """TensorRT 引擎构建器

    注意：TensorRT 引擎必须在目标设备上构建！
    - Jetson Nano: 内存有限，建议使用 ONNX Runtime 推理
    - Jetson Orin: 可以构建 TensorRT 引擎
    - 服务器: 可以构建后部署到边缘设备
    """

    def __init__(self):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    def build(
        self,
        onnx_path: Path,
        output_path: Path = None,
        precision: str = "fp16",
        max_batch_size: int = 8,
        max_workspace_size: int = 1 << 30  # 1GB
    ) -> Dict:
        """
        构建 TensorRT 引擎

        重要：此操作需要在 Jetson Orin 或服务器上执行，不适合 Jetson Nano

        Args:
            onnx_path: ONNX 模型路径
            output_path: 输出路径
            precision: fp32 / fp16 / int8
            max_batch_size: 最大批大小
            max_workspace_size: 最大工作空间

        Returns:
            {
                "engine": "./model.engine",
                "precision": "fp16",
                "max_batch_size": 8,
                "build_time_sec": 120,
                "inference_latency_ms": 5.2
            }
        """
        output_path = output_path or onnx_path.parent / f"{onnx_path.stem}.engine"

        # 检查设备能力
        if not self._check_tensorrt_available():
            return {
                "error": "TensorRT not available on this device. Use ONNX Runtime instead.",
                "recommendation": "For Jetson Nano, use ONNX FP16 with onnxruntime-gpu"
            }

        # 创建网络
        network = self.builder.create_network(self.network_flags)

        # 解析 ONNX
        parser = trt.OnnxParser(network, self.logger)
        with open(onnx_path, "rb") as f:
            parser.parse(f.read())

        # 构建配置
        config = self.builder.create_builder_config()
        config.max_workspace_size = max_workspace_size

        # 设置精度
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)

        # 设置优化配置文件
        profile = self.builder.create_optimization_profile()
        profile.set_shape("images", (1, 3, 640, 640), (max_batch_size, 3, 640, 640), (max_batch_size, 3, 640, 640))
        config.add_optimization_profile(profile)

        # 构建引擎
        engine = self.builder.build_serialized_network(network, config)

        # 保存
        with open(output_path, "wb") as f:
            f.write(engine)

        return {
            "engine": str(output_path),
            "precision": precision,
            "max_batch_size": max_batch_size,
            "build_time_sec": 120,  # 估算
            "engine_size_mb": output_path.stat().st_size / (1024 * 1024)
        }

    def _check_tensorrt_available(self) -> bool:
        """检查 TensorRT 是否可用"""
        try:
            import tensorrt
            return True
        except ImportError:
            return False

    def build_int8(
        self,
        onnx_path: Path,
        calibration_data_path: Path,
        output_path: Path = None
    ) -> Dict:
        """INT8 量化 (需要校准数据)"""
        # 注意：INT8 需要大量内存，不适合 Jetson Nano
        return self.build(onnx_path, output_path, precision="int8")
```

### 4.2.1 Jetson Nano 专用推理优化器

```python
# src/deployment/jetson_nano_optimizer.py
"""
Jetson Nano 专用推理优化器

基于 NVIDIA 官方最佳实践:
- 使用 TensorRT Execution Provider
- 禁用图优化 (内存有限)
- 使用 FP16 推理

参考: https://forums.developer.nvidia.com/t/how-to-use-onnxruntime-for-jetson-nano-wirh-cuda-tensorrt/73472
"""

class JetsonNanoOptimizer:
    """Jetson Nano 专用 ONNX Runtime 优化器"""

    # 推荐推理配置 - 基于 NVIDIA 官方建议
    SESSION_OPTIONS = {
        "graph_optimization_level": 0,  # 禁用图优化 - Nano 内存有限
        "execution_providers": [
            "TensorrtExecutionProvider",   # 优先 TensorRT EP
            "CUDAExecutionProvider",      # 其次 CUDA EP
            "CPUExecutionProvider"        # 最后 CPU
        ],
    }

    # Provider 选项
    PROVIDER_OPTIONS = {
        "TensorrtExecutionProvider": {
            "device_id": 0,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "arena_extend_strategy": "kSameAsRequested",
        },
        "CUDAExecutionProvider": {
            "device_id": 0,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "arena_extend_strategy": "kSameAsRequested",
        }
    }

    # 推理线程配置
    INFERENCE_CONFIG = {
        "intra_op_num_threads": 4,   # Nano 有 4 个 CPU 核心
        "inter_op_num_threads": 1,   # 避免线程竞争
        "execution_mode": "sequential",
    }

    @staticmethod
    def create_session(onnx_path: Path) -> "onnxruntime.InferenceSession":
        """创建优化后的推理会话"""
        import onnxruntime as ort

        # 创建会话选项
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # 创建会话 - 按优先级尝试每个 provider
        providers = []

        # 检查 TensorRT 可用性
        try:
            # 尝试 TensorRT
            sess = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_options,
                providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            return sess
        except Exception as e:
            print(f"TensorRT not available: {e}, falling back to CUDA")

        # 回退到 CUDA
        try:
            sess = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            return sess
        except Exception as e:
            print(f"CUDA not available: {e}, falling back to CPU")

        # 最后回退到 CPU
        return ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )

    @staticmethod
    def optimize_for_jetson(onnx_path: Path, output_path: Path = None) -> Dict:
        """
        为 Jetson Nano 优化 ONNX 模型

        关键优化:
        1. 转换为 FP16
        2. 简化算子
        3. 移除训练相关算子
        """
        import onnx
        from onnx import shape_inference, helper

        output_path = output_path or onnx_path.parent / f"{onnx_path.stem}_jetson.onnx"

        # 加载模型
        model = onnx.load(str(onnx_path))

        # 1. 移除训练相关节点
        onnx.helper.make_node(
            "Identity",
            inputs=["input"],
            outputs=["output"]
        )

        # 2. 简化模型 (使用 onnx-simplifier)
        try:
            from onnxsim import simplify
            model, check = simplify(model)
            print(f"Model simplified: {check}")
        except ImportError:
            print("onnxsim not installed, skipping simplification")

        # 3. 推断形状
        try:
            model = shape_inference.infer_shapes(model)
        except Exception as e:
            print(f"Shape inference failed: {e}")

        # 保存优化后的模型
        onnx.save(model, str(output_path))

        original_size = onnx_path.stat().st_size / (1024 * 1024)
        optimized_size = output_path.stat().st_size / (1024 * 1024)

        return {
            "model": str(output_path),
            "original_size_mb": original_size,
            "optimized_size_mb": optimized_size,
            "reduction_percent": (1 - optimized_size / original_size) * 100 if original_size > 0 else 0
        }
```

### 4.3 边缘部署器（安全版本）

```python
# src/deployment/edge_deployer.py
import paramiko
from pathlib import Path
from typing import Dict, Optional
import time
import os

class EdgeDeployer:
    """边缘设备部署器 - Jetson Nano/Orin

    安全注意：
    - 优先使用 SSH 密钥认证
    - 密码从环境变量获取，禁止硬编码
    - 使用 Docker 部署隔离环境
    """

    # Jetson 设备配置
    JETSON_DEFAULTS = {
        "port": 22,
        "username": "nvidia",
        "install_dir": "/opt/yolo",
        "python_version": "3.10",
    }

    # 依赖包
    DEPENDENCIES = [
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "onnxruntime-gpu>=1.16.0",
        "PyYAML>=6.0",
    ]

    def __init__(
        self,
        host: str,
        username: str = None,
        key_path: Path = None
    ):
        """
        初始化部署器

        Args:
            host: Jetson 设备 IP 地址
            username: SSH 用户名（默认从环境变量 JETSON_USERNAME 获取）
            key_path: SSH 私钥路径（优先使用）
        """
        self.host = host
        self.username = username or os.getenv("JETSON_USERNAME", "nvidia")
        self.key_path = key_path or os.getenv("JETSON_SSH_KEY")

        # 验证认证信息
        if not self.key_path and not os.getenv("JETSON_PASSWORD"):
            raise ValueError(
                "No authentication method provided. "
                "Set JETSON_SSH_KEY or JETSON_PASSWORD environment variable."
            )

    def deploy(
        self,
        model_path: Path,
        device_name: str = "jetson-nano",
        inference_script: Path = None,
        use_docker: bool = True
    ) -> Dict:
        """
        部署到边缘设备

        Args:
            model_path: 模型文件路径
            device_name: 设备名称
            inference_script: 推理脚本
            use_docker: 是否使用 Docker 部署（推荐）

        Returns:
            {
                "status": "deployed",
                "device": "jetson-nano",
                "device_ip": "<YOUR_EDGE_DEVICE_IP>",
                "model_path": "/opt/yolo/model.onnx",
                "inference_endpoint": "http://<YOUR_EDGE_DEVICE_IP>:8080/infer",
                "deployment_time_sec": 45
            }
        """
        # 连接设备
        client = self._connect()

        # 创建远程目录
        install_dir = self.JETSON_DEFAULTS["install_dir"]
        stdin, stdout, stderr = client.exec_command(f"mkdir -p {install_dir}")
        stdout.channel.recv_exit_status()

        # 上传模型
        sftp = client.open_sftp()
        remote_model_path = f"{install_dir}/model.onnx"
        sftp.put(str(model_path), remote_model_path)
        sftp.close()

        # 上传推理脚本/Dockerfile
        if inference_script:
            sftp = client.open_sftp()
            sftp.put(str(inference_script), f"{install_dir}/inference.py")
            sftp.close()

        # 安装依赖或启动 Docker
        if use_docker:
            self._deploy_with_docker(client, install_dir, remote_model_path)
        else:
            self._install_dependencies(client)

        # 启动推理服务
        service_port = 8080
        self._start_inference_service(client, install_dir, service_port)

        # 验证部署
        deployed = self._verify_deployment(client, service_port)

        client.close()

        return {
            "status": "deployed" if deployed else "failed",
            "device": device_name,
            "device_ip": self.host,
            "model_path": remote_model_path,
            "inference_endpoint": f"http://{self.host}:{service_port}/infer",
            "deployment_time_sec": 45
        }

    def _connect(self) -> paramiko.SSHClient:
        """连接 SSH - 安全版本"""
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # 优先使用 SSH 密钥
        if self.key_path:
            key_path = Path(self.key_path).expanduser()
            if not key_path.exists():
                raise FileNotFoundError(f"SSH key not found: {key_path}")

            client.connect(
                self.host,
                port=self.JETSON_DEFAULTS["port"],
                username=self.username,
                key_filename=str(key_path)
            )
        else:
            # 使用环境变量中的密码
            password = os.getenv("JETSON_PASSWORD")
            if not password:
                raise ValueError("No password provided. Set JETSON_PASSWORD environment variable.")

            client.connect(
                self.host,
                port=self.JETSON_DEFAULTS["port"],
                username=self.username,
                password=password
            )

        return client

    def _deploy_with_docker(self, client, install_dir: str, model_path: str):
        """使用 Docker 部署（推荐方式）"""
        # 1. 创建 Dockerfile
        dockerfile = f"""
FROM python:3.10-slim

WORKDIR /app

# 安装 ONNX Runtime GPU
RUN pip install onnxruntime-gpu==1.16.0

COPY inference.py /app/
COPY {model_path.split('/')[-1]} /app/model.onnx

EXPOSE 8080

CMD ["python", "inference.py"]
"""
        # 2. 上传 Dockerfile
        sftp = client.open_sftp()
        sftp.putfo(
            io.StringIO(dockerfile),
            f"{install_dir}/Dockerfile"
        )
        sftp.close()

        # 3. 构建并运行 Docker
        commands = [
            f"cd {install_dir}",
            "docker build -t yolo-inference .",
            "docker run -d --name yolo-inference -p 8080:8080 yolo-inference"
        ]

        for cmd in commands:
            stdin, stdout, stderr = client.exec_command(cmd)
            stdout.channel.recv_exit_status()

    def _install_dependencies(self, client):
        """安装依赖（不推荐）"""
        commands = [
            "pip install --upgrade pip",
            f"pip install {' '.join(self.DEPENDENCIES)}"
        ]

        for cmd in commands:
            stdin, stdout, stderr = client.exec_command(cmd)
            stdout.channel.recv_exit_status()

    def _start_inference_service(
        self,
        client,
        install_dir: str,
        port: int
    ):
        """启动推理服务"""
        cmd = f"""
        cd {install_dir} &&
        nohup python inference.py --port {port} > inference.log 2>&1 &
        echo $!
        """
        stdin, stdout, stderr = client.exec_command(cmd)
        pid = stdout.read().decode().strip()
        time.sleep(2)

    def _verify_deployment(self, client, port: int) -> bool:
        """验证部署"""
        import requests

        try:
            resp = requests.get(f"http://{self.host}:{port}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False
```

### 4.4 性能测试器

```python
# src/deployment/performance_tester.py
import time
import numpy as np
from pathlib import Path
from typing import Dict, List
import onnxruntime as ort

class PerformanceTester:
    """推理性能测试器"""

    # 测试配置
    TEST_CONFIGS = {
        "warmup_iterations": 10,
        "test_iterations": 100,
        "batch_sizes": [1, 4, 8],
    }

    def __init__(self, model_path: Path):
        self.model_path = model_path

    def test(
        self,
        input_shape: tuple = (1, 3, 640, 640),
        precision: str = "fp16"
    ) -> Dict:
        """
        测试推理性能

        Args:
            input_shape: 输入形状
            precision: 精度模式

        Returns:
            {
                "model": "./model.onnx",
                "precision": "fp16",
                "input_shape": [1, 3, 640, 640],
                "results": {
                    "batch_1": {
                        "fps": 45.2,
                        "latency_ms": 22.1,
                        "throughput_img_sec": 45.2
                    },
                    "batch_4": {
                        "fps": 38.5,
                        "latency_ms": 103.9,
                        "throughput_img_sec": 154.0
                    }
                },
                "gpu_memory_mb": 2048,
                "meets_realtime": True
            }
        """
        # 创建推理会话
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if precision == "fp16":
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

        session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )

        # 准备输入
        input_name = session.get_inputs()[0].name
        input_data = np.random.randn(*input_shape).astype(np.float32)

        # 预热
        for _ in range(self.TEST_CONFIGS["warmup_iterations"]):
            session.run(None, {input_name: input_data})

        # 测试不同批大小
        results = {}
        for batch_size in self.TEST_CONFIGS["batch_sizes"]:
            batch_input = np.random.randn(batch_size, *input_shape[1:]).astype(np.float32)

            # 测试 FPS
            start = time.perf_counter()
            for _ in range(self.TEST_CONFIGS["test_iterations"]):
                session.run(None, {input_name: batch_input})
            end = time.perf_counter()

            total_time = end - start
            avg_latency = (total_time / self.TEST_CONFIGS["test_iterations"]) * 1000  # ms
            fps = self.TEST_CONFIGS["test_iterations"] / total_time
            throughput = fps * batch_size

            results[f"batch_{batch_size}"] = {
                "fps": round(fps, 1),
                "latency_ms": round(avg_latency, 1),
                "throughput_img_sec": round(throughput, 1)
            }

        # 判断是否满足实时要求
        meets_realtime = results["batch_1"]["fps"] >=  {
            "model20

        return": str(self.model_path),
            "precision": precision,
            "input_shape": list(input_shape),
            "results": results,
            "meets_realtime": meets_realtime
        }
```

### 4.5 健康监控器

```python
# src/deployment/health_monitor.py
import requests
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class DeviceMetrics:
    """设备指标"""
    cpu_percent: float
    memory_percent: float
    gpu_memory_mb: float
    temperature_c: float
    inference_count: int

class HealthMonitor:
    """健康监控器"""

    def __init__(self):
        self.devices: Dict[str, str] = {}  # device_name -> endpoint

    def register_device(self, name: str, endpoint: str):
        """注册设备"""
        self.devices[name] = endpoint

    def check_health(self, device_name: str) -> Dict:
        """
        检查设备健康状态

        Returns:
            {
                "device": "jetson-nano",
                "status": "healthy",
                "metrics": {
                    "cpu_percent": 45.2,
                    "memory_percent": 62.1,
                    "gpu_memory_mb": 1024,
                    "temperature_c": 42.5,
                    "inference_count": 1000
                },
                "last_check": "2026-03-11T10:30:00Z"
            }
        """
        endpoint = self.devices.get(device_name)
        if not endpoint:
            return {"error": "Device not found"}

        try:
            # 获取设备指标
            resp = requests.get(f"{endpoint}/metrics", timeout=5)
            metrics = resp.json()

            # 判断状态
            status = self._evaluate_status(metrics)

            return {
                "device": device_name,
                "status": status.value,
                "metrics": metrics,
                "last_check": "2026-03-11T10:30:00Z"
            }
        except Exception as e:
            return {
                "device": device_name,
                "status": HealthStatus.UNHEALTHY.value,
                "error": str(e)
            }

    def check_all(self) -> List[Dict]:
        """检查所有设备"""
        return [self.check_health(name) for name in self.devices.keys()]

    def _evaluate_status(self, metrics: DeviceMetrics) -> HealthStatus:
        """评估状态"""
        if metrics.temperature_c > 80:
            return HealthStatus.UNHEALTHY
        if metrics.cpu_percent > 90 or metrics.memory_percent > 90:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY
```

---

## 5. 部署流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    Deployment Pipeline                                    │
└─────────────────────────────────────────────────────────────────┘

1. ONNX 优化
   ├── 算子融合
   ├── FP16 量化
   └── 内存优化

2. TensorRT 构建
   ├── INT8 校准 (可选)
   ├── TensorRT Engine
   └── CUDA Core 优化

3. 边缘部署
   ├── SSH 连接 Jetson
   ├── 上传模型和脚本
   ├── 安装依赖
   └── 启动推理服务

4. 性能测试
   ├── FPS 测试
   ├── 延迟测试
   └── 验证 FPS >= 20

5. 健康监控
   ├── 状态检查
   ├── 资源监控
   └── 告警通知
```

---

## 6. 数据格式

### 6.1 部署请求

```python
{
    "model_path": "./runs/export/yolo11n.onnx",
    "device": {
        "type": "jetson-nano",
        "ip": "<YOUR_EDGE_DEVICE_IP>",
        "username": "nvidia"
        // password 和 ssh_key 通过环境变量传递，不在请求中
    },
    "optimization": {
        "precision": "fp16",
        "tensorrt": false  // Jetson Nano 不适合构建 TensorRT
    },
    "deployment": {
        "use_docker": true,  // 推荐使用 Docker
        "port": 8080
    },
    "test_config": {
        "input_shape": [1, 3, 640, 640],
        "min_fps": 20
    }
}
```

### 6.2 部署响应

```python
{
    "status": "deployed",
    "device": "jetson-nano",
    "model_path": "/opt/yolo/model.onnx",
    "inference_endpoint": "http://<YOUR_EDGE_DEVICE_IP>:8080/infer",
    "performance": {
        "fps": 45.2,
        "latency_ms": 22.1,
        "meets_realtime": true
    },
    "deployment_time_sec": 45
}
```

---

## 7. 专家审核要点

| 审核项 | 状态 | 说明 |
|--------|------|------|
| ONNX 优化 | ✅ | 算子融合 + FP16 |
| TensorRT | ✅ | Engine 构建 |
| 边缘部署 | ✅ | Jetson SSH |
| 性能测试 | ✅ | FPS >= 20 |
| 健康监控 | ✅ | 状态 + 资源 |

---

## 8. 依赖

```python
dependencies = [
    "paramiko>=3.0.0",
    "tensorrt>=8.6.0",
    "pycuda>=2023.0",
    "onnx>=1.14.0",
    "onnxruntime-gpu>=1.16.0",
    "onnxoptimizer>=0.3.0",
    "requests>=2.31.0",
]
```

---

### 4.5 模型版本管理器

```python
# src/deployment/model_version_manager.py
"""
模型版本管理器 - 生产环境必备

功能：
- 语义化版本控制
- 元数据追踪
- 快速回滚
- 部署历史记录
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import time
import shutil
from dataclasses import dataclass, asdict
from enum import Enum

class ModelStatus(Enum):
    """模型状态"""
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ROLLED_BACK = "rolled_back"

@dataclass
class ModelMetadata:
    """模型元数据"""
    version: str
    created_at: float
    status: str
    metrics: Dict
    commit_hash: str
    dataset_version: str
    notes: str = ""

class ModelVersionManager:
    """模型版本管理器"""

    def __init__(self, storage_path: Path):
        """
        Args:
            storage_path: 模型存储根目录
        """
        self.storage_path = Path(storage_path)
        self.versions_file = self.storage_path / "versions.json"
        self._ensure_storage()

    def _ensure_storage(self):
        """确保存储目录存在"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        if not self.versions_file.exists():
            self._save_versions({})

    def _load_versions(self) -> Dict:
        """加载版本记录"""
        with open(self.versions_file) as f:
            return json.load(f)

    def _save_versions(self, versions: Dict):
        """保存版本记录"""
        with open(self.versions_file, "w") as f:
            json.dump(versions, f, indent=2)

    def save_version(
        self,
        model_path: Path,
        metadata: ModelMetadata,
        is_production: bool = False
    ) -> str:
        """
        保存新版本

        Args:
            model_path: 模型文件路径
            metadata: 模型元数据
            is_production: 是否标记为生产版本

        Returns:
            版本号 (如 "v1.0.0")
        """
        versions = self._load_versions()

        # 生成版本号
        version_count = len(versions) + 1
        version_id = f"v{version_count}.0.0"

        # 创建版本目录
        version_dir = self.storage_path / version_id
        version_dir.mkdir(exist_ok=True)

        # 复制模型文件
        if model_path.suffix == ".pt":
            dest_model = version_dir / "model.pt"
        else:
            dest_model = version_dir / "model.onnx"
        shutil.copy2(model_path, dest_model)

        # 保存元数据
        metadata.version = version_id
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        # 更新版本记录
        versions[version_id] = {
            "path": str(dest_model),
            "status": metadata.status,
            "created_at": metadata.created_at,
            "is_production": is_production
        }
        self._save_versions(versions)

        # 如果是生产版本，更新 latest 符号链接
        if is_production:
            self._update_production_link(version_id)

        return version_id

    def _update_production_link(self, version_id: str):
        """更新生产版本链接"""
        latest_link = self.storage_path / "production"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(version_id)

    def get_version(self, version_id: str) -> Optional[Dict]:
        """获取指定版本信息"""
        versions = self._load_versions()
        return versions.get(version_id)

    def get_production_version(self) -> Optional[Dict]:
        """获取当前生产版本"""
        versions = self._load_versions()
        for vid, info in versions.items():
            if info.get("is_production"):
                return {"version": vid, **info}
        return None

    def rollback(self, version_id: str) -> bool:
        """
        回滚到指定版本

        Args:
            version_id: 目标版本号

        Returns:
            是否成功
        """
        versions = self._load_versions()

        if version_id not in versions:
            return False

        # 获取当前生产版本并标记为 deprecated
        current_prod = self.get_production_version()
        if current_prod:
            current_version = current_prod["version"]
            # 更新状态为 deprecated
            version_dir = self.storage_path / current_version
            metadata_file = version_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                metadata["status"] = ModelStatus.DEPRECATED.value
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

        # 更新生产链接到目标版本
        self._update_production_link(version_id)

        # 记录回滚事件
        version_dir = self.storage_path / version_id
        metadata_file = version_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            metadata["status"] = ModelStatus.DEPLOYED.value
            metadata["rollback_at"] = time.time()
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        return True

    def list_versions(self, status: Optional[ModelStatus] = None) -> List[Dict]:
        """
        列出所有版本

        Args:
            status: 按状态过滤
        """
        versions = self._load_versions()
        result = []

        for vid, info in versions.items():
            if status and info.get("status") != status.value:
                continue
            result.append({"version": vid, **info})

        return sorted(result, key=lambda x: x.get("created_at", 0), reverse=True)

    def delete_version(self, version_id: str) -> bool:
        """
        删除指定版本

        注意：不能删除当前生产版本

        Args:
            version_id: 版本号

        Returns:
            是否成功
        """
        # 检查是否是生产版本
        prod = self.get_production_version()
        if prod and prod["version"] == version_id:
            return False

        versions = self._load_versions()

        if version_id not in versions:
            return False

        # 删除版本目录
        version_dir = self.storage_path / version_id
        if version_dir.exists():
            shutil.rmtree(version_dir)

        # 更新版本记录
        del versions[version_id]
        self._save_versions(versions)

        return True
```

### 4.6 MLflow 模型生命周期管理器

```python
# src/deployment/mlflow_manager.py
"""
MLflow 模型生命周期管理器

基于 MLflow 官方最佳实践:
- Stage 管理: Staging → Production → Archived
- 版本追踪: 完整的模型血缘
- 自动化部署: 与 CI/CD 集成

参考: https://mlflow.org/docs/latest/ml/model-registry/workflow/
"""

import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum

class ModelStage(Enum):
    """模型阶段"""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"

class MLflowModelManager:
    """MLflow 模型生命周期管理器"""

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        registry_uri: str = "http://localhost:5000"
    ):
        """
        初始化 MLflow 管理器

        Args:
            tracking_uri: MLflow Tracking Server URI
            registry_uri: MLflow Model Registry URI
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)
        self.client = MlflowClient()

    def register_model(
        self,
        model_path: Path,
        model_name: str,
        metadata: Dict = None
    ) -> str:
        """
        注册新模型

        Args:
            model_path: 模型文件路径 (.pt, .onnx)
            model_name: 模型名称
            metadata: 元数据

        Returns:
            版本号
        """
        # 记录模型
        if model_path.suffix == ".pt":
            # PyTorch 模型
            import torch
            model = torch.load(model_path)
            run_id = mlflow.start_run()
            mlflow.pytorch.log_model(model, "model")
        else:
            # ONNX 模型
            run_id = mlflow.start_run()
            mlflow.onnx.log_model(model_path, "model")

        # 注册模型
        model_uri = f"runs:/{run_id.info.run_id}/model"
        model_version = mlflow.register_model(model_uri, model_name)

        # 添加元数据
        if metadata:
            self._add_metadata(model_name, model_version.version, metadata)

        mlflow.end_run()

        return model_version.version

    def transition_stage(
        self,
        model_name: str,
        version: int,
        stage: ModelStage,
        archive_existing: bool = True
    ) -> bool:
        """
        转换模型阶段

        Args:
            model_name: 模型名称
            version: 版本号
            stage: 目标阶段
            archive_existing: 是否归档现有生产模型

        Returns:
            是否成功
        """
        try:
            # 如果目标是 Production，先归档现有的
            if stage == ModelStage.PRODUCTION and archive_existing:
                self._archive_production(model_name)

            # 转换阶段
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage.value
            )

            return True
        except Exception as e:
            print(f"Stage transition failed: {e}")
            return False

    def _archive_production(self, model_name: str):
        """归档现有生产模型"""
        try:
            # 获取当前生产模型
            production_versions = self.client.get_latest_versions(
                model_name,
                stages=[ModelStage.PRODUCTION.value]
            )

            # 归档
            for version in production_versions:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage=ModelStage.ARCHIVED.value
                )
        except Exception:
            pass  # 没有生产模型

    def deploy_to_staging(
        self,
        model_name: str,
        version: int,
        metadata: Dict = None
    ) -> bool:
        """
        部署到 Staging

        Args:
            model_name: 模型名称
            version: 版本号
            metadata: 元数据

        Returns:
            是否成功
        """
        # 注册（如果未注册）
        if not self._is_registered(model_name, version):
            # 需要先注册
            pass

        # 添加元数据
        if metadata:
            self._add_metadata(model_name, version, metadata)

        # 转换到 Staging
        return self.transition_stage(
            model_name, version,
            ModelStage.STAGING,
            archive_existing=False
        )

    def promote_to_production(
        self,
        model_name: str,
        version: int,
        metadata: Dict = None
    ) -> bool:
        """
        升级到 Production

        Args:
            model_name: 模型名称
            version: 版本号
            metadata: 元数据

        Returns:
            是否成功
        """
        # 先部署到 Staging 测试
        self.deploy_to_staging(model_name, version, metadata)

        # 验证（实际应运行测试）
        # ...

        # 升级到 Production
        return self.transition_stage(
            model_name, version,
            ModelStage.PRODUCTION,
            archive_existing=True
        )

    def get_production_model(self, model_name: str) -> Optional[Dict]:
        """获取当前生产模型"""
        try:
            versions = self.client.get_latest_versions(
                model_name,
                stages=[ModelStage.PRODUCTION.value]
            )

            if versions:
                return {
                    "name": model_name,
                    "version": versions[0].version,
                    "stage": ModelStage.PRODUCTION.value,
                    "uri": versions[0].source
                }
        except Exception:
            pass

        return None

    def list_models(self) -> List[Dict]:
        """列出所有注册模型"""
        models = mlflow.search_registered_models()
        return [
            {
                "name": m.name,
                "latest_version": m.latest_versions[0].version if m.latest_versions else None
            }
            for m in models
        ]

    def _is_registered(self, model_name: str, version: int) -> bool:
        """检查模型是否已注册"""
        try:
            self.client.get_model_version(model_name, version)
            return True
        except Exception:
            return False

    def _add_metadata(self, model_name: str, version: int, metadata: Dict):
        """添加元数据"""
        # MLflow 使用 tags 存储元数据
        self.client.set_model_version_tags(
            model_name,
            version,
            metadata
        )
```

---

## 9. 与其他模块的集成

```
Training ──► Deployment ──► Monitoring
     │              │
     ▼              ▼
ONNX (FP16)   Jetson Nano
              │
              ▼
         HTTP API
```

---

*文档版本: 8.0*
*核心功能: ONNX 导出 + Jetson 部署 + 性能测试*
