# ADR 0005: Deployment Targets

## Status
Accepted

## Context
We need to support multiple deployment platforms for trained YOLO models. Different hardware requires different optimization formats for optimal inference performance.

## Decision
Support multiple export formats with priority deployment targets:

### Supported Formats

| Format | Platform | Precision | Status |
|--------|----------|-----------|--------|
| ONNX | General/CPU | FP32, FP16 | Implemented |
| TensorRT | NVIDIA Jetson/GPU | FP32, FP16, INT8 | Implemented |
| TorchScript | PyTorch | FP32 | Implemented |
| RKNN | Rockchip NPU | FP16, INT8 | Planned (future) |

### Platform-Specific Configurations

#### NVIDIA Jetson Orin
- **Format**: TensorRT Engine (.engine)
- **Precision**: FP16 (default), INT8 (with calibration)
- **Dynamic**: Yes (for variable input sizes)
- **Workspace**: 4GB

#### NVIDIA Jetson Nano
- **Format**: TensorRT Engine (.engine)
- **Precision**: FP16
- **Dynamic**: No (fixed input)
- **Workspace**: 2GB

#### General Deployment (CPU/Cloud)
- **Format**: ONNX
- **Precision**: FP32 (default), FP16
- **Simplify**: Yes (ONNX optimizer)
- **Opset**: 13

### Implementation
- `ModelExporter` class handles all export logic
- `EdgeDeployer` class handles deployment to devices
- Benchmark data available for Jetson platforms

## Gaps (NOT YET IMPLEMENTED)

### Current State
- Export functionality implemented
- Edge deployment (SCP to Jetson) is a placeholder
- INT8 calibration requires manual dataset
- RKNN export NOT implemented

### What Needs to Be Done
1. **Edge Deployment**: Implement actual SCP/SSH deployment to Jetson devices
2. **RKNN Support**: Add Rockchip NPU export (requires rknn-toolkit2)
3. **TFLite Export**: Add TensorFlow Lite for mobile edge devices
4. **Auto-Detection**: Auto-detect best format based on target device
5. **Deployment Verification**: Add inference testing after deployment

## Consequences

### Easier
- Single training pipeline for multiple deployment targets
- Platform-specific optimizations available
- Clear export path for production deployment

### More Difficult
- Need to test on actual target hardware
- Different precision levels require validation
- Platform-specific build tools required (TensorRT, RKNN)
