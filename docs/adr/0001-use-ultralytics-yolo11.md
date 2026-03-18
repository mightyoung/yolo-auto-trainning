# ADR 0001: Use Ultralytics YOLO11

## Status
Accepted

## Context
We need to select a base model for object detection training. The project requires a well-maintained, production-ready YOLO implementation with active community support and regular updates.

## Decision
Use Ultralytics YOLO11 as the base model for object detection training.

### Model Selection
- **Primary Model**: YOLO11 (yolo11m for balanced speed/accuracy)
- **Variant Options**: yolo11n (nano), yolo11s (small), yolo11m (medium), yolo11l (large), yolo11x (extra-large)

### Training Approach
- **Standard Training**: Direct training with configurable epochs, batch size, and image size
- **Hyperparameter Optimization**: Ray Tune for HPO (see ADR-0002)
- **Transfer Learning**: Support for pretrained weight fine-tuning with frozen backbone layers
- **Knowledge Distillation**: Support for teacher-student distillation using official Ultralytics API

### IMPORTANT: Knowledge Distillation Limitation
YOLO11 does NOT have built-in knowledge distillation support in the traditional sense. The current implementation uses the official Ultralytics `teacher` parameter which was available in earlier versions but may have limited functionality. For true KD:
- Consider using YOLOv8 with `teacher` parameter
- Or implement custom KD loss outside the official API

## Consequences

### Easier
- Well-documented API with comprehensive tutorials
- Active community and fast bug fixes
- Built-in model export to multiple formats (ONNX, TensorRT, TorchScript)
- Integrated hyperparameter tuning via Ray Tune

### More Difficult
- Custom model modifications require understanding ultralytics internals
- Knowledge distillation is limited (not a native first-class feature)
- Need to monitor Ultralytics releases for API changes
