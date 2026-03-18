# ADR 0002: MLflow for Experiment Tracking

## Status
Accepted

## Context
We need a centralized system for tracking machine learning experiments, including metrics, parameters, artifacts, and model versions. The system should integrate with the training pipeline and provide a UI for comparing runs.

## Decision
Use MLflow as the experiment tracking and model registry solution.

### MLflow Components Used
1. **MLflow Tracking Server**: For logging experiments, metrics, parameters, and artifacts
2. **MLflow Model Registry**: For versioning and managing trained models

### Implementation Details
- **Tracking URI**: Configurable via `MLFLOW_TRACKING_URI` environment variable
- **Default**: `http://localhost:5000` (local MLflow server)
- **Experiment Name**: YOLO training runs grouped by dataset/task
- **Artifacts**: Model weights, training logs, evaluation results stored in artifact store

### Integration Points
- Training runner logs metrics via `MLflowTracker` class
- Model exports automatically logged as artifacts
- Training parameters stored for reproducibility

## Gaps (NOT YET IMPLEMENTED)

### Current State
- `src/training/mlflow_tracker.py` exists but integration with `YOLOTrainer` is NOT complete
- MLflow is listed in requirements but automatic logging is not wired up
- No MLflow server configured in docker-compose by default

### What Needs to Be Done
1. **Integrate MLflow with YOLOTrainer**: Add automatic metric logging during training
2. **Configure MLflow Server**: Add MLflow container to docker-compose
3. **Model Registry Setup**: Configure model registry for version tracking
4. **Artifact Storage**: Configure artifact storage (local filesystem or cloud)
5. **Callback Integration**: Add MLflow callbacks for training events

## Consequences

### Easier
- Standard ML experiment tracking across all training runs
- Model versioning and lineage tracking
- UI for comparing experiments and visualizing metrics
- Integration with popular ML frameworks

### More Difficult
- Need to maintain MLflow server infrastructure
- Artifact storage management at scale
- Requires additional configuration for production deployment
