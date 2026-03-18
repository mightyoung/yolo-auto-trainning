# ADR 0006: Data Platform Architecture

## Status
Accepted

## Context
We need a system for discovering, managing, and versioning training data. The platform should support multiple data sources and ensure data quality before training.

## Decision
Use a multi-source data discovery system with quality gates:

### Data Sources Supported
1. **Roboflow**: Public and private datasets, annotation tools
2. **Kaggle**: Competition datasets, community datasets
3. **HuggingFace**: Pre-trained datasets, community datasets

### Data Discovery Implementation
- `DatasetDiscovery` class searches across sources
- Relevance scoring based on task keywords
- Returns dataset metadata (size, classes, format)

### Dataset Format
- YOLO format (YAML with train/val/test paths)
- Automatic conversion from other formats
- Class mapping management

## Gaps (NOT YET IMPLEMENTED)

### Current State
- Dataset discovery API endpoints exist (Roboflow, Kaggle, HuggingFace)
- Basic search and metadata retrieval implemented
- Dataset download to local storage implemented
- DVC for versioning NOT integrated
- Data quality gates NOT implemented

### What Needs to Be Done

#### 1. DVC Integration (Data Version Control)
- Initialize DVC in project
- Track dataset versions in Git
- Support dataset sharing via DVC remote
- Pipeline integration for reproducible training

#### 2. Data Quality Gates
- Automatic validation on dataset download
- Checks: class distribution, image quality, annotation validity
- Minimum dataset size requirements
- Duplicate detection
- Report generation before training

#### 3. Data Pipeline
- Automated data preprocessing
- Data augmentation pipelines
- Train/val/test split management
- Cache management for downloaded datasets

## Architecture Diagram (Future State)
```
┌─────────────────────────────────────────────────────────────┐
│                    Data Platform                            │
├─────────────────────────────────────────────────────────────┤
│  Discovery Layer                                           │
│  ├── Roboflow API                                          │
│  ├── Kaggle API                                           │
│  └── HuggingFace API                                      │
├─────────────────────────────────────────────────────────────┤
│  Quality Gates                                             │
│  ├── Class Distribution Check                              │
│  ├── Image Quality Validation                              │
│  ├── Annotation Validation                                 │
│  └── Minimum Size Check                                    │
├─────────────────────────────────────────────────────────────┤
│  Version Control (DVC)                                     │
│  ├── Dataset Versioning                                    │
│  ├── Pipeline Tracking                                     │
│  └── Remote Storage                                        │
└─────────────────────────────────────────────────────────────┘
```

## Consequences

### Easier
- Standardized dataset format across sources
- Reproducible experiments via data versioning
- Quality gates prevent bad training runs

### More Difficult
- DVC requires Git workflow integration
- Additional storage for dataset versions
- Quality gate configuration per project
- Network overhead for dataset downloads
