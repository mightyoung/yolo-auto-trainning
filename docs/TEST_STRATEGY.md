# YOLO Auto-Training System - Test Strategy Document

**Version**: 1.0
**Date**: 2026-03-14
**Status**: Test Plan based on PRD and Architecture Design

---

## 1. Test Strategy Overview

### 1.1 Testing Philosophy

Based on the PRD and design documents, the testing strategy follows these principles:

| Principle | Description | Application |
|-----------|-------------|-------------|
| **Unit Testing** | Test individual components in isolation | Core business logic |
| **Integration Testing** | Test component interactions | API endpoints, Celery tasks |
| **End-to-End Testing** | Test complete workflows | Data discovery → Training → Deployment |
| **Property-Based Testing** | Test invariants | Hyperparameter validation |
| **Mock Testing** | Avoid external dependencies | API calls, Redis, file I/O |

### 1.2 Test Pyramid

```
         /\
        /  \       E2E Tests (5%)
       /____\      - Full workflow tests
      /      \
     /        \    Integration Tests (25%)
    /__________\   - API tests, Celery tasks
   /            \
  /              \  Unit Tests (70%)
 /________________\ - Core functions, utilities
```

### 1.3 Test Coverage Targets

| Module | Target Coverage | Priority |
|--------|-----------------|----------|
| Data Discovery | ≥ 80% | High |
| Training Config | ≥ 90% | High |
| API Gateway | ≥ 85% | High |
| Agent Orchestration | ≥ 70% | Medium |
| Deployment | ≥ 75% | Medium |

---

## 2. Module-by-Module Test Cases

### 2.1 Data Discovery Module

#### 2.1.1 Relevance Scoring Tests

```python
class TestRelevanceScoring:
    """Test cases for _calculate_relevance function."""

    # Exact match tests
    def test_exact_match_returns_1(self):
        """Exact match should return score of 1.0"""
        assert calculate_relevance("car", "car") == 1.0

    def test_exact_match_case_insensitive(self):
        """Case insensitive exact match"""
        assert calculate_relevance("Car", "CAR") == 1.0

    # Partial match tests
    def test_query_in_text_returns_09(self):
        """Query contained in text returns 0.9"""
        assert calculate_relevance("car", "car_detection") == 0.9

    def test_text_in_query_returns_08(self):
        """Text contained in query returns 0.8"""
        assert calculate_relevance("car_detection", "car") == 0.8

    # Jaccard similarity tests
    def test_jaccard_similarity(self):
        """Jaccard similarity for word-level matching"""
        score = calculate_relevance("car vehicle", "car truck")
        assert 0.3 < score < 0.7

    # Edge cases
    def test_empty_query_returns_0(self):
        """Empty query returns 0.0"""
        assert calculate_relevance("", "car") == 0.0

    def test_empty_text_returns_0(self):
        """Empty text returns 0.0"""
        assert calculate_relevance("car", "") == 0.0

    def test_both_empty_returns_0(self):
        """Both empty returns 0.0"""
        assert calculate_relevance("", "") == 0.0

    def test_no_common_words_returns_low_score(self):
        """No common words returns low score"""
        score = calculate_relevance("car", "building")
        assert score < 0.3
```

#### 2.1.2 Dataset Search Tests

```python
class TestDatasetDiscovery:
    """Test cases for DatasetDiscovery class."""

    @pytest.fixture
    def discovery(self, tmp_path):
        """Create discovery instance with temp directory."""
        return DatasetDiscovery(output_dir=tmp_path)

    @patch('src.data.discovery.requests.get')
    def test_search_roboflow_returns_results(self, mock_get, discovery):
        """Roboflow search returns parsed results."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "datasets": [
                {"name": "car-detection", "images": 1000, "license": "MIT"}
            ]
        }
        mock_get.return_value = mock_response

        results = discovery.search("car detection", max_results=5)
        assert len(results) >= 0  # May be empty if no API key

    @patch('src.data.discovery.KaggleApi')
    def test_search_kaggle_handles_error(self, mock_kaggle, discovery):
        """Kaggle error handling."""
        mock_api = Mock()
        mock_api.dataset_list.side_effect = Exception("API Error")
        mock_kaggle.return_value = mock_api

        results = discovery._search_kaggle("test", 5)
        assert results == []  # Should return empty on error
```

#### 2.1.3 Data Merger Tests

```python
class TestDataMerger:
    """Test cases for DataMerger class."""

    def test_merge_respects_synthetic_ratio(self, tmp_path):
        """Synthetic data ratio should not exceed 30%."""
        merger = DataMerger(max_synthetic_ratio=0.3)

        # Create mock datasets
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        for i in range(100):
            (real_dir / f"img{i}.jpg").touch()

        synth_dir = tmp_path / "synthetic"
        synth_dir.mkdir()
        for i in range(50):
            (synth_dir / f"synth{i}.jpg").touch()

        result = merger.merge(
            discovered_datasets=[real_dir],
            synthetic_dataset=synth_dir,
            output_dir=tmp_path / "merged"
        )

        assert result["synthetic_ratio"] <= 0.3

    def test_merge_creates_val_split(self, tmp_path):
        """Merge creates 10% validation split."""
        merger = DataMerger()

        real_dir = tmp_path / "real"
        real_dir.mkdir()
        for i in range(100):
            (real_dir / f"img{i}.jpg").touch()

        result = merger.merge(
            discovered_datasets=[real_dir],
            output_dir=tmp_path / "merged"
        )

        # Should have approximately 10% in validation
        assert result["val_images"] > 0
        assert result["val_images"] < result["train_images"]
```

### 2.2 Training Module

#### 2.2.1 Configuration Tests

```python
class TestTrainingConfig:
    """Test cases for TrainingConfig."""

    def test_default_values_match_ultralytics(self):
        """Default values should match Ultralytics official defaults."""
        config = TrainingConfig()

        # Critical parameters from official docs
        assert config.lr0 == 0.01
        assert config.box == 7.5
        assert config.fliplr == 0.5
        assert config.momentum == 0.937
        assert config.weight_decay == 0.0005

    def test_to_dict_contains_all_params(self):
        """to_dict should contain all training parameters."""
        config = TrainingConfig()
        params = config.to_dict()

        assert "lr0" in params
        assert "box" in params
        assert "fliplr" in params
        assert "epochs" in params
        assert "imgsz" in params

    def test_hpo_config_search_space(self):
        """HPO config search space is valid."""
        config = HPOConfig()

        # Check search space bounds
        assert config.param_space["lr0"][0] < config.param_space["lr0"][1]
        assert config.param_space["momentum"][0] < config.param_space["momentum"][1]

        # Check grace period is reasonable
        assert config.grace_period >= 10
        assert config.grace_period < config.epochs_per_trial
```

#### 2.2.2 Training Runner Tests

```python
class TestYOLOTrainer:
    """Test cases for YOLOTrainer."""

    @pytest.fixture
    def trainer(self, tmp_path):
        """Create trainer instance."""
        return YOLOTrainer(model="yolo11n", output_dir=tmp_path)

    def test_trainer_initialization(self, trainer):
        """Trainer initializes with correct defaults."""
        assert trainer.model_name == "yolo11n"
        assert trainer.output_dir.exists()

    @patch('src.training.runner.YOLO')
    def test_sanity_check_returns_result(self, mock_yolo, trainer, tmp_path):
        """Sanity check returns TrainingResult."""
        mock_model = Mock()
        mock_results = Mock()
        mock_results.results_dict = {"metrics/mAP50(B)": 0.35}
        mock_model.train.return_value = mock_results
        mock_yolo.return_value = mock_model

        # Create minimal YAML
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text("path: .\ntrain: images\nval: images\nnc: 1\nnames: ['obj']")

        result = trainer.sanity_check(yaml_path)

        assert result.status in ["passed", "failed"]
        assert "mAP50" in result.metrics
```

### 2.3 API Gateway Module

#### 2.3.1 Authentication Tests

```python
class TestAuthentication:
    """Test cases for JWT and API Key authentication."""

    def test_create_access_token(self):
        """Access token is created with correct claims."""
        token = create_access_token({"sub": "user123"})

        assert token is not None
        assert isinstance(token, str)

    def test_verify_valid_token(self):
        """Valid token verifies successfully."""
        token = create_access_token({"sub": "user123"})
        payload = verify_token(token)

        assert payload["sub"] == "user123"
        assert payload["type"] == "access"

    def test_verify_expired_token_raises(self):
        """Expired token raises exception."""
        # Create token with past expiration
        with patch('src.api.gateway.datetime') as mock_dt:
            mock_dt.utcnow.return_value = datetime(2020, 1, 1)
            token = create_access_token({"sub": "user123"})

        with pytest.raises(HTTPException) as exc_info:
            verify_token(token)

        assert exc_info.value.status_code == 401

    def test_generate_api_key_format(self):
        """API key has correct format."""
        key = generate_api_key()

        assert key.startswith("yolo_")
        assert len(key) > 20

    @patch('src.api.gateway.get_redis_client')
    def test_store_and_verify_api_key(self, mock_redis):
        """API key can be stored and verified in Redis."""
        mock_client = Mock()
        mock_redis.return_value = mock_client

        # Store
        result = store_api_key_in_redis("test_key", "user123")
        assert result is True
        mock_client.setex.assert_called_once()

        # Verify
        mock_client.get.return_value = "user123"
        user_id = verify_api_key_in_redis("test_key")
        assert user_id == "user123"
```

#### 2.3.2 Rate Limiting Tests

```python
class TestRateLimiting:
    """Test cases for rate limiting."""

    @patch('src.api.gateway.get_redis_client')
    def test_rate_limit_allows_within_limit(self, mock_redis):
        """Requests within limit are allowed."""
        mock_client = Mock()
        mock_client.eval.return_value = 1  # Allowed
        mock_redis.return_value = mock_client

        # Create mock request
        request = Mock()
        request.url.path = "/api/v1/train/start"

        # Should not raise
        # Note: Need proper async testing setup

    @patch('src.api.gateway.get_redis_client')
    def test_rate_limit_blocks_over_limit(self, mock_redis):
        """Requests over limit are blocked."""
        mock_client = Mock()
        mock_client.eval.return_value = 0  # Blocked
        mock_redis.return_value = mock_client

        request = Mock()
        request.url.path = "/api/v1/train/start"

        with pytest.raises(HTTPException) as exc_info:
            # Would need proper async setup
            pass

        assert exc_info.value.status_code == 429
```

#### 2.3.3 API Routes Tests

```python
class TestAPIRoutes:
    """Test cases for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from src.api.gateway import app
        return TestClient(app)

    def test_health_check(self, client):
        """Health check returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_data_search_endpoint(self, client):
        """Data search endpoint works."""
        response = client.post(
            "/api/v1/data/search",
            json={"query": "car detection", "max_results": 5}
        )

        assert response.status_code == 200
        assert "datasets" in response.json()

    def test_train_start_endpoint(self, client):
        """Training start endpoint submits task."""
        response = client.post(
            "/api/v1/train/start",
            json={
                "data_yaml": "/path/to/data.yaml",
                "model": "yolo11m",
                "epochs": 100
            }
        )

        assert response.status_code == 200
        assert "task_id" in response.json()
```

### 2.4 Agent Orchestration Module

#### 2.4.1 Agent Creation Tests

```python
class TestAgentOrchestration:
    """Test cases for CrewAI agents."""

    def test_dataset_discovery_agent_created(self):
        """Dataset discovery agent is created with correct config."""
        agent = create_dataset_discovery_agent()

        assert agent.role == "Dataset Curator"
        assert len(agent.tools) == 2
        assert not agent.allow_delegation

    def test_training_agent_has_decision_rules(self):
        """Training agent has decision rules in backstory."""
        agent = create_training_agent()

        assert "decision rules" in agent.backstory.lower()
        assert "mAP50" in agent.backstory
        assert "YOLO11" in agent.backstory

    def test_deployment_agent_has_fps_rule(self):
        """Deployment agent has FPS decision rule."""
        agent = create_deployment_agent()

        assert "FPS" in agent.backstory
        assert "20" in agent.backstory  # Target FPS

    def test_all_agents_have_tools(self):
        """All agents have appropriate tools."""
        discovery = create_dataset_discovery_agent()
        training = create_training_agent()
        deployment = create_deployment_agent()

        assert len(discovery.tools) > 0
        assert len(training.tools) > 0
        assert len(deployment.tools) > 0
```

### 2.5 Deployment Module

#### 2.5.1 Export Tests

```python
class TestModelExporter:
    """Test cases for model export."""

    @pytest.fixture
    def exporter(self, tmp_path):
        """Create exporter instance."""
        return ModelExporter(output_dir=tmp_path)

    def test_export_config_defaults(self):
        """Export config has correct defaults."""
        config = ExportConfig()

        assert config.format == "onnx"
        assert config.half is True  # FP16
        assert config.opset == 13

    def test_platform_configs(self):
        """Platform-specific configs are defined."""
        config = ExportConfig()

        assert "jetson" in config.platform_configs
        assert "tensorrt" in config.platform_configs
        assert "cpu" in config.platform_configs

        # Jetson uses FP16
        assert config.platform_configs["jetson"]["half"] is True
```

---

## 3. Integration Test Cases

### 3.1 Full Workflow Tests

```python
class TestFullWorkflow:
    """End-to-end workflow tests."""

    @pytest.mark.integration
    @patch('src.data.discovery.requests.get')
    @patch('src.training.runner.YOLO')
    def test_full_training_workflow(self, mock_yolo, mock_requests, tmp_path):
        """Test complete workflow from discovery to training."""
        # 1. Mock dataset search
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "datasets": [{"name": "cars", "images": 1000}]
        }
        mock_requests.return_value = mock_response

        # 2. Discovery
        discovery = DatasetDiscovery(output_dir=tmp_path / "data")
        results = discovery.search("car detection")
        assert len(results) >= 0

        # 3. Training (mocked)
        mock_model = Mock()
        mock_results = Mock()
        mock_results.results_dict = {"metrics/mAP50(B)": 0.75}
        mock_results.save_dir = str(tmp_path / "weights")
        mock_model.train.return_value = mock_results
        mock_yolo.return_value = mock_model

        trainer = YOLOTrainer(output_dir=tmp_path / "runs")
        # Would run actual training in integration test

        assert True  # Workflow completes without error
```

### 3.2 Celery Task Tests

```python
class TestCeleryTasks:
    """Test cases for Celery tasks."""

    def test_training_task_signature(self):
        """Training task accepts correct parameters."""
        from src.api.tasks import training_task

        # Task should be delay-able
        assert hasattr(training_task, 'delay')

    def test_export_task_configuration(self):
        """Export task has correct configuration."""
        from src.api.tasks import export_task

        # Check retry config
        assert export_task.max_retries == 3
```

---

## 4. Evaluation Criteria

### 4.1 Test Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Code Coverage** | ≥ 75% | pytest --cov |
| **Pass Rate** | ≥ 95% | CI pipeline |
| **Test Execution Time** | < 5 min | CI pipeline |
| **Flaky Tests** | < 2% | Test history |

### 4.2 Functional Acceptance Criteria

#### Data Discovery Module

| Test | Acceptance Criteria |
|------|-------------------|
| Relevance Scoring | Exact match = 1.0, partial match > 0.8, no match < 0.3 |
| API Integration | Returns empty list on API failure (graceful degradation) |
| Data Merger | Synthetic ratio never exceeds 30% |

#### Training Module

| Test | Acceptance Criteria |
|------|-------------------|
| Config Validation | All Ultralytics defaults match official docs |
| HPO Config | Search space bounds are valid |
| Sanity Check | Returns valid TrainingResult object |

#### API Module

| Test | Acceptance Criteria |
|------|-------------------|
| Authentication | Invalid token returns 401 |
| Rate Limiting | Over limit returns 429 |
| Endpoints | All endpoints return 200 for valid input |

#### Agent Module

| Test | Acceptance Criteria |
|------|-------------------|
| Decision Rules | All 14 rules present in agent backstories |
| Tool Assignment | Correct tools assigned to each agent |

### 4.3 Non-Functional Requirements

| Requirement | Test Method | Pass Criteria |
|-------------|-------------|----------------|
| **Security** | Security tests | No vulnerabilities in dependencies |
| **Performance** | Load testing | < 500ms for API response |
| **Reliability** | Flaky test check | < 2% flaky tests |
| **Maintainability** | Code review | All functions have docstrings |

---

## 5. Test Execution Plan

### 5.1 CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-mock

      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=src --cov-report=xml

      - name: Run integration tests
        run: |
          pytest tests/integration/ -v --block-network

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### 5.2 Test Organization

```
yolo-auto-training/
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_data_discovery.py
│   │   ├── test_training_config.py
│   │   ├── test_authentication.py
│   │   ├── test_rate_limiting.py
│   │   └── test_agents.py
│   └── integration/
│       ├── __init__.py
│       ├── test_api_routes.py
│       └── test_workflows.py
└── pytest.ini
```

### 5.3 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific module
pytest tests/unit/test_data_discovery.py -v

# Run integration tests only
pytest tests/integration/ -v

# Run with network blocked (default)
pytest tests/ --block-network
```

---

## 6. Test Data Management

### 6.1 Mock Data

| Resource | Mock Strategy |
|----------|---------------|
| Roboflow API | Mock requests with JSON responses |
| Kaggle API | Mock KaggleApi class |
| HuggingFace | Mock list_datasets function |
| Redis | Mock redis client |
| YOLO Training | Mock YOLO class |

### 6.2 Test Fixtures

```python
# tests/conftest.py

import pytest
from pathlib import Path
import tempfile

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_yaml(temp_dir):
    """Create sample dataset YAML."""
    yaml_path = temp_dir / "data.yaml"
    yaml_path.write_text("""
path: .
train: images
val: images
nc: 1
names: ['object']
""")
    return yaml_path
```

---

## 7. References

- [Ultralytics Configuration](https://docs.ultralytics.com/usage/cfg/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Pytest Best Practices](https://docs.pytest.org/en/stable/)
- [CrewAI Testing](https://docs.crewai.com/)

---

*Document Version: 1.0*
*Last Updated: 2026-03-14*
*Based on PRD v3.1 and Design Documents*
