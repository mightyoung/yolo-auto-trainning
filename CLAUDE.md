# CLAUDE.md

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **yolo-auto-training** (782 symbols, 1436 relationships, 59 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/yolo-auto-training/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/yolo-auto-training/context` | Codebase overview, check index freshness |
| `gitnexus://repo/yolo-auto-training/clusters` | All functional areas |
| `gitnexus://repo/yolo-auto-training/processes` | All execution flows |
| `gitnexus://repo/yolo-auto-training/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

- Re-index: `npx gitnexus analyze`
- Check freshness: `npx gitnexus status`
- Generate docs: `npx gitnexus wiki`

<!-- gitnexus:end -->

---

## Build, Test & Development Commands

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=src --cov=business-api --cov=training-api

# Run a single test file
pytest tests/unit/test_training_config.py -v

# Run linting
ruff check src/ business-api/ training-api/

# Type checking
mypy src/ business-api/ training-api/
```

### Service Entry Points

```bash
# Business API (port 8000) - handles data discovery, agent orchestration, task scheduling
uvicorn business-api.src.api.gateway:app --host 0.0.0.0 --port 8000 --reload

# Training API (port 8001) - runs on GPU server, handles YOLO training, HPO, model export
uvicorn training-api.src.api.gateway:app --host 0.0.0.0 --port 8001 --reload

# Legacy monolithic API
uvicorn src.api.gateway:app --host 0.0.0.0 --port 8080 --reload

# Celery worker for async tasks
celery -A business-api.src.api.tasks worker --loglevel=info
```

### Docker Compose

```bash
# All-in-one: Redis + Business API + Celery worker + GPU training
docker-compose up -d --build

# With full MLOps stack (Prometheus + Grafana + ELK)
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

---

## Architecture Overview

### Three-API Architecture

The system consists of **three independent FastAPI services** that communicate over HTTP:

```
┌─────────────────────────────────────────────────────┐
│  Business API (port 8000) — business-api/src/api/   │
│  Entry: gateway.py                                  │
│  Handles: Auth, Dataset Discovery, CrewAI Agents,   │
│           Task Scheduling → Training API            │
└─────────────────────┬───────────────────────────────┘
                      │ Internal HTTP
┌─────────────────────▼───────────────────────────────┐
│  Training API (port 8001) — training-api/src/api/   │
│  Entry: gateway.py                                  │
│  Handles: YOLO Training, Ray Tune HPO, Model Export│
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Legacy API (port 8080) — src/api/                  │
│  Entry: gateway.py                                  │
│  Handles: End-to-end pipeline, Celery tasks        │
└─────────────────────────────────────────────────────┘
```

### Business API Routes (`business-api/src/api/`)

| Router | File | Purpose |
|--------|------|---------|
| `data_router` | routes.py | Roboflow/Kaggle/HuggingFace dataset search |
| `train_router` | routes.py | Training job submission via TrainingAPIClient |
| `deploy_router` | routes.py | Model export orchestration |
| `agent_router` | agent_routes.py | CrewAI multi-agent workflow triggers |
| `callback_router` | routes.py | MLflow/webhook callbacks |
| `analysis_router` | routes.py | Deepanalyze endpoints |

### Training API Routes (`training-api/src/api/`)

| Router | File | Purpose |
|--------|------|---------|
| `internal_router` | routes.py | Internal training endpoints (YOLO runner, HPO) |
| `model_router` | model_routes.py | Model management (list, delete, download) |

### Key Classes

- **`YOLOTrainer`** (`training-api/src/training/runner.py`) — wraps ultralytics YOLO with MLflow tracking, supports HPO via Ray Tune
- **`PipelineExecutor`** (`src/pipeline/orchestrator.py`) — task dependency graph executor for end-to-end pipelines
- **`DatasetDiscovery`** (`src/data/discovery.py`) — multi-source dataset search with relevance scoring
- **CrewAI Agents** (`business-api/src/agents/orchestration.py`) — Dataset Curator, Data Engineer, ML Engineer, DevOps Engineer agents

### Service Communication

- Business API → Training API: `TrainingAPIClient` (`business-api/src/api/training_client.py`) makes HTTP calls with `TRAINING_API_KEY`
- Both APIs use Redis for caching and Celery for async task queuing
- MLflow tracks experiments; Prometheus exports metrics at `/metrics`

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `REDIS_URL` | Redis connection | `redis://localhost:6379/0` |
| `TRAINING_API_URL` | Training API address | `http://localhost:8001` |
| `TRAINING_API_KEY` | Internal API key | `default-key` |
| `JWT_SECRET_KEY` | JWT signing key | auto-generated |
| `DEEPSEEK_API_KEY` | LLM for CrewAI agents | required |
| `MLFLOW_TRACKING_URI` | MLflow server | `http://localhost:5000` |

---

## Testing Patterns

- **Unit tests**: `tests/unit/` — isolated module testing with mocks
- **Integration tests**: `tests/integration/` — API route testing with test client
- **conftest.py** at both `tests/` and `business-api/tests/` provides fixtures
- Key fixtures: `test_client` (FastAPI TestClient), `mock_redis`, `mock_training_api`
