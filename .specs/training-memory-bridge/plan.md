# Training Memory Bridge Plan

## Design

1. Add bounded attempt-memory helpers to `training-api/src/api/store/task_store.py`.
2. Normalize every task record on read/write so `attempt_memory` and `latest_attempt` always exist.
3. Append typed attempt records from `_run_training_sync()` for completion, failure, cancellation, and retry.
4. Append a retry attempt when `get_training_status()` auto-resubmits stuck or failed jobs.

## Risks

- Response models should continue to ignore these internal fields
- Task-store changes must not break existing JSON persistence

## Validation

- Add focused tests for task-store normalization and attempt append behavior
- Run the affected training and business unit suites
