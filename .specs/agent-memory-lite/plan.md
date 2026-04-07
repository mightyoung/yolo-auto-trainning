# Agent Memory Lite Plan

## Design

1. Add a small helper module under `business-api/src/agents/` for:
   - sanitizing training status snapshots
   - building typed attempt records
   - appending bounded attempt history into Redis hash state
   - extracting normalized training submission params from mixed legacy/new agent state

2. Extend `task_registry` with internal attempt-memory normalization so business tasks can reuse the same schema later.

3. Wire orchestration polling to:
   - store `training_summary` instead of raw poll payloads
   - append attempt records on success/failure/timeout

4. Wire `AutoAdjustAgent` to:
   - use normalized submission extraction instead of legacy `params` only
   - append typed attempt records on LR-adjust and data-expansion outcomes
   - store sanitized worker status snapshots

## Risks

- Existing agent UI may still read legacy fields, so compatibility fields must remain
- Redis hash history must be bounded

## Validation

- Add focused unit tests for helper behavior
- Run current business/training unit suites plus agent tests
