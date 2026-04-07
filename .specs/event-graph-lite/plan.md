# Event Graph Lite Plan

## Design

1. Add a small helper module under `business-api/src/agents/` for:
   - building typed event nodes and edges
   - capping graph size and edge fan-out
   - projecting attempt records into event graph updates

2. Extend the business task registry schema with a bounded `event_graph` field so task-level relationships can be persisted without changing the public API surface.

3. Extend the training task store with the same additive field so retry and terminal events can be represented consistently on the training side.

4. Update plateau, retry, and export paths to record graph edges for:
   - task submitted
   - training started
   - plateau decision selected
   - auto-adjust retried
   - export completed/failed
   - task adjusted_from / adjusted_to

## Risks

- The graph must remain additive and not break existing task consumers
- Graph edges should not duplicate attempt memory noise
- Size bounds must keep Redis payloads small

## Validation

- Add focused unit tests for graph construction and bounded updates
- Run the affected business and training unit suites
