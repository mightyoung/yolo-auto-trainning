# Event Graph Lite Spec

## Goal

Promote the bounded attempt memory into a lightweight task event graph so the project can track causal chains across retries, plateau recovery, exports, and follow-up tasks without introducing a full graph-memory subsystem.

## Scope

- Business-side task registry
- Training-side task store
- Plateau / retry / export / adjusted-task relationships
- Structured graph summaries exposed through task records

## Non-goals

- No general graph database
- No unbounded cross-task traversal
- No free-form narrative memory
- No changes to external API response contracts beyond compatible additive fields

## Acceptance Criteria

- Task records can expose a bounded `event_graph` summary alongside `attempt_memory`
- Retry / plateau / export / adjusted-from relationships are recorded as typed graph edges
- Graph summaries remain bounded and schema-safe
- Focused tests cover edge construction, boundedness, and migration compatibility
