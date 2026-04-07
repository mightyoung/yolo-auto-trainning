# Training Memory Bridge Spec

## Goal

Extend the training API task store so it keeps the same bounded attempt-memory shape as the business-side agent registry.

## Scope

- Training API task store normalization
- Training training completion / retry / failure events
- Auto-resubmit metadata on task records

## Non-goals

- No full cross-service memory graph
- No new planner or ToT layer
- No changes to business-side agent orchestration in this phase

## Acceptance Criteria

- Training task records always expose `attempt_memory` and `latest_attempt`
- Terminal training outcomes append typed attempt records
- Auto-resubmit writes a structured retry attempt
- History remains bounded
- Focused tests cover the new task-store behavior
