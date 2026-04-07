# Agent Memory Lite Spec

## Goal

Apply a narrow, production-oriented subset of the paper findings:

- Reflexion-lite: store typed retry/attempt memory for agent-driven recovery paths
- AgentSys-lite: keep worker outputs summarized and schema-bounded before they enter top-level orchestration state

## Scope

- Business agent orchestration only
- Auto-adjust retry and training polling paths
- Internal task registry support for attempt-memory slots

## Non-goals

- No full graph memory
- No generic Tree-of-Thought planner
- No free-form long reflection prompts
- No training-side prompt or policy learning

## Acceptance Criteria

- Raw worker/training poll payloads are no longer stored verbatim in orchestration Redis state
- Agent retry/failure/success events are captured as typed attempt records
- Attempt-memory helpers cap history and avoid unbounded prompt/state growth
- New behavior is covered by focused unit tests
