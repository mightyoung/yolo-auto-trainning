# Plateau Search Lite Spec

## Goal

Add a bounded Tree-of-Thoughts style decision step only for plateau recovery, so the auto-adjust path can compare a small set of candidate actions before triggering a retry.

## Scope

- Business-side plateau recovery
- Auto-adjust LR decay and data-expansion decisions
- Structured decision trace storage in attempt memory

## Non-goals

- No general-purpose planner for all agent traffic
- No recursive or unbounded branching
- No change to the training API signal contract
- No new LLM dependency in the decision path

## Acceptance Criteria

- Plateau recovery can generate 2-4 bounded candidates from the current signal bundle
- A deterministic scoring step selects one candidate and records the rationale
- The selected action and rejected alternatives are stored in bounded attempt memory
- Existing LR decay and data-expansion behaviors still work when they are selected
- Focused tests cover candidate generation, scoring, and selected-action logging
