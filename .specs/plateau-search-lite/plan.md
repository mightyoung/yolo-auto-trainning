# Plateau Search Lite Plan

## Design

1. Add a small helper module under `business-api/src/agents/` that:
   - builds a bounded set of plateau recovery candidates
   - scores each candidate deterministically from the current signal bundle
   - selects a winner and returns a compact rationale
   - emits a typed decision trace for attempt memory

2. Update `AutoAdjustAgent` to:
   - call the candidate builder before acting on LR decay / data expansion
   - log the selected candidate and rejected alternatives
   - preserve current action implementations as execution primitives

3. Keep the planner deterministic and schema-bounded:
   - no free-form chain-of-thought text
   - no recursive expansion
   - no more than four candidates per signal bundle

## Risks

- The new decision layer must not delay normal plateau handling enough to matter
- If the planner overfits the current signals, it could suppress valid recovery paths
- Existing Redis/UI consumers may need compatibility fields preserved

## Validation

- Add unit tests for candidate generation and scoring
- Add integration-style tests for the selected-action trace in auto-adjust
- Run the affected business and training unit suites
