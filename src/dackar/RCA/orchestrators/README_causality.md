
# Causality Engine

Implements a first deterministic `RuleBasedCausalityEngine` for the RCA workflow.

## Responsibilities
- consume event + telemetry_summary + kg_context
- generate failure-mode and past-event causal hypotheses
- score candidates across structural / temporal / evidence / governance dimensions
- emit `causality_candidates.json`

## Files
- `causality_engine.py`
