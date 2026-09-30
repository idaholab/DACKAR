# Phase 2 Implementation Plan (Reasoning + Coverage Enforcement)

**Date:** April 25, 2026  
**Depends on:** `wave1_implementation_checklist_april_25.md` (completed)  
**Source of truth:** `rca_metamodel.md` + `rca_metamodel_decision_log_april_25.md`

---

## Phase 2 Goal

Implement core metamodel reasoning logic while remaining backward-compatible:

- move from metadata scaffolding to real category/chain reasoning
- enforce applicability-first coverage policy (`applicable|not_applicable|unknown`)
- produce candidate-level causal category and chain position with explicit confidence
- generate deterministic coverage and rule-out records across categories `A-L`
- apply uncertainty propagation and expose sensitivity diagnostics

Phase 2 does **not** make fields required yet (strict mode is Phase 4).

---

## Locked Decisions Implemented in Phase 2

1. Applicability-first policy before coverage enforcement.
2. Hybrid category assignment (deterministic first, LLM fallback only when needed).
3. Deterministic chain-position assignment with confidence.
4. Category-specific evidence minima gate for posture ceilings.
5. Missing stream data degrades confidence (not contradiction).
6. Mandatory attempt at category `L` (candidate or explicit ruled-out/not-applicable rationale).
7. OE availability-aware posture behavior (`external_oe_unavailable` semantics).

---

## In Scope

- `orchestrators/causality_engine_v32.py` reasoning upgrades
- `orchestrators/rca_reasoning_orchestrator.py` propagation and manifest surface
- `synthesis/rca_synthesizer_v31.py` analyst-facing surfacing of new Phase 2 outputs
- `validation/schema_validator.py` semantic checks for Phase 2 logic consistency
- targeted schema updates if new optional fields are needed
- unit tests for new logic contracts

---

## Out of Scope (Deferred)

- hard contradiction and near-tie gating as hard decision blockers (Phase 3)
- hard-gate override lifecycle and reinstatement governance (Phase 3)
- required-field strict schema cutover (`metamodel_compliance_level=full`) (Phase 4)

---

## A) Engine Reasoning Work (`causality_engine_v32.py`)

### A1. Deterministic category classifier

Add a dedicated classifier function:

- input: candidate + KG context + operational context + PM/CMMS + signal evidence
- output:
  - `primary_causal_category`
  - `category_assignment_method`
  - `category_assignment_confidence`
  - `category_assignment_rationale` (short machine-readable string)
  - optional `category_alternatives` (top-2)

Deterministic precedence examples:

- FM internal mechanisms -> `A`
- dependency/support degradation -> `B`
- directional upstream/downstream signals -> `C` / `D`
- operating envelope violations -> `E`
- environmental disturbances -> `F`
- procedure execution vs baseline/config logic -> `G` / `I`
- design/spec adequacy issues -> `H`
- surveillance/test adequacy indicators -> `J`
- vendor/traceability defects -> `K`
- program/systemic recurrence signals -> `L`

### A2. Applicability assessment engine (`A-L`)

Implement category-level applicability pass:

- output per category: status + rationale + evidence_refs
- status in `{applicable, not_applicable, unknown}`
- unknown is conservative, never silently skipped

### A3. Coverage enforcement engine

After candidate generation and category assignment:

- for each category in `A-L`:
  - if `applicable|unknown`: require candidate count > 0 or explicit rule-out
  - if `not_applicable`: require rationale
- populate top-level `category_coverage` deterministically from live candidate pool

### A4. Rule-out generator

Generate structured `ruleout` entries with controlled reason taxonomy:

- `physically_impossible`
- `timeline_inconsistent`
- `barrier_held`
- `no_supporting_data`
- `category_not_applicable`
- `outside_investigation_scope`
- `superseded_by_higher_fidelity_evidence`
- `analyst_excluded`

### A5. Chain-position reasoning

Refine chain position logic:

- use temporal relation + causal role + dependency direction
- emit:
  - `chain_position`
  - `chain_position_confidence`
  - `chain_position_rationale`

### A6. Category-specific minimum evidence gate

Add minima checker per category:

- if minima missing, cap candidate evidence posture at `insufficient`/`weak`
- preserve current scoring pipeline, but annotate capped posture reason

### A7. Uncertainty propagation

Compute stream quality and apply confidence multiplier:

- stream qualities: temporal/logical/documentary/OE
- `Q = weighted_mean(q_streams)`
- `score_final = score_raw * Q`
- emit:
  - `quality_multiplier`
  - stream quality breakdown
  - `data_limited_conclusion` trigger when critical stream below floor

### A8. OE availability-aware posture

When fleet/industry OE unavailable:

- mark OE stream as `insufficient`, not contradicted
- set candidate/run-level metadata to support `external_oe_unavailable` surfacing downstream

---

## B) Orchestrator + Manifest Wiring (`rca_reasoning_orchestrator.py`)

### B1. Persist new reasoning outputs

Ensure persisted artifacts include:

- applicability map with rationales
- category coverage map derived from actual candidates and rule-outs
- category assignment and chain reasoning fields per candidate
- uncertainty propagation diagnostics

### B2. Run manifest Phase 2 sections

Add/extend manifest blocks:

- `coverage_summary` (resolved from live outputs)
- `applicability_summary`
- `uncertainty_summary` (stream quality, multiplier stats, data-limited flags)
- `metamodel_migration.phase = "wave2"`

### B3. RCA card attention signal plumbing

Inject attention flags for:

- unresolved coverage categories
- categories with unknown applicability in high-impact classes (`B`, `F`, `I`, `L`)
- data-limited conclusions
- external OE unavailable

---

## C) Synthesizer Updates (`rca_synthesizer_v31.py`)

### C1. Executive summary augmentation

Consume Phase 2 metadata and surface:

- concise category coverage posture
- unknown-applicability highlights
- uncertainty/degraded-confidence rationale

### C2. Primary hypothesis rationale enrichment

Augment `why_primary` with Phase 2 details:

- category assignment rationale
- chain-position rationale
- stream quality caveats when confidence is degraded

Do not change fallback contract shape in Phase 2.

---

## D) Validator Enhancements (`validation/schema_validator.py`)

### D1. Consistency checks

Add semantic checks:

- category coverage aligns with candidate/category counts
- applicability statuses valid and coherent with coverage status
- `not_applicable` categories include rationale
- `unknown` categories in high-impact classes generate warnings if unresolved
- `L` category always represented in coverage records

### D2. Confidence-quality checks

Validate uncertainty block:

- stream qualities in `[0,1]`
- multiplier in `[0,1]`
- score consistency sanity check when provided

All Phase 2 checks should remain compat-safe (warnings where strict enforcement is deferred).

---

## E) Schema Deltas (Optional/Backward-Compatible)

If needed, add optional fields to schema(s):

- candidate-level:
  - `category_assignment_rationale`
  - `category_alternatives`
  - `chain_position_rationale`
  - `quality_multiplier`
  - `stream_quality`
  - `data_limited_conclusion`
- top-level:
  - `applicability_summary`
  - `uncertainty_summary`

No new required fields in Phase 2.

---

## F) Unit Test Plan

Add tests (new files suggested):

1. `test_phase2_category_assignment.py`
   - deterministic mapping coverage across representative category signals
2. `test_phase2_applicability_coverage.py`
   - applicability-first logic and coverage enforcement behavior
3. `test_phase2_chain_position.py`
   - chain position assignment and confidence behavior by temporal relation
4. `test_phase2_uncertainty_propagation.py`
   - score degradation and data-limited flags
5. `test_phase2_category_minima.py`
   - posture capping when category minima evidence is missing
6. `test_phase2_manifest_summary.py`
   - manifest coverage/applicability/uncertainty summaries emitted correctly

Regression tests to run/update:

- `test_manifest_quality.py`
- `test_causality_scoring.py`
- `test_pipeline_alignment_plan.py`
- `test_synthesizer_validation.py`

---

## G) Acceptance Criteria (Phase 2 Done)

Phase 2 is complete when:

1. Candidates carry deterministic category and chain-position assignments with confidence fields.
2. `applicability_assessment` and `category_coverage` are computed from reasoning outputs (not scaffolds).
3. Coverage policy is enforced for `applicable|unknown` categories with rule-out support.
4. Category `L` is always represented (candidate or explicit rationale path).
5. Uncertainty propagation is applied and surfaced in artifacts.
6. OE unavailability is correctly represented as insufficient data, not contradiction.
7. Unit tests for Phase 2 contracts pass, and legacy behavior remains compatible.

---

## H) Recommended Delivery Sequence

1. Engine category/applicability/coverage logic (`A1-A4`)
2. Chain position and minima gate (`A5-A6`)
3. Uncertainty propagation + OE availability behavior (`A7-A8`)
4. Orchestrator/manifest propagation (`B1-B2`)
5. Synthesizer surfacing (`C1-C2`)
6. Validator semantic checks (`D1-D2`)
7. Tests and stabilization (`F`)

---

## I) Risks and Mitigations

- **Risk:** category assignment ambiguity for sparse evidence  
  **Mitigation:** deterministic-first with confidence + explicit unknown status

- **Risk:** false confidence from partial data  
  **Mitigation:** mandatory uncertainty multiplier + data-limited flags

- **Risk:** compatibility regressions in existing pipelines  
  **Mitigation:** optional fields only; maintain legacy IDs and core outputs

- **Risk:** overconstrained coverage in thin datasets  
  **Mitigation:** applicability-first gating and structured rule-out paths

