# Wave 1 Implementation Checklist (Metamodel Migration)

**Date:** April 25, 2026  
**Scope:** Wave 1 only (schema and metadata, non-breaking)  
**Source of truth:** `rca_metamodel.md` + `rca_metamodel_decision_log_april_25.md`

---

## Wave 1 Goal

Introduce metamodel metadata fields and audit structures without breaking existing runs:

- Add optional fields to artifacts and schemas.
- Persist new metadata in run artifacts and manifest.
- Keep legacy `candidate_id` behavior intact.
- Add non-breaking validations and unit tests.

No strict enforcement in Wave 1.

---

## A) Schemas: Add Optional Metamodel Fields

### 1) `schemas/causality_candidates.v3_2.schema.json`

Add optional candidate-level properties:

- `canonical_candidate_key` (string)
- `primary_causal_category` (enum `A` through `L`)
- `chain_position` (enum `initiating|contributing|consequence`)
- `chain_position_confidence` (number `0..1`, nullable)
- `event_scope_id` (string or null)
- `category_assignment_method` (enum `deterministic|llm_fallback|analyst_override`)
- `category_assignment_confidence` (number `0..1`, nullable)
- `category_applicability` (enum `applicable|not_applicable|unknown`)
- `ruleout` (object, optional)
  - `reason_code` (enum from locked taxonomy)
  - `reason_detail` (string, optional)
  - `ruled_out_by` (enum `engine|analyst`)
  - `ruled_out_at` (date-time, optional)

Add optional top-level blocks:

- `category_coverage` (object keyed by `A..L`; status + rationale metadata)
- `applicability_assessment` (object keyed by `A..L`; status + rationale metadata)
- `metamodel_compliance` (object)
  - `level` (`partial|full`)
  - `version` (string)

Keep all new fields optional in Wave 1.

### 2) `schemas/causality_candidates.json`

Mirror the same optional fields (or add compatibility references if this file remains canonical for validation paths).

### 3) `schemas/run_manifest.json`

Add optional metadata under `pipeline_config`:

- `metamodel_compliance_level` (`partial|full`)
- `metamodel_decision_log_version` (string)
- `near_tie_delta` (number)
- `critical_stream_floor` (number)
- `oe_reinstatement_threshold` (number)

Add optional audit blocks:

- `metamodel_migration` (phase, compatibility flags)
- `coverage_summary` (`A..L` category status summary)

### 4) `schemas/rca_card.json`

Add optional executive summary extensions:

- `category_coverage_flags` (array of strings)
- `external_oe_unavailable` (boolean)

No new required fields in Wave 1.

---

## B) Engine Metadata Emission (Non-Breaking)

### 5) `orchestrators/causality_engine_v32.py`

Populate optional fields during candidate generation and refinement:

- retain legacy `candidate_id`
- emit `canonical_candidate_key` (initial compatibility construction)
- emit placeholder/initial:
  - `primary_causal_category` when derivable
  - `chain_position` when derivable
  - assignment method/confidence fields

Add top-level compatibility metadata in output:

- `metamodel_compliance.level = "partial"`
- scaffolded `category_coverage` and `applicability_assessment` structures

Important Wave 1 behavior:

- Do not reject candidates missing new fields.
- Do not change existing ranking thresholds/retention behavior.

---

## C) Orchestration and Manifest Wiring

### 6) `orchestrators/rca_reasoning_orchestrator.py`

Ensure new metadata propagates to persisted artifacts:

- carry through `category_coverage`, `applicability_assessment`, `metamodel_compliance`
- write metamodel config defaults into `run_manifest.pipeline_config`
- write migration status and defaults into `run_manifest`

Keep legacy outputs and artifact names unchanged.

### 7) `orchestrators/artifact_store.py` (if needed)

Confirm no filtering strips newly added optional keys.

---

## D) Synthesis and RCA Card Surface

### 8) `synthesis/rca_synthesizer_v31.py`

Surface new metadata into analyst-facing flags (non-blocking):

- append attention flags for missing category coverage rationale when present
- append `external_oe_unavailable` context when set upstream

Do not alter fallback behavior semantics in Wave 1.

---

## E) Validation Layer (Warn-Compatible)

### 9) `validation/schema_validator.py`

Add semantic checks for new fields in compatibility mode:

- validate enum/value ranges when fields exist
- emit warnings (not hard failures) when metamodel metadata is incomplete in `partial` mode
- preserve current strictness for pre-existing required fields

Bundle checks (optional):

- if `category_coverage` present, ensure keys are valid categories `A..L`
- if `ruleout.reason_code` present, ensure controlled taxonomy value

---

## F) Unit Tests to Add/Update

Add focused tests under `unit_tests`:

### 10) New tests

- `test_metamodel_schema_compatibility.py`
  - old payloads still validate
  - new optional fields validate when present

- `test_metamodel_manifest_metadata.py`
  - `run_manifest` includes metamodel config fields when emitted

- `test_category_coverage_scaffold.py`
  - engine output includes category scaffold with valid keys when enabled

- `test_ruleout_reason_taxonomy.py`
  - accepted reason codes pass; unknown codes fail semantic check

### 11) Existing tests likely to touch

- `test_manifest_quality.py`
- `test_synthesizer_validation.py`
- `test_pipeline_alignment_plan.py`
- `test_review_hooks.py`

Update only as needed to account for additive optional fields.

---

## G) Wave 1 Acceptance Criteria

Wave 1 is done when all are true:

1. Existing test cases run without breaking legacy payload consumers.
2. New optional fields are emitted in at least one representative run artifact.
3. `run_manifest` records `metamodel_compliance_level = partial` and locked default thresholds.
4. Validator accepts legacy payloads and validates new fields when present.
5. No strict enforcement of A-L coverage or chain/category requirements yet.

---

## H) Out of Scope for Wave 1 (Deferred)

Do not implement yet:

- mandatory applicability-first coverage enforcement
- hard contradiction/near-tie gating changes
- strict uncertainty propagation formula wiring
- hard-gate override lifecycle controls
- required-field schema cutover (`full` mode)

These belong to Waves 2-4.

---

## I) Recommended Execution Order (Within Wave 1)

1. Update schemas.
2. Update engine metadata emission.
3. Wire orchestrator and manifest propagation.
4. Update validator semantics.
5. Update synthesizer surface flags.
6. Add tests, then run unit suite.

