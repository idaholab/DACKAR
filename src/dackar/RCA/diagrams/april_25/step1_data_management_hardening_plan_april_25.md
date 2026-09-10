# Step 1 — Data Management Hardening Plan

**Date:** 2026-04-25  
**Step:** Step 1 — Data Management (from `rca_metamodel.md`)  
**Depends on:** Step 0 (completed)  
**Source of truth:** `rca_metamodel.md`, JSON schemas in `src/dackar/RCA/schemas/`

---

## Goal

Make Step 1 fully Green by ensuring:
- every input data element is quality-assessed at intake
- quality signals from individual artifacts drive the coverage report
- paired-data coupling is enforced (SOE + protection logic)
- per-artifact quality propagates into confidence multipliers and uncertainty summary
- strict-mode validation blocks silent use of degraded critical inputs

---

## Current state audit

### Already implemented ✅
| Item | Location |
|---|---|
| `_build_data_coverage_summary` emitting `complete\|partial\|missing` for three source families | `rca_reasoning_orchestrator.py` |
| Degraded-mode analyst acknowledgement gate | `review_hooks`, `schema_validator.py` |
| Coverage quality factor applied to `quality_multiplier` | `causality_engine_v32.py` |
| Strict full-mode validation for coverage uncertainty/review-hook signals | `schema_validator.py` |
| `has_soe_log`, `has_alarm_log`, `has_protection_logic_context`, `has_configuration_change_records` flags in `run_context.input_refs` and `scope_snapshot.data_availability` | `rca_reasoning_orchestrator.py`, `run_context.json` |

### Gaps ❌
| Gap | Impact |
|---|---|
| `soe_log.quality`, `alarm_log.quality`, `telemetry_summary.signals[].data_quality` fields are **not consumed** in coverage report | Quality-degraded inputs treated as fully healthy |
| `upstream_anomaly_inputs` source family only counts TSKR patterns — ignores soe/alarm artifact quality | Misleading `complete` status when underlying logs are degraded |
| **No source families** for `soe_log`, `alarm_log`, `protection_logic_context`, `configuration_change_records`, `telemetry_detail` in coverage report | Coverage gaps invisible to scoring and review routing |
| **Paired-data coupling not enforced**: `soe_log` available but `protection_logic_context` absent triggers no warning | Barrier logic gate runs in degraded mode silently |
| No Step 1 DoD or readiness matrix row in backlog | Tracking gap |

---

## In Scope

- Expand `_build_data_coverage_summary` to assess per-artifact quality for all Step 1 inputs
- Add new source families to coverage report: `soe_log`, `alarm_log`, `protection_logic_context`, `configuration_change_records`, `telemetry_detail`
- Implement paired-data coupling check: `soe_log` ↔ `protection_logic_context`
- Propagate per-artifact quality degradation into `coverage_factor` (and thereby into `quality_multiplier`)
- Add strict full-mode semantic validation for paired-data requirement
- Write Step 1 DoD + readiness matrix row
- Tests (schema, semantic, behavior)

## Out of Scope

- Fetching or parsing the actual SOE/alarm/protection-logic data — these remain optional inputs; coverage report only assesses what is provided
- Redesigning the scoring pipeline weights
- Step 2+ changes

---

## Workstreams

### WS1 — Expand `_build_data_coverage_summary`

**File:** `orchestrators/rca_reasoning_orchestrator.py`

Current signature only takes `kg_context`, `tskr_patterns`, `evidence_bundle`, `causality_candidates`.

**Changes:**
- Add optional params: `run_context`, `soe_log`, `alarm_log`, `protection_logic_context`, `configuration_change_records`, `telemetry_summary`
- Add new source family assessors:

| New source family | Quality signals consumed | Status logic |
|---|---|---|
| `telemetry_detail` | `signals[].data_quality.missing_fraction`, `flatline_detected`, `outlier_fraction` | `complete` if no flags; `partial` if any degraded; `missing` if no signals |
| `soe_log` | `quality.clock_sync_ok`, `dropped_record_count`, `duplicate_record_count` | `complete` if clock_sync_ok + zero drops; `partial` if minor issues; `missing` if not provided |
| `alarm_log` | `quality.missing_fraction`, `clock_sync_ok` | `complete` / `partial` / `missing` |
| `protection_logic_context` | presence only at Step 1 (no internal quality fields) | `complete` if present; `missing` if absent but `soe_log` is present (paired requirement) |
| `configuration_change_records` | `quality.coverage_status` | maps directly to `complete\|partial\|missing`; `missing` if not provided |

- Update `overall_status` aggregation to include all new families
- Pass `run_context` to read `input_refs.has_*` flags for presence checks

### WS2 — Paired-data coupling enforcement

**File:** `orchestrators/rca_reasoning_orchestrator.py`

- After building the coverage report, add a `paired_data_checks` block:
  - `soe_protection_logic_pairing`: `ok` if `has_soe_log == has_protection_logic_context`; `warning` if `has_soe_log=True` and `has_protection_logic_context=False`; `n/a` if both absent
- Surface paired-data warnings into `review_hooks.degraded_reasons`

### WS3 — Per-artifact quality propagation into coverage factor

**File:** `orchestrators/causality_engine_v32.py` (`_coverage_quality_profile`)

- Extend `_coverage_quality_profile` to consider new source families
- Weighted average should still prioritize `kg_context` (structural) and `upstream_anomaly_inputs` (signal) over optional artifact quality
- Proposed weights:
  - `kg_context`: 0.40
  - `upstream_anomaly_inputs` / `telemetry_detail`: 0.30 (combined)
  - `chroma_corpus`: 0.15
  - `soe_log`, `alarm_log`, `protection_logic_context`, `configuration_change_records`: 0.15 (combined, only when present)

### WS4 — Strict semantic validation

**File:** `validation/schema_validator.py`

Add checks to `_semantic_checks_run_manifest`:
1. If `source_families.soe_log.status != missing` and `source_families.protection_logic_context.status == missing` → full-mode **error** (paired requirement violated)
2. If `source_families.telemetry_detail.status == missing` → full-mode **error** (telemetry is mandatory)
3. If any new source family missing and `overall_status` still reported as `complete` → full-mode **error** (overall status inconsistent)

Add checks to `_semantic_checks_run_context`:
1. If `input_refs.has_soe_log=True` and `input_refs.has_protection_logic_context=False` → full-mode **warning** at scoping time

### WS5 — Tests

**Files:** `unit_tests/test_phase4_strict_mode.py`, `unit_tests/test_manifest_quality.py`, new `unit_tests/test_step1_data_coverage.py`

Test cases:
- Coverage report includes all new source families when artifacts provided
- `telemetry_detail` status degrades when signals have high `missing_fraction`
- `soe_log` status correctly assessed from quality fields
- Paired-data check raises warning when SOE present but protection logic absent
- Full-mode validation fails when paired requirement violated
- Full-mode validation fails when `overall_status=complete` but required family missing
- Quality multiplier is lower when new source families degrade
- Full suite stays green

---

## Implementation Sequence

1. WS1: expand `_build_data_coverage_summary` with new source families (non-breaking defaults)
2. WS2: add `paired_data_checks` block and review-hook surfacing
3. WS3: extend `_coverage_quality_profile` for new families with weights
4. WS4: add semantic validation rules
5. WS5: write tests
6. Run targeted tests → full `src/dackar/RCA/unit_tests` → update backlog + metamodel

---

## Acceptance Criteria

- Coverage report contains source families for: `kg_context`, `chroma_corpus`, `upstream_anomaly_inputs`, `telemetry_detail`, `soe_log`, `alarm_log`, `protection_logic_context`, `configuration_change_records`
- Per-artifact quality fields (not just presence) drive `partial` vs `complete` for telemetry, SOE, and alarm families
- Paired-data check surfaces in `review_hooks.degraded_reasons` when SOE present but protection logic absent
- Full-mode validation fails when paired requirement is violated
- Coverage factor updated to include new families
- Targeted and full test suites pass

---

## Step 1 Definition of Done (to be frozen in backlog after plan approval)

### Required inputs
| Input | Schema | Mandatory? | Quality fields consumed |
|---|---|---|---|
| `event` | `event.json` | Mandatory | — (structural, no quality fields) |
| `telemetry_summary` | `telemetry_summary.json` | Mandatory | `signals[].data_quality.missing_fraction`, `flatline_detected`, `outlier_fraction` |
| `operational_context` | `operational_context.json` | P0 | — |
| `pm_compliance` | `pm_compliance.json` | P1 / auto-built | `summary.data_quality_confidence`, `data_quality_notes` |
| `soe_log` | `soe_log.json` | Conditional P0 | `quality.clock_sync_ok`, `dropped_record_count` |
| `protection_logic_context` | `protection_logic_context.json` | Conditional P0 paired with SOE | presence |
| `alarm_log` | `alarm_log.json` | Conditional P0 | `quality.missing_fraction`, `clock_sync_ok` |
| `configuration_change_records` | `configuration_change_records.json` | P1 | `quality.coverage_status` |
| `cmms_context` | `cmms_context.json` | P1 | — |

### Required outputs/artifacts
- `run_manifest.coverage_summary` with all source families present and quality-driven
- `run_manifest.coverage_summary.paired_data_checks` with `soe_protection_logic_pairing` status
- `run_manifest.review_hooks.degraded_reasons` includes paired-data warnings
- `run_manifest.uncertainty_summary` reflects quality degradation from new source families
- Candidate `quality_multiplier` reflects coverage from all source families

### Required decision checkpoints
- Analyst acknowledgement required when coverage is degraded (already implemented)
- Paired-data warning surfaces in review routing

### Required tests
- Three layers: schema, semantic strict-mode, behavior (scenario pipeline outcomes)
- Full `src/dackar/RCA/unit_tests` suite remains green

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| New source families always `missing` (not provided) → overall always `partial` | Use `has_*` flags to conditionally include families; when not provided, treat as `not_assessed` rather than `missing` in overall rollup |
| Weight changes to coverage factor cause regression in existing scoring tests | Use backwards-compatible defaults; only apply new weights when new families are present |
| Paired-data check too strict for common setups without protection logic | Keep it as `warning` in compat/partial mode; only `error` in full mode |
