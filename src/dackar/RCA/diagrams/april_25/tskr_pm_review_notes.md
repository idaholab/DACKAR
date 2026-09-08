---
name: TSKR and PM Compliance Review Notes
description: Comprehensive systems engineering review of tskr_temporal_scorer.py and pm_compliance module, including bugs, integration gaps, fixes already applied, and implementation strategy
type: project
---

# TSKR & PM Compliance Review — May 2026

## Context

Branch: `mandd/rca`. This document captures the full state of analysis and work done on two RCA subsystems across two sessions.

---

## PM Compliance Module — Changes Already Applied

### Bug Fix: Ishikawa field name mismatch
**File**: `src/dackar/RCA/orchestrators/ishikawa_evaluator.py` (line ~167)

`_maintenance_rows()` was reading `pm_compliance.get("overdue_tasks")` but the aggregator writes `overdue_items`. Fixed with primary + fallback:

```python
overdue = pm_compliance.get("overdue_items") or pm_compliance.get("overdue_tasks") or []
```

Notes key also changed from `"overdue_tasks"` → `"overdue_items"`.

### Fixture Schema Upgrades (TC-2, TC-3, TC-5)

All three were pre-Wave-2 schema. Added to each:
- `assessment_date`, `look_back_window_days`, `fmea_pm_linkage_available: false`
- `component_id` on all check entries
- `overdue_by_days` on failing checks
- `overdue_items` array (replacing legacy `overdue_tasks`)
- `data_quality_notes` array
- `summary`: `overall_compliance`, `maintenance_induced_risk`, `has_scope_gaps_for_primary_fm`, `data_quality_confidence`

**TC-6** remains the reference fixture with complete schema including `fmea_pm_linkage_available: true` and `components[]` with scope_gaps.

### Unit Tests: `_apply_pm_corrective_actions`
**File**: `src/dackar/RCA/unit_tests/test_pm_corrective_actions.py`

26 tests written and passing. Tests cover:
- Guards (5): no pm, no linkage, no components, empty scope gaps, missing primary FM
- Priority rules (4): high risk → high priority, medium risk → medium priority, no risk → medium fallback, mixed components
- Action structure (7): required keys, action_id format, recommended_action text, source_ref, fmea_pm_linkage flag, target_component, fm_id
- Primary FM filtering (3): only gaps for primary FM included, secondary FMs excluded, non-matching component skipped
- Deduplication (3): same gap appears once, different gaps both appear, different components both appear
- Edge cases: multiple gaps on one component, action_id uniqueness across gaps

Required neo4j mock pattern before import:
```python
for _mod in ("neo4j", "py2neo", "chromadb", "langchain_chroma", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
from orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator
```

---

## TSKR Temporal Scorer — Full Integration Map

**File**: `src/dackar/RCA/orchestrators/tskr_temporal_scorer.py` (~1195 lines)
**Class**: `TSKRTemporalScorerV1`

### Method Signature

```python
def score(
    self,
    event: JsonDict,                          # event_id, asset_id, timestamp_start/end, severity
    telemetry_summary: JsonDict,              # signals[].sensor_id, anomalies[].timestamp_start/end/severity_score
    kg_context: JsonDict,                     # failure_modes[], past_events[], components[], out_of_boundary_anomalies[]
    operational_context: Optional[JsonDict],  # currently unused in score logic
    run_context: JsonDict,                    # run_id for provenance
    signal_evidence: Optional[JsonDict],      # augmented_anomaly_set[], per_candidate_chain_score{}
    alarm_log: Optional[JsonDict],            # alarms[].activated_at/cleared_at + quality.clock_sync_ok
    soe_log: Optional[JsonDict],              # records[].timestamp + quality.clock_sync_ok
) -> JsonDict:
```

### Where It Is Called

**Orchestrator**: `rca_reasoning_orchestrator.py`, lines 462–471 (`orchestrate()`)

Pipeline sequence:
1. Stage 5A — KG context built (line 390)
2. Stage 5B — CMMS context augmented into `past_events` (line 422); temporal metadata enriched (line 447)
3. Stage 5B.2 — Signal evidence built (line 452)
4. **→ Stage 6 — TSKR called** (line 461); result schema-validated and persisted to artifact store
5. Stage 7 — Causality engine receives `tskr_patterns` (line 474)
6. Stage 8 — Ishikawa evaluator receives `tskr_patterns` (line 615)
7. Stage 9 — Synthesizer receives `tskr_patterns` (line 690)
8. Stage 10 — Attention flags applied using `tskr_patterns` (line 708)

The orchestrator uses dynamic signature inspection (lines 898–902) to pass `alarm_log`/`soe_log` only if the scorer's current version accepts them.

---

### Output Schema (`patterns[]` — key fields)

```
pattern_id: "TSKR::{fm_id}"
target_id: str (fm_id)
component_id: str

# Allen relation results
relation: "precedes"|"overlaps"|"contains"|"during"|"follows"|"unknown"
operator_family: "interval_interval"|"interval_only"|"anomaly_only"|null
mean_lag_hours: float|null
std_lag_hours: float|null

# Scores
confidence: float [0,1]          ← primary output; used in causality engine (35% weight)
support: float [0,1]             ← composite; used in causality engine (15% weight)
signal_support_score: float
recurrence_support_score: float
telemetry_support: float

# Latency alignment
latency_alignment_score: float   ← used in causality engine (25% weight)
latency_violation_type: "none"|"too_fast"|"too_slow"|"not_available"
expected_latency_min/max_hours: float|null
observed_lag_hours: float|null

# Telemetry
anomaly_count: int
matching_signal_ids: [str]
lag_consistency: float [0,1]

# Recurrence (from past_events)
recurrence_count: int
effective_recurrence_count: float
recurrence_trend: "increasing"|"decreasing"|"stable"|"insufficient_data"
unresolved_recurrence_count: int
exact_doc_ids_count: int

# Semantic recurrence (Phase A / §4.3)
semantic_match_count: int
near_match_count: int
fm_resolution_ambiguous: bool    ← triggers orchestrator attention flag

# Flags
temporal_contradiction: bool     ← -0.20 penalty to confidence; -0.15 to support
novel_pattern: bool              ← documentary_novel AND signal_novel
near_match_pattern: bool         ← triggers orchestrator attention flag
```

### Confidence Formula

```python
confidence = clamp01(normalized_weighted_sum([
    (max(anomaly_score, telemetry_support), 0.45),
    (latency_score,                         0.30),
    (chain_pos,                             0.10),
    (history_score,                         0.10),
    (anomaly_count_score,                   0.15),
    (lag_consistency_score,                 0.10),
]) - (0.20 if temporal_contradiction else 0.0))

support = clamp01(
    0.35 * history_score
  + 0.35 * telemetry_support
  + 0.15 * anomaly_count_score
  + 0.15 * lag_consistency_score
  - (0.15 if temporal_contradiction else 0.0)
)
```

---

### Downstream Consumers

#### Causality Engine (`causality_engine_v32.py`)

- **Line 244**: `_index_tskr_patterns()` — builds `{target_id → [patterns]}` dict
- **Line 4064**: `_lookup_tskr_pattern()` — returns highest-confidence pattern for a given FM
- **Line 2821**: `_temporal_score_for_fm()` — builds candidate temporal score:
  ```
  temporal = 0.35 * confidence + 0.25 * relation_precedence + 0.25 * latency_alignment + 0.15 * support
  if temporal_contradiction: temporal -= 0.25
  ```
- Fields read: `confidence`, `support`, `relation`, `mean_lag_hours`, `latency_alignment_score`, `temporal_contradiction`, `expected_latency_min/max_hours`, `observed_lag_hours`, `latency_violation_type`, `pattern_id`, `window_start/end`, `operator_family`
- `recurrence_count`, `recurrence_trend` are **NOT** read by causality engine — they flow to synthesizer/RCA card only

#### Ishikawa Evaluator (`ishikawa_evaluator.py`)

- **Line 125** `_measurement_rows()`: builds `pattern_map = {target_id → pattern}` from `tskr_patterns.patterns[]`
- Only reads `relation` from patterns — annotates each sensor's Ishikawa row with `temporal_relation`
- All other pattern fields are ignored here

#### Orchestrator Attention Flags (`rca_reasoning_orchestrator.py`)

Three functions, all called at line 708 post-synthesis:

| Function | Lines | Reads | Triggers When |
|---|---|---|---|
| `_apply_recurrence_match_quality_attention_flags()` | 5301–5320 | `summary.high_cr_match_failure_rate`, `unmatched_cr_count`, `unmatched_cr_rate` | CR-to-FM match rate < 70% |
| `_apply_near_match_pattern_attention_flags()` | 5323–5348 | `patterns[].near_match_pattern` | Any pattern has `near_match_pattern=True` |
| `_apply_fm_resolution_ambiguity_flags()` | 5380–5405 | `patterns[].fm_resolution_ambiguous` | Any pattern has `fm_resolution_ambiguous=True` |

Note: There is **no** attention flag for `recurrence_trend == "increasing"` — this is G3 below.

#### RCA Synthesizer (`rca_synthesizer_v31.py`)

- Line 371: First 10 patterns + summary compacted into LLM context
- Used for temporal support narrative, recurrence arguments, confidence framing
- `recurrence_count`, `recurrence_trend`, `unresolved_recurrence_count` reach the LLM here

---

### `past_events` Schema in `kg_context`

Confirmed consistent across TC-3, TC-5, TC-6:

```json
{
  "event_id": str,
  "asset_id": str,
  "component_id": str,
  "timestamp_start": ISO8601,
  "timestamp_end": ISO8601,
  "severity": str,
  "event_type": str,
  "matched_failure_mode_ids": [str],   ← CRITICAL for recurrence matching
  "time_distance_days": int,            ← extracted as most_recent_days_ago (stale — see B2)
  "resolved": bool (optional),
  "source_doc_id": str (optional)
}
```

> **Note**: The previous session's claim that TC-6 uses `related_failure_modes`/`occurred_at` was **not confirmed** by the deep read. The schema appears consistent. G1 (schema normalization) may be a lower-risk item than originally assessed — verify by running TC-6 end-to-end and checking `recurrence_count` in the output patterns.

---

## Confirmed Bugs

| # | Location | Bug | Severity |
|---|----------|-----|----------|
| B1 | `_build_recurrence_profile` L739-742 | OR-matching: event counted for FM when `component_id` matches even if `matched_failure_mode_ids` excludes that FM — inflates recurrence when multiple FMs share a component | High |
| B2 | `_build_recurrence_profile` | `most_recent_days_ago` set from `time_distance_days` (KG snapshot-relative) rather than computed from `event.timestamp_start` vs past event `timestamp_start` — stale recency bonus | Medium |
| B3 | `_stage_b_allen_relation_by_component` | Last-write-wins when multiple `out_of_boundary_anomalies` match same component — highest-priority Allen relation (per `CAUSAL_PRIORITY` order) should win instead | Medium |
| B4 | `_recurrence_trend` | Half/half interval comparison: requires ≥3 intervals, no smoothing — single outlier interval flips trend | Low |
| B5 | `score()` | `signal_novel` computed globally before per-FM loop — once any FM sees a new signal, all FMs get `signal_novel=False` | Low |
| B6 | `_build_recurrence_profile` | `count` (total events) and `unresolved_count` computed from different filtered event sets | Low |

## Integration Gaps

| # | Gap | Impact |
|---|-----|--------|
| G1 | `past_events` schema normalization — if KG ever emits `related_failure_modes`/`occurred_at`, recurrence scoring is dead. Normalization layer absent. | Medium (verify TC-6 first) |
| G2 | `pm_compliance.overdue_items` not wired into scorer — overdue PM for a component should modulate `history_score` | Medium |
| G3 | No attention flag for `recurrence_trend == "increasing"` — accelerating failure rate is not surfaced to orchestrator/analyst | Medium |
| G4 | Matched CR/WO event IDs not stored in pattern output — no traceability from pattern score to source CRs | Low |
| G5 | `_anomaly_count_score` is FM-agnostic — same bonus regardless of which FM is being scored | Low |
| G6 | End-to-end semantic recurrence path (CR text → NLP → FM match) untested across all test cases | Low |

---

## Implementation Strategy Plan

### Phase 0 — Done ✓
- [x] Fix Ishikawa `overdue_tasks` → `overdue_items` field name mismatch
- [x] Upgrade TC-2, TC-3, TC-5 fixtures to current pm_compliance schema
- [x] Write 26 unit tests for `_apply_pm_corrective_actions` (all passing)
- [x] Full systems-level analysis of TSKR scorer and integration map

### Phase 1 — Done ✓
- [x] **B1** OR-matching → guarded component fallback in `_build_recurrence_profile` (lines 778–784)
- [x] **B3** Last-write-wins → CAUSAL_PRIORITY selection in `_stage_b_allen_relation_by_component` (lines 406–427)
- [x] **G1** `_normalize_past_events()` added — remaps `related_failure_modes`/`occurred_at` to canonical schema transparently (lines 432–458)

### Phase 1 — Done ✓ (correctness)

- [x] **B1** `_build_recurrence_profile`: guarded component fallback — component-only match used only when `matched_failure_mode_ids` is absent
- [x] **B3** `_stage_b_allen_relation_by_component`: CAUSAL_PRIORITY rank selection replaces last-write-wins
- [x] **G1** `_normalize_past_events()`: transparent remapping of `related_failure_modes`/`occurred_at` to canonical fields; enables TC-6 recurrence scoring

### Phase 2 — Done ✓ (integration)

- [x] **B2** `most_recent_days_ago` computed from `event_start − past_event.timestamp_start`; timezone normalization added; fallback to `time_distance_days` when no `event_start`
- [x] **G3** `attention_flags: ["accelerating_recurrence"]` emitted in pattern when `trend == "increasing"`; `_apply_accelerating_recurrence_attention_flags()` added to orchestrator and wired at line 711
- [x] **G2** `pm_compliance` accepted by `score()` and `_score_failure_mode_pattern()`; +0.05 per matching overdue item, capped at +0.15; surfaced as `pm_overdue_boost`; backward-compatible via `inspect.signature` in `_build_tskr_patterns()`

### Phase 3 — Done ✓ (robustness & traceability)

- [x] **B4** `_recurrence_trend`: OLS linear regression on interval sequence replaces fragile half/half ratio; slope normalised by mean interval; threshold ±0.10
- [x] **B5** `_extract_signal_ids_for_fm()` added; filters anomalous sensors by `signal.parameter` vs `fm.expected_symptom_types`; `signal_novel`, `novel_pattern`, and `matching_signal_ids` all use per-FM result; falls back to global list when FM has no symptom types
- [x] **G4** `contributing_event_ids` added to `RecurrenceProfile` and pattern dict; collects `event_id`/`source_doc_id`/`cr_id`/`wo_id` from matched past events; no duplicates
- [x] **B6** `unresolved_count` now computed from `matching` (all matched events), not `dated` (events with parseable timestamps); aligns denominator with `count`

### Phase 4 — Done ✓ (testing)

- [x] `unit_tests/test_tskr_phase_fixes.py` — 51 new tests, all passing
- [x] Full suite: **1622 passed** (up from 1571 baseline)

| Section | # tests | Key assertions |
|---------|---------|---------------|
| B1 | 5 | FM-specific count, component fallback only when no FM IDs, two FMs on same component stay independent |
| B3 | 5 | Higher CAUSAL_PRIORITY wins regardless of order; independent components tracked separately |
| G1 | 4 | New-schema fields remapped; canonical fields not clobbered; recurrence count > 0 for TC-6 style |
| B2 | 4 | Live delta computed; stale `time_distance_days` ignored; timezone-naive handled |
| G2 | 6 | No boost without compliance; cap at 0.15; legacy `overdue_tasks` key accepted |
| G3 | 4 | Flag absent for stable trend; set for shrinking intervals; escalated to RCA card |
| B4 | 7 | Insufficient-data guards; increasing/decreasing/stable; outlier robustness |
| B5 | 7 | Symptom-type mismatch → empty; match works; multi-type; global fallback; end-to-end in pattern |
| G4 | 5 | IDs captured; all events covered; surfaced in pattern; no duplicates |
| B6 | 4 | No-timestamp events counted; resolved=None not counted; count and unresolved share denominator |

---

## Key File Locations

| File | Role |
|------|------|
| `src/dackar/RCA/orchestrators/tskr_temporal_scorer.py` | TSKR scorer (~1195 lines) — all bug fixes go here |
| `src/dackar/RCA/orchestrators/ishikawa_evaluator.py` | Ishikawa matrix builder — field name fix already applied |
| `src/dackar/RCA/orchestrators/rca_reasoning_orchestrator.py` | Main orchestrator — Stage 6 (TSKR call), Stage 10 (attention flags) |
| `src/dackar/RCA/orchestrators/causality_engine_v32.py` | Causality engine — `_index_tskr_patterns`, `_lookup_tskr_pattern`, temporal scoring |
| `src/dackar/RCA/synthesis/rca_synthesizer_v31.py` | Synthesizer — receives tskr_patterns for LLM context |
| `src/dackar/RCA/unit_tests/test_pm_corrective_actions.py` | 26 unit tests for Wave 4 method |
| `src/dackar/RCA/tests/test_case_6/fixtures/kg_context.json` | Reference TC-6 kg_context — verify past_events schema |

---

## Test Suite

| Milestone | Count |
|-----------|-------|
| Baseline (start of sessions) | 1571 |
| After Phase 4 (current) | **1622** |
| New tests added | 51 (in `test_tskr_phase_fixes.py`) |
