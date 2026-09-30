# RCA Workflow Development Backlog (From `rca_metamodel.md`)

Date: 2026-04-25  
Source: `src/dackar/RCA/diagrams/april_25/rca_metamodel.md`

---

## Purpose

Translate the metamodel requirements into an actionable implementation backlog for the new RCA workflow, with:

- priority (`P0`, `P1`, `P2`)
- workflow step mapping (`0` to `6`)
- concrete acceptance criteria

---

## Priority Legend

- `P0` = required for minimum metamodel-compliant workflow behavior
- `P1` = strong functional completeness / analyst-trust improvements
- `P2` = advanced depth and long-tail capability

---

## Systematic Workflow Hardening Plan (Control Framework)

Use this framework to keep workflow Steps `0-6` in a consistently development-ready state, with data processing correctness and logic integrity verified end-to-end.

### 1) Freeze a Definition of Done per Step (`0-6`)

For each step, define:

- required inputs (schema + semantic expectations)
- required outputs/artifacts
- required decision checkpoints
- required tests (schema + logic + regression)

Source of truth remains `rca_metamodel.md`; operational checklist remains this file.

### 2) Build and maintain a Step Readiness Matrix (single control panel)

Track these dimensions per step:

- data coverage (all data elements mapped to schemas)
- processing coverage (data consumed by logic, not only present)
- logic validity (behavior matches metamodel rules)
- auditability (decision trail, rationale, timestamps, evidence refs)
- determinism (same inputs -> same replay-critical outputs)
- test status (unit/semantic/full-suite)

Status convention per row: `Green | Yellow | Red`.

### 3) Enforce contracts first, then logic depth

Execution order:

- contracts: every required data element has schema and validation
- semantic validation: block "present but wrong meaning" in full mode
- logic depth: ensure data is used in scoring, gates, and conclusion as metamodel expects
- auditability: elimination/override/reinstatement must be explainable and traceable

### 4) Execute hardening by step order (not by random feature)

Recommended sequence:

1. Step 0 (Scoping): versioned scope revision lifecycle (trigger, boundary delta, analyst decision, timestamp, downstream consumption).
2. Step 1 (Data management): tighten paired-data checks and quality-propagation semantics.
3. Steps 2/3/3.5 (Expansion + pattern recognition): formalize novel-pattern/scope-expansion hooks and analyst checkpoint persistence.
4. Steps 4/5/6: maintain current strength and close residual depth gaps (especially human/organizational contributors).

### 5) Add three test layers for every change

- schema tests: structure correctness
- semantic strict-mode tests: rule correctness
- behavior tests: scenario-based pipeline outcomes

Always run:

- targeted suites for touched logic
- full `src/dackar/RCA/unit_tests` as green baseline gate

### 6) Keep traceability artifacts current after each completed item

- update metamodel status section in `rca_metamodel.md`
- update backlog checklist in `rca_workflow_development_backlog_april_25.md`
- keep "what remains" explicit, short, and prioritized

---

## Wave 5 Implementation Checklist (Executable Subset)

Use this as the next implementation wave focused on highest-impact `P0` completion.

### Addressed So Far (Wave 5)

The following Wave 5 checklist elements have already been implemented and verified:

- Rule-out taxonomy + ruled-out standby registry fields are enforced.
- Canonical candidate 4-tuple is enforced in generation output and strict validation.
- Full-mode category coverage A-L now enforces: applicable categories must be `candidate_scored` or `ruled_out`.
- Physical plausibility hard gate now executes pre-ranking and emits binary pass/fail rationale per candidate.
- Timeline consistency hard gate now executes in normal/degraded mode and records binary pass/fail rationale per candidate.
- Barrier logic hard gate now executes in normal/degraded mode and records binary pass/fail rationale per candidate.
- Step-1 source-family coverage report now emits `complete | partial | missing` across KG, Chroma corpus, and anomaly inputs.
- Progression gate now requires analyst acknowledgement when coverage is degraded (`partial`/`missing`).
- Coverage quality flags now propagate into candidate `quality_multiplier` and run-level `uncertainty_summary`.
- Full-mode semantic validation now blocks degraded runs that omit required coverage uncertainty/review-hook signals.
- Per-stream posture emission (`temporal`, `logical`, `documentary`, `oe`) is implemented.
- Posture aggregation includes contradiction blocking for primary selection.
- Near-tie flags and sensitivity outputs are emitted.
- OE-based reinstatement path with rationale/provenance is implemented.
- RCA card includes explicit proximate/contributing/root depth summary fields.
- Recommended actions now carry explicit depth targets (`proximate|contributing|root`) and are auto-expanded to cover resolved depth layers.
- RCA card includes unresolved gaps and effectiveness monitoring plan fields.
- Analyst checkpoints for steps `0,1,2,3,3.5,4,5,6` are persisted in `run_manifest`.
- Immutable decision trail entries exist for rule-out/reinstatement/final decision events.
- Reinstatement decision-trail entries now include rationale, evidence references, and timestamp in full-mode validation.
- Replayability signature now captures deterministic gate/posture/ranking digest for same-input reproducibility checks.
- Hard-gate unit coverage now verifies elimination ordering and first-failure rationale capture.
- Step-0 scope management now persists versioned scope revisions (trigger/boundary/decision/timestamp) and tracks active approved scope in `run_context`.
- Strict-mode/coverage/posture/near-tie/OE/depth unit tests have been added.
- Full `src/dackar/RCA/unit_tests` suite is green at this checkpoint.

Primary implementation files:

- `src/dackar/RCA/orchestrators/causality_engine_v32.py`
- `src/dackar/RCA/orchestrators/rca_reasoning_orchestrator.py`
- `src/dackar/RCA/synthesis/rca_synthesizer_v31.py`
- `src/dackar/RCA/validation/schema_validator.py`
- `src/dackar/RCA/schemas/run_manifest.json`
- `src/dackar/RCA/schemas/rca_card.json`
- `src/dackar/RCA/schemas/causality_candidates.json`
- `src/dackar/RCA/schemas/causality_candidates.v3_2.schema.json`
- `src/dackar/RCA/schemas/run_context.json`
- `src/dackar/RCA/unit_tests/test_phase4_strict_mode.py`
- `src/dackar/RCA/unit_tests/test_manifest_quality.py`
- `src/dackar/RCA/unit_tests/test_refine_with_evidence.py`
- `src/dackar/RCA/unit_tests/test_synthesizer_fallback.py`
- `src/dackar/RCA/unit_tests/test_pipeline_alignment_plan.py`

### Still Open (Next 5 items)

Status tags: `[x]` done, `[ ]` not started, `(in progress)` currently being implemented.

Execute next in this order:

1. Enforce canonical candidate 4-tuple in generation output and validation checks. **[x] completed (2026-04-25)**  
   (Checklist A.1)
2. Enforce category coverage A-L: candidate-scored or explicit rule-out for each applicable category. **[x] completed (2026-04-25)**  
   (Checklist A.2)
3. Ensure physical plausibility gate executes before ranking and logs binary pass/fail rationale. **[x] completed (2026-04-25)**  
   (Checklist A.4)
4. Ensure timeline consistency gate runs in normal and degraded mode (when latency params missing). **[x] completed (2026-04-25)**  
   (Checklist A.5)
5. Ensure barrier logic gate executes where barrier/protection inputs exist; emit degraded-mode flag otherwise. **[x] completed (2026-04-25)**  
   (Checklist A.6)

After these five are complete, continue with:

- Step 1 coverage gating/acknowledgement/uncertainty hardening (Checklist B).
- Remaining Step 6 and auditability gaps (`recommended_actions` depth mapping, override/reinstatement evidence refs, replayability checks).
- Remaining hard-gate rationale unit tests (Checklist E.2).

### A. Step 4/5 Governance Core

- [x] Enforce canonical candidate 4-tuple in generation output and validation checks.
- [x] Enforce category coverage A-L: candidate-scored or explicit rule-out for each applicable category.
- [x] Add/validate rule-out reason taxonomy and ruled-out standby registry fields.
- [x] Ensure physical plausibility gate executes before ranking and logs binary pass/fail rationale.
- [x] Ensure timeline consistency gate runs in normal and degraded mode (when latency params missing).
- [x] Ensure barrier logic gate executes where barrier/protection inputs exist; emit degraded-mode flag otherwise.
- [x] Emit per-stream posture (`temporal`, `logical`, `documentary`, `oe`) for every retained candidate.
- [x] Apply posture aggregation rules so any contradicted stream blocks primary by default.
- [x] Emit near-tie flags and ranking sensitivity table in outputs.
- [x] Support OE-based reinstatement from standby with mandatory rationale and provenance.

### B. Step 1 Coverage and Uncertainty Hardening

- [x] **B.1** Emit structured data coverage report (`complete | partial | missing`) by source family.
- [x] **B.2** Gate progression on analyst acknowledgement when coverage is degraded.
- [x] **B.3** Propagate quality flags into candidate quality multiplier and run-level uncertainty summary.
- [x] **B.4** Add semantic validation to prevent silently ignoring missing critical inputs in full mode.

### C. Step 6 Depth-Complete RCA Card

- [x] Add explicit proximate/contributing/root conclusion fields to RCA card outputs.
- [x] Ensure recommended actions map to all causal depth levels (not proximate only).
- [x] Add unresolved-gap section describing what missing evidence could change the conclusion.
- [x] Add effectiveness monitoring plan fields (indicator, threshold, review horizon).

### D. Analyst Decision-Point and Auditability Controls

- [x] Persist analyst checkpoints for steps 0, 1, 2, 3, 3.5, 4, 5, and 6.
- [x] Record override/reinstatement rationale with evidence references and timestamp.
- [x] Ensure replayability: same inputs reproduce same gate/posture/ranking outputs.
- [x] Add immutable decision trail entries for rule-out and reinstatement events.

### E. Validation and Test Gates

- [x] Add unit tests for category coverage enforcement and strict-mode failures.
- [x] Add unit tests for hard-gate elimination ordering and rationale capture.
- [x] Add unit tests for per-stream posture and aggregation (including contradiction block).
- [x] Add unit tests for near-tie and sensitivity outputs.
- [x] Add unit tests for OE reinstatement path and provenance weighting.
- [x] Add unit tests for depth-complete conclusion outputs in RCA card.
- [x] Run full `src/dackar/RCA/unit_tests` suite and keep green baseline before merge.

### Wave 5 Exit Criteria

- [x] No regression in existing RCA unit tests.
- [x] Full-mode semantic validation passes for generated artifacts.
- [x] RCA card includes depth-complete conclusion + monitoring plan.
- [x] Coverage report and ruled-out log are present and auditable in run artifacts.
- [x] Analyst review behavior is deterministic for contradiction and near-tie scenarios.

---

## Backlog by Workflow Step

## Step 0 — Scoping

### Item 0.1 — Iterative scope revision mechanism (`P0`)

Implement explicit scope revision checkpoints so scope can expand/contract as new causal relations appear.

Status: **[x] completed (2026-04-25)** for initial lifecycle implementation in `run_context` (versioned revisions + active approved scope tracking + strict full-mode validation + tests).

Acceptance criteria:

- Scope revisions are represented as versioned records in run artifacts.
- Each revision includes trigger, changed boundary, analyst decision, and timestamp.
- Downstream stages consume latest approved scope version.

### Step 0 Definition of Done (Frozen)

#### Required inputs (cross-referenced against JSON schemas)

| Input | Schema | Mandatory? | Role at scoping |
| --- | --- | --- | --- |
| `event` | `event.json` | **Mandatory** | `event_id`, `asset_id`, `timestamp_start/end`, `severity`, `event_type`, `actuation_type`, `trigger_source`, `symptom_signature` |
| `telemetry_summary` | `telemetry_summary.json` | **Mandatory** | Anomaly window and signals that constrain the time boundary |
| `operational_context` | `operational_context.json` | **P0 — strongly recommended** | `mode`, `percent_rated_power`, `train_configuration`, `recent_alarms[].system_affected`, `recent_operations` → shapes system/train boundary and operating envelope |
| `pm_compliance` | `pm_compliance.json` | **P1 — often auto-built** | Overdue PM flags → informs whether Categories I/J are in initial scope |
| `cmms_context` | `cmms_context.json` | **P1** | CRs/WOs → shapes recurrence and additional component boundary (fetched in Stage B5 if not provided at intake) |
| `soe_log` | `soe_log.json` | **Conditional P0 when available** | Protection signal sequence → time window precision and barrier boundary. Tightly coupled with `protection_logic_context` |
| `alarm_log` | `alarm_log.json` | **Conditional P0 when available** | System-level alarms → which systems/trains are in scope |
| `protection_logic_context` | `protection_logic_context.json` | **Conditional P0 when SOE available** | Trip/permissive/interlock logic → required for barrier logic gate in non-degraded mode |
| `configuration_change_records` | `configuration_change_records.json` | **P1** | Recent ECNs, setpoint changes → feeds initial Category H/I scope |

**Not inputs to Step 0** (generated by downstream steps): `kg_context`, `tskr_patterns`, `causality_candidates`, `evidence_bundle`, `signal_evidence`, `rca_card`, `run_manifest`.

#### Required outputs/artifacts
- `run_context.scope_management` with `active_scope_version`, `scope_revisions`, and approved revision marker.
- `run_context.input_refs` carries:
  - `active_scope_version`, `active_scope_revision_id`
  - `event_severity`, `event_type`, `actuation_type`, `trigger_source`
  - `has_operational_context`, `has_pm_compliance`, `has_cmms_context`, `has_soe_log`, `has_alarm_log`, `has_protection_logic_context`, `has_configuration_change_records`
- `scope_snapshot` captures:
  - `asset_ids`, `component_ids` (enriched from soe_log + cmms_context)
  - `system_boundary` (from operational_context alarms + alarm_log)
  - `change_control_systems` (from configuration_change_records)
  - `operating_context` (mode, percent_rated_power, train_id, train_in_service)
  - `event_context` (severity, event_type, actuation_type, trigger_source)
  - `data_availability` flags for all optional inputs
- `run_manifest.scope_revision_summary` + `pipeline_config.scope_runtime` for audit/replay visibility.

#### Required decision checkpoints
- Initial intake decision captured as revision `scope_version=0`.
- Subsequent revisions capture analyst decision (`accepted|deferred|rejected`) and timestamp.

#### Required tests
- Schema checks for `run_context`.
- Full-mode semantic checks for scope lifecycle consistency.
- Behavior tests: initial scope with rich operational_context, alarm_log, soe_log enrichment.
- Behavior tests: accepted/deferred/rejected revision progression.

### Step Readiness Matrix (Step 0 Snapshot)

| Step | Data coverage | Processing coverage | Logic validity | Auditability | Determinism | Test status | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 — Scoping | Green | Green | Green | Green | Green | Green | **Green** |

Step 0 residual gap (to reach full Green across all future revisions):

- Auto-wiring of scope-expansion triggers from Step 2 and Step 3.5 into this lifecycle is still pending (handled in `phase2_scope_expansion_hooks_plan_april_25.md`).

## Step 1 — Data Management

### Item 1.1 — KG and corpus coverage gating (`P0`) — [x] completed (2026-04-25)

Gate execution on data adequacy (KG completeness + Chroma corpus coverage + upstream anomaly input coverage).

Acceptance criteria:

- Emits a structured coverage report with per-domain status (`complete | partial | missing`).
- Missing/stale coverage affects confidence via uncertainty propagation (not informational only).
- Analyst must explicitly acknowledge degraded data mode before continuing.

**What was implemented (2026-04-25):**
- `_build_data_coverage_summary` expanded to 8 source families: `kg_context`, `chroma_corpus`, `upstream_anomaly_inputs`, `telemetry_detail`, `soe_log`, `alarm_log`, `protection_logic_context`, `configuration_change_records`.
- Per-artifact quality fields consumed for each new family (e.g. `soe_log.quality.clock_sync_ok`, `telemetry_summary.signals[].data_quality.missing_fraction`).
- Optional families with no input use status `not_assessed` — they do not degrade `overall_status`.
- Paired-data check block (`paired_data_checks.soe_protection_logic_pairing`) added.

### Item 1.2 — External input quality propagation (`P0`) — [x] completed (2026-04-25)

Propagate anomaly/SOE/alarm quality limitations into subsequent scoring and confidence.

Acceptance criteria:

- Data quality flags from Step 1 are consumed in Steps 3-5.
- Candidate-level `quality_multiplier` and run-level uncertainty summary reflect these flags.

**What was implemented (2026-04-25):**
- `_coverage_quality_profile` in `causality_engine_v32.py` extended with weighted coverage factor across all 8 families; `not_assessed` families do not penalize the factor.
- Weights: `kg_context` 40%, `upstream_anomaly_inputs` 20%, `chroma_corpus` 15%, `telemetry_detail` 10%, assessed optional families share 15%.
- Paired-data warning (SOE present, protection logic absent) surfaced in `review_hooks.degraded_reasons`.
- Full-mode strict validation: missing Step 1 families, missing telemetry, paired-data violation, `overall_status` inconsistency, missing `paired_data_checks` block all emit errors.

### Step 1 Definition of Done (Frozen)

#### Required inputs
| Input | Schema | Mandatory? | Quality fields consumed |
|---|---|---|---|
| `event` | `event.json` | Mandatory | — |
| `telemetry_summary` | `telemetry_summary.json` | Mandatory | `signals[].data_quality.missing_fraction`, `flatline_detected`, `outlier_fraction` |
| `operational_context` | `operational_context.json` | P0 | — |
| `pm_compliance` | `pm_compliance.json` | P1 | `summary.data_quality_confidence` |
| `soe_log` | `soe_log.json` | Conditional P0 | `quality.clock_sync_ok`, `dropped_record_count` |
| `protection_logic_context` | `protection_logic_context.json` | Conditional P0 (paired with SOE) | presence |
| `alarm_log` | `alarm_log.json` | Conditional P0 | `quality.missing_fraction`, `clock_sync_ok` |
| `configuration_change_records` | `configuration_change_records.json` | P1 | `quality.coverage_status` |
| `cmms_context` | `cmms_context.json` | P1 | — |

#### Required outputs/artifacts
- `run_manifest.coverage_summary.source_families` with all 8 families present
- `run_manifest.coverage_summary.paired_data_checks.soe_protection_logic_pairing`
- `run_manifest.review_hooks.degraded_reasons` includes paired-data warning when applicable
- `run_manifest.uncertainty_summary` reflects quality degradation from new source families
- Candidate `quality_multiplier` reflects weighted coverage factor

#### Required tests
- 24 tests in `test_step1_data_coverage.py` (behavior, coverage factor, strict-mode semantic)
- Full suite 787 passed

### Step Readiness Matrix (Step 1 Snapshot)

| Step | Data coverage | Processing coverage | Logic validity | Auditability | Determinism | Test status | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 — Data Management | Green | Green | Green | Green | Green | Green | **Green** |

Step 1 residual gap:
- Feeding new optional inputs (`soe_log`, `alarm_log`, etc.) through the actual `run()` entry point requires callers to pass them. The internal pipeline wiring is complete; live adapter integration is a deployment concern.

## Step 2 — KG Expansion

### Step 2b — Temporal Search (`P0`) — [x] completed (2026-04-25)

For each component/asset in the KG find top-N past events, ranked by recency and relevance.

**What was implemented (2026-04-25):**
- `_enrich_past_events_temporal_metadata` added to orchestrator — post-processes `kg_context.past_events` after all injections (KG + CMMS) are complete.
- Each past event tagged with `in_precursor_window` (bool) and `window_tier` (primary / extended / historical / unknown) using configurable `precursor_window_days` (default 180).
- `per_component_past_events` index built in `seed_context`: `{component_id: [event_id, ...]}` capped to `per_component_past_event_top_n` (default 5) by priority score.
- `temporal_search_summary` built in `seed_context`: component count, in-window/out-of-window/unknown counts, source breakdown (kg / cmms_cr / cmms_wo).
- `run_manifest.pipeline_config.temporal_search` populated from `seed_context.temporal_search_summary`.
- `_classify_past_event_source` helper encapsulates CMMS ID prefix convention.
- `kg_context.json` schema updated with `in_precursor_window` and `window_tier` on `past_events` items.
- 20 tests in `test_step2b_temporal_search.py` — all pass.

**Residual gaps (intentionally deferred):**
- Chroma historical query per component (Step 3 already covers documentary pattern retrieval)
- Step 2a (Architectural Search / MBSE cross-reference)
- Step 2d (Fleet/industry OE similar event identification)

### Step Readiness Matrix (Step 2b Snapshot)

| Step | Data coverage | Processing coverage | Logic validity | Auditability | Determinism | Test status | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2b — Temporal Search | Green | Green | Green | Green | Green | Green | **Green** |

---

## Step 2c — Allen Relation Map

Build a signed temporal event graph between the triggering event and all anomaly, alarm, and SOE record inputs using Allen interval algebra.

**What was implemented (2026-04-25):**
- `_build_allen_relation_map` static method added to orchestrator.
  - Accepts `event`, `telemetry_summary`, `alarm_log`, `soe_log`.
  - Anomaly nodes sourced from `telemetry_summary.signals[].anomaly_window`.
  - Alarm nodes sourced from `alarm_log.alarms[]`; alarms without `acknowledged_at` treated as point events.
  - SOE records sourced from `soe_log.records[]`; always point events; large logs capped at `max_soe_nodes` (default 200).
  - Allen relation computed via existing `temporal_relations.allen_relation()` for all three input types.
  - Clock-sync failure flags (`soe_clock_sync_ok`, `alarm_clock_sync_ok`) force relation to `"unknown"` for that input family.
  - Causal candidate flag set to `True` when relation ∈ {precedes, overlaps, contains}.
- `allen_relation_map.json` schema created in `schemas/` — defines event_interval, quality_flags, summary, and nodes array.
- `event` parameter threaded into `_stage_g_finalize_manifest` and its call site in `run()`.
- `allen_relation_map` wired into run manifest at top level and within `artifacts.allen_relation_map` (summary only).
- `temporal_relations` imports added to orchestrator (`Interval`, `allen_relation`, `RELATION_SCORE`, `PRECEDES`, `OVERLAPS`, `CONTAINS`).
- 24 tests in `test_step2c_allen_relation_map.py` — all pass.

**Residual gaps (intentionally deferred):**
- Scoring-layer integration: `allen_base_score` not yet forwarded into `causality_candidates` scoring pipeline (Step 4 work).
- Step 2a (Architectural Search / MBSE cross-reference)
- Step 2d (Fleet/industry OE similar event identification)

### Step 2c Definition of Done (Frozen)

| # | Criterion | Met? |
| --- | --- | --- |
| 1 | Anomaly, alarm, and SOE nodes processed with Allen algebra | Yes |
| 2 | CRs, WOs, KG past events explicitly excluded | Yes |
| 3 | Clock-sync failure yields `unknown` relation (not a crash) | Yes |
| 4 | Large SOE logs capped safely | Yes |
| 5 | `allen_relation_map` present in manifest (top-level + artifacts block) | Yes |
| 6 | JSON schema for new artifact exists | Yes |
| 7 | 24 tests pass, zero regressions | Yes |

### Step Readiness Matrix (Step 2c Snapshot)

| Step | Data coverage | Processing coverage | Logic validity | Auditability | Determinism | Test status | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2c — Allen Relation Map | Green | Green | Green | Green | Green | Green | **Green** |

---

## Phase 3b — Scope-Expansion Hooks (Steps 2c / 3 / 3.5 → Step 0)

Wire the outputs of Step 2c (Allen map), signal evidence propagation chains, and TSKR novel patterns into the Step 0 scope revision lifecycle as auto-detected "expansion suggested" signals, surfaced at Human Decision checkpoints.

**What was implemented (2026-04-25):**
- `_detect_scope_expansion_signals` static method:
  - **Source 1 — Allen relation map:** causal-candidate nodes whose `component_id` is not in the current scope boundary → `out_of_scope_causal_component` (severity: warning).
  - **Source 2 — signal evidence propagation chains:** chain components outside scope → `out_of_scope_propagation_component` (severity: warning).
  - **Source 3 — TSKR patterns:** patterns with `novel_pattern=True` or `match_count=0` → `novel_signal_pattern` (severity: info).
  - De-duplicates by `signal_id`; skips filtering when scope boundary is empty.
- `_inject_scope_expansion_signals` method: merges new signals into `run_context.scope_management.expansion_suggestions`; idempotent on `signal_id`.
- Wired into `run()` after Step 2c map build, before `_stage_g_finalize_manifest`; saves updated `run_context` when new signals are detected.
- `scope_expansion_summary` (total_signals, pending_analyst_decision, by_trigger_type) computed in `_stage_g_finalize_manifest` and propagated to manifest top level.
- `review_hooks.analyst_decisions_required` and `review_hooks.degraded_reasons` populated when pending signals > 0.
- `run_context.json` schema updated with `expansion_suggestions` array in `scope_management`.
- **Step 2d stub:** `similar_event_list` artifact wired into manifest with `status: "not_implemented"`; `similar_event_list.json` schema created (plant/fleet/industry event structure with confidence weights).
- 17 tests in `test_step3b_scope_expansion_hooks.py` — all pass; total suite 848 passed.

**Step 2d full implementation planned (2026-04-25):** detailed plan in `step2d_similar_event_plan_april_25.md`. Three-tier architecture:
- **Plant:** direct query of `kg_context.past_events` (always runs, in-memory, scored on 5 match dimensions).
- **Fleet:** pluggable `SimilarEventAdapter` Protocol backed by `LLMOEAdapter` calling a fine-tuned fleet OE API.
- **Industry:** same adapter, separate endpoint for INPO SOER / EPRI / NRC LER databases.
Matched events are surfaced in OE reinstatement rationale; `unresolved_gaps` emits when plant count is zero or a tier is degraded.

**Step 2d implementation completed (2026-04-25):** 23 tests pass, 1003 full-suite tests pass, zero regressions.

**Implemented changes:**
- **Schema upgrade (`similar_event_list.json`):** added `query_terms`, `summary` (plant/fleet/industry counts, `degraded_tiers`, `any_plant_match`), and richer event fields (`match_dimensions`, `root_cause_label`, `resolution`, `lessons_learned_ref`, `contributing_categories`, `window_tier`, `actuation_type`).
- **`adapters/` package** created with two new files:
  - `similar_event_adapter.py` — `SimilarEventAdapter` Protocol (`@runtime_checkable`); `TIER_CONFIDENCE_MULTIPLIERS` dict (plant 1.0, fleet 0.80, industry 0.60).
  - `llm_oe_adapter.py` — `LLMOEAdapter`: structured prompt builder; POST to fleet/industry endpoints; parses list/dict/wrapped responses; `degraded` flag on any failure; `last_error` attr.
- **`RCAReasoningOrchestrator`:**
  - `similar_event_adapter: Optional[Any] = None` class field + `set_similar_event_adapter()` method.
  - New static `_query_plant_past_events(...)`: scores `kg_context.past_events` on 5 dimensions (component +0.40, FM +0.25, event_type +0.15, actuation +0.10, precursor-window +0.10); returns top-N sorted by confidence_weight.
  - New instance method `_build_similar_event_list(...)`: plant tier always runs; fleet/industry via injected adapter; merges results; emits `query_terms`, `summary`, `status`, `provenance`.
  - New static `_annotate_candidates_with_oe_evidence(...)`: annotates top candidates' `oe_reinstatement_evidence` with matched events (threshold ≥ 0.30); matches on `component_id` or `failure_mode_id`/`failure_signature`.
  - `similar_event_list` built before `synthesize()` (so OE gaps feed the RCA card); re-used in `_stage_g_finalize_manifest` via `pre_computed_similar_event_list` param.
- **`synthesize()` / `_fallback_card()`** in synthesizer: both accept `similar_event_list: Optional[JsonDict] = None`; passed to `_build_unresolved_gaps`.
- **`_build_unresolved_gaps`**: new `similar_event_list` param; emits gap when `plant_count=0`; emits gap per degraded tier.
- **Manifest `artifacts.similar_event_list`** block upgraded from stub to live summary with `plant_count`, `fleet_count`, `industry_count`, `any_plant_match`, `degraded_tiers`.

**Analyst workflow integration:**
- After pipeline completes, `run_manifest.scope_expansion_summary.pending_analyst_decision > 0` signals that the analyst must act on expansion suggestions before writeback.
- Analyst uses `apply_scope_revision()` to accept/defer/reject each suggestion, which updates `run_context.scope_management`.

### Phase 3b Definition of Done (Frozen)

| # | Criterion | Met? |
| --- | --- | --- |
| 1 | Three signal sources: Allen map, propagation chains, TSKR novel patterns | Yes |
| 2 | `expansion_suggestions` persisted in `run_context.scope_management` | Yes |
| 3 | Idempotent injection (re-runs don't duplicate signals) | Yes |
| 4 | `scope_expansion_summary` in manifest with pending count and by_trigger_type | Yes |
| 5 | `review_hooks.analyst_decisions_required` populated when pending > 0 | Yes |
| 6 | `run_context.json` schema documents `expansion_suggestions` | Yes |
| 7 | Step 2d `similar_event_list` stub wired with schema | Yes |
| 8 | 17 tests pass, zero regressions | Yes |

### Step Readiness Matrix (Phase 3b Snapshot)

| Phase | Signal detection | Injection/persistence | Manifest surfacing | Analyst hooks | Test status | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| 3b — Scope-Expansion Hooks | Green | Green | Green | Green | Green | **Green** |

---

## Step 3.5 — Signal Pattern Recognition (Signal Lessons Learned)

Match current anomaly, alarm, and SOE log patterns against historical signatures for same/similar components; emit `signal_lessons_learned` artifact with matched patterns and novel-pattern flags.

**What was implemented (2026-04-25):**
- `novel_pattern` boolean added to every `tskr_patterns.patterns[]` entry — True when `recurrence_count == 0`, history score < 0.20, and no signal IDs match.
- `n_novel_patterns` / `has_novel_patterns` added to `tskr_patterns.summary`.
- `alarm_log` and `soe_log` threaded into `TSKRTemporalScorerV1.score()` via new `_extract_alarm_windows` and `_extract_soe_windows` static methods; alarm/SOE point-event windows merged into the anomaly pool for pattern scoring; clock-sync failures mark windows as degraded without crashing.
- Backward-compatible: `inspect.signature` guard in `_build_tskr_patterns` skips new params when scorer doesn't support them.
- `_build_signal_lessons_learned` static method (orchestrator):
  - Splits tskr_patterns into `matched_patterns` (recurrence_count > 0 or support ≥ 0.20) and `novel_patterns`.
  - Attaches `causal_explanation` and `resolution_summary` from recurrence profile fields.
  - Populates `input_sources` list reflecting which log types contributed windows.
- `signal_lessons_learned.json` schema created in `schemas/`.
- `signal_lessons_learned` wired into manifest top level and `artifacts.signal_lessons_learned` (summary: `total_matched`, `novel_pattern_flag`, `n_novel_patterns`, `input_sources`).
- Novel-pattern outcomes already forwarded to scope-expansion suggestions via Phase 3b hooks.
- 19 tests in `test_step35_signal_lessons_learned.py` — all pass; total suite 867 passed.

### Step 3.5 Definition of Done (Frozen)

| # | Criterion | Met? |
| --- | --- | --- |
| 1 | `novel_pattern` flag on every `tskr_patterns` pattern entry | Yes |
| 2 | Alarm and SOE windows merged into anomaly pool for pattern scoring | Yes |
| 3 | Clock-sync failure handled gracefully (degraded flag, no crash) | Yes |
| 4 | `signal_lessons_learned` artifact with matched/novel separation | Yes |
| 5 | Causal explanation and resolution summary from recurrence data | Yes |
| 6 | `signal_lessons_learned.json` schema exists | Yes |
| 7 | Artifact wired into manifest (top-level + artifacts block) | Yes |
| 8 | 19 tests pass, zero regressions | Yes |

### Step Readiness Matrix (Step 3.5 Snapshot)

| Step | Data coverage | Processing coverage | Logic validity | Auditability | Determinism | Test status | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3.5 — Signal Pattern Recognition | Green | Green | Green | Green | Green | Green | **Green** |

## Step 4 — Candidate Generation and Initial Ranking

### Item 4.2 — A-L category coverage enforcement (`P0`)

For each event, each applicable category must have scored candidate(s) or explicit rule-out.

Acceptance criteria:

- Coverage report exists for categories A-L with status and rationale.
- `unknown` coverage status is not allowed in strict/full mode.

## Step 5 — Ranking and Evidence Assessment

### Item 5.1 — Hard Gate phase completeness (`P0`)

Implement strict elimination-first gates:

- Gate 1 Physical plausibility
- Gate 2 Timeline consistency (with degraded mode when latency unavailable)
- Gate 3 Barrier logic

Acceptance criteria:

- Gate outcomes are binary and auditable per candidate.
- Eliminated candidates move to ruled-out log with gate reason.
- Degraded-mode behavior is explicit and test-covered.

### Item 5.2 — Sensitivity table (`P1`) [x] completed (2026-04-25)

Project composite-score deltas for top-N candidates if each currently missing/not_assessed/partial data source were available at full quality.

**Implemented (2026-04-25):**

- `sensitivity_table.json` schema — `event_id`, `generated_at`, `summary`, `rows`, `provenance`.  Each row: `candidate_id`, `candidate_rank`, `source_family`, `current_status`, `current_composite_score`, `estimated_composite_if_available`, `estimated_score_delta`, `would_change_ranking`.
- `RuleBasedCausalityEngineV32._build_sensitivity_table` static method added: iterates degraded source families (missing / not_assessed / partial), patches each to `complete`, re-runs `_coverage_quality_profile`, and rescales `composite_raw` by the new factor to produce an upper-bound estimate.
- Called from `refine_with_evidence` after final candidate ranking; `sensitivity_table` keyed into `payload`.
- Orchestrator `_stage_g_finalize_manifest`: `RuleBasedCausalityEngineV32._build_sensitivity_table(...)` called with final retained candidates + coverage summary; result stored at `run_manifest.sensitivity_table` and summarised in `run_manifest.artifacts.sensitivity_table`.
- `analyst_attention_flags` receives `"SENSITIVITY: missing data could alter candidate ranking — review sensitivity_table"` when `summary.any_ranking_change_possible` is True.
- 25 tests in `test_step5_sensitivity_table.py`; 892 tests pass, zero regressions.

### Step 5 Definition of Done (Frozen)

| # | Criterion | Met? |
| --- | --- | --- |
| 1 | Hard gates (Physical Plausibility, Timeline, Barrier Logic) implemented and auditable | Yes |
| 2 | v1→v2 rank delta (`_build_scoring_evolution`) surfaced in manifest | Yes |
| 3 | Near-tie detection and flagging in `decision_posture` | Yes |
| 4 | OE reinstatement path with rationale | Yes |
| 5 | Replayability signature (SHA-256) | Yes |
| 6 | Sensitivity table: per-candidate, per-source delta estimation | Yes |
| 7 | `analyst_attention_flags` injection when ranking change is possible | Yes |
| 8 | `sensitivity_table.json` schema exists | Yes |
| 9 | 25 sensitivity-table tests pass, 892 total tests pass | Yes |

### Step Readiness Matrix (Step 5 Snapshot)

| Step | Data coverage | Processing coverage | Logic validity | Auditability | Determinism | Test status | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 — Ranking & Evidence | Green | Green | Green | Green | Green | Green | **Green** |

## Step 6 — Conclusion

### Item 6.3 — Effectiveness monitoring plan and unresolved gaps (`P1`) [x] completed (2026-04-25)

Complete forward-looking closure requirements.

**Implemented (2026-04-25):**

- **WS1 — Human Performance Assessment** (`_build_human_performance_assessment`):
  - New `human_performance_assessment` block added to `rca_card.json` schema (required fields: `applicable`, `category_flags`, `findings`, `provenance_note`).
  - Scans retained candidates for H/I/J/K categories; maps category → `performance_mode` (execution_error / procedure_gap / knowledge_gap / supervisory_gap) and AP-913 regulatory reference.
  - `corrective_action_ids` cross-referenced from `recommended_actions.linked_candidate_id`.
  - Injected deterministically after both LLM and fallback card paths so the field is always present.
- **WS2 — Deepened `_build_unresolved_gaps`**:
  - Now accepts `causal_depth_summary`, `sensitivity_any_change`, and `novel_pattern_flag`.
  - Emits a gap entry when the contributing layer (G–K) is empty.
  - Emits a gap entry when the root layer (Category L) is unresolved.
  - Emits a gap entry when the sensitivity table flags a possible ranking change from missing data.
  - Emits a gap entry when novel TSKR patterns are detected.
  - Cap raised from 6 to 8 items.
- **WS3 — Depth-stratified `_build_effectiveness_monitoring_plan`**:
  - Three depth profiles: proximate (equipment health, 90d), contributing (process/procedure, 180d), root (programmatic/fleet OE, 365d).
  - New `causal_depth_level` and `success_criteria` fields on every plan item (added to `rca_card.json` schema).
  - Plan cap raised to 5 items; fallback entry also carries both new fields.
- **WS4 — `depth_incomplete_reason` in `causal_depth_summary`**:
  - `depth_incomplete_reason` string field added to `rca_card.json` `causal_depth_summary` object (optional, present only when `depth_complete=false`).
  - `_build_causal_depth_summary` now computes which layers are incomplete and assembles an analyst-readable explanation.
  - Fallback card also populates the reason string.
- **41 tests** in `test_step6_conclusion.py` — all pass; **933 total tests pass, zero regressions**.

### Step 6 Definition of Done (Frozen)

| # | Criterion | Met? |
| --- | --- | --- |
| 1 | `primary_hypothesis` with cause_label, confidence_label, composite_score | Yes |
| 2 | `contributing_causes[]` with candidate_id, rationale, category | Yes |
| 3 | `causal_depth_summary` — proximate / contributing / root + `depth_complete` | Yes |
| 4 | `depth_incomplete_reason` when depth_complete=false | Yes |
| 5 | `unresolved_gaps[]` — depth layers, sensitivity, novel patterns, data quality | Yes |
| 6 | `effectiveness_monitoring_plan[]` — depth-stratified indicators, success_criteria | Yes |
| 7 | `human_performance_assessment` — applicable flag, category_flags, findings, regulatory refs | Yes |
| 8 | `barrier_analysis` summary | Yes |
| 9 | `analyst_review` — decision_required, questions, writeback_recommendation | Yes |
| 10 | 41 targeted tests pass, 933 total tests pass | Yes |

### Step Readiness Matrix (Step 6 Snapshot)

| Step | Data coverage | Processing coverage | Logic validity | Auditability | Determinism | Test status | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 — Conclusion | Green | Green | Green | Green | Green | Green | **Green** |

---

## Phase 5 — Full Workflow Logic Audit + Finding 4 Fix (completed 2026-04-25)

### Phase 5 Audit Findings

| # | Finding | Resolution |
| --- | --- | --- |
| A | `_build_unresolved_gaps` was reading `sensitivity_any_change`/`novel_pattern_flag` from `run_context` (dead reads) | Fixed: reads from `causality_candidates["sensitivity_table"]` and `tskr_patterns` parameter |
| B | `allen_relation_map` not fed into synthesizer | Accepted by design: manifest-level only |
| C | Phase 3b scope signals emitted after card build (cannot influence same-run card) | Accepted by design: next-run analyst inputs |
| D | `generate()` does not consume `coverage_summary` | Accepted by design: quality penalty in `refine_with_evidence` only |
| E | `sensitivity_table.json` `top_n_candidates` had `minimum: 1` (breaks empty-candidate runs) | Fixed: `minimum: 0` |
| F | `_build_signal_lessons_learned` could emit null `pattern_id` / `confidence` | Fixed: string/float coercion with fallback |
| G | Allen map not the temporal score source (TSKR is proxy) | Declared limitation: consistent with metamodel notes |
| H | Category E `operating_point` not in scoring | **Planned (2026-04-25)** — plan in `finding_h_operating_point_plan_april_25.md` |
| I | Physical plausibility gate uses structural proxy in degraded mode | Declared limitation: barrier gate relies on KG safety functions — **Fixed by Finding I** |
| **4** | **Categories F/K/L have dedicated schemas but absent from `coverage_summary`** | **Fixed — see below** |

### Finding 4 — Categories F / K / L Coverage Visibility (completed 2026-04-25)

**Problem:** `environmental_monitoring` (Cat. F), `vendor_supply_chain_records` (Cat. K), and `training_records` (Cat. L) had JSON schemas but the orchestrator's `run()` signature did not accept them, and `_build_data_coverage_summary` did not assess their presence. Their absence was invisible to the analyst and failed validator checks in full mode.

**Changes:**
- `run()` now accepts `environmental_monitoring`, `vendor_supply_chain_records`, `training_records` as optional `JsonDict` parameters; each is also checked against `input_refs` availability flags.
- `_build_data_coverage_summary` assesses each family with field-level quality heuristics, assigning `not_assessed` / `complete` / `partial`.
- `_stage_g_finalize_manifest` signature and its call site in `run()` updated to thread the three new args.
- Validator `all_expected_families` set (full mode) expanded to include the three families — `not_assessed` is a valid status.
- `ALL_EXPECTED_FAMILIES` in `test_step1_data_coverage.py` updated; all 24 Step 1 tests pass; 933 total tests pass, zero regressions.

---

## Finding G — Wire `allen_base_score` into Causality Scoring (completed 2026-04-25)

**Status:** Complete — detailed plan in `finding_g_allen_scoring_plan_april_25.md`

**Problem:** The `allen_relation_map` (Step 2c) attaches an `allen_base_score` and `allen_relation_to_event` to every anomaly/alarm/SOE node, but `refine_with_evidence` never saw the map — it was built *inside* `_stage_g_finalize_manifest`, which runs **after** scoring. Allen scores were computed and discarded.

**Changes implemented (6 workstreams):**
- **WS1 (Sequencing):** Extracted `_build_allen_relation_map` call to before `refine_with_evidence` in `run()`; result stored as `pre_refine_allen_map`, reused in `_detect_scope_expansion_signals` and passed to `_stage_g_finalize_manifest` as `pre_computed_allen_map` — no rebuild.
- **WS2 (Engine):** `refine_with_evidence` accepts `allen_relation_map: Optional[JsonDict] = None`. New static helpers:
  - `_build_allen_component_index`: indexes causal nodes by component_id; applies SOE clock-sync discount (×0.80); collects `follow_ids` set.
  - `_apply_allen_temporal_blend`: blend α=0.25 (`new_temporal = 0.75×old + 0.25×allen`); only raises temporal; updates `composite_raw` / `composite_score` by temporal weight delta; sets `temporal_contradiction=True` for `follows` nodes.
- **WS3 (Gates):** No code change needed — `_apply_timeline_consistency_gate` already reads `temporal_evidence["temporal_contradiction"]`; Allen contradiction automatically triggers the gate and rules out the candidate.
- **WS4 (Rationale):** `_update_score_rationale_for_refinement` appends Allen blend note (`allen_base_score`, `relation`, `blended_temporal`) or contradiction note to `score_rationale["temporal"]`.
- **WS5 (Tests):** 25 tests in `test_finding_g_allen_scoring.py` — all pass; **958 total tests pass, zero regressions**.
- **WS6 (Docs):** This section + metamodel update.

**Blend formula:** `new_temporal = 0.75 × TSKR_temporal + 0.25 × allen_score` (causal-match only; no-match leaves temporal unchanged).  
**Contradiction:** Allen `follows` → `temporal_evidence["temporal_contradiction"] = True` → timeline gate rules out the candidate.  
**Backward-compatible:** `allen_relation_map=None` default; zero impact on existing tests.

### Finding G Definition of Done

| # | Criterion | Met? |
| --- | --- | --- |
| 1 | Allen map built before `refine_with_evidence` in `run()` | Yes |
| 2 | `_stage_g_finalize_manifest` uses pre-computed map (no rebuild) | Yes |
| 3 | `_build_allen_component_index` static helper with SOE clock discount | Yes |
| 4 | `_apply_allen_temporal_blend` static helper with α=0.25 blend and follows contradiction | Yes |
| 5 | `scores["allen_temporal_score"]`, `scores["allen_relation"]`, `scores["allen_blend_applied"]` stored | Yes |
| 6 | `temporal_contradiction` flagged for `follows` nodes → timeline gate fires | Yes |
| 7 | 25 targeted tests pass | Yes |
| 8 | Full suite 958 tests pass, zero regressions | Yes |

### Step Readiness Matrix (Finding G Snapshot)

| Dimension | Before | After |
| --- | --- | --- |
| Allen map build timing | After refine (manifest only) | Before refine (scoring input) |
| Temporal score source | TSKR only | TSKR primary + Allen blend (α=0.25) |
| Contradiction detection | TSKR/spaCy | TSKR/spaCy + Allen `follows` |
| Score traceability | No Allen fields | `allen_temporal_score`, `allen_relation`, `allen_blend_applied` |

---

## Finding I — Direct `protection_logic_context` Read in Hard Gates (completed 2026-04-25)

**Status:** Complete — 22 tests pass, 980 full-suite tests pass, zero regressions.

**Problem:** Both protection-evidence hard gates operated on structural proxies. The physical plausibility gate only checked `scores["structural"] < 0.20`. The barrier logic gate relied on a circular `ruleout.reason_code = "barrier_held"` pre-condition that was never externally set — it effectively always passed. `protection_logic_context`, which carries direct `barrier_states` and `logic_sets` data, was never read by either gate.

**Implemented changes:**

- **WS1 — `_build_plc_barrier_index`** (new static helper in causality engine): Parses `barrier_states` into `{sf_id → state}` dict; flattens all `input_signals` and `output_signals` from every `logic_set` into a single `set[str]`. Both are built once per `refine_with_evidence` call.
- **WS2 — `_apply_physical_plausibility_gate`** updated: Accepts `plc_logic_signal_ids` and `plc_sf_state`. When `component_id` is in `plc_logic_signal_ids`, sets `plc_consulted=True` and enriches the `rationale` with whether any `affected_safety_functions` has `state="held"`. Gate continues to pass; PLC information is contextual only.
- **WS3 — `_apply_barrier_logic_gate`** updated: Accepts `plc_sf_state`. Iterates `affected_safety_functions`, looks each `sf_id` up in `plc_sf_state`; `state∈{failed,degraded}` → forces gate failure (`plc_forced_fail=True`); `state="held"` → noted in rationale but does not block. `plc_consulted` flag added to the gate record.
- **WS4 — `refine_with_evidence`** updated: New param `protection_logic_context: Optional[JsonDict] = None`. Builds PLC index once via `_build_plc_barrier_index`; passes `plc_logic_signal_ids` and `plc_sf_state` into both gate calls.
- **WS5 — Orchestrator** updated: `refine_kwargs` includes `protection_logic_context` via inspect-guard (same pattern as Allen). `_run_auto_reentry_if_needed` signature and internal `refine_with_evidence` call updated.
- **WS6 (Tests):** 22 tests in `test_finding_i_plc_gates.py`:
  - `_build_plc_barrier_index`: 6 tests (None, empty, barrier states, signal collection, multi-set merge, null-signal skip)
  - `_apply_physical_plausibility_gate`: 5 tests (baseline, low structural, PLC consulted, not consulted, held-SF rationale note)
  - `_apply_barrier_logic_gate`: 6 tests (no-PLC degraded pass, failed SF blocks, degraded SF blocks, held SF passes with note, unknown passes, SF not in PLC not consulted)
  - Integration with `refine_with_evidence`: 4 tests
  - Orchestrator parameter declaration: 1 test

### Step Readiness Matrix (Finding I Snapshot)

| Dimension | Before | After |
| --- | --- | --- |
| Physical plausibility data source | `scores["structural"]` only | `scores["structural"]` + PLC logic_signal membership |
| Barrier gate fail trigger | Never (circular ruleout pre-condition) | PLC `barrier_state ∈ {failed, degraded}` for matched sf_id |
| Barrier gate traceability | No PLC fields | `plc_consulted` flag + PLC state notes in rationale |
| Degraded mode clarity | Always degraded when no barrier_signal | `plc_consulted=True` upgrades from degraded when PLC present |
| Test count | 958 | 980 (+22) |

---

**Barrier-held threshold:** `barrier_signal ≥ 0.80` — only fails candidates whose failure mode is fundamentally about a safety-critical function (not ordinary components associated with safety functions).  
**Backward-compatible:** `protection_logic_context=None` default; no existing test affected.

---

## Finding H — Category E `operating_point` in Scoring (completed 2026-04-25)

**Plan doc:** `finding_h_operating_point_plan_april_25.md`  
**Status:** Complete ✅

**Problem:** Category E candidates (demand/transient/envelope causes) scored identically regardless of whether the plant was at 100% power on a power ramp vs cold shutdown. `operational_context` was collected (Finding 4) but not used in causal scoring.

**Changes implemented (5 workstreams):**
- **WS1 (Helper):** New `_operating_point_score` static helper in `causality_engine_v32.py`.
  - Mode base lookup table (7 modes: `power_ramp`=0.70 → `shutdown`=0.20).
  - Cat E–only power modifier: `percent_rated_power × 0.30` for high-demand keywords; `(1−pnorm) × 0.25` for standby keywords.
  - Universal train OOS + standby bonus: `+0.15` when `in_service=False` and standby keyword in fm text.
  - Returns `(0.0, "not_assessed")` when `operational_context` is absent — never penalises candidates.
  - Score capped at 1.0.
- **WS2 (Structural):** `_build_failure_mode_candidates` updated:
  - **Sequencing fix:** `_infer_primary_category_for_failure_mode` moved to *before* the structural assembly block so `op_delta` can use the correct category.
  - Operating-point delta `op_delta = 0.12 × op_score` added to structural sum (max +0.12).
  - `scores["operating_point_score"]` and `scores["operating_point_note"]` stored on every candidate.
- **WS3 (Rationale):** `_build_failure_mode_candidates` initial `score_rationale["structural"]` string now includes `op_point ... → delta ...` when active. `_update_score_rationale_for_refinement` appends the operating-point note to `score_rationale["structural"]` after refinement.
- **WS4 (Tests):** 20 tests in `test_finding_h_operating_point.py` — all pass; **1023 total tests pass, zero regressions**.
- **WS5 (Docs):** This section + metamodel update.

**Score contribution:** `op_delta ∈ [0, 0.12]` — advisory, additive to structural, never dominant.  
**Category E interaction:** Power-level modifier (±) applies only to Cat E candidates; mode base and train bonus apply to all categories.  
**Backward-compatible:** `operational_context=None` → `op_score=0.0`, `op_note="not_assessed"`, zero delta.

### Finding H Definition of Done

| # | Criterion | Met? |
| --- | --- | --- |
| 1 | `_operating_point_score` static helper with mode table, Cat E power modifier, train bonus | Yes |
| 2 | Category inference moved before structural assembly | Yes |
| 3 | `op_delta` (≤ 0.12) added to structural | Yes |
| 4 | `operating_point_score` / `operating_point_note` stored in `scores` dict | Yes |
| 5 | Initial `score_rationale["structural"]` includes op note when active | Yes |
| 6 | `_update_score_rationale_for_refinement` appends op note post-refinement | Yes |
| 7 | 20 targeted tests pass | Yes |
| 8 | Full suite 1023 tests pass, zero regressions | Yes |

### Step Readiness Matrix (Finding H Snapshot)

| Dimension | Before | After |
| --- | --- | --- |
| Operational mode in scoring | Ignored | Mode base table (7 modes) → structural delta |
| Power level in scoring | Ignored | Cat E: `percent_rated_power` × keyword modifier |
| Train OOS in scoring | Ignored | Standby-mechanism candidates: +0.15 |
| Category inference timing | After structural assembly | Before structural assembly |
| Score traceability | No op fields | `operating_point_score`, `operating_point_note`, rationale note |
| Test count | 1003 | 1023 (+20) |

---

## Scope-Revision Downstream Propagation (completed 2026-04-25)

**Plan doc:** `scope_revision_downstream_plan_april_25.md`  
**Status:** Complete ✅

**Problem:** Step 0 scope revision lifecycle writes decisions (accept/defer/reject) but those decisions had no downstream enforcement. `causality_engine.generate()` fed all KG failure modes through unconditionally regardless of the analyst-approved scope boundary. The scope version number was surfaced in the manifest but had zero mechanical effect.

**Changes implemented (6 workstreams):**
- **WS1 (Helpers):** Two new static methods in `rca_reasoning_orchestrator.py`:
  - `_resolve_approved_scope_boundary(run_context) → Optional[FrozenSet[str]]`: walks `scope_revisions[]` backwards to find the latest accepted revision; returns normalised (lower-case) frozenset of `component_ids` when `active_scope_version > 0`; returns `None` when version == 0 (discovery mode) or boundary is empty.
  - `_apply_scope_boundary_filter(candidates, boundary, scope_version) → JsonDict`: soft-filters candidates — moves out-of-scope `component_id`s to `ruled_out[]` with `reason_code="scope_filtered"` and `hard_gate=False`; stores `scope_filter_applied`, `scope_filter_version`, `scope_filter_filtered_count`, `scope_filter_filtered_component_ids` on the candidates dict.
- **WS2 (`apply_scope_revision` enhancement):** When `analyst_decision="accepted"` and caller supplies no explicit `scope_snapshot`, the method auto-builds one by copying the latest accepted snapshot and unioning `changed_boundary.added_component_ids` / subtracting `removed_component_ids`. Removes the burden of snapshot reconstruction from callers.
- **WS3 (`resolve_expansion_suggestion`):** New instance method. Atomic: marks a suggestion's `analyst_decision`, stores `resolution_timestamp` and optional `analyst_rationale`, and — when accepted — delegates to `apply_scope_revision` with the suggestion's `suggested_component_ids` as `added_component_ids`. Raises `ValueError` on unknown `signal_id`.
- **WS4 (Wire into `run()`):** After `generate()` and before `refine_with_evidence`, calls `_resolve_approved_scope_boundary`; if non-None applies `_apply_scope_boundary_filter`. Stores `scope_filter` summary in `pipeline_runtime` and surfaces it in `run_manifest.artifacts.scope_filter`.
- **WS5 (Tests):** 26 tests in `test_scope_revision_downstream.py` — all pass; **1049 total tests pass, zero regressions**.
- **WS6 (Schema + Docs):** `causality_candidates.json` `reason_code` enum extended with `"scope_filtered"` at both `ruleout` locations. This section + metamodel update.

**Filter mode:** Soft (rule-out, never delete) — preserves full audit trail; analyst can widen scope to reinstate.  
**Discovery mode:** Version 0 → filter not activated; all KG candidates flow through.  
**Backward-compatible:** Existing run_contexts with `active_scope_version=0` are unaffected.

### Scope-Revision Downstream Definition of Done

| # | Criterion | Met? |
| --- | --- | --- |
| 1 | `_resolve_approved_scope_boundary` returns None for v0, frozenset for v≥1 | Yes |
| 2 | `_apply_scope_boundary_filter` moves out-of-scope → ruled_out with scope_filtered | Yes |
| 3 | `apply_scope_revision` auto-merges added/removed component IDs | Yes |
| 4 | `resolve_expansion_suggestion` atomically marks suggestion + updates scope | Yes |
| 5 | Filter wired into `run()` between generate() and refine_with_evidence | Yes |
| 6 | `scope_filter` block in `run_manifest.artifacts` | Yes |
| 7 | 26 targeted tests pass | Yes |
| 8 | Full suite 1049 tests pass, zero regressions | Yes |

### Step Readiness Matrix (Scope-Revision Downstream Snapshot)

| Dimension | Before | After |
| --- | --- | --- |
| Scope decision enforceability | Documentation only | Filters generate() output when version > 0 |
| Expansion suggestion lifecycle | pending → never resolved | accept/defer/reject → scope version bumped |
| Out-of-scope candidates | Scored and ranked to analyst | Moved to ruled_out with scope_filtered reason |
| `apply_scope_revision` usability | Caller must reconstruct full snapshot | Auto-merged from changed_boundary |
| Manifest traceability | Scope version only | Full scope_filter block (filtered count, CIDs) |
| Test count | 1023 | 1049 (+26) |


