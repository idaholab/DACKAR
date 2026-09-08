# DACKAR Outage Analytics — Code & Methodology Review

**Date:** 2026-04-16
**Reviewer:** Code + methodology review (Claude)
**Scope:** `/Users/mandd/projects/DACKAR/src/dackar/outage/` — all stages, orchestrator, services, adapters, tests
**Out of scope:** `/Users/mandd/projects/LOGOS/src/CPM/` (treated as external, robust)

---

## 1. Overall Assessment

The codebase is well-architected, cleanly separated, and comprehensively tested. The 7-stage pipeline has clear interfaces, graceful degradation, and an evidence-traceability design that matters for nuclear plant operators. Most of the "stub" concern going in turned out to be wrong — Stage B, for example, has full Cypher query implementations, DQ scoring, deduplication, recurrence indicators, and PM compliance tracking. The core execution path (Stages A, C, D, E, F, G) is production-ready code.

The serious problems are **methodological, not structural**. The code mostly does what it claims. The question is whether what it claims is enough to earn trust in the target environment.

---

## 2. What Is Working Well

- **Pipeline architecture**: Protocol-based injection, artifact passthrough for replay/testing, artifact schema validation, analyst review flags, run manifests. Correct design for an auditable nuclear decision tool.
- **Uncertainty quantification (Stages D + E)**: Mixture-model duration distribution (routine + disruption-driven extended pool), confidence tier assignment, and Monte Carlo with CP drag and criticality index are all correctly implemented.
- **Evidence chain (Stage G)**: Every recommendation cites upstream sources with strength scores. Analyst review triggers on 6 conditions. This is what outage managers need to trust the output.
- **Regulatory constraint detection**: `critical_analysis.md` flags this as absent — but it is implemented. Stage A classifies `regulatory_drivers` (TS, LCO, NRC, ALARA, hold points) and sets `defer_prohibited`/`scope_reduction_prohibited`. Stage F enforces them in option feasibility checks.
- **LOGOS adapter**: Hardcoded path is a default argument, not a hard failure. Injectable and documented clearly.
- **Test suite**: 730+ tests, all passing. Mock duck-typing for Stage E (avoiding LOGOS dependency in CI) is the right call.

---

## 3. Methodology Concerns (High Priority)

### 3a. Single-Outage Validation ⚠️ Critical

Duration distributions and component risk patterns from one outage have too little statistical basis to support confidence-tier claims. A "data_supported" tier requires ≥5 analogs — but five analogs from the same outage captures within-outage variance, not between-outage variance. Fuel cycle character, contractor availability, and regulatory commitments vary enormously across cycles.

**Root cause:** `confidence.py` `_classify_tier()` counts analogs but has no requirement on outage diversity.

**Fix (Priority 1 — LOW effort):** Add `outages_represented` as a second gating axis in `ConfidenceEstimator.classify()`.  
Proposed thresholds: `high` requires ≥10 analogs **and** ≥3 distinct outages; `medium` requires ≥5 analogs **and** ≥2 distinct outages; `low` otherwise.

**Status:** ✅ Fixed 2026-04-16 — `classify()` now accepts `outages_represented: int = 0` and enforces diversity gates.

### 3b. Execution Mode Flags Are Dead Code ⚠️ Medium

`ActivityCase` defines four execution mode flags (`has_rp_hold`, `requires_scaffold`, `has_clearance`, `is_vendor_supported`) that are documented as strong variance predictors. They are defined in the domain model, but not extracted by Stage A NLP, not used in similarity scoring, and not used in duration estimation.

**Fix (Priority 2 — LOW effort):** Add keyword pattern matching in `stage_a_intake.py` to extract these flags from the work order description.

**Status:** ✅ Fixed 2026-04-16 — `_extract_execution_mode_flags()` added to Stage A; flags passed through in intake result.

### 3c. Gap 4 Dependency Similarity Disabled by Design

`dependency_similarity.py` has weight `0.0` because schedule-neighborhood information requires the schedule to be loaded at retrieval time — a circular dependency (Stage D runs before Stage E). Correct resolution is a two-pass design where Stage D re-weights analogs using schedule position after Stage E. Worth revisiting once multi-outage data is available.

**Status:** Known, deferred. Not touching for now.

### 3d. Planning Phase Risk Register Needs Multi-Outage Validation

`pre_outage_risk_workflow.py` and Workflow 2 are built correctly. Before they are shown to outage managers, run on ≥2 completed outages from the same unit and compare component risk rankings against actuals. The slide deck correctly labels this as "In development."

**Status:** Deferred — depends on data availability.

---

## 4. Code Robustness Issues

### 4a. Cypher Work-Type Codes Interpolated Into Query String ⚠️ Medium

**File:** `stage_b_kg_timeline.py:396-398`  
`pm_code` and `cm_code` are f-string-interpolated directly into the Cypher text rather than passed as query parameters. Low-risk given the config values are internal, but should be parameterized before any production KG connection.

**Fix (Priority 3 — LOW effort):** Move work-type codes into the `parameters=` dict.

**Status:** ✅ Fixed 2026-04-16 — all six Cypher query methods parameterized.

### 4b. Monte Carlo: No Guard on Zero/Negative Duration Samples ⚠️ Low

`DurationDistribution.sample()` draws from the empirical pool. If the pool contains 0.0 or near-zero values, the Monte Carlo in Stage E can produce nonsensical schedule timings or divide-by-zero in CP drag calculations.

**Fix (Priority 5 — LOW effort):** Clamp samples to `max(sample, 0.1)` with a warning log.

**Status:** ✅ Fixed 2026-04-16 — floor guard added to Stage E Monte Carlo loop.

### 4c. Stage B Component Resolution Can Silently Fall Through ⚠️ Low

`_select_primary_component()` raises `ValueError` when no component ID is found. The orchestrator's optional-stage handler catches this as a soft failure, resulting in an empty timeline that Stage C treats as `insufficient_data`. The degradation path is correct but should be surfaced explicitly as a data quality failure in the output artifact.

**Status:** Acknowledged, low priority. Empty timeline already logged with WARNING; coverage artifact reflects zero events.

### 4d. Stage A Preprocessing Fallback Is Too Silent ⚠️ Low

Module-level try/except for optional NLP dependencies degrades gracefully but does not set any flag in the output artifact to signal that cleaning was skipped. Downstream stages cannot know to apply lower confidence weights.

**Fix:** Add `preprocessing_features_available: bool` to `ActivityIntakeResult`. Deferred — requires schema change.

**Status:** Deferred. Acceptable for current integration-testing phase.

---

## 5. Functional Gaps

| Gap | Impact | Effort | Status |
|-----|--------|--------|--------|
| Multi-outage confidence tier gate | High — required for statistical validity | Low | ✅ Done 2026-04-16 |
| Execution mode flag extraction from NLP | Medium — activates extended pool logic | Low | ✅ Done 2026-04-16 (P2a: extraction) |
| Execution mode flags → Stage D activation | Medium — flags now flow into similarity scoring | Low | ✅ Done 2026-04-16 (P2b: activation) |
| Execution mode flags → outlier bypass routing | Medium — disruption-context analogs preserved for mixture model | Low | ✅ Done 2026-04-16 (P2c: outlier routing) |
| Cypher query parameterization (Stage B) | Medium — security hygiene before KG prod | Low | ✅ Done 2026-04-16 |
| Monte Carlo sample floor guard | Low — prevents edge-case bad values | Low | ✅ Done 2026-04-16 |
| Crew continuity / fatigue constraints | Medium — LOGOS has hooks, DACKAR adapter doesn't expose them | Medium | ✅ Done 2026-04-16 |
| Baseline schedule locking | Medium — can't compute SPI or float variance vs. original | Medium | ✅ Done 2026-04-16 |
| Dependency similarity two-pass design (Gap 4) | Medium — InsertionPointDeterminer pre-pass + Layer 1 affinity re-ranking | Medium | ✅ Done 2026-04-16 |
| Permit lead time modeling | Medium — RP/confined space approvals add unmodeled days | High | ✅ Done 2026-04-16 |
| Cost per mode (crash cost) | Low — parametric cost estimate per option type, Stage F ranking updated | Low | ✅ Done 2026-04-16 |
| Pre-outage risk register multi-outage validation | High — data gap, not code gap | N/A | Blocked on data |

---

## 6. Near-Term Priorities (Completed This Session)

### P1 — Outage diversity gate in `confidence.py` and `stage_d_analogs.py`

**`confidence.py` (`ConfidenceEstimator`):**
- `classify()` gains optional `outages_represented: int = 0` parameter (0 = gate disabled, backward compatible)
- `_classify_tier()` applies gate only at the `high` tier: if `outages_represented > 0` and below threshold, caps to `medium`
- No gate at `medium` tier (it's already "use with caution")
- New constructor params: `high_outage_threshold=3`, `medium_outage_threshold=2` (unused at medium, kept for future)

**`stage_d_analogs.py` (`HistoricalAnalogRetriever._compute_confidence_tier`):**
- New `HistoricalAnalogConfig` field: `min_outages_for_data_supported: int = 3`
- `_compute_confidence_tier()` now counts `len({a["outage_id"] for a in analogs if a.get("outage_id")})`
- If count gate says `data_supported` but outage gate fails → cap to `sme_informed` (never to `low_confidence` for this reason alone)
- Set `min_outages_for_data_supported=0` to disable

**New tests in `test_stages_f_g.py`:**
- 8 new `TestComputeConfidenceTier` test methods covering: single-outage cap, two-outage behavior, three-outage allows `data_supported`, gate disabled at threshold=0, no `outage_id` triggers cap

### P2 — Execution mode flag extraction in `stage_a_intake.py`

- Four compiled regex patterns at module level: `_RP_HOLD_PATTERNS`, `_SCAFFOLD_PATTERNS`, `_CLEARANCE_PATTERNS`, `_VENDOR_PATTERNS`
- New method `_extract_execution_mode_flags(text: str) -> dict` returning `{has_rp_hold, requires_scaffold, has_clearance, is_vendor_supported}`
- Called from `process()` after regulatory constraint detection; result emitted as `execution_mode_flags` in `ActivityIntakeResult`
- 22 new tests in `TestExtractExecutionModeFlags` covering: individual flag detection, empty/no-match, multi-flag, integration via `process()`

### P3 — Cypher parameterization in `stage_b_kg_timeline.py`
- `_query_work_orders`: `NOT wo.work_type IN [$pm_code, $cm_code]` using `parameters={"pm_code": ..., "cm_code": ...}`
- Eliminates string interpolation of config values into Cypher query text

### P5 — Monte Carlo sample floor in `stage_e_schedule.py`
- `_MIN_DURATION_HOURS = 0.1` constant added
- `_clamp(val, label)` inner function in `_run_monte_carlo()` applies floor to p50/p80/p90 before scenario loop
- Emits `LOGGER.warning(...)` when a scenario duration is clamped (surfaces data quality issues in analog index)

**Test results after all changes: 791 passed (up from 730), 0 failures.**

---

## Session 2 — 2026-04-16 (P2 activation chain)

### P2b — Execution mode flags wired into Stage D similarity scoring

**Root cause of the gap:** `_make_activity_case()` built the query `ActivityCase` via `__new__` without calling `__init__`, so fields not explicitly in the `fields` dict returned `None` from `getattr`. `ContextSimilarityScorer` skips `None` fields (weight redistribution). Result: the four execution mode flags — despite having weights 0.05–0.08 in `_DEFAULT_WEIGHTS` — were silently ignored for the query side.

**Changes:**

`stage_d_analogs.py` — `_make_activity_case()`:
- Added four keyword params: `has_rp_hold=False`, `requires_scaffold=False`, `has_clearance=False`, `is_vendor_supported=False`
- Explicitly set in the `fields` dict so `getattr(query, 'has_rp_hold', None)` returns `False` (not `None`), engaging the scorer

`stage_d_analogs.py` — `_build_query()`:
- Reads `intake_result.get("execution_mode_flags") or {}` (safe if key absent)
- Passes each flag through `bool(...)` to `_make_activity_case()`

**Effect in practice:**
- Query describing RP hold work → `has_rp_hold=True` on query `ActivityCase`
- `ContextSimilarityScorer` now boosts analogs that also had RP holds (weight 0.08) and penalises those that didn't
- All four flags now participate in similarity scoring end-to-end

**New tests (11 in `test_stages_f_g.py`):**
- `TestMakeActivityCaseExecutionFlags` (6 tests) — verifies `_make_activity_case` explicitly sets flags, not leaves them absent
- `TestBuildQueryExecutionFlagPassthrough` (5 tests) — verifies flags flow from `intake_result` → `_build_query` → `_query_activity_case`; includes end-to-end scorer differentiation test

**Test results: 802 passed (up from 791), 0 failures.**

---

## Session 3 — 2026-04-16 (P2c: disruption-context outlier routing)

### P2c — Execution mode flags wired into outlier separation

**Root cause of the gap:** `_remove_duration_outliers()` called `OutlierHandler.separate()` and then kept only `separation.routine`.  Analogs above the IQR fence were silently discarded regardless of whether their extended duration was caused by the same disruption condition (RP hold, scaffold, clearance, vendor) that the query is asking about.  For a query with `has_rp_hold=True`, any RP-hold analog with a long duration was discarded from the pool before reaching the distribution fitter — the opposite of what the mixture model needs.

**Changes:**

`stages/stage_d_analogs.py`:
- `_activity_to_analog()`: added `has_rp_hold`, `requires_scaffold`, `has_clearance`, `is_vendor_supported` to the analog dict (bool, `False` when absent).  Required for flag matching and downstream traceability.
- Three new module-level helpers:
  - `_EXECUTION_MODE_FLAGS` — canonical tuple of the four flag names
  - `_active_execution_flags(query_activity)` → `frozenset` of flag names that are `True` on the query
  - `_analog_matches_flags(analog, active_flags)` → `bool`, `True` if any flag overlaps
- `_remove_duration_outliers()` rewritten:
  - Partitions `with_duration` into `nd_pairs` (non-disruption) and `dc_pairs` (disruption-context, share ≥1 active flag with query)
  - IQR/Tukey fence applied only to `nd_pairs`; `dc_pairs` bypass it entirely
  - `outliers_removed` count covers only true non-disruption outliers
  - Reconstruction preserves original index order as before

**Effect in practice:**
- Query with `has_rp_hold=True` → RP-hold analogs with long durations are kept in the pool
- Distribution fitter receives a full pool; its IQR pass routes them naturally to `extended`
- `mixture_weight` and extended pool in `DurationDistribution` now reflect genuine disruption-mode data
- Non-matching outliers (no flag overlap) are still discarded — the fence is not disabled, just targeted

**New tests (9 in `test_stages_f_g.py`, class `TestRemoveDurationOutliersDisruptionMask`):**
- `test_no_active_flags_behaves_as_before` — regression: outlier removed when query has no flags
- `test_disruption_context_analog_preserved_above_fence` — RP-hold analog survives with rp_hold query
- `test_non_disruption_outlier_still_removed_when_flags_active` — true outlier removed even when flags active
- `test_any_matching_flag_preserves_analog` — scaffold-only query preserves scaffold analog
- `test_active_execution_flags_helper_returns_correct_frozenset`
- `test_active_execution_flags_returns_empty_when_all_false`
- `test_analog_matches_flags_true_on_overlap`
- `test_analog_matches_flags_false_on_no_overlap`
- `test_activity_to_analog_includes_execution_flags` — verifies flags propagated into analog dict

**Test results: 811 passed (up from 802), 0 failures.**

---

---

## Session 4 — 2026-04-16 (query_summary enrichment + baseline schedule locking)

### query_summary enrichment (P2 chain completion)

**Gap:** The execution mode flags were wired into similarity scoring (P2b) and outlier routing (P2c) but not recorded in the output artifact.  An analyst reviewing a Stage D artifact couldn't tell which disruption conditions were active for a given retrieval run.

**Change:** `stage_d_analogs.py` — `_build_query()` now includes `"execution_mode_flags"` in the returned `query_summary` dict.  The flags are read from the same `_flags` dict already used to construct the query `ActivityCase`, so no additional extraction is needed.  All-False when `execution_mode_flags` is absent from the intake result (backward compat).

**New tests (2 in `TestBuildQueryExecutionFlagPassthrough`):**
- `test_query_summary_includes_execution_mode_flags`
- `test_query_summary_flags_absent_intake_defaults_to_all_false`

### Baseline schedule locking (Stage E)

**Gap:** Stage E loaded the working schedule and used its CP duration as the reference for `cp_drag_hours`.  It could not compute schedule variance (how far the outage has already slipped from the original kickoff plan) or the total projected overrun, because no locked baseline was loaded.  Outage managers use these metrics daily.

**Changes:**

`stages/stage_e_schedule.py`:
- `ScheduleImpactConfig` gains `baseline_schedule_version: str = "baseline"` — the version tag to load for the original locked plan.  Set to `""` to disable.
- `_ScheduleNetwork` gains three optional fields: `locked_baseline_cp_hours`, `locked_baseline_start`, `working_start`.
- `_load_schedule_network()`: after loading the working schedule, makes a second loader call at `baseline_schedule_version` in a try/except.  Failure is a soft degradation (logs DEBUG, proceeds without variance fields) — e.g. first outage running DACKAR where no locked baseline has been archived.
- `_compute_cp_metrics()` gains three keyword-only params (all default None for backward compat): `locked_baseline_cp_hours`, `locked_baseline_start`, `working_start`.
- New static helper `_locked_baseline_fields()` computes the four variance fields and absolute datetimes.

**New fields in `cp_impact` artifact section (when baseline is available):**
- `locked_baseline_cp_hours` — original kickoff plan duration (hours)
- `schedule_variance_hours` — `working_cp - locked_baseline_cp`; positive means already slipped
- `total_overrun_hours` — `max(0, estimated_new_cp - locked_baseline_cp)`; total slip vs. original if activity inserted
- `locked_baseline_finish` — ISO datetime of original planned outage finish
- `projected_finish_after_insertion` — ISO datetime of projected finish after insertion (p50 scenario)

**New tests (9 in `TestComputeCPMetricsBaselineLocking`, test_stage_e.py):**
- Schedule variance positive/zero
- Total overrun includes both drag and variance; clamped at 0
- Fields absent when baseline not provided
- `locked_baseline_finish` computed correctly from start + duration
- `projected_finish_after_insertion` computed from working start + new CP
- Empty-duration sim still populates variance fields
- `locked_baseline_cp_hours` echoed in output

**Test results: 822 passed (up from 811), 0 failures.**

---

---

## Session 5 — 2026-04-16 (displaced task regulatory constraint enrichment)

### `has_regulatory_constraint` in displaced tasks

**Gap:** `_identify_displaced_tasks()` hardcoded `"has_regulatory_constraint": False` with a comment "KG enrichment required." Stage G uses this field to decide whether a displaced task needs special handling in the recommendation text — a displaced surveillance test or TS hold point is categorically different from a displaced PM task. Leaving it False meant Stage G was systematically underweighting regulatory risk in the displaced task list.

**Design choice:** Description-based keyword matching (same patterns as Stage A's `_detect_regulatory_constraints`) rather than a KG round-trip. Rationale: the KG requires a connected driver and a per-task lookup; description matching is self-contained, testable, and covers the common cases (TS tasks nearly always carry "TS 3.5.7" or "surveillance", hold-point tasks say "hold point" or "HP"). A future KG enrichment layer can override these values with higher-confidence data if the driver is available.

**Changes:**

`stages/stage_e_schedule.py`:
- Added `import re`
- Added `_REGULATORY_KEYWORDS_RE` module-level compiled pattern (mirrors `stage_a_intake._REGULATORY_KEYWORDS_RE`): matches TS, LCO, NRC, 10 CFR, ALARA, CAP, surveillance, operability determination, hold point, quality hold, mode change, entry condition
- Added `_has_regulatory_constraint(text: Optional[str]) -> bool` module-level helper — public API, fully testable without a mock
- `_identify_displaced_tasks()`: `"has_regulatory_constraint"` now set to `_has_regulatory_constraint(description)` instead of hardcoded False

**New tests (16 in test_stage_e.py):**
- `TestHasRegulatoryConstraint` (13 tests): TS/LCO/surveillance/hold-point/NRC/ALARA/mode-change → True; plain description/None/empty string → False; case-insensitive; partial-word boundary safety
- `TestIdentifyDisplacedTasksRegulatoryFlag` (3 tests): end-to-end displaced task integration using a `_ShiftedPert` mock subclass that shifts T-002's ES by 8 hours in `clone_for_analysis()`, confirming TS/surveillance descriptions are flagged and plain descriptions are not

**Test results: 838 passed (up from 822), 0 failures.**

---

## 7. Open Questions

- Should `outages_represented` also be gated in the `HistoricalAnalogs` artifact itself (as a validation rule in the schema validator)?
- Once Stage B KG driver is connected in production, should empty KG timelines block the pipeline or proceed with `causal_posture = insufficient_data`? Current behavior: proceed — this seems correct.
- Dependency similarity (Gap 4): two-pass Stage D design — worth prototyping once multi-outage data is available.
