# Deep Review and Test Strategy — Unexpected Activity Pipeline
**Date:** 2026-04-16  
**Scope:** Stages A–G, orchestrator, outage_uncertainty modules  
**Reviewer:** Claude (claude-sonnet-4-6)  
**Source files:** All stage implementations, protocols, schemas, tests, and documentation as of 2026-04-16 (838 tests passing)

---

## Executive Summary

- **The pipeline architecture is sound.** Protocol-based injection, artifact passthrough for replay, run manifests, schema validation, and evidence traceability are all correctly designed for an auditable nuclear decision tool. The 838-test suite with zero failures is a strong signal of implementation quality.
- **Top concern: scoring model has a structural bias.** Stage F's composite risk formula conflates "risk of taking action" with "risk of not taking action" in a way that can produce systematically wrong rankings when the regulatory urgency is high. The `causal_urgency` dimension uses the same urgency value for both action and non-action options, but with opposite sign — meaning high urgency simultaneously increases the cost of not acting AND decreases the cost of acting, which can produce unrealistically low risk scores for `insert_now` and unrealistically high scores for `defer` even before the regulatory block fires.
- **Top gap: no TS action level deadline clock.** Outage managers managing an active LCO or TS action level need to know how many hours remain before the action level expires. This is the single most time-critical piece of information in an in-outage decision, and no stage surfaces it.
- **Top robustness risk: Stage G `MONITOR` status is nearly unreachable.** The condition requires `dist_tier == "low_confidence"` AND `analog_count == 0` AND `criticality_label == "non_critical"` — but `criticality_label` is read from `insertion_options["schedule_summary"]`, a field that does not exist in the artifact schema. When the field is absent the default is `"non_critical"`, which means the MONITOR condition becomes reachable only if Stage F happens to populate `schedule_summary`. No test exercises this path end-to-end.
- **The multi-outage diversity gate is implemented correctly** in both `ConfidenceEstimator` and `HistoricalAnalogRetriever`. The execution mode flag chain (Stage A → D → E) is fully wired. Cypher parameterization and the MC floor guard are in place. The baseline schedule locking and displaced-task regulatory enrichment are complete.

---

## 1. Requirements Analysis

### 1.1 What Outage Managers Need

Derived from `critical_analysis.md`, `outage_analytics_overview.md`, `unexpected_activities_notes.md`, `test_case_spec.md`, and the domain knowledge in `outage_planner_gap_analysis.md`.

When an unexpected activity is found during an outage, a manager needs to answer seven questions in roughly this order:

1. **Legality:** Can I even defer or reduce scope, or does a TS/LCO/NRC commitment prohibit it — and if so, how much time do I have before action level expires?
2. **Identity:** What exactly is this activity — component, system, failure mode, discipline? Is it a truly new finding or an escalation of existing planned work?
3. **History:** Has this happened before on this component? What were the outcomes? How long did it take?
4. **Duration:** How long will this take in the best and worst cases? What is the p80 — the number I should protect against?
5. **Schedule impact:** What does this do to my critical path and my finish date? Which downstream tasks get displaced?
6. **Options:** What can I actually do — insert now, defer, partial scope, parallel execution, escalate? What does each cost in time and money?
7. **Decision with traceability:** A specific, defensible recommendation with sources, so I can brief management and document why I made the call.

Beyond those seven questions, the following operational needs recur in the literature:

- **Crew and resource availability** at the proposed execution window
- **Permit and approval lead times** before work can start
- **Communication chain guidance** — who must be notified (shift supervisor, licensing, NRC)
- **Learning loop** — capturing the actual outcome (actual duration, what happened) so the next cycle benefits

### 1.2 Requirements Coverage Matrix

| Requirement | Covered By Stage | Status | Gap |
|---|---|---|---|
| Regulatory legality check (can I defer?) | A (detection) + F (clearance check) | Covered | No TS action level deadline clock |
| Active LCO clock / time to action level expiry | A (structured fields: `active_lco`, `lco_number`) | Partial | `active_lco` is boolean only; hours remaining not computed anywhere |
| Emergence type classification | A (`emergence_type`, 4-rule chain) | Covered | — |
| Component identity and NER | A (NER pipeline, 3-layer) | Covered | ~~Fallback NLP degrades silently (Issue N1)~~ ✅ Fixed 2026-04-16 |
| Active LCO clock / time to action level expiry | A (`_compute_lco_clock`) | ~~Partial~~ ✅ Fixed 2026-04-16 | `lco_clock_status` + `hours_to_action_level` + `lco_action_level_expires_at` in process() output; LCO flags + warning prefix in Stage G |
| Component history and recurrence | B (KG timeline), C (Allen chain) | Covered | KG driver absent → empty timeline, no hard failure surfaced to user |
| Frequency and trend of prior failures | B (`recurrence_indicators`) | Covered | PM compliance assumes fixed 180-day interval when not in KG |
| Duration estimate (p50/p80/p90) | D (analog retrieval + distribution fitting) | Covered | Single-outage analog pool → tier cap at `sme_informed` (by design) |
| Mixture model for disrupted executions | D (execution mode flags → outlier bypass) | Covered after P2c fix | — |
| Critical path drag and float analysis | E (Monte Carlo, LOGOS Pert) | Covered | Schedule loader/graph builder absent → RuntimeError, no soft degradation |
| Permit lead time | E (permit_lead_time block) | Covered | RP hold + clearance + scaffold modeled; vendor mobilization not included |
| Displaced tasks with regulatory flag | E (`_identify_displaced_tasks` + `_has_regulatory_constraint`) | Covered after Session 5 | Keyword-based only, no KG round-trip |
| Baseline schedule variance (already slipped?) | E (`locked_baseline_fields`) | Covered after Session 4 | Absent when no baseline version archived — field is null, not warned |
| Crew resource availability | E (`crew_continuity`, `_check_resource_conflicts`) | Covered | No per-shift named-crew tracking (see Gap 5 in `outage_planner_gap_analysis.md`) |
| Options generation with cost model | F (7 option types + parametric cost) | Covered | Pre-outage staging upgrade has a latent bug (Issue N2) |
| Regulatory block enforcement on options | F (`_check_regulatory_clearance`) | Covered | ~~`escalate_to_management` always marked regulatory_cleared=True even for NRC commitment activities — correct by design but should be explicit (Issue N3)~~ ✅ Fixed 2026-04-16 |
| Risk scoring and ranking | F (5-dimension composite) | Covered | ~~Scoring formula has structural bias (Issue N4)~~ ✅ Fixed 2026-04-16 |
| Decision status (PROCEED/DEFER/ESCALATE/MONITOR) | G | Partial | MONITOR condition reads non-existent field (Issue N5) |
| Evidence chain traceability | G (4-source chain) | Covered | ~~max_evidence_items cap can truncate regulatory evidence (Issue N6)~~ ✅ Fixed 2026-04-16 |
| Analyst review flags | G (8 flags) | Covered | — |
| Reject-with-reason feedback loop | G (`analyst_review.reviewer_decision`) | ✅ Fixed 2026-04-16 | `CompletionFeedbackWorkflow` routes on `reviewer_decision`; rejected recs stored with `RECOMMENDATION_REJECTED:` prefix |
| Actual duration feedback to analog index | Orchestrator `record_completion()` | ✅ Fixed 2026-04-16 | ~~Issue N7~~ — `feedback_writer` wired; `CompletionFeedbackWorkflow.ingest()` closes the loop (M5) |
| Communication chain (who to notify) | Not implemented | Gap | No stage produces notification chain guidance |
| TS action level deadline tracking | Not implemented | Gap | Nuclear critical — see Section 1.3 |

### 1.3 Missing Requirements

The following requirements are present in the domain literature but absent from both the documentation and the implementation:

**M1 — TS Action Level Countdown (Critical)**  
When an LCO entry condition is met, Technical Specifications typically give the licensee a defined action level window (e.g., 72 hours for Mode 3, 8 hours for Mode 1). The pipeline captures `active_lco` as a boolean and `lco_number` as a string, but nowhere computes or surfaces hours-remaining before the action level deadline. This is arguably the most urgent piece of information a manager needs after identifying a TS-applicable activity. Without it, the PROCEED recommendation carries an implied "when convenient" framing that is operationally wrong for LCO-driven activities.

**M2 — Notification and Communication Chain**  
Nuclear plants have regulatory notification requirements: NRC must be notified within 1, 4, or 24 hours depending on the event class. Maintenance Rule function status, shift supervisor logs, and licensing calls are all time-ordered. The pipeline produces no guidance on who to notify, in what order, or on what timeline. This is a non-optional part of outage manager decision support.

**M3 — Safety Function Conflict Detection**  
The pipeline checks regulatory constraint keywords but does not reason about whether the activity affects a safety function's operability while the plant is in a mode where that function is required. For example, working on Train A of the RHR while Train B is already in maintenance (single failure vulnerability) is a different risk level than either train alone. The `SystemStatePool` in LOGOS handles mutual exclusion, but Stage F options never consult current system state.

**M4 — ALARA Dose Rate Estimation**  
`alara_requirement` is detected as a regulatory driver and sets `defer_prohibited=False` (correctly — ALARA doesn't prohibit deferral, but it affects option ranking). However, no stage estimates or surfaces the expected cumulative dose for the activity at the insertion point. ALARA optimization is explicitly mentioned as a first-class constraint in `outage_analytics_overview.md` (Slide 3), and the absence of a dose estimate means the outage manager cannot evaluate whether the recommended insertion window is radiation-safe.

~~**M5 — Post-Completion Feedback Ingestion**~~  
✅ Fixed 2026-04-16 — `workflows/completion_feedback_workflow.py` implements `CompletionFeedbackWorkflow.ingest(recommendation, actual_duration_hours, ...)`. Routes on `analyst_review.reviewer_decision`: accepted/modified/None → `orchestrator.record_completion()` called normally; rejected → called with `RECOMMENDATION_REJECTED: {reviewer_notes}` prefix in `outcome_notes` for negative learning; pending → skip with `IngestionReceipt(skipped=True)`. 30 tests in `TestIngestionRouting`, `TestRejectionPath`, `TestGuardConditions`, `TestFieldForwarding`, `TestReceiptFields`, `TestNoFeedbackWriterWired`.

---

## 2. Stage-by-Stage Logic Review

### 2.1 Stage A — Activity Intake

**What it does:** Text cleaning, abbreviation expansion, 3-layer NER, reference resolution, 4-rule emergence type classification, 12-pattern regulatory constraint detection, execution mode flag extraction, data quality score computation.

**Algorithm correctness:**

The data quality formula at line ~225 of the pseudocode is:
```
dq = 0.35 × field_completeness + 0.25 × ner_yield + 0.25 × (1.0 − abbr_rate) + 0.15 × source_confidence
```
Weights sum to 1.0. Formula is correct.

The emergence type 4-rule priority chain is correct in logic but has one ambiguity: Rule 2 fires for any regulatory keyword in the text OR any structured TS/NRC/LCO field. The structured field path sets confidence 0.90; the text-only path sets confidence 0.85. If both fire, the structured-field confidence (0.90) takes precedence. This is the intended behavior but is not documented in the rule engine comments.

**Edge cases:**

- `raw_description = ""` produces an empty entity list and `ner_yield = 0`. The fallback emergence type is `truly_unplanned` at confidence 0.45 (insufficient signal). Correct behavior, but downstream stages will see zero resolved components and Stage B will raise `ValueError`. The orchestrator catches this as a soft failure via `optional_failures` — verify this path is exercised in `test_orchestrator_e2e.py` (it is not: the e2e tests always inject a non-empty description).
- `known_component_id` is passed through to `resolved_component_ids` in `_resolve_references`. If the caller provides a component ID that does not exist in the KG, Stage B's query returns zero events without raising. The KG absence is invisible to Stage C and D. No test verifies this case.
- Execution mode flag `_VENDOR_PATTERNS` matches "technical representative" — this is a common phrase in plant activity descriptions that does not always indicate OEM vendor involvement. May produce false positive `is_vendor_supported` flags, biasing similarity scoring toward longer-duration analogs.

**Schema issues:**

The `emergent_activity.json` schema requires `"source_system"` with enum `["P6", "CMMS", "CAP", "manual", "other"]`. Stage A's `_SOURCE_CONFIDENCE` map uses lowercase keys `maximo`, `primavera`, `p6`, `sap`, `manual`. The schema enum does not include `maximo`, `primavera`, or `sap`, yet those are the most common production values. Either the schema or the source confidence map is out of sync. **Issue N8 (Medium).**

### 2.2 Stage B — KG Timeline Builder

**What it does:** Queries Neo4j for component events (CRs, WOs, PM, CM, abnormal events, inspections) within a 5-year window. Deduplicates by `source_doc_id`. Computes recurrence indicators and PM compliance.

**Algorithm correctness:**

PM compliance calculation uses `default_pm_interval_days = 180` when no interval is stored on the KG component node. 180 days is a reasonable default for many components, but for quarterly surveillances (91 days) or annual PMs (365 days), this produces incorrect overdue flags. If the component has no PM history in the window, `pm_compliance_status` defaults to `unknown` — correct.

The trend computation (first-half vs. second-half event frequency, threshold 1.5x) is a coarse heuristic. It can produce `stable` for genuinely increasing trends when the sample size is small. With fewer than 3 events it returns `insufficient_data` — correct, but the threshold of 3 is low.

The `max_events = 100` truncation keeps the **most recent** events (tail of the sorted list). For a 5-year window this is correct behavior for causal chain analysis (recent events are more causally relevant). However, if there were a burst of events in the last 6 months (100+ events from a recurring failure), all pre-burst history would be dropped. No test exercises this edge case.

**Edge cases:**

- `detection_timestamp = None`: `_window_start_iso` returns None, so `window_start` is None. The Cypher query uses `WHERE cr.initiated_date >= $window_start AND cr.initiated_date <= $detection_ts`. With `window_start = None`, the Neo4j query behavior depends on how py2neo handles null parameters — it may return all events or raise. **Issue N9 (Medium):** Add an explicit guard in `build()` that falls back to `utcnow() - 5 years` when `detection_timestamp` is absent, logging a WARNING.
- `kg_driver = None`: The `if kg_driver available:` block is skipped, producing an empty timeline. The orchestrator registers this as an optional failure. Stage C then produces `causal_posture = "insufficient_data"` with zero chain links. Stage D runs independently (does not depend on B). Stage F receives an empty temporal chain and proceeds to option generation. This is the correct graceful degradation path.

**Cypher parameterization:**

Fixed in Session 3 (code review Session P3). All six query methods use `parameters=` dict. No residual Cypher injection risk.

### 2.3 Stage C — Temporal Chain Scorer

**What it does:** Allen interval algebra over the component event timeline. Produces per-link relation scores, confidence scores, and a chain summary with causal_posture.

**Algorithm correctness:**

The confidence formula:
```
confidence = 0.55 × data_quality_score + 0.30 × lag_plausibility + 0.15 × relation_score
```
Weights sum to 1.0. Correct.

The lag plausibility decay:
```
lag_plausibility = max(0.1, 1.0 - (onset_lag_hours - 24.0) / 720.0)
```
At 744 hours (31 days), plausibility = max(0.1, 1.0 - 720/720) = 0.1. At 360 hours (15 days), plausibility ≈ 0.53. This decay models the intuition that a prior event from years ago is a weaker causal candidate. The 720-hour (30-day) normalization constant is a tunable but currently undocumented design choice.

**Issue N10 (Low):** The lag plausibility formula uses onset lag relative to the event START. For a CR that was open for months (e.g., a slow degradation observed 6 months before the emergent activity), the lag from event START is long even though the event was ONGOING at activity onset — which is exactly the OVERLAPS relation, scored at 0.90. In that case the confidence formula produces:
```
0.55 × dq + 0.30 × 0.1 + 0.15 × 0.90 = 0.55 × dq + 0.165
```
versus an ideal score of `0.55 × dq + 0.30 × 1.0 + 0.135`. For long-running OVERLAPS relations the lag plausibility artificially depresses confidence. Fix: for OVERLAPS and CONTAINS relations, set lag plausibility to 1.0 unconditionally (the temporal relationship already encodes precedence).

**Edge cases:**

- Both `detection_timestamp` and `actual_start` are None: `_parse_activity_interval` returns `(None, None), is_point=True`. For each event, `act_start = None`, so `_allen_relation` returns `_UNKNOWN` (relation score 0.0). Chain summary produces `causal_posture = "insufficient_data"`. Correct behavior.
- `planned_duration_hours = 0`: `timedelta(hours=0)` produces `end_iso = start_iso` (zero-width interval). `is_point = False` (because `end_iso` is not None). The activity is treated as a zero-duration interval, which will make any event that is simultaneous appear as `DURING` rather than `OVERLAPS`. This is an edge case that may occur for instantaneous events (mode changes). Low impact.

### 2.4 Stage D — Historical Analog Retriever

**What it does:** Builds query ActivityCase, retrieves candidates, scores and filters by similarity threshold, removes outliers (with execution mode flag bypass), fits duration distribution, computes confidence tier.

**Algorithm correctness:**

The two-pass outlier design is correct:
- Pass 1: IQR fence removes noise from the non-disruption sub-pool only; disruption-context analogs bypass
- Pass 2: `OutlierHandler.separate()` classifies disruption-context analogs into the `extended` pool for the mixture model

The confidence tier two-gate logic (count gate + outage diversity gate) is correct per the P1 fix. The implementation caps at `sme_informed` when the count gate passes but the outage diversity gate fails, rather than degrading all the way to `low_confidence`. This is the intended conservative-but-not-punitive behavior.

**Issue N11 (Medium) ~~— `_compute_confidence_tier` does not pass `outages_represented` to `ConfidenceEstimator.classify()`~~** ✅ **Fixed 2026-04-16** (test coverage only — code was already correct)

The code already passes `outages_represented` to `classify()` and `_activity_to_analog` already includes `outage_id`. The gap was absence of tests. 5 regression tests added in `TestComputeConfidenceTierWithEstimator`: CE is called when injected; `outages_represented` is correctly counted from analog `outage_id` fields; tier mapping (`high→data_supported`, `medium→sme_informed`, `low→low_confidence`); fallback to count-based logic when `classify()` raises; `outage_id` propagation through `_activity_to_analog`.

**Issue N12 (Low) — `similarity_threshold = 0.60` is a fixed constant:**  
The threshold is a constructor config with a reasonable default. However, when the retrieval index is absent (stub mode), all candidates are returned with similarity score 1.0 (stub always returns fixed scores). The threshold is never violated in test mode. First production deployment will see a real rejection rate, potentially leaving zero analogs for components with no close matches.

### 2.5 Stage E — Schedule Impact Assessor

**What it does:** Loads LOGOS schedule, determines insertion point, computes float analysis, runs Monte Carlo via LOGOS Pert, identifies displaced tasks, checks resource conflicts, assesses crew continuity, computes permit lead times, optionally loads locked baseline.

**Algorithm correctness:**

The Monte Carlo loop adds `permit_lead_hours` to every sampled activity duration before passing it to LOGOS Pert. This correctly propagates the permit overhead through CP drag computation. The `_MIN_DURATION_HOURS = 0.1` floor guard (P5 fix) prevents zero-duration samples from poisoning the CP arithmetic.

The `_compute_cp_metrics` method derives CP drag as `median_project_duration - schedule_network.baseline_cp_hours`. This is the correct definition when baseline is the working schedule. With the locked baseline enabled, `total_overrun_hours = max(0, estimated_new_cp - locked_baseline_cp)`. Correct.

**Issue N13 (High) ~~— Stage E raises `RuntimeError` when schedule_loader or schedule_graph_builder is None~~** ✅ **Fixed 2026-04-16**  
~~The current behavior:~~
```python
if self.schedule_loader is None or self.schedule_graph_builder is None:
    raise RuntimeError(...)
```
~~The orchestrator only handles this as an optional failure if the exception is caught. Looking at `_stage_e_schedule` in the orchestrator, it passes the result directly to the next stage without a try/except — the RuntimeError will propagate and kill the pipeline rather than producing a partial assessment. Stage E should produce a stub artifact with `cp_impact.cp_drag_hours = null` and a note explaining the absence of schedule data, rather than raising. This matches how Stage B handles the absent KG driver.~~

**Implementation:** `_stage_e_schedule()` in the orchestrator now catches `RuntimeError` from `schedule_impact_assessor.assess()`, logs a WARNING, appends to `optional_failures`, and returns `_stub_schedule_impact_artifact()` — a fully-schema-compliant artifact with null CP metrics identified by `schedule_version_id == "STUB::no_schedule"` (Y8 fix: `schedule_loader_unavailable` field was removed as it is not in the schema). Stages F and G continue normally.

**Issue N14 (Medium) — `schedule_summary` is not produced by Stage F:**  
Stage G's `_determine_decision_status` reads `insertion_options.get("schedule_summary")` to get `criticality_label` for the MONITOR condition. Stage F produces `ranking_summary` (a dict with `options_generated`, `feasible_count`, etc.) but does NOT produce a `schedule_summary` key. The MONITOR branch uses `criticality_label` from the non-existent key, defaults to `"non_critical"`, and so the MONITOR condition depends entirely on `dist_tier == "low_confidence" and analog_count == 0`. This is technically reachable but the `schedule_summary` field name mismatch is misleading and could cause a future developer to add a `schedule_summary` key to Stage F (which would change behavior). Either document that `schedule_summary` is intentionally absent and the default `"non_critical"` is the intended fallback, or add a `schedule_summary` to Stage F's output.

### 2.6 Stage F — Insertion Option Generator

**What it does:** Generates up to 7 option types, checks regulatory clearance, computes parametric costs, scores each option on 5 dimensions, ranks, selects recommended.

**Algorithm correctness:**

Weights sum: 0.35 + 0.15 + 0.20 + 0.20 + 0.10 = 1.00. Correct. *(Updated 2026-04-16: confidence 0.25→0.15, urgency 0.10→0.20, N4 fix)*

**Issue N4 (High) ~~— Scoring formula double-counts urgency for the "best" option~~** ✅ **Fixed 2026-04-16**

~~For an `insert_now` option with high causal urgency (`urgency = 0.80`, e.g., `causal_posture = "supported"`):~~
~~The urgency dimension (weight 0.10) was the ONLY dimension that distinguishes the schedule-neutral `defer` option (cp_impact=0) from the `insert_now` option. This caused DEFER to score lower than INSERT_NOW for sub-48h CP drag activities with high urgency.~~

**Implementation:** `_W_CONFIDENCE` reduced from 0.25 → 0.15; `_W_URGENCY` increased from 0.10 → 0.20 (weights still sum to 1.00). With the new weights, for a `supported` posture (`urgency=0.80`) and 40 h drag on a 480 h baseline, `insert_now` (risk ≈ 0.195) scores lower than `defer` (risk ≈ 0.235). Added regression test `test_high_urgency_insert_now_beats_defer_at_sub48h_drag` in `TestCausalUrgencyDirection`.

**Issue N2 (Medium) ~~— Pre-outage staging upgrade logic has a timezone comparison bug~~** ✅ **Fixed 2026-04-16**

~~`emergent_activity.get("outage_start")` was not in the schema, so the pre-outage staging upgrade never fired and `pre_outage_staging` was unreachable.~~

**Implementation:** `outage_start` added to `emergent_activity.json` as an optional `date-time` field with description explaining its role in the staging upgrade. 4 regression tests added in `TestGenerateContingencyBuffer`: upgrade fires when `detection_ts < outage_start`; no upgrade when detection after outage start; no upgrade when `actual_start` is set; rationale mentions staging.

**Issue N3 (Low) ~~— escalate_to_management is always regulatory_cleared=True~~** ✅ **Fixed 2026-04-16**

~~This is correct by design: escalation is always a legal option. However, when the activity has an NRC commitment deadline, the escalation option's rationale does not include the deadline. The outage manager escalating to management needs to know that the NRC notification clock has already started.~~

**Fix:** `_generate_escalate()` now accepts `intake_result` and appends a TS/LCO deadline note to the rationale when `has_regulatory_constraint=True` or `active_lco=True`. 5 tests in `TestN3EscalateRegulatoryDeadlineNote`.

### 2.7 Stage G — Recommendation Synthesizer

**What it does:** Selects primary option, determines decision status, builds executive summary, assembles evidence chain, computes attention flags, builds history and schedule summaries, sets analyst review flag.

**Algorithm correctness:**

**Issue N5 (Medium) ~~— MONITOR condition reads missing field~~** ✅ **Fixed 2026-04-16**

~~Stage F does not produce a `schedule_summary` key, so `criticality_label` was always `"non_critical"` (the default), making MONITOR reachable whenever `dist_tier == "low_confidence" and analog_count == 0` — including for activities with critical-path impact.~~

**Implementation:** `_determine_decision_status` gains a `schedule_impact_assessment: Optional[JsonDict] = None` parameter. `criticality_label` is now read from `schedule_impact_assessment["float_analysis"]["criticality_label"]` — the same source as `_build_schedule_summary`. The call site in `synthesize()` passes `schedule_impact_assessment`. Three regression tests added: MONITOR blocked when critical, MONITOR allowed when non-critical, None schedule defaults to non_critical.

**Issue N6 (Low) — evidence chain cap can truncate regulatory evidence:**

`config.max_evidence_items = 10` (default). The evidence chain populates: temporal chain links first (up to 3), then historical analogs (up to 3), then schedule analysis (1), then condition reports (1). Regulatory constraint information is surfaced in `regulatory_flags` at the artifact level, not in the evidence chain itself. If all 8 evidence slots are filled and the regulatory context is the most important evidence item, it is not represented in the evidence chain. The regulatory flag is still present in `regulatory_flags` but may be overlooked by a user reading only the evidence chain. Suggest adding a `regulatory_constraint_evidence` entry as entry[0] in the evidence chain whenever `has_regulatory_constraint` is True.

**Evidence chain quality issue:**

For the schedule_analysis evidence entry, `supports` is computed as:
```python
supports = criticality != "critical" or cp_drag < 8.0
```
This means the schedule analysis "supports" the recommendation when the activity is non-critical OR when CP drag is less than 8 hours. For a critical-path activity with exactly 8.0 h drag, `cp_drag < 8.0` is False, so `supports = False`. At 7.99 h drag, `supports = True`. The 8.0-hour threshold is arbitrary and produces a cliff edge. This affects the narrative direction in the evidence chain (the "supports" field influences how the evidence is framed in the executive summary).

**History summary outage IDs (§8a traceability):**

Stage G's `_build_history_summary` correctly extracts distinct `outage_id` values from the analogs list. This satisfies the §8a requirement from `critical_analysis.md`. Confirmed correct.

### 2.8 Orchestrator

**Thread safety (H6 fix):**

The orchestrator builds a fresh `run_context` with a new UUID per `run()` invocation. Stage D passes `qac` explicitly through the call chain rather than storing it as instance state. No shared mutable state between concurrent calls. Thread-safe.

**Schedule context (two-pass design):**

The pre-pass `InsertionPointDeterminer` is optional and falls back gracefully to `None` when not injected. The schedule is loaded twice when the determiner is NOT injected (once in the pre-pass, once in Stage E) — except that when the determiner is absent, there is no pre-pass at all. The double-load issue only occurs when the determiner IS injected: the determiner loads the schedule during the pre-pass, then Stage E loads it again for Monte Carlo. This is documented in the orchestrator source. Fix: cache the `OutageData` in `ScheduleContext` and have Stage E read from the cache.

**Optional failure handling:**

Stage B failure is handled via `optional_failures`. Stage E failure raises `RuntimeError` and is NOT caught. **This is Issue N13 again** — confirm whether the orchestrator's `_stage_e_schedule` has a try/except around the `assess()` call.

Looking at `outage_activity_orchestrator.py` lines 226–234, `_stage_e_schedule` calls `self.schedule_impact_assessor.assess(...)` with no try/except. The `RuntimeError` raised by Stage E when schedule_loader is absent propagates through the orchestrator and kills the pipeline. The e2e tests work because they inject a mock assessor that returns a pre-built dict. First production run without LOGOS will fail at Stage E.

---

## 3. Cross-Stage Data Flow Analysis

### Critical path: intake → recommendation

| Field | Produced By | Consumed By | Risk if Absent |
|---|---|---|---|
| `intake_result.resolved_component_ids[0]` | Stage A | Stage B (KG query) | Stage B raises ValueError → optional failure → empty timeline |
| `intake_result.regulatory_drivers` | Stage A | Stage F (clearance check), Stage G (regulatory_flags) | No regulatory blocking; compliance risk |
| `intake_result.execution_mode_flags` | Stage A | Stage D (similarity, outlier routing), Stage E (permit lead time) | Execution mode flags all default False; mixture model degrades |
| `intake_result.has_regulatory_constraint` | Stage A | Stage G (attention flag, executive summary) | Regulatory warning omitted from conclusion |
| `intake_result.unknown_abbreviation_rate` | Stage A | Stage G (HIGH_ABBR_RATE flag) | Flag never raised regardless of NLP quality |
| `component_event_timeline.events` | Stage B | Stage C (Allen algebra), Stage G (history summary, evidence chain) | Empty chain → `causal_posture = insufficient_data` |
| `temporal_event_chain.summary.causal_posture` | Stage C | Stage F (`_POSTURE_TO_URGENCY` lookup), Stage G (evidence chain) | Defaults to `insufficient_data` → urgency 0.40 (neutral) |
| `temporal_event_chain.summary.strongest_link_id` | Stage C | Stage G (evidence chain item 1) | Evidence chain omits temporal chain evidence |
| `historical_analogs.duration_distribution` | Stage D | Stage E (MC sampling), Stage F (option cost/confidence) | MC uses empty distribution → stub p50/p80/p90 → all options score identically on confidence dimension |
| `historical_analogs.duration_distribution.confidence_tier` | Stage D | Stage F (option confidence), Stage G (tier, attention flags) | All options default to `low_confidence` confidence |
| `historical_analogs.analogs[*].outage_id` | Stage D | Stage D (diversity gate) | Outage diversity gate disabled; single-outage pool not caught |
| `schedule_impact_assessment.cp_impact.cp_drag_hours` | Stage E | Stage F (escalation threshold, scoring), Stage G (executive summary) | Scoring and summary produce generic non-critical output |
| `schedule_impact_assessment.float_analysis.criticality_label` | Stage E | Stage F (option generation), Stage G (attention flag) | All options use `non_critical` defaults |
| `schedule_impact_assessment.displaced_tasks[*].has_regulatory_constraint` | Stage E | Stage G (DISPLACED_REGULATORY flag) | Flag never raised |
| `schedule_impact_assessment.crew_continuity` | Stage E | Stage F (`_resolve_crew_count`) | Falls back to `config.default_crew_count = 2` |
| `insertion_options.recommended_option_id` | Stage F | Stage G (primary recommendation, decision status) | Decision status → INCONCLUSIVE |
| `insertion_options.options[*].risk_score` | Stage F | Stage G (primary recommendation) | INCONCLUSIVE if no feasible option |
| `insertion_options.schedule_summary` | Not produced | Stage G (MONITOR condition) | MONITOR branch reads missing key → defaults silently |

### Dead fields (written but never read downstream)

- `intake_result.provenance.ner_pipeline_version` — set to None, never consumed
- `historical_analogs.retrieval_summary.prescorer_candidates` — if populated, not consumed by G
- `schedule_impact_assessment.notes` — written as metadata, not consumed by F or G
- `insertion_options.min_cost_option_id` — computed by F, not read by G

### Fields read by downstream stages that may be null without warning

- `schedule_impact_assessment.cp_impact.locked_baseline_cp_hours` — null when no baseline version archived; Stage G executive summary does not mention baseline absence
- `schedule_impact_assessment.permit_lead_time` — present but not surfaced in Stage G executive summary or evidence chain

---

## 4. Scoring Model Coherence

### Stage F composite risk formula analysis

**Definition:** `risk = 0.35 × cp_impact + 0.25 × (1 − confidence) + 0.20 × resource_score + 0.10 × urgency_score + 0.10 × cost_score`

**Self-consistency check:**

The `defer_to_post_outage` option always has `cp_impact_hours = 0`, so `cp_impact_score = 0`. Its `confidence = 0.85` (hardcoded in `_generate_defer`), so `confidence_score = 0.15`. Its `resource_score = 0` (no conflicts). Its `cost_score` is effectively 0 in-outage (deferred work costs are `deferred_duration × labor_rate` but this is not part of the in-outage cost estimate). Its `urgency_score = urgency` (non-action type).

For `causal_posture = "supported"` (urgency = 0.80):
```
risk(defer) = 0 + 0.25×0.15 + 0 + 0.10×0.80 + cost≈0 = 0.0375 + 0.08 = 0.1175
```

For `insert_now` with 20 h drag on 400 h baseline (cp_impact_score = 0.05):
```
risk(insert_now) = 0.35×0.05 + 0.25×(1−0.65) + 0 + 0.10×0.20 + 0.10×cost_norm
                 = 0.0175 + 0.0875 + 0.02 + cost ≈ 0.125 + cost
```
Even with zero cost, `risk(insert_now) ≈ 0.125 > risk(defer) = 0.1175`. The formula recommends DEFER over INSERT_NOW even in a "supported" causal chain scenario with 20 hours of CP drag — until the regulatory block fires on defer.

This confirms Issue N4: the scoring model relies on the regulatory clearance check to block defer, rather than making defer score higher than insert_now in urgent scenarios through the numeric formula alone. This creates a single point of failure: if any regulatory keyword is missed by Stage A, the system will recommend DEFER when it should PROCEED.

**Known failure modes:**

| Scenario | Expected Decision | Actual Scoring Outcome |
|---|---|---|
| TS-applicable activity, high urgency, 20 h CP drag | PROCEED (TS blocks defer) | Defer scores LOWER numerically; only blocked by regulatory check |
| No TS reference, high urgency, 20 h CP drag | PROCEED | Defer wins numerically; system recommends DEFER incorrectly |
| Zero CP drag, low urgency, no analogs | MONITOR | MONITOR reachable if zero analogs AND low_confidence tier |
| All options infeasible | INCONCLUSIVE | Correct |
| All options regulatory-blocked | INCONCLUSIVE | Correct |

### Stage G decision logic coherence

The decision status hierarchy (INCONCLUSIVE → ESCALATE → DEFER → MONITOR → PROCEED) is applied in priority order. ESCALATE requires `primary_option.option_type == "escalate_to_management"` — which is only generated when CP drag exceeds `escalate_if_cp_drag_exceeds_hours = 24.0`. For activities with 23.9 h CP drag (just below threshold), no escalate option is generated, and the system produces PROCEED even though the drag is nearly equal to the escalation threshold. The 24-hour constant is not configurable per-activity or per-outage phase.

---

## 5. Robustness Analysis

| Scenario | Current Behavior | Risk Level | Recommendation |
|---|---|---|---|
| KG driver absent | Stage B produces empty timeline; pipeline continues; causal_posture = insufficient_data | Low | Document this path as expected degradation; add note to executive summary |
| Zero analogs | `low_confidence` tier; fallback policy used; MONITOR or PROCEED depending on CP | Medium | Verify fallback policy is injected; add test for zero-analog full-pipeline run |
| Single analog | `sme_informed` tier (count = 1 ≥ min_analogs_for_sme_informed); distribution fitted on one point | Low | Verify p50=p80=p90 for single-point distribution; add explicit test |
| `detection_timestamp` missing | Stage B window_start = None → Cypher query behavior undefined; Stage C returns insufficient_data | **High** | Issue N9: add explicit guard in Stage B |
| `regulatory_drivers = []` (empty list) | All options regulatory-cleared; correct behavior | Low | — |
| `regulatory_drivers = None` | Stage F reads `intake_result.get("regulatory_drivers", [])` → `[]` safe default | Low | — |
| Stage E stub (no LOGOS) | RuntimeError propagates; pipeline dies | **High** | Issue N13: add soft degradation path |
| `p50 = 0.0` from distribution | MC samples clamped to _MIN_DURATION_HOURS = 0.1; WARNING logged | Low (fixed) | Verify warning surfaces in logs |
| Concurrent pipeline runs | No shared mutable state; thread-safe via explicit qac passing | Low | — |
| `planned_duration_hours = None` with no actual timestamps | `is_point = True` in Stage C; all relations are SIMULTANEOUS or PRECEDES/FOLLOWS at point | Low | — |
| Source system "maximo" in input | `source_confidence = 0.90` (correct) but schema enum rejects it | Medium | Issue N8: fix schema enum |
| `outage_start` field absent | Pre-outage staging upgrade never fires (option type stays `add_contingency_buffer`) | Medium | Issue N2: add `outage_start` to `emergent_activity.json` schema |
| Very long activity description (>2000 chars) | No length guard; NER pipeline may be slow or produce unusual entity counts | Low | Add truncation at Stage A with WARNING |

---

## 6. Gaps and Missing Features

Issues not previously documented (new in this review):

**N1 (Medium) — Stage A NLP fallback is silent:**  
When `_CLEANERS_AVAILABLE = False` (missing outage_uncertainty.preprocessing), Stage A uses basic whitespace normalization only. The output artifact has no field indicating that preprocessing was degraded. Downstream stages cannot apply lower confidence weights. The `code_review_and_gap_tracker.md` notes this as deferred but it has not been fixed.

**N2 (Medium) — Pre-outage staging option type unreachable in standard pipeline:**  
`emergent_activity["outage_start"]` is not in the schema. The option type upgrade from `add_contingency_buffer` to `pre_outage_staging` never fires through the standard pipeline. Either add `outage_start` to the schema, or derive it from `outage_id` via a lookup in `_generate_contingency_buffer`.

**N3 (Low) ~~— Escalate option does not surface TS action level deadline~~** ✅ **Fixed 2026-04-16**  
~~When escalating an LCO/TS-applicable activity, the escalation rationale does not include the LCO action level time remaining. The outage manager escalating to management needs this number immediately.~~  
**Fix:** `_generate_escalate()` now accepts `intake_result: Optional[JsonDict]`. When `intake_result.has_regulatory_constraint` is `True` or `emergent_activity.active_lco` is set, a TS/LCO deadline note is appended to the rationale (⚠ symbol + LCO number if present + instruction to confirm action-level hours with licensing before briefing management). Call site updated; 5 regression tests added in `TestN3EscalateRegulatoryDeadlineNote`.

**N4 (High) ~~— Scoring formula has structural bias toward DEFER for non-TS urgent activities~~** ✅ **Fixed 2026-04-16**  
~~As analyzed in Section 4. The `causal_urgency` weight (0.10) is insufficient to overcome `cp_impact` for activities with meaningful but sub-threshold CP drag. The regulatory check is the only guard for TS-applicable cases, leaving non-TS urgent activities at risk of a DEFER recommendation.~~  
**Fix:** `_W_URGENCY` 0.10 → 0.20, `_W_CONFIDENCE` 0.25 → 0.15. Regression test added.

**N5 (Medium) — MONITOR decision status reads non-existent `schedule_summary` field from Stage F:**  
The MONITOR condition silently uses `"non_critical"` as the criticality_label because `insertion_options["schedule_summary"]` does not exist. This produces subtly wrong behavior (MONITOR is more reachable than it should be). Fix: either have Stage F produce a `schedule_summary` key, or remove the `criticality_label` guard from the MONITOR condition and document the change.

**N6 (Low) ~~— Evidence chain truncation can exclude regulatory evidence~~** ✅ **Fixed 2026-04-16**  
~~The cap of `max_evidence_items = 10` can exclude regulatory constraint evidence if temporal chain + analogs + schedule analysis already fills the chain.~~  
**Fix:** `_assemble_evidence_chain()` now accepts `intake_result: Optional[JsonDict]`. When `has_regulatory_constraint` is True, a `regulatory_constraint` evidence entry is inserted at index 0 before all other items — guaranteeing it survives the `max_evidence_items` slice. Call site in `synthesize()` updated. 4 tests added in `TestN6RegulatoryEvidencePinned`.

**N7 (High) ~~— No feedback loop from actual execution to analog index~~** ✅ **Fixed 2026-04-16**  
~~The `test_completion_feedback.py` test file exists but the implementation is not wired into the pipeline.~~  
**Fix:** `OutageActivityOrchestrator` now accepts `feedback_writer: Optional[CompletionFeedbackWriter]` field and exposes `record_completion()` public method. When `feedback_writer` is None, the method logs a warning and returns a no-op `CompletionRecord`. Full wiring example documented in field docstring.

**N8 (Medium) ~~— Schema/code mismatch: emergent_activity source_system enum~~** ✅ **Fixed 2026-04-16**  
~~Schema validation was rejecting records from Maximo, SAP, and Primavera; `CMMS` and `CAP` schema values were falling through to `"unknown"` confidence (0.40) in Stage A.~~  
**Fix:** (1) `emergent_activity.json` enum expanded with `maximo`, `sap`, `primavera`. (2) `_SOURCE_CONFIDENCE` in Stage A extended: `cmms → 0.80`, `cap → 0.70`, `other → 0.40`. Confidence ordering: maximo ≥ sap ≥ cmms > cap > manual > other ≥ unknown. 4 regression tests added to `TestComputeDataQuality`.

**N9 (Medium) ~~— Stage B: no guard for missing detection_timestamp in KG query window~~** ✅ **Fixed 2026-04-16**  
~~`_window_start_iso` returned `None` when `detection_timestamp` was absent, passing Python `None` to py2neo as `$window_start` (undefined behaviour across driver versions).~~  
**Fix:** `_window_start_iso` now falls back to `datetime.now(UTC)` when `before_ts` is None or unparseable — return type changed from `Optional[str]` to `str`. `build()` logs a `WARNING` when `detection_timestamp` is absent so operators can diagnose missing input data. 2 tests updated + 2 new tests in `TestWindowStartIso`.

**N10 (Low) ~~— Stage C: lag plausibility artificially depresses confidence for long-running OVERLAPS events~~** ✅ **Fixed 2026-04-16**  
~~Events with OVERLAPS or CONTAINS relations should have lag plausibility forced to 1.0 since the temporal relationship already establishes precedence. The current formula uses event START as lag reference, which penalizes long-running prior events.~~  
**Fix:** `_compute_confidence()` now guards for `allen_relation in (_OVERLAPS, _CONTAINS)` before the decay formula and forces `lag_plausibility = 1.0` unconditionally for those relations. All other relations unchanged. 4 tests added in `TestComputeConfidence` (`test_n10_*`).

**N11 (Medium) — Stage D: when ConfidenceEstimator is injected, outage diversity gate depends on correct `outage_id` population in analogs:**  
The P1 fix added `outage_id` to `_activity_to_analog()`, but test fixtures predate P1 and may not include the field. Confirm that `outages_represented` is correctly computed (not zero) in production mode, and add a test with a ConfidenceEstimator injected.

**N12 (Low) ~~— Stage D: similarity_threshold behavior in production vs. test~~** ✅ **Fixed 2026-04-16**  
~~Stub retrieval indices return fixed similarity scores. First production deployment will see real threshold rejections. Add a metric: `retrieval_summary.candidates_below_threshold` so operators can monitor retrieval quality.~~  
**Fix:** `_score_and_filter()` return type changed from `List[JsonDict]` to `Tuple[List[JsonDict], int]`. Before the neighbor-selection step, counts all scored candidates with score < `config.similarity_threshold`; returns the count as the second element. `_build_retrieval_summary()` accepts and stores `candidates_below_threshold`. Call site in `retrieve()` updated. 6 tests added in `TestN12CandidatesBelowThreshold`.

**N13 (High) ~~— Stage E: RuntimeError when schedule loader is absent kills entire pipeline~~** ✅ **Fixed 2026-04-16**  
~~Should produce a stub artifact with null CP metrics instead of raising, matching Stage B's graceful degradation pattern. This is the most likely failure mode for the first production pilot.~~  
**Fix:** Orchestrator `_stage_e_schedule()` catches `RuntimeError`, appends to `optional_failures`, returns `_stub_schedule_impact_artifact()`. Review hooks surface `schedule_loader_unavailable: True`.

**N14 (Medium) — Stage F produces `ranking_summary` but Stage G reads `schedule_summary`:**  
Dead field reference in Stage G's MONITOR condition. Addressed in Issue N5 above.

---

### New issues found — April 2026 second-pass code review

The following issues were identified during a full functional pass of all stage implementations, cross-referenced against the schemas. All are new — not in the original tracker above.

~~**X1 (Medium) — PRECEDES can never reach `"strong"` causal_strength:**~~  
✅ **Fixed 2026-04-16** — `_RELATION_SCORES[PRECEDES]` raised from `0.75` → `0.80`. At 0.75 the maximum product was `0.75 × 0.9625 = 0.721 < 0.75` (strong threshold), making "strong" unreachable regardless of data quality. At 0.80, `0.80 × 0.9625 = 0.77 ≥ 0.75` — reachable for short-lag, high-quality PRECEDES links. Module docstring and `_RELATION_SCORES` comment updated. Two tests updated/added in `TestAssignCausalStrength`: `test_precedes_moderate_confidence_moderate` (expected value updated) and `test_precedes_high_confidence_strong` (new). One `_compute_confidence` test that hardcoded the old relation score was updated.

~~**X2 (Low) — `lco_number` not propagated through Stage A:**~~  
✅ **Fixed 2026-04-16** — `"lco_number": emergent_activity.get("lco_number")` added to Stage A `process()` return dict alongside the M1 clock fields. `activity_intake_result.json` schema updated with `lco_number` (and the previously-missing M1 fields `preprocessing_available`, `execution_mode_flags`, `lco_action_level_expires_at`, `hours_to_action_level`, `lco_clock_status` — all produced by Stage A but absent from the schema). 3 tests added in `TestX2LcoNumberForwarding`.

~~**X3 (High) — `rejection_reason` violates `analyst_review` schema:**~~  
✅ **Fixed 2026-04-16** — `analyst_review` had `additionalProperties: false`. `_determine_analyst_review()` returned `rejection_reason: None` as part of the §8d feedback loop fix, but the field was absent from the schema. `rejection_reason` added to `analyst_review.properties` in `outage_activity_recommendation.json` with `type: ["string", "null"]` and a description explaining its role in the rejection path.

~~**X4 (High) — Stage B `ValueError` propagates uncaught, kills pipeline:**~~  
✅ **Fixed 2026-04-16** — `_stage_b_kg_timeline()` now accepts `optional_failures: List[JsonDict]` (mirroring Stage E's signature). `build()` is wrapped in `try/except ValueError`; on failure a WARNING is logged, the error is appended to `optional_failures`, and `_stub_component_event_timeline()` is returned — a fully-structured artifact with empty events, `kg_driver_available: False`, and `data_coverage: 0.0`. The call site in `run()` passes `optional_failures`. `_stub_component_event_timeline()` added alongside `_stub_schedule_impact_artifact()` in the orchestrator module. 4 tests added in `TestX4StageBValueErrorDegradation` in `test_orchestrator_e2e.py`.

~~**X5 (High) — `"regulatory_constraint"` not in `evidence_chain source_type` enum:**~~  
✅ **Fixed 2026-04-16** — The N6 fix inserted a `source_type="regulatory_constraint"` evidence entry at index 0 of the evidence chain. The schema `outage_activity_recommendation.json` `evidence_chain[*].source_type` enum did not include this value, causing strict validation failure on any artifact with regulatory constraints. `"regulatory_constraint"` added to the `source_type` enum.

~~**X6 (Medium) — `outage_ids` not in `history_summary` schema:**~~  
✅ **Fixed 2026-04-16** — `"outage_ids": {"type": "array", "items": {"type": "string"}}` added to `history_summary.properties` in `outage_activity_recommendation.json` with a §8a traceability description. Existing test `test_outage_ids_field_present` already validates the field value; no new tests needed.

---

### New issues found — April 2026 third-pass code review

These issues were identified during a second full functional pass (stages A–G, orchestrator, schemas) focused on code/schema divergence that is masked by `NoOpSchemaValidator`. All are new to this session.

~~**Y1 (High) — Stage C `chain_links` output uses wrong field names:**~~  
✅ **Fixed 2026-04-16** — `_score_event()` returned `event_id`, `event_type`, `event_timestamp`, `data_quality_score`. The `temporal_event_chain.json` schema requires `prior_event_id`, `prior_event_type` as required fields under `additionalProperties: false`. `data_quality_score` is not in the schema properties. Fixed by renaming the three keys (`prior_event_id`, `prior_event_type`, `prior_event_timestamp`) and removing `data_quality_score` from the output dict. Stage G only reads `link_id`, `allen_relation`, `relation_score`, `onset_lag_hours`, `causal_strength` — no downstream breakage. 2 tests added in `TestStageCScoring`: `test_y1_chain_link_uses_prior_event_id` and `test_y1_prior_event_id_value_matches_input`.

~~**Y2 (Critical) — Stage A `_REGULATORY_PATTERNS` driver_type values do not match schema enum:**~~  
✅ **Fixed 2026-04-16** — `_REGULATORY_PATTERNS` produced 6 driver_type values not in the `activity_intake_result.json` schema enum (`"ts_surveillance"`, `"nrc_commitment"`, `"cap_commitment"`, `"hold_point"`, `"alara_constraint"`, `"license_basis_inspection"`, `"other"`): `"technical_specification"` → `"ts_surveillance"`, `"limiting_condition_for_operation"` → `"ts_surveillance"`, `"nrc_regulation"` → `"nrc_commitment"`, `"alara_requirement"` → `"alara_constraint"`, `"corrective_action_program"` → `"cap_commitment"`, `"surveillance_requirement"` → `"ts_surveillance"`, `"operability_determination"` → `"license_basis_inspection"`, `"mode_change_constraint"` → `"other"`. The two hardcoded driver_type strings in `_detect_regulatory_constraints` structured-field path (`"technical_specification"` and `"limiting_condition_for_operation"`) were also updated to `"ts_surveillance"`. Stage F's `_DEFER_PROHIBITED_TYPES` and `_SCOPE_REDUCTION_PROHIBITED_TYPES` sets were updated to the new values (critical: `_SCOPE_REDUCTION_PROHIBITED_TYPES` only checks driver_type, not `defer_prohibited`, so the rename would have silently unblocked scope reduction for TS-constrained activities). Tests in `TestDetectRegulatoryConstraints` and `TestCheckRegulatoryClearance` updated. 4 new Y2 tests added in `TestDetectRegulatoryConstraints`.

~~**Y3 (Low) — `"contradicted_with_support"` causal_posture value not in schema enum:**~~  
✅ **Fixed 2026-04-16** — `_summarize_chain()` has dead code path producing `"contradicted_with_support"` (currently unreachable via the `if has_contradiction` guard, but present in Stage F's `_POSTURE_TO_URGENCY` dict). `temporal_event_chain.json` `causal_posture` enum did not include this value. Added `"contradicted_with_support"` to the enum.

~~**Y4 (Medium) — Stage B `pm_compliance_status` uses `"current"` not in schema enum:**~~  
✅ **Fixed 2026-04-17** — `"current"` → `"compliant"` at the within-interval branch in `_compute_recurrence_indicators()`. 2 tests added in `TestComputeRecurrenceIndicators`: `test_y4_pm_compliance_status_is_schema_valid` (all three code paths) and `test_y4_within_interval_produces_compliant_not_current`.

~~**Y5 (Medium) — Stage B `data_coverage` return fields don't match schema:**~~  
✅ **Fixed 2026-04-17** — `_compute_data_coverage()` renamed `earliest_event_date` → `earliest_event`, `latest_event_date` → `latest_event`, and removed the non-schema fields `window_start`, `window_end`, `has_gaps`. Added `data_quality_summary: None`. The unused `window_start` local variable was also removed. 4 tests added in `TestComputeDataCoverage`.

~~**Y6 (Medium) — Stage B `recurrence_indicators` includes non-schema fields:**~~  
✅ **Fixed 2026-04-17** — Removed `min_inter_event_days` and `pm_overdue_days` from the `_compute_recurrence_indicators()` return dict. Both were computed but neither appears in the `component_event_timeline.json` `recurrence_indicators` schema under `additionalProperties: false`. The local variables are retained for computation but not emitted. 3 tests added in `TestComputeRecurrenceIndicators`.

~~**Y7 (Medium) — `_stub_component_event_timeline()` has schema violations:**~~  
✅ **Fixed 2026-04-17** — Four changes: (1) `"data_coverage": 0.0` → proper object with all schema fields (`total_events`, `outages_represented`, `earliest_event`, `latest_event`, `data_quality_summary`); (2) removed `"component_type": None` (not in schema and not emitted by `build()` either); (3) kept `"kg_driver_available": False` and added the field to `component_event_timeline.json` schema (it's set by `build()` and consumed by the orchestrator at lines 914 and 996 — it's a legitimate artifact field); (4) replaced non-schema `recurrence_indicators` fields `last_failure_date` → `last_cm_date` and removed `inter_event_period_days`, added `mean_inter_event_days: None`. 3 tests added in `TestX4StageBValueErrorDegradation`.

~~**Y8 (High) — `_stub_schedule_impact_artifact()` has multiple schema violations:**~~  
✅ **Fixed 2026-04-17** — Stub completely rebuilt from schema: (1) removed `schedule_loader_unavailable`, `crew_continuity`, `permit_lead_time` (not in schema under `additionalProperties: false`); (2) `criticality_label = "unknown"` → `"non_critical"`; (3) `available_float_before` → `available_float_before_hours`, `remaining_float_hours` → `remaining_float_after_hours`; (4) added required `is_critical_path_impact: False`; (5) added required `estimated_new_cp_hours: 0.0` to `cp_impact`; (6) `schedule_version_id: None` → `"STUB::no_schedule"` (schema requires string); (7) `duration_estimate: {}` → proper object with all required fields. Stub is identified by `schedule_version_id == "STUB::no_schedule"`. Two existing tests updated; 1 new comprehensive Y8 schema-compliance test added in `TestN13StageEStubDegradation`.

---

## 7. Test Strategy

### 7.1 Philosophy

The existing 838-test suite is strong on unit coverage of individual stage methods. The gaps are:

1. **No test exercises Stage E RuntimeError path** (missing schedule loader)
2. **No test for empty raw_description end-to-end** (Stage B ValueError propagation)
3. **No test for MONITOR decision status** through a full orchestrator run
4. **No test with injected ConfidenceEstimator** verifying outage diversity gate
5. **No test for the pre-outage staging option type** (requires `outage_start` field)
6. **No adversarial input test** (very long descriptions, malformed timestamps, extreme CP drag values)

New tests should be placed in `tests/` following the existing pattern: `_OUTAGE_ROOT` path insertion at the top, duck-typed mock objects for LOGOS dependencies.

### 7.2 Unit Test Matrix

| Stage | Method | Test Cases | Priority |
|---|---|---|---|
| A | `_classify_emergence_type` | pre-classified pass-through at confidence 1.0; regulatory keyword only; scope + WO; scope no WO; schedule opt; degradation keyword; empty text | High |
| A | `_extract_execution_mode_flags` | RP hold variants; scaffold; LOTO/clearance; vendor OEM; multi-flag; empty | Medium (existing) |
| A | `_compute_data_quality` | all fields present; missing timestamp; missing component; manual source; unknown source | Medium |
| A | `process` with missing raw_description | empty string → abbr_rate=0, no entities, truly_unplanned at 0.45 | High |
| B | `build` with detection_timestamp = None | should not raise; should use fallback window | High |
| B | `build` with max_events cap | 150 events → only last 100 retained | Medium |
| B | `_compute_recurrence_indicators` | 0 events; 1 event; 2 events (insufficient_data); increasing trend; decreasing trend | Medium |
| C | `_allen_relation` | all 7 relations; epsilon boundary cases; both endpoints None | High |
| C | `_compute_confidence` | OVERLAPS with 1000h lag (should give low plausibility); lag = None; negative lag | Medium |
| C | Long-running OVERLAPS event | verify confidence is not artificially low (Issue N10) | Medium |
| D | `_compute_confidence_tier` with injected ConfidenceEstimator | 10 analogs from 1 outage → sme_informed cap; 10 from 3 outages → data_supported | High |
| D | `_remove_duration_outliers` with mixed flags | disruption-context analog above fence preserved; non-disruption outlier removed | Medium (existing) |
| D | Zero analogs below threshold | fallback_used=True; confidence_tier=low_confidence; p50=None | High |
| E | `assess` with schedule_loader = None | RuntimeError currently; should produce stub artifact | High |
| E | `_run_monte_carlo` with p50 = 0.0 | all samples clamped to 0.1h; warning logged | Low (existing) |
| E | `_compute_cp_metrics` with locked baseline | variance positive when working > baseline; overrun clamped at 0 | Medium (existing) |
| E | `_has_regulatory_constraint` | TS/LCO/surveillance/hold-point/NRC/ALARA/mode-change → True; plain → False | Low (existing) |
| F | `_score_option` urgency direction | action option with urgency=0.80 scores LOWER than non-action; verify formula | High |
| F | `_generate_defer` feasibility | safety_related=True → infeasible; active_lco=True → infeasible; both False → feasible | High |
| F | `_generate_contingency_buffer` with outage_start absent | option_type stays `add_contingency_buffer` (never `pre_outage_staging`) | Medium |
| F | `_check_regulatory_clearance` | escalate always cleared; insert_now always cleared; defer blocked by TS; scope_reduction blocked by surveillance | High (existing) |
| F | All options infeasible or blocked | `recommended_option_id = None`; INCONCLUSIVE in G | High |
| F | CP drag exactly equal to escalation threshold | no escalate option generated at threshold; generated at threshold+0.01 | Medium |
| G | `_determine_decision_status` MONITOR | zero analogs, low_confidence tier, no CP impact → MONITOR | High |
| G | `_determine_decision_status` with missing schedule_summary | verify default "non_critical" behavior; document intent | Medium |
| G | Evidence chain regulatory entry first | when has_regulatory_constraint=True, first evidence entry is regulatory | Low |
| G | `_build_history_summary` | outage_ids deduplicated; empty analogs list | Medium |

### 7.3 Integration Test Scenarios

**IT-01: Full pipeline, no backends (all stubs)**
- Input: activity with regulatory keyword "TS 3.4.6" and non-empty description
- Expected: ESCALATE or PROCEED with regulatory flag in executive summary; `has_regulatory_constraint = True`; defer option marked `regulatory_cleared = False`
- Purpose: Verify regulatory flow end-to-end

**IT-02: Full pipeline, zero analogs**
- Input: activity description with no close matches in stub retrieval index (similarity always < threshold)
- Expected: `confidence_tier = low_confidence`; `fallback_used = True`; decision status MONITOR or INCONCLUSIVE
- Purpose: Verify fallback policy and low-analog path

**IT-03: Full pipeline, missing detection_timestamp**
- Input: activity with `detection_timestamp = None`
- Expected: Pipeline completes (no RuntimeError from Stage B); `causal_posture = insufficient_data`; MONITOR or PROCEED; WARNING logged
- Purpose: Verify Issue N9 fix when applied

**IT-04: Full pipeline, Stage E absent (no schedule loader)**
- Input: any activity with no schedule_loader injected
- Expected: Pipeline produces partial recommendation with null CP metrics; PROCEED or MONITOR based on analog data alone; WARNING logged
- Purpose: Verify Issue N13 fix when applied

**IT-05: Execution mode flag pipeline**
- Input: activity description containing "RP hold", high-dose work area
- Expected: `execution_mode_flags.has_rp_hold = True`; Stage D returns RP-hold analogs above IQR fence; Stage E includes `rp_hold_lead_time_hours` in `permit_lead_time.total_lead_hours`
- Purpose: Verify P2a/P2b/P2c/permit_lead_time chain

**IT-06: Multi-outage diversity gate**
- Input: 10 analogs all from outage_id = "RF-20"; injected ConfidenceEstimator with default thresholds
- Expected: `confidence_tier = sme_informed` (capped from data_supported)
- Purpose: Verify P1 fix is correctly applied in end-to-end retrieval

### 7.4 End-to-End Golden-Path Tests

**E2E-01: RCP Seal Leak (Scenario 1, Demo Workflow 1)**
- `emergent_activity`: `RCCPUMP-1A-SEAL-LEAK`, emergence_type pre-set to `regulatory_driven`, TS 3.4.6 reference
- Stub backends: KGDriver with 3 prior CRs (seal leakage history); RetrievalIndex with 5 analogs averaging 18 h; schedule with 48 h CP drag
- Expected final status: `ESCALATE`; `regulatory_cleared = False` for defer; `has_regulatory_constraint = True`; evidence chain includes temporal chain link and top analogs
- Assert: `outage_activity_recommendation.decision_status == "ESCALATE"`, `outage_activity_recommendation.executive_summary.primary_conclusion` contains "REGULATORY CONSTRAINT PRESENT"

**E2E-02: Snubber Scope Expansion (Scenario 2, Demo Workflow 1)**
- `emergent_activity`: `SNUBBER-3A-INSP-SCOPE`, emergence_type `scope_expansion`, no regulatory keywords
- Stub backends: minimal CR history; 3 analogs averaging 12 h; schedule with 28 h float, 0 h CP drag
- Expected final status: `PROCEED` with insert_now; no regulatory flags; `confidence_tier = sme_informed`
- Assert: `outage_activity_recommendation.decision_status == "PROCEED"`, `insertion_options.recommended_option_id` maps to `insert_now` option

**E2E-03: Unknown component, no KG history, no analogs**
- `emergent_activity`: minimal record with obscure component ID, no existing records
- Stage B returns empty timeline; Stage D returns zero analogs with fallback
- Expected: `decision_status == "MONITOR"` (no CP impact + low_confidence + zero analogs); `analyst_review.required == True`; all 3 low-evidence attention flags raised
- Assert: MONITOR status, analyst_review.required=True, attention flags include `low_confidence_recommendation`, `low_analog_count`, `fallback_distribution_used`

### 7.5 Adversarial / Edge Case Tests

**ADV-01: Extremely long raw_description (5000 characters)**
- Verify Stage A completes without timeout; `abbr_rate` is computed; NER does not crash
- Verify no truncation of first 255 chars for entity extraction when entity is in the tail

**ADV-02: All-uppercase description (maximum abbreviation density)**
- `raw_description = "1SJ MOV 101 PKG LKG TS 3 5 7 SEE WO 483921 CR 29847"`
- Verify `abbr_rate` > 0.25 triggers WARNING; `_FLAG_HIGH_ABBR_RATE` raised in Stage G

**ADV-03: Non-ASCII characters in description**
- `raw_description` contains UTF-8 characters (e.g., from copy-paste of a French or Spanish CAP system)
- Verify `_TAG_ID_RE` and `_WO_REF_RE` do not raise; `abbr_rate` is computed on ASCII tokens only

**ADV-04: CP drag exceeds outage duration**
- `cp_drag_hours = 800.0` on a 600-hour outage; `baseline_cp_hours = 600.0`
- `cp_impact_score = min(1.0, 800/600) = 1.0` — clamped correctly
- Verify `escalate_to_management` option is generated; decision status ESCALATE

**ADV-05: Regulatory_drivers list with duplicate driver types**
- Two TS entries with same `driver_type`; both with `defer_prohibited = True`
- Verify regulatory block message does not repeat both entries; deduplicated

**ADV-06: `p50_hours` and `p80_hours` both zero from distribution**
- `contingency_buffer_hours = max(0, p80 - p50) = 0` → zero-width buffer
- Verify contingency buffer option is still generated (not filtered out as nonsensical)
- Verify cost estimate for this option is zero and does not divide by zero

**ADV-07: All displaced tasks have `has_regulatory_constraint = True`**
- Schedule with 10 displaced surveillance tasks
- Verify `_FLAG_DISPLACED_REGULATORY` raised; executive summary flags this prominently

### 7.6 Test Fixtures Status ✅ Updated 2026-04-16

| Fixture | Location | Purpose | Status |
|---|---|---|---|
| `_StubKGDriver` with empty result | `tests/` | Stage B graceful degradation path | In `demo_scenarios.py`; needs separate test version |
| `_StubRetrievalIndex` with zero results (all below threshold) | `tests/test_stages_f_g.py` | Zero-analog path | Not present |
| ~~`_StubScheduleLoader` that returns RuntimeError stub~~ | ~~`tests/test_stage_e.py`~~ | ~~Issue N13 test~~ | ✅ **Covered 2026-04-16** — `TestN13StageEStubDegradation` in `test_orchestrator_e2e.py` |
| ~~Activity with `detection_timestamp = None`~~ | ~~`tests/test_stage_b.py`~~ | ~~Issue N9 test~~ | ✅ **Covered 2026-04-16** — `TestWindowStartIso::test_none_before_ts_falls_back_to_utcnow` |
| ~~Activity with `source_system = "maximo"`~~ | ~~`tests/test_stages_a_c.py`~~ | ~~Issue N8 test~~ | ✅ **Covered 2026-04-16** — `TestComputeDataQuality::test_n8_*` (4 tests) |
| ~~Analogs with explicit `outage_id` field~~ | ~~`tests/test_stages_f_g.py`~~ | ~~Outage diversity gate with injected ConfidenceEstimator~~ | ✅ **Covered 2026-04-16** — `TestComputeConfidenceTierWithEstimator` (5 tests) |
| ~~Activity with `outage_start` present~~ | ~~`tests/test_stages_f_g.py`~~ | ~~Pre-outage staging option type reachability~~ | ✅ **Covered 2026-04-16** — `TestGenerateContingencyBuffer` (5 tests) |
| ~~Full pipeline run returning MONITOR status~~ | `tests/test_orchestrator_e2e.py` | MONITOR path | ✅ **Covered 2026-04-17** — `SCENARIO_UNKNOWN_COMPONENT` (empty analogs, non-critical schedule, safety_related=True); `TestMonitorScenario` 12 tests |

### 7.7 Coverage Targets

| Stage | Achievable Coverage | Meaningful Coverage | Notes |
|---|---|---|---|
| Stage A | 90%+ | High | Core NLP paths depend on injected backends; regex paths fully testable |
| Stage B | 85%+ | High | KG paths need stub driver; recurrence indicators fully testable |
| Stage C | 95%+ | High | Pure algorithmic; all Allen relations testable without dependencies |
| Stage D | 85%+ | High | Retrieval pipeline needs stub index; outlier and distribution logic fully testable |
| Stage E | 75%+ | High | MC simulation and float analysis testable via mock Pert; schedule loader path needs stub |
| Stage F | 90%+ | High | All 7 option generators + scoring fully testable; cost model testable |
| Stage G | 90%+ | High | Pure logic; all decision paths testable with constructed inputs |
| Orchestrator | 80%+ | Medium | End-to-end flow testable; optional failure paths need adversarial fixtures |
| outage_uncertainty modules | 85%+ | High | ConfidenceEstimator, DistributionFitter, OutlierHandler are pure Python |

**Status as of 2026-04-17:** All Y-series issues (Y1–Y8) resolved. Full issue set H1–H6, M1, M2–M7 [doc-only], N1–N3, N4, N5, N6, N7–N10, N11, N12, N13, D1–D3, X1–X6, Y1–Y8 resolved with direct unit tests. ~~Full pipeline run returning MONITOR status~~ ✅ **Fixed 2026-04-17** — `SCENARIO_UNKNOWN_COMPONENT` added to `demo_scenarios.py` (empty analog list + non-critical schedule + safety_related=True to block defer); `TestMonitorScenario` (12 tests) added to `test_orchestrator_e2e.py`; documents E2E-03. ~~Adversarial input tests~~ ✅ **Fixed 2026-04-17** — ADV-01–ADV-07 implemented: `TestAdversarialInputStageA` (10 tests in `test_stages_a_c.py`) covers long descriptions, all-caps high-abbr-rate, non-ASCII inputs; `TestAdversarialStageF` (11 tests in `test_stages_f_g.py`) covers CP drag > baseline clamping, escalate generation at extreme drag, duplicate driver type deduplication, and zero p50/p80 division-free contingency buffer; `TestAdversarialStageG` (7 tests in `test_stages_f_g.py`) covers high-abbr-rate flag, analyst review required, and all-displaced-regulatory flag + schedule summary. Also fixed `_check_regulatory_clearance` to deduplicate driver_type strings in block reason message (ADV-05 fix). N13 inline note updated to remove stale `schedule_loader_unavailable` reference. 1,350 tests passing, 0 failures.

---

## Appendix A: Priority Order for New Issues

| Issue | Severity | Effort | Recommended Action |
|---|---|---|---|
| ~~N13 — Stage E kills pipeline on absent schedule loader~~ | ~~High~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — stub artifact + optional_failures + review hook |
| ~~N7 — No feedback loop (actual duration → analog index)~~ | ~~High~~ | ~~High~~ | ✅ Fixed 2026-04-16 — `feedback_writer` field + `record_completion()` on orchestrator |
| ~~N4 — Scoring bias toward DEFER for non-TS urgent activities~~ | ~~High~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — urgency weight 0.10→0.20, confidence 0.25→0.15; regression test added |
| ~~D1 — `kg_unavailable` not a first-class review hook flag~~ | ~~Medium~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — Stage B artifact adds `kg_driver_available`; orchestrator surfaces `kg_unavailable` in review_hooks |
| ~~D2 — `InsertionOptionGenerator` has no plugin registry~~ | ~~Medium~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — `extra_option_generators` constructor param + `register_option_generator()` method |
| ~~D3 — `escalate_to_management` cost missing decision latency~~ | ~~Medium~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — `escalate_decision_delay_hours` config field; `decision_latency_cost_usd` added to cost estimate |
| ~~N8 — Schema/source_system enum mismatch~~ | ~~Medium~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — schema enum expanded; `cmms`/`cap`/`other` added to confidence map |
| ~~N9 — Stage B null detection_timestamp in KG query~~ | ~~Medium~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — `_window_start_iso` fallback to utcnow(); WARNING in `build()` |
| ~~N2 — Pre-outage staging option type unreachable~~ | ~~Medium~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — `outage_start` added to `emergent_activity.json`; 4 tests added |
| ~~N5 — MONITOR reads missing `schedule_summary` field~~ | ~~Medium~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — reads `schedule_impact["float_analysis"]["criticality_label"]`; 3 tests added |
| ~~N11 — Outage diversity gate with injected ConfidenceEstimator~~ | ~~Medium~~ | ~~Medium~~ | ✅ Fixed 2026-04-16 — 5 regression tests added; code was already correct |
| ~~M1 — TS action level countdown missing~~ | ~~Critical~~ | ~~Medium~~ | ✅ Fixed 2026-04-16 — `_compute_lco_clock()` in Stage A; 3 clock fields in `process()` output; 2 LCO flags + clock prefix in Stage G; 26 tests |
| ~~M5 — Learning loop unimplemented~~ | ~~High~~ | ~~High~~ | ✅ Fixed 2026-04-16 — `CompletionFeedbackWorkflow.ingest()` bridges Stage G artifact → `record_completion()`; accepted/rejected/modified/pending routing; 30 tests in `test_completion_feedback_workflow.py` |
| ~~N1 — NLP fallback silent~~ | ~~Medium~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — `preprocessing_available: bool` added to `process()` output; 3 tests in `TestN1PreprocessingAvailableField` |
| ~~N10 — Stage C lag plausibility for OVERLAPS events~~ | ~~Low~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — `lag_plausibility=1.0` for OVERLAPS/CONTAINS; 4 tests in `TestComputeConfidence` |
| ~~N6 — Evidence chain can exclude regulatory evidence~~ | ~~Low~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — regulatory entry pinned at index 0; 4 tests in `TestN6RegulatoryEvidencePinned` |
| ~~N3 — Escalate option missing TS deadline note~~ | ~~Low~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — LCO deadline note appended to rationale; 5 tests in `TestN3EscalateRegulatoryDeadlineNote` |
| ~~N12 — Missing `candidates_below_threshold` metric~~ | ~~Low~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — field added to `retrieval_summary`; `_score_and_filter` returns `Tuple[List, int]`; 6 tests in `TestN12CandidatesBelowThreshold` |
| ~~X4 — Stage B `ValueError` propagates uncaught, kills pipeline~~ | ~~High~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — `try/except ValueError` + `_stub_component_event_timeline()`; 4 tests in `TestX4StageBValueErrorDegradation` |
| ~~X3 — `rejection_reason` violates `analyst_review` schema~~ | ~~High~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — `rejection_reason` added to `analyst_review.properties` in schema |
| ~~X5 — `"regulatory_constraint"` not in `source_type` enum~~ | ~~High~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — `"regulatory_constraint"` added to `evidence_chain source_type` enum |
| ~~X6 — `outage_ids` not in `history_summary` schema~~ | ~~Medium~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — `"outage_ids"` added to `history_summary.properties` in schema |
| ~~X1 — PRECEDES can never reach `"strong"` causal_strength~~ | ~~Medium~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — `_RELATION_SCORES[PRECEDES]` 0.75 → 0.80; 2 tests updated/added in `TestAssignCausalStrength` |
| ~~X2 — `lco_number` not propagated through Stage A~~ | ~~Low~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — `lco_number` added to Stage A output + intake schema; 3 tests in `TestX2LcoNumberForwarding` |
| ~~Y2 — Stage A `_REGULATORY_PATTERNS` driver_type values not in schema enum~~ | ~~Critical~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — 8 patterns remapped to schema-valid values; Stage F prohibit sets updated; 4 tests in `TestDetectRegulatoryConstraints` |
| ~~Y1 — Stage C `chain_links` output uses `event_id`/`event_type` instead of `prior_event_id`/`prior_event_type`~~ | ~~High~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — fields renamed in `_score_event()`, `data_quality_score` removed; 2 tests added |
| ~~Y3 — `"contradicted_with_support"` not in `temporal_event_chain` `causal_posture` enum~~ | ~~Low~~ | ~~Low~~ | ✅ Fixed 2026-04-16 — added to schema enum |
| ~~Y8 — `_stub_schedule_impact_artifact()` has multiple schema violations~~ | ~~High~~ | ~~Low~~ | ✅ Fixed 2026-04-17 — stub rebuilt from schema; 1 comprehensive Y8 test added |
| ~~Y4 — Stage B `pm_compliance_status = "current"` not in schema enum~~ | ~~Medium~~ | ~~Low~~ | ✅ Fixed 2026-04-17 — `"current"` → `"compliant"`; 2 tests added |
| ~~Y5 — Stage B `data_coverage` field names don't match schema~~ | ~~Medium~~ | ~~Low~~ | ✅ Fixed 2026-04-17 — renamed 2 fields, removed 3 non-schema fields; 4 tests added |
| ~~Y6 — Stage B `recurrence_indicators` includes non-schema fields~~ | ~~Medium~~ | ~~Low~~ | ✅ Fixed 2026-04-17 — removed `min_inter_event_days` and `pm_overdue_days`; 3 tests added |
| ~~Y7 — `_stub_component_event_timeline()` has schema violations~~ | ~~Medium~~ | ~~Low~~ | ✅ Fixed 2026-04-17 — stub rebuilt; `kg_driver_available` added to schema; 3 tests added |
