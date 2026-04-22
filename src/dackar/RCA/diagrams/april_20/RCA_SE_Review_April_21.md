# RCA Pipeline — Comprehensive Systems Engineering Review
**Date**: April 21, 2026 · **Sprint 1 applied**: April 21, 2026 · **Sprint 2 applied**: April 21, 2026 · **Option B applied**: April 21, 2026 · **Sprint 5 applied**: April 21, 2026 · **Post–April 21 SE code batch applied**: April 21, 2026 · **Sprint 6 applied**: April 21, 2026 · **Sprint 7 applied**: April 21, 2026
**Baseline**: Orchestrator v3.2 · Schema set v3.2 · RCA_pipeline_stages.md (April 21)
**Reviewer perspective**: Systems Engineer / RCA Practitioner (nuclear power plant)
**Scope**: Cross-document synthesis covering all four companion documents:
- `RCA_pipeline_flowchart.md` — architecture reference
- `RCA_pipeline_stages.md` — per-stage drill-down (primary working document)
- `RCA_workflow_april_2.md` — formal method specification
- `RCA_Systems_Engineering_Review_April_20.md` — prior review baseline

**Predecessor**: `RCA_Systems_Engineering_Review_April_20.md` identified C1–C5, H1–H6, M1–M8. This document does **not** repeat those findings; it references them by ID and adds new findings organized by theme. The combined finding set is summarised in the Priority Matrix (§10).

---

## 1. Executive Summary

The pipeline is architecturally sound for a decision-support tool. The dual-scoring-pass design, schema-validated artifact contracts, and deterministic fallback path reflect careful engineering. However, a systematic review across all four documents reveals **33 new findings** (6 critical, 12 high, 15 medium) in addition to the 21 findings from the April 20 review. Three architectural patterns account for most of the severity:

1. **The pipeline is strictly feedforward with no recovery paths.** An error, omission, or data-quality problem at any stage propagates silently to all downstream stages. There is no retry, no feedback loop, and no mid-pipeline abort for partial failures.

2. **The scoring system is under-constrained.** Weights are unempirical, do not sum-to-1.0 by enforcement, and have no feedback mechanism from analyst outcomes. Two scoring dimensions (operational context, safety significance) are computed but not consumed. The evidence score is used in pre-evidence filtering — a circular reference.

3. **The pipeline is opaque at the stage boundary.** Validation is late-binding (Stage J only). No artifact is schema-validated at the producing stage. Silent failures — silent degradation of BM25, silent no-op of `component_ids` filter, silent empty of `kg_context.past_events[]` — are the dominant error mode. An engineer operating the system has no mechanism to detect that a run's evidence retrieval degraded silently to dense-only.

---

## 2. Scope and Method

### What was reviewed
Every stage from A through J was assessed for: (1) functional correctness against its stated purpose, (2) internal logical consistency, (3) consistency with `RCA_Data_Management_Strategy.md` as the authoritative data reference, (4) consistency across stage boundaries (output of stage N is the actual input to stage N+1), and (5) regulatory adequacy for nuclear plant CAP submittal.

### What was not reviewed
Code-level correctness beyond the pseudo-code in `RCA_pipeline_stages.md`. Performance and scalability. Infrastructure and deployment.

### Reference test case
Throughout this review, the reference scenario is: **PWR Unit 2, condenser vacuum loss, EVT-U2-2024-0847**. True root cause = air in-leakage through expansion joint. Contributing cause = HVAC fan bearing failure → elevated ambient → accelerated thermal fatigue. Discriminating signal = dissolved oxygen 142 ppb (normal < 10 ppb), which contradicts the fouling hypothesis.

---

## 3. Cross-Pipeline Architectural Issues

### 3.1 Strictly Sequential Pipeline with No Recovery Paths (CRITICAL — ✅ FIXED latest batch)

**Finding**: The pipeline has one control path. If Stage B builds an incomplete neighborhood (missed component), Stage C scores temporal patterns on an incomplete failure mode set, Stage D generates candidates from that incomplete set, Stage E retrieves evidence for those candidates, and Stage H synthesizes from that evidence — all without any mechanism to detect or recover from the upstream omission.

**No feedback loops exist**:
- Stage E's retrieval results cannot expand Stage D's candidate set
- Stage F's post-evidence ranking cannot trigger a second Stage B expansion
- Stage H's synthesis cannot flag that candidate generation was incomplete and trigger re-entry

This is an architectural constraint, not a code bug. It means the system's output quality is bounded by Stage B's completeness, and Stage B's completeness is bounded by KG coverage — which is never checked at runtime.

**Consequence**: In the reference test case, if Stage B missed the HVAC fan (upstream contributing cause), the contributing cause pathway is unrecoverable regardless of what evidence Stage E retrieves.

**Recommendation**: At minimum, add a **pipeline re-entry hook** when Stage F's post-evidence primary differs from Stage D's pre-evidence primary. The new primary should trigger a targeted second KG expansion for its upstream components. Short-term: surface any rank-inversion as `analyst_attention_flags["rank_inversion_detected"]` with explicit text prompting the analyst to verify KG coverage for the new primary.

**Status (Latest batch)**: Extended to include in-run automatic recovery.
- `RCAReasoningOrchestrator._compute_reentry_hook` now emits a structured re-entry recommendation when rank inversion is detected (`pre_evidence_top_candidate_id` != `post_evidence_top_candidate_id`), including targeted `target_component_ids` and explicit follow-up action text.
- `review_hooks` now carries `reentry_hook`, `degraded_run`, and `degraded_reasons`.
- `writeback_ready` is now blocked when re-entry is recommended (degraded branch), preventing silent promotion of unstable runs.
- New bounded auto re-entry loop (`enable_auto_reentry`, `auto_reentry_max_attempts`) now re-runs Stage B→F in the same execution when rank inversion is detected, using targeted focus components from `reentry_hook.target_component_ids`.
- Re-entry execution metadata (`attempt_count`, per-attempt status/details) is now recorded in `run_manifest.pipeline_config.reentry_execution`.
- A dedicated `reentry_execution` artifact is now persisted and validated each run for audit traceability (in addition to the run-manifest summary block).

---

### 3.2 Silent Failure Propagation — No Mid-Pipeline Error Protocol (HIGH — ✅ PARTIALLY FIXED latest batch)

**Finding**: No stage defines what it does when its input artifact is incomplete or its own output is incomplete. The four modalities are:
- Stage A: abort if `event_id` or `asset_id` missing; otherwise continue with warnings
- Stages B–G: continue silently on partial data
- Stage H: produce rca_card even if zero candidates survived filtering
- Stage J: validate after the fact

The gap: Stages B through G have no defined failure contract. A Stage B that finds zero failure modes does not abort, warn loudly, or modify the pipeline's behavior at Stage D — Stage D simply generates zero candidates, Stage E retrieves nothing, and Stage H synthesizes an `insufficient_evidence` conclusion with no indication that the root cause was zero KG coverage, not genuine absence of evidence.

**Recommendation**: Define a **stage-level health contract** for each stage:
- **Green**: output complete, all fields populated
- **Yellow**: output partial, missing optional fields, downstream stages adjusted
- **Red**: output structurally incomplete, downstream stages must treat as degraded run

Carry `pipeline_health` field in each artifact, and have Stage J's `next_step` routing include a `degraded_run` branch.

**Status (Latest batch)**:
- Stage-level health contracts are now materialized in `run_manifest.stage_health` with per-stage `status` + `issues` blocks (`stage_b_kg_context`, `stage_c_temporal`, `stage_d_causality`, `stage_e_evidence`, `stage_g_structuring`, `stage_i_archive`).
- Overall `pipeline_health` now folds stage-level red/yellow outcomes into final run routing signals.
- Stage-policy hooks are now configurable (`config.extra.stage_policy_hooks`) with per-stage/per-status actions (`analyst_review` | `validation_remediation` | `hard_stop`), and review hooks now emit `stage_policy_violations` plus `stage_remediation_playbooks` for actionable remediation routing.

**Still open**: runtime enforcement breadth remains policy-dependent — non-fatal hook absence/degradation paths (e.g., archive hook unavailable) currently degrade to yellow remediation rather than unconditional abort.

---

### 3.3 KG Population is a Silent Prerequisite with No Runtime Governance (CRITICAL — ✅ FIXED latest batch)

**Finding**: The KG is populated offline before the pipeline runs. The pipeline consumes it as if it were complete and current. But the KG population process has **no defined runtime governance**:
- No failure mode count check at Stage B (zero failure modes is not caught)
- No FMEA currency check (failure mode latency bounds may be stale — see April 20 review M2)
- No modification date check (KG may not reflect recent engineering changes)
- `kg_snapshot_version` was never populated (Gap 1, now Fixed) but its existence only records what version was loaded, not whether it is current relative to the event's date

**Specific risk**: For EVT-U2-2024-0847, if the expansion joint thermal fatigue failure mode was added to the KG after the event occurred but the pipeline is run against the KG snapshot that existed at event time, the FMEA data is correct but timing-unstable — a replay of the run at a later date would produce a different result.

**Recommendation**: Add to Stage B: (1) minimum failure mode count assertion (configurable floor per asset class); (2) `fmea_currency_warning` flag if any failure mode's `fmea_revision_date` predates the event by more than `fmea_staleness_threshold_days` (configurable, default 730); (3) explicit KG-to-event timestamp comparison in `kg_context.kg_snapshot_version`.

**Status (Latest batch)**:
- `Neo4jKGContextBuilder._fetch_failure_modes` now carries `fmea_revision_date` and `revision_date` into `kg_context.failure_modes[]`.
- `RCAReasoningOrchestrator._compute_kg_governance` adds runtime governance checks:
  - configurable minimum failure mode floor (`kg_min_failure_modes_default`, optional `kg_min_failure_modes_by_asset_class` override),
  - FMEA staleness check against `event.timestamp_start` (`fmea_staleness_threshold_days`, default 730),
  - KG snapshot-vs-event timestamp consistency check using `kg_snapshot_version`.
- `run_manifest` now includes `kg_governance` and pipeline/review routing consumes this signal via `degraded_run`.
- Strict hard-abort semantics are now enforced at source: when `strict_red_state_governance` and `hard_abort_on_kg_red_state` are enabled and `kg_governance.status == "red"`, the run is terminated immediately with an explicit abort reason in `run_status`.

---

### 3.4 Input Artifacts Have No Provenance or Currency Guarantee (MEDIUM — NEW)

**Finding**: `event.json`, `telemetry_summary.json`, `operational_context.json`, and `pm_compliance.json` are consumed at Stage A with key-presence validation only. No artifact carries a `generated_at` timestamp that is checked against the event's `timestamp_start`. A stale `telemetry_summary` assembled from a different time window would pass all current validation.

**Consequence**: A `pm_compliance.json` that reflects PM status as of 30 days before the event would not trigger any warning. A PM that lapsed the day before the event would be invisible.

**Recommendation**: Add to Stage A cross-artifact temporal consistency checks:
- `telemetry.window.end >= event.timestamp_start` (telemetry covers the event)
- `pm_compliance.assessment_date` within `pm_staleness_threshold_days` of `event.timestamp_start`
- `operational_context.as_of_timestamp` within `oc_staleness_threshold_hours` of `event.timestamp_start`

**Status (Post–April 21 batch)**: **Partial** — `orchestrators/input_guards.py` → `build_input_guards` writes non-blocking checks (telemetry window vs event start, optional PM/OC currency) onto `run_context["input_guards"]` (also persisted with `run_context` when saved). Hardening beyond warnings (e.g. strict aborts) is still open per full recommendation.

---

### 3.5 No Event Scoping Check — Single-Event Assumption Unchecked (MEDIUM — NEW)

**Finding**: The pipeline assumes it is analyzing a single, bounded event. No check verifies:
- That the `event_id` is not part of a multi-event sequence (overlapping events on the same asset)
- That the `telemetry.window` covers a single anomaly and not multiple distinct anomalies from different root causes
- That `operational_context.recent_alarms` do not contain alarms from a prior, unrelated event that are still open

**Consequence**: If two events overlap on the same asset (a common scenario in nuclear plants where a primary trip leads to secondary equipment stress), the pipeline will conflate their causal chains. The candidate set will include failure modes relevant to both events, scores will be diluted, and the synthesis will be unable to separate them.

**Recommendation**: Add to Stage A: check whether `event_id` appears in `operational_context.recent_alarms` (would indicate a second, concurrent event referencing this one). Emit `analyst_attention_flags["possible_multi_event_overlap"]` if detected.

**Status (Post–April 21 batch)**: **Partial** — `build_input_guards` can emit a `possible_multi_event_overlap` path when `recent_alarms` reference another `related_event_id` or suggest correlated events (heuristic). Broader scoping (telemetry window, multiple distinct anomalies) remains open.

---

### 3.6 Co-Pilot Interaction Model Not Defined (HIGH — NEW)

**Finding**: The pipeline is described as "decision support, not decision automation." The analyst is explicitly the final decision-maker. However, no **structured interaction protocol** defines when and how the analyst engages during the pipeline run:
- There is no mechanism for the analyst to provide mid-pipeline guidance (e.g., "this component is known to be out of the causal path — exclude it")
- The `analyst_override` mechanism operates post-synthesis only
- Stage G (Ishikawa) is optional but the pipeline does not ask the analyst whether to run it
- The `analyst_review_questions` in the rca_card are generated but there is no workflow for capturing answers

**Consequence**: The analyst's role is effectively "accept or override the machine conclusion." This is not RCA co-piloting — it is rubber-stamping or rejection. An RCA engineer with domain knowledge about this specific plant could dramatically improve Stage D's candidate quality if they could provide a hint at Stage B, but there is no mechanism to do so.

**Recommendation**: Define three analyst touch-points:
1. **Pre-Stage D**: Analyst reviews `kg_context` — can mark specific failure modes as "excluded_by_analyst" with justification
2. **Pre-Stage H**: Analyst reviews `causality_candidates v2` — can mark any candidate as "priority_review" to ensure it passes top-k selection regardless of score
3. **Post-Stage H**: Existing `analyst_override` — extend to include diff tracking as noted in April 20 review H4

---

## 4. Scoring System Integrity

### 4.1 Evidence Score Used in Pre-Evidence Filtering — Circular Reference (CRITICAL — ✅ FIXED Sprint 1)

**Finding** (referenced in `RCA_pipeline_stages.md` Stage D): Stage D applied a dual threshold: `composite ≥ 0.30 AND evidence_score ≥ 0.35`. The `evidence_score` at Stage D was a **pre-evidence proxy** — computed from KG document ranking signals before any document content is retrieved. Applying the 0.35 threshold to this proxy was a category error.

**Fix applied**: Added `minimum_pre_evidence_threshold = 0.10` to `CausalityEngineConfigV32`. Stage D candidate generation (failure mode and event analog paths) now uses `minimum_pre_evidence_threshold` (0.10). Stage F post-evidence check (`_refresh_candidate_confidence_and_thresholds`) continues to use `minimum_evidence_threshold` (0.35) against actual retrieved evidence scores. Files changed: `causality_engine_v32.py` lines 75, 295, 454.

---

### 4.2 `telemetry_anomaly_precedes` Seed Type Not Mapped in TOPOLOGY_BASE (HIGH — ✅ FIXED Sprint 2)

**Finding**: Stage B (proposed telemetry-driven expansion) would add components to the neighborhood with `seed_match_type = "telemetry_anomaly_precedes"`. Stage D's `TOPOLOGY_BASE` mapping controls the structural score for each `seed_match_type`. `telemetry_anomaly_precedes` is not defined in `TOPOLOGY_BASE`.

**Consequence**: All failure modes belonging to telemetry-expanded components would fall to the default structural score (~0.40). This is too low — a component with a preceding temporal anomaly is a stronger causal signal than structural proximity alone. The default score would cause telemetry-expanded candidates to systematically rank below structurally adjacent candidates that have no temporal evidence at all.

**Fix applied**: Refactored inline `if/elif` in `_structural_score_for_fm` into class-level `_SEED_STRUCTURAL_SCORES` dict. Added: `"telemetry_anomaly_precedes": 0.80`, `"telemetry_anomaly_simultaneous": 0.70`. File: `causality_engine_v32.py`.

---

### 4.3 Governance Score Asset-Level, Not Failure-Mode-Specific (MEDIUM — ✅ FIXED Sprint 6)

**Finding**: Per `RCA_workflow_april_2.md` §6: "governance score is currently computed at the asset level and applied uniformly to all candidates regardless of whether the failed PM item is related to the specific failure mode being scored." Per §13, this is marked `[FIXED]`.

**Residual gap**: Even with candidate-specific PM linkage, the governance dimension weight (0.10) was applied identically to all failure mode types.

**Fix applied (Sprint 6)**:
- `RuleBasedCausalityEngineV32._governance_weight_for_fm(superclass)` classifies FM superclass by keyword: maintenance-preventable keywords (bearing, seal, lubrication, wear, fouling, calibration, …) → weight 0.20; external-cause keywords (environmental, design, vendor, manufacturing, flood, …) → weight 0.02; default → 0.10. Maintenance classification takes precedence when both keyword sets match (e.g. "external corrosion" → 0.20, since corrosion is addressable by PM).
- `_combine_scores` now accepts optional `weights_override`; FM candidates pass their category weight at scoring time and `scores["governance_weight"]` persists it for Stage F re-compute in `_refresh_candidate_confidence_and_thresholds`.
- Tests: `unit_tests/test_causality_scoring.py` — 5 new NM3 tests.

---

### 4.4 Scoring Weights Unempirical and Not Sum-Constrained (MEDIUM — PARTIALLY FROM APRIL 20 REVIEW M1)

**Finding**: Per `RCA_workflow_april_2.md` §6: "weights not normalized: No constraint enforces that scoring weights sum to 1.0." The April 20 review noted that weights are engineering judgments without empirical calibration (M1 in that review).

**Combined concern**: The weights have two independent problems:
1. **No sum constraint** — a configuration change to one weight doesn't require adjusting others, allowing silent composite score inflation
2. **No empirical basis** — analyst override rates per scoring dimension are never tracked, so there is no feedback signal to improve calibration

**Concrete gaps**:
- `telemetry` weight (0.20) is applied even when `instrument_validity_flag` indicates the sensor is out-of-calibration
- `structural` weight (0.30) rewards KG topological proximity, not physical causal plausibility (April 20 review M5)
- `evidence` weight (0.20) at Stage D measures KG document registry completeness, not evidence quality

**Recommendation**: (1) Add JSON schema constraint: `sum(weights.values()) == 1.0` with validation error on deviation > 0.001. (2) Add `weight_override_flags` to `run_context` that allow per-run weight adjustments with analyst justification. (3) Track `dimension_disagreement_rate` (analyst override correlated to dimension) in run manifest for future calibration.

---

### 4.5 Two Independent Allen Computations Without Coordination (HIGH — ✅ PARTIALLY FIXED latest batch)

**Finding**: Allen interval relations are computed at **two separate stages**:
1. Stage B (proposed): timestamp pre-filter to decide neighborhood inclusion — a binary classification (include/exclude)
2. Stage C (TSKR): full temporal scoring against FMEA latency windows for each failure mode

Both compute the same fundamental relation (anomaly window vs event interval), but they use different inputs (Stage B uses raw anomaly timestamps; Stage C uses TSKR patterns built from all anomalies) and different granularity.

**Problem**: There is no handshake between these two computations. The Allen relation produced at Stage B for a given sensor is not stored and not consumed by Stage C. Stage C recomputes temporal relations independently. This creates two inconsistency risks:
- Stage B includes a component (relation = PRECEDES), but Stage C's TSKR pattern for the same sensor's failure mode shows FOLLOWS — now a component is in the neighborhood but its TSKR pattern contradicts its inclusion reason
- Stage B excludes a component (relation = FOLLOWS), but Stage C has no TSKR pattern for it — a silent coverage gap

**Fix applied (latest batch)**:
- `TSKRTemporalScorerV1` now reads `kg_context.out_of_boundary_anomalies[]` and builds a per-component Stage B Allen relation map.
- For each FM pattern, when the corresponding Stage B relation is `follows`, Stage C now sets `stage_b_temporal_contradiction: true` and forces `temporal_contradiction: true`.
- Pattern output now carries `stage_b_allen_relation` + `stage_b_temporal_contradiction` for traceability of the handshake path.

**Residual note**: This implements the Stage B -> Stage C contradiction handshake. Coverage quality still depends on population completeness of `out_of_boundary_anomalies` at neighborhood-build time.

---

### 4.6 Symptom Matching Not Implemented (MEDIUM — ✅ FIXED latest batch)

**Original finding**: `event.symptom_signature.symptom_types` and `anomaly_pattern` were documented as available but not reliably translated into scoring value.

**Fix applied (latest batch)**:
- `RuleBasedCausalityEngineV32._symptom_match_score` now uses normalized symptom text (`case`, `_`, `-`, punctuation) for both event and failure-mode sides before scoring.
- Pattern matching moved from strict equality to alias-aware similarity (`gradual_drift` vs `drift`, `surge` vs `spike`, etc.) via `_pattern_similarity_score`.
- `expected_symptoms` fallback parsing now supports both semicolon and comma delimiters, then normalizes terms before overlap scoring.
- Symptom-type overlap now combines phrase-level F1 with token-overlap fallback to avoid brittle misses from formatting variants.
- Telemetry rationale now reports pattern match/mismatch using the same normalized similarity path used in scoring.

**Verification**:
- Added/updated unit coverage in `unit_tests/test_symptom_match.py`:
  - alias normalization for anomaly patterns;
  - case/delimiter normalization for symptom type matching.

**Residual note**: This closes the implementation gap for deterministic symptom matching in Stage D scoring; future ontology-level symptom semantics can still improve precision but are no longer a blocker for this finding.

---

## 5. Data Quality, Traceability, and Provenance

### 5.1 Source Tier Metadata Not Yet Tagged in Chroma (HIGH — ✅ FIXED through latest batch)

**Finding**: `RCA_Data_Management_Strategy.md` defines a 6-tier evidence authority hierarchy:
```
plant_instance > plant_procedure > plant_fmea > plant_family > oe_iris > oe_adams
```
Stage 5B is responsible for tagging each embedded document with `source_tier` metadata. Stage F's refinement formula should incorporate authority weight. However:
1. `source_tier` tagging is not yet implemented in Stage 5B
2. The refinement formula in Stage F (`refined_evidence = 0.30×prior + 0.55×best_support + 0.15×context − 0.45×contradiction`) does not use `source_tier` at all

**Consequence**: A maintenance procedure snippet (tier 2 = plant_procedure) that says "check bearing clearance" scores identically to a closed FMEA finding (tier 3 = plant_fmea) that says "bearing degradation confirmed at this unit." The authority hierarchy designed to weight evidence by epistemic value is entirely inert.

**Fix applied (partial)**:
- Stage 5B: `source_tier: "plant_instance"` now set on both CR and WO Chroma document metadata (`cmms_context_builder.py`). Other document types (EDMS, FMEA, OE) carry their tiers when ingestion for those sources is implemented.
- Stage F: `_AUTHORITY_WEIGHTS` class constant added to `RuleBasedCausalityEngineV32` with all 6 tiers. Refinement formula now multiplies `support_score * authority_weight` where `authority_weight = _AUTHORITY_WEIGHTS.get(ev.get("best_source_tier"), 1.0)`. Backward compatible: defaults to 1.0 until the evidence retriever populates `best_source_tier` in per-candidate summaries (Sprint 3 work).

**Status update (latest batch)**: `ChromaEvidenceRetriever._build_candidate_evidence_summary` now emits `best_source_tier` from the strongest supporting snippet per candidate. Stage F authority weighting is now active in `RuleBasedCausalityEngineV32.refine_with_evidence`.

**Original recommendation** (for reference): (1) Stage 5B must set `source_tier` metadata on every Chroma document at ingestion. (2) Stage F refinement formula should incorporate `authority_weight[source_tier]` as a multiplier on `best_support_score`:
```python
authority_weight = {"plant_instance": 1.0, "plant_procedure": 0.8, "plant_fmea": 0.7,
                    "plant_family": 0.5, "oe_iris": 0.4, "oe_adams": 0.3}
refined_evidence = clamp(
    0.30 * prior + 0.55 * best_support * authority_weight[source_tier] + 0.15 * context
    - 0.45 * contradiction, 0.0, 1.0
)
```

---

### 5.2 `component_ids` Filter in Chroma Is a No-Op (HIGH — ✅ FIXED latest batch)

**Finding**: Per `RCA_workflow_april_2.md` §step E: "the Chroma metadata structure does not support direct list-valued filter on `component_ids`. The filter is built but silently ignored."

**Consequence**: For multi-component assets, Stage E retrieves all documents for the asset regardless of which component a candidate belongs to. A candidate for pump bearing degradation retrieves the same document set as a candidate for pump seal degradation. Retrieval precision is bounded by asset, not component.

**Compounding issue**: Because the filter is silently ignored (no log warning, no metadata flag in the evidence bundle), neither the analyst nor any downstream stage can detect that retrieval was component-unscoped. The `evidence_bundle` looks complete when it is actually under-targeted.

**Recommendation**: Short-term: add explicit warning in retrieval metadata `filter_applied: false, reason: "component_ids_not_supported"` so analysts know precision is reduced. Medium-term: restructure Chroma document metadata to support component-level scoping — embed a `primary_component_id` field at ingestion and filter on single values.

**Status (Latest batch)**:
- Candidate query plans now carry targeted `candidate_component_ids` (derived from candidate KG path + FM/component mapping), and Stage E filter construction now prefers these scoped IDs over whole-asset component lists.
- This materially narrows retrieval scope for candidate-support and contradiction checks.
- Chroma ingestion now writes scalar `primary_component_id` metadata (derived from `component_id/component_ids`) and Stage E retrieval now applies component scope as a true Chroma index-level filter (`primary_component_id IN [...]`).
- Compatibility fallback remains for legacy records missing `primary_component_id`: retrieval logs `legacy_post_filter` strategy and exposes `component_filter_mode: "index_filter_with_legacy_post_filter"` in provenance until collections are re-ingested.

---

### 5.3 BM25 Silent Degradation on Disk-Loaded Collections (HIGH — CONFIRMED FROM WORKFLOW)

**Finding**: Per `RCA_workflow_april_2.md` §step E: "When collections are loaded from disk (not ingested in the same process), `state.bm25_docs` is empty and hybrid retrieval degrades silently to dense-only. No warning is emitted."

**Consequence**: The primary use case — a pipeline run that loads a pre-archived Chroma collection — silently loses keyword-precision retrieval. For nuclear terminology (component tag numbers like `1-RC-P-001A`, procedure numbers like `OP-1.6.2`), keyword retrieval is essential. Dense-only retrieval cannot distinguish `1-RC-P-001A` from `2-RC-P-001A`.

**Recommendation**: Add `bm25_available: bool` to the `evidence_bundle.retrieval_metadata`. Set it to `False` when `state.bm25_docs` is empty. Include a `retrieval_quality_warning` in `analyst_attention_flags` when `bm25_available == False`: "Keyword retrieval unavailable — tag number matching is reduced. Re-ingest Chroma collection to restore hybrid retrieval."

---

### 5.4 FM-to-CR Matching Depends on NER Quality (MEDIUM — ✅ PARTIALLY FIXED latest batch)

**Finding**: Stage C's recurrence profile relies on matching past CR/WO records to failure modes. This matching is done via entity normalization (EntityNormalizer) — the NLP pipeline maps free-text CR descriptions to KG failure mode IDs. If NER quality is poor (surface form not in the trained vocabulary, or component alias not resolved), a past CR about "seal leakage" that corresponds to failure mode `FM-031` will not be matched and will not contribute to the recurrence profile.

**Consequence**: The recurrence score for `FM-031` will be artificially low, causing Stage D to under-rank a failure mode that has actually recurred. This is particularly dangerous for the reference test case's recurrence trap: "most recent similar event 18 months prior had fouling as confirmed root cause — recurrence scorer must not over-weight recency."

**Fix applied (latest batch)**:
- Stage C (`TSKRTemporalScorerV1`) now computes CR-match quality diagnostics in `tskr_patterns.summary`:
  - `total_cr_count`
  - `unmatched_cr_count`
  - `unmatched_cr_rate`
  - `high_cr_match_failure_rate` (threshold > 30%)
- Orchestrator now promotes high CR-match failure to `rca_card.executive_summary.analyst_attention_flags` so analysts see recurrence-quality risk before writeback decisions.

**Residual note**: This adds runtime observability/guardrails for NER miss risk; improving underlying FM-to-CR entity linking accuracy remains an open modeling/data task.

---

### 5.5 Allen Interval Endpoint Completeness — Open vs Closed Intervals (MEDIUM — ✅ PARTIALLY FIXED latest batch)

**Finding**: TSKR Allen interval relations depend on the endpoints of anomaly intervals. The `telemetry_summary.signals[].anomalies[]` schema does not specify whether `timestamp_start` and `timestamp_end` represent open or closed interval endpoints. The Allen algebra is well-defined only when endpoints are unambiguous.

**Specific case**: If an anomaly ends exactly at `event.timestamp_start` (same value), is the Allen relation PRECEDES (open interval, anomaly ended before event began) or MEETS (closed interval, anomaly ended at event start)? These produce different TSKR pattern labels and different latency calculations.

**Consequence**: For anomalies with timestamps near the event boundary, the Allen relation is indeterminate and the temporal score is unreliable. The latency window used in `latency_violation_type` computation may be off by the granularity of the timestamp field.

**Fix applied (latest batch)**:
- `schemas/telemetry_summary.json` now defines anomaly `interval_type` with values `closed | open | half_open_start | half_open_end`, defaulting to `closed`.
- `temporal_relations.allen_relation()` now accepts `interval_type` and applies endpoint semantics in boundary-touching cases.
- `TSKRTemporalScorerV1` now propagates anomaly `interval_type` into Allen classification (missing/invalid values normalize to `closed`).

**Residual note**: Pipeline semantics are now deterministic and explicit; external anomaly producers should still publish a documented convention in data strategy docs for end-to-end governance.

---

### 5.6 Historical Event Pool at Stage D Still Empty (HIGH — ✅ PARTIALLY FIXED latest batch)

**Finding**: Stage D generates two types of candidates: failure mode candidates and historical event analog candidates. The historical event pool is intended to come from `kg_context.past_events[]` — accepted RCA conclusions written back from closed CAP items. Per `RCA_Data_Management_Strategy.md` §10 and the updated `RCA_pipeline_stages.md`, **this list is currently empty** because CAP write-back has not been implemented.

**Consequence**: The `hypothesis_type: "past_event_analog"` candidate generation path is dead code in production. The recurrence profile in Stage C relies on CMMS records (via the `cmms_context` introduced in the Stage 5B/C revision), but Stage D cannot generate a past-event analog candidate even when recurrence is strong because the analog pool is empty.

**Compounding issue**: The April 20 review H1 noted that past event analogs as primary hypothesis are logically invalid. The issue is moot for now (empty pool), but when CAP write-back is implemented, this architectural issue becomes live. The fix (past events as evidence, not candidates) should be designed now before the pool is populated.

**Recommendation**: (1) Treat `kg_context.past_events[]` items as supporting evidence for the matching failure mode candidate, not as independent candidates. Add `past_event_support` as a sub-field of the recurrence score, not a separate hypothesis type. (2) Track CAP write-back implementation as a dependency for enabling recurrence-based reasoning.

**Status (Latest batch)**:
- `RCAReasoningOrchestrator` now augments `kg_context.past_events[]` with synthetic CMMS-derived historical events (from `cmms_context.cr_records[]` / `wo_records[]`) when available, so recurrence reasoning is no longer strictly blocked on CAP write-back.
- Injected entries are marked as `source: "cmms_context"` and tracked in `kg_context.seed_context.cmms_past_events_injected`.
- Added canonical historical-event support modeling in `kg_context.seed_context`:
  - `canonical_event_graph` (nodes/edges linking prior events to the current event),
  - `historical_support_channels` with explicit `mode: "support_channel_only"` and channel counts for recurrence support usage.

**Still open**: full CAP write-back population of the canonical event graph from accepted RCA/CAP outcomes (current graph bootstrap is CMMS + existing KG history only).

---

### 5.7 Two-Path Document Ingestion Not Yet Routed (MEDIUM — ✅ PARTIALLY FIXED latest batch)

**Finding**: `RCA_Data_Management_Strategy.md` defines two ingestion paths:
- **Path A**: Structured documents (CR/WO with structured JSON fields) → direct field extraction → Chroma
- **Path B**: Unstructured documents (PDF SOPs, FMEA PDFs, ECA narratives) → mdParser → chunk → Chroma

Stage 5B is responsible for routing each document type to the correct path. The current implementation does not differentiate between these paths — all documents go through the same chunking pipeline regardless of whether they have structured fields available.

**Consequence**: CR and WO records with `condition_assessment.as_found_condition` and `failure_mode_refs` structured fields are chunked as plain text, losing the structured signal that Path A would preserve. The document type semantics gap in Stage E (noted in workflow §step E and April 20 review) is partly caused by this ingestion routing failure — if structured fields are not preserved in Chroma metadata, keyword role classification cannot use them.

**Recommendation**: In Stage 5B's ingestion logic, check `doc_type` before chunking: if CR or WO, extract `as_found_condition`, `failure_mode_refs`, `extracted_causal_statements` as dedicated Chroma metadata fields (not embedded in chunk text). Stage E can then use these fields directly in role classification without keyword matching.

**Status (Latest batch)**:
- Stage 5B now routes CMMS CR/WO payloads through an explicit **Path-A structured metadata** path in `cmms_context_builder.get_chroma_documents` (`ingestion_path: "path_a_structured"`).
- Injected CMMS docs now include retrieval-critical identity fields (`doc_id`, `doc_type`, `asset_id`, `component_id/component_ids`) and flattened structured fields (`ca_as_found_condition`, `ca_as_left_condition`, `failure_mode_refs`, `failure_mode_refs_text`, `causal_statements_text`).
- `RCAReasoningOrchestrator` now injects CMMS document refs into `kg_context.documents[]` (`CMMS::CR::*` / `CMMS::WO::*`) so Stage E doc-id filtering does not silently exclude newly injected CMMS evidence.

**Residual note**: Path-B unstructured corpus still uses existing chunk pipelines; richer per-doc routing policies (for non-CMMS semi-structured sources) remain future hardening work.

---

## 6. Stage-by-Stage New Findings

### 6.1 Stage A — Additional Gaps

**A1 — No cross-artifact timestamp consistency** (MEDIUM — ⚠️ PARTIALLY ADDRESSED): `telemetry.window.end` is not verified to be ≥ `event.timestamp_start`. A telemetry summary from the wrong time window passes silently. *Post–April 21: non-blocking `input_guards` on `run_context` surface this class of issue; behavior is warning-style, not a hard block.*

**A2 — Severity not used to adjust evidence floor** (MEDIUM — ✅ FIXED Sprint 6): `event.severity` is now stored in `run_context.input_refs.event_severity` and consumed by `_compute_review_hooks`. `RuleValidatedRCASynthesizerV31._SEVERITY_SCORE_FLOORS` maps severity 1–5 to composite score floors (0.30 / 0.32 / 0.35 / 0.45 / 0.55). A `passed_severity_gate` flag blocks `writeback_ready` and adds a `degraded_reasons` message when the primary composite does not clear the floor; gate is bypassed when severity is absent. Tests: `unit_tests/test_review_hooks.py` — 5 new A2 tests.

**A3 — `output_dir` writability not verified at entry** (LOW — ✅ FIXED): Stage I will fail mid-pipeline if `output_dir` is not writable. Stage A should verify writeability and abort early with a clear error rather than failing at persistence time. *Post–April 21: `orchestrators/input_guards.assert_output_dir_writable` is invoked at pipeline start in `rca_reasoning_orchestrator.py` when using `FileArtifactStore`.*

---

### 6.2 Stage B — Additional Gaps

**B1 — `telemetry_anomaly_precedes` not in TOPOLOGY_BASE** (see §4.2 above — HIGH)

**B2 — Document window undifferentiated by type** (MEDIUM — ✅ FIXED Sprint 7): The ±90-day document ranking window is applied uniformly to all document types. FMEA documents, ECAs, and RCA conclusions are timeless engineering knowledge — they should not be subject to recency decay. CRs and WOs are time-bound operational observations — recency is appropriate. Applying recency decay to FMEA documents penalizes the most authoritative source in the corpus. **Fix**: `kg_context_builder.py` Cypher query now exempts `ECA` and `RCA` from the date-window filter (added to the `doc_type IN [...]` bypass alongside `SOP`, `FMEA`, `MANUAL`, `BULLETIN`); recency-proximity bonus removed for ECA/RCA in the Python enrichment loop — authority is independent of temporal proximity.

**B3 — KG document references include no `doc_type` filter at retrieval time** (LOW): `kg_context.documents[]` carries `doc_type` but all types are retrieved together. Downstream stages must filter by type themselves. A `doc_type_breakdown` count field would give Stage 5B a pre-fetch view of corpus composition.

---

### 6.3 Stage C — Additional Gaps

**C1 — TSKR keeps only first pattern per target_id** (HIGH — ✅ CONFIRMED FIXED): `_index_tskr_patterns()` uses `setdefault(target_id, []).append(p)` to collect all patterns; `_lookup_tskr_pattern()` returns `max(patterns, key=lambda p: p.get("confidence") or 0.0)`. No code change needed.

**C2 — Anomaly timestamp weighting all-equal** (MEDIUM — CONFIRMED WORKFLOW): High-severity anomalies should weight lag estimation more than low-severity or noisy signals. Currently all anomalies are equal.

**C3 — Stage B / Stage C Allen coordination gap** (HIGH — see §4.5 above)

**C4 — Tone discretization vocabulary undefined** (HIGH — ✅ PARTIALLY FIXED latest batch): TSKR now applies an explicit deterministic tone vocabulary (`npp_tone_v1`) in `tskr_temporal_scorer.py`:
- per-anomaly tone classification (`trip_band_persistent`, `alert_band_persistent`, `watch_band_persistent`, `transient_excursion`) is derived from severity + duration;
- summary/provenance now expose `tone_vocabulary_version`, `dominant_tone`, `tone_counts`, and `tone_calibration_uncertainty`;
- telemetry-support scoring now consumes tone classes rather than only raw severity labels.

**Residual note**: this closes the "undefined vocabulary" implementation gap, but threshold calibration remains heuristic and should be validated against plant/parameter-specific acceptance criteria.

---

### 6.4 Stage D — Additional Gaps

**D1 — Evidence threshold applied pre-evidence** (see §4.1 above — CRITICAL)

**D2 — `telemetry_anomaly_precedes` not in TOPOLOGY_BASE** (see §4.2 above — HIGH)

**D3 — Historical event pool empty until CAP write-back** (see §5.6 above — HIGH)

**D4 — `review_alternative_gap` uses only composite delta** (MEDIUM — ✅ FIXED): The review alternative rescue condition checks whether a filtered candidate's composite score is within `review_alternative_gap` (0.10) of the single passing candidate. It does not check `evidence_posture`. A candidate that is `evidence_posture: "contradicted"` should not be rescued as a review alternative regardless of score proximity — a contradicted candidate needs analyst attention, not elevation to the review set. *Post–April 21: `RuleBasedCausalityEngineV32._eligible_review_alternative` returns `False` when `evidence_posture == "contradicted"`; unit coverage in `unit_tests/test_review_alternative.py`.*

---

### 6.5 Stage E — Additional Gaps

**E1 — `component_ids` filter no-op** (see §5.2 — ✅ FIXED latest batch): `storage/chroma_store.py` now indexes scalar `primary_component_id` metadata at ingest and applies true index-level Chroma filtering for component-scoped queries; `evidence_bundle.provenance.component_filter_mode` now distinguishes `index_filter` vs `index_filter_with_legacy_post_filter` compatibility mode.

**E2 — BM25 silent degradation** (see §5.3 — ✅ Fixed Sprint 4): `LOGGER.warning` emitted when `retrieval_mode == "dense_only"`; `retrieval_quality_warning` and `bm25_available` fields added to `evidence_bundle.provenance`.

**E3 — No evidence retrieval for `out_of_boundary_anomalies`** (HIGH — ✅ FIXED latest batch): Stage E now adds targeted `query_type: "out_of_boundary"` plans (intent `kg_gap_investigation`) for entries in `kg_context.out_of_boundary_anomalies[]`, so KG-gap components are explicitly searched in the evidence pass.

**E4 — Contradicting evidence retrieval not structurally specified** (MEDIUM — ✅ PARTIALLY FIXED latest batch): Stage E now adds a structural contradiction path in `evidence_retriever._assess_hit_against_candidate`:
- explicit causal-attribution cues (`root cause`, `caused by`, `determined to be`, etc.) with **low candidate alignment** in `candidate_contradiction` queries now trigger `structural_contradiction_hit`;
- this adds deterministic contradiction boost and traceability fields (`structural_contradiction_hit`, `structural_contradiction_score`) on evidence assessment output;
- unit coverage added in `unit_tests/test_evidence_scorer.py` for both positive and negative structural-contradiction paths.

**Residual note**: structural contradiction now also checks structured `failure_mode_refs` / structured causal text alignment to reduce false contradiction hits; full ontology-backed contradiction typing remains open.

**E5 — Keyword role classification uses brittle surface matching** (MEDIUM — CONFIRMED WORKFLOW): "Loss of lubrication" vs "lube oil degradation" are semantically equivalent but won't match the same keyword set. This is a known limitation for nuclear terminology.

---

### 6.6 Stage F — Additional Gaps

**F1 — `contradicted` evidence posture missing** (HIGH — ✅ FIXED Sprint 2): When `n_support == 0` and `n_contra > 0`, the posture classification previously fell through to `"weak"` for weak contradictions (contradiction_score < 0.45). Added condition: `if support_score == 0.0 and contradiction_score > 0.0: return "contradicted"` before the existing `>= 0.45` threshold check. The `>= 0.45` condition now handles the case where there is some support but contradiction dominates. File: `causality_engine_v32.py` `_evidence_posture()`.

**F2 — Authority tier not used in refinement formula** (see §5.1 — HIGH)

**F3 — `review_alternative_gap` on composite only** (see D4 — ✅ FIXED — same change as D4)

---

### 6.7 Stage H — Additional Gaps

**H1 — Top-k selection ignores `review_required` flag** (HIGH — ✅ FIXED): Stage H selects the top-5 candidates by composite score for synthesis. If a candidate outside the top-5 has `evidence_posture: "contradicted"` or `temporal_contradiction_flag: True`, it is silently dropped. An engineer reviewing a later RCA that asks "was hypothesis X considered?" will find no record of it in the rca_card. The top-k selection must retain all candidates with `review_required: True` regardless of rank. *Post–April 21: `refine_with_evidence` in `causality_engine_v32.py` sets `review_required` (near-ties, contradicted posture, temporal contradiction); `RCASynthesizerV31._select_candidates` in `synthesis/rca_synthesizer_v31.py` extends the synthesis set with up to `max_synthesis_extra_review_candidates` (default 8) such rows in score order after the top-N pass.*

**H2 — Evidence not balanced across candidates in synthesis** (MEDIUM — ✅ PARTIALLY FIXED latest batch): Fallback synthesis now uses balanced evidence selection (`_balanced_fallback_evidence`) before card assembly:
- preserves score ranking, but first attempts to include per-card-candidate evidence slices (supporting and contradicting where available) for primary + top alternatives;
- prevents one-sided primary-only evidence rows when alternative-linked snippets are available in the selected evidence pool;
- unit coverage added in `unit_tests/test_synthesizer_fallback.py`.

**Residual note**: deterministic fallback balancing remains in place, and LLM-path cards now receive a deterministic post-pass that backfills missing alternative-linked evidence rows when available in the evidence pool. Prompt quality still influences narrative composition.

**H3 — `minimum_evidence_gate` string comparison bug** (MEDIUM — ✅ CONFIRMED FIXED / Sprint 3): Current code uses `.strip().lower()` on `support_role` and `float()` cast on `composite_score`. Regression test `test_gate_whitespace_padded_support_role_passes` locks in the fix.

**H4 — LLM hallucination gate is warning, not error** (HIGH — ✅ FIXED): A hallucinated candidate ID from the LLM path that passes schema validation is flagged as a warning. It should be a hard error that triggers the fallback path. With the current behavior, a hallucinated candidate could reach the rca_card if the cross-artifact check tolerates it. *Post–April 21: if `primary_hypothesis.candidate_id` is present and not in the union of input candidate ids (or `"NONE"`), the LLM `rca_card` is **discarded** and the deterministic path is used — see `synthesis/rca_synthesizer_v31.py` (LLM card assembly).*

**H5 — `writeback_ready: False` always; `requires_human_review: True` always** (CRITICAL — ✅ FIXED Sprint 1): Three interlocked fixes were applied:
1. **Confidence cap removed** (`rca_synthesizer_v31.py`): `and not fallback_used` removed from the high-confidence condition; `if fallback_used: cap at medium` block removed. The deterministic fallback path can now produce `"high"` confidence when evidence warrants it.
2. **`not fallback_used` gate removed from `writeback_ready`** (`rca_reasoning_orchestrator.py` `_compute_review_hooks`): the deterministic fallback is production-quality and should be writeback-eligible when all other gates pass.
3. **`requires_human_review` computed, not hardcoded** (`rca_reasoning_orchestrator.py`): replaced `"requires_human_review": True` with a derived expression based on `decision_required`, `all_claims_cited`, `passed_minimum_evidence_gate`, `outputs_ok`, and `decision_status`. Existing test `test_writeback_blocked_when_fallback_used` updated to `test_writeback_allowed_when_fallback_used_and_all_conditions_met`; `test_requires_human_review_always_true` updated to `test_requires_human_review_computed_from_conditions`.

---

### 6.8 Stage I — Additional Gaps

**I1 — No `run_complete` sentinel file** (HIGH — ✅ Fixed Sprint 3): `run_status.json` is written at run start with `run_complete: false` and overwritten at run end with `run_complete: true`. Stage J can now distinguish partial from complete runs.

**I2 — Writes are not atomic** (HIGH — ✅ Fixed Sprint 3): `FileArtifactStore._write_atomic()` uses `tempfile.mkstemp` + `Path.replace()` (POSIX-atomic rename). Readers can never observe a partial write.

**I3 — Chroma archive failure not caught** (MEDIUM — ✅ PARTIALLY FIXED latest batch): Stage I now executes an explicit Chroma archive hook (when available) and records `chroma_archive` status in manifest/pipeline health. Archive exceptions are now promoted to `pipeline_health: red` with remediation routing (`next_step: validation_remediation`), and strict policy (`hard_fail_on_chroma_archive_error`) hard-aborts the run with `run_status.aborted=true` + reason.  
Remaining gap: archive hook availability is backend-dependent; when no archive hook is exposed by the evidence store, Stage I reports yellow degradation rather than red failure.

---

### 6.9 Stage J — Additional Gaps

**J1 — Schema validation late-binding** (CRITICAL — ✅ FIXED Sprint 1 + Sprint 5): `cmms_context` was the one artifact with a schema (`cmms_context.json`) that was saved without validation. Fixed: added `cmms_context` to `RCAArtifactValidator.CORE_ARTIFACTS` (`schema_validator.py`) and changed the bare `artifact_store.save` in `build_cmms_context()` to `_validate_and_persist(optional=True)`. Schema errors in `cmms_context` are now caught and logged at Stage 5B.

The broader architectural issue is now fully addressed: all intermediate artifacts (`kg_context`, `tskr_patterns`, `causality_candidates`, `evidence_bundle`, `rca_card`) already went through `_validate_and_persist` — this was already correct. `run_context.json` and `run_manifest.json` schemas were created and both artifact types wired into `CORE_ARTIFACTS` in Sprint 5 (NC5).

**J2 — `run_complete` check absent** (MEDIUM — ✅ Fixed Sprint 4): `FileArtifactStore.is_run_complete(run_id)` reads `run_status.json` and returns `True` only for completed runs. `load(run_id, artifact_name)` provides safe external artifact access. External callers (notebooks, replay scripts) should call `is_run_complete()` before loading or re-validating artifacts.

**J3 — `next_step` routing not wired to analyst workflow** (MEDIUM — ✅ PARTIALLY FIXED latest batch): Orchestrator now consumes `review_hooks.next_step` to build workflow dispatch payloads:
- `RCAReasoningOrchestrator._build_workflow_dispatch` maps `next_step` to a target queue (`writeback`, `analyst_review`, `validation_remediation`) and emits dispatch metadata (`dispatch_ref`, `target_queue`, `dispatched_at`);
- dispatch payload is persisted as `workflow_dispatch` artifact (when enabled) and summarized under `run_manifest.review_hooks.workflow_dispatch` + `run_manifest.artifacts.workflow_dispatch`;
- optional transport execution hook (`workflow_dispatch_adapter.dispatch`) now records `transport_status`, `transport_ref`, and `transport_error` for external handoff visibility.

**Residual note**: adapter transport wiring is now first-class, but concrete production transport implementations (email/CMMS API/ticketing connectors) remain integration work.

---

## 7. Schema and Implementation Defects (Known)

The following defects are confirmed open per `RCA_workflow_april_2.md` §4 and §15. They are collected here for completeness:

| # | Defect | Severity | Status |
|---|--------|----------|--------|
| S1 | `confidence_label` case mismatch (uppercase in schema, lowercase in runtime) | HIGH | ✅ Confirmed already fixed — all schemas use lowercase enum |
| S2 | `kg_context` dual schema (v2 schema differs from runtime output) | HIGH | ✅ Fixed Option B — deleted orphaned `orchestrators/kg_context.json`; `schemas/kg_context.json` is sole canonical schema |
| S3 | TSKR index keeps only first pattern per target_id | HIGH | ✅ Confirmed already fixed — `_index_tskr_patterns` collects all patterns per target; `_lookup_tskr_pattern` selects highest-confidence via `max()` |
| S4 | `component_ids` filter is a no-op (silent) | HIGH | ✅ Fixed latest batch — scalar `primary_component_id` indexed at ingest + index-level component filter in retrieval; legacy fallback retained until re-ingest |
| S5 | BM25 unavailable on disk-loaded collections (silent) | HIGH | ✅ Fixed Sprint 4 — LOGGER.warning + retrieval_quality_warning in provenance |
| S6 | `writeback_ready: False` always | CRITICAL | ✅ Fixed Sprint 1 — `not fallback_used` gate removed; see §6.7 H5 |
| S7 | `requires_human_review: True` always | CRITICAL | ✅ Fixed Sprint 1 — now computed from quality gates; see §6.7 H5 |
| S8 | LLM hallucination gate is warning not error | HIGH | ✅ Fixed Post–April 21 — unknown `primary_hypothesis.candidate_id` discards LLM card; see §6.7 H4 |
| S9 | Weights not enforced to sum to 1.0 | MEDIUM | ✅ Fixed Sprint 2 — `__post_init__` raises `ValueError` if `abs(sum - 1.0) > 0.001` |
| S10 | `_passes_minimum_evidence_gate` string comparison | MEDIUM | ✅ Confirmed already fixed — `.strip().lower()` and `float()` cast both present; regression test added (`test_gate_whitespace_padded_support_role_passes`) |

---

## 8. Regulatory and Operational Compliance Gaps

### 8.1 Contributing Causes Not Representable (CRITICAL — APRIL 20 C2 — ✅ FIXED latest batch)

`rca_card` now includes a first-class `contributing_causes[]` block in schema and synthesis output. `RuleValidatedRCASynthesizerV31` normalizes LLM-supplied contributing causes and emits deterministic contributing-cause rows in fallback synthesis for near-primary alternatives. This closes the representability gap (while preserving analyst review requirements for acceptance/writeback).

### 8.2 Safety Function Impact Never Reaches rca_card (CRITICAL — APRIL 20 C5 — ✅ FIXED latest batch)

Safety-function impact now propagates to the synthesized card. `RuleValidatedRCASynthesizerV31` applies safety-significance post-processing that:
- inspects `affected_safety_functions` on the selected primary candidate,
- appends an explicit safety-significance entry to `executive_summary.analyst_attention_flags`,
- escalates `recommended_actions[].priority` to at least `high` (and to `critical` for reactor-protection-class impact).
- normalizes alias forms (e.g., `RPS`, `ESFAS`, `ECCS`, underscore/hyphen variants) so safety-tier mapping is resilient to KG naming variations.

This closes both the C5 propagation gap and the C4 priority-derivation gap for action prioritization.

### 8.3 Barrier Analysis Completely Absent (HIGH — WORKFLOW GAP 3 — ✅ PARTIALLY FIXED latest batch)

A deterministic `barrier_analysis` artifact now exists and is persisted each run. The orchestrator computes barrier state from KG safety functions, candidate-level `affected_safety_functions`, and optional Ishikawa process/procedure signals, then propagates:
- top-level `barrier_analysis` into `run_manifest`,
- compact `barrier_analysis` summary into `rca_card`.

Depth-pass extension (this session): barrier significance now influences candidate scoring and action shaping:
- `causality_engine_v32` now derives a `barrier_signal` from candidate-linked safety functions and applies a bounded structural score delta (FM and historical-analog paths),
- candidate `scores.barrier_signal` and structural rationale now explicitly expose barrier contribution,
- synthesizer action normalization now adds barrier-weighted rationale text and applies a degraded-barrier-count priority floor.

This closes the prior "no barrier concept in outputs" gap and partially closes "barriers do not influence ranking/action behavior."

**Still open**: deeper defense-in-depth ontology in KG semantics (barrier types, failure states, dependencies) and PRA-calibrated risk coupling (see §8.4).

### 8.4 Risk Significance Score Not Derived (HIGH — ✅ PARTIALLY FIXED latest batch)

`affected_safety_functions` are now converted into a deterministic `risk_significance_scalar` and tier in Stage D/F (`causality_engine_v32.py`), with bounded governance adjustment (`governance_risk_delta`) carried in candidate `scores`.

Status (latest batch):
- Added deterministic safety-function risk mapping (`critical/high/medium`) with alias-hardened text normalization and bounded scalar derivation.
- Candidate scoring now stores `scores.risk_significance_scalar`, `scores.risk_significance_tier`, `scores.governance_base`, and `scores.governance_risk_delta`.
- Stage F refinement now re-applies the risk-governance adjustment so composite scoring remains risk-aware after evidence updates.
- Stage H action shaping now uses risk significance in recommended action priority flooring and rationale text.
- Analyst-facing card sections now surface risk context (executive attention flags + `primary_hypothesis.why_primary`/`uncertainties`) without schema changes.

Still open:
- PRA-coupled calibration of scalar/tier thresholds and uncertainty bounds (current mapping remains deterministic heuristic).

### 8.5 Change Analysis Unstructured (MEDIUM — WORKFLOW GAP 7)

`operational_context.nearby_maintenance` influences CCF scoring but is not surfaced as a named "change analysis" artifact. A change analysis — systematically correlating maintenance, setpoint changes, and configuration changes against the event timeline — is a first-class RCA methodology requirement (INPO AP-923 §3.8) that the pipeline does not produce.

### 8.6 Analyst Override Has No Audit Trail Diff (HIGH — APRIL 20 H4)

Reiterated for regulatory emphasis: the `analyst_override` artifact records the final decision but not the delta from the system recommendation. Nuclear plant CAP audit trails must show both states and the analyst's justification. See April 20 H4 for recommended `diff` structure.

### 8.7 INPO AP-913 Completeness Criteria Not Mapped (MEDIUM — ✅ FIXED latest batch)

`run_manifest` now includes an `ap913_completeness` block with booleans for:
- `root_cause_identified`
- `direct_cause_identified`
- `contributing_causes_identified`
- `extent_of_condition_assessed`
- `effectiveness_review_defined`

The checklist is computed in `RCAReasoningOrchestrator._compute_ap913_completeness()` and persisted each run.

**Current shape**:
```json
{
  "root_cause_identified": true,
  "direct_cause_identified": true,
  "contributing_causes_identified": false,
  "extent_of_condition_assessed": false,
  "effectiveness_review_defined": false
}
```

---

## 9. Cross-Cutting Observations

### 9.1 Common Cause Failure Architecturally Invisible

The KG models component containment and connectivity but does not model **shared cause mechanisms**: shared cooling medium, shared power supply, shared maintenance history, shared vendor batch. The CCF scoring in Stage D uses proximity heuristics (components in the same sub-system scored for CCF), but true CCF detection requires a common-cause group taxonomy that does not exist in the KG schema. A multi-component failure caused by a contaminated lube oil batch is undetectable by the current system.

### 9.2 Recurrence Not Normalized by Equipment Population

Recurrence scoring counts events and computes inter-event intervals but does not normalize by equipment population or operating hours. A failure mode that occurred once in 40 years of operation on a single component is treated the same as one that occurred once in 3 months. See April 20 review M7.

### 9.3 Ishikawa Taxonomy vs Nuclear Standard Alignment

The Stage G Ishikawa categories (`equipment_hardware`, `process_procedure`, `measurement_instrumentation`, `environment_operating_context`, `maintenance_human_factors`) differ from INPO AP-923's 4M+E taxonomy (`Man`, `Machine`, `Method`, `Material`, `Environment`). Neither is wrong, but the mismatch means the Ishikawa output cannot be directly compared to industry-standard RCA reports without relabeling.

### 9.4 No Cross-Run Comparison or Fleet Awareness

Each pipeline run is fully isolated. There is no mechanism to compare the current run's candidates or evidence against prior runs on the same asset or same failure mode class. The recurrence profile in Stage C is the only cross-run signal, and it is bounded by CMMS retention. A systematic failure mode appearing at multiple units within a short window (a classic fleet-wide OE trigger) is invisible to any single run.

### 9.5 Stage G Optional Without Quality Impact Assessment

Stage G (Ishikawa) is optional. When it is skipped, Stage H runs without the Ishikawa matrix. The rca_card carries no indication that Stage G was skipped and no assessment of what causal branches were therefore not systematically examined. An analyst reviewing the rca_card cannot tell whether the absence of a `maintenance_human_factors` entry means "maintenance was investigated and found not contributory" or "Stage G was not run."

**Recommendation**: Add `ishikawa_run: bool` and `ishikawa_skip_reason` to `run_manifest`. When `ishikawa_run == False`, add an `analyst_attention_flag`: "Ishikawa structuring was not performed — human performance and organizational factor branches were not systematically evaluated."

**✅ FIXED Sprint 7**: `_apply_ishikawa_skip_attention_flag` static method added to orchestrator; called in `run()` after other `_apply_*` hooks. `pipeline_config.ishikawa_run: bool` and `pipeline_config.ishikawa_skip_reason: str|null` added to both the runtime manifest and `schemas/run_manifest.json`. Deduplication guard prevents double-injection. Tests: `test_ishikawa_skip_flag_injected_when_matrix_absent`, `test_ishikawa_skip_flag_not_injected_when_matrix_present`, `test_ishikawa_skip_flag_not_duplicated_on_double_call`, `test_manifest_pipeline_config_has_ishikawa_run_false_when_disabled`, `test_manifest_pipeline_config_has_ishikawa_run_true_when_matrix_present`.

### 9.6 OE Documents Require Fleet-Level Linkage Not Yet Supported

Per `RCA_workflow_april_2.md` §OE section: OE documents (INPO OE, NRC Information Notices, EPRI Technical Reports) have fundamentally different semantics from plant-internal documents. Their linkage is through failure mode similarity and system type, not plant topology. Recency decay should not apply. None of these requirements are implemented — OE documents, if ingested, would go through the same pipeline as plant-internal documents, losing their distinct epistemic role.

---

## 10. Priority Matrix and Remediation Roadmap

### New Critical Findings (must address before production deployment)

| ID | Finding | Section |
|----|---------|---------|
| NC1 | ✅ Strictly sequential pipeline — rank-inversion reentry hook + bounded auto re-entry loop (latest batch) | §3.1 |
| NC2 | ✅ KG population silent prerequisite — `_compute_kg_governance` + hard-abort on red state (latest batch) | §3.3 |
| NC3 | ✅ Evidence threshold applied pre-evidence (circular reference) | §4.1 |
| NC4 | ✅ `writeback_ready` and `requires_human_review` permanently broken | §6.7 H5 |
| NC5 | ✅ Schema validation late-binding — `cmms_context` fixed; `run_context.json` + `run_manifest.json` created; both wired into `CORE_ARTIFACTS` | §6.9 J1 |
| NC6 | ✅ Contributing causes — `contributing_causes[]` added to rca_card schema + synthesis (Post–April 21 batch) | §8.1 |

### New High Findings (investigation quality and audit defensibility)

| ID | Finding | Section |
|----|---------|---------|
| NH1 | Silent failure propagation — no mid-pipeline error protocol | §3.2 |
| NH2 | Co-pilot interaction model not defined | §3.6 |
| NH3 | ⚠️ Allen Stage B/Stage C coordination — partial (handshake flagging implemented; source coverage still dependent) | §4.5 |
| NH4 | `telemetry_anomaly_precedes` not in TOPOLOGY_BASE | §4.2 |
| NH5 | Historical event pool empty until CAP write-back | §5.6 |
| NH6 | Source tier metadata not tagged in Chroma | §5.1 |
| NH7 | ✅ `component_ids` filter no-op (closed latest batch) | §5.2 |
| NH8 | BM25 silent degradation (confirmed) | §5.3 |
| NH9 | No evidence retrieval for `out_of_boundary_anomalies` | §6.5 E3 |
| NH10 | `contradicted` evidence posture missing | §6.6 F1 |
| NH11 | ✅ Top-k selection / `review_required` — **fixed** (Post–April 21 batch) | §6.7 H1 |
| NH12 | ✅ LLM primary candidate id — **fixed** (reject unknown id; use deterministic path) (Post–April 21 batch) | §6.7 H4, S8 |

### New Medium Findings (correctness and data quality)

| ID | Finding | Section |
|----|---------|---------|
| NM1 | ⚠️ Input currency — **partial** (`run_context.input_guards` warnings) | §3.4 |
| NM2 | ⚠️ Event scoping / overlap — **partial** (heuristic in `input_guards`) | §3.5 |
| NM3 | ✅ Governance weight now FM-category-specific via `_governance_weight_for_fm` (Sprint 6) | §4.3 |
| NM4 | Scoring weights unempirical, no sum constraint | §4.4 |
| NM5 | ✅ Symptom matching implemented and normalized (closed latest batch) | §4.6 |
| NM6 | ⚠️ FM-to-CR matching quality telemetry + analyst flagging — partial | §5.4 |
| NM7 | ⚠️ Allen interval endpoint completeness — partial (`interval_type` schema + scoring semantics added) | §5.5 |
| NM8 | ⚠️ Two-path ingestion routing + structured Path-A metadata wired for CMMS CR/WO — partial | §5.7 |
| NM9 | ⚠️ Tone vocabulary (`npp_tone_v1`) + summary/provenance traceability added — partial | §6.3 C4 |
| NM10 | ✅ `review_alternative` vs `contradicted` — **fixed** (same as D4) | §6.4 D4, §6.6 F3 |
| NM11 | ⚠️ Structural contradiction scoring tightened with structured FM-ref alignment — partial | §6.5 E4 |
| NM12 | ⚠️ Evidence balancing now applies to fallback + deterministic LLM post-pass — partial | §6.7 H2 |
| NM13 | ⚠️ Chroma archive failure not caught — partial (strict fail path implemented; backend hook availability remains) | §6.8 I3 |
| NM14 | ⚠️ `next_step` dispatch + optional adapter transport status/ref tracking — partial | §6.9 J3 |
| NM15 | ✅ INPO AP-913 completeness criteria mapped (closed latest batch) | §8.7 |

### Previously Identified (April 20 Review) — Status

| ID | Finding | Prior Status | Updated Notes |
|----|---------|-------------|---------------|
| C1 | Closed-world KG assumption | Open | NC3 (evidence threshold) exacerbates this |
| C2 | Single primary cause architecture | Open | Also NC6 (contributing causes) |
| C3 | Confidence always capped at medium | Open | Also NC4 (writeback broken) |
| C4 | Action priority not from safety significance | ✅ Fixed latest batch | Safety-significance post-processing now elevates action priority; see §8.2 |
| C5 | Safety function never flows to rca_card | ✅ Fixed latest batch | Safety-significance now surfaced in analyst flags + action priority; see §8.2 |
| H1 | Past event analog as primary hypothesis | Open | Moot until CAP write-back (NH5) |
| H2 | Evidence retrieval cannot rescue filtered candidates | Open | Root cause is NC3 |
| H3 | Score evolution not named artifact | Open | ✅ Fixed Sprint 3 — `scoring_evolution.json` persisted as dedicated artifact in `run()` when pre-refine snapshot exists |
| H4 | Analyst override no diff record | Open | ✅ Fixed Sprint 5 — `_compute_primary_diff` added; `primary_diff` field in `override_record` captures before/after delta for `primary_candidate_change` overrides |
| H5 | Evidence excerpts are summaries | Open | Still open |
| H6 | Recommended actions not validated against posture | Open | ✅ Fixed Sprint 5 — `_POSTURE_WARNINGS` dict + `posture_warning` field added to every action row in `_normalize_recommended_actions`; triggered for `contradicted` and `no_data` postures |

---

## 11. Suggested Fix Sequencing

The following sequencing minimizes dependency conflicts and maximizes early risk reduction:

**Sprint 1 — Break the permanently-broken signals:** ✅ COMPLETE (April 21, 2026)
1. ✅ Remove fallback confidence cap — `rca_synthesizer_v31.py`: removed `and not fallback_used` from high-confidence condition and `if fallback_used: cap at medium` block
2. ✅ Remove `not fallback_used` from `writeback_ready` gate and compute `requires_human_review` from quality conditions — `rca_reasoning_orchestrator.py` `_compute_review_hooks`
3. ✅ Relax Stage D evidence threshold from 0.35 to 0.10 via `minimum_pre_evidence_threshold` — `causality_engine_v32.py`; Stage F retains 0.35 on post-evidence scores
4. ⚠️ Progressive validation for `cmms_context` — `schema_validator.py` (added to CORE_ARTIFACTS) + `rca_reasoning_orchestrator.py` (bare save → `_validate_and_persist`). `run_context` and `run_manifest` schemas deferred to Sprint 2.

**Sprint 2 — Scoring integrity:** ✅ COMPLETE (April 21, 2026)
4. ✅ `contradicted` posture for zero-support + any-contradiction case — `causality_engine_v32.py` `_evidence_posture()`
5. ⚠️ `source_tier` metadata added to CMMS Chroma docs (`cmms_context_builder.py`); `_AUTHORITY_WEIGHTS` constant + authority multiplier wired in Stage F refinement formula — dormant until evidence retriever populates `best_source_tier` (Sprint 3)
6. ✅ `telemetry_anomaly_precedes` (0.80) + `telemetry_anomaly_simultaneous` (0.70) added; `_structural_score_for_fm` refactored to `_SEED_STRUCTURAL_SCORES` dict — `causality_engine_v32.py`
7. ✅ Weight sum constraint: `__post_init__` raises `ValueError` if `abs(sum(weights) - 1.0) > 0.001` — `causality_engine_v32.py`

**Option B — Schema audit pass:** ✅ COMPLETE (April 21, 2026)
8. ✅ S1 (`confidence_label` case mismatch) — confirmed already fixed; no code change needed
9. ✅ S2 (dual `kg_context` schema) — deleted orphaned `orchestrators/kg_context.json`; `schemas/kg_context.json` is sole canonical schema
10. ✅ S3 (TSKR first-pattern-only) — confirmed already fixed via `setdefault + max(confidence)`; no code change needed
*(NC6 contributing causes closed in Post–April 21 batch; NH2 analyst touch-points remain open)*

**Sprint 3 — Traceability and persistence integrity:** ✅ COMPLETE (April 21, 2026)
11. ✅ S10 regression test (`test_gate_whitespace_padded_support_role_passes`) — confirmed already fixed; test locks in `.strip().lower()` normalisation (H3 / S10)
12. ✅ `scoring_evolution.json` persisted as dedicated named artifact in `run()` when pre-refine snapshot exists (H3)
13. ✅ `run_status.json` sentinel written at run start (`run_complete: false`) and sealed at run end (`run_complete: true`) — Stage J can distinguish partial from complete runs (I1)
14. ✅ `FileArtifactStore._write_atomic()` — `tempfile.mkstemp` + `Path.replace()` POSIX-atomic rename; readers never observe partial writes (I2)
*(NC6 contributing causes closed in Post–April 21 batch; NH2 analyst touch-points remain open)*

**Sprint 4 — Data quality and corpus integrity:** ✅ COMPLETE (April 21, 2026)
15. ✅ `FileArtifactStore.is_run_complete(run_id)` + `load(run_id, artifact_name)` — safe external artifact access; callers check completion before loading (J2)
16. ✅ BM25 silent degradation — `LOGGER.warning` emitted when `retrieval_mode == "dense_only"`; `retrieval_quality_warning` and `bm25_available` fields added to `evidence_bundle.provenance` (NH8 / S5 / E2)
17. ⚠️ `component_ids` post-filter — confirmed pre-existing; `LOGGER.warning` added when active; `component_filter_mode: "post_filter"` in provenance. True index-level filter is medium-term work (NH7 / S4 / E1)
*(NH3 Allen coordination has now advanced via Stage B->C contradiction handshake.)*

**Sprint 5 — rca_card quality (Option B):** ✅ COMPLETE (April 21, 2026)
16. ✅ Analyst override audit trail diff — `_compute_primary_diff` + `primary_diff` in `override_record` (H4)
17. ✅ Posture-aware recommended actions — `_POSTURE_WARNINGS` + `posture_warning` on every action row (H6)
18. ✅ `run_context.json` + `run_manifest.json` schemas created; both added to `CORE_ARTIFACTS` (NC5)

**Sprint 6 — Scoring quality:** ✅ COMPLETE (April 22, 2026)
19. ✅ Severity-adjusted evidence floor — `_SEVERITY_SCORE_FLOORS` + `passed_severity_gate` in `_compute_review_hooks`; `event_severity` stored in `run_context.input_refs` (A2)
20. ✅ FM-category governance weight — `_governance_weight_for_fm` + `weights_override` in `_combine_scores`; weight persisted in `scores["governance_weight"]` for Stage F re-compute (NM3)

**Post–April 21 batch — Regulatory completeness:** ✅ COMPLETE (April 21, 2026)
21. ✅ Contributing causes (`contributing_causes[]` in rca_card schema + synthesis) (NC6)
22. ✅ Safety function propagation to rca_card (C5)
23. ✅ Safety significance override on recommended action priority (C4)
24. ✅ AP-913 completeness checklist in run_manifest (NM15)

**Sprint 7 — Audit visibility ✅ COMPLETE April 21, 2026:**
25. ✅ Stage G skip surfaced to analyst — `ishikawa_run: bool` + `ishikawa_skip_reason` in `run_manifest`; `analyst_attention_flag` when skipped (§9.5)
26. ✅ ECA/RCA documents exempted from ±90-day recency window + recency bonus removed (B2)

**Sprint 8 — Evidence quality (next):**
27. Evidence excerpts passed through as source text, not summaries (April 20 H5)

### Post–April 21 implementation batch (code) — log and resolution status

**Purpose**: Record the **Post–April 21 SE code batch** so this review file matches the branch: what was **fully resolved**, what was **partially mitigated**, and **where** in code (for audit trail).

| Resolution | Review IDs | Files (primary) |
|------------|------------|-----------------|
| **Full** | §3.1 short-term; §5.1 source-tier activation; §6.1 **A3**; §6.4 **D4**; §6.5 **E3**; §6.6 **F3**; §6.7 **H1** / **NH11**; §6.7 **H4** / **S8** / **NH12**; §8.1 (NC6); §8.7 (NM15); **NM10** (same rule as D4) | See table below. |
| **Partial** | §3.4 A1 / **NM1**; §3.5 **NM2**; §6.1 **A1** (warnings only) | `input_guards.py`, `rca_reasoning_orchestrator.py` |
| **Explicitly not in scope (unchanged)** | Co-pilot touchpoints, PRA-calibrated risk derivation, etc. | — (see §8.4) |

| Review ref | Change |
|------------|--------|
| **§3.1 (rank inversion + auto re-entry)** | `orchestrators/rca_reasoning_orchestrator.py` — structured `reentry_hook` plus bounded automatic in-run re-entry loop (`enable_auto_reentry`, `auto_reentry_max_attempts`) now re-executes Stage B→F with targeted `focus_component_ids`; re-entry execution details are logged in `run_manifest.pipeline_config.reentry_execution` and persisted as first-class `reentry_execution` artifact (`schemas/reentry_execution.json`). |
| **§3.4 A1 / NM1** | `orchestrators/input_guards.py` — `build_input_guards` stores non-blocking `input_guards` on `run_context` (telemetry window vs event; optional PM/OC currency). **Strict** cross-artifact policy (abort / gate) not implemented. |
| **§3.5 NM2** | `build_input_guards` — heuristic `possible_multi_event_overlap` when `operational_context.recent_alarms` reference another `related_event_id` or suggest correlated events. **Broader** scoping checks remain open. |
| **§6.1 A3** | `input_guards.assert_output_dir_writable` at pipeline start from `rca_reasoning_orchestrator.py` when using `FileArtifactStore`. |
| **§6.4 D4 / §6.6 F3** | `orchestrators/causality_engine_v32.py` — `_eligible_review_alternative` returns `False` for `evidence_posture: "contradicted"`. Tests: `unit_tests/test_review_alternative.py`. |
| **§6.7 H1 / NH11** | `refine_with_evidence` sets `review_required` (e.g. near-ties, contradicted posture, temporal flags); `synthesis/rca_synthesizer_v31.py` — `RCASynthesizerV31._select_candidates` appends up to `max_synthesis_extra_review_candidates` (default 8) extra `review_required` rows after top-N. |
| **§6.7 H4 / S8 / NH12** | `rca_synthesizer_v31.py` — if LLM `primary_hypothesis.candidate_id` is not in the input candidate set (or `"NONE"`), the LLM `rca_card` is **dropped** and the deterministic path runs. |
| **§5.1 / F2 / NH6 (remaining part)** | `orchestrators/evidence_retriever.py` now populates `candidate_evidence_summary[].best_source_tier` from strongest supporting snippet metadata (`source_tier`), enabling Stage F authority weighting in `causality_engine_v32.py`. |
| **§6.5 E3** | `evidence_retriever._build_queries` now appends one targeted `out_of_boundary` query per `kg_context.out_of_boundary_anomalies[]` row (`query_intent: "kg_gap_investigation"`). |
| **§8.1 NC6** | `schemas/rca_card.json` adds required `contributing_causes[]`; `synthesis/rca_synthesizer_v31.py` now normalizes LLM and deterministic contributing-cause rows. |
| **§8.2 C5 / C4** | `synthesis/rca_synthesizer_v31.py` safety-significance post-processing now maps `affected_safety_functions` to analyst attention flags and escalates `recommended_actions[].priority` (critical/high/medium floor by safety tier), including alias normalization for common safety-system abbreviations. |
| **§3.2 short-term degraded-run visibility** | `pipeline_health` contract added to `evidence_bundle`, `causality_candidates`, and `run_manifest` (green/yellow/red + issues list). |
| **§3.2 stage-level policy extension** | `rca_reasoning_orchestrator.py` now computes `run_manifest.stage_health` (including `stage_i_archive`), propagates stage-level red/yellow issues into `pipeline_health`, and applies configurable `stage_policy_hooks` + `stage_remediation_playbooks` in review routing (`stage_policy_violations`, `stage_hard_stop_required`). |
| **§5.2 component filtering precision** | `evidence_retriever.py` derives `candidate_component_ids`; `storage/chroma_store.py` now writes `primary_component_id` at ingestion and applies index-level Chroma filter on that scalar key in `query_doc_type`, with legacy post-filter fallback only for old records lacking the new key. |
| **§5.6 historical event support-channel modeling** | `rca_reasoning_orchestrator.py` augments `kg_context.past_events` from `cmms_context` CR/WO history and now builds `seed_context.canonical_event_graph` + `seed_context.historical_support_channels` (`mode: support_channel_only`) to represent historical analogs as recurrence/evidence channels rather than standalone primary tracks. |
| **§3.3 strict governance hard-abort** | `rca_reasoning_orchestrator.py` now enforces hard-abort when `kg_governance.status == red` under strict policy (`strict_red_state_governance` + `hard_abort_on_kg_red_state`), recording explicit abort reason in `run_status`. |
| **§6.8 I3 Chroma archive failure handling** | `rca_reasoning_orchestrator.py` now adds Stage I `chroma_archive` execution/status plumbing into `stage_health`, `pipeline_health`, and `run_manifest` (`pipeline_config` + `artifacts`); archive exceptions are red-state failures and can hard-abort via `hard_fail_on_chroma_archive_error`, with explicit abort reason persisted to `run_status`. |
| **§8.3 barrier analysis** | New `barrier_analysis` artifact (`schemas/barrier_analysis.json`) computed in orchestrator and propagated to `run_manifest` and `rca_card.barrier_analysis` summary; depth pass adds `barrier_signal`-aware structural scoring in `causality_engine_v32.py` and barrier-weighted action rationale/priority shaping in `rca_synthesizer_v31.py`. |
| **§8.4 risk significance scalar** | `causality_engine_v32.py` now derives deterministic `risk_significance_scalar`/tier from `affected_safety_functions` and applies bounded governance adjustment (`governance_base` + `governance_risk_delta`); `rca_synthesizer_v31.py` now uses risk context to floor recommended action priority and append risk-weighted rationale. |
| **§8.7 NM15** | `ap913_completeness` block computed in orchestrator (`_compute_ap913_completeness`) and required in `run_manifest.json`. |

**`run_context` / consumers**: `input_guards` is stored under `run_context["input_guards"]` (merged when the orchestrator builds/persists `run_context.json`). Schemas for `run_context` typically allow additional properties; any consumer should treat unknown keys as optional telemetry.

---

*End of review — April 21, 2026*
*Total new findings: 33 (6 Critical, 12 High, 15 Medium)*
*Total prior findings: 21 (5 Critical, 6 High, 8 Medium) — see RCA_Systems_Engineering_Review_April_20.md*
*Combined open finding count: 54*
*Sprint 1 closed: NC3 ✅, NC4 ✅, NC5 ⚠️ (partial), S6 ✅, S7 ✅ — 4 fully closed, 1 partial. Remaining open: 49 findings*
*Sprint 2 closed: NH4 ✅, NH6 ⚠️ (partial — source_tier tagged; authority weight dormant), F1 ✅, S9 ✅ — 3 fully closed, 1 partial. Remaining open: 46 findings*
*Option B closed: S1 ✅ (was already fixed), S2 ✅ (orphaned orchestrators/kg_context.json deleted), S3 ✅ (was already fixed), C1 ✅ (was already fixed) — 4 closed. Remaining open: 42 findings*
*Sprint 3 closed: S10 ✅ (was already fixed; regression test added), H3 ✅ (scoring_evolution.json dedicated artifact), I1 ✅ (run_status sentinel), I2 ✅ (atomic writes via mkstemp+rename) — 4 closed. Remaining open: 38 findings*
*Sprint 4 closed: J2 ✅ (is_run_complete + load on FileArtifactStore), S5/E2/NH8 ✅ (BM25 warning + provenance field), S4/E1/NH7 ⚠️ (partial at Sprint 4; closed in latest batch via index-level `primary_component_id` filter).*
*Sprint 5 closed: H4 ✅ (analyst override primary_diff audit trail), H6 ✅ (posture_warning on all action rows), NC5 ✅ (run_context.json + run_manifest.json schemas + CORE_ARTIFACTS) — 3 fully closed. Remaining open: 32 findings (before Post–April 21 code batch).*

*Post–April 21 SE code batch: **closed** in code/docs — S8 (LLM primary id), NH11 (top-k + `review_required`), NH12 (LLM gate / same as S8), D4 + F3 + NM10 (review alternative vs `contradicted`), A3 (output dir); **short-term / partial** — §3.1 rank-inversion flag only; NM1/NM2/A1 via `input_guards`.*
*Sprint 6 closed: A2 ✅ (severity-adjusted evidence floor: `_SEVERITY_SCORE_FLOORS` + `passed_severity_gate`), NM3 ✅ (FM-category governance weight: `_governance_weight_for_fm` + `weights_override` in `_combine_scores`) — 2 fully closed. Remaining open: 28 findings.*
*Sprint 7 closed: §9.5 ✅ (Ishikawa skip surfaced: `ishikawa_run`/`ishikawa_skip_reason` in manifest + `_apply_ishikawa_skip_attention_flag`), B2 ✅ (ECA/RCA exempted from ±90-day window filter + recency bonus removed in `kg_context_builder.py`) — 2 fully closed. Remaining open: 26 findings.*
*Latest batch (this session): **closed/advanced** — NC6 (`contributing_causes[]`), NM15 (`ap913_completeness`), E3 (`out_of_boundary_anomalies` retrieval), remaining NH6/F2 gap (`best_source_tier` population), C5/C4 safety-function-to-rca_card propagation + priority mapping, **plus §3.1/§3.3 closure pass** (bounded automatic in-run re-entry loop + strict KG red-state hard-abort policy), **§3.2 policy-hook pass** (`stage_policy_hooks` + stage remediation playbooks + `stage_policy_violations` routing), **§5.2 closure pass** (candidate-scoped filtering + index-level `primary_component_id` Chroma filtering with legacy fallback mode), **§5.6 support-channel pass** (CMMS-derived `past_events` injection + canonical event graph/support-channel modeling in `seed_context`), **§8.4 risk-significance pass** (deterministic candidate risk scalar + Stage F governance adjustment + risk-aware action priority/rationale), and **§6.8 I3 archive-governance pass** (Stage I Chroma archive failure now feeds red/yellow health + remediation routing + strict abort policy). Informal roll-up: remaining open findings are now concentrated in PRA-calibrated risk modeling, backend-specific archive hook coverage, and deeper ontology-level barrier semantics.*
