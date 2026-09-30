# Code Review — Unexpected Activity Analysis Pipeline (Stages A–G)

**Scope:** `stages/stage_a_intake.py` through `stage_g_recommendation.py`,
`orchestrators/outage_activity_orchestrator.py`, `orchestrators/protocols.py`,
`stages/insertion_point_determiner.py`, `stages/completion_feedback.py`  
**Date:** 2026-04-16

---

## Summary

The pipeline is architecturally sound: protocol-based injection, schema-validated
artifacts, explicit regulatory-constraint propagation, and a well-reasoned
evidence-chain traceability design.  The core logic of every stage is implemented
and the unit-test coverage reported is substantial.

Seven issues are high enough severity to affect correctness of the recommendation
output in real-world cases.  Eight are medium-severity logic errors or design
inconsistencies.  The remainder are low-severity hygiene items.

---

## Critical / High Severity

### H1 — `_FLAG_CP_IMPACT` is never raised (Stage G)

**Location:** `stage_g_recommendation.py` → `_compute_attention_flags()`

```python
float_analysis = schedule_impact.get("float_analysis") or {}
if float_analysis.get("is_critical_path_impact"):   # ← field does not exist
    flags.append(_FLAG_CP_IMPACT)
```

Stage E's `float_analysis` sub-object contains `criticality_label`, `float_consumed_hours`,
`remaining_float_hours`, and `near_critical_float_threshold`.  There is no boolean field
`is_critical_path_impact`.  The flag is therefore **never appended**, regardless of whether
the activity lands on the critical path.

**Impact:** Outage managers reviewing the recommendation never see `critical_path_impact`
in the attention flags even for activities that consumed all network float.  The flag was
designed specifically to surface this situation.

**Fix:**
```python
if float_analysis.get("criticality_label") == "critical":
    flags.append(_FLAG_CP_IMPACT)
```

---

### H2 — Contingency buffer feasibility uses the wrong field as float proxy (Stage F)

**Location:** `stage_f_options.py` → `_generate_contingency_buffer()`

```python
available_float: float = float(
    float_analysis.get("available_float_before")   # absent in most Stage E outputs
    or float_consumed                              # ← used as fallback
    or 0.0
)
remaining_float: float = max(0.0, available_float - p50)
feasible = buffer_hours <= remaining_float or criticality_label == "non_critical"
```

`float_consumed_hours` is the float the emergent activity *consumes*, not the float
*available* at the insertion point.  When `available_float_before` is absent (the common
case with stub or simplified Stage E outputs), `available_float` becomes a small number
(hours consumed), so `remaining_float = max(0, small_number - p50)` is almost always 0.
The buffer option is then marked `feasible=False` even when the schedule has ample slack.

**Impact:** The `add_contingency_buffer` option is systematically flagged infeasible in
environments without a wired LOGOS schedule loader, causing Stage F to surface fewer
valid options and Stage G to produce more INCONCLUSIVE outcomes than warranted.

**Fix:** When `available_float_before` is absent fall back to
`remaining_float_hours` from Stage E (the explicit field for remaining float
after the activity), not `float_consumed_hours`.  If neither is present, default
to a permissive estimate and note the uncertainty rather than declaring infeasible.

```python
available_float: float = float(
    float_analysis.get("available_float_before")
    or float_analysis.get("remaining_float_hours")
    or float("inf")   # unknown → permissive
)
```

---

### H3 — Allen relation DURING / OVERLAPS-from-right mis-classification (Stage C)

**Location:** `stage_c_temporal_chain.py` → `_allen_relation()`

```python
if a_s < b_s - eps:          # a_e is within [b_s-eps, b_e+eps]
    return _OVERLAPS
if a_e <= b_e + eps:         # A entirely within B
    return _DURING
# A started within B but extends beyond B → SIMULTANEOUS
return _SIMULTANEOUS
```

The final `else` branch covers the case where the prior event *starts inside* the
emergent activity's window but *extends beyond its end*.  Allen's algebra calls this
the "overlapped-by" inverse — a prior event that was already in progress when the
activity started and continues afterwards.  This is a meaningful causal signal (the
prior condition is still active at activity onset and persists through it) but it
scores **0.50** (SIMULTANEOUS) instead of **0.90** (OVERLAPS).

By contrast, the OVERLAPS branch handles the reverse case correctly: prior event that
started *before* the activity and ends *during* it, scoring 0.90.

**Impact:** Any prior event that started before the emergent activity and is still ongoing
at the end of the activity window is under-scored by 0.40 points.  For long-duration
degradation events (e.g. a trending leak that spans the entire outage) this mis-classification
causes the causal relevance score to drop from `strong` to `moderate`.

**Fix:** The final branch before `return _SIMULTANEOUS` should return `_OVERLAPS`
(or a new `_OVERLAPPED_BY` alias that maps to the same score), since the causal
signal is equivalent:

```python
# A started within B but extends beyond B: prior event active through full window
return _OVERLAPS   # same causal weight as forward overlap
```

---

### H4 — Similarity weight config in Stage D does not govern actual scoring (Stage D)

**Location:** `stage_d_analogs.py` → `HistoricalAnalogConfig`

```python
lexical_weight: float = 0.30
semantic_weight: float = 0.40
component_weight: float = 0.20   # ← dead letter
context_weight: float = 0.10
```

The injected `SimilarityEngine` from `outage_uncertainty` uses its own
`SimilarityAggregator` (weights: lexical 0.20, semantic 0.40, context 0.40).
These weights are fixed at construction time and are **not** read from
`HistoricalAnalogConfig`.  The four config fields document a 4-component
model (`lexical / semantic / component / context`) that does not correspond
to the actual 3-component model (`lexical / semantic / context`) used by the
engine — `component_family` similarity is already embedded inside the context
scorer.

**Impact:** An operator tuning `HistoricalAnalogConfig.component_weight` to
`0.40` expecting the component-matching signal to dominate will see no
change in retrieval behaviour.  There is no way to override these weights
through Stage D's config as currently wired.

**Fix (two options):**
1. Remove the four weight fields from `HistoricalAnalogConfig` and document that
   scoring weights are controlled via the injected `SimilarityAggregator`.
2. Pass `config.lexical_weight` / `config.semantic_weight` / `config.context_weight`
   to `SimilarityAggregator(weights={...})` at construction so the config actually
   governs scoring.

The `component_weight` field should be removed regardless — it implies an
aggregation dimension that does not exist as a separate scorer.

---

### H5 — Outlier removal uses float equality to identify retained samples (Stage D)

**Location:** `stage_d_analogs.py` → `_remove_duration_outliers()`

```python
routine_remaining = list(separation.routine)
for (i, a), d in zip(nd_pairs, nd_durations):
    try:
        routine_remaining.remove(d)   # exact float equality
        kept_nd.append((i, a))
    except ValueError:
        pass  # classified as extended — drop
```

`list.remove()` uses `==` comparison.  The concern: `separation.routine` is
produced by `OutlierHandler.separate()` which returns the original values
unchanged (no arithmetic), so the values in `routine_remaining` *should* be
identical objects to those in `nd_durations`.  However this relies on the
invariant that `OutlierHandler.separate()` never copies through any arithmetic
operation.  If that invariant breaks in a future refactor (e.g. weight
normalisation that rounds values, or a new separation strategy that clips
extremes), the matching silently fails and **all analogs are dropped** (each
`remove()` raises `ValueError` and is caught silently).

**Fix:** Track by index, not by value.

```python
routine_set = set(separation.routine)     # fine for unique values
# For duplicates, use position-based matching instead:
extended_values = set(separation.extended)
kept_nd = [
    (i, a) for (i, a), d in zip(nd_pairs, nd_durations)
    if d not in extended_values
]
```

This is equivalent for unique values and correctly handles duplicates when the
separation result has equal floats in the routine pool.

---

### H6 — `_query_activity_case` is mutable instance state — thread-unsafe (Stage D)

**Location:** `stage_d_analogs.py` → `HistoricalAnalogRetriever`

```python
self._query_activity_case = None  # set in _build_query(), consumed by multiple methods
```

The per-call query state is stored on the instance.  Two concurrent calls to `retrieve()`
on the same retriever share this attribute and will silently overwrite each other.  The
outlier removal step (`_remove_duration_outliers()`) also reads this state.

**Impact:** Silent incorrect scoring in any future async/batch processing mode.

**Fix:** Pass `query_activity_case` as an explicit parameter through the private
call chain (`_retrieve_candidates(query, qac)`, `_score_and_filter(query, qac)`,
`_remove_duration_outliers(analogs, qac)`) and remove the instance attribute.
This also makes the data flow explicit and easier to test.

---

### H7 — MONITOR decision status is effectively unreachable (Stage G)

**Location:** `stage_g_recommendation.py` → `_determine_decision_status()`

```python
if cp_impact == 0.0 and analog_count == 0 and dist_tier == "low_confidence":
    return _MONITOR
```

All three conditions must hold simultaneously.  In practice:

- `cp_impact == 0.0` only when the primary option is `defer` (cp_impact=0 by design) or
  `insert_now` with zero float consumed.  Defer maps to `_DEFER` before this check.
- `analog_count == 0` implies `low_confidence` (the two gate redundantly), but with zero
  analogs the fallback policy still produces a distribution, so `p50` may be non-zero,
  which means Stage E likely produced non-zero float analysis, making `cp_impact > 0`.

The practical result: **MONITOR is never returned** in normal pipeline execution.
A non-critical activity with no historical data and no schedule impact (the ideal
MONITOR case) returns PROCEED because `cp_impact` is slightly above 0.

**Fix:** Relax the MONITOR condition to require only low confidence + no clear
action path, independent of the exact cp_impact value:

```python
if primary_option is None:
    return _INCONCLUSIVE
if dist_tier == "low_confidence" and analog_count == 0:
    if criticality_label in ("non_critical",):
        return _MONITOR
```

---

## Medium Severity

### M1 — `causal_posture = "contradicted"` overrides strong supporting evidence (Stage C)

**Location:** `stage_c_temporal_chain.py` → `_summarize_chain()`

```python
if has_contradiction:
    posture = "contradicted"
elif "strong" in strengths:
    posture = "supported"
```

A single FOLLOWS link (a prior event that occurred *after* the emergent activity onset —
labelled temporal_contradiction) forces `causal_posture = "contradicted"` even when 10
other links are classified as `strong`.  In Stage F this maps to `urgency = 0.70`, and
in Stage G triggers `_FLAG_TEMPORAL_CONTRADICTION`.

Contradictions can arise from data-entry errors (wrong timestamps) or legitimately
asynchronous workflows (e.g. a CR raised after the fact for documentation).  Discarding
all supporting evidence in such cases is overly conservative.

**Fix:** Introduce a `"contradicted_with_support"` posture for chains where
contradictions coexist with strong positive evidence, and map it to an appropriate
intermediate urgency in Stage F.  This also better serves the evidence-traceability
principle: the analyst should see *both* the contradiction *and* the supporting
evidence, not just the contradiction.

---

### M2 — Deferred option cost includes future-cycle labor at face value (Stage F)

**Location:** `stage_f_options.py` → `_compute_cost_estimate()` / `_compute_option_cost()`

```python
deferred_labor_cost = round(deferred_duration_hours * crew_count * labor_rate, 2)
total = round(labor_cost + schedule_extension_cost + crash_premium + deferred_labor_cost, 2)
```

For `defer_to_post_outage`, `deferred_duration_hours = p50` and `duration_hours = 0`.
The `total_cost_usd` therefore includes the full future-cycle labor at today's rates.
Since `max_cost` is computed across all candidates (including defer), this inflates the
cost denominator for the scoring normalisation and artificially raises the cost score
of non-defer options.

Additionally: no time-discounting.  A deferred job 18 months away costs less in
present-value terms than the same job today.  Presenting them as equivalent overstates
the apparent cost of deferral.

**Fix (pragmatic):** Exclude `deferred_labor_cost` from the `total_cost_usd` used for
normalised cost scoring, keeping it as a separate informational field.  The deferred
cost should inform the outage manager's decision but should not enter the comparative
risk score.

---

### M3 — Parallel option confidence is hardcoded, ignoring analog data quality (Stage F)

**Location:** `stage_f_options.py` → `_generate_parallel_option()`

```python
option = _make_option(
    ...
    confidence=0.65,   # hardcoded, ignores distribution confidence_tier
)
```

Every other action option derives its confidence from `_tier_confidence(dist.get("confidence_tier"))`:
- `data_supported` → 0.85
- `sme_informed` → 0.65
- `low_confidence` → 0.40

The parallel option ignores this completely.  For `data_supported` cases the parallel
option confidence (0.65) is lower than `insert_now` (0.85), unfairly penalising the
option in the composite risk score.  For `low_confidence` cases it's too high (0.65 vs. 0.40).

**Fix:**
```python
dist = schedule_impact.get("cp_impact") or {}
# read from historical_analogs (pass through) or fall back to 0.65
confidence = _tier_confidence(dist.get("confidence_tier")) if dist else 0.65
```

---

### M4 — Pre-outage staging detection uses unreliable string comparison (Stage F)

**Location:** `stage_f_options.py` → `_generate_contingency_buffer()`

```python
if not actual_start and outage_start and detection_ts:
    option_type = _PRE_STAGE
```

The condition doesn't actually compare `detection_ts` against `outage_start` — it
checks only that both are truthy.  Any activity with a detection timestamp and outage
start populated will silently become `pre_outage_staging`, including activities detected
during the outage (when staging is no longer possible).

**Fix:** Parse both timestamps and compare:
```python
if not actual_start and outage_start and detection_ts:
    dt_detect = _parse_dt(detection_ts)
    dt_outage = _parse_dt(outage_start)
    if dt_detect and dt_outage and dt_detect < dt_outage:
        option_type = _PRE_STAGE
```

---

### M5 — Single-sample distributions give false precision (Stage D)

**Location:** `stage_d_analogs.py` → `_fit_duration_distribution()`

With `min_analogs_for_sme_informed = 1`, a single analog produces a degenerate
distribution where `p50 = p80 = p90 = actual_duration_hours`.  This is presented
to Stage E and Stage F with tier `sme_informed` (confidence = 0.65), conveying
precision that does not exist.

**Fix:** Require `n ≥ 2` for a fitted distribution.  For `n = 1` use the fallback
path but annotate it distinctly (`"single_analog_prior"`) so Stage G can raise a
dedicated analyst flag.

---

### M6 — `outlier_iqr_factor` in `HistoricalAnalogConfig` is a dead config field (Stage D)

**Location:** `stage_d_analogs.py` → `HistoricalAnalogConfig`

```python
outlier_iqr_factor: float = 1.5
"""IQR multiplier for duration outlier removal (Tukey fence).
Passed to OutlierHandler(strategy='iqr').  The IQR factor is not directly
configurable on OutlierHandler; Tukey 1.5 IQR is the default."""
```

The docstring acknowledges the field cannot be passed to `OutlierHandler` because
that class hardcodes its IQR multiplier.  The config field therefore has no effect.

**Fix:** Either (a) add an `iqr_factor` parameter to `OutlierHandler.__init__()` and
wire it through, or (b) remove the field from `HistoricalAnalogConfig` and note in the
docstring that the IQR factor is fixed at 1.5.  Leaving a non-functional config field
is a maintenance trap.

---

### M7 — `regulatory_keywords_path` in Stage A config is documented but not wired

**Location:** `stage_a_intake.py` → line ~815

The config accepts a `regulatory_keywords_path` parameter to allow operators to
supply plant-specific regulatory keyword lists, but `_detect_regulatory_constraints()`
always uses the inline hardcoded regex patterns.  The path is never read.

**Impact:** Operators configuring this field see no change in regulatory detection.
More importantly, plant-specific TS numbers (e.g. "TS 3.4.6") not in the default
regex set are never detected.

**Fix:** Load and compile a supplementary pattern from `regulatory_keywords_path`
when provided, then union it with the default patterns before scanning.

---

### M8 — Orchestrator comment over-claims schedule load savings from two-pass design

**Location:** `outage_activity_orchestrator.py` → `insertion_point_determiner` docstring

> "Stage E reuses insertion point (avoids double schedule load)"

The two-pass pre-computation avoids re-*determining the insertion point* in Stage E
(by passing `schedule_context`), but Stage E still loads the full schedule independently
for the Monte Carlo simulation.  The schedule is loaded twice: once in the pre-pass
and once in Stage E.

This is a misleading comment, not a code bug, but it may lead to incorrect performance
expectations when the schedule loader has significant latency (e.g. live P6 API call).

**Fix:** Update the docstring to clarify what is and is not saved.  If true schedule
load deduplication is desired, cache the loaded `OutageData` in `ScheduleContext` and
pass it to Stage E.

---

## Low Severity / Style

### L1 — `_DictActivityCase.__getattr__` returns `None` for missing attributes silently

Any attribute access on the fallback `_DictActivityCase` returns `None` rather than
raising `AttributeError`.  This makes debugging harder when an expected field is simply
absent: the scorer sees `None` and applies missing-field redistribution silently.
Adding a `__repr__` noting it is a fallback object would help diagnostics.

---

### L2 — `confidence_tier` override comment is misleading (Stage D)

```python
# Authoritative tier is count-based (not the fitter's internal tier), so we override here.
distribution["confidence_tier"] = confidence_tier
```

When `confidence_estimator` is injected, `_compute_confidence_tier()` delegates to
`ConfidenceEstimator` (similarity-aware, not count-only).  The comment incorrectly
implies the tier is always count-based.  This will confuse maintainers who inject
a `ConfidenceEstimator` and wonder why "count-based" is mentioned.

---

### L3 — `pipeline_version` is `None` in all produced artifacts (Stage G)

Both `pipeline_version` and `provenance.pipeline_version` are hardcoded to `None`.
Without a version field, artifacts from different pipeline generations are
indistinguishable in storage and cannot be selectively re-run.

---

### L4 — `_generate_parallel_option()` returns a `List[JsonDict]` while all other generators return a single `JsonDict`

The inconsistent return type makes the caller (`generate()`) use
`candidates.extend(...)` for parallel and `candidates.append(...)` for all others.
An empty list (no parallel window identified) silently contributes zero options.
If the pattern is intentional (parallel can generate zero or more options), document
this explicitly; otherwise normalise to `Optional[JsonDict]` and filter None.

---

### L5 — Bare `except Exception:` in Stage A (at least 6 locations)

All are tagged `# noqa: BLE001` indicating intentional broad catch.  These are
acceptable for an NLP pipeline with optional backends, but each site should log
at least `LOGGER.debug(... exc_info=True)` to preserve the exception context for
diagnostics.  Several currently only log `LOGGER.debug(...)` with a plain string.

---

## Design Observations (not bugs, but worth discussing)

### D1 — Stage B produces an empty timeline when the KG driver is absent ✅ fixed 2026-04-16

~~An empty `ComponentEventTimeline` propagates through Stage C (no links → `causal_posture = "insufficient_data"`, urgency = 0.40 in Stage F) and Stage G (empty `evidence_chain` from B/C).  This is graceful degradation by design, but the absence of the KG driver is never surfaced as a first-class flag to the analyst.  A deployment without the KG driver silently provides weaker recommendations without informing the operator why.  Consider adding a `kg_unavailable` flag to the run manifest.~~

Added `"kg_driver_available": bool` to the Stage B artifact.  Orchestrator `_finalize_manifest()` propagates it into `artifacts.component_event_timeline.kg_driver_available`.  `_compute_review_hooks()` derives `kg_unavailable = not kg_driver_available` and surfaces it as a first-class flag in `review_hooks`.

### D2 — Stage F option set is fixed at design time; no extensibility hook ✅ fixed 2026-04-16

~~The seven option types are generated by named private methods wired directly in `generate()`.  Adding a new option type (e.g. a partial-completion option, or a regulatory relief request) requires modifying the class.  A plugin registry (`_option_generators: list[callable]`) would allow domain-specific option types to be injected at the same level as other dependencies.~~

Added `extra_option_generators: List[Callable]` to `InsertionOptionGenerator.__init__()` and a `register_option_generator(fn)` convenience method.  Registered callables receive the full `generate()` kwargs and may return `JsonDict | List[JsonDict] | None`.  They are invoked after the built-in generators, before regulatory clearance, cost estimation, and scoring.  Exceptions in extra generators are caught and logged, never aborting the pipeline.

### D3 — Cost model uses a single global `outage_day_cost_per_hour` across all options ✅ fixed 2026-04-16

~~The schedule extension cost is identical for all options, including `escalate_to_management` whose CP drag is the same as `insert_now`.  In practice, escalation delays decision-making, adding a "decision latency" overhead not captured in the model.  A `decision_delay_hours` field on the escalate option would make the cost more accurate.~~

Added `escalate_decision_delay_hours: float = 4.0` to `InsertionOptionConfig`.  `_generate_escalate()` embeds `decision_delay_hours` in the option dict and mentions it in the rationale.  `_compute_cost_estimate()` accepts `decision_delay_hours` and computes `decision_latency_cost_usd = delay × outage_day_cost`, included in `total_cost_usd` (the outage clock runs during the decision window).  `_compute_option_cost()` threads the value from the option dict for the escalate case only.

---

## Issue Priority Summary

| ID | Stage | Severity | Short description |
|----|-------|----------|-------------------|
| H1 | G | **Critical** | ~~`_FLAG_CP_IMPACT` never raised — wrong field name~~ ✅ fixed 2026-04-16 |
| H2 | F | **High** | ~~Contingency buffer falsely infeasible — wrong float proxy~~ ✅ fixed 2026-04-16 |
| H3 | C | **High** | ~~OVERLAPS-from-right mis-classified as SIMULTANEOUS (-0.40 score)~~ ✅ fixed 2026-04-16 |
| H4 | D | **High** | ~~Similarity weights in config are dead letters — don't govern engine~~ ✅ fixed 2026-04-16 |
| H5 | D | **High** | ~~Float equality for outlier removal — silent data loss on future refactor~~ ✅ fixed 2026-04-16 |
| H6 | D | **High** | ~~Mutable instance state `_query_activity_case` — thread-unsafe~~ ✅ fixed 2026-04-16 |
| H7 | G | **High** | ~~MONITOR status effectively unreachable~~ ✅ fixed 2026-04-16 |
| M1 | C | Medium | ~~Single contradiction overrides all strong support in causal posture~~ ✅ fixed 2026-04-16 |
| M2 | F | Medium | ~~Deferred cost compared at face value — inflates cost scoring~~ ✅ fixed 2026-04-16 |
| M3 | F | Medium | ~~Parallel option confidence hardcoded, ignoring analog tier~~ ✅ fixed 2026-04-16 |
| M4 | F | Medium | ~~Pre-staging detection uses truthy check, not timestamp comparison~~ ✅ fixed 2026-04-16 |
| M5 | D | Medium | ~~Single-analog distribution gives false p80=p90=p50 precision~~ ✅ fixed 2026-04-16 |
| M6 | D | Medium | ~~`outlier_iqr_factor` config field has no effect~~ ✅ fixed 2026-04-16 |
| M7 | A | Medium | ~~`regulatory_keywords_path` config field not wired~~ ✅ fixed 2026-04-16 |
| M8 | Orch | Medium | ~~Comment overstates schedule load savings from two-pass design~~ ✅ fixed 2026-04-16 |
| L1–L5 | Various | Low | ~~Style / maintainability~~ ✅ fixed 2026-04-16 |
| D1 | B + Orch | Design | ~~Empty timeline when KG driver absent not surfaced as first-class flag~~ ✅ fixed 2026-04-16 |
| D2 | F | Design | ~~Option set fixed at design time; no extensibility hook~~ ✅ fixed 2026-04-16 |
| D3 | F | Design | ~~Cost model missing decision latency for escalate option~~ ✅ fixed 2026-04-16 |
