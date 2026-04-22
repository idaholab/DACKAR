# FMEA Handling in the RCA Pipeline
## Specification and Design Notes

**Date**: April 22, 2026
**Status**: Design — not yet implemented
**Companion documents**: `RCA_pipeline_stages.md` · `RCA_stage_B5_signal_evidence_spec.md` · `RCA_Data_Management_Strategy.md`

---

## 1. Common FMEA Table Structure

FMEAs exist in many formats across industries and individual organisations. Before specifying how the RCA pipeline handles them, it is important to establish what fields are genuinely common, what fields vary, and what fields are entirely absent from standard formats.

### 1.1 Fields present in virtually all FMEA formats

These columns appear across AIAG FMEA (4th/5th ed.), IEC 60812, MIL-STD-1629A, and most nuclear-utility-specific templates:

| Canonical field | Typical column names | Content |
|----------------|---------------------|---------|
| `item_function` | Item, Function, Component, System Element | What the component or function is |
| `failure_mode` | Failure Mode, Potential Failure Mode, FM | The specific way the item fails |
| `potential_causes` | Potential Cause(s) of Failure, Failure Mechanism, Root Cause | What could cause this failure mode to occur — **often a list** |
| `local_effect` | Local Effect, Failure Effect (Local), Effect on Sub-system | Observable consequence at the component level |
| `system_effect` | Next Higher Effect, Subsystem Effect, Effect on System | Consequence propagated up to the next assembly or system |
| `end_effect` | End Effect, Mission Effect, Safety Effect, Effect on Plant | Ultimate consequence at the mission or safety level |
| `detection_method` | Detection Method, Current Controls (Detection), How Detected | How the failure mode would be identified — procedure, alarm, test, inspection |
| `corrective_actions` | Recommended Actions, Corrective Actions, Compensatory Measures | What should be done if this failure mode occurs or is found |
| `severity` | Severity, S, Criticality Rank | Seriousness of the end effect — numeric scale |

> **Key structural point**: every FMEA distinguishes at least two dimensions for each failure mode — **causes** (what produces the failure) and **effects/consequences** (what the failure produces downstream). Detection methods span both: they may detect the cause before it produces the failure, or detect the effect after it has occurred. The RCA pipeline must treat these as distinct fields, not collapse them.

### 1.2 Fields common in some formats but absent in others

| Field | Present in | Absent in |
|-------|-----------|-----------|
| `occurrence` (O) | AIAG, IEC 60812, most nuclear utility templates | MIL-STD-1629A (uses failure rate λ instead) |
| `detection_rating` (D) | AIAG, IEC 60812 | MIL-STD-1629A |
| `rpn` (S × O × D) | AIAG, IEC 60812 | MIL-STD-1629A (uses criticality Cm = β × α × λp × t) |
| `failure_rate` (λ) | MIL-STD-1629A | Most commercial AIAG-derived templates |
| `failure_mode_ratio` (α) | MIL-STD-1629A | Most commercial templates |
| `mission_phase` | MIL-STD-1629A, aerospace/DOE | Most commercial/nuclear utility templates |
| `compensatory_measures` | Nuclear utility templates, IEC 60812 | AIAG |
| Safety function impact | Nuclear utility templates | Standard commercial formats |
| Tech spec / LCO applicability | Nuclear utility templates | All non-nuclear formats |
| Single failure criterion | Nuclear utility templates | All non-nuclear formats |
| Common cause failure flag | Some nuclear utility templates | Most commercial formats |

### 1.3 Fields absent from all standard FMEA formats

These fields appear in the current RCA pipeline's FMEA schema but are **not present in any standard FMEA format**. They are custom enrichments that must be added deliberately:

| Field | Used by pipeline | Why absent from standard FMEAs |
|-------|-----------------|-------------------------------|
| `expected_latency_min/max_hours` | Stage C TSKR latency alignment | Standard FMEAs describe effects qualitatively, not with propagation time bounds |
| `expected_anomaly_pattern` | Stage C + Stage D symptom matching | The signal pattern vocabulary (step_change, gradual_drift, etc.) is not a standard FMEA concept |
| `applies_to_component_id` (instance-level link) | Stage 0 KG APPLIES_TO edge | Standard FMEAs are class-level (component type), not instance-level |

---

## 2. What the RCA Pipeline Actually Needs from FMEAs

Tracing through each pipeline stage, the table below maps what the pipeline consumes, where it comes from in the FMEA, and how critical the field is when absent.

| Pipeline consumer | FMEA field required | Standard availability | Impact when absent |
|-------------------|--------------------|----------------------|--------------------|
| Stage 0 — KG APPLIES_TO edge | `component_type` | Universal | FM node created but not linked to any equipment — orphaned, invisible at Stage B |
| Stage B — failure mode retrieval | `failure_mode_name`, `failure_mechanism` | Universal | No FM nodes in KG → empty candidate pool |
| Stage B — safety function linkage | `end_effect` (safety function impact) | Nuclear templates only | Safety function nodes unlinked; C5 finding recurs |
| Stage C — latency alignment | `expected_latency_min/max_hours` | **Absent from all standard formats** | `latency_violation_type: unknown`; alignment score degrades to floor |
| Stage C — anomaly pattern filter | `expected_anomaly_pattern` | **Absent from all standard formats** | Pattern matching disabled; Stage C scores all anomaly types equally |
| Stage D — symptom matching | `local_effect`, `expected_symptoms` | Partially (free text) | Symptom match quality depends on NER over free text |
| Stage D — governance scoring | `severity`, `occurrence`, `detection`, `rpn` | AIAG/IEC only | MIL-STD plants: governance score undefined |
| Stage H — recommended actions | `corrective_actions` | Universal (free text) | Recommended actions fall back to generic templates |
| PM Compliance — scope analysis | `failure_mode_name` (for PM↔FM linkage) | Universal | Linkage degrades to free-text fuzzy matching |

### 2.1 The latency argument

`expected_latency_min/max_hours` is the most consequential missing field. It is the primary discriminating input to Stage C's TSKR scoring — the mechanism by which two failure modes with the same Allen relation (both PRECEDE the event) are separated. Without it, Stage C cannot distinguish a bearing wear failure mode with an expected 2–6 hour lead time from a seal degradation failure mode with an expected 48–72 hour lead time.

However, **latency values are not present in any standard FMEA format**, and the few cases where they are added by nuclear utilities are typically rough expert estimates with wide uncertainty bounds. Calibrating a scoring function to uncertain bounds of uncertain quality is architecturally fragile.

Stage B.5 (topology-driven anomaly fetch and propagation chain construction, see `RCA_stage_B5_signal_evidence_spec.md`) provides a more robust substitute: it derives the actual temporal ordering of anomalies from plant historian data, cross-referenced against the topology. This is data-driven rather than estimate-driven. An anomaly that demonstrably preceded the event in plant data, and whose sensor is topologically upstream of the event asset, provides stronger temporal evidence than a latency bound asserted in a document that may not have been updated since original plant construction.

**Architectural consequence**: `expected_latency_min/max_hours` should be treated as an optional enrichment field, not a required one. Stage C's latency alignment scoring should be active when the field is populated, but should degrade to a neutral score (not a penalty) when it is absent — rather than the current `latency_violation_type: unknown` floor. The primary temporal discriminator is Stage B.5's propagation chain, not Stage C's latency alignment.

This reframing has an additional benefit: it makes the pipeline functional on the first day of deployment, before any FMEA enrichment has been performed, using only the standard fields present in whatever FMEA the plant already has.

---

## 3. FMEA Ingestion Normalization Layer

### 3.1 Purpose

The current `fmeaParser.py` handles column naming variations via regex matching (`DEFAULT_COLUMN_MAP`) but does not handle structural differences between FMEA formats. A plant using MIL-STD-1629A will have criticality numbers where the parser expects RPN; a plant using a function-based FMEA will have no component_type column at all. The normalization layer sits between raw FMEA files and the KG ingest, making format differences explicit and handled rather than silently ignored.

### 3.2 Format profiles

Define a small set of named format profiles that capture the most common structural variants. Each profile declares which columns are present, what they map to in the canonical schema, and what derivation rules apply when a canonical field must be computed from available ones.

```python
FMEA_FORMAT_PROFILES = {
    "aiag_4th":  AiagFmeaProfile(),      # AIAG FMEA 4th edition
    "aiag_5th":  Aiag5thFmeaProfile(),   # AIAG FMEA 5th edition (DFMEA/PFMEA split)
    "mil_std_1629a": MilStd1629aProfile(),
    "iec_60812": Iec60812Profile(),
    "nuclear_generic": NuclearGenericProfile(),  # common nuclear utility template
    "auto": AutoDetectProfile(),         # attempts format detection from column headers
}
```

Each profile is a dataclass that declares:
- `column_map`: `{canonical_field: [candidate_column_names]}` — ordered list of regex patterns to try
- `derived_fields`: `{canonical_field: derivation_fn}` — functions that compute a field from other available fields
- `required_fields`: fields whose absence is a critical ingestion error
- `optional_fields`: fields whose absence is flagged but non-blocking

### 3.3 Canonical field set

The normalization layer maps any input FMEA to this canonical schema, regardless of source format. Fields marked (E) are enrichments — not expected from standard FMEAs, populated by the enrichment workflow (§4).

```
REQUIRED (must be present for KG ingestion to proceed):
  component_type          str    — equipment type; matched to KG domain_category
  failure_mode_name       str    — short FM label
  failure_mechanism       str    — underlying physical/chemical mechanism

STANDARD OPTIONAL (present in most formats):
  local_effect            str    — observable consequence at component level
  system_effect           str    — consequence at system/train level
  end_effect              str    — consequence at mission/safety level
  potential_causes[]      list   — list of cause statements (may be single string in some formats)
  detection_method        str    — how this FM would be detected
  corrective_actions[]    list   — recommended responses
  severity                int    — 1–10 or equivalent scale
  occurrence              int    — 1–10 (AIAG) or derived from λ (MIL-STD)
  detection_rating        int    — 1–10 (AIAG) or derived
  rpn                     int    — computed as S×O×D if absent but S,O,D present
  safety_function_impact  str    — nuclear templates; linked to KG safety function nodes
  tech_spec_applicability str    — nuclear templates

ENRICHMENT FIELDS (E) — populated by §4 workflow, not expected from raw FMEA:
  expected_latency_min_hours    float   — lower bound of propagation delay
  expected_latency_max_hours    float   — upper bound of propagation delay
  expected_anomaly_pattern      str     — step_change | gradual_drift | spike | ...
  fmea_revision_date            date    — for KG staleness tracking
```

### 3.4 Derivation rules

When a canonical field is absent but can be computed from available fields, the normalization layer applies derivation rules automatically. Derived values are tagged with `derivation_method` in the KG node for traceability.

| Canonical field | Derivation rule | Condition |
|----------------|-----------------|-----------|
| `rpn` | S × O × D | All three present; RPN absent |
| `occurrence` | Derived from λ × mission_time (MIL-STD) | MIL-STD profile; λ and mission_time present |
| `severity` | Derived from criticality rank mapping table | MIL-STD profile; criticality present, severity absent |
| `end_effect` | Copied from `system_effect` if only one effect column exists | Single-effect FMEA formats |
| `potential_causes` | Split `failure_mechanism` on delimiter (`;`, `/`, `and`) | `potential_causes` absent but `failure_mechanism` is a compound string |
| `expected_anomaly_pattern` | NLP classification of `local_effect` text against the 7-value enum | `expected_anomaly_pattern` absent; `local_effect` present — **low confidence, flagged for review** |

### 3.5 Field quality classification

For every ingested FMEA, the normalization layer produces a per-field quality report:

| Status | Meaning |
|--------|---------|
| `present_native` | Field found directly in source FMEA with expected type |
| `derived` | Field computed from other present fields via derivation rule |
| `nlp_inferred` | Field inferred by NLP from text fields (low confidence — always flagged) |
| `missing_critical` | Required field absent — FM node ingested with null; KG governance warning written |
| `missing_optional` | Optional field absent — null in KG; no warning, but surfaced in ingestion report |
| `missing_enrichment` | Enrichment field (E) absent — expected; no warning; enrichment workflow is the intended path |

### 3.6 Multi-level effect handling

Standard FMEAs with local / next-higher / end effect columns must not be flattened to a single `local_effect`. The normalization layer maps all three to distinct KG properties:

```
failure_mode.local_effect        → component-level observable consequence
failure_mode.system_effect       → train/system-level consequence
failure_mode.safety_effect       → safety function impact (end effect)
```

When the FMEA has only one effect column, it is stored as `local_effect` and the other two are null. `safety_effect` is the field that Stage B queries when building the safety function linkage — if it is null, the APPLIES_TO → safety_function edge is not written.

### 3.7 Cause and consequence separation

The distinction between causes and consequences is structurally critical for RCA. A failure mode's **potential causes** are what Stage D uses to reason about upstream contributors. The **effects** are what Stage D uses for symptom matching and safety significance. The normalization layer must maintain this separation even when source FMEAs are ambiguous.

When a FMEA has a single "failure mechanism" column that conflates cause and mechanism, the normalization layer stores it as `failure_mechanism` (cause-side). When a FMEA has multiple effect levels, all are preserved. When cause and effect are in the same column (rare but observed in informal FMEAs), a simple heuristic (effect markers: "results in", "causes", "leads to"; cause markers: "due to", "caused by", "from") can separate them — but this is flagged as `nlp_inferred` with low confidence.

### 3.8 KG governance impact

The ingestion normalization layer writes a summary to the `kg_provenance` node (see Stage 0 spec in `RCA_pipeline_stages.md`):

```json
"fmea_ingestion_quality": {
    "total_fms_ingested":          int,
    "critical_field_missing_count": int,   // missing component_type or failure_mode_name
    "enrichment_field_missing_count": int, // missing latency, anomaly_pattern
    "derived_field_count":         int,
    "nlp_inferred_field_count":    int,
    "orphaned_fm_count":           int,    // no APPLIES_TO edge (component_type not in KG)
    "profile_used":                str,
    "format_autodetect_confidence": float  // only when profile = "auto"
}
```

This feeds `_compute_kg_governance()` at Stage A: high `enrichment_field_missing_count` sets the KG to `yellow` (runs proceed with a `fmea_enrichment_incomplete` flag in `run_context`); high `critical_field_missing_count` or `orphaned_fm_count` above threshold sets `red`.

---

## 4. FMEA Enrichment Workflow

### 4.1 Purpose

The enrichment workflow is a **human-in-loop step that runs before the RCA pipeline**, not inside it. Its purpose is to annotate KG failure mode nodes with the non-standard fields that the pipeline benefits from but cannot derive automatically from raw FMEA data. It is triggered when the KG governance check at Stage A finds a `fmea_enrichment_incomplete` flag, or explicitly by the analyst before a planned RCA investigation.

The workflow is deliberately limited in scope: it only asks for the fields that have the highest impact on pipeline quality and that cannot be reliably inferred from other data. It does not attempt to replace the FMEA — it extends it for the specific purpose of RCA decision support.

### 4.2 Target enrichment fields and their priority

Given the architectural argument in §2.1, the enrichment fields are ordered by impact:

| Priority | Field | Impact if absent | Effort to provide |
|----------|-------|-----------------|-------------------|
| 1 | `expected_anomaly_pattern` | Stage D symptom matching disabled for this FM | Low — SME selects from 7-value enum based on physical knowledge |
| 2 | `safety_effect` / safety function link | Safety function impact not propagated to rca_card | Low — SME confirms or denies link from existing safety analysis |
| 3 | `expected_latency_min/max_hours` | Stage C latency alignment uses floor score | Medium — requires engineering judgement or test data |
| 4 | Cause-consequence split refinement | NLP-inferred split may be wrong | Medium — SME reviews NLP result and corrects |

Note: `expected_latency_min/max_hours` is priority 3, not 1. The topology-driven anomaly sequencing of Stage B.5 provides temporal evidence that does not depend on these values. Enriching latency bounds is valuable but not a prerequisite for a functioning RCA run.

### 4.3 Workflow trigger conditions

The enrichment workflow should be triggered when **any** of the following conditions hold for the failure modes in the current event's KG neighborhood:

```python
def should_trigger_enrichment(kg_context, enrichment_thresholds) -> bool:
    fms = kg_context["failure_modes"]

    # Fraction of FMs with missing anomaly pattern
    pattern_missing_ratio = sum(
        1 for fm in fms if not fm.get("expected_anomaly_pattern")
    ) / max(len(fms), 1)

    # Any FM linked to a safety function but missing safety_effect
    safety_gap = any(
        fm for fm in fms
        if fm.get("linked_safety_function") and not fm.get("safety_effect")
    )

    return (
        pattern_missing_ratio > enrichment_thresholds["pattern_missing_ratio"]  # default 0.5
        or safety_gap
    )
```

### 4.4 Enrichment interface specification

The enrichment workflow is presented to the SME as a structured review form, one failure mode at a time. The form is pre-populated with all available FMEA data and any NLP-inferred values (clearly marked as unconfirmed). The SME confirms, corrects, or provides the missing values.

**Per-FM review form fields:**

```
Component:          [component_type]  [asset_id if instance-level]
Failure Mode:       [failure_mode_name]
Failure Mechanism:  [failure_mechanism]
Local Effect:       [local_effect]

--- Enrichment fields ---

Anomaly Pattern     [dropdown: step_change | gradual_drift | spike |
(required):          oscillation | dropout | sustained_exceedance | unknown]
                    [NLP suggestion if available, marked "unconfirmed"]

Safety Function     [yes / no / unknown]
Impact:             [if yes → select from KG safety function list]

Latency Bounds      Min: [__] hours    Max: [__] hours
(optional):         [basis: FMEA text | test data | engineering judgement | literature]

Cause / Effect      [review NLP split if flagged as low confidence]
Review:             Causes: [editable list]
                    Effects: [editable list]
```

Each submitted enrichment is written to the KG with full provenance:

```json
{
    "fm_id": "FM-001",
    "field": "expected_anomaly_pattern",
    "value": "gradual_drift",
    "enrichment_source": "human_review",
    "reviewer_id": "eng_jsmith",
    "review_timestamp": "2026-04-22T14:30:00Z",
    "basis": "engineering_judgement",
    "confidence": "high"
}
```

### 4.5 Enrichment scope — neighborhood-first strategy

A full plant-wide FMEA enrichment is impractical before a first RCA run. The workflow uses a **neighborhood-first strategy**: enrich only the failure modes in the KG neighborhood of the current event, not the entire FMEA. This makes the pre-run enrichment step bounded and focused.

The neighborhood is determined by Stage B's component expansion (same query used at run time). For a typical 2-hop neighborhood, this covers 10–50 failure modes — a manageable review session for a single engineer.

Enrichments are persistent in the KG and accumulate across RCA events. Over time, the most frequently investigated equipment will have fully enriched failure modes, and the enrichment step for those components becomes a verification rather than a creation task.

### 4.6 Enrichment status tracking

The `kg_provenance` node tracks enrichment progress:

```json
"fmea_enrichment_status": {
    "total_fms_in_kg":             int,
    "pattern_enriched_count":      int,
    "latency_enriched_count":      int,
    "safety_effect_confirmed_count": int,
    "enrichment_coverage_pct":     float,   // pattern_enriched / total
    "last_enrichment_run":         "date-time"
}
```

Stage A reads `enrichment_coverage_pct` for the current event's neighborhood and includes it in `run_context.kg_enrichment_coverage`. When coverage is low (< threshold, default 0.50), `run_context.fmea_enrichment_incomplete = true` is set, which Stage J surfaces in `run_manifest.analyst_attention_flags[]`.

---

## 5. Impact on Stage C Latency Alignment

The architectural argument in §2.1 requires a change to how Stage C treats missing latency bounds. The current behaviour penalises absent latency with `latency_violation_type: unknown` and a reduced score. This is inappropriate: it penalises failure modes that were never given a chance to be enriched, rather than deferring to the data-driven temporal evidence from Stage B.5.

**Revised Stage C latency alignment behaviour:**

| Condition | Current behaviour | Revised behaviour |
|-----------|------------------|------------------|
| `expected_latency_min/max_hours` present | Score by alignment | Unchanged |
| Both absent, `signal_evidence` chains available | `unknown` floor score | Neutral score (0.50) — defer to Stage B.5 chain_position_score |
| Both absent, no `signal_evidence` | `unknown` floor score | Neutral score (0.50) — latency dimension abstained |
| Observed lag outside bounds | `too_fast` / `too_slow` penalty | Unchanged |

A neutral score (0.50) means the latency dimension neither helps nor hurts the candidate — it abstains. This is the correct behaviour when no information is available: the other four scoring dimensions carry the ranking without an uninformative penalty distorting the result.

```python
def score_latency(
    mean_lag_h: float | None,
    latency_min_h: float | None,
    latency_max_h: float | None,
) -> tuple[float, str]:
    """
    Returns (latency_alignment_score, violation_type).
    Abstains (0.50, "not_available") when bounds are missing.
    """
    if latency_min_h is None or latency_max_h is None:
        return 0.50, "not_available"   # revised: neutral abstention, not floor penalty
    if mean_lag_h is None:
        return 0.50, "not_available"
    if mean_lag_h < latency_min_h:
        penalty = min(0.40, (latency_min_h - mean_lag_h) / latency_min_h)
        return clamp01(0.80 - penalty), "too_fast"
    if mean_lag_h > latency_max_h:
        penalty = min(0.40, (mean_lag_h - latency_max_h) / latency_max_h)
        return clamp01(0.80 - penalty), "too_slow"
    return 1.0, "none"
```

`latency_violation_type` enum gains `"not_available"` as an explicit value (replacing `"unknown"`) to clearly signal abstention in the artifact.

---

## 6. Implementation Checklist

### Phase 1 — Normalization layer (`doc_parsers/fmea_normalizer.py`)
- [ ] Define `FmeaFormatProfile` base class and `column_map`, `derived_fields`, `required_fields` interface
- [ ] Implement `AiagFmeaProfile` (4th and 5th edition)
- [ ] Implement `MilStd1629aProfile` — criticality → severity derivation, λ → occurrence derivation
- [ ] Implement `Iec60812Profile`
- [ ] Implement `NuclearGenericProfile`
- [ ] Implement `AutoDetectProfile` — column header matching with confidence score
- [ ] Implement derivation rules (§3.4): RPN, occurrence from λ, effect splitting, NLP pattern inference
- [ ] Implement field quality classification (§3.5): `present_native` / `derived` / `nlp_inferred` / `missing_*`
- [ ] Implement multi-level effect handler (§3.6): local / system / safety effect separation
- [ ] Implement cause-consequence separator (§3.7): heuristic + NLP with confidence flag
- [ ] Write ingestion quality report to `kg_provenance.fmea_ingestion_quality`
- [ ] Update `kg_ingest_fmea_workflow.py` to use normalizer as pre-processing step
- [ ] Update `schemas/fmea_ingestion_report.json` — new schema for quality report artifact

### Phase 2 — Enrichment workflow (`RCA/fmea_enrichment/`)
- [ ] `enrichment_trigger.py` — `should_trigger_enrichment()` logic (§4.3)
- [ ] `enrichment_form_builder.py` — builds per-FM review form pre-populated from KG
- [ ] `enrichment_writer.py` — writes enrichment records to KG with provenance (§4.4)
- [ ] `enrichment_status_tracker.py` — maintains `kg_provenance.fmea_enrichment_status`
- [ ] CLI entry point: `python -m dackar.RCA.fmea_enrichment --event event.json` — runs neighborhood-first enrichment session
- [ ] Update `_compute_kg_governance()` in orchestrator to read `enrichment_coverage_pct`

### Phase 3 — Stage C latency fix (`orchestrators/tskr_temporal_scorer.py`)
- [ ] Update `score_latency()` to return `(0.50, "not_available")` when bounds absent (§5)
- [ ] Add `"not_available"` to `latency_violation_type` enum in `schemas/tskr_patterns.json`
- [ ] Update Stage C confidence formula comment to reflect latency as optional enrichment
- [ ] Add test: `test_latency_abstention` — absent bounds → score = 0.50, not floor

### Phase 4 — Tests
- [ ] `unit_tests/test_fmea_normalizer.py`:
  - [ ] `test_aiag_column_mapping` — standard AIAG columns map correctly
  - [ ] `test_milstd_criticality_to_severity_derivation` — criticality → severity computed
  - [ ] `test_milstd_lambda_to_occurrence_derivation` — λ × t → occurrence computed
  - [ ] `test_rpn_derivation` — S × O × D computed when RPN absent
  - [ ] `test_multi_level_effect_separation` — local / system / safety stored separately
  - [ ] `test_cause_consequence_separation` — compound string split correctly
  - [ ] `test_nlp_pattern_inference_flagged` — inferred pattern has `nlp_inferred` status
  - [ ] `test_missing_critical_field_governance_warning` — missing component_type → red governance
  - [ ] `test_missing_enrichment_field_no_warning` — missing latency → yellow, not red
  - [ ] `test_autodetect_profile` — column headers match correct profile
- [ ] `unit_tests/test_fmea_enrichment.py`:
  - [ ] `test_trigger_condition_pattern_missing` — > 50% FMs missing pattern → trigger
  - [ ] `test_trigger_condition_safety_gap` — safety function linked but no safety_effect → trigger
  - [ ] `test_enrichment_provenance_written` — reviewer_id, timestamp, basis stored in KG
  - [ ] `test_neighborhood_first_scope` — enrichment query returns only neighborhood FMs
  - [ ] `test_enrichment_coverage_tracking` — coverage_pct updates correctly after session

---

## 7. Open Questions

1. **NLP anomaly pattern inference quality**: the derivation rule for `expected_anomaly_pattern` from `local_effect` free text (§3.4) is flagged as low confidence. The quality of this inference depends on how descriptive the local_effect text is. A systematic evaluation of this against a labelled set of FMEA records is needed before the inference can be trusted even as a starting suggestion in the enrichment form.

2. **MIL-STD criticality → RPN equivalence**: the derivation of `severity` from criticality category (Class I–IV in MIL-STD-1629A) and of `occurrence` from λ × t is an approximation. The two scales are not directly equivalent, and using derived RPN values in the governance score could introduce systematic bias for MIL-STD plants. A plant-specific calibration table may be needed.

3. **Instance-level vs. class-level FMEAs**: standard FMEAs are always class-level (`component_type`). The `APPLIES_TO` edge from a `failure_mode` node to a specific `element_usage` (plant instance) requires the KG query `WHERE domain_category = component_type`. If the same component type appears in multiple trains (A-train pump, B-train pump), both instances get the same failure modes — which is correct for the failure mode catalog but incorrect if the FMEA was written for a specific instance (e.g., a vendor-specific FMEA for one particular pump model installed only in Train A). The normalization layer has no mechanism to scope FMEAs to specific instances. This is flagged as a future requirement.

4. **FMEA revision management**: nuclear utility FMEAs are living documents revised after plant modifications, operating experience, and regulatory changes. The KG currently tracks `fmea_revision_date` per failure mode node, but there is no mechanism to detect when a FMEA has been revised and the KG needs partial update. A delta-ingestion capability (compare current FMEA against KG state, update only changed rows) is needed to avoid full re-ingestion on every revision.

5. **Enrichment governance for safety-significant FMs**: enrichment fields for safety-significant failure modes (those linked to safety functions or with high severity) should require dual review — an additional reviewer must confirm the SME's entry before it is written to the KG. The current enrichment writer has no approval workflow. For a production nuclear deployment, this is likely a regulatory requirement.
