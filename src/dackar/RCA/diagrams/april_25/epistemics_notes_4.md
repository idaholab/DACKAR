# Epistemics Module — Integration Notes

**Date:** 2026-04-29  
**Scope:** Impact analysis of `dackar.RCA.epistemics` on the RCA workflow (`rca_workflow_reference_guide_april_25.md`).  
**Status:** Working notes — pre-implementation, pre-ADR.

---

## 1. What the epistemics module is arguing (core claim)

The pipeline currently has a **provenance and shape** model for data, but not a **semantic contribution** model. It knows what a CR *is* and where it came from, but has no unified vocabulary for what a CR *does* in the causal story.

That vocabulary is currently reinvented independently in at least three places:

- **Retriever** — `doc_type` priority weights (`evidence_retriever.py`)
- **Causality engine** — scoring dimensions, pre/post-refine logic (`causality_engine_v32.py`)
- **Synthesizer** — implicit prompt design (`rca_synthesizer_v31.py`)

The epistemics module is the **single place** that answers: *"what epistemic role does this datum play in the causal story?"*

### 1.1 Three-layer architecture

The module separates three concerns that must remain distinct. If they blur, the original problem is recreated in a more complex form.

| Layer | Question | Examples |
|---|---|---|
| **Classification** | What is this data element? | CR = monitors performance; ECA = analyzes past degradation |
| **Routing** | Where does it go in scoring? | Analyzes → `support_score`; monitors → `contextual_score` |
| **Control** | What constraints does it impose? | Caps, flags, override rules, attention flags |

All design decisions below are organized within this three-layer structure.

---

## 2. Four-way epistemic classification (Layer 1)

The fundamental question the classification answers is: **what relationship does a data element have to equipment performance?** Four classes fall out naturally and together are complete with respect to the data elements the pipeline ingests (§2.1 of the workflow guide).

### 2.1 Affects performance

Things that **act on** the equipment. Candidate causes or contributors — can appear at any causal depth (proximate, contributing, root).

| Data element | Notes |
|---|---|
| Work orders | Physical activity performed on equipment — repair, adjustment, replacement; primary role is the activity, not the embedded observation |
| Operational context | Operating mode, power level, recent operator actions |
| Configuration change records | Engineering changes, setpoint packages — changed the equipment's operating basis |
| Vendor and supply chain records | Lot numbers, batch certifications — potential common-cause contributors |
| Training records | Personnel qualification — affects human execution quality |
| PM compliance | What was or wasn't done to the equipment; maintenance posture |

### 2.2 Monitors performance

Things that **observe** the equipment's state. Evidence of a condition, not causes of it. In scoring, monitors signals route to temporal and telemetry dimensions — they do not enter the evidence dimension that drives contributing and root cause depth. Note: this is a scoring routing rule, not an absolute engineering statement. Repeated monitor evidence across trains or environmental monitor data during a Category F event can materially support contributing-cause reasoning — but that reasoning must be made explicit through the attention flag and analyst acknowledgment mechanism (§4.2), not implicit through score inflation.

| Data element | Notes |
|---|---|
| Telemetry summary | Anomaly records, pattern type, severity |
| Alarm log | Activations with timestamps; detection events, not causal events |
| SOE log | Discrete state transitions; millisecond-resolution detection record |
| Environmental monitoring | Ambient conditions at event time |
| Condition reports | Human-generated observation that a condition exists; embedded preliminary cause assessment is a secondary contribution, not the primary role |

### 2.3 Analyzes past degradation

Things whose **primary purpose is causal interpretation** of a specific past event or condition. Not preliminary field observations — documents written specifically to answer "what happened and why." The CR says "we saw this"; the ECA says "here is what it means" — they are successive steps in the interpretive chain, not competing documents in the same class.

| Data element | Notes |
|---|---|
| ECAs | Engineering condition assessment — causal interpretation of a degradation; high authority |
| RCAs | Formal root cause conclusion; highest authority within class |
| OE documents (INPO, NRC LERs, EPRI) | Fleet/industry causal analogy; authority discounted by tier distance (plant 1.0 / fleet 0.80 / industry 0.60) |
| Similar event list items | Structured analogy; not direct observation of current event |
| KG past events (closed with confirmed cause) | Historical causal conclusions with confirmed linkage; authority depends on closure status |

### 2.4 Characterizes the system

Things that define the **reference frame** against which all other data elements are interpreted. Not about a specific degradation event — the standing model of the system.

| Data element | Notes |
|---|---|
| Equipment model / KG | Components, failure modes, topology, safety functions — the hypothesis space |
| Protection logic | Trip setpoints, barrier states, permissive logic — design response boundaries |
| FMEA (as KG nodes and Chroma text) | Catalog of possible failure mechanisms per component class |
| SOPs (as prescriptive rules) | What should happen under given conditions; discriminating logic when threshold-specific |

### 2.5 Dual-role data elements

Some data elements carry contributions from more than one class. Policy must assign a primary role and route the secondary contribution to its correct dimension explicitly.

| Data element | Primary role | Secondary role | Policy |
|---|---|---|---|
| Work orders | Affects performance (the physical activity) | Monitors performance (as-found/as-left observation) | Activity → governance dimension; as-found observation → `contextual_score` at most, not `support_score` |
| Condition reports | Monitors performance (field observation) | Analyzes past degradation (preliminary cause assessment) | Preliminary cause text → `contextual_score` only; lower authority than a dedicated ECA/RCA |
| Environmental monitoring | Monitors performance (ambient readings) | Affects performance (if the condition is the candidate mechanism — Category F) | Resolves at candidate assignment: if the environmental condition is the candidate mechanism, shift to affects; if background context, retain as monitors |
| OE / similar events | Analyzes past degradation (causal analogy) | None — tier confidence multiplier already captures epistemic distance | Apply tier multiplier; never treat as direct observation of current equipment |

---

## 3. Routing (Layer 2)

### 3.1 Scoring dimensions mapped to epistemic classes

The table below describes the **target state** after the epistemics module is introduced — not the current pipeline, which routes all Chroma hits into the evidence dimension indiscriminately.

| Scoring dimension | Primary epistemic class | Current state | Target state |
|---|---|---|---|
| Structural | Characterizes the system | Correct — KG topology and failure mode applicability | Unchanged |
| Temporal | Monitors performance | Partially correct — TSKR blends monitors and analyzes terms in a flat sum | Separated into signal support score (monitors) and recurrence support score (analyzes) |
| Telemetry | Monitors performance | Correct — anomaly severity and pattern fit | Unchanged |
| Evidence | Analyzes past degradation | Incorrect — all Chroma hit types contribute regardless of class | Restricted to analyzes-class hits only; other classes routed per table in §3.2 |
| Governance | Affects performance | Correct — PM compliance posture | Unchanged |

The hard gates (physical plausibility, barrier logic, timeline consistency) are purely "characterizes the system" logic — they enforce the reference frame. This is why they are binary while sub-scores are continuous.

The composite score mixes two fundamentally different epistemic operations:
- **Scoring within the reference frame** — structural and governance dimensions
- **Weighing evidence against the reference frame** — temporal, telemetry, and evidence dimensions

### 3.2 Evidence blend routing table

The routing table below is the primary executable artifact of the epistemics module for Step 5c. It must be implemented as a **versioned config artifact**, not documentation. Every input must resolve to exactly one path (mutually exclusive, collectively exhaustive). Unit tests must cover ambiguous cases and missing-metadata cases explicitly.

| Epistemic class | Chroma hit types | `support_score` | `contextual_score` | `contradiction_score` | Excluded |
|---|---|---|---|---|---|
| Analyzes past degradation | ECA, RCA, OE docs, closed KG past events | ✓ | — | ✓ | — |
| Monitors performance | CR, WO as-found observations | — | ✓ | — | — |
| Affects performance | WO activities | — | — | — | ✓ (already in governance) |
| Characterizes the system — discriminating | FMEA with quantitative thresholds or diagnostic logic, SOP diagnostic steps | — | ✓ (bounded) | ✓ | — |
| Characterizes the system — plausibility | FMEA general text, manuals, bulletins | — | — | — | ✓ (already in structural) |

### 3.3 Fallback hierarchy for hybrid documents

CRs often lack `finding_status`. WOs may not have clean `ca_as_found_condition`. OE ingestion may be inconsistent across plants. Routing must define a deterministic fallback order when metadata is missing or inconsistent — otherwise identical records can route differently depending on ingestion quality.

**Priority chain:**

```
finding_status → authority_level → doc_type → default_class
```

**Degraded classification flag:**

| Routing level used (`classification_resolution_level`) | `degraded_classification` | Reason |
|---|---|---|
| `finding_status` | False | Semantic field tied to epistemic intent |
| `authority_level` | False | Semantic field tied to epistemic intent |
| `doc_type` | **True** | Syntactic proxy only — epistemic intent not confirmed |
| `default_class` | **True** | No metadata available |

`authority_level` is still a semantic field — fallback to it is acceptable without flagging. Fallback to raw `doc_type` is where silent epistemic drift begins.

**`classification_resolution_level` field:** every annotation must record which level was used (`"finding_status"` / `"authority_level"` / `"doc_type"` / `"default"`), not just whether it was degraded. This gives auditability, debugging leverage, and plant-to-plant comparability.

When `degraded_classification = True`, this must be surfaced in `run_manifest.epistemics_summary` and counted per artifact type. A high degraded-classification rate is a signal that the plant's document ingestion pipeline needs metadata enrichment before epistemics routing can be trusted.

### 3.4 Allen blend routing

The temporal blend in Step 5c currently applies `allen_base_score` from all signal sources equally. Corrected routing:

```python
# Current — all signals contribute equally
temporal_refined = max(old_temporal, 0.75 × old_temporal + 0.25 × allen_base_score)

# Corrected — only affects-class signals raise the causal score
allen_causal_score = allen_base_score if source_class == "affects_performance" else 0.0
temporal_refined   = max(old_temporal, 0.75 × old_temporal + 0.25 × allen_causal_score)
```

Monitors signals' Allen relations are still computed and retained — they continue to feed the timeline consistency gate and timeline construction for the analyst. They do not raise the temporal causal score.

### 3.5 TSKR blend restructuring

The current TSKR flat blend conflates monitors and analyzes contributions:

```
raw = (0.45 × anomaly_score          # monitors performance
     + 0.30 × onset_score            # monitors performance
     + 0.10 × chain_score            # monitors performance
     + 0.10 × history_score          # analyzes past degradation
     + 0.15 × anomaly_count_score    # monitors performance
     + 0.10 × lag_consistency_score) # monitors performance
```

Target state — two explicit intermediate scores:

- **Signal support score** (monitors): anomaly, onset, chain, anomaly count, lag consistency. Answers: *what did the signals show, and does the timing fit?*
- **Recurrence support score** (analyzes): history score from past events. Answers: *is this a known pattern?*

Combined with explicit weights into the temporal sub-score. The weights need not change immediately — the grouping makes the epistemics independently testable and interpretable.

Alarm and SOE contributions to TSKR are restricted to onset timing only — feeding onset score and lag consistency score, not anomaly score or anomaly count score.

---

## 4. Control (Layer 3)

### 4.1 Three design invariants

These invariants must be stated explicitly in the workflow document and enforced in implementation. They protect the robustness of the pipeline against future drift.

**Invariant 1 — KG-anchored hypothesis space**
The candidate-generation backbone remains equipment-model anchored. No data element from any epistemic class can generate a hypothesis outside the equipment model.

**Invariant 2 — Deterministic, versioned routing**
Epistemic routing is deterministic and versioned. The same input with the same metadata always produces the same routing decision. `policy_version` covers both the epistemics config and the engine's hint-mapping table and is stamped on `run_manifest.pipeline_config`.

**Invariant 3 — Analyzes-class inputs can only modify scores, never expand the hypothesis space**
No path exists from `similar_event_list` → `generate()`. No path exists from OE documents → new failure mode creation. Analyzes-class inputs calibrate confidence in existing candidates; they do not generate new ones. This invariant is especially important as LLM OE retrieval becomes richer — it protects the physical grounding of the system.

### 4.2 "Observationally strong but causally ungrounded" state

A candidate can accumulate high scores on structural, temporal, and telemetry dimensions purely from KG plausibility and monitors signals, with zero analyzes or affects support, and still rank highly. The epistemics module fixes the evidence dimension but leaves this indirect path open through score accumulation. This state must be named and controlled explicitly.

**Defining `affects_support` computably**

`affects_support` is not any affects-class signal in the run — it is specifically affects-class signals tied to the candidate's component in the precursor window. A PM non-compliance on a different component does not ground a hypothesis about this one.

```python
affects_support = any(
    element.source_class == "affects_performance"
    and element.component_id == candidate.component_id
    and element.timestamp within event.precursor_window
    for element in annotated_elements
)
```

This covers: PM non-compliance on the candidate component within the window; operational context deviation tied to the candidate component's operating envelope; configuration changes that touched the candidate component.

> **v1 limitation — upstream affects not included in grounding test.** Many real causes are upstream (Category B and C): a support system failure or a common-cause batch defect acting on the candidate component will have `component_id` pointing to the upstream component, not the candidate. These are excluded from `affects_support` in v1 unless explicitly linked via KG topology. That linkage is a second-pass enrichment, not a first-version requirement. This definition must not be read as complete — it is conservative by design. Upstream grounding will require a topology-aware extension in a future version.

**Defining `analyzes_support` computably**

```python
analyzes_support = count(
    element.source_class == "analyzes_past_degradation"
    and element.component_id == candidate.component_id
    and element.fm_id == candidate.fm_id
    and not element.superseded
    for element in annotated_elements
)
```

**Trigger condition (strictly rule-based, not heuristic):**

```python
observationally_ungrounded = (
    analyzes_support == 0
    and affects_support == False
    and candidate.composite_score >= ranking_threshold
)
```

**Control behavior:**

- `confidence_label` capped at **medium** — hard cap, not a default
- `analyst_attention_flag` fired: *"candidate is observationally strong but causally ungrounded — no affects-class precursor and no analyzes-class conclusion support this hypothesis"*
- Analyst explicit acknowledgment required before this candidate can be selected as primary hypothesis and before writeback is allowed
- This state **cannot be overridden** by the confidence override mechanism (§4.3) — it is a harder control than the confidence policy

### 4.3 Confidence label policy and override mechanism

**Default policy:**

A candidate requires at least one analyzes-class element and at least one affects-class element in the precursor window for a `confidence_label` of "high." This is a policy choice, not an engineering truth — it will be validated against TC-1 through TC-7 before being treated as fixed.

**Override eligibility** (rule-based, not discretionary):

Both conditions must be met:
1. Temporal sub-score and telemetry sub-score both exceed threshold X (to be calibrated against show-and-tell set)
2. No contradictions from any hard gate

**Override recording** (for audit):

```
confidence_label = "high"
confidence_override_applied = True
causal_grounding_absent = True   # preserved even with override
```

The override does not remove `causal_grounding_absent` — downstream consumers and auditors must be able to see that the high confidence label was reached without full epistemic grounding. Analyst acknowledgment is required before writeback when `confidence_override_applied = True`.

### 4.4 Threshold recalibration as a design-coupled artifact

The `minimum_evidence_threshold` recalibration is not a later tuning step — it is part of the design. The epistemics routing and the threshold value are inseparable: shipping one without the other produces a pipeline that is logically correct but operationally broken because many runs will fall below threshold due to the removal of FMEA and CR score inflation.

**Calibration profile structure:**

```yaml
calibration_profile:
  name: string
  version: string
  baseline_data_coverage_signature:
    has_eca: bool
    has_rca: bool
    has_cr_with_finding_status: bool
    has_wo_with_as_found: bool
    chroma_doc_type_distribution: dict
  derived_thresholds:
    minimum_evidence_threshold: float
    confidence_override_threshold_temporal: float
    confidence_override_threshold_telemetry: float
```

**Key constraint:** a calibration profile is only valid for runs whose data coverage signature is compatible with the profile's `baseline_data_coverage_signature`. A profile calibrated on a rich ECA/CR environment is not valid for a CR-only environment. If a run's coverage signature does not match the active profile, this must be flagged in `run_manifest.pipeline_config`.

**Procedure:** calibration runs TC-1 through TC-7, establishes threshold values, names the profile, and stamps it on all subsequent `run_manifest.pipeline_config` records. The profile is a versioned artifact stored alongside `EpistemicsConfig`.

---

## 5. Tension point 1 — Alarms and SOE as causal candidates

### 5.1 The original problem

In the Allen relation map (workflow §2.3), an alarm that **PRECEDES** the event receives base score 0.75 and `causal_candidate = True`. The pipeline blends this into the candidate's temporal sub-score in Step 5c:

```
temporal_refined = max(old_temporal, 0.75 × old_temporal + 0.25 × allen_base_score)
```

The consequence is that an alarm preceding an event raises the causal score of any associated candidate — treating temporal precedence as causal evidence. A configuration change that preceded the event and an alarm that fired three minutes before trip were treated as equivalent causal evidence. They are not.

### 5.2 How the four-way classification resolves it

Alarms and SOE records belong unambiguously to **monitors performance**. The resolution follows from class membership alone — no KG topology check is needed. The corrected Allen blend (§3.4) applies `allen_base_score` only for affects-class sources. Monitors signals continue to feed the timeline consistency gate and analyst timeline but do not raise the temporal causal score.

### 5.3 What stays the same

Alarms and SOE records retain their role in scope construction at Step 0 — a scoping function, not an epistemic one.

---

## 6. Tension point 2 — FMEA double-counting

### 6.1 The original problem

Every candidate is anchored to a KG failure mode node (characterizes the system). Chroma also holds FMEA text as retrievable documents. When Chroma returns an FMEA chunk, the engine treats it as supporting evidence, raising the evidence sub-score. But an FMEA chunk only says "this failure mode is plausible for this component class" — which the KG already asserted. The candidate receives credit in the evidence dimension for something the structural dimension already captured. In FMEA-heavy indexes with sparse plant-specific CRs and ECAs, this caused candidates to rank artificially high.

### 6.2 How the four-way classification resolves it

FMEA content belongs to **characterizes the system** — the same class as the KG itself. It cannot contribute new information to the evidence dimension. Per the routing table (§3.2), FMEA general text is excluded from the evidence blend entirely; FMEA discriminating content (quantitative thresholds, diagnostic logic) routes to `contextual_score` bounded, and may contribute to `contradiction_score` when a candidate violates a specified threshold.

`support_score` now only receives analyzes-class content — formal causal conclusions about real degradation events on real equipment. The evidence dimension becomes genuinely discriminating between "this failure mode is theoretically possible" and "this failure mode has been formally analyzed on this equipment."

### 6.3 Downstream effect

Candidates supported only by FMEA text and CR observations will score lower on the evidence dimension and may not clear `minimum_evidence_threshold`. This is correct behavior — and it is why threshold recalibration (§4.4) is design-coupled, not optional.

---

## 7. RCA workflow impact — three affected areas

The four-way classification and the SE mental model produce a structured inference the pipeline should mirror: the reference frame defines what is possible, affects data generates hypotheses, monitors data sequences and constrains them, analyzes data calibrates confidence.

### 7.1 SE mental model — reasoning principles

The SE's fundamental question is: **what changed?** The four classes map onto four successive cognitive steps:

- *"What characterizes this equipment and its design basis?"* — **Characterizes the system.** The reference frame. Non-negotiable: if a candidate violates it, the candidate is ruled out regardless of signals.
- *"What was acting on the equipment in the precursor window?"* — **Affects performance.** The candidate causes.
- *"What did we observe about the equipment's state?"* — **Monitors performance.** These sequence and constrain candidates. An alarm does not tell you what caused the failure — it tells you when the failure became detectable.
- *"What have we concluded about similar degradations before?"* — **Analyzes past degradation.** These calibrate confidence. Past analysis raises or lowers confidence in a candidate but does not generate new ones.

The most common reasoning errors from mixing classes: conflating detection timing with causation (treating a PRECEDES alarm as causal evidence); conflating plausibility with observation (treating an FMEA entry as evidence that a failure actually occurred).

Applied to scoring: the SE would distinguish *"what did the signals show?"* (monitors, timing and constraint) from *"is this a known recurring pattern?"* (analyzes, confidence calibration). These are different cognitive operations and must not be summed in a flat blend.

### 7.2 Step 4: Candidate generation (Phase A epistemics)

**What Step 4 currently does**

`generate()` scores every failure mode from the KG on five dimensions using inputs available before Chroma retrieval exists: `kg_context`, `tskr_patterns`, `telemetry_summary`, `signal_evidence`, `operational_context`, `pm_compliance`, `alarm_log`, `soe_log`. Every input at this stage belongs to characterizes the system, monitors performance, or affects performance. Nothing from analyzes past degradation exists yet except the history score component of TSKR.

**Deviations from the SE mental model**

*Deviation 1 — TSKR flat blend conflates monitors and analyzes:* the six-term weighted sum mixes five monitors terms with one analyzes term (history score) without distinguishing their epistemic class. A high temporal score cannot be attributed to strong signal evidence vs recurrence history. Target state: restructure into signal support score and recurrence support score per §3.5.

*Deviation 2 — Alarm and SOE misuse in TSKR:* `alarm_log` and `soe_log` feed TSKR for onset timing but are currently used interchangeably with telemetry anomalies in anomaly scoring. Target state: restrict alarm and SOE contributions to onset score and lag consistency score only.

*Deviation 3 — Affects contributions split across dimensions:* the operating-point delta within structural and PM compliance within governance are both affects-class but split across dimensions for implementation reasons. This is a labeling inconsistency, not a scoring error. Out of scope for the epistemics module.

### 7.3 Step 5c: Evidence re-scoring / `refine_with_evidence` (Phase B + C epistemics)

**What the SE mental model says should be happening here**

By Step 5c the SE has a ranked candidate list. The question is: *"does what we know from past analyses of similar degradations confirm or contradict these candidates?"* This is purely an analyzes-class operation — calibrating confidence in existing candidates.

**Deviations from the SE mental model**

*Deviation 1 — `support_score` receives all Chroma hit types indiscriminately:* ECAs, RCAs, CRs, WOs, FMEAs, SOPs, OE documents all contribute to `support_score`. Target state: routing table in §3.2 applies — only analyzes-class hits enter `support_score`.

*Deviation 2 — Allen blend raises causal score from monitors signals:* corrected per §3.4.

*Deviation 3 — No supersession within analyzes class:* a preliminary CR assessment and a formal ECA on the same component and failure mode can both contribute to `support_score` independently. Target state: `resolve_supersession` (host TBD — see §8 open decisions) — when both a CR and an ECA exist for the same component/FM pair, the CR is marked superseded on the causal question; its `support_score` contribution is zeroed and it is retained as provenance only.

**Consequence for dual-threshold check**

Candidates supported only by FMEA text and CR observations will score lower and may not clear `minimum_evidence_threshold = 0.35`. This is correct. Recalibration per §4.4 is required before production runs.

**"Observationally strong but causally ungrounded" — indirect path**

Even with correct evidence routing, a candidate can still accumulate high scores on structural, temporal, and telemetry dimensions from KG plausibility and monitors signals alone. The control behavior in §4.2 applies: `observationally_ungrounded` flag, confidence capped at medium, analyst acknowledgment required.

### 7.4 Step 6d: Synthesis (Phase D + E epistemics)

**What the SE mental model says should be happening here**

The SE question shifts: *"given everything we know, what is the most defensible account of this failure, and what does it leave unresolved?"* A system engineer writing an RCA conclusion is always implicitly doing epistemic classification — the epistemics digest makes that reasoning explicit and auditable.

**Epistemics digest — minimal v1 schema**

The digest must be structurally constrained before implementation begins to prevent the LLM and fallback synthesis paths from diverging. A v1 schema based on counts and flags is sufficient to enforce consistency:

```yaml
epistemics_digest:
  candidate_id: string
  analyzes_support_count: int
  analyzes_support_items:
    - source_id: string
      authority_level: string        # "mandatory" | "guidance" | "informational"
      superseded: bool
  affects_support_present: bool
  affects_support_items:
    - source_id: string
      component_id: string
      within_precursor_window: bool
  observationally_ungrounded: bool
  causal_grounding_absent: bool      # preserved even when confidence_override_applied
  degraded_classification_count: int # count of hits that fell back to doc_type or default
  confidence_cap: "medium" | null    # set when observationally_ungrounded = true
```

This schema is the contract between the epistemics module and the synthesizer. Both the LLM prompt construction and the deterministic fallback template must consume this structure and not reconstruct epistemic roles from raw `doc_type` fields. The digest is also the primary input to `run_manifest.epistemics_summary`.

**Three synthesis deviations**

*Deviation 1 — Confidence language not grounded in evidence class:* `confidence_label` currently derives from composite score alone. A candidate scoring high on monitors and FMEA plausibility looks identical to one scoring high on ECA conclusions and affects-class precursors. Target state: two-factor confidence assessment per §4.3, with override mechanism and `causal_grounding_absent` flag preserved.

*Deviation 2 — Recommended actions don't distinguish causal depth by evidence class:* the SE mental model requires:
- Proximate causes supported by monitors evidence (observation) and analyzes evidence (past conclusion)
- Contributing causes supported by affects evidence (what was acting on the equipment)
- Root causes supported by systemic analyzes evidence (OE, recurrence patterns, organizational findings)

If a root cause candidate has no analyzes-class support, the synthesizer must flag the root cause assignment as analytically ungrounded.

*Deviation 3 — Unresolved gaps not typed by what would close them:* target state — gaps are typed by epistemic class:

| Gap type | Meaning | Action implied |
|---|---|---|
| Monitors gap | Signal or observation needed but absent | Instrument review, historian retrieval, alarm log recovery |
| Affects gap | Unknown activity or condition in precursor window | Maintenance record retrieval, shift log review, configuration history |
| Analyzes gap | No formal causal conclusion for a known past event | CR closure investigation, ECA commissioning, OE search |

**Epistemics digest per candidate**

Before synthesis runs, the module produces a structured digest per candidate:
- What analyzes-class conclusions support or contradict this candidate
- What affects-class data generated this candidate as a hypothesis
- What monitors-class data constrains its timing and severity
- What characterizes-class data bounds what is physically possible

The synthesizer consumes this digest. The LLM path receives principled inputs for confidence language and causal depth. The deterministic fallback receives explicit rules. Both paths produce auditable outputs.

**Manifest consequence**

`run_manifest.epistemics_summary` records: counts by epistemic class per candidate, supersession edges count, `degraded_classification` counts by artifact type, `classification_resolution_level` distribution, and calibration profile reference.

---

## 8. Open decisions

**Resolved by the four-way classification:**

- ~~Exact discount multiplier for `redundant_with_kg` FMEA hits~~ — FMEA general text excluded from evidence blend entirely; no multiplier needed
- ~~Whether alarm `exclude_from_direct_support` requires KG topology check~~ — class membership is the rule; no topology check required
- ~~Secondary role math~~ — dual-role elements route each contribution to a different scoring dimension; no within-dimension blending of roles

**Still open:**

- **FMEA discriminating content threshold** — what field predicates in Chroma metadata reliably identify quantitative thresholds vs qualitative descriptions without re-running NLP at query time. Options: structured FMEA fields at ingest, lightweight keyword classifier, NER tagging at index time.
- **Confidence override thresholds X** — temporal and telemetry score floors for override eligibility; to be established by calibration against TC-1 through TC-7.
- **Supersession within analyzes class** — CR superseded by ECA/RCA is the simple case. Hard cases include conflicting ECAs and OE-vs-plant-RCA disagreement. A first-version authority hierarchy within the analyzes class: plant RCA > plant ECA > plant CR preliminary assessment > fleet OE > industry OE. When two elements conflict on the same component/FM pair, higher authority wins. Equal authority: more recent wins. Equal authority and equal recency: analyst flag required, no automatic resolution. Note: this hierarchy applies within a single component/FM pair — conflict handling across multiple components or FM pairs will need additional rules in a future version. Full hierarchy rules need ADR before implementation.
- **Calibration profile compatibility check** — first-version rule: flag a mismatch if any field in `baseline_data_coverage_signature` that is `True` in the profile is `False` in the run's actual coverage. Conservative but safe. Fields that most materially affect threshold behavior: `has_eca`, `has_cr_with_finding_status`, and whether the analyzes-class fraction of Chroma hits exceeds a minimum proportion. A run with no ECAs must not use a profile calibrated on an ECA-rich environment. Note: this rule is intentionally strict and will trigger frequently in early deployments — that is acceptable. Noisy flagging during rollout is preferable to silent miscalibration. Flag frequency should be monitored and used to drive ingestion pipeline improvements, not to relax the compatibility rule.
- **`resolve_supersession` host** — orchestrator (natural hook after Allen map, before `refine_with_evidence`) vs engine (if engine ingests single merged evidence list). Needs ADR.
- **JSON shape for stored annotation** — snake_case nested `epistemic` sub-object on each evidence hit; whether secondary roles are a list or weighted dict.
- **Root cause analyzes-gap flagging threshold** — how much analyzes-class support is required before a root cause assignment is considered grounded; whether this is a hard flag or a graduated attention flag.

---

## 9. Reproducibility risks

**Risk 1 — Policy version coverage**

`EpistemicPolicyOutput` hints (`evidence_prior_bias`, `refinement_bias`) are applied by the engine. If the engine's hint-mapping changes without an epistemics config version bump, reproducibility breaks invisibly. `policy_version` must cover both the epistemics config and the engine's mapping table, stamped on `run_manifest.pipeline_config`.

**Risk 2 — Threshold recalibration between runs**

`minimum_evidence_threshold` and confidence override thresholds are versioned configuration parameters within the calibration profile (§4.4). Changing them between runs changes rankings without a code change. The calibration profile name and version must be stamped on `run_manifest.pipeline_config` for every run. A run whose coverage signature does not match the active calibration profile must be flagged.

**Risk 3 — Silent epistemic drift from degraded classification**

When routing falls back to `doc_type` or `default_class`, identical records may route differently across plants and ingestion runs. `classification_resolution_level` recorded on every annotation and `degraded_classification` counts in the manifest are the primary defenses. A high degraded-classification rate must trigger an ingestion quality review, not silent acceptance.

---

## 10. Document cross-references

| Document | Relevant sections |
|---|---|
| `rca_workflow_reference_guide_april_25.md` | §2.1 (data inventory), §2.2 (workflow steps), §2.3 (Allen relations), §2.4 (pattern recognition), §5.6 (Step 4), §5.8 (Step 5), §5.12 (Step 6) |
| `rca_architecture_assessment_and_epistemics_april_26.md` | §3.0 (data element taxonomy), §3.3 (seams), §4.4 (data model), §4.8 (integration phases), §4.12 (open decisions), §4.13 (workflow alignment), §4.14 (final check) |
| `Architecture_Assessment.md` (april_20) | Document-type treatment, FMEA double-counting, SOP as discriminating logic |
