# RCA Workflow — Systems Engineering Review
**Reviewer perspective**: Systems Engineer / RCA Practitioner (nuclear power plant)
**Date**: April 20, 2026
**Baseline**: Orchestrator v3.2, Schema set v3.2, `RCA_workflow_april_2.md` (April 6)
**Scope**: Premises soundness · Logic completeness · Data-assessment functional linkage

---

## 1. Overview Assessment

The core workflow is well-structured and architecturally sound for a **decision-support tool**. The multi-stage artifact chain, dual scoring pass (pre/post evidence), schema-validated artifact contracts, and deterministic fallback path reflect careful engineering.

However, from the standpoint of a practicing RCA engineer who must defend conclusions to a licensing authority, there are **three structural concerns** that cut across the entire workflow, plus a set of specific logic gaps and linkage breaks. These are organized below in order of RCA consequence severity.

---

## 2. Premises and Assumptions — Soundness Review

### 2.1 Closed-World Assumption on the KG (HIGH RISK)

**Premise**: The causal search space is bounded by the failure modes present in the KG neighborhood of the affected asset. If a failure mode is not in the KG, it cannot be generated as a candidate.

**Problem**: This is a **closed-world assumption** — the system can only find what it already knows. For novel, first-of-kind failure modes (first occurrence at this plant, failure mode not yet coded into the FMEA/KG), the pipeline will generate zero valid candidates and fall through to `insufficient_evidence`. The engineer receives no diagnostic signal that the cause is novel — only that confidence is low.

**RCA consequence**: In the reference test case (condenser vacuum loss, EVT-U2-2024-0847), if "expansion joint thermal fatigue" were not a coded failure mode in the KG, the system would miss the true root cause entirely. There is no mechanism to prompt the engineer to extend the search beyond the KG.

**Recommendation**: When zero candidates survive Stage D filtering, the synthesizer should explicitly flag **"search space exhaustion"** as a distinct executive summary status (separate from "insufficient_evidence"), prompting the engineer that the true cause may lie outside the current KG.

---

### 2.2 Telemetry Validity Assumption (MEDIUM RISK)

**Premise**: TSKR (Stage C) uses telemetry anomaly timestamps as the ground truth for failure propagation timing. The anomaly pattern and latency windows are taken at face value.

**Problem**: No sensor validity check is performed before or during temporal scoring. A measured anomaly could be caused by:
- Sensor drift or calibration error (the anomaly is the instrument, not the process)
- Spurious trip or out-of-calibration condition
- Process noise above a poorly-set threshold

If the telemetry anomaly is instrument-induced, the TSKR latency computation is meaningless, and any failure mode whose temporal signature happens to align with the spurious anomaly will receive an undeserved temporal boost.

**RCA consequence**: A false temporal alignment could elevate a wrong candidate above the true cause. This is especially dangerous for instrumentation-class failures (which are common in nuclear plants) where the anomaly itself is the failure mode, not a symptom of a downstream mechanical failure.

**Recommendation**: The `telemetry_summary` schema should carry an `instrument_validity_flag` (calibrated / out-of-cal / under-investigation) per signal. The TSKR scorer should reduce pattern confidence when the validity flag is not "calibrated." This is a data-model gap, not just a code gap.

---

### 2.3 FMEA Data Currency Assumption (MEDIUM RISK)

**Premise**: `kg_context.failure_modes[].expected_latency_min_hours`, `expected_latency_max_hours`, `expected_symptoms`, and `expected_anomaly_pattern` are assumed accurate and current.

**Problem**: FMEA data in the KG is populated at plant startup or during outages. Between updates, plant modifications (new components, changed service conditions, different operating modes) can invalidate the expected latency and symptom data. The KG has no FMEA revision date per failure mode, and no mechanism warns the scorer that a latency window may be stale.

**RCA consequence**: A latency violation (`latency_violation_type: too_fast`) might be diagnosed as contradicting evidence against a failure mode, when in reality the FMEA window is outdated and the actual failure occurred well within plausible physics.

**Recommendation**: Add `fmea_revision_date` and `fmea_confidence_level` (validated / preliminary / inherited) to the `failure_modes[]` schema. Candidates with preliminary or inherited FMEA data should have their temporal score uncertainty flagged explicitly in the TSKR output.

---

### 2.4 Scoring Weight Calibration Assumption (MEDIUM RISK)

**Premise**: The 5-dimensional scoring weights (structural 0.30, temporal 0.20, telemetry 0.20, evidence 0.20, governance 0.10) are assumed to be appropriate priors for nuclear equipment RCA.

**Problem**: These weights are engineering judgments, not empirically calibrated values. There is no feedback mechanism connecting analyst acceptance/rejection of RCA conclusions to weight refinement. A weight configuration that systematically over-weights structural proximity (KG topology) relative to evidence quality could produce confidently-wrong rankings.

**Concrete risk**: The structural dimension weights KG topology proximity — a failure mode that is topologically close to the affected asset will score well structurally even if there is no physical causal pathway. KG topology is not the same as causal proximity.

**Recommendation**: Define a weight validation protocol using the reference test case suite (Section 12 of the spec). Track analyst override rates per scoring dimension across real RCA runs to identify dimensions that are consistently wrong. Document the weight provenance in the scoring configuration schema.

---

### 2.5 Single Primary Cause Assumption (HIGH RISK)

**Premise**: The pipeline resolves to ONE primary hypothesis plus alternatives. The rca_card structure has exactly one `primary_hypothesis` block.

**Problem**: Nuclear equipment events frequently involve **concurrent contributing causes** — a degraded component that failed because of a procedural deviation compounded by adverse operating conditions. These are not alternative hypotheses (one is right, the others are wrong) — they are **co-necessary causes** that must ALL be addressed in the corrective action program to prevent recurrence.

The current architecture can represent this situation only as "mixed" evidence posture or through the alternatives block with a weak "reason_not_primary" narrative, but it provides no formal mechanism to declare two or more causes as jointly necessary.

**RCA consequence**: A corrective action that addresses only the primary hypothesis may fail to prevent recurrence if a contributing cause is left unaddressed. INPO and IAEA RCA guidance explicitly requires identification of contributing causes as a separate structured output, not just a de-prioritized alternative.

**Recommendation**: Add a `contributing_causes[]` array to the `rca_card` schema, distinct from `alternatives[]`. Contributing causes are causes that are necessary but insufficient on their own. Each should carry the same candidate linkage, evidence citation, and recommended action structure as the primary.

---

## 3. Logic — Correctness and Completeness

### 3.1 No Feedback Loop Between Evidence and KG Context (LOGIC GAP)

**The gap**: Stage B (KG context) defines the causal search space. Stage E retrieves evidence only for candidates generated by Stage D. Stage F can re-rank candidates based on evidence, but the KG context that generated them (Stage B) was computed before any evidence was seen.

If Stage F demotes candidate A and promotes candidate B, the evidence that elevated B was retrieved using A-focused queries (from Stage D). The KG neighborhood, failure mode set, and document selection were all optimized for A, not B. The evidence for B is therefore incomplete.

**RCA consequence**: The final primary hypothesis could be supported by evidence retrieved under an incorrect initial framing. A second-pass KG expansion focused on the post-evidence winner would retrieve more relevant failure modes and documents.

**Note**: This is an architectural trade-off that is acceptable for decision-support (the analyst is expected to exercise judgment), but it must be surfaced explicitly in the analyst_attention_flags when the post-evidence primary differs from the pre-evidence primary.

---

### 3.2 Ishikawa as Post-Hoc Translation, Not Investigation Driver (LOGIC GAP)

**The gap**: Stage G (Ishikawa) is described explicitly as "translation, not new inference." The Ishikawa diagram is produced after all causal reasoning is complete and serves as a structured presentation of conclusions already reached.

**Problem from RCA methodology standpoint**: In formal RCA practice (INPO AP-923, NRC RIS 2006-13), the Ishikawa (fishbone) diagram is the **investigation framework** — it defines which cause categories to investigate before data collection begins. Using it only as a post-hoc translator inverts the intended methodology and means the investigation may not have systematically explored all Ishikawa branches.

**Specific concern**: The `maintenance_human_factors` branch of the Ishikawa is populated from evidence snippets mentioning maintenance keywords. There is no structural guarantee that the human performance dimension was systematically explored during candidate generation — it can only surface if a maintenance-related document was retrieved. A human performance cause with no documentation trail will produce an empty `maintenance_human_factors` branch and be silently missed.

**Recommendation**: The Ishikawa categories should drive **evidence retrieval query planning** in Stage E. For each Ishikawa branch, generate at least one targeted evidence query, independent of whether candidates were generated for that branch. This transforms the Ishikawa from a translator into a systematic investigation completeness check.

---

### 3.3 Past Event Analogs as Primary Hypothesis — Logical Conflict (LOGIC GAP)

**The gap**: `hypothesis_type: "past_event_analog"` candidates can be ranked as the primary hypothesis.

**Problem**: A past event is not a causal mechanism — it is evidence of recurrence pattern. Asserting "past event EVT-2024-001 is the primary hypothesis" is logically meaningless: it says the current event resembles a past event, but says nothing about the current event's cause. The true root cause is the same failure mode that caused the past event (or a new failure mode if conditions changed).

**RCA consequence**: An rca_card with `primary_hypothesis.hypothesis_type = "past_event_analog"` would fail technical review because it does not identify a mechanism. An engineer submitting this to the CAP would have it returned.

**Recommendation**: Past event analogs should be evidence, not candidates. They should appear in the `evidence[]` block of the rca_card with `support_role: "supporting"` and `source_type: "past_event"`, and should inform the recurrence_score of the failure mode candidate they match, not compete with failure mode candidates for primary position.

---

### 3.4 Confidence Always Capped at Medium — Calibration Is Broken (LOGIC GAP)

**The gap**: `fallback_used: True` always (because DummyLLMClient always fails). The confidence calibration rule caps at "medium" when fallback is used. Therefore, **no RCA run can ever produce a high-confidence conclusion** in the current system.

**RCA consequence**: Engineers will learn that the confidence label is always "medium" and stop using it as a discriminator. A case with overwhelming supporting evidence, no contradictions, a clear temporal alignment, and a known recurrence pattern will be labeled identically to a case with weak evidence and ambiguous temporal signals. The confidence label becomes noise.

**This is not just a technical debt item — it is an active quality degradation.** The deterministic fallback path produces conclusions that are just as grounded as the LLM path (and more reproducible). The confidence cap should not apply to the fallback path; it should apply only when confidence would otherwise rely on LLM-generated narrative that cannot be independently verified.

**Recommendation**: Remove the fallback confidence cap. Apply the confidence cap only when `rca_card.primary_hypothesis.narrative` contains LLM-generated text that has not passed semantic validation.

---

### 3.5 Recommended Action Priority Not Driven by Safety Significance (LOGIC GAP)

**The gap**: Recommended actions carry a `priority` field (critical/high/medium/low) and `action_type`, but neither is derived from `kg_context.safety_functions[]` or any safety significance signal.

**Problem**: A failure mode affecting a safety-related function should always generate at least one `priority: critical` action, regardless of composite score. Currently, a high-scoring candidate with weak governance could generate a "low" priority action on a safety-related component because the synthesizer's fallback action priority logic is based on composite_score thresholds, not safety classification.

**RCA consequence**: The corrective action priority written back to the CAP may not reflect the actual safety significance of the finding. This is a regulatory concern — nuclear plant CAPs have priority tiers that must align with 10 CFR 50 Appendix B requirements.

**Recommendation**: Add a `safety_significance_override` rule to the recommended action generator: if any `affected_safety_functions[].category` == "safety_related" and the candidate is the primary hypothesis, the minimum priority for all linked actions is "high"; for "safety_critical" category, minimum is "critical."

---

### 3.6 Evidence Retrieval Cannot Rescue Filtered Candidates (STRUCTURAL GAP)

**The gap**: Candidates filtered out by Stage D's dual threshold (composite ≥ 0.30 AND evidence ≥ 0.35) never reach Stage E evidence retrieval. If the true root cause is filtered at Stage D because it has sparse pre-existing documentation (evidence_score < 0.35), no amount of actual retrieved evidence can reinstate it.

**Concrete case**: A novel failure mode in a well-maintained plant with few historical CRs might score correctly on structural, temporal, and telemetry dimensions but fail the evidence_score threshold because no documents about it exist yet. The dual threshold eliminates it before evidence retrieval could confirm it.

**Recommendation**: Relax the pre-evidence filtering: apply only the structural+temporal composite threshold at Stage D (composite ≥ 0.25), defer evidence-score threshold to Stage F where actual evidence is available. This avoids over-filtering on a dimension (evidence) that hasn't been evaluated yet.

---

### 3.7 No Propagation Path Physics Validation (STRUCTURAL GAP)

**The gap**: Each candidate carries a `kg_path` — the graph traversal path from root cause node to affected asset. This path is used in structural scoring (topology proximity) but is never validated for physical plausibility.

**Problem**: KG edges carry relationship types (flows_to, controlled_by, serves, part_of, etc.) but these types do not carry propagation physics (energy transfer mechanism, flow medium, signal type). A path that traverses `component_A → serves → component_B → flows_to → component_C` may be topologically short but physically implausible as a causal chain.

**RCA consequence**: The structural score rewards topological proximity, but proximity in the KG does not guarantee a physically plausible causal mechanism. A high structural score is not equivalent to a plausible failure path.

**Recommendation**: Document this limitation explicitly in the `score_rationale["structural"]` narrative for each candidate so the engineer can assess KG path plausibility during review.

---

## 4. Data-Assessment Functional Linkage

### 4.1 Safety Function Impact Siloed — Never Flows to Output (LINKAGE BREAK)

**Observation**: `kg_context.safety_functions[]` is populated with affected safety functions. This information is referenced in TSKR and candidate scoring but **never propagates forward** to:
- `rca_card.primary_hypothesis.why_primary[]`
- `rca_card.recommended_actions[].priority`
- `rca_card.executive_summary.analyst_attention_flags[]`
- `run_manifest.review_hooks`

An engineer reading the rca_card has no direct indication that the primary failure mode affects a safety function unless they separately examine the kg_context artifact. The rca_card should be self-contained for regulatory review.

**Recommendation**: The synthesizer should check `candidate.affected_safety_functions` and surface any safety function impact in `analyst_attention_flags` and in `why_primary[]` narrative. This is a functional linkage gap with regulatory consequence.

---

### 4.2 Score Evolution (v1 → v2) Not a Named Artifact (LINKAGE BREAK)

**Observation**: The spec states: *"The delta between v1 and v2 ranking is the most important diagnostic signal for whether evidence retrieval is providing discriminating value."* Yet this delta is **not a named artifact**. Comparing v1 vs. v2 requires loading two JSON files and performing a diff.

**Consequence**: The most diagnostic signal in the entire pipeline — whether the evidence retrieval is working — is invisible to the engineer in the standard workflow. The run_manifest captures final state only.

**Recommendation**: Add `scoring_evolution` to the run_manifest or as a named artifact:
```json
{
  "candidate_id": "FM::FM-001",
  "v1_rank": 1,
  "v2_rank": 3,
  "v1_composite": 0.71,
  "v2_composite": 0.48,
  "evidence_delta": -0.35,
  "posture_change": "supported → contradicted"
}
```
This is essential for analyst review and for diagnosing retrieval quality.

---

### 4.3 CMMS Records Not Explicitly Linked to Candidates (LINKAGE BREAK)

**Observation**: CMMS CR/WO records from Stage 5B are injected into the Chroma vector store and retrieved through semantic similarity. The `cmms_context` artifact carries the raw records. The `evidence_bundle` may contain snippets derived from CMMS records.

**Problem**: There is no explicit cross-reference between:
- `cmms_context.cr_records[i].cr_id` → `evidence_bundle.snippets[j].doc_id` → `causality_candidates.v2.supporting_evidence_refs[]`

An engineer reviewing the RCA cannot trace "this specific WO → this evidence snippet → this score change" without manually correlating three artifacts by doc_id string matching.

**Recommendation**: When CMMS records are embedded and indexed in Chroma, the `doc_id` used for the Chroma record should be explicitly recorded in `cmms_context.cr_records[].chroma_doc_id`. This creates a traceable audit chain from CMMS source record to evidence score contribution.

---

### 4.4 Analyst Override Audit Trail Incomplete (LINKAGE BREAK)

**Observation**: The `analyst_override_processor` applies analyst decisions to the rca_card. The override is persisted as an artifact. However, the override artifact records **what** was overridden (final decision) but not **what changed** in the rca_card as a result.

**Regulatory concern**: For nuclear plant CAP, the audit trail must show the original system recommendation, the analyst's override, and the specific reasoning. If an analyst overrides the primary hypothesis from "bearing degradation" to "seal failure," the audit record must show both states and the analyst's justification.

**Recommendation**: The override processor should produce a `diff` section in the `analyst_override` artifact:
```json
{
  "field": "primary_hypothesis.candidate_id",
  "system_value": "FM::FM-001",
  "analyst_value": "FM::FM-005",
  "analyst_rationale": "..."
}
```

---

### 4.5 Evidence Snippets vs. Summary — What Does the Engineer See? (LINKAGE BREAK)

**Observation**: The `evidence_bundle` carries both:
1. Per-candidate summary (`supporting_count`, `best_support_score`, snippet IDs)
2. Actual snippet objects (`snippet_id`, `text`, `doc_id`, `page`)

The synthesizer's fallback path uses **summary statistics**, not actual snippets, to construct `rca_card.evidence[]` rows. The `excerpt` field in each evidence row is populated from summary-derived text, not from the actual retrieved snippet.

**Consequence**: The engineer reading the rca_card sees evidence rows with excerpts that are summary descriptions, not verbatim document text. They cannot navigate to the source document from the rca_card. The evidence rows appear more grounded than they are.

**Recommendation**: The fallback synthesizer should populate `rca_card.evidence[].excerpt` from `evidence_bundle.snippets[].text` (verbatim, truncated to N characters) and `rca_card.evidence[].source_id` from `evidence_bundle.snippets[].doc_id`. The analyst must be able to navigate from the rca_card to the source document.

---

### 4.6 Recommended Actions Not Validated Against Failure Mode Characteristics (LINKAGE BREAK)

**Observation**: Recommended actions carry `linked_candidate_id` and `target_component_id`, but there is no validation that the `action_type` is appropriate for the failure mode's characteristics.

**Examples of functional disconnects**:
- A failure mode with `evidence_posture: "contradicted"` could still generate `action_type: "immediate_corrective"` actions
- A failure mode with `decision_status: "insufficient_evidence"` could generate `priority: "critical"` actions
- A past_event_analog candidate (logically problematic as noted in §3.3) could generate actions against a component even if the failure mechanism is not established

**Recommendation**: Add validation gates in the recommended action generator:
- If `evidence_posture` is "contradicted" or "weak": action_type must be "monitoring" or "engineering_evaluation", not "immediate_corrective"
- If `decision_status` is "insufficient_evidence": no action_type of "immediate_corrective" or "long_term_corrective" permitted; must be "engineering_evaluation"
- Actions linked to past_event_analog candidates must explicitly note the hypothesis is recurrence-based, not mechanism-confirmed

---

### 4.7 Recurrence Model Missing Failure Rate Normalization (DATA MODEL GAP)

**Observation**: The recurrence model counts events and computes inter-event intervals. `mean_inter_event_days` is the primary recurrence metric.

**Problem**: Inter-event interval is not the same as failure rate. A component that failed once after 40 years of operation versus one that failed once after 3 months have the same `same_component_event_count = 1`, but radically different risk significance. Operating hours, start/stop cycles, and equipment age are not captured in the recurrence model.

**Consequence**: The recurrence_score could rank a first-occurrence-in-40-years event the same as a fourth-occurrence-in-two-years event if both have similar inter-event intervals in the available history window.

**Recommendation**: Add `equipment_operating_hours` and `equipment_age_years` to the `past_events[]` schema. Normalize recurrence scores by operating exposure, not raw calendar time.

---

### 4.8 No Corpus Completeness Check Before Evidence Retrieval (DATA MODEL GAP)

**Observation**: Stage E retrieves evidence from the Chroma vector store without any check of corpus coverage for the event's asset, document types, or time window.

**Problem**: If the relevant document types (FMEA, ECA, CR, WO) are absent or sparsely represented for a specific component in the vector store, the evidence retrieval will return low-scoring or irrelevant results. The evidence_bundle will show low counts, and Stage F will penalize well-supported candidates. The engineer cannot tell whether low evidence counts mean "no evidence exists" or "relevant documents were not ingested."

**Recommendation**: Add a corpus coverage summary to the `evidence_bundle` metadata:
```json
{
  "corpus_coverage": {
    "asset_id": "PUMP-001",
    "doc_type_counts": {"CR": 12, "WO": 8, "FMEA": 0, "SOP": 2, "ECA": 0},
    "date_range_coverage": {"earliest": "2020-01-01", "latest": "2026-04-01"},
    "coverage_warning": "FMEA documents absent for this asset — temporal scoring may be unreliable"
  }
}
```

---

## 5. Summary of Findings

### Critical (regulatory consequence if unaddressed)

| ID | Finding | Section |
|----|---------|---------|
| C1 | Closed-world assumption — novel failure modes silently missed | §2.1 |
| C2 | Single primary cause architecture — concurrent causes not representable | §2.5 |
| C3 | Confidence always capped at medium — calibration broken in production | §3.4 |
| C4 | Recommended action priority not driven by safety significance | §3.5 |
| C5 | Safety function impact never propagates to rca_card output | §4.1 |

### High (investigation quality and audit defensibility)

| ID | Finding | Section |
|----|---------|---------|
| H1 | Past event analogs as primary hypothesis — logically invalid | §3.3 |
| H2 | Evidence retrieval cannot rescue Stage D filtered candidates | §3.6 |
| H3 | Score evolution (v1→v2) not a named artifact | §4.2 |
| H4 | Analyst override audit trail incomplete — no diff record | §4.4 |
| H5 | Evidence excerpts in rca_card are summaries, not verbatim text | §4.5 |
| H6 | Recommended actions not validated against failure mode evidence posture | §4.6 |

### Medium (correctness and data quality)

| ID | Finding | Section |
|----|---------|---------|
| M1 | Telemetry validity not checked before TSKR temporal scoring | §2.2 |
| M2 | FMEA data currency not tracked per failure mode | §2.3 |
| M3 | Ishikawa as post-hoc translator — human performance branch silently empty | §3.2 |
| M4 | No feedback loop from post-evidence winner to KG context | §3.1 |
| M5 | KG path not validated for physical plausibility | §3.7 |
| M6 | CMMS records not explicitly linked to candidates in evidence chain | §4.3 |
| M7 | Recurrence model missing failure rate normalization | §4.7 |
| M8 | No corpus completeness check before evidence retrieval | §4.8 |

---

## 6. Items Already Tracked (Confirmed Consistent with Code)

The following items in the open issues tracker (`RCA_workflow_april_2.md`, §15 / §13) are confirmed observed and consistent with the codebase as explored:

- Confidence label case mismatch (uppercase vs. lowercase) across schema files — **confirmed open**
- kg_context dual schema — **confirmed open, tolerated via stop_on_validation_error=False**
- TSKR index stores only first pattern per target_id — **confirmed open**
- BM25 unavailable on disk-loaded collections without warning — **confirmed open**
- LLM hallucinated candidate ID is warning, not hard error — **confirmed open**
- writeback_ready always False (fallback always used) — **confirmed**
- Defense-in-depth barrier analysis absent — **confirmed, no barrier concept in KG schema**
- No risk significance scalar from safety functions — **confirmed**
- Change analysis unstructured — **confirmed**
- Sensitivity analysis missing — **confirmed**

---

*End of review — April 20, 2026*
