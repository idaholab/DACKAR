# RCA WORKFLOW ASSESSMENT

## 1. Overall Logic Flow — Strengths and Structural Concerns

### What works well
The seven-step pipeline is logically ordered and the separation between hypothesis generation (KG-bound) and evidence retrieval (Chroma) is architecturally sound. Requiring every hypothesis to be anchored to a KG node before evidence is consulted is the right discipline — it prevents the retriever from inventing causes from document noise.
The dual-artifact approach (rca_card for analysts, run_manifest for auditors) is also correct. These serve genuinely different audiences and should not be collapsed.

### Structural concerns
The Allen map executes after Chroma retrieval, but the document describes it as Step 2c. This is not a minor sequencing note — it means that candidate generation (Step 4) and initial ranking happen without the full Allen temporal picture. Step 4 uses Allen scores embedded in tskr_patterns, which is a compressed proxy. If that proxy is wrong or incomplete, the wrong candidates go to Chroma, and the evidence retrieved is biased toward those candidates. The misalignment between conceptual order and execution order is a real architectural risk, not just a documentation problem. The document acknowledges it but doesn't justify why the Allen map can't be built before Step 4.
Auto-reentry (Step 5d) re-runs Steps 1–5 with a wider KG hop limit, but there is no mechanism to ensure the expanded KG doesn't introduce failure modes that are topology-adjacent but physically irrelevant. A wider neighborhood retrieves more nodes; more nodes generate more candidates; with no additional domain gate on relevance, the expanded run can dilute the ranking. The document doesn't describe how candidate explosion from reentry is controlled beyond top_k_candidates.
The scope revision mechanism (§2.5.3) is asymmetric in risk. Accepting an expansion adds components and runs Step 4's boundary filter on re-run. But the expansion decision is made by the analyst after seeing a run that was already shaped by the original scope. There is a circularity risk: the original scope determines what evidence was retrieved, which shapes what signals appear in the Allen map, which determines what expansion suggestions are generated. An analyst could rationally accept expansions that are artifacts of an incomplete first run rather than genuine causal leads.

## 2. Causal Reasoning — Issues and Limitations

### The five-dimension scoring blend

The composite score formula is:
0.30 × structural + 0.20 × temporal + 0.20 × telemetry + 0.20 × evidence + 0.10 × governance
Several concerns here:
The weights are static across all failure mode superclasses. A latent design deficiency (Category H) and an active bearing failure (Category A) are scored on the same dimensional blend with the same weights. For a design deficiency, temporal precedence is essentially guaranteed (the flaw predates the event by design), so the temporal sub-score provides no discrimination. For an equipment-internal failure, structural topology fit is less predictive than telemetry pattern. Static weights across radically different failure mode types will systematically over- or under-rank certain categories. The document acknowledges governance weight "can vary per failure mode superclass" but doesn't generalize this to other dimensions.
The evidence sub-score starts very low at Step 4 and jumps at Step 5. This is by design, but it creates a threshold discontinuity: minimum_pre_evidence_threshold is 0.10 at Step 4 and minimum_evidence_threshold is 0.35 after Step 5. A candidate that barely passes at 0.11 in Step 4 could be eliminated at 0.35 after refinement, while a candidate that was artificially boosted by a high structural score might survive Step 4 and then anchor the Chroma retrieval queries. The Chroma queries are built from the top-N Step 4 candidates, so the query space is already biased before evidence is consulted.
The raise-only rule for Allen scores in temporal refinement is asymmetric and potentially misleading. The blend 0.75 × TSKR + 0.25 × Allen can only raise a candidate's temporal score. This means a candidate with a FOLLOWS relation — a clear temporal contradiction — has its temporal_contradiction flag set, but the composite score itself is not lowered by the Allen signal; the hard gate is the only mechanism to eliminate it. If the gate doesn't fire (e.g. because the protection logic data is absent), a temporally contradicted candidate remains in the ranked list with an artificially inflated temporal score. This is a meaningful safety gap.

### The three hard gates

Gate 1 (physical plausibility) depends on "component state" which is largely inferred from the KG, not from real-time data. The KG is populated by engineers and may not reflect the actual state of the component at the time of the event. If the KG says a valve is normally open but it was locked out for maintenance, Gate 1 may incorrectly pass or fail candidates based on stale topology.
Gate 3 (barrier logic) explicitly degrades when protection logic data is absent. The document states that "if SOE is present but protection logic is absent, the timeline and barrier gates operate in degraded mode." At a nuclear facility, the barrier gate is arguably the most safety-critical filter in the pipeline. Running in degraded mode without a hard analyst hold on the result is a design choice that deserves more explicit treatment. The current behavior is to set an analyst_decisions_required flag, but this is a soft mechanism.
There is no gate that checks whether the ranked primary hypothesis is physically sufficient to explain the observed symptoms. The gates are all eliminative — they remove impossible candidates. No gate confirms that the surviving top candidate is a complete explanation. A partial cause could rank first because it has good documentary support, while the true initiating failure mode has no prior CRs and scores low on evidence.

### Category coverage enforcement

The requirement that every run must produce at least one candidate in each of Categories A–L (or explicitly rule out the category) is correct and important. However, the mechanism is passive: "a category with no candidates is always recorded in category_coverage with reason_code = no_supporting_data." This means the pipeline can produce a formally complete RCA with twelve entries saying "no supporting data" for ten categories. The analyst attention flag fires only when status is "unknown" — not when it's "no_supporting_data" for a high-impact category. An L-category (systemic/organizational) root cause could be silently underweighted if the KG has no training_records and the Chroma store has no confirmed RCA documents with L-category findings.

## 3. Key Assumptions — Where They May Not Hold
Assumption: Telemetry anomaly detection upstream is reliable. The document explicitly states "anomaly detection quality upstream directly limits what the pipeline can conclude." This is correct but understated as a risk. At many nuclear facilities, the plant historian is configured for compliance monitoring, not anomaly characterization. Gradual drift over weeks or months — the signature of classic degradation failures — may not appear as a clean gradual_drift anomaly type in a pre-processed summary. The pipeline has no way to distinguish "no anomaly detected" from "anomaly present but not characterized upstream." This matters most for Categories A (equipment-internal degradation) and J (inspection program gap).
Assumption: The KG failure mode taxonomy is complete and current. Every hypothesis is anchored to a KG node. If the FMEA-derived failure mode library doesn't include a failure mechanism that occurs in practice — either because it's novel, because the FMEA was done at a lower fidelity level, or because the equipment was modified without a FMEA update — the pipeline cannot generate that hypothesis regardless of how strong the evidence is. This is a known limitation, but the pipeline provides no signal when it might be in this condition. A run with a sparse or stale failure mode library produces the same format output as a run with a complete one, making it hard for the analyst to recognize when the hypothesis space is fundamentally incomplete.
Assumption: The documentary record (Chroma) is representative and current. The evidence sub-score depends heavily on what's in the vector store. At plants with poor CR closure discipline — where CRs remain open for months, or findings are written in vague language — the evidence sub-score for genuine causes will be systematically low. The pipeline's scoring will favor failure modes that have been written up clearly in prior CRs, which biases toward previously-identified causes and against genuinely novel ones. This is the inverse of what you want when the novelty flag fires.
Assumption: The recurrence count is meaningful as a proxy for root cause persistence. The history score adds +0.10 if any events are unresolved and +0.15 if the trend is increasing. But "unresolved" in the CR/WO sense means the corrective action hasn't been closed — it doesn't mean the root cause wasn't addressed. Many nuclear plants carry large backlogs of technically open CRs that are operationally managed. High unresolved recurrence count may reflect backlog management culture rather than genuine root cause persistence.
Assumption: The causal chain position (initiating / contributing / consequence) can be reliably assigned from temporal data alone. In practice, many nuclear events involve nearly simultaneous multi-train responses where the temporal ordering of signals is within the Allen epsilon (0.5 hours). In those cases, chain position defaults to "contributing," which is a conservative but informationally empty assignment. For fast transients — reactor trips, ECCS actuations — the entire causal sequence may unfold in seconds, making a 30-minute epsilon meaningless.

## 4. Human and Organizational Factors — A Structural Gap
Categories G, I, and L are architecturally second-class citizens in this pipeline. Their primary evidence sources are:

Chroma-indexed work orders and procedures (text retrieval)
Training records (structured data)
OE documents (text retrieval)

But the pipeline's scoring engine is fundamentally oriented toward equipment failure modes. The structural sub-score (weight 0.30) measures KG topology fit, which is meaningful for Categories A–F but largely undefined for Category L (systemic/organizational weakness). A Category L hypothesis has no "component" in the topology sense, no telemetry signature, and no FMEA failure mode node. Its composite score will be dominated by whatever documentary evidence exists and its governance sub-score — making it structurally disadvantaged in the ranking against equipment-origin hypotheses even when organizational weakness is the genuine root cause.
The Ishikawa matrix (Step 6a) is described as "optional" and is the only mechanism that explicitly evaluates the personnel, procedure, and management branches. Making the organizational root cause analysis optional in a pipeline intended for nuclear RCA is a significant design choice that the document doesn't address as a limitation.

## 5. Smaller but Consequential Issues
The novel_pattern flag requires three simultaneous conditions, including "no signal IDs resolved to this failure mode." A failure mode that has a signal match but zero documentary recurrence — arguably the most dangerous scenario (new signal, no history) — is not flagged as novel. The SE reading the manifest might assume "not novel" means "we've seen this before," when it may mean "we've seen a signal match but have no documented prior event."
The similar event confidence multipliers (fleet 0.80, industry 0.60) are hardcoded and not documented as to their origin. These are not trivial values — they directly affect whether a known industry failure pattern will be ranked above a plant-specific candidate. There is no sensitivity analysis provided and no mechanism for a site to calibrate them based on their OE program quality.
The LLM hallucination guard (§B.1, §5.12) discards the entire LLM output if the primary hypothesis candidate_id is invalid. This is correct but incomplete: the LLM could produce a valid candidate_id for the primary hypothesis while hallucinating contributing causes, evidence citations, or recommended actions. The guard only checks the primary hypothesis ID, leaving the rest of the card unchecked for factual accuracy against the evidence bundle.
The writeback_ready flag is a composite of many conditions, but the document doesn't specify what happens when a run is marked writeback_ready and the analyst disagrees with the top hypothesis. The flag signals process readiness, not factual correctness. An analyst who disagrees with the ranking has no pipeline mechanism to record a dissent and proceed — the only option appears to be holding the record, which may conflict with regulatory timeline requirements.

## Summary of Priority Issues

| Priority | Issue | 
| :--- | :--- |
| High | Allen map built after Step 4 — temporal picture incomplete during hypothesis generation |
| High | No confirmatory gate — surviving top candidate may be partial or insufficient explanation |
| High | Barrier gate degrades silently when protection logic absent; soft hold mechanism only |
| High | Organizational categories (G/I/L) structurally disadvantaged in composite scoring |
| Medium | Static scoring weights across failure mode superclasses
| Medium | novel_pattern condition 3 allows signal-matched first-occurrences to escape the flag |
| Medium | Recurrence count conflates genuine persistence with CR backlog culture |
| Medium | Scope expansion suggestion circularity — early scope shapes what expansions are suggested |
| Medium | LLM hallucination guard checks only primary hypothesis ID |
| Lower | Fleet/industry confidence multipliers are undocumented and uncalibrated |
| Lower | writeback_ready has no analyst dissent pathway |

Overall this is a well-structured decision-support architecture. The most important design principle — that the analyst retains final authority — is correctly stated and structurally enforced in most places. The gaps above are areas where that principle could be undermined in practice by subtle scoring artifacts, silent degradation, or structural biases in the causal taxonomy treatment.

# Addressing Issues

## 1. Allen Map Built After Step 4 — Temporal Picture Incomplete During Hypothesis Generation
Root problem: Step 4 uses a compressed Allen proxy embedded in tskr_patterns rather than the full relation map, meaning candidate ranking happens on incomplete temporal information.
Suggestions:
The most direct fix is to move Allen map construction to execute between Steps 1 and 4, making it a true prerequisite to candidate generation rather than a conceptual one. The reason it currently runs after Chroma retrieval appears to be a dependency on evidence refinement inputs — but the Allen classification itself only needs event, telemetry_summary, alarm_log, and soe_log, none of which require Chroma. The map can be built in two passes: a lightweight first pass before Step 4 that classifies all available signal intervals for use in initial scoring, and a refinement pass after Chroma retrieval that adds any evidence-derived temporal signals. This preserves the current refinement logic while giving Step 4 a real temporal foundation.
If moving the map is architecturally expensive, a lower-cost alternative is to make the TSKR proxy more explicit about its limitations. Currently tskr_patterns carries Allen-classified timing but doesn't surface which candidates have a proxy score versus a full Allen score. Adding a temporal_score_quality field ("proxy" vs "full_allen") would allow Step 4 to flag candidates whose ranking depends heavily on proxy temporal data, giving analysts a targeted signal about where the ranking is most uncertain.

## 2. No Confirmatory Gate — Surviving Top Candidate May Be Partial or Insufficient Explanation
Root problem: All three hard gates are eliminative. No mechanism confirms that the top-ranked candidate is causally sufficient to explain the full observed symptom set.
Suggestions:
Introduce a symptom coverage check as a post-ranking assessment (not a hard gate, to avoid false eliminations). For each candidate that passes the three hard gates, compute a symptom_coverage_ratio: the fraction of observed telemetry anomalies and alarm activations that are consistent with the candidate's expected failure mode signature. A candidate that explains 40% of observed symptoms while ranking first should be flagged differently than one that explains 90%.
This ratio doesn't need to be a blocking gate — it can be an attention flag with a configurable threshold (e.g., flag when symptom_coverage_ratio < 0.60 for the top candidate). The flag text should explicitly state: "The primary hypothesis accounts for N of M observed anomalies. The following anomalies are unexplained by this hypothesis: [list]." This directly addresses the partial-cause problem and gives the analyst specific evidence to investigate.
A complementary approach is residual anomaly tracking: after the top candidate is selected, tag each signal in allen_relation_map as either "explained" (consistent with primary hypothesis failure mode signature) or "residual" (not explained). Residual signals that have causal timing (OVERLAPS or CONTAINS) are surfaced in rca_card.unresolved_gaps with explicit language that they may indicate a co-existing or alternative root cause. This is lower implementation cost than a full coverage gate and directly useful to the analyst.

## 3. Barrier Gate Degrades Silently When Protection Logic Absent
Root problem: The barrier logic gate — arguably the most safety-critical filter — operates in degraded mode when protection_logic_context is absent, with only a soft analyst_decisions_required flag as the hold mechanism.
Suggestions:
Introduce a barrier gate degradation severity level distinct from other data coverage gaps. Unlike missing vendor records or training data, absent protection logic directly compromises a hard safety gate. The manifest should report this not just as a data coverage gap but as a gate integrity flag — a separate field that the writeback readiness check explicitly evaluates.
Specifically, add a condition to writeback_ready: if the barrier gate operated in degraded mode and analyst_review.barrier_gate_degraded_acknowledged is not explicitly set to true in the card JSON, writeback_ready must be false regardless of other conditions. This makes the barrier degradation an active analyst decision rather than a passive flag. The analyst must affirmatively state they have reviewed the barrier status through another means (physical walkdown, PLC historian query, operator statement) before the record can proceed.
A second suggestion is to define a protection logic surrogate hierarchy. When protection_logic_context is absent, the pipeline currently uses whatever SOE data is available in degraded mode. Make this hierarchy explicit and documented: SOE actuation records are a tier-1 surrogate; alarm log actuation alarms are tier-2; operator shift log entries are tier-3; no surrogate means full degradation. Each tier carries a defined barrier_gate_confidence level that is reported in the manifest and contributes to the attention flag severity. This gives the analyst a clearer picture of how degraded the gate actually is, rather than a binary present/absent flag.

## 4. Organizational Categories (G/I/L) Structurally Disadvantaged in Composite Scoring
Root problem: The composite scoring formula is designed around equipment failure modes with KG topology, telemetry signatures, and FMEA nodes. Categories G, I, and L have none of these, making them structurally low-scoring regardless of evidence strength.
Suggestions:
The most important fix is category-specific scoring profiles. Rather than one static weight vector for all candidates, define at minimum three profiles:

Equipment-origin (Categories A–F): current weights — structural 0.30, temporal 0.20, telemetry 0.20, evidence 0.20, governance 0.10
Human/procedural (Categories G, I): structural weight reduced (topology fit is not meaningful here), evidence weight raised, with the structural dimension replaced by a procedural compliance dimension that scores deviation from procedure baseline using configuration_change_records and work order text
Organizational/systemic (Categories L): structural and telemetry weights collapsed to near-zero; evidence and a new recurrence pattern dimension dominate; the recurrence dimension should specifically credit unresolved_recurrence_count and increasing trend across multiple event types, not just same-FM recurrence

Additionally, make the Ishikawa matrix non-optional for events above a defined severity threshold. The current design makes it optional as a config flag, which means organizational factor analysis can be bypassed entirely. For any event where the RCA card safety_significance field is above a site-defined floor, enable_ishikawa should be forced to true and the writeback check should fail if ishikawa_matrix is absent. The document already has the severity significance postprocessing step — this is a natural extension of it.
A third suggestion is to add a Category L floor check: if the pipeline produces a complete run and no Category L candidate has a composite score above a minimum threshold (e.g., 0.20), and the event has any recurrence (recurrence_count > 0) or any unresolved prior events, an explicit attention flag should fire stating that organizational root cause was not established despite a recurrence pattern. This forces the analyst to actively evaluate and document why L does not apply, rather than letting it silently score low.

## 5. Static Scoring Weights Across Failure Mode Superclasses
Root problem: The same dimensional weights apply to failure modes whose causal evidence looks completely different. This systematically biases ranking toward failure modes that happen to align with the weight vector's implicit assumptions.
Suggestions:
Implement weight profiles tied to failure mode superclass (already partially done for governance; generalize it). The KG already carries a superclass field on each failure mode node — this is the natural key for weight selection. Define profiles for at minimum: mechanical degradation, electrical/I&C, design deficiency, procedural deviation, and environmental/external. Each profile specifies its own weight vector and any applicable sub-score modifiers.
The weight selection logic in the causality engine then becomes:
profile = scoring_profiles[failure_mode.superclass]
composite = sum(profile.weights[dim] × scores[dim] for dim in dimensions)
This is a low-risk change structurally because the output format doesn't change — only the inputs to the weighted sum. For auditability, the score_rationale field on each candidate should record which weight profile was applied, so analysts and reviewers can see that a design deficiency candidate used a different structural weight than a bearing failure candidate.
A complementary improvement is to add weight sensitivity output to the run manifest. For the top three candidates, report how the ranking would change if the structural weight shifted ±0.10 and the evidence weight shifted ±0.10. This is a simple parametric calculation that can run at manifest finalization with no additional data, and it gives analysts a direct read on whether the ranking is robust to reasonable disagreements about dimensional importance.

## 6. novel_pattern Condition 3 Allows Signal-Matched First-Occurrences to Escape the Flag
Root problem: A failure mode with a signal match but zero documentary history is not flagged as novel, which could mislead an analyst into thinking there is prior experience with the pattern when there is only a signal correlation.
Suggestions:
Decompose the single novel_pattern flag into two distinct flags that answer different questions:

documentary_novel: True when recurrence_count == 0 and history_score < 0.20 — no prior documentary record regardless of signal matching
signal_novel: True when no signal IDs resolve to this failure mode — no prior signal pattern match

The current novel_pattern flag is essentially the AND of both conditions. Splitting them into separate fields allows the analyst to distinguish four meaningful states:
documentary_novelsignal_novelMeaningFalseFalseKnown pattern, documented history — lowest concernFalseTrueDocumented in CRs/WOs but no signal match — possible documentation quality issueTrueFalseNew to the documentary record but signal-matched — this is the dangerous case currently not flaggedTrueTrueGenuinely novel — current behavior correctly flags this
The "documentary novel but signal-matched" case is particularly important: it means the equipment is showing a signal pattern consistent with a known failure mode type, but the plant has no CR/RCA record of it. This could mean the failure is genuinely new, or it could mean prior occurrences were not properly documented. Either finding warrants an explicit attention flag with different recommended analyst actions than a fully novel pattern.

## 7. Recurrence Count Conflates Genuine Persistence With CR Backlog Culture
Root problem: High unresolved_recurrence_count may reflect a plant's CR backlog management practice rather than an unresolved physical root cause, leading to inflated history scores for some failure modes.
Suggestions:
Add resolution quality weighting to the recurrence profile. Rather than treating all unresolved CRs equally, weight them by how long they have been open and whether a corrective action has been assigned (even if not closed). A CR open for 8 years with no corrective action assigned is a genuine root cause persistence signal. A CR open for 3 months with an active work order is likely administrative backlog.
The history score bonus for unresolved events should be conditioned on resolution quality:
unresolved_weight = sum(
    1.0 if no_corrective_action_assigned
    else 0.4 if corrective_action_open_days > 365
    else 0.1
    for each unresolved event
) / max(1, len(unresolved_events))

history_score bonus = 0.10 × unresolved_weight  # currently a flat +0.10
This requires the CMMS records to carry corrective_action_status and ca_open_days fields, which are standard in most plant CMMS systems (Maximo, SAP PM). The cmms_context schema should be extended to capture these fields explicitly.
A second suggestion is to add a CR closure rate metric at the run level, not just the candidate level. If the overall plant CR closure rate for this equipment class is low (which can be computed from kg_context.past_events aggregate statistics), the manifest should note that recurrence counts may be systematically overstated for all candidates in this run. This is a run-level data quality flag, not a per-candidate adjustment, and it gives reviewers important context for interpreting all recurrence-derived scores.

## 8. Scope Expansion Suggestion Circularity
Root problem: The expansion suggestions generated in Step 6h are shaped by the scope of the run that generated them. An incomplete first run may suggest expansions that are artifacts of missing data rather than genuine causal leads.
Suggestions:
Implement expansion suggestion confidence scoring based on the quality of the run that generated them. An expansion suggestion generated from a run with degraded TSKR, missing SOE, or a KG governance warning should carry a lower confidence than one generated from a complete run. The expansion_suggestions entries should include a suggestion_confidence field that reports the data quality of the originating run, so the analyst can weight the suggestion accordingly.
Add a counter-factual check before surfacing expansion suggestions: for each suggested component, query whether it has any failure modes in the KG that are consistent with the current event's telemetry anomaly types. If the suggested component has no plausible failure modes that match the observed signal patterns, the suggestion should be suppressed or at minimum labeled "topology-adjacent but no matching failure mode" rather than presented as a neutral expansion option. This filters out suggestions that are structural neighbors but not causal candidates.
On the process side, consider requiring a minimum of two runs before an expansion is recommended as ready to accept. The first run generates suggestions; the second run (with any available data backfill) confirms or withdraws them. This breaks the circularity by requiring that an expansion survive a second evidence pass before the analyst is asked to make a decision. The analyst_decisions_required field in the manifest can be conditioned on run_count >= 2 for expansion suggestions.

## 9. LLM Hallucination Guard Checks Only Primary Hypothesis ID
Root problem: The guard discards the card if the primary hypothesis candidate_id is invalid, but contributing causes, evidence citations, and recommended actions are not verified against the actual evidence bundle.
Suggestions:
Extend the hallucination guard into a structured post-generation validation pass that checks all ID-bearing and citation-bearing fields against the actual pipeline outputs before the card is accepted. The validation should cover at minimum:

All candidate_id references in alternatives[] and contributing_causes[] must exist in causality_candidates
All evidence_id references in citations[] must exist in evidence_bundle
All target_component_id values in recommended_actions[] must exist in kg_context.components
No recommended_actions[] entry should cite a barrier or safety function that the barrier_analysis reports as intact without explicit qualification

Any field that fails validation should be either removed from the card (if non-essential) or cause the card to fall back to the deterministic template for that specific section rather than discarding the entire card. A partial fallback approach — where the LLM primary hypothesis is accepted but contributing causes fall back to deterministic template — preserves LLM value where it's valid while preventing hallucinated content from reaching the analyst.
Additionally, add a semantic consistency check between the LLM narrative text and the structured fields. If the primary hypothesis narrative text mentions a component name or failure mode that doesn't appear in the top candidate's fm_id or component_id, flag it as a narrative inconsistency for analyst review. This can be implemented with simple named entity extraction against the KG vocabulary — it doesn't require a second LLM call.

## 10. Fleet/Industry Confidence Multipliers Are Undocumented and Uncalibrated
Root problem: The hardcoded multipliers (fleet 0.80, industry 0.60) apply uniformly regardless of the quality or relevance of the specific OE source, and their origin is not documented.
Suggestions:
Replace the hardcoded tier multipliers with source-level confidence metadata returned by the adapter. The SimilarEventAdapter protocol should be extended to return a source_confidence field per result alongside the match score. The orchestrator then applies this per-result confidence rather than a flat tier discount. For LLMOEAdapter, source confidence can be derived from the LLM's own relevance score, the recency of the source document, and whether the event type is an exact match.
For sites that want to retain simple tier multipliers, move the multipliers from hardcoded constants into OrchestratorConfig so they are site-configurable and appear in the run_manifest.pipeline_config snapshot. This makes them auditable — a reviewer can see what discount was applied and whether it matches the site's OE program quality assessment.
Document a calibration methodology in Appendix C: sites should compare their similar_event_list rankings against known analogues from INPO SOERs or NRC LERs over a set of historical events, and adjust the industry multiplier based on how often industry-tier matches correctly identified the root cause before it was confirmed locally. This is not a training procedure — it's a configuration review process that any site OE coordinator can perform annually.

## 11. writeback_ready Has No Analyst Dissent Pathway
Root problem: An analyst who disagrees with the top hypothesis has no pipeline mechanism to record a dissent and proceed — the only option is to hold the record, which may conflict with regulatory timelines.
Suggestions:
Add a formal analyst override mechanism to the analyst_review block of the RCA card. The override allows the analyst to:

Accept a hypothesis that is not the pipeline's top-ranked candidate as the primary conclusion
Record a structured rationale for the override
Proceed to writeback with the override documented

The override should be captured in a structured field, not free text, so it can be reported systematically:
json"analyst_override": {
    "active": true,
    "analyst_selected_candidate_id": "CAND-007",
    "pipeline_top_candidate_id": "CAND-002",
    "override_rationale_category": "physical_inspection_finding" 
        | "additional_evidence_not_in_corpus" 
        | "expert_judgment" 
        | "other",
    "override_rationale_text": "...",
    "analyst_id": "...",
    "override_timestamp": "..."
}
When analyst_override.active is true, writeback_ready can be set to true provided the override fields are complete, even if the pipeline's own scoring conditions are not fully met. The override is recorded in the decision_trail in the manifest, making the divergence between pipeline ranking and analyst conclusion fully auditable.
This mechanism also generates valuable feedback data: a systematic pattern of overrides for a particular failure mode superclass or causal category is a direct signal that the scoring weights or KG coverage need recalibration for that equipment type. Tracking override frequency by category over time is the most honest calibration signal available.

## Cross-Cutting Suggestion: Uncertainty Propagation
One issue that underlies several of the problems above is that the pipeline produces point-estimate scores without representing uncertainty. A candidate with a composite score of 0.62 built from complete data is fundamentally different from a score of 0.62 built from missing SOE, a proxy Allen score, and no Chroma evidence — but both look identical in the ranked list.
Consider adding a score confidence interval to each candidate's output, derived from data completeness. The width of the interval would be a function of how many sub-scores were computed from full data versus degraded proxies. A candidate with narrow intervals and a score of 0.62 is a strong result. A candidate with wide intervals and a score of 0.62 is essentially a guess. Displaying this visually in the RCA card — even as a simple "confidence band" label alongside the score — would help analysts calibrate their trust in the ranking without requiring them to read the full manifest sensitivity table.

---

## Implementation Plan (2026-05-06)

### Grouping by architectural layer

| Layer | Issues | Risk |
|---|---|---|
| Attention flags / workflow control (additive, no scoring changes) | 3, 5, 11, 9 | Low |
| Output enrichment (new fields on existing artifacts) | 2, 7, 12, 14 | Low |
| Synthesizer (isolated from scoring path) | 10, 13 | Low–Medium |
| Scoring architecture (causality engine, ranking) | 1, 4, 6 | Medium–High |
| CMMS schema dependent | 8 | External dependency |

---

### Phase 1 — Safety-critical hardening (no scoring changes)

All three are new attention flags or writeback conditions. They touch `_apply_*_attention_flags()` and `_build_workflow_dispatch()` only — no effect on ranking.

| Issue | What changes | Why now |
|---|---|---|
| **3** — Barrier gate degradation | Add `barrier_gate_degraded_acknowledged` to `writeback_ready` conditions; add surrogate hierarchy (SOE tier-1 / alarm tier-2 / shift log tier-3) with `barrier_gate_confidence` in manifest | Most safety-critical gap; purely additive to writeback logic |
| **5** — Fast transient Allen epsilon | Detect fast-transient events from `event_type` (reactor trip, ECCS actuation); flag when all causal signals collapse within epsilon window; emit `allen_epsilon_unreliable` attention flag | Concrete failure condition for a specific, common nuclear event type; attention flag only, no scoring change |
| **11** — Category L floor check | New `_apply_category_l_floor_attention_flags()`: if no L-candidate above threshold AND `recurrence_count > 0` or unresolved prior events exist, fire explicit flag | L silently low-scores with no alert; additive method, zero scoring impact |

**Files touched:** `rca_reasoning_orchestrator.py` (three methods), `run_manifest` schema (one new field).

---

### Phase 2 — Output enrichment (additive fields, no scoring changes)

New fields on existing artifacts. No ranking changes. Each is independent.

| Issue | What changes | Key decision needed |
|---|---|---|
| **7** — `novel_pattern` decomposition | Split into `documentary_novel` (no CR/WO history) + `signal_novel` (no signal match); retain `novel_pattern` for backward compat | Whether to deprecate `novel_pattern` immediately or keep both |
| **2** — Symptom coverage / confirmatory check | Post-ranking: compute `symptom_coverage_ratio` per candidate from `expected_anomaly_pattern` on FM nodes vs. observed signals; tag residual signals in `rca_card.unresolved_gaps` | Requires `expected_anomaly_pattern` populated in KG — check coverage before implementing |
| **9** — Scope expansion confidence | Add `suggestion_confidence` to expansion suggestions derived from originating run's data quality flags; add counter-factual FM match check (suppress if no FM matches observed anomaly types) | Minor |
| **12** — Confidence multipliers to config | Move `TIER_CONFIDENCE_MULTIPLIERS` from `similar_event_adapter.py` to `OrchestratorConfig`; appear in manifest `pipeline_config` snapshot | Trivial |
| **14** — Uncertainty propagation | Add `score_confidence_interval` per candidate: width derived from count of sub-scores on full data vs. degraded proxies; manifest-level output first | How to define "degraded proxy" precisely — needs agreed definition |
| **8** — Recurrence quality weighting (moved from Phase 5) | `unresolved_weight` formula using `cr_records[].status` + derived `ca_open_days` from existing CMMS fields; `cmms_recurrence_quality: "weighted"\|"flat"` in manifest | No CMMS adapter changes required; uses existing `cmms_context.json` `status` and `closed_date` fields |

**Files touched:** `rca_reasoning_orchestrator.py`, `causality_engine_v32.py` (output only), `tskr_patterns` schema, `similar_event_adapter.py`, `OrchestratorConfig`, history scoring logic in `causality_engine_v32.py`.

---

### Phase 3 — Synthesizer hardening (isolated from scoring)

Both touch `rca_synthesizer_v31.py` and the card schema. Independent of scoring path.

| Issue | What changes |
|---|---|
| **10** — Extended hallucination guard | Validate all ID-bearing fields: `candidate_id` in `alternatives[]`/`contributing_causes[]`, `evidence_id` in `citations[]`, `component_id` in `recommended_actions[]`; partial section fallback rather than full discard |
| **13** — Analyst dissent pathway | New `analyst_override` block in `analyst_review`: `active`, `analyst_selected_candidate_id`, `override_rationale_category` (4-value enum), `override_rationale_text`, `analyst_id`, `override_timestamp`; `writeback_ready` accepts override if block is complete |

**Files touched:** `rca_synthesizer_v31.py`, RCA card JSON schema.

---

### Phase 4 — Scoring architecture (highest risk, needs careful test design)

Two inter-dependent changes. Must be done together and require regression testing across all TC-1–TC-7 cases.

| Issue | What changes | Options |
|---|---|---|
| **1** — Allen map pre-Step-4 | **Option A (full fix):** Move Allen map build to between Steps 1 and 4; `generate()` consumes it directly. **Option B (interim):** Add `temporal_score_quality: "proxy" \| "full_allen"` field to `tskr_patterns`; no execution order change. | A fixes the root problem but is higher risk; B gives analyst visibility immediately with zero risk; not mutually exclusive |
| **4 + 6** — Category-specific scoring profiles | Define at minimum three weight profiles keyed by FM superclass: equipment-origin (A–F, current weights), human/procedural (G/I, structural ↓ evidence ↑), organizational (L, structural+telemetry → near-zero, evidence + recurrence dominate); `score_rationale` records which profile applied | Start with G/I/L profiles only, or generalize to all superclasses immediately? |

**Files touched:** `causality_engine_v32.py` (core scoring loop), `CausalityEngineConfigV32`, `rca_reasoning_orchestrator.py` (Allen map call site).

---

### Phase 5 — (Reserved)

Issue 8 moved to Phase 2 (Decision D3: graceful degradation with existing CMMS schema fields; no external dependency). Phase 5 is available for future items.

---

### Open decisions before coding

| # | Decision | Options | Implication | **Outcome (2026-05-06)** |
|---|---|---|---|---|
| **D1** | Issue 1 — Allen map: fix root cause or add quality field first? | Option A: move map before Step 4 (Phase 4, medium risk); Option B: add `temporal_score_quality` field (Phase 2, zero risk); both | B can ship immediately; A requires test design for scoring regression | **Option B now.** Add `temporal_score_quality: "proxy" \| "full_allen"` in `_apply_allen_temporal_blend()` per candidate. Option A only if TC-8 demonstrates proxy-vs-full divergence on a real event. |
| **D2** | Issue 4+6 — Scoring profiles: G/I/L only or all superclasses? | Narrow (G/I/L first, extend later); Broad (full superclass generalization now) | Narrow reduces regression surface but requires a second scoring change later | **Narrow weights now; broad dispatch infrastructure from day one.** Implement G/I/L profiles immediately. Extend `_governance_weight_for_fm()` into `_scoring_profile_for_fm()` returning a full weight dict keyed by superclass — so adding A–F profiles later is a data change only, no code change. |
| **D3** | Issue 8 — CMMS schema: implement now or defer? | Implement if fields exist in current Maximo/SAP exports; defer to Phase 5 if not | Verify `corrective_action_status` and `ca_open_days` field availability before scheduling | **Implement now with graceful degradation; move to Phase 2.** Existing `cmms_context.json` already has `status` (open/closed) and `closed_date` on CR records — sufficient to derive `ca_open_days`. Fields optional: if absent, fall back to current flat +0.10 bonus; `run_manifest.data_coverage_summary` records `cmms_recurrence_quality: "weighted"` or `"flat"`. No CMMS adapter schema changes required. |

### Decision Outcomes — Implementation Implications

**D1:** `temporal_score_quality` is set in `causality_engine_v32._apply_allen_temporal_blend()`. Logic: if `_build_allen_component_index()` returned a non-empty entry for the candidate's `component_id`, quality = `"full_allen"`; otherwise quality = `"proxy"` even if the Allen map exists. Per-component, not per-run.

**D2:** Extend `_governance_weight_for_fm()` into a `_scoring_profile_for_fm(superclass)` method returning `{"structural": ..., "temporal": ..., "telemetry": ..., "evidence": ..., "governance": ...}`. Pass as `weights_override` to `_combine_scores()`. `score_rationale` records which profile applied. Prerequisite: verify KG FM nodes for G/I/L categories carry `superclass` values that match the dispatch key — keyword inference alone is not reliable enough (see SE review below).

**D3:** Implement `unresolved_weight` formula in the history scorer using existing CMMS fields. `corrective_action_status` maps to `cr_records[].status`; `ca_open_days` = `(event_date − cr.created_date).days` when `status == "open"`, or `(cr.closed_date − cr.created_date).days` when closed. Record `cmms_recurrence_quality` in manifest. Issue 8 moves from Phase 5 to Phase 2.

---

## Code-Verified Review Notes (2026-05-06)

*Added after code-to-text verification of the implementation against the claims in this review.*

### Technical Accuracy

All major factual claims are confirmed against the implementation:

- Allen map builds at Step 5b — after Step 5a Chroma retrieval, before Step 5c refinement; Step 4 runs on TSKR-embedded Allen proxy only ✓
- Raise-only rule for Allen temporal blend: `new_temporal = max(old_temporal, new_temporal)` — confirmed in code ✓
- Barrier gate degrades when `protection_logic_context` absent; hold mechanism is `analyst_decisions_required` (soft) ✓
- `novel_pattern` requires three simultaneous conditions ✓
- Fleet/industry multipliers hardcoded as `{"plant": 1.00, "fleet": 0.80, "industry": 0.60}` in `adapters/similar_event_adapter.py` ✓
- LLM hallucination guard checks only `primary_hypothesis.candidate_id`; contributing causes, citations, and recommended actions are not validated ✓
- `minimum_pre_evidence_threshold = 0.10` at Step 4 vs. `minimum_evidence_threshold = 0.35` at Step 5 — threshold discontinuity confirmed ✓
- Category L structural sub-score (weight 0.30) measures KG topology fit — meaningless for a systemic/organizational cause with no topology node ✓

**One precision issue — Allen map timing (§1 Structural Concerns):**

The review states "The Allen map executes after Chroma retrieval." This is accurate per execution order (5a Chroma → 5b Allen map → 5c refinement), but it slightly misframes the root problem. The critical gap is not the Allen map's position relative to Chroma retrieval — it is its position relative to *Step 4 candidate generation*, which runs before both. The Allen map is unavailable to Step 4 whether it is built at 5b or at 5a; moving it to just before Chroma retrieval would not fix anything. The two-pass suggestion in §1 (lightweight pre-Step-4 pass + refinement pass at 5b) correctly identifies the fix, but the framing of the problem should be "Allen map absent during candidate generation" rather than "Allen map after Chroma retrieval."

### Arguments That Could Be Sharpened

**Scope expansion circularity (§1):** The concern is real but the failure mode is more specific than stated. The circularity matters most when the *first run* operated with a degraded data source (e.g., no SOE log). In that case, the Allen proxy in `tskr_patterns` is computed from telemetry and alarms only, and the expansion signals in Step 6h are drawn from that incomplete picture. An expansion suggestion generated from a complete run carries much lower circularity risk. The suggestion in §8 (expansion suggestion confidence scoring derived from the originating run's data quality) is the right fix, but the problem statement would be stronger if it specified the degraded-first-run scenario rather than the general case.

**Auto-reentry candidate explosion (§1):** The `top_k_candidates` cap does constrain the output size. The sharper concern is that the KG governance check does not re-run after neighborhood expansion — if the expanded hop radius includes components with stale or low-fidelity failure mode entries, those candidates enter the ranking without a governance flag. This is more specific and more actionable than "candidate explosion."

### Gaps — Capabilities Not Assessed

Three implemented capabilities that partially address concerns raised in this review are not mentioned:

**1. Signal episode search and cross-pattern linkage (optional extensions)**

`PatternSearcher` (gated by `enable_signal_episode_search`) and `CrossPatternLinker` (gated by `enable_cross_pattern_linkage`) are implemented and address two concerns raised in §2 and §3:

- The "recurrence count conflates backlog with persistence" concern (§7): cross-pattern linkage joins signal episodes with extracted causal chains from historical documents, providing a richer recurrence signal than the CMMS-derived unresolved count.
- The "Chroma biases toward documented failure modes" concern (§3): `DocExtractionStore` is a second, separate Chroma store holding extracted causal chain records. It provides an evidence path that is structurally independent of the main evidence retrieval query.

Both extensions are disabled by default. The review is correct to evaluate the default execution path, but a note that the architecture has a development path toward these concerns would be balanced.

**2. DocExtractionStore as a second evidence path**

The review focuses on the single main Chroma store for documentary evidence. The `DocExtractionStore` (a second Chroma collection holding `HistoricalDocExtraction` objects — extracted causal chains from past RCA/ECA documents) provides a complementary retrieval path via `CrossPatternLinker`. Its epistemic classification system (`affects_performance`, `monitors_performance`, `analyzes_past_degradation`, `characterizes_system`) and `semantic_contribution = similarity_score × confidence_weight × cause_is_symptom_factor` formula partially mitigate the "evidence sub-score biased toward well-documented failure modes" concern for plants that have historical RCA documents ingested into this store.

**3. Writeback dissent vs. existing hold mechanism**

The review correctly identifies the absence of a formal analyst override pathway. The current state is not quite binary: `decision_status = "review_required"` with `analyst_decisions_required` does capture a hold, and `coverage_acknowledgement_required` forces the analyst to explicitly acknowledge degraded coverage before writeback. The gap is specifically the absence of a structured pathway to *proceed with a different conclusion than the pipeline's top-ranked candidate* — the analyst can hold, but cannot dissent-and-proceed. The `analyst_override` JSON structure in §11 is the right fix; the problem statement is accurate but the current state has more nuance than "the only option is to hold the record."

### Priority Calibration

Two additions to the priority table are recommended:

| Priority | Issue | Basis |
|---|---|---|
| **High** | Fast transients — Allen epsilon (0.5 h) is meaningless for reactor trips and ECCS actuations where causal sequence unfolds in seconds | Buried in §3 Key Assumptions; severity warrants High |
| **Medium** | Category L silent underweighting — `no_supporting_data` never triggers attention flag; an RCA with zero L-category candidates above threshold produces no alert | Described in §2 coverage enforcement narrative; not in priority table |

The existing High/Medium calibration for the other items is appropriate.

### Suggestion Feasibility

| Suggestion | Feasibility | Implementation note |
|---|---|---|
| Two-pass Allen map (lightweight pre-Step-4 pass) | High | Allen classifier only needs `event`, `telemetry_summary`, `alarm_log`, `soe_log` — no Chroma dependency; clean insertion between Steps 1 and 4 |
| `temporal_score_quality` field ("proxy" vs "full_allen") | High | Low-risk alternative; one field added to existing `tskr_patterns` output per candidate |
| Symptom coverage ratio / residual anomaly tracking | Medium | Requires a FM → expected signal type map in the KG; not currently stored on `failure_mode` nodes beyond `expected_anomaly_pattern` |
| Barrier degradation severity + `barrier_gate_degraded_acknowledged` in writeback | High | Straightforward extension to writeback_ready condition list; `barrier_analysis` already carries gate integrity info |
| Category-specific scoring profiles | High | KG already carries `superclass` on failure mode nodes; `score_rationale` already records per-candidate inputs; structural change is minimal |
| `documentary_novel` + `signal_novel` flag decomposition | High | Clean split with no schema breaking change; most actionable suggestion in §5–11 |
| Resolution quality weighting for recurrence | Medium | Requires `corrective_action_status` and `ca_open_days` in CMMS schema — availability varies by site CMMS discipline |
| Expansion suggestion confidence scoring | High | Natural fit; `suggestion_confidence` can be derived from existing data quality fields already in the run manifest |
| Extended hallucination guard (section-by-section fallback) | Medium | Right direction, but partial fallback is architecturally harder than it appears — the LLM card is a coherent narrative; cleanly separating primary hypothesis from contributing causes requires structured LLM output format changes |
| Site-configurable multipliers in `OrchestratorConfig` | High | Trivial move from hardcoded constants; immediate auditability benefit with no scoring change |
| Analyst override with structured JSON | High | Well-designed; the four `override_rationale_category` values cover the realistic override scenarios; produces calibration signal over time |
| Score confidence intervals derived from data completeness | Medium | Derivation from data completeness is correct approach; visual representation in the card requires UI/renderer changes; manifest-level output is the lower-cost first step |

---

## SE Implementation Review — Code-Level Probe (2026-05-06)

*Third-pass review: each proposed fix evaluated against the actual code. Probes whether the fix will work as described, identifies what already exists, and flags design gaps in the proposed approach. All findings reference code confirmed in the repository as of 2026-05-06.*

### Pre-existing Infrastructure — Code-Verified Findings

**Issue 13 — Analyst override (Phase 3): CONFIRMED SOLVED.**
`_apply_accepted_state()` (line 414, `synthesis/analyst_override_processor.py`) sets exactly:
- `analyst_review["writeback_recommendation"] = "ready_if_accepted"`
- `executive_summary["decision_status"] = "candidate_ready"`
- `analyst_review["decision_required"] = False`

`writeback_ready` reads `decision_required` from the card (line 4681: `bool(analyst_review.get("decision_required", True))`). After `_apply_accepted_state()`, all three `writeback_ready` conditions that the override affects are satisfied. Phase 3 Issue 13 is a **connector verification only** — confirm the orchestrator's override path (line ~5873) calls `_build_workflow_dispatch()` after the processor. No schema or processor implementation needed.

**Remaining gap:** `degraded_reasons` is computed independently from the card state. An analyst override does NOT clear barrier degradation reasons. This is correct safety behavior — a hypothesis override and a barrier-gate acknowledgement are separate decisions and should remain separate.

**Issue 7 — novel_pattern: TSKR scorer IS in the repo.**
`orchestrators/tskr_temporal_scorer.py:1102` assigns `novel_pattern` with exactly:
```python
"novel_pattern": bool(
    effective_recurrence_count == 0   # no documentary history
    and history_score < 0.20          # weak documentary evidence
    and not bool(signal_ids)          # no signal match
)
```
The `documentary_novel` / `signal_novel` split maps directly:
- `documentary_novel = (effective_recurrence_count == 0 and history_score < 0.20)`
- `signal_novel = not bool(signal_ids)`
- Current `novel_pattern = documentary_novel AND signal_novel`

**Primary file to touch: `orchestrators/tskr_temporal_scorer.py`.** The plan listed wrong files (`rca_reasoning_orchestrator.py`, `causality_engine_v32.py`). The orchestrator only reads the field, not sets it.

**D2 — Scoring profiles: KG `superclass` field has zero coverage.**
KG demo nodes have no `superclass` or `causal_category` field — FM nodes carry only `fm_id` references via `failure_mode_refs`. The causality engine's `_infer_primary_category_for_failure_mode()` keyword-matches on `name + superclass + failure_mechanism + event_type`, but if `superclass` is null/empty, keyword matching must rely on `name` and `failure_mechanism` alone. KG population helpers (`demos/kg_population_helpers.py`) do not set `superclass` on FM nodes.

**D2 is blocked on a KG population prerequisite.** Before implementing scoring profiles, either: (a) add a controlled-vocabulary `causal_category` field to FM KG nodes and use it as the primary dispatch key, or (b) verify that existing `name`/`failure_mechanism` text is sufficient for keyword inference across G/I/L categories in the production KG. **Option (a) is mandatory** — keyword inference alone is not reliable enough for a safety-critical dispatch.

**Issue 8 — CMMS quality weighting: fully implementable with existing data.**
`_augment_kg_context_with_cmms_past_events()` (line 1737) already converts CMMS CR records to `past_events` entries with `resolved=False` for open CRs and `time_distance_days = days_before_event`. The causality engine's `_recurrence_features_for_candidate()` (line 3409) counts `pe.get("resolved") is False` → `unresolved_fm_count`. The `time_distance_days` field is already on each `past_events` entry.

**No schema changes needed.** The quality-weighted formula replaces the flat `0.10 × count` in `_recurrence_score_from_features()` by summing per-event weights derived from `time_distance_days`:
```python
# proposed quality weight per unresolved event
weight = 1.0 if time_distance_days > 365 else 0.4 if time_distance_days > 90 else 0.1
unresolved_boost = min(0.20, 0.10 × sum(weights_per_fm_event))
```
The data is already in scope. Change is self-contained in `_recurrence_features_for_candidate()` and `_recurrence_score_from_features()`.

---

### Phase-by-Phase Probe

**Phase 1, Issue 3 — Barrier gate degradation:**

`hard_gates["barrier_logic"]["degraded_mode"]` is already tracked per candidate (`causality_engine_v32.py:2343`). The orchestrator's `_build_workflow_dispatch()` detects SOE-present / PLC-absent and appends a `degraded_reasons` entry (line ~4702). `writeback_ready` fails when `degraded_reasons` is non-empty (line 4763: `and not degraded_reasons`). The gap is confirmed: there is no `barrier_gate_degraded_acknowledged` field that lets an analyst clear this specific entry and proceed. `coverage_acknowledgement_required` / `coverage_acknowledged` exist for data coverage gaps but are distinct.

**Implementation path:** Add `barrier_gate_degraded_acknowledged` to `analyst_review` in the card schema. Modify `_build_workflow_dispatch()`: before appending the barrier degradation entry to `degraded_reasons`, check if `analyst_review.barrier_gate_degraded_acknowledged == true`; if so, skip appending. This restores the writeback path without changing the barrier gate logic itself.

**Surrogate hierarchy (SOE tier-1 / alarm tier-2 / shift log tier-3):** Not yet implemented — barrier gate records only `degraded_mode: True/False`. Building the hierarchy requires `_apply_barrier_logic_gate()` to classify what surrogates it found. This is additive but independent of the acknowledged-flag fix. **Decouple the two**: ship the acknowledgement flag first (Phase 1), surrogate hierarchy classification second (can be Phase 2).

**Phase 1, Issue 5 — Fast transient Allen epsilon:**

`event.event_type` is a plain string field — confirmed at `rca_reasoning_orchestrator.py:3164`. No enum constraint exists. The Allen map builder does not emit attention flags; the flag must be emitted by the orchestrator after `_build_allen_relation_map()` returns. Implementation is self-contained.

**Confirmed design constraint:** `FAST_TRANSIENT_EVENT_TYPES` must be a `Set[str]` in `OrchestratorConfig`, not hardcoded. No facility-wide event type enum exists. Defaults: `{"reactor_trip", "eccs_actuation", "turbine_trip", "loss_of_feedwater"}`.

**Phase 1, Issue 11 — Category L floor check:**

`cmms_context` IS in scope in the orchestrator's main run loop and is passed to the synthesizer. It is NOT currently passed to any attention flag method. The new `_apply_category_l_floor_attention_flags()` must accept `cmms_context` as an explicit parameter. Use `cmms_context["recurrence_summary"]["open_cr_count"]` for the recurrence signal — already available without per-candidate joins.

**Confirmed design constraint:** The floor threshold must be in `OrchestratorConfig` (not hardcoded at 0.20). A site with sparse L-category KG coverage needs a lower threshold to avoid alert fatigue.

**Phase 2, Issue 2 — Symptom coverage ratio:**

`expected_anomaly_pattern` is **confirmed absent** from the KG. The demo KG nodes carry no such field (`kg_population_demo/nodes.json` inspected; KG population helpers do not set it). **Issue 2 full implementation is blocked on a KG population sprint.** Remove from Phase 2. The residual anomaly tracking variant — tag Allen nodes as "explained"/"residual" against the top candidate's FM name/mechanism text using simple substring matching — does not require this field and can ship independently as a Phase 2 item with a clear scope boundary.

**Phase 2, Issue 8 — CMMS recurrence quality (moved from Phase 5):**

`cmms_context.json` already has `cr_records[].status` (open/closed/cancelled/unknown) and `cr_records[].closed_date`. `ca_open_days` derivation: if `status == "open"`, use `(event_date − cr.created_date).days`; if `status == "closed"`, use `(closed_date − created_date).days`. No CMMS adapter schema changes required.

**Graceful degradation:** if `closed_date` is null on a closed CR (data quality issue), fall back to flat +0.10 bonus and record `cmms_recurrence_quality: "flat"` in manifest. The formula produces a `cmms_recurrence_quality: "weighted"` result only when all unresolved CRs have sufficient date fields.

**Phase 2, Issue 14 — Score confidence intervals:**

Degradation signals for the interval width are scattered: `hard_gates["barrier_logic"]["degraded_mode"]` (candidate-level), `temporal_contradiction` and `latency_violation_type` (temporal score dict), Allen map quality flags (run-level), missing TSKR scorer (run-level). Aggregating these into a per-candidate interval requires a post-scoring pass.

**Implementation path:** After `_combine_scores()` runs for a candidate, compute `n_full` (sub-scores from complete data) and `n_degraded` (sub-scores from proxy or missing data). Width = `f(n_degraded / 5)` — a simple fractional degradation. Store as `score_confidence_interval: {"lower": ..., "upper": ..., "width": ..., "degraded_dimension_count": n_degraded}` on each candidate. Do the manifest-level summary for top-3 candidates first; per-candidate output in a second pass.

**Phase 3, Issue 10 — Extended hallucination guard:**

The LLM produces the RCA card as a coherent JSON object. Partial fallback requires cleanly separating sections after generation. The synthesizer already has `_build_fallback_card()` for deterministic output. A per-section fallback means: accept `primary_hypothesis` + narrative from LLM, replace `contributing_causes` from deterministic template when `contributing_causes` contains invalid candidate IDs.

**Architectural concern:** The LLM narrative text (executive_summary.narrative) may reference contributing causes by name. If those causes are replaced by deterministic alternatives, the narrative becomes internally inconsistent — the prose refers to entities not in the structured fields. Add `synthesis_quality: "full_llm" | "partial_llm" | "deterministic"` to the card so analysts and reviewers know which sections are LLM-generated and which are template-generated. Document that under `"partial_llm"`, narrative consistency is not guaranteed.

**Phase 3, Issue 13 — Analyst override (see above):**
Verify `_apply_accepted_state()` field assignments before writing new code. If the existing implementation already sets the right fields, this phase is complete with minor routing verification only.

**Phase 4, Issue 1 — D1 (`temporal_score_quality`):**

`temporal_score_quality` must be set in `_apply_allen_temporal_blend()` per candidate based on whether `_build_allen_component_index()` returned a non-empty entry for the candidate's `component_id`. If the Allen map exists but the component had no classified signals, the score remains `"proxy"` even after the blend pass. This is per-component logic, not per-run. A candidate whose component has no Allen node gets `temporal_score_quality: "proxy"` regardless of whether other candidates in the same run got `"full_allen"`.

**Phase 4, Issue 4+6 — D2 (scoring profiles):**

`_infer_primary_category_for_failure_mode()` uses keyword matching on the FM node's `name + superclass + failure_mechanism + event_type`. For a Category G (human performance) failure mode, the FM `superclass` field must contain keywords from `_CATEGORY_KEYWORDS["G"]` for dispatch to work.

**Critical dependency:** If KG FM nodes for human factors and organizational categories use controlled vocabulary that doesn't match these keywords (e.g., superclass = "HFE" rather than "human performance"), the dispatch will silently apply the equipment-origin weight profile to G/I/L candidates. Before implementing D2: run a KG query against all FM nodes with categories G, I, L and verify their `superclass` values match the keyword set. If they don't, add a direct `causal_category` field to FM KG nodes as an authoritative label alongside the keyword-inferred category, and use it as the primary dispatch key. Keyword inference as fallback only.

---

### Priority Adjustments — Final Code-Verified Status

| Issue | Plan Phase | Status | Finding |
|---|---|---|---|
| **13** — Analyst override | Phase 3 | **Scope reduced** | Schema + processor fully implemented. `_apply_accepted_state()` confirmed to set all required `writeback_ready` fields. Phase 3 = connector verification only. |
| **7** — novel_pattern decomposition | Phase 2 | **File corrected** | Primary change is `orchestrators/tskr_temporal_scorer.py:1102`. Three conditions confirmed. Plan listed wrong files. |
| **2** — Symptom coverage (full) | Phase 2 | **BLOCKED — remove** | `expected_anomaly_pattern` confirmed absent from KG. Move full implementation post KG-population sprint. Residual-anomaly variant (substring match) can ship independently. |
| **D2** — Scoring profiles | Phase 4 | **PREREQUISITE ADDED** | KG nodes carry no `superclass` or `causal_category` field. Must add controlled-vocabulary `causal_category` to FM KG nodes before D2 is implementable. Keyword inference alone is unsafe for safety-critical dispatch. |
| **8** — CMMS quality weighting | Phase 2 (moved) | **Confirmed implementable** | `_augment_kg_context_with_cmms_past_events()` already sets `resolved` from CMMS status and `time_distance_days` on past_events. Change confined to `_recurrence_features_for_candidate()` and `_recurrence_score_from_features()`. No external dependencies. |
| **3** — Barrier gate surrogate hierarchy | Phase 1 | **Decouple** | Ship `barrier_gate_degraded_acknowledged` flag first; surrogate hierarchy is independent, move to Phase 2. |
| **5** — Fast transient event types | Phase 1 | **Config constraint confirmed** | `event_type` is a plain string; no enum. `FAST_TRANSIENT_EVENT_TYPES` must be `Set[str]` in `OrchestratorConfig`. |
| **11** — Category L floor | Phase 1 | **Parameter gap confirmed** | `cmms_context` is in scope in `run()` but NOT passed to any attention flag method. New method signature must include it explicitly. |
| **D1** — `temporal_score_quality` | Phase 4 (Option B) | **Logic confirmed** | Set per-component in `_apply_allen_temporal_blend()` based on component_id match in Allen index. Not per-run. |
| **10** — Partial hallucination fallback | Phase 3 | **Narrative risk noted** | Add `synthesis_quality: "full_llm"\|"partial_llm"\|"deterministic"` field to card; document that partial fallback does not guarantee narrative-structure consistency. |



## Phase 1 Implementation Record (2026-05-06)

<!-- @code: orchestrators/rca_reasoning_orchestrator.py | RCAReasoningOrchestrator -->
<!-- @status: implemented -->
<!-- @reviewed: 2026-05-06 -->

Three Phase 1 changes are now implemented and tested in `rca_reasoning_orchestrator.py`. All are additive (no scoring changes, no ranking effects). 1388 unit tests pass.

### Issue 3 — Barrier Gate Degradation Acknowledgement

**What changed:** `_compute_review_hooks()` now reads `analyst_review.barrier_gate_degraded_acknowledged` (bool, default `False`) before appending the SOE/PLC pairing degradation reason to `degraded_reasons`. When the analyst sets this field to `True`, the barrier-gate degradation entry is suppressed and the `writeback_ready` conjunction can proceed — provided all other conditions are met.

**Why:** The barrier logic gate runs in degraded mode when SOE is present but `protection_logic_context` is absent. Previously the analyst had no way to record that they had verified barrier status by an alternate means (physical walkdown, PLC historian, operator statement) and proceed to writeback. They could only hold the record.

**Analyst action:** Set `rca_card.analyst_review.barrier_gate_degraded_acknowledged = true` after verifying barrier status through an alternate means. The intent and means of verification should be recorded in `analyst_review` free-text fields for auditability.

**Schema field added to `analyst_review` block:**
```json
"barrier_gate_degraded_acknowledged": false
```

**Test file:** `unit_tests/test_phase1_hardening.py` — `TestBarrierGateDegradedAcknowledged` (6 tests).

---

### Issue 5 — Fast-Transient Allen Epsilon Flag

**What changed:** New static method `_apply_fast_transient_attention_flags(rca_card, event, allen_relation_map, fast_transient_event_types)`. Called in `run()` after `_apply_ishikawa_skip_attention_flag`. New config field `OrchestratorConfig.fast_transient_event_types: Set[str]` (default: `{"reactor_trip", "eccs_actuation", "turbine_trip", "loss_of_feedwater"}`).

**Firing condition:** `event.event_type` in `fast_transient_event_types` AND `allen_relation_map.summary.causal_nodes > 0`.

**Why:** For fast transients, the entire causal sequence may unfold in seconds. The Allen epsilon of 0.5 hours is larger than the causal sequence itself, making all interval-based relation assignments unreliable. Without this flag, a run on a reactor trip produces temporal scores that look authoritative but are based on meaningless interval classifications.

**Flag text placed in `rca_card.executive_summary.analyst_attention_flags`:**
> *"Fast-transient event detected (event_type='reactor_trip'). Allen temporal epsilon (0.5 h) exceeds the causal sequence duration — interval relation assignments for N causal signal(s) may be unreliable. Verify causal ordering using SOE or PLC timestamps at sub-minute resolution before accepting temporal-score contributions for this run."*

**Site customization:** Add or remove event types via `OrchestratorConfig(fast_transient_event_types={...})`.

**Test file:** `unit_tests/test_phase1_hardening.py` — `TestFastTransientAttentionFlags` (8 tests).

---

### Issue 11 — Category L Organizational Floor Check

**What changed:** New static method `_apply_category_l_floor_attention_flags(rca_card, causality_candidates, cmms_context, category_l_score_floor)`. Called in `run()` after `_apply_ishikawa_skip_attention_flag`. New config field `OrchestratorConfig.category_l_score_floor: float` (default: `0.20`).

**Firing condition:** No Category L candidate has `composite_score >= category_l_score_floor` AND at least one recurrence signal is present (`causality_candidates.recurrence_summary.candidate_count_with_recurrence > 0` OR `cmms_context.recurrence_summary.open_cr_count > 0`).

**Why:** A Category L (systemic/organizational) root cause is structurally low-scoring because the composite formula is oriented toward equipment failure modes with KG topology and telemetry signatures. Without this flag, L-category hypotheses can silently score below any reporting threshold with no analyst alert — even when a clear recurrence pattern exists that is a classic indicator of organizational factors.

**Flag text placed in `rca_card.executive_summary.analyst_attention_flags`:**
> *"No Category L (systemic/organizational) candidate reached the score floor (0.20) despite a recurrence signal (recurrence history present; 3 open CR(s) in CMMS). (Category L coverage status: no_supporting_data) Document explicitly why organizational root cause does not apply before writeback."*

**Site customization:** Lower `category_l_score_floor` (e.g., `0.10`) for KGs with sparse L-category nodes to avoid alert fatigue.

**Test file:** `unit_tests/test_phase1_hardening.py` — `TestCategoryLFloorAttentionFlags` (9 tests).

---

## Phase 2 Implementation Record (2026-05-06)

<!-- @code: orchestrators/tskr_temporal_scorer.py | TSKRTemporalScorerV1._score_failure_mode_pattern -->
<!-- @code: orchestrators/rca_reasoning_orchestrator.py | RCAReasoningOrchestrator -->
<!-- @code: orchestrators/causality_engine_v32.py | RuleBasedCausalityEngineV32 -->
<!-- @status: implemented -->
<!-- @reviewed: 2026-05-06 -->

Five Phase 2 changes are now implemented. All are additive — no ranking or scoring-path changes. Issue 14 (score confidence interval) deferred to Phase 4 (depends on D1 `temporal_score_quality` field). 1417 unit tests pass.

### Issue 7 — `novel_pattern` Decomposition

**What changed:** `tskr_temporal_scorer.py:_score_failure_mode_pattern()` now emits two new fields alongside the retained `novel_pattern`:

```python
"documentary_novel": bool(effective_recurrence_count == 0 and history_score < 0.20),
"signal_novel":      not bool(signal_ids),
"novel_pattern":     bool(...)   # AND of both — backward compat
```

**Four analyst-meaningful states:**

| `documentary_novel` | `signal_novel` | Meaning |
|---|---|---|
| False | False | Known pattern, documented history — lowest concern |
| False | True | Documented in CRs/WOs but no signal match — possible documentation quality issue |
| **True** | **False** | **New to documentary record but signal-matched — dangerous case, now explicitly flagged** |
| True | True | Genuinely novel — prior behavior correctly captured this |

**Test file:** `unit_tests/test_phase2_enrichment.py` — `TestNovelPatternDecomposition` (5 tests).

---

### Issue 12 — Tier Confidence Multipliers to `OrchestratorConfig`

**What changed:** `OrchestratorConfig` now has `tier_confidence_multipliers: Dict[str, float]` (default: `{"plant": 1.00, "fleet": 0.80, "industry": 0.60}`). The `_build_similar_event_list()` instance method now reads from `self.config.tier_confidence_multipliers` instead of the module-level constant. The module-level `TIER_CONFIDENCE_MULTIPLIERS` in `similar_event_adapter.py` is retained for backward compatibility. Values appear in `run_manifest.pipeline_config.tier_confidence_multipliers` for audit.

**Site customization:** `OrchestratorConfig(tier_confidence_multipliers={"plant": 1.0, "fleet": 0.90, "industry": 0.75})`.

**Test file:** `TestTierMultipliersConfig` (3 tests).

---

### Issue 9 — Scope Expansion Suggestion Confidence

**What changed:** Each signal dict in `run_context.scope_management.expansion_suggestions` now has two new fields: `suggestion_confidence` (`"low"` | `"medium"`) and `suggestion_confidence_reason` (string or None).

**Confidence derivation by source:**

| Source | Confidence | Reason |
|---|---|---|
| Allen map — clean quality flags | `"medium"` | None |
| Allen map — SOE clock sync failed | `"low"` | `"soe_clock_sync_failed"` |
| Allen map — alarm clock sync failed | `"low"` | `"alarm_clock_sync_failed"` |
| Allen map — SOE nodes capped | `"low"` | `"soe_nodes_capped"` |
| Signal propagation chain | `"medium"` | None |
| TSKR novel pattern | `"low"` | `"novel_pattern_sparse_evidence"` |

**Test file:** `TestScopeExpansionSuggestionConfidence` (7 tests).

---

### Issue 8 — CMMS Recurrence Quality Weighting

**What changed:** `causality_engine_v32._recurrence_features_for_candidate()` now computes a time-weighted unresolved boost for FM-level past events. Weight per unresolved event: `1.0` if `time_distance_days > 365`, `0.4` if `> 90`, `0.1` otherwise. Graceful fallback to flat formula when `time_distance_days` is absent on any event. The `_recurrence_score_from_features()` method accepts a new optional `weighted_unresolved_fm_boost` parameter that replaces the flat count when provided.

**New field on recurrence features:** `cmms_recurrence_quality: "weighted" | "flat"`.

**Run-level manifest field:** `pipeline_config.cmms_recurrence_quality` — `"weighted"` when all candidates used the quality formula, `"flat"` otherwise, `"n/a"` when no candidates.

**Data source:** `time_distance_days` is already set on `past_events` by `_augment_kg_context_with_cmms_past_events()`. No CMMS schema changes required.

**Test file:** `TestCmmsRecurrenceQualityWeighting` (6 tests).

---

### Issue 2 (residual variant) — Residual Anomaly Tracking

**What changed:** New static method `_apply_residual_anomaly_gaps(rca_card, allen_relation_map, causality_candidates)`, called from `run()` after the card is built. Tags each causal-candidate Allen node as "explained" (component matches primary hypothesis) or "residual" (causal but on a different component).

**New field on `rca_card`:** `unresolved_gaps`:
```json
{
  "explained_causal_node_count": 1,
  "residual_causal_node_count": 1,
  "residual_nodes": [
    {
      "node_id": "N-002",
      "node_type": "anomaly",
      "component_id": "COMP-B",
      "allen_relation_to_event": "overlaps",
      "allen_score": 0.8,
      "gap_label": "Causal signal on component 'COMP-B' (Allen: overlaps) is not explained by primary hypothesis (bearing_degradation)."
    }
  ],
  "assessment": "partial"
}
```
`assessment` values: `"complete"` (no residuals), `"partial"` (some explained, some residual), `"unexplained"` (all causal nodes on other components).

Nodes with `allen_relation_to_event == "follows"` are excluded (temporal contradiction — handled by the barrier gate, not a gap).

When residual nodes exist, an attention flag is added to `rca_card.executive_summary.analyst_attention_flags`.

**Scope boundary:** This is the residual-anomaly variant only. Full symptom coverage ratio (Issue 2 proper) remains blocked on `expected_anomaly_pattern` field being added to FM KG nodes.

**Test file:** `TestResidualAnomalyGaps` (8 tests).

---

## Phase 3 Implementation Record (2026-05-06)

<!-- @code: synthesis/rca_synthesizer_v31.py | RuleValidatedRCASynthesizerV31._validate_and_repair_llm_sections -->
<!-- @status: implemented -->
<!-- @reviewed: 2026-05-06 -->

Two Phase 3 issues resolved. 1433 unit tests pass.

### Issue 10 — Extended Hallucination Guard in Synthesizer

**Gap (pre-Phase 3):** The existing hard-reject gate in `synthesize()` only checked `primary_hypothesis.candidate_id` against `_all_input_candidate_ids`. If the LLM invented a candidate ID in `contributing_causes[]` or `alternatives[]`, the card was kept and the fabricated IDs surfaced in `_validate_card_semantics` only as error strings — they were not removed from the output. A fabricated ID in `recommended_actions[i].linked_candidate_id` or `evidence[i].linked_candidate_id` was similarly only flagged, not repaired.

**What changed:** New static method `_validate_and_repair_llm_sections(card, all_input_candidate_ids)` in `rca_synthesizer_v31.py`, called between the primary hard-reject check and `_validate_card_semantics`:

- **`contributing_causes[]`** — entries whose `candidate_id` is not in `_all_input_candidate_ids` are removed entirely.
- **`alternatives[]`** — same removal rule.
- **`recommended_actions[i].linked_candidate_id`** — nullified (set to `None`) if not in valid set.
- **`evidence[i].linked_candidate_id`** — nullified if not in valid set.
- Returns `int` repair count; a count > 0 sets `synthesis_quality = "partial_llm"`.

**No template replacement** — fabricated contributing causes are dropped, not substituted. This preserves narrative coherence for the remaining sections (a deterministic replacement string would reference a cause the LLM never analysed).

**New `validation_status.synthesis_quality` field:**

| Value | Condition |
|---|---|
| `"full_llm"` | LLM card accepted with zero repairs |
| `"partial_llm"` | LLM card accepted but ≥1 hallucinated secondary ID was removed/nullified |
| `"deterministic"` | Fallback card used (LLM failed hard-reject or generation error) |

The field is also preset in `_normalize_llm_output()` (`"full_llm"`) and `_fallback_card()` (`"deterministic"`) so the key is always present regardless of path.

**Primary hypothesis remains protected by the earlier hard-reject gate** — an invented primary ID still discards the entire LLM card.

**Files changed:** `synthesis/rca_synthesizer_v31.py` only.

**Test file:** `unit_tests/test_phase3_hallucination_guard.py` — `TestValidateAndRepairLlmSections` (10 tests) + `TestSynthesisQualityField` (6 tests).

---

### Issue 13 — Connector Verification (override path)

**Finding:** No code change needed.

`apply_override()` in `rca_reasoning_orchestrator.py` calls `AnalystOverrideProcessor().apply()` → `_apply_accepted_state()`, which sets `writeback_recommendation = "ready_if_accepted"`, `decision_status = "candidate_ready"`, and `decision_required = False` directly on the card. The writeback path reads these fields from the card — it does not re-evaluate `_build_workflow_dispatch()`. The connector is therefore correct: an accepted override immediately makes the card writable without requiring a full re-run.

The pre-flight analysis confirmed that the concern in the original issue was based on a misread of the execution sequence. No behavioral gap exists.

---

## Phase 4a Implementation Record (2026-05-06)

<!-- @code: orchestrators/causality_engine_v32.py | RuleBasedCausalityEngineV32._apply_allen_temporal_blend -->
<!-- @status: implemented -->
<!-- @reviewed: 2026-05-06 -->

One change. No scoring-path changes; field is purely informational. 1440 unit tests pass.

### Issue 1 / D1 — `temporal_score_quality` per-component field

**What changed:** `_apply_allen_temporal_blend()` in `causality_engine_v32.py` now sets `candidate["scores"]["temporal_score_quality"]` at every return path:

| Execution path | Value |
|---|---|
| Component found in `causal_scores` (Allen match, blend applied or clamped) | `"full_allen"` |
| Component not found in `causal_scores` (no Allen node for this component) | `"proxy"` |
| Component in `follow_ids` (temporal contradiction — blend suppressed) | `"proxy"` |

**Key invariant:** quality is per-component, not per-run. Within a single `refine_with_evidence()` call, a candidate whose component has an Allen causal node receives `"full_allen"` while a candidate on a different component with no node receives `"proxy"` — even though both candidates are in the same event analysis.

**Analyst interpretation:**
- `"full_allen"` — temporal score reflects direct interval-algebra measurement from plant SOE/alarm data; higher confidence in the timing dimension.
- `"proxy"` — temporal score comes from TSKR keyword-inference proxy; timing evidence is estimated, not measured. Treat temporal component of composite score with correspondingly lower confidence.

**Note on SOE node exclusion:** `_build_allen_component_index()` excludes SOE and alarm nodes from `causal_scores` (only `anomaly`-type causal nodes contribute). A component that has only SOE/alarm nodes in the Allen map will have no entry in `causal_scores` and will therefore receive `temporal_score_quality = "proxy"` — correctly reflecting that no anomaly-class signal grounded the Allen blend.

**Files changed:** `orchestrators/causality_engine_v32.py` only.

**Tests added:** 7 new tests in `unit_tests/test_finding_g_allen_scoring.py` under the `# Phase 4a` section (`test_tsq_*`).

**Remaining Phase 4 work:**
- **Phase 4b** — Issue 14 (score confidence interval): can now be implemented; depends on `temporal_score_quality` being present (satisfied by this change).
- **Phase 4c** — D2 (scoring profiles for G/I/L categories): **COMPLETE** — Steps 1–3 (config/dispatch/output fields) and Steps 4a–4c (schema, engine curated-read, helper function, fixture updates) all implemented. Steps 5–6 pending SE review and production KG access.

---

## Phase 4b Implementation Record (2026-05-06)

<!-- @code: orchestrators/causality_engine_v32.py | RuleBasedCausalityEngineV32._apply_score_confidence_interval -->
<!-- @status: implemented -->
<!-- @reviewed: 2026-05-06 -->

One change. No scoring changes; field is purely informational. 1458 unit tests pass.

### Issue 14 — `score_confidence_interval` per candidate

**Gap (pre-Phase 4b):** Candidates carried a single composite score with no indication of how much trust to place in that number. A score of 0.62 from five fully-grounded dimensions is a qualitatively different result from the same score when two of those dimensions are proxy or absent — but both looked identical in output.

**What changed:** New static method `_apply_score_confidence_interval(candidate)` in `causality_engine_v32.py`, called after the gate loop (physical plausibility, timeline consistency, barrier logic) and before the sort. Each candidate gets:

```json
"score_confidence_interval": {
  "lower": 0.40,
  "upper": 0.80,
  "width": 0.40,
  "degraded_dimension_count": 2,
  "degraded_dimensions": ["temporal", "evidence"]
}
```

**Degradation signals (one per scoring dimension):**

| Dimension | Degraded when |
|---|---|
| `structural` | `hard_gates.physical_plausibility.degraded_mode == True` |
| `temporal` | `scores.temporal_score_quality == "proxy"` (no Allen causal match for this component) |
| `telemetry` | `scores.telemetry == 0.0` (no telemetry signal contributed) |
| `evidence` | `observationally_ungrounded == True` (no affects-class evidence) |
| `governance` | `hard_gates.barrier_logic.degraded_mode == True` |

**Width formula:** `width = n_degraded / 5` (linear 0.0 → 1.0).

**Interval:** symmetric around composite score, clamped to [0, 1]:
- `lower = max(0.0, composite_score − width / 2)`
- `upper = min(1.0, composite_score + width / 2)`

**Analyst interpretation:** A narrow interval (width ≤ 0.2) with a high composite score is a strong result. A wide interval (width ≥ 0.6) means the score is driven heavily by proxy or absent data — the number should inform investigation priority, not substitute for it.

**Dependency on Phase 4a:** The `temporal` dimension uses `temporal_score_quality` introduced in Phase 4a. Without Phase 4a, temporal would always appear as degraded regardless of Allen coverage.

**Files changed:** `orchestrators/causality_engine_v32.py` only.

**Tests added:** 18 tests in `unit_tests/test_phase4b_score_confidence_interval.py` — `TestApplyScoreConfidenceInterval` (14 unit tests) + `TestScoreConfidenceIntervalIntegration` (3 integration tests).

---

## Phase 4c Implementation Record (2026-05-06) — Steps 1–3

<!-- @code: orchestrators/causality_engine_v32.py | RuleBasedCausalityEngineV32._scoring_profile_for_fm -->
<!-- @code: orchestrators/causality_engine_v32.py | CausalityEngineConfigV32 -->
<!-- @status: implemented -->
<!-- @reviewed: 2026-05-06 -->

Steps 1–3 implemented (infrastructure + wiring). Steps 4a–4c also implemented (see §Phase4c-Steps4). 1504 unit tests pass.

### Steps 1–3 — Scoring Profiles: Config, Dispatch, and Output Field

**What changed:**

**Step 1 — `scoring_profiles` in `CausalityEngineConfigV32`.**
New field `scoring_profiles: Optional[Dict[str, Dict[str, float]]]`, initialized in `__post_init__` from the module-level `_DEFAULT_SCORING_PROFILES` constant. Every profile is validated at startup: must have exactly the five scoring dimensions and sum to 1.0. Sites override via `CausalityEngineConfigV32(scoring_profiles={...})` — partial overrides (only the categories that differ) are supported.

**Step 2 — `_scoring_profile_for_fm(category)`.**
New instance method on `RuleBasedCausalityEngineV32`. Replaces `_governance_weight_for_fm()` at the `generate()` call site. Returns a copy of the full five-dimension weight dict for the given category letter; falls back to category `"A"` for unrecognised categories. `_governance_weight_for_fm()` is retained as dead code for backward compat with any external callers.

`_refresh_candidate_confidence_and_thresholds()` updated to prefer `scores["scoring_profile_weights"]` as the full `weights_override` when present; falls back to the legacy `scores["governance_weight"]`-only override for candidates generated before this change.

**Step 3 — `score_profile_applied` and `scoring_profile_weights` fields.**
Every FM candidate now carries in its `scores` dict:
- `score_profile_applied` — profile name string (e.g. `"human_performance"`, `"organizational"`)
- `scoring_profile_weights` — copy of the full weight dict used for this candidate's composite
- `governance_weight` — retained (equals `scoring_profile_weights["governance"]`) for backward compat

**SE-assessed default profiles (all sum to 1.00):**

| Cat | Profile | structural | temporal | telemetry | evidence | governance | Rationale |
|---|---|---|---|---|---|---|---|
| A–F | `equipment_origin` | 0.30 | 0.20 | 0.20 | 0.20 | 0.10 | Current weights — unchanged for equipment-origin hypotheses |
| G | `human_performance` | 0.05 | 0.10 | 0.05 | 0.65 | 0.15 | Procedure/WO record dominates; telemetry/topology uninformative |
| H | `design_deficiency` | 0.15 | 0.05 | 0.20 | 0.45 | 0.15 | Temporal guaranteed (latent flaw); telemetry captures margin exceedances |
| I | `change_control` | 0.05 | 0.25 | 0.10 | 0.45 | 0.15 | Change date vs. event date is primary causal test; temporal raised vs. G |
| J | `surveillance` | 0.05 | 0.05 | 0.05 | 0.55 | 0.30 | Regulatory/OE record is primary diagnostic; governance raised |
| K | `vendor_procurement` | 0.10 | 0.10 | 0.05 | 0.50 | 0.25 | Traceability + industry OE dominate |
| L | `organizational` | 0.05 | 0.05 | 0.05 | 0.60 | 0.25 | No topology/telemetry signal; documentary record and CAP are sole sources |

**Note on plant-dependence:** H telemetry (0.20) and J governance (0.30) weights are site-configurable precisely because their informativeness depends on historian coverage depth and CAP/OE integration fidelity. Sites with sparse telemetry for design-deficiency hypotheses should lower H telemetry and raise H evidence accordingly.

**Files changed:** `orchestrators/causality_engine_v32.py` only.

**Tests:** 23 tests in `unit_tests/test_phase4c_scoring_profiles.py`.

### Steps 4–6 — Blocked

| Step | What | Blocker |
|---|---|---|
| 4 | Add `causal_category` controlled-vocabulary field to FM KG nodes; update `kg_population_helpers.py` | KG schema/population work — out of engine scope |
| 5 | Query production KG to verify G–L category coverage | After step 4 |
| 6 | Before/after ranking comparison on show-and-tell cases | After step 5 |

Until step 4 is complete, G–L candidates in the production KG whose `superclass`/`name` keywords don't match `_CATEGORY_KEYWORDS` will infer as category `"A"` and receive `equipment_origin` weights. The `score_profile_applied` field makes this visible in every candidate's output — analysts and calibration engineers can query it to identify misclassifications before the KG is updated.

---

## Phase 4c Steps 4a–4c Implementation Record (2026-05-06) {#Phase4c-Steps4}

<!-- @code: orchestrators/causality_engine_v32.py | RuleBasedCausalityEngineV32._infer_primary_category_for_failure_mode -->
<!-- @code: demos/kg_population_helpers.py | assign_causal_category -->
<!-- @schema: schemas/kg_context.json -->
<!-- @status: implemented -->
<!-- @reviewed: 2026-05-06 -->

Steps 4a–4c are the prerequisite KG-side changes that unlock authoritative scoring profile dispatch. All three steps complete. 1504 unit tests pass.

### Step 4a — `causal_category` and `causal_category_source` in `kg_context.json` schema

Two new optional properties added to the `failure_modes[]` item schema:

```json
"causal_category": {
  "type": "string",
  "enum": ["A","B","C","D","E","F","G","H","I","J","K","L"],
  "description": "Authoritative causal category letter (A–L). Takes precedence over runtime keyword inference when present."
},
"causal_category_source": {
  "type": "string",
  "enum": ["curated", "inferred"],
  "description": "How causal_category was assigned: 'curated' = reviewed by a qualified SE; 'inferred' = derived by automated keyword matching."
}
```

`causal_category_source` makes the provenance of every dispatch decision visible in the output — analysts can immediately distinguish SE-reviewed assignments from automated inference.

**File changed:** `schemas/kg_context.json`.

### Step 4b — Engine reads curated `causal_category` before keyword inference

`_infer_primary_category_for_failure_mode()` updated with a curated-first guard:

```python
curated = str(fm.get("causal_category") or "").strip().upper()
if curated in cls._CATEGORY_PROFILE_NAMES:   # all 12 categories
    return curated, []                        # no keyword inference; no alternatives
# … existing keyword inference …
```

The guard short-circuits for all 12 valid letters, including `"A"` (which was previously the default-only category, not in `_CATEGORY_KEYWORDS`). When a curated value is present, `category_alternatives` is empty — the analyst card does not show spurious alternatives for a category that has been reviewed.

**Motivating case (TC-6):** `FM-MFPB-PROC-CRITERION-GAP` has `superclass = "procedure_acceptance_criteria_gap"` — keyword inference returns `"A"` (no keyword hit). With `causal_category = "I"` (curated), the engine correctly dispatches to the `change_control` profile (evidence = 0.45, temporal = 0.25), which appropriately weights the change-date vs. event-date test for a procedural criterion gap.

**Also added to `kg_population_helpers.py`:** `assign_causal_category(fm_node) -> Tuple[str, str]` — shared function for any population pipeline to call before ingesting FM nodes. Returns `(category, source)`. If `fm_node["causal_category"]` is already a valid letter, returns `(value, "curated")`; otherwise keyword-matches on `name + superclass + failure_mechanism` and returns `(inferred, "inferred")`. `_CAUSAL_CATEGORY_KEYWORDS` in helpers mirrors `_CATEGORY_KEYWORDS` in the engine so keyword inference is consistent across both modules.

**Files changed:** `orchestrators/causality_engine_v32.py`, `demos/kg_population_helpers.py`.

### Step 4c — All 7 show-and-tell fixture `kg_context.json` files updated

SE-assessed `causal_category` and `causal_category_source` values added to every FM node in all 7 test case fixtures. Selection rationale:

| Fixture | FM node | Category | Source | Rationale |
|---|---|---|---|---|
| TC-1 | FM_BEARING_WEAR | A | inferred | Mechanical wear — no keyword hit → A correct |
| TC-2 | FM_HVAC_SUPPORT_DEGRAD | B | curated | Support system (HVAC) — B correct; keyword could miss |
| TC-2 | FM_AIR_INLEAK / FM_COND_FOULING / FM_FWCV_INSTAB / FM_VAC_INST_BIAS | A | inferred | Equipment-origin degradation |
| TC-3 | FM-CW-TEMP-RISE / FM-HVAC-DEGRAD | B | curated | Support system boundary |
| TC-3 | FM-CND-AIR-INLEAK / FM-CND-TUBE-FOUL / FM-CND-TUBE-LEAK | A | inferred | Equipment-origin |
| TC-4 | FM-FW-TRANSIENT | E | inferred | Off-design transient keyword match |
| TC-4 | FM-NI-SPURIOUS / FM-CRD-MECHANICAL | A | inferred | No keyword hit |
| TC-5 | FM-HPCI-CCF-COUPLING | A | curated | CCF handled by scoring delta; category A so profile doesn't interfere |
| TC-5 | FM-HPCI-A/B-WEAR | A | inferred | Equipment wear |
| TC-6 | FM-MFPB-LUBE-OIL-OMISSION | G | curated | Maintenance omission — human performance |
| TC-6 | FM-MFPB-PROC-CRITERION-GAP | I | curated | Procedure criterion gap — change control |
| TC-6 | FM-MFPB-LO-BYPASS-NOT-RESTORED | G | curated | Restoration step omitted — human performance |
| TC-6 | FM-MFPB-LO-COOLER-FOULING / FM-MFPB-BEARING-WEAR | A | inferred | Equipment-origin |
| TC-7 | FM-RCPC-SEAL-CV-DRIFT | A | curated | Instrument drift — stays A; no change-control flag |
| TC-7 | FM-SWHX4C-FOULING | B | curated | Support system heat exchanger |
| TC-7 | FM-RCPC-SEAL-WEAR | A | inferred | Mechanical seal wear |

**Files changed:** all 7 `tests/test_case_N/fixtures/kg_context.json` files.

### Tests

23 new tests in `unit_tests/test_phase4c_causal_category_dispatch.py`:

- **`TestAssignCausalCategory`** (10 tests) — curated round-trip, override of keyword inference, case/whitespace normalization, invalid value fallthrough, empty node, per-category acceptance
- **`TestEngineReadsCuratedCategory`** (8 tests) — curated G bypasses inference, curated I bypasses inference, missing field uses inference, curated A accepted (not treated as missing), empty/None/invalid values fall through
- **`TestTC6CuratedCategoryDispatch`** (5 tests) — fixture schema check, curated field values, `generate()` produces G-category candidates, G candidates use `human_performance` profile

### Steps 5–6 — Pending SE review and production KG access

| Step | What | Status |
|---|---|---|
| 5 | Query production KG: count FM nodes by `causal_category`; identify nodes with `causal_category_source = "inferred"` that SE should review | Blocked on production KG access |
| 6 | Before/after ranking comparison on show-and-tell cases with G–L categories | After step 5 |

The 7 fixture updates (step 4c) provide a controlled before/after baseline for show-and-tell cases. Steps 5–6 extend this to the full production KG.
