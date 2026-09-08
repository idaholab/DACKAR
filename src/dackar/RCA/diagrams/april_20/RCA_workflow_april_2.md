# Formal Method Specification — RCA with Knowledge Graph and Chroma
Updated: April 6th, 2026
Baseline: Orchestrator v3.2, Schema set v3.2, validated fixture-driven workflow
The method is intended to be machine-readable, auditable, and explainable for nuclear equipment reliability investigations
Knowledge-graph structure (Neo4j) and vector retrieval (Chroma) are as of now both inputs to the RCA pipeline
LLM output is advisory. The analyst remains the final decision-maker
The deterministic fallback synthesis path is currently the production path. The LLM path (OllamaLLMClient) exists but is not yet validated end-to-end.

## 1. Purpose and Scope
This method defines the current RCA workflow for single-event analysis augmented by historical events, document corpora, system architecture, telemetry, and maintenance/governance data. The target user is a system engineer or reliability engineer performing structured troubleshooting with analyst sign-off.
The workflow is explicitly decision support, not decision automation. All final technical and regulatory conclusions remain the responsibility of the licensed engineer. The system does not self-approve a conclusion.

## 2. End-to-End Artifact Chain
Important architectural note: The candidate set goes through two distinct scoring passes. The pre-evidence candidates (v1) are ranked on structure, timing, and telemetry alone. The post-evidence candidates (v2) are re-ranked after the evidence bundle is retrieved. The delta between v1 and v2 ranking is the most important diagnostic signal for whether evidence retrieval is providing discriminating value.
 
## 3. System Architecture and Interfaces
The current implementation uses Neo4j for graph traversal and Chroma for vector retrieval. The orchestrator stitches together typed stage outputs rather than free-form intermediate text. This keeps the method inspectable and simplifies validation.
### LLM Client Status
Two LLM clients are implemented:
DummyLLMClient — development stub that intentionally raises, forcing the deterministic fallback path. Currently the default in build_dev_orchestrator().
OllamaLLMClient — real implementation using the Ollama /api/generate endpoint with format: json. Exists in the codebase but not yet validated end-to-end against the full schema contract.
The deterministic fallback path in RuleValidatedRCASynthesizerV31._fallback_card() is the main output: it deterministically performs confidence calibration, recurrence interpretation, common-cause interpretation, evidence posture derivation, and analyst review question generation.

## 4. Formal Data-Model Expectations
The canonical machine contract lives in the JSON schemas under schemas/. Two important schema consistency issues are currently open and must be resolved before the next iteration:
Issue — confidence label case mismatch: causality_candidates.json (runtime schema) uses lowercase confidence labels ("high", "medium", "low", "speculative"). causality_candidates.v3_2.schema.json uses uppercase ("HIGH", "MEDIUM", "LOW", "SPECULATIVE"). The causality engine v32 produces lowercase. The schema validator will reject output depending on which file it loaded. Resolution required before next release.
Issue — kg_context dual schema: The v2 kg_context.json schema has a different upstream_paths structure than the runtime kg_context produced by Neo4jKGContextBuilder. The orchestrator currently runs with stop_on_validation_error=False to tolerate this. Resolution required for strict validation runs.

## 5. Knowledge Graph Design
The KG is the structural backbone of the RCA method. It defines the causal search space in Stage 1 and preserves accepted RCA conclusions for future recurrence analysis after analyst write-back.
### KG Governance — Open Issues
The following governance questions are unresolved and must be addressed before production deployment:
Initial population: Where does the KG come from? Plant MBSE models, vendor data, or manual entry? What is the seeding process for a new unit?
FMEA taxonomy maintenance: Who owns the failure mode taxonomy? On what cadence is it updated? Who resolves new failure modes identified during RCA?
Currency after plant modifications: When an engineering change notice modifies a system boundary or installs a new component, what is the process for updating the KG before the next event analysis?
Unresolved entity alias review: The NLP pipeline flags unresolved entity aliases (surface forms that could not be mapped to canonical KG nodes). Who reviews these, and on what cadence? Currently: no defined process.

## 6. Candidate Scoring and Confidence Model
### Pre-Evidence Scoring (Stage 3 — V32)
Each candidate is scored across five dimensions before any documents are retrieved:
Known gap — governance uniformity: The governance score is currently computed at the asset level and applied uniformly to all candidates regardless of whether the failed PM item is related to the specific failure mode being scored. This must be replaced with candidate-specific PM linkage in the next iteration.
Known gap — symptom matching: The event.symptom_signature.symptom_types and anomaly_pattern fields, and the corresponding kg_context.failure_modes[].expected_symptoms and expected_anomaly_pattern fields, are not currently used in scoring. A failure mode that expects gradual_drift and whose component shows gradual_drift anomalies should receive a structural bonus. This is unimplemented.
Known gap — weights not normalized: No constraint enforces that scoring weights sum to 1.0. A configuration change to one weight without adjusting others will silently produce an unnormalized composite score.
### Post-Evidence Refinement (Stage 5 — V32)
After evidence retrieval, the evidence score is refined using:
refined_evidence = clamp(
    0.30 * prior_evidence
  + 0.55 * best_support_score
  + 0.15 * best_context_score
  - 0.45 * best_contradiction_score,
  0.0, 1.0
)
The composite score is recomputed using the same weights with the refined evidence score. Candidates are re-sorted and re-screened against thresholds. If only one candidate passes, the best failed candidate is evaluated for retention as a review alternative if its score gap is within review_alternative_gap (default 0.10) and it is not temporally contradicted.
### Confidence Calibration (Stage 7 — Synthesizer)
The final confidence_label on the primary hypothesis is not a simple threshold on the composite score. It is computed by _calibrate_primary_confidence() using:
Supporting evidence count and score
Contradicting evidence count and score
Evidence posture label (supported / mixed / contextual_only / contradicted / weak)
Score gap to runner-up candidate
Temporal posture (supported / partial / contradicted / weak)
Temporal contradiction flag
Recurrence score and confidence
Common-cause score and confidence
Whether the fallback path was used (caps at medium if True)
This multi-dimensional calibration is intentional. A candidate with a high composite score but zero supporting evidence and a temporal contradiction must not be labeled high confidence. The calibration model enforces this constraint.

## 7. Formal Workflow Steps
### Step A — Run Setup and Input Validation
Class: RCAReasoningOrchestrator._stage_a_build_run_context()
Validate the incoming event and telemetry summary against their JSON schemas and semantic rules. Establish the run_id, pipeline configuration, and input validation result. Persist run_context. This stage does not perform causal reasoning.
### Step B — KG Context Construction
Class: Neo4jKGContextBuilder.build()
Resolve seed components from the event asset and telemetry signal identifiers (using four lookup strategies: monitored_variable_id, sensor_id, tag_id, ID, aliases — with confidence ranking and deduplication). Expand the graph neighborhood within max_hops. Fetch failure modes, ranked documents, and ranked past events. The KG context defines the causal search space for all subsequent stages.
### Step C — Temporal Reasoning (TSKR)
Class: TSKRTemporalScorerV1.score()
For each failure mode in the KG context, compute: temporal relation (precedes / simultaneous / follows / unknown), mean lag and std lag against anomaly timestamps, latency alignment score against expected FMEA latency bounds, latency violation type (none / too_fast / too_slow / unknown), temporal contradiction flag, history support score from past event overlap, and combined pattern confidence and support scores. Currently heuristic — the scorer's weakest stage.
Known gap — TSKR index limitation: The TSKR index keeps only the first pattern per target_id. Failure modes with multiple TSKR patterns at different latencies lose all but the first. Fix required: store a list and select highest-confidence pattern, or aggregate.
Known gap — anomaly timestamp weighting: All anomaly timestamps are treated as equal regardless of severity or signal quality. High-severity anomalies should contribute more to lag estimation than low-severity or noisy ones.
### Step D — Candidate Generation (Pre-Evidence)
Class: RuleBasedCausalityEngineV32.generate()
Generate failure mode and historical event candidates. Score each across five dimensions. Enrich with recurrence features (from past_event_index) and common-cause features (from common_cause_index). Apply dual threshold screening (composite ≥ 0.30 AND evidence ≥ 0.35). Retain top-k. Store filtered candidates with filter reason. Compute recurrence_summary and common_cause_summary. Output is causality_candidates v1.
Note: V31 (RuleBasedCausalityEngineV31) is retained for cross-validation and ablation testing alongside V32. Both engines are intentionally kept — V31 provides a structural/evidence-only baseline, V32 adds TSKR Allen-relation scoring and EntityNormalizer. V32 is the production engine.
### Step E — Evidence Retrieval
Class: ChromaEvidenceRetriever.retrieve()
Build hypothesis-guided query plans for each top candidate: one support query and one contradiction query per candidate, plus failure mode context, component context, and operational context queries. Execute each query against the Chroma vector store with asset, doc_type, doc_id, and component filters. Score each retrieved snippet for support, contradiction, and context roles using keyword cue detection, candidate term overlap, authority weight, and extraction quality. Deduplicate and rank. Build per-candidate evidence summary (supporting counts, contradiction counts, best scores, snippet IDs). Output is evidence_bundle.
Known gap — component_ids filter is a no-op: The Chroma metadata structure does not support direct list-valued filter on component_ids. The filter is built but silently ignored. All documents for the asset are searched regardless of component scoping. This reduces retrieval precision for assets with many components.
Known gap — BM25 unavailable on disk-loaded collections: When collections are loaded from disk (not ingested in the same process), state.bm25_docs is empty and hybrid retrieval degrades silently to dense-only. No warning is emitted. Fix required: add explicit BM25 availability flag in retrieval metadata.
Known gap — keyword-based role classification: Support, contradiction, and context role classification uses keyword tokenization against snippet text. Semantically equivalent terms ("loss of lubrication" vs "lube oil degradation") will not match. This is a known precision limitation for the nuclear terminology domain.
Known gap — document type semantics not differentiated: All document types (CR, WO, SOP, ECA, RCA, FMEA) pass through the same keyword classifier. A SOP's conditional statement ("fouling may cause backpressure rise") scores identically to a confirmed WO finding. The structured fields in the document schema (condition_assessment.as_found_condition, failure_mode_refs, extracted_causal_statements) are not used in role classification. This is one of the highest-priority improvements for retrieval quality.
### Step F — Candidate Refinement (Post-Evidence)
Class: RuleBasedCausalityEngineV32.refine_with_evidence()
For each retained candidate, update the evidence score using the refinement formula. Recompute composite score and threshold pass/fail. Derive evidence_posture label. Re-sort and re-screen candidates. Evaluate review alternative rescue if only one candidate passes. Mark evidence_refinement_applied: True in provenance. Output is causality_candidates v2.
This is architecturally the most important stage. The delta between the Stage D ranking and the Stage F ranking is the system's primary discriminating contribution. If the ranking does not change after evidence refinement, evidence retrieval is not providing useful information and should be diagnosed.
### Step G — Ishikawa Structuring
Class: HeuristicIshikawaEvaluatorV1.evaluate()
Translate all prior reasoning into five engineering RCA categories: equipment_hardware, process_procedure, measurement_instrumentation, environment_operating_context, maintenance_human_factors. This stage performs translation, not new inference. Every row carries links to its source artifact, linked candidate IDs, supporting evidence IDs, and telemetry signals. Output is ishikawa_matrix.
Known gap — Ishikawa depth: The evaluator is heuristic. Row strength values are approximations. The process_procedure category is populated from top evidence snippets rather than from actual procedural deviation analysis. Human performance and organizational factor analysis (required by IAEA TECDOC-1756 and INPO AP-923) is not implemented.
### Step H — RCA Synthesis
Class: RuleValidatedRCASynthesizerV31.synthesize()
Select top candidates and top evidence. Attempt LLM structured generation (currently always fails → fallback). Build deterministic RCA card from the ranked candidates and evidence bundle. Derive executive summary, primary hypothesis with why_primary[] and uncertainties[], alternatives with supports/weaknesses/citations, evidence rows with support_role, recommended actions, and analyst review questions. Apply multi-dimensional confidence calibration. Set validation status flags. Output is rca_card.
Current production path: Deterministic fallback (_fallback_card()). The LLM path (OllamaLLMClient) exists in the codebase but is not validated.
Known gap — LLM hallucination gate: If the LLM path is used and produces a hallucinated candidate ID that passes schema validation, the cross-artifact check in the validator will catch it as a warning (not an error). This should be upgraded to a hard error that triggers the fallback path.
### Step I — Artifact Persistence
Class: FileArtifactStore
Write all stage outputs as JSON files under {output_dir}/{run_id}/. Validation reports are persisted alongside each artifact.
### Step J — Validation and Run Manifest
Classes: RCAArtifactValidator.validate_run_bundle(), RCAReasoningOrchestrator._stage_g_finalize_manifest()
Run two-layer validation: JSON schema checks per artifact, and cross-artifact semantic checks (event_id / asset_id consistency across all artifacts, subgraph_id consistency, primary candidate ID exists in candidates, cause_label matches, evidence source IDs exist in bundle, candidate count arithmetic consistency, temporal posture vs contradiction flag consistency). Compute review hooks. Output is run_manifest.
requires_human_review: True always. writeback_ready: False always in current implementation because fallback_used: True.

## 8. RCA Workflow — File-to-Stage Map
### Supporting Components

## 9. Current Implementation Status
### Core RCA Workflow
### Artifacts
### Infrastructure

## 10. Known Gaps and Prioritized Technical Debt
### Priority 1 — Must Fix Before Next Release
### Priority 2 — High Value for Next Iteration
### Priority 3 — Architecture and Maintainability
### Deferred (Not in Scope for Next Iteration)
Human performance analysis (task analysis, error precursor identification, organizational factor assessment)
Precursor detection (weak signal aggregation before event)
NLP/NER parsing pipeline for ProcessedTextRecord population
Fleet-wide recurrence (cross-unit, cross-plant similarity)
PRA / risk significance integration
Effectiveness review tracking

## 11. Acceptance Criteria and Developer Guidance
Reproducibility: Identical inputs must produce identical causality_candidates and validation outcomes. The pipeline is fully deterministic when DummyLLMClient is used.
Grounding: Primary causal claims in the RCA card must reference evidence or be explicitly marked low-confidence/speculative. all_claims_cited must be True before write-back is considered.
Auditability: Every artifact carries provenance (generated_at, generated_by, run_id, pipeline_version). Every KG commit carries analyst sign-off context. The run_manifest is the permanent audit record.
Explainability: The RCA result must be defensible through ranked candidates, temporal evidence, Ishikawa categories, and evidence citations — not only by narrative summary. The score_rationale block on each candidate must be populated.
Schema hygiene: Resolve confidence label case mismatch and kg_context dual schema before enabling strict validation. All schemas must have additionalProperties: false for deterministic validation behavior.
Repository hygiene: Schemas in a single discoverable directory (schemas/). Example fixture bundles synchronized with schemas. V31 causality engine retained for cross-validation and ablation testing alongside V32 (not deprecated — intentional dual-engine design).

## 12. Reference Test Case
Primary validation scenario: PWR Unit 2 — Condenser Vacuum Loss / Turbine Load Runback (EVT-U2-2024-0847)
True root cause: Air in-leakage through turbine exhaust duct expansion joint
Contributing cause: HVAC fan bearing failure → elevated pit ambient temperature → accelerated expansion joint thermal fatigue
Red herring: Condenser waterbox tube cleaning 21 days prior (found acceptable)
Recurrence trap: Most recent similar event (18 months prior) had fouling as confirmed root cause — recurrence scorer must not over-weight recency
Key discriminating signal: Hotwell dissolved oxygen = 142 ppb (normal < 10 ppb). DO elevation is caused by air in-leakage and is NOT produced by tube fouling. The condenser tube inspection WO explicitly states "within normal limits, zero tubes plugged" — this is the primary contradicting evidence for the fouling hypothesis.
### Assertion Targets

## 13. Next Iteration Priorities
### Must Do
Resolve confidence_label case mismatch (uppercase vs lowercase) across schema files
Resolve kg_context dual schema — pick v2 as canonical, update KGContextBuilder output
Add sum-to-1.0 constraint on scoring_config.weights in schema and validator
[FIXED] Fix TSKR index to store all patterns per target_id, select highest-confidence
Make LLM hallucinated candidate ID a hard validation error (not warning)
Refactor rca_reasoning_orchestrator.py — split KGContextBuilder, IshikawaEvaluator, factory function into separate modules
### High Value
[FIXED] Implement candidate-specific governance scoring — link failed PM items to specific failure mode candidates by component and check_type
[FIXED] Add symptom-to-failure-mode matching using symptom_types and anomaly_pattern fields
[FIXED] Fix component_ids filter in Chroma — or document explicitly as no-op with warning
[FIXED] Add BM25 availability flag to retrieval metadata with explicit warning on degradation
[FIXED] Differentiate document type role classification — use condition_assessment.as_found_condition for WO classification; treat SOP diagnostic rules as discriminating logic rather than supporting evidence
Add time_distance_days weighting to evidence prior scoring in the causality engine
Fix _passes_minimum_evidence_gate string comparison to use .strip() and explicit casting
Add analyst_override schema and ingestion pathway
### Defer
NLP/NER parsing pipeline
Human performance analysis tools
Precursor detection
PRA integration
CAP integration (Maximo/SAP PM)
Production deployment concerns

## 14. Notes on TSKR
Nuclear plant telemetry is exactly the kind of noisy, multi-variate, interval-based time series that TSKR was designed for. The hierarchical Tone → Chord → Phrase structure maps naturally onto plant failure propagation:
Tones → discretized parameter states ("lube oil pressure low," "vibration elevated," "temperature rising")
Chords → coincident degradation signatures (multiple parameters anomalous simultaneously)
Phrases → temporal sequences of failure propagating through a system
Use TSKR for: justifying the representation choice over Allen's relations; the Tone discretization approach for converting continuous plant parameters into symbolic interval states; the margin-closedness concept for pruning temporal pattern results.
Don't rely on TSKR for: full mining pipeline implementation, lag/confidence modeling, or event-relative temporal scoring — these are gaps that require additional domain engineering work.
The most important next step for making TSKR useful is defining the Tone discretization vocabulary for nuclear plant parameters — what thresholds and durations make a "lube oil low" Tone versus a transient fluctuation. This is domain engineering work that no paper will provide, and it is the foundation everything else rests on. The FMEA expected_latency_min_hours and expected_latency_max_hours fields in the KG are the starting point for this vocabulary.

## 15. Open Issues Tracker

# Gaps from Requirements document
Gap 1 — kg_snapshot_version never populated (§6 Model Governance) [Fixed]
The schema declares it as "Required for reproducibility — RCA must be replayable against the KG state that existed at event time." Neo4jKGContextBuilder.build() never queries Neo4j for a snapshot version or timestamp and never sets the field. Every run produces a kg_context that silently omits it, making the output non-reproducible against KG changes — which is a regulatory defensibility problem.
Fix: Query CALL dbms.components() YIELD versions or CALL apoc.meta.stats() at build time and store the result, or at minimum stamp generated_at as the snapshot version.

Gap 2 — recent_alarms is ingested but never used in causal scoring (§3.3) [Fixed]
operational_context.recent_alarms carries alarm_id, priority, system_affected, setpoint, actual_value. The causal engine has zero alarm handling — no field is read, no score is adjusted. A critical alarm on the same component as a candidate is strong corroborating structural signal; a low-priority alarm on a different system is irrelevant noise. The engine currently treats all alarm states identically: ignored.
Fix: In _structural_score_for_fm, extract alarms from operational_context and boost structural score when a critical/high alarm's system_affected matches the candidate's component or its upstream.

Gap 3 — Defense-in-depth barrier analysis completely absent (§3.2, §3.12)
The document makes barrier analysis a first-class requirement: "which barriers failed, which succeeded." This is distinct from safety functions — barriers are the physical/functional layers (fuel cladding, RCS pressure boundary, containment) that prevent radioactive release. Safety function mapping (Issue #8 we just implemented) is one step, but there is no barriers concept in:
kg_context.json schema
Neo4j KG context builder
Candidate scoring
rca_card recommended actions
The document also calls for "defense-in-depth barrier identification" and "visualization of degraded defense layers" explicitly in §3.2 and "identification of barrier weaknesses" in §3.12. This is a structural gap in the architecture — the KG schema doesn't model barriers at all.

Gap 4 — No risk significance score derived from safety function impact (§3.12)
affected_safety_functions on each candidate tells you which safety functions could be affected, but there's no risk_significance scalar derived from it. A candidate affecting "Core Cooling" is orders of magnitude more safety-significant than one affecting "Instrument Air." The document requires "qualitative risk significance scoring" as part of PRA integration. Currently the field is populated but never used to adjust confidence or flag candidates for priority review.

Gap 5 — mdParser.py is missing doc types and section roles (§3.1, §3.7)
FIELD_LABEL_MAP covers CR, WO, SOP, ECA, OTHER — but:
No FMEA section role map — the parser has no concept of FMEA-specific sections (failure mechanism, detection method, mitigation). FMEA chunks are parsed as OTHER, losing structured signal. [Fixed]
No INPO OE / fleet-level documents — the document calls for "industry OE reports" and "operating experience repositories" as a primary data source (§3.7). There is no "OE" or "INPO" entry in FIELD_LABEL_MAP. [Fixed]
No protection logic extraction — no section roles for permissive/interlock/trip references in any doc type, which §3.2 requires.
No alarm reference extraction — alarm IDs mentioned in CR narratives or operator logs are not tagged. [should be fixed now]

Gap 6 — governance score rationale is non-traceable (§3.8, §7) [Fixed]
The score_rationale["governance"] on every candidate says "Governance score derived from PM compliance signals." — a generic string that never identifies which PM checks matched, what keywords triggered them, or what the check result was. The document requires "audit trail from raw data to final conclusion" and "evidence traceability for every inference." The current rationale doesn't satisfy this for governance.

Gap 7 — Change analysis is unstructured (§3.8)
The document requires a "change analysis workflow" as a named RCA methodology. operational_context.nearby_maintenance is used in CCF scoring, but there is no structured change analysis that systematically correlates:
Maintenance completed in the window
Setpoint/procedure revisions
Configuration changes (ECNs)
...against the event timeline. Currently maintenance proximity influences common_cause scoring heuristically, but it's not surfaced as a named "change analysis" artifact in the output the engineer reviews.

Gap 8 — Sensitivity analysis missing (§3.9)
All candidate scores are point estimates. The document explicitly requires "sensitivity analysis under data-quality assumptions." If extraction_quality on a key piece of evidence went from 0.6 to 0.9, how much does the composite score change? Which candidates are score-stable vs. fragile? This is never computed or surfaced. The evidence_gap flag (implemented this session) covers the zero-evidence case, but not the quality-sensitivity case.

Gap 9 — No FMEA doc type in mdParser.py and no FMEA-specific structured output feeding the document schema [FMEA parser added]
FMEAs carry failure_mechanism, rpn, expected_latency — all fields that the kg_context.json failure_modes schema defines. But the parser has no FMEA section map, so FMEA documents are chunked as generic text. The structured fields in failure_modes in kg_context must currently be manually populated in the KG rather than being extracted by the parser.

Gap 10 — Visualize timing of events and RCA retrieved data
There must a way to visualize present information retrieved and generated by the RCA workflow. This should be interactive

# On OE documents
OE reports (INPO OE, NRC Information Notices, Generic Letters, EPRI Technical Reports, fleet-wide SOER/SOER) are fundamentally different from plant-internal documents in three ways:
Scope is fleet-wide, not asset-specific. A CR is tied to a specific equipment tag at a specific plant. An INPO OE report says "this failure mode has occurred at 7 plants on this component family." There is no asset_id to resolve, no component_id — the linkage is through failure mode similarity and system type, not plant topology.
Authority and epistemic role are distinct. OE is guidance/informational — it can't confirm your specific event's cause. Its value is plausibility amplification: "this has happened before at similar plants." That's different from an ECA which confirms a cause at this plant.
Temporal semantics are different. OE from 2005 about valve stiction degradation is still fully relevant today — it's closer to FMEA (timeless engineering knowledge) than to a CR (time-bound operational observation). Recency decay should not apply.

# On FMEA documents
FMEA is class-level, not instance-level. This has important consequences:
A single FMEA row for "centrifugal pump / seal degradation" applies to every centrifugal pump in the plant
The FMEA says nothing about which specific pump P-101A will fail — only that pumps of that type can fail that way
Individual pump P-101A is a KG element_usage node that realizes the component_type = centrifugal_pump

The mapping chain is therefore:

FMEA row  →  failure_mode node  →  component_type (class)
             			                            ↑
             			                 KG realizes edge
             			                            ↑
             		                 individual element_usage node (P-101A, P-101B, …)

This means the FMEA parser cannot write applies_to_component_id = P-101A directly — it should write applies_to_component_type = centrifugal_pump and the KG loader resolves which specific element_usage nodes are instances of that type at query time.

There are two common exceptions worth noting:
Some plants maintain component-specific FMEA addenda for high-safety-significance items (unique service conditions, known material susceptibility). These override the class-level RPN with a component-specific one.
The Maintenance Rule (10 CFR 50.65 in nuclear) assigns functional failure criteria at the functional group level, not the component type level. This maps to safety functions, not failure modes directly.

# On NER 

# Unit test suite for RCA workflow
test_evidence_scorer.py — _assess_hit_against_candidate (the most bug-prone function in the pipeline. Tests each scoring pathway in isolation)

test_evidence_dedup.py — _dedupe_and_rank

test_evidence_summary.py — _build_candidate_evidence_summary

test_causality_scoring.py — shared scoring helpers (v31 + v32)

test_causality_engine_generate.py — generate() stage (v31 + v32)

test_refine_with_evidence.py — v32 refine_with_evidence() only

test_synthesizer_validation.py — _validate_card_semantics

test_synthesizer_fallback.py — _fallback_card evidence scoping (Fix for A10)

test_query_builder.py — _build_queries

## Stage-boundary integration tests
test_stage_de_contract.py — causality engine → evidence retriever handoff
test_stage_ef_contract.py — evidence retriever → refine_with_evidence handoff

Approximate count: ~55 tests across 9 files. Each test is < 30 lines — a minimal fixture dict, one call, one or two asserts.
Priority order to build them (highest ROI first):
test_evidence_scorer.py + test_evidence_dedup.py — these guard the two bugs we just fixed
test_refine_with_evidence.py — the v32-specific threshold logic
test_causality_engine_generate.py — FM/analog separation (Fix 2)
test_synthesizer_validation.py + test_synthesizer_fallback.py — A10 guard
test_causality_scoring.py — scoring arithmetic
test_query_builder.py + stage contracts — retrieval contract
