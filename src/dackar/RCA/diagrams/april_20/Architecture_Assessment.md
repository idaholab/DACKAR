# Architecture Assessment
**Status pass**: April 23, 2026 — inline `[FIXED]`, `[PARTIAL]`, `[OPEN]` annotations added against v32 implementation.
Pipeline structure: 
KG narrowing → temporal scoring → candidate generation → evidence retrieval → evidence refinement → synthesis. 
## Critical Issues
**[FIXED — v32]** ~~The TSKR index only keeps the first pattern per target_id.~~ `_index_tskr_patterns()` in v32 uses `index.setdefault(target_id, []).append(p)` — all patterns per target_id are accumulated. `_lookup_tskr_pattern()` returns the highest-confidence one. The multi-pattern case is handled correctly.

**[FIXED — v32]** ~~The governance score rewards PM failures, not penalizes them. A bearing wear candidate and an instrument drift candidate both get the same governance score boost if any PM check failed anywhere on the asset.~~ `_governance_details()` in v32 applies 3-tier candidate-specific matching: (1) structural — `check.component_id == component_id`; (2) FM-level — `fm_id in check.applicable_fm_ids`; (3) keyword fallback against `fm_name` + component name only (not all-text). Candidates on the same asset but unrelated to the failed PM item receive no governance boost.
**[PARTIAL]** Evidence role assignment and lexical overlap in `_assess_hit_against_candidate`.

*Fixed (Apr 23, 2026)*: Query-intent circular labeling removed. A `+0.15` boost keyed on `query_type` ("candidate" → support, "candidate_contradiction" → contradiction) caused deduplication winner-takes-all to determine the role rather than snippet content. Neutral snippets in the 0.15–0.30 content-score gap were systematically mislabeled "supporting" when the support query ranked them higher. The boost block has been removed; role classification now relies solely on content signals: semantic_relevance (3-tier: Chroma vector score → encoder cosine similarity → lexical fallback), support/contradiction cue detection, CA structured fields, spaCy hedge discount, and causal attribution detection.

*Still open*: The `candidate_term_overlap` lexical fallback (last-resort path when no vector score and no encoder is configured) still matches only the `cause_label` string against snippet tokens — "loss of lubrication" will not match "lube oil degradation." This path is bypassed when a Chroma vector score or an encoder is available, so it does not affect production runs with a configured embedder. It remains a gap for zero-dependency / offline deployments and for any snippet where the vector score is absent (e.g., BM25-only hits from disk-loaded collections).

**[OPEN]** The `_evidence_score_for_fm` and `_evidence_score_for_past_event` methods don't fully account for document age in the evidence PRIOR (pre-retrieval). Note: `_recency_factor()` IS already called by both methods and uses `time_distance_days` to weight retrieved evidence — the remaining gap is that the pre-retrieval evidence prior (Stage D) does not use document age from `kg_context.documents`. A 10-year-old SOP contributes the same as a CR filed last week. The `kg_context.json` schema has `time_distance_days` on documents — that field should feed into the evidence prior score in the causality engine.

### Causality Engine — Specific Gaps vs. Engineer Needs
**[FIXED — v32]** ~~Symptom-to-failure-mode matching is absent.~~ `_symptom_match_score()` in v32 (lines 1194–1271) matches `anomaly_pattern` against `expected_anomaly_pattern` (60% weight) and `symptom_types` against `expected_symptom_types` / `expected_symptoms` (40% weight). The resulting delta is applied in `_evaluate_candidates()` as `symptom_delta = 0.40 * (symptom_score - 0.5)` — a high-severity step-change event now scores differently from a gradual-drift event against the same candidate.

**[FIXED — v32]** ~~Common cause scoring doesn't use train configuration.~~ `_common_cause_features_for_candidate()` in v32 reads `operational_context.train_configuration.in_service`; an OOS train contributes a CCF cluster signal (score 1.0) while unknown in_service contributes 0.0 (no speculation). The clearest loss-of-redundancy signal is now incorporated.

**[FIXED — v32]** ~~Recurrence analysis doesn't distinguish resolved from unresolved past events.~~ `_recurrence_features_for_candidate()` in v32 (lines 2068–2112) computes `unresolved_fm_count` and `unresolved_component_count` from past events where `resolved == False`. Events with `resolved == None` (unknown) are excluded from the unresolved count rather than being assumed resolved.

**[FIXED — v32]** ~~The temporal scorer treats all anomaly timestamps as equal.~~ `TSKRTemporalScorerV1` in v32 adds `_severity_weight()` (lines 446–459) which combines the `severity_score` field with `_INSTRUMENT_VALIDITY_MULTIPLIER` (calibrated/invalid/failed flags) to produce a per-window weight in [0.1, 1.0]. `_effective_anomaly_count()` is a severity-weighted sum, not a raw count. High-severity step-change anomalies now contribute more than low-severity noise spikes.

## Retrieval Layer Issues
**[OPEN]** `ProcessedRecordStore` is fully in-memory with no persistence. If the process restarts, all hydration data is gone. For a tool that may need to run multiple analyses in sequence or be interrupted during an investigation, this means re-loading JSONL files each time. This is probably acceptable for a dev deployment but needs to be noted as a production gap.

**[FIXED — v32]** ~~`_normalize_filter_meta` silently drops `component_ids`.~~ `query_doc_type()` in `chroma_store.py` now translates `component_ids` to a `primary_component_id: {$in: [...]}` Chroma where-clause filter. A `_doc_matches_component_ids()` post-filter handles legacy records that predate the `primary_component_id` field. Component-level filtering is now functional.

**[FIXED]** ~~BM25 silent degradation on disk-loaded collections.~~ `chroma_store.py:727` stamps `_bm25_available` on every returned document; `load_collection()` warns at collection-load time; `query_doc_type()` warns at query time with the `hybrid_weight` value. `evidence_retriever.py` reads `_bm25_available` per hit, derives `retrieval_mode` (`dense_only / hybrid / unknown`), propagates `retrieval_mode` + `retrieval_quality_warning` into the evidence bundle metadata, and sets pipeline health to yellow when dense-only.

## Schema Builder and KG Ingest
**[OPEN]** `kg_schema_builder_workflow.py` uses ad-hoc label resolution that will silently create wrong-typed nodes. `_label_candidates` generates Pascal-case variants from snake_case names, which works for `processed_text_record → ProcessedTextRecord` but will fail silently for any label that doesn't follow this exact convention. If a TOML schema uses `AbnormalEvent` and the code generates `Abnormalevent`, it creates a new unlabeled node type rather than erroring. The label resolution needs an explicit registry lookup with a hard failure on miss.

**[OPEN]** The `causes` edge in the graph builder is overloaded across at least six semantically distinct relationships. `Anomaly → Event`, `PMCheck → Event`, `HistoricalEvent → CurrentEvent`, and `Alarm → Event` all use `"causes"` as the relationship type. This makes the KG unusable for path-strength analysis — when you traverse `CAUSES` edges you can't distinguish "this alarm occurred simultaneously with the event" from "this failure mode was confirmed as the root cause." Each of these needs a distinct edge type.

**[OPEN]** `_prefix` produces IDs like `FM:loss of lubrication` if the source key contains spaces. Neo4j node IDs with spaces are valid but will cause problems in Cypher when used unquoted. The prefix function should normalize the value portion: lowercase, replace spaces with underscores or hashes. The `_norm_text_key` helper exists in the same file but isn't used in `_prefix`.

## Orchestrator Structure
**[FIXED]** ~~`rca_reasoning_orchestrator.py` is doing too much.~~ The orchestrator has been split into 12 separate modules in `orchestrators/`: `causality_engine_v32.py`, `causality_engine_v31.py`, `ishikawa_evaluator.py`, `evidence_retriever.py`, `kg_context_builder.py`, `tskr_temporal_scorer.py`, `temporal_relations.py`, `signal_evidence_builder.py`, `artifact_store.py`, `llm_clients.py`, `input_guards.py`. The main orchestrator now delegates to these specialized sub-modules.

**[OPEN]** `build_dev_orchestrator` exposes `stop_on_validation_error` but the orchestrator will raise on the first validation failure regardless of whether the error is in an optional artifact. If `ishikawa_matrix` fails schema validation (it's newly added and marked heuristic), the entire run aborts even though the RCA card could still be synthesized from the other artifacts. Validation failures on optional stages should be logged and surfaced in the manifest but should not abort the run unless the failed artifact is required by a downstream stage.

## Document-type-aware treatment 
The current implementation treats document types differently in some ways but not nearly as differently as their fundamentally different nature requires.
What each document type actually communicates — and what the code misses
### Condition Reports (CRs)
What they are: An engineer's observation of an abnormal condition at a specific time. They contain a symptom description, an initial assessment, and sometimes a preliminary cause statement.
What they uniquely provide: Temporal grounding. A CR has a created_at timestamp and a reported_date. The cause statement in a CR is an initial hypothesis, not a confirmed finding. The CR filed on Day 9 of the precursor period in the test case ("possible fouling or CW temperature effect") was written before the DO data showed air in-leakage — it reflects the state of knowledge at that moment, not the confirmed root cause.
What the code misses: It doesn't distinguish between a CR's preliminary cause statement and a confirmed cause statement. Both score the same. It also doesn't use the CR timestamp to assess whether the document predates the discriminating evidence — a CR written before the hotwell DO elevated should carry less causal weight than one written after.
### Work Orders (WOs)
What they are: Records of maintenance activities with as-found and as-left condition assessments. They contain measurements, observations, and often explicit "within normal limits" or "outside acceptance criteria" language.
What they uniquely provide: The closest thing to ground truth about equipment condition at a specific point in time. WO-2024-11847 saying "zero tubes plugged, within normal limits" is a direct physical inspection result — it's qualitatively different from a CR's operator observation or a SOP's conditional guidance.
What the code partially handles: The contradiction cue detection does pick up "within normal limits" and "acceptable" from WOs, which is why WO-2024-11847 correctly scores as contradicting for the fouling hypothesis in the test case. But the code doesn't know this is a physical inspection result versus a textual description — it's purely the keyword match that does the work.
What the code misses: **[FIXED]** The `condition_assessment` block — with explicit `as_found_condition` and `as_left_condition` enumerations (`acceptable/degraded/failed/unknown`) — is now used in Stage 4. `evidence_retriever.py:485–513` reads both fields and applies ±0.35 score adjustments to support/contradiction scores. The structured condition data is richer and more reliable than keyword detection in free text.
### SOPs (Standard Operating Procedures)
What they are: Prescriptive documents describing how things should work and what actions operators should take. They contain diagnostic logic, acceptance criteria, and step-by-step guidance.
What they uniquely provide: Engineering knowledge about the expected behavior of systems and the diagnostic rules that distinguish failure modes. SOP-U2-CND-001 Step 4.2 in the test case says "DO above 20 ppb is indicative of air in-leakage — tube fouling does not cause dissolved oxygen elevation." That is an expert diagnostic rule, not an observation.
What the code misses: The system treats this diagnostic rule the same as any other supporting snippet. It scores as contextual or supporting for air in-leakage because it contains "air in-leakage" near positive language. But its real value is as a discriminating rule — it explicitly says fouling does NOT produce DO elevation. A document-type-aware system would recognize SOP diagnostic rules as having special standing: they are the engineering basis for distinguishing between hypotheses, not just additional evidence for one of them.
SOPs also carry a temporal peculiarity — they describe steady-state expected behavior, not event-specific observations. The code applies the same temporal relevance logic to SOPs as to CRs, which is conceptually wrong.
### ECAs (Engineering Cause Analyses)
What they are: Formal engineering documents that present a structured causal argument, often with an explicitly stated confirmed root cause, contributing factors, evidence items, and recommended corrective actions.
What they uniquely provide: The closest thing to a confirmed causal conclusion in the document corpus. An ECA is more epistemically reliable than a CR (which is preliminary) and more specific than an FMEA (which is generic). ECA-2022-1103 in the test case explicitly states "root cause confirmed as tube fouling" with "DO was within normal limits during that event at 7.2 ppb" — that last detail is the key discriminating fact that distinguishes the prior event from the current one.
What the code misses: **[PARTIAL]** ECA/RCA now receive `+0.22×rf` epistemic weight vs CR/WO at `+0.15×rf` in `_evidence_score_for_fm` and `_evidence_score_for_past_event`. The scoring improvement comes from `doc_type` weighting only. The ECA's `causal_factors[]` and `evidence_items[]` structured arrays are still not parsed — these fields contain machine-readable causal knowledge the system could use directly but the retriever works only on embedded text snippets.
### RCAs (Root Cause Analysis reports)
What they are: Comprehensive post-event analyses, similar to ECAs but typically more extensive, including event timelines, causal factor charts, barrier analyses, and effectiveness reviews.
What they uniquely provide: Confirmed causal chains with explicit evidence linkage. They also often contain the negative findings — what was ruled out and why — which are directly useful for hypothesis discrimination.
What the code misses: **[PARTIAL]** RCA `doc_type` now receives higher epistemic weight (`+0.22×rf`). The confirmed causal status and ruling-out logic (what was eliminated and why) are still not structurally parsed. The ruling-out logic is particularly valuable and completely invisible to the current keyword-based classifier.
### FMEAs (Failure Mode and Effects Analyses)
What they are: Engineering design documents that enumerate possible failure modes for components, their effects, detection methods, and risk priority numbers.
What they uniquely provide: They define the possibility space — what failures are physically conceivable for each component. They're not observations of what happened; they're engineering predictions of what could happen.
What the code misses: The FMEA is already partially absorbed into the KG through the failure_modes nodes — so its structural content is already influencing Stage 3. When a FMEA snippet is retrieved in Stage 4, it's effectively double-counting information already present in the KG, but the system doesn't know this. A FMEA saying "air in-leakage is a known failure mode for expansion joints" scores as supporting evidence for the air in-leakage hypothesis, when it really just confirms that the hypothesis is physically plausible — something Stage 3 already established.
The right architecture would recognize that each document type occupies a different position on two axes:
Observation vs. prescription: CRs and WOs describe what was observed. SOPs and FMEAs describe what should or could happen. ECAs and RCAs describe what was confirmed.
Temporal specificity: CRs, WOs, and ECAs are event-specific. SOPs, FMEAs, and manuals are time-independent engineering knowledge.
A properly differentiated Stage 4 would use these distinctions to:
Treat confirmed findings in ECAs and RCAs as higher-confidence evidence than preliminary observations in CRs **[PARTIAL — doc_type weighting implemented; structured `causal_factors` parsing still missing]**
Use SOP diagnostic rules as discriminating logic rather than supporting evidence for individual hypotheses **[OPEN]**
Use FMEA content only to confirm physical plausibility, not as independent supporting evidence (since that's already captured in Stage 3) **[OPEN]**
Weight WO condition assessments by their structured `as_found_condition` and `as_left_condition` fields rather than keyword detection **[FIXED — evidence_retriever.py:485–513]**
Apply temporal discounting to CR preliminary assessments when later ECAs or RCAs have superseded them **[OPEN]**

## What's Working Well
The refine_with_evidence method in v32 is architecturally correct — using retrieved evidence to update the prior evidence score and rerank candidates is exactly the right two-pass design. The _eligible_review_alternative logic that rescues near-threshold candidates when only one candidate passes is also good; it prevents the pathological case where a low evidence threshold eliminates all alternatives and the analyst gets a single candidate with no context.
The TSKRTemporalScorerV1 latency alignment model is more principled than most heuristic temporal scorers. The separation of latency_alignment_score from temporal_contradiction is correct — something can be partially misaligned without being contradicted, and the synthesizer's confidence calibration handles this distinction properly.
The py2neo_workflow.py wrapper with its batch upsert design and safe token validation is solid infrastructure. The label-grouping approach in upsert_nodes_batch and upsert_edges_batch is efficient and correct.

## Priority Recommendations Before Next Iteration
Ranked by impact on RCA quality (updated April 23, 2026 against v32):

✅ ~~Fix governance scoring — make it candidate-specific, not asset-wide~~ — Fixed in v32 (`_governance_details()` 3-tier matching)
✅ ~~Add symptom-to-failure-mode structural matching using `symptom_types` and `anomaly_pattern`~~ — Fixed in v32 (`_symptom_match_score()`)
✅ ~~Fix the TSKR index to handle multiple patterns per target~~ — Fixed in v32 (`_index_tskr_patterns()` accumulates all patterns)
✅ ~~Make `component_ids` filtering functional in Chroma~~ — Fixed in v32 (`query_doc_type()` translates to `primary_component_id` filter)

**Remaining open items ranked by impact:**
1. Fix the `causes` edge overloading in the KG builder — six semantically distinct relationships all use the same edge type, breaking path-strength analysis
2. Incorporate `time_distance_days` into evidence PRIOR scoring (Stage D pre-retrieval) — `_recency_factor()` already handles post-retrieval weighting; the pre-retrieval prior gap remains
3. ✅ ~~Query-intent circular labeling in `_assess_hit_against_candidate`~~ — removed `query_type` score boost; role now content-only. *Remaining gap*: `candidate_term_overlap` lexical fallback still misses terminological variants ("lube oil degradation" ≠ "loss of lubrication") on offline/no-encoder deployments
4. Parse ECA/RCA structured arrays (`causal_factors[]`, `evidence_items[]`) — doc_type weighting is in place but structured content is still ignored
5. ✅ ~~Fix BM25 silent degradation~~ — already fixed; `_bm25_available` flag stamped on documents, `retrieval_mode` propagated to evidence bundle, pipeline health → yellow when dense-only
6. Fix `stop_on_validation_error` — optional artifact validation failures should log and continue, not abort the run
7. Fix KG label resolution — replace ad-hoc `_label_candidates` with explicit registry lookup and hard failure on miss
8. Normalize `_prefix` value portion — spaces in FM IDs cause unquoted Cypher issues
