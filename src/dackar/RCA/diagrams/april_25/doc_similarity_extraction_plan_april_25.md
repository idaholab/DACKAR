# Document Similarity Search via NER-Based Extraction — Design and Development Plan

**Date:** 2026-04-28
**Context:** RCA pipeline — Step 3 (recurrence matching) and Step 2d (similar event search)
**Status:** Phases 1–3 implemented; Phase 4 (calibration) planned

---

## 1. Problem Statement

Step 3 (documentary pattern recognition) and Step 2d (similar event identification) both rely on matching past plant records — CRs, WOs, and prior RCA findings — against the current event's failure modes. The current implementation matches on structured fields: `component_id`, `fm_id`, `event_type`, and `actuation_type`.

This approach has a fundamental limitation: **most CRs and WOs do not carry a structured `fm_id`**. Records are written in free text by operators and maintenance staff using inconsistent terminology, abbreviations, and varying levels of detail. Formal failure mode IDs are only present when a prior RCA explicitly closed the record against the FMEA taxonomy — which is the exception, not the rule.

Consequences:
- `recurrence_count` in Step 3 understates true recurrence because records describing the same failure mode in different words are not matched
- `novel_pattern` flags fire on failure modes that have prior occurrences, simply because those occurrences were never formally tagged
- Step 2d plant-tier similar event scoring is similarly limited to records with clean structured fields

The goal is to enable **semantic matching** on what records actually say — identified effect, assessed cause, inferred failure mode — rather than on exact ID fields.

---

## 2. Proposed Approach

### 2.1 Two-stage pipeline

```
Stage 1 — Structured extraction (at ingestion time)
    CR / WO / RCA document
        → HybridNERPipeline + causal_condition_adapter  (primary)
        → LLM fallback for low-confidence or absent causal language only
        → one extraction record per identified causal chain in the document
            { identified_effect, assessed_cause, inferred_fm_label,
              fm_id_candidate, confidence, cause_is_symptom,
              extraction_version, embedding_model_version }
        → stored as metadata in Chroma collection "doc_extractions"

Stage 2 — Embedding similarity search (at RCA run time)
    current event failure mode description
        → embed query (identified_effect + assessed_cause + inferred_fm_label)
        → cosine similarity against pre-computed embeddings of past extraction records
        → ranked list of semantically similar past records
        → feeds Step 3 recurrence pool and Step 2d similar event list
```

### 2.2 Why pre-ingestion, not in-pipeline

Extraction runs **once per document at ingestion time**, not during each RCA run. This keeps the pipeline deterministic and avoids latency during execution. Extracted fields are versioned metadata stored alongside the raw record — the pipeline consumes them as structured inputs, not as live outputs.

### 2.3 Document scope boundary

Extraction applies to the following document types only:

| Type | Included | Rationale |
|---|---|---|
| `CR` — Condition Report | Yes | Primary source of plant recurrence history |
| `WO` — Work Order | Yes | Contains as-found condition and cause assessments |
| `RCA` — Root Cause Analysis report | Yes | Highest-quality cause assignments; `fm_id` may already be present |
| `ECA` — Engineering Condition Assessment | Yes | Contains failure mechanism analysis |
| `FMEA` | No | Defines the failure mode taxonomy itself — used as reference context, not as a past event |
| `SOP`, `MANUAL`, `BULLETIN`, `OE` | No | Describe procedures and industry experience, not plant-specific past events; recurrence is a plant-record concept |

Extending extraction to OE documents is a future option (fleet-level recurrence) but is out of scope for this implementation.

---

## 3. Extraction Step (Stage 1)

### 3.1 Extraction granularity — one record per causal chain

A single CR or WO may describe multiple failure modes, effects, and cause-effect pairs. Producing one extraction record per document would silently lose this information and create coarse embeddings that match poorly against specific failure mode queries.

**Decision: produce one extraction record per identified causal chain** within a document. A causal chain is a linked (cause, effect) pair extracted by `causal_condition_adapter`. A document with two independent cause-effect descriptions produces two extraction records, both referencing the same `doc_id`.

Consequences:
- The `"doc_extractions"` Chroma collection stores multiple records per source document
- Deduplication at query time uses `doc_id` to avoid double-counting the same document as multiple recurrence hits
- A document with no extractable causal chain produces one extraction record with `identified_effect` and `assessed_cause` set to null and `confidence = low`

### 3.2 Target fields

For each extracted causal chain the extractor produces:

| Field | Description | Example |
|---|---|---|
| `doc_id` | Source document identifier | `"CR-2026-00123"` |
| `chain_index` | Index of this causal chain within the document (0-based) | `0` |
| `identified_effect` | Observable symptom or functional impact reported | `"pump outlet pressure dropped below setpoint"` |
| `assessed_cause` | Cause as documented in the record — may be a symptom; never inferred beyond what is written | `"bearing degradation due to loss of lubrication"` |
| `inferred_fm_label` | Best-match failure mode label from the FMEA taxonomy, or free text if no match | `"bearing wear — lubrication starvation"` |
| `fm_id_candidate` | Candidate `fm_id` from the equipment model if resolvable at ingestion time; null otherwise | `"FM-PUMP-BRG-002"` |
| `confidence` | Extraction confidence: `high` / `medium` / `low` | `"medium"` |
| `cause_is_symptom` | True if `assessed_cause` describes an observable effect rather than a root mechanism | `true` |
| `as_found` | Component health state at inspection time | `"degraded"` |
| `as_left` | Component health state after corrective action | `"acceptable"` |
| `procedural_deviation_score` | Heuristic score (0–1) for procedural deviation language; stored as metadata for Category G filtering, not used in similarity scoring | `0.0` |
| `extraction_version` | Combined version string: NER pipeline version + gazetteer version + LLM model+prompt version (if fallback used) | `"ner-v1.2_gaz-v3_llm-none"` |
| `embedding_model_version` | Name and version of the embedding model used to compute the stored vector | `"nomic-embed-text-v1.5"` |

### 3.3 Extraction path

**Step A — Entity extraction via `HybridNERPipeline`**

Run the hybrid NER pipeline on the document text. The following entity groups are relevant:

| NER group | Maps to extraction field | Notes |
|---|---|---|
| `G5_FAILURE_OUTCOME` | `identified_effect` | Observable failure states: "failed to start", "trip occurred", "leak detected" |
| `G4_MECHANISM_PROCESS` | `assessed_cause` / `inferred_fm_label` | Degradation mechanisms: "bearing wear", "corrosion", "cavitation", "fatigue" |
| `G1_PHYSICAL_COMPONENT` | Component context for `fm_id_candidate` resolution | Identifies the affected hardware |
| `G8_PROPERTIES` | Supporting context for effect characterization | Temperature, vibration, pressure readings |

`cause_is_symptom` is derived from entity group membership: if the extracted cause entity belongs to `G5` (observable outcome) rather than `G4` (degradation mechanism), the record is describing a symptom, not a mechanism.

**Step B — Causal relation extraction via `causal_condition_adapter`**

Run `extract_stage5_causal_condition()` on the same text. This produces:
- Linked (cause, effect) pairs via `CausalSentence` dependency-tree parsing, with `CausalSimple` as fallback — these populate `assessed_cause` and `identified_effect` as a pair rather than as independent entities
- `as_found` / `as_left` health state classification — stored as metadata for filtering, not embedded
- `procedural_deviation_score` — stored as metadata for Category G filtering only; not included in the similarity embedding

**Step C — `fm_id_candidate` resolution**

Attempt to map `inferred_fm_label` to a specific `fm_id` from the equipment model. This resolution is **deferred to the first RCA run**, not performed at ingestion time, to avoid adding a KG dependency to the ingestion pipeline.

Resolution mechanism: at the start of Step 3 / Step 2d, the orchestrator performs a one-time batch lookup — embedding similarity of each unique `inferred_fm_label` in the `"doc_extractions"` collection against the KG failure mode list for the current event's asset neighborhood. Resolved `fm_id_candidate` values are written back to the collection as a metadata update. Resolution threshold: cosine similarity ≥ 0.80; below this, `fm_id_candidate` remains null. When multiple FMs score above threshold, the highest-scoring match is selected and the runner-up is stored as `fm_id_candidate_alt` for transparency.

**Step D — LLM fallback**

Invoke the LLM only when Steps A and B both fail to produce any extraction (no causal pairs, no G4/G5 entities found). The LLM is used strictly for **extraction** — it must identify cause and effect phrases that are present in the text but not captured by rule-based methods. It must **not** infer mechanisms that are not stated or implied in the document. Using the LLM to deepen a symptom-only record into a mechanism-level cause is explicitly prohibited — `cause_is_symptom` must remain `True` and `assessed_cause` must reflect what is written, not what the LLM infers.

The `LLMDisambiguator` in the hybrid NER pipeline and the LLM fallback in `causal_condition_adapter` provide the implementation base for this path.

### 3.4 Confidence assignment

| Condition | Confidence |
|---|---|
| Gazetteer hit (score ≥ 0.85) + causal relation extracted by `CausalSentence` | `high` |
| Anchored NP or fuzzy gazetteer hit + `CausalSimple` extraction | `medium` |
| LLM fallback used, or only `G5` entities found with no `G4` mechanism | `low` |
| No entities extracted (null record) | `low` + route to human review |

### 3.5 Known extraction limitations

- **Symptom vs. cause confusion:** CRs frequently record observable effects rather than FMEA-level mechanisms. The `cause_is_symptom` flag signals this — downstream consumers weight these records lower. The LLM is never used to resolve symptoms into mechanisms; that inference is the analyst's responsibility.
- **Sparse records:** short WOs with no causal language produce null extraction records with `confidence = low`. These are stored but carry zero weight in recurrence scoring unless a human reviewer promotes them.
- **Gazetteer coverage:** the NER gazetteer must be seeded with the plant's FMEA taxonomy and component terminology. Coverage gaps directly limit extraction quality and must be assessed before implementation begins.
- **Consistency:** `extraction_version` must combine NER pipeline version, gazetteer version, and LLM model+prompt version. `embedding_model_version` must be tracked separately. Any change to either requires re-extraction and re-embedding of all affected records.

---

## 4. Embedding Similarity Search (Stage 2)

### 4.1 What is embedded

Each extraction record is embedded as a concatenation of its semantic fields:

```
embed_text = f"{identified_effect} | {assessed_cause} | {inferred_fm_label}"
```

`as_found`, `as_left`, and `procedural_deviation_score` are **not** included in the embedding — they are used only as metadata filters at query time. `as_found` and `as_left` have low semantic content ("degraded", "acceptable") that would dilute the embedding; they are more useful as hard filters (e.g., only retrieve records where `as_found = degraded` when the candidate involves a degraded component).

This embedding is computed at ingestion time. If `identified_effect` or `assessed_cause` is null (sparse record), only the non-null fields are concatenated.

**Note on metadata storage:** `identified_effect`, `assessed_cause`, and `inferred_fm_label` are stored in Chroma metadata in addition to being embedded as document content. This allows `SemanticMatch` objects retrieved by `DocExtractionStore.query()` to carry the full text fields for analyst audit trail display, without requiring a separate document fetch.

### 4.2 Query construction at run time

For each active failure mode candidate in the current event:

```
query_text = f"{fm.name} | {fm.expected_symptoms}"
```

The `event.symptom_description` field listed in earlier drafts was removed from the query because it is not reliably available on the FM dict at TSKR scoring time; the FM name and expected symptoms provide sufficient discriminative signal. Cosine similarity is computed against all stored extraction record embeddings. **The embedding model used at query time must match `embedding_model_version` stored in the collection** — a mismatch must raise an error, not produce silently wrong similarity scores.

### 4.3 Scoring and recurrence count semantics

Semantic matches **supplement** but do not replace exact-match recurrence. The two pools are kept separate and combined as follows:

```
exact_pool    = past events matched by component_id and fm_id  (integer count, weight 1.0)
semantic_pool = top-k extraction records above similarity_threshold
                    weighted by: similarity_score × confidence_weight × cause_is_symptom_factor

# Fractional recurrence contribution per semantic match:
semantic_contribution = similarity_score × confidence_weight × cause_is_symptom_factor

# where:
#   confidence_weight:       high=1.0 / medium=0.7 / low=0.3
#   cause_is_symptom_factor: 0.5 if cause_is_symptom=True, else 1.0

# Effective recurrence count for history_score:
effective_recurrence_count = len(exact_pool) + sum(semantic_contribution for each semantic match)
```

`effective_recurrence_count` is a float. The `history_score` lookup table (0→0.0, 1→0.35, 2-3→0.55, 4-6→0.70, >6→0.80) applies to `floor(effective_recurrence_count)` — **these thresholds were calibrated for integer exact-match counts and will require recalibration** once labeled data is available. The bonuses (recency, trend, unresolved) continue to apply only to exact-match records, not semantic matches, since semantic records do not carry resolution status.

**Deduplication:** when building the semantic pool, group results by `doc_id` and keep only the highest-scoring chain per document to avoid double-counting a single CR that contains multiple causal chains. Deduplication is performed inside `DocExtractionStore.query()` before results are returned.

**`novel_pattern` semantics with near-miss records:** a record with similarity just below `similarity_threshold` does not increment `effective_recurrence_count`. To avoid silent misclassification, a `near_match_pattern` flag is added alongside `novel_pattern`:

```
novel_pattern      = (effective_recurrence_count == 0)
near_match_pattern = (novel_pattern == True AND
                      any semantic match exists with similarity in [similarity_threshold - 0.10, similarity_threshold))
```

`near_match_pattern = True` appears in `tskr_patterns` and triggers an attention flag on the RCA card: "novelty uncertain — near-threshold semantic match found; analyst review recommended."

### 4.4 Integration points

| Pipeline step | Current behavior | With semantic search |
|---|---|---|
| **Step 3 — recurrence pool** | Exact match on `component_id` and `fm_id` | Augmented with weighted semantic matches; `effective_recurrence_count` replaces integer `recurrence_count` in `history_score` |
| **Step 2d — plant tier** | Scoring on `matched_component_ids`, `matched_failure_mode_ids`, `event_type`, `actuation_type` | `semantic_similarity_score` dimension added at weight 0.10; existing five dimensions renormalized × 0.90 so total remains 1.0. Applies to CMMS-sourced past events only (KG-native events have no extraction record). |

### 4.5 Similarity threshold and scoring parameters

| Parameter | Suggested default | Exposed in | Notes |
|---|---|---|---|
| `similarity_threshold` | 0.75 | `OrchestratorConfig.semantic_similarity_threshold`, `TSKRTemporalScorerConfig.semantic_similarity_threshold` | Primary inclusion threshold; requires empirical calibration against labeled data before production use |
| `near_match_window` | 0.10 | `OrchestratorConfig.near_match_window`, `TSKRTemporalScorerConfig.near_match_window` | Width of the soft zone below threshold that triggers `near_match_pattern` |
| `fm_id_resolution_threshold` | 0.80 | `OrchestratorConfig.fm_id_resolution_threshold` | Minimum similarity for `fm_id_candidate` resolution (Step C) |
| `confidence_weight` | high=1.0 / medium=0.7 / low=0.3 | `SemanticMatch.confidence_weight` (property) | Scales semantic contribution per record |
| `cause_is_symptom_factor` | 0.5 | `SemanticMatch.cause_is_symptom_factor` (property) | Halves contribution of records where `cause_is_symptom=True` |
| `top_k_semantic` | 5 | `OrchestratorConfig.top_k_semantic`, `TSKRTemporalScorerConfig.top_k_semantic` | Maximum semantic matches retrieved per failure mode before deduplication |
| `enable_semantic_recurrence` | `False` | `OrchestratorConfig.enable_semantic_recurrence`, `TSKRTemporalScorerConfig.enable_semantic_recurrence` | Feature gate — off by default; set True to activate semantic augmentation |

---

## 5. Precision and False Positive Risk

In a nuclear context, a false positive — a semantic match that causes the analyst to treat a novel event as a known recurrence — is as dangerous as a missed recurrence. The following controls address this risk:

- **Fractional contributions:** semantic matches contribute fractional values to `effective_recurrence_count`, not integer increments. A single low-confidence near-match cannot push a novel event above the recurrence threshold on its own.
- **`cause_is_symptom_factor`:** records that describe symptoms rather than mechanisms receive half weight, reducing the risk of symptom-level false matches inflating recurrence counts.
- **`near_match_pattern` flag:** near-threshold matches are surfaced to the analyst rather than silently included or excluded.
- **`fm_id_candidate` deferred resolution threshold (0.80):** conservative threshold for FM assignment prevents spurious ID matches.
- **LLM extraction-only constraint:** the LLM is prohibited from inferring mechanisms not present in the text, removing a key source of false specificity.
- **Human review path for `low` confidence records:** null and low-confidence extractions do not enter the recurrence pool until a human reviewer promotes them.
- **Feature gate:** `enable_semantic_recurrence = False` by default in both `OrchestratorConfig` and `TSKRTemporalScorerConfig`. Semantic augmentation must be explicitly enabled; existing pipeline behaviour is unchanged until it is.

### Success criteria

The feature is considered validated when, on a labeled evaluation set of plant records:

| Metric | Minimum acceptable | Target |
|---|---|---|
| Precision (true positive semantic matches / all semantic matches above threshold) | ≥ 0.80 | ≥ 0.90 |
| Recall (true positive semantic matches / all known same-FM record pairs) | ≥ 0.70 | ≥ 0.85 |
| False positive rate on negative pairs (different FM, same component) | ≤ 0.10 | ≤ 0.05 |
| `novel_pattern` flag accuracy (correctly identifies first occurrences) | ≥ 0.90 | ≥ 0.95 |
| Regression: no change to exact-match `recurrence_count` for well-tagged records | 100% | 100% |

---

## 6. Data Flow and Storage

```
Ingestion pipeline
    source document (CR / WO / RCA / ECA only — see §2.3)
        ├── existing path → Chroma chunk store (full text, existing metadata)
        └── new path → HybridNERPipeline + causal_condition_adapter
                            → LLM fallback if Steps A+B both fail
                            → one extraction record per causal chain
                            → embed(identified_effect | assessed_cause | inferred_fm_label)
                            → stored in Chroma collection: "doc_extractions"
                                metadata: doc_id, chain_index, doc_type,
                                          identified_effect, assessed_cause, inferred_fm_label,
                                          fm_id_candidate (null until resolved),
                                          fm_id_candidate_alt (null until resolved),
                                          confidence, cause_is_symptom,
                                          as_found, as_left,
                                          procedural_deviation_score,
                                          extraction_version, embedding_model_version

RCA run — one-time batch step (before Step 3 / Step 2d)
    for each unique inferred_fm_label in "doc_extractions" not yet resolved:
        → embed(inferred_fm_label)
        → similarity lookup against KG failure mode list for current asset neighborhood
        → write fm_id_candidate (and fm_id_candidate_alt) back to collection if similarity ≥ 0.80

RCA run — Step 3 (when enable_semantic_recurrence = True)
    → embed(fm.name | fm.expected_symptoms)
    → Chroma similarity query on "doc_extractions" (model version guard enforced)
    → top-k results deduplicated by doc_id inside DocExtractionStore.query()
    → weighted semantic contributions merged into effective_recurrence_count
    → near_match_pattern flag set if only near-threshold matches found
```

---

## 7. Development Tasks

### Phase 1 — Extraction adapter ✅ Complete

**Files:** `src/dackar/RCA/doc_extraction/schema.py`, `src/dackar/RCA/doc_extraction/adapter.py`, `src/dackar/RCA/doc_extraction/__init__.py`
**Tests:** `src/dackar/RCA/unit_tests/test_doc_extraction_adapter.py` (36 tests, all passing)

- [x] Define `DocExtractionRecord` dataclass with `embed_text()`, `is_null_record()`, `as_chroma_metadata()` helpers (`schema.py`)
- [x] Define `ConfidenceLevel` enum (`high` / `medium` / `low`)
- [x] Implement `DocExtractionAdapter` wrapping `HybridNERPipeline` + `causal_condition_adapter`; one record per causal chain; null record for documents with no causal language
- [x] Implement `cause_is_symptom` detection (G5 overlap check on cause text)
- [x] Implement confidence assignment per §3.4: HIGH = CausalSentence + gazetteer ≥ 0.85 + stmt_confidence ≥ 0.60; MEDIUM = CausalSentence or CausalSimple with mechanism; LOW = LLM fallback or no G4 entity
- [x] Skip negated / conjectural causal statements (absence, not occurrence)
- [x] `fm_id_candidate` always `None` at extraction time (deferred to Step C)
- [x] Dependency injection for `_causal_extractor` to avoid spacy module-level import at test collection time
- [x] `EXTRACTABLE_DOC_TYPES = frozenset({"CR", "WO", "RCA", "ECA"})` enforced; `ValueError` on out-of-scope types

**Implementation notes:**
- `causal_condition_adapter` is loaded lazily via `_get_causal_extractor()` to avoid transitive spacy import at test collection time
- Negated/conjectural statements produce a null record (not silently dropped) so the document is never missing from the store
- LLM fallback path is handled by `causal_condition_adapter` itself; `extractor_used == "LLM_implicit"` triggers LOW confidence

---

### Phase 2 — Embedding store ✅ Complete

**Files:** `src/dackar/RCA/doc_extraction/store.py`
**Tests:** `src/dackar/RCA/unit_tests/test_doc_extraction_store.py` (41 tests, all passing)

- [x] `DocExtractionStore(persist_directory, embed_model, ollama_base_url, fm_resolution_threshold)` with lazy Chroma collection init
- [x] `upsert(record) → record_id` and `upsert_batch(records) → int`; Chroma record ID format `"{doc_id}::chain::{chain_index}"`
- [x] `query(query_text, top_k, similarity_threshold, near_match_window, filter_meta) → (matches, near_matches)`:
  - Converts Chroma cosine distance to similarity via `sim = 1.0 - distance`
  - Deduplicates by `doc_id` (best chain per document)
  - Returns `(matches, near_matches)` where near_matches contains results in `[threshold - window, threshold)`
- [x] `_assert_model_version()`: raises `EmbeddingModelVersionError` on stored-vs-query model mismatch
- [x] `resolve_fm_candidates(fm_list, resolution_threshold)`: batch KG FM resolution; writes `fm_id_candidate` + `fm_id_candidate_alt` back to collection for above-threshold matches
- [x] `count()` and `delete_by_doc_id()` utilities
- [x] `_build_where_clause()`: builds Chroma `$eq` / `$in` / `$and` filter from metadata dict
- [x] `SemanticMatch.semantic_contribution` property: `similarity_score × confidence_weight × cause_is_symptom_factor`
- [x] All Chroma / LangChain imports are lazy (inside methods) — module importable without Chroma installed

**Bug fix applied (April 2026):**
`as_chroma_metadata()` now stores `identified_effect`, `assessed_cause`, `inferred_fm_label` in the metadata dict in addition to using them as the Chroma document content. This allows `SemanticMatch` objects to carry the full text fields for analyst audit display. Previously these fields were stored only as document content and always returned `None` from query results.

---

### Phase 3 — Pipeline integration ✅ Core complete; Step 2d dimension deferred

**Files:** `src/dackar/RCA/orchestrators/tskr_temporal_scorer.py`, `src/dackar/RCA/orchestrators/rca_reasoning_orchestrator.py`
**Tests:** `src/dackar/RCA/unit_tests/test_doc_extraction_pipeline_integration.py` (45 tests, all passing)

#### Step 3 — effective recurrence count ✅

- [x] `TSKRTemporalScorerConfig` gains 4 semantic fields: `enable_semantic_recurrence` (default `False`), `semantic_similarity_threshold` (0.75), `near_match_window` (0.10), `top_k_semantic` (5)
- [x] `TSKRTemporalScorerV1.__init__` accepts `doc_extraction_store` parameter
- [x] `_score_from_effective_count(effective_count: float, profile: RecurrenceProfile) → float`: same tier brackets as `_score_from_recurrence_profile` applied to `floor(effective_count)` per §4.3
- [x] `_score_failure_mode_pattern`: when `enable_semantic_recurrence=True` and store present, queries store using `fm.name | fm.expected_symptoms`; accumulates `semantic_contribution` values; re-scores history via `_score_from_effective_count` when contributions are positive; sets `near_match_pattern=True` when only near-threshold matches found and exact pool is empty
- [x] Pattern output gains: `effective_recurrence_count` (float), `semantic_match_count` (int), `near_match_count` (int), `near_match_pattern` (bool); `novel_pattern` updated to use `effective_recurrence_count`
- [x] Graceful fallback when `DocExtractionStore.query()` raises — semantic fields default to zero, pattern proceeds

#### Orchestrator integration ✅

- [x] `OrchestratorConfig` gains 5 semantic fields: `enable_semantic_recurrence`, `semantic_similarity_threshold`, `near_match_window`, `fm_id_resolution_threshold`, `top_k_semantic`
- [x] `RCAReasoningOrchestrator` gains `doc_extraction_store` field and `set_doc_extraction_store()` setter
- [x] `_apply_tskr_runtime_overrides()` propagates all semantic config params and the store into the scorer at run time
- [x] `run()` calls `resolve_fm_candidates()` after `kg_context` is built when `enable_semantic_recurrence=True` (Step C, §3.3)
- [x] `_apply_near_match_pattern_attention_flags()`: adds analyst attention flag to RCA card when any pattern has `near_match_pattern=True`
- [x] `_build_semantic_recurrence_provenance()`: summarises `semantic_recurrence_used`, `semantic_match_count`, `near_match_count`, `near_match_fm_ids` across all TSKR patterns; included in `run_manifest.pipeline_config.semantic_recurrence`

#### Step 2d semantic dimension ⏳ Deferred to Phase 3b

`_query_plant_past_events()` was not modified in Phase 3. Adding a `semantic_similarity_score` dimension at weight 0.10 with renormalized existing weights requires deciding how to associate past events (identified by `event_id`) with their extracted document records in the store — past events in `kg_context.past_events` reference `fm_id` and `component_id` but not `doc_id`. This join requires either a `source_doc_id` field on past events (a schema change) or a separate lookup path. Deferred to Phase 3b pending schema alignment.

---

### Phase 3b — Step 2d semantic dimension (planned)

- [ ] Add `source_doc_id` field to `kg_context.past_events` records (KG schema change)
- [ ] Modify `_query_plant_past_events()`: for each past event with a `source_doc_id`, retrieve the highest-similarity extraction record from the store as `semantic_similarity_score`; add as dimension at weight 0.10 with existing dimensions renormalized (component 0.36, FM 0.23, event_type 0.14, actuation 0.09, window 0.09, semantic 0.10; total → 1.01 before normalization — renormalize to 1.0)
- [ ] Alternatively, query the store directly using top causality candidates' FM descriptions and match by `fm_id_candidate` field in the store records
- [ ] Add `semantic_similarity_score` to `SimilarEvent` output schema

---

### Phase 4 — Calibration and validation (planned)

- [ ] Build labeled evaluation set: known CR/WO pairs that are same-FM (positive) and different-FM same-component (negative)
- [ ] Calibrate `similarity_threshold` and `fm_id_resolution_threshold` against labeled set; target metrics in §5
- [ ] Recalibrate `history_score` lookup table for `effective_recurrence_count` as a float (current thresholds calibrated for integer exact-match counts — see §4.3 note)
- [ ] Regression test: exact-match `recurrence_count` unchanged for well-tagged records (TC-1 through TC-7)
- [ ] Verify `near_match_pattern` fires correctly at threshold boundary
- [ ] Verify precision / recall / false positive rate meet §5 success criteria

---

## 8. Testing Strategy

### Unit tests — Phase 1 (36 tests, `test_doc_extraction_adapter.py`)
- `DocExtractionAdapter`: given a CR text with two independent cause-effect pairs, assert two extraction records are produced with correct `chain_index` values
- `cause_is_symptom`: G5-only record → `True`; G4 record → `False`; G4+G5 compound → `False`
- `confidence` assignment: test cases spanning all three levels including LLM fallback trigger
- Negated/conjectural statements: skipped; all-negated document → null record
- `fm_id_candidate` always `None` at extraction time
- Document type enforcement: `ValueError` for out-of-scope types
- NER pipeline failure: graceful fallback to empty entity lists
- Schema helpers: `embed_text()`, `is_null_record()`, `as_chroma_metadata()`

### Unit tests — Phase 2 (41 tests, `test_doc_extraction_store.py`)
- `SemanticMatch` computed properties: `confidence_weight`, `cause_is_symptom_factor`, `semantic_contribution`
- `upsert` / `upsert_batch`: record ID format, embedding model version written back, null records stored
- `query`: threshold filtering, deduplication by `doc_id`, near_match band, top_k cap, semantic text fields round-trip
- Embedding model version guard: mismatch raises `EmbeddingModelVersionError`; match and empty collection do not raise
- `resolve_fm_candidates`: resolves above-threshold labels, skips below-threshold, writes `fm_id_candidate_alt`
- `count()`, `delete_by_doc_id()`
- `_build_where_clause`: scalar, list, multi-field, empty filter
- `_meta_to_semantic_match`: handles missing/malformed metadata

### Unit tests — Phase 3 (45 tests, `test_doc_extraction_pipeline_integration.py`)
- `TSKRTemporalScorerConfig`: semantic field defaults and overrides
- `TSKRTemporalScorerV1`: store parameter accepted; disabled when `enable_semantic_recurrence=False`
- `_score_from_effective_count`: all tier brackets, trend/unresolved/recency bonuses, clamp
- `_score_failure_mode_pattern` with semantic disabled: baseline fields present, store not called
- `_score_failure_mode_pattern` with semantic enabled: effective count updated, history rescored, `near_match_pattern` logic, query args forwarded correctly
- Near-match-only: `near_match_pattern=True`, effective count unchanged, `novel_pattern=True`
- Graceful fallback on store exception
- Empty FM name skips query
- `novel_pattern` uses `effective_recurrence_count`
- `OrchestratorConfig`: semantic field defaults and overrides
- `set_doc_extraction_store()` setter
- `_apply_tskr_runtime_overrides`: semantic config propagated to scorer
- `_apply_near_match_pattern_attention_flags`: flag added, idempotent, absent when no near-match
- `_build_semantic_recurrence_provenance`: correct aggregation

### Integration tests (planned — Phase 4)
- Ingest sample CRs → verify `"doc_extractions"` collection contains correct records
- Run `fm_id_candidate` batch resolution → verify above-threshold FMs resolved
- Run Step 3 with and without semantic augmentation on a fixture with same FM in differently-worded CRs → verify `effective_recurrence_count` increases
- Verify embedding model version mismatch raises error at query time

### Regression tests (planned — Phase 4)
- TC-1 through TC-7 with `enable_semantic_recurrence=True`: verify no change to candidate ranking or gate outcomes for cases with well-tagged KG records
- Verify `novel_pattern` semantics unchanged for failure modes with zero exact-match and zero semantic match

---

## 9. Open Questions

1. **Chroma collection strategy:** separate `"doc_extractions"` collection vs. additional vectors in existing document chunks — trade-off between query isolation and operational complexity. Separate collection is implemented as recommended.
2. **Re-extraction policy on CMMS updates:** when a CR is updated in CMMS and re-ingested, extraction records for that `doc_id` should be invalidated and re-run via `delete_by_doc_id()` + re-upsert. A re-extraction trigger keyed on `doc_id` + `extraction_version` is needed; the mechanism for detecting CMMS record updates is deployment-specific.
3. **Gazetteer coverage assessment:** the NER gazetteer must cover the plant's FMEA taxonomy before extraction quality can be evaluated. This assessment should be the first task before Phase 1 development begins.
4. **Human review tooling:** the operational path for low-confidence records (who reviews, what interface, how promoted to the store) is outside the current codebase scope and requires a product decision.
5. **`history_score` recalibration data requirements:** recalibrating the lookup table thresholds for `effective_recurrence_count` as a float requires a labeled dataset of plant records with known recurrence ground truth. The size and source of this dataset need to be defined before Phase 4 can be scoped.
6. **Step 2d source_doc_id schema gap:** `kg_context.past_events` records do not currently carry a `source_doc_id` field linking them to their originating CR/WO. Resolving this is a prerequisite for Phase 3b. The simplest fix is to add `source_doc_id` to the KG ingestion workflow for CMMS-sourced past events.
7. **Double-counting risk between exact pool and semantic pool:** there is no deduplication between past events in `kg_context.past_events` (the exact pool) and extraction records in the semantic pool. A single CR could appear in both if it was both formally tagged with `fm_id` in the KG and extracted into the semantic store. The risk is low (formally tagged CRs are uncommon) but should be addressed in Phase 3b by checking `source_doc_id` overlap when the field becomes available.
