# Step 3 / Step 3.5 Modification Review

## Context

This review evaluates two proposed enhancements to the RCA pipeline:

1. Document similarity + semantic extraction (NER + embeddings)
2. Temporal episode pattern matching (logs, SOE, anomaly sequences)

The objective is to assess how these proposals improve:
- Step 3 — documentary pattern recognition
- Step 3.5 — signal pattern recognition

The reference baseline is the current RCA workflow, where Step 3 operates on structured past events (`kg_context.past_events`) and Step 3.5 classifies patterns using `tskr_patterns` outputs and recurrence metadata.

---

## 1. Semantic Extraction for Step 3 (Primary Improvement)

The current Step 3 logic assumes that past events carry structured identifiers such as `fm_id`. In practice, most CRs and WOs do not satisfy this assumption, which directly affects `recurrence_count` and `novel_pattern` accuracy. This limitation is already acknowledged in the RCA workflow: recurrence depends on correct failure-mode attribution, which is often missing or inconsistent in historical records.

The proposed semantic extraction pipeline addresses this by shifting the problem upstream. Instead of relying on structured IDs at run time, it extracts causal information at ingestion time and stores it as structured, versioned metadata. During the RCA run, Step 3 can then operate on this enriched representation using similarity rather than exact matching.

This aligns well with the existing architecture. Step 3 already consumes structured metadata derived from `kg_context.past_events`; the proposal simply improves the quality of that metadata. Importantly, extraction is performed outside the RCA run, so determinism is preserved. The pipeline continues to operate on stored artifacts, not on live LLM outputs.

The decision to generate one extraction record per causal chain is also consistent with the RCA model. Candidates in Step 4 are defined at the failure-mode level, and chain-level extraction ensures that similarity operates at the same granularity. Document-level embeddings would be too coarse and would dilute the relationship between symptom, cause, and failure mode.

From a system perspective, this modification directly strengthens the inputs to:
- Step 3 recurrence profiling (via improved `recurrence_count` and `history_score`)
- Step 4 temporal scoring (through better recurrence signals embedded in `tskr_patterns`)

---

## 2. Required Corrections Before Adoption

The proposal is sound, but two issues must be addressed before it can be considered reliable.

The first is double counting. The same CR can appear both in `kg_context.past_events` (exact match path) and in the semantic extraction store. Without a shared identifier such as `source_doc_id`, the same historical evidence may be counted twice, artificially inflating recurrence. This directly impacts `effective_recurrence_count` and therefore `history_score`. This must be resolved at the data model level before enabling semantic recurrence.

The second issue is calibration. The current history scoring logic is based on integer recurrence counts. Introducing fractional contributions from semantic matches changes the meaning of the score, but the thresholds remain unchanged. Until recalibration is performed, the scoring remains internally consistent but not physically meaningful. For development this is acceptable, but it should not be treated as validated output.

A third concern is failure-mode resolution. Mapping `inferred_fm_label` to a KG `fm_id` using embeddings introduces a new source of silent error. If the mapping is incorrect, recurrence is attributed to the wrong failure mode. This does not break the pipeline structurally, but it degrades trust in the output. Conservative thresholds and audit visibility are required.

---

## 3. Temporal Pattern Matching (Not a Step 3 Fix)

The temporal pattern matching proposal introduces a different abstraction. Instead of reasoning over failure modes, it operates on episodes derived from event density across alarms, SOE, and anomalies. Similarity is computed using co-occurrence, ordering, and frequency metrics.

This is conceptually aligned with Step 2d (similar event identification), not Step 3. Step 3 is failure-mode-centric and feeds directly into candidate ranking. The temporal pattern method is episode-centric and produces similarity between operational signatures.

Because of this mismatch, integrating it directly into Step 3 would complicate the pipeline without addressing the core limitation of documentary recurrence. It introduces a new data model (episodes), new parameters (density thresholds, similarity weights), and new calibration requirements, none of which map directly to the A–L causal category framework.

However, the approach is still valuable. It provides a structured way to retrieve historical events based on signal behavior, which complements the existing plant-tier matching in Step 2d. It should therefore be positioned as an enhancement to Step 2d or as an additional evidence source, not as a replacement for Step 3.

---

## 4. Cross-Pattern Recognition (Recommended Next Step)

To keep the development aligned with the RCA workflow, the role of Step 3 and Step 3.5 should be stated more precisely.

In the current workflow, Step 3 is the documentary pattern-recognition layer. It operates on `kg_context.past_events` and produces a recurrence profile per failure mode. Its outputs are quantities such as `recurrence_count`, `recurrence_trend`, `unresolved_recurrence_count`, `history_score`, and `novel_pattern`. In other words, Step 3 does not natively retrieve "similar events" in a broad sense; it evaluates whether the current failure mode has documentary support in historical CRs, WOs, and prior RCA findings.

Step 3.5 is the signal pattern-recognition layer. In the current workflow, it classifies the current pattern as matched or novel using the TSKR outputs and the historical support already assembled for Step 3. It does not yet retrieve historical SOE or alarm episodes as first-class objects. That capability would come from the proposed temporal pattern-matching framework, which should be treated as an extension rather than assumed to already exist.

With that distinction in place, the intended future architecture becomes clearer. The goal is not simply to "cross" two sets of similar records. The goal is to build explicit links between three entities: the current event's signal signature, historical signal episodes, and historical documents such as CRs and WOs. Once those links exist, the system can answer the more useful engineering question: whether a current signal signature has appeared before and, if so, what documented cause or failure mode was associated with it.

This is more powerful than either layer alone. Documentary recurrence can fail when the historical record lacks a clean `fm_id`. Signal recurrence can fail when a historical SOE or alarm pattern exists but was never formally closed through RCA. If the system can connect a historical signal episode to a CR or WO and that document semantically aligns with a current candidate failure mode, the analyst receives a much stronger and more traceable form of support.

This should be implemented as a linking layer, not as a merged score at the outset. The linking layer should remain explicit and auditable so that analysts can see where signal similarity, document similarity, and candidate alignment reinforce each other and where they do not.

### 4.1 Proposed Linkage Data Structures

The cleanest model is to represent the new capability as a chain of linked artifacts rather than as one opaque similarity object.

A historical signal episode should be represented independently from a historical document record. The episode object is the retrieval result from the temporal pattern-matching subsystem; the document object is the extraction result from the semantic recurrence subsystem. The bridge between them should be explicit.

A practical minimal schema is:

```python
@dataclass
class HistoricalSignalEpisode:
    episode_id: str
    asset_id: str
    window_start: datetime
    window_end: datetime
    source_types: list[str]              # alarm, soe, anomaly
    event_set: frozenset[str]
    event_seq: list[str]
    freq_vec: dict[str, int]
    similarity_to_current: float
    linked_doc_ids: list[str]            # populated by linkage step
    index_status: str                    # "indexed" | "no_episodes_indexed" | "stale"
    # "indexed"              — episode is from a current, populated index; eligible
    #                          for linkage
    # "no_episodes_indexed"  — the index contained no episodes for this asset
    #                          neighborhood; linkage must not be attempted (§4.11)
    # "stale"                — the index was built outside the configured staleness
    #                          window; linkage is allowed but flagged low-confidence
```

```python
@dataclass
class HistoricalDocExtraction:
    doc_id: str
    doc_type: str                        # CR, WO, RCA, ECA
    asset_id: str | None
    event_time_start: datetime | None    # time of the plant event, not document creation
    event_time_end: datetime | None
    event_time_confidence: str           # "explicit" | "inferred" | "absent"
    # "explicit"  — event_time fields populated from a structured timestamp field
    # "inferred"  — derived from surrounding document context; less reliable
    # "absent"    — no event-time information available; level-2 temporal linkage
    #               cannot run; linkage falls through to level-3 (semantic/FM)
    identified_effect: str | None
    assessed_cause: str | None
    inferred_fm_label: str | None
    fm_id_candidate: str | None
    fm_id_candidate_alt: str | None
    fm_resolution_status: str            # "auto_resolved" | "ambiguous" | "unresolved"
    fm_resolution_score: float | None
    confidence: str                      # high, medium, low
    cause_is_symptom: bool
    source_episode_ids: list[str]        # populated by linkage step
```

```python
@dataclass
class CrossPatternLink:
    link_id: str
    episode_id: str
    doc_id: str
    asset_match: bool
    time_overlap_hours: float | None
    temporal_link_skipped: bool          # True when event_time_confidence == "absent"
    linkage_precedence_level: int        # 1 = direct reference, 2 = temporal+asset,
    #                                      3 = semantic/FM (see §4.5)
    component_overlap: list[str]
    fm_alignment_score: float | None
    signal_similarity_score: float
    document_similarity_score: float | None
    link_confidence: float
    provenance: dict[str, Any]
```

```python
@dataclass
class CandidateCrossPatternEvidence:
    candidate_id: str
    component_id: str
    fm_id: str
    linked_episode_ids: list[str]
    linked_doc_ids: list[str]
    best_link_score: float
    support_posture: str                 # reinforcing, conflicting, weakly_supporting,
    #                                      unresolved — see §4.7
    reinforcement_strength: str          # "single" | "multiple_consistent" | "mixed"
    # populated only when support_posture == "reinforcing" or "weakly_supporting"
    # "single"             — exactly one reinforcing link above threshold
    # "multiple_consistent"— two or more reinforcing links, all pointing to the same
    #                         FM or cause; treated as qualitatively stronger but not
    #                         numerically accumulated in Phase 1 (see §4.6)
    # "mixed"              — two or more links, not all pointing to the same FM/cause;
    #                         treated as weakly_supporting regardless of individual scores
    # None when support_posture == "conflicting" or "unresolved"
    linkage_outcome: str                 # "linked" | "no_data" | "no_match" |
    #                                      "below_threshold" — see §4.11
    evidence_paths: list[CrossPatternLink]
```

### 4.2 How the Links Should Be Built

The links should not be inferred from one signal alone. They should be created using a small set of explicit checks.

The first check is asset and component compatibility. A historical episode and a historical document should belong to the same asset or to a valid mapped neighborhood.

The second check is temporal compatibility. If a CR or WO corresponds to the same operational event as a historical SOE or alarm episode, their event-time windows should overlap or fall within a configurable tolerance. Temporal compatibility uses `event_time_start` / `event_time_end` on `HistoricalDocExtraction`, not document creation time. When `event_time_confidence == "absent"`, this check cannot run and `temporal_link_skipped = True` is recorded on the `CrossPatternLink`. The linkage then falls through to level 3.

The third check is semantic compatibility. If the document extraction resolves to an `fm_id_candidate`, that failure mode should align with the same FM family or candidate set implicated by the historical signal episode.

**Episode-to-candidate mapping rule:** An episode links only to candidates whose `fm_id` overlaps with the episode's implicated FM set — that is, candidates where `fm_id` matches `fm_id_candidate` on any document linked to the episode, or where embedding similarity between the episode's event labels and the candidate's FM description exceeds `fm_alignment_score_threshold`. An episode does not link to all active candidates indiscriminately. When multiple candidates qualify, a separate `CrossPatternLink` is built for each qualifying candidate-episode pair. When no candidate FM matches, the episode produces no `CandidateCrossPatternEvidence` entry but is retained in `cross_pattern_evidence.json` for audit purposes.

**Link confidence formula:**

```python
# All terms that are present contribute at their full weight.
# Missing terms (None or empty) default to 0.0 and the formula is
# renormalized over present terms only, so that a missing dimension
# does not silently deflate the score relative to other links.

present_weights = {}

present_weights["signal"]   = 0.30   # always present
present_weights["temporal"] = 0.20   if time_overlap_hours is not None else 0.0
present_weights["fm"]       = 0.20   if fm_alignment_score is not None else 0.0
present_weights["document"] = 0.30   if document_similarity_score is not None else 0.0

total_weight = sum(present_weights.values())   # always > 0 since signal is always present

link_confidence = (
    present_weights["signal"]   * signal_similarity_score +
    present_weights["temporal"] * (temporal_compatibility_score or 0.0) +
    present_weights["fm"]       * (fm_alignment_score or 0.0) +
    present_weights["document"] * (document_similarity_score or 0.0)
) / total_weight
```

Missing dimensions are always recorded in `provenance` so the analyst can see which terms contributed. A link where only `signal_similarity_score` is present is valid but carries lower weight in practice because `total_weight == 0.30` means the confidence is bounded by the signal score alone.

Temporal compatibility may be implemented as a gate rather than a score component: links where the event-time gap between episode and document exceeds `temporal_compatibility_max_gap_days` (default 180 days) are suppressed entirely. The gate approach is preferred when asset-class temporal norms are well understood; the weighted formula is preferred during initial deployment. The mode is controlled by `CrossPatternConfig.temporal_compatibility_mode` (`"gate"` or `"formula"`).

This formula should not be used to modify candidate ranking initially. It should only be used to sort and filter cross-pattern evidence presented to the analyst and to downstream synthesis.

### 4.3 Where This Should Live in the Pipeline

The linkage layer should not live inside Step 3 itself. Step 3 should remain the place where documentary recurrence per failure mode is computed. Likewise, candidate generation in Step 4 should remain based on the KG and TSKR artifacts, not on cross-pattern links.

The most natural place for the new capability is between Step 2d and Step 5, with final visibility in Step 5 and Step 6.

Step 2d should be extended to return historical signal episodes as explicit artifacts when the temporal pattern-matching subsystem is enabled. After Step 2d completes, but before Step 5 evidence refinement, the orchestrator should run a new linkage stage. Step 5 should then consume `cross_pattern_evidence` alongside `evidence_bundle` and `allen_relation_map` as auxiliary evidence only. Step 6 surfaces the result in the RCA card and run manifest.

This placement keeps the pipeline stable:
- Step 3 remains documentary recurrence.
- Step 3.5 remains signal-pattern classification.
- Step 2d becomes the source of `HistoricalSignalEpisode` objects.
- The new linkage stage becomes a bridge artifact.
- Step 5 and Step 6 are the places where the bridge is consumed and surfaced.

### 4.4 Development Recommendation

Development should proceed in two phases.

The first phase should be evidence-only. Build the data structures, create the linkage stage, persist the artifact, and expose it in the RCA card and run manifest. This is enough to validate whether the bridge is useful without destabilizing the ranking engine.

The second phase should consider controlled score influence, but only after calibration. If cross-pattern linkage proves reliable, it can later contribute a small bounded increment to evidence posture or analyst attention logic. It should not become a dominant driver of candidate ranking unless the team has labeled data showing that the links are both precise and stable.

---

### 4.5 Linkage Precedence Rules

The linkage process evaluates candidates in the following sequence:

1. **Direct reference (highest confidence)**
   Match via `source_doc_id`, work order ID, or event reference.

2. **Temporal + asset/component alignment**
   Same asset (or validated KG neighborhood); overlapping or near-overlapping *event-time* windows (not document creation time). Requires `event_time_confidence != "absent"`; when absent this level is skipped, `temporal_link_skipped = True` is set, and linkage falls through to level 3.

3. **Semantic / FM alignment (fallback only)**
   Similar `inferred_fm_label` or matching `fm_id_candidate`. Used only when stronger signals are absent.

Tie-breaking always favors higher-precedence matches. **Redundancy suppression rule:** when a level-1 link exists for an episode-document pair, all lower-precedence links for that same pair are suppressed from `evidence_paths` entirely.

---

### 4.6 Cardinality and Aggregation Rules

The linkage model handles one-to-many and many-to-one relationships:

- One signal episode → multiple documents
- One document → multiple signal episodes
- One document → multiple causal chains

`CandidateCrossPatternEvidence.best_link_score` is derived from the single highest-confidence link path. Additional paths are retained in `evidence_paths`.

**Reinforcement strength rule:** when `support_posture == "reinforcing"`, the number and consistency of independent links must be made visible without numerically accumulating them into a score. The `reinforcement_strength` field on `CandidateCrossPatternEvidence` encodes this:

- `"single"` — one reinforcing link above threshold
- `"multiple_consistent"` — two or more reinforcing links, all pointing to the same FM or cause; surfaced to the analyst as qualitatively stronger
- `"mixed"` — two or more links not all pointing to the same FM or cause; `support_posture` is downgraded to `"weakly_supporting"` regardless of individual link scores

This keeps Phase 1 safe — no numerical accumulation — while ensuring that multiple independent historical confirmations are visible and interpretable by the analyst.

---

### 4.7 Negative Case Handling (Support Posture and Linkage Outcome)

The system must represent both the result of linkage reasoning and the reason no linkage was possible. These are distinct and must not be conflated.

**`support_posture`** — result of attempted linkage (populated when `linkage_outcome == "linked"`):

| Value | Meaning |
|---|---|
| `reinforcing` | Signal and document evidence align on the same FM/cause |
| `conflicting` | Signal and document evidence point to different FM/cause |
| `weakly_supporting` | Only one side (signal or document) provides support, or `reinforcement_strength == "mixed"` |
| `unresolved` | Linkage was attempted but produced no result above threshold |

**`linkage_outcome`** — reason no linkage was possible (always populated):

| Value | Meaning | RCA card wording |
|---|---|---|
| `"linked"` | At least one valid link was built | Normal cross-pattern summary |
| `"no_data"` | `index_status == "no_episodes_indexed"` or no document extractions exist for this asset | "No historical signal or document data available for cross-pattern assessment." |
| `"no_match"` | Index is populated but search returned no episodes above the similarity floor | "No historically similar signal episodes were found for this event." |
| `"below_threshold"` | Episodes and documents were found but no link survived the `link_confidence_threshold` | "Historical episodes and documents were found but could not be reliably linked." |

These are meaningfully different in RCA. `"no_data"` means the absence of evidence is itself informative about data coverage. `"no_match"` means the pattern may be genuinely novel. `"below_threshold"` means partial evidence exists but is not strong enough to assert. Collapsing all three into an empty result or `"unresolved"` would hide information the analyst needs to interpret the output correctly.

**Behavioral rule for `conflicting` posture:** a `conflicting` support posture for any top-ranked candidate must trigger an analyst attention flag and set `review_required = True` on that candidate, consistent with how near-tie logic operates in the evidence refinement pass. This applies in Phase 1 even though composite scoring is unchanged.

**RCA card wording for `linkage_outcome`:** the card must use the wording in the table above, not a generic "no cross-pattern evidence found." The run manifest must retain the `linkage_outcome` value and any available counts (episodes retrieved, documents retrieved, links attempted, links above threshold) for traceability.

---

### 4.8 First-Phase Behavior Boundary

In the initial implementation, cross-pattern linkage must remain **non-intrusive** to the scoring engine.

It must NOT modify:
- `composite_score`
- hard gate outcomes
- candidate ranking order
- `score_rationale` or any scoring-related field

It should ONLY:
- generate `cross_pattern_evidence.json`
- add analyst attention flags for `conflicting` posture and strong reinforcement
- enrich RCA card narrative using the wording rules in §4.7

The prohibition on `score_rationale` modification is important: if cross-pattern evidence appears in a scoring field during Phase 1 it creates an audit ambiguity about whether the ranking was influenced by an unvalidated layer.

---

### 4.9 Artifact Contract

**`cross_pattern_evidence.json`** must persist:
- All `CrossPatternLink` objects, including `linkage_precedence_level` and `temporal_link_skipped`
- Per-candidate summaries (`CandidateCrossPatternEvidence`), including `linkage_outcome` and `reinforcement_strength`
- A top-level summary: total episodes retrieved, total documents matched, total links built, per-candidate `linkage_outcome` distribution

Integration points:
- **Step 5:** artifact is available but not used in scoring
- **Step 6:** summary injected into `rca_card`; structured summary added to `run_manifest.artifacts`

The run manifest must retain `linkage_outcome`, precedence-level distribution, and `temporal_link_skipped` counts for every candidate, so reviewers can assess the quality of the cross-pattern assessment without reading the full artifact.

---

### 4.10 Semantic Recurrence Safeguards (Pre-Calibration Controls)

**Double-counting guard (mandatory for Phase 3):**
At query time, any semantic match whose `doc_id` is in the exact-match pool must be excluded via the `exact_doc_ids` parameter on `DocExtractionStore.query()`. This guard is required before `source_doc_id` is fully propagated through the schema.

**Conservative history-score policy:**
Until recalibration is complete, semantic contributions must not elevate `history_score` into the same tier as any exact-match recurrence. Semantic-only recurrence may increase score within the lowest tier only. This cap is named `semantic_recurrence_capped` and appears as a boolean field in `tskr_patterns` pattern output.

**FM resolution threshold (fixed default 0.88):**
- ≥ 0.88 → `fm_resolution_status = "auto_resolved"`
- [0.80, 0.88) → `fm_resolution_status = "ambiguous"`; requires analyst review before contributing to recurrence counting
- < 0.80 → `fm_resolution_status = "unresolved"`; `fm_id_candidate` remains `None`

Top-2 FM candidates with similarity scores are always returned regardless of resolution status.

**FM resolution ambiguity attention flags:**
`fm_resolution_ambiguous` must be propagated from `tskr_patterns` into an analyst attention flag at Step 6 via `_apply_fm_resolution_ambiguity_flags()` in the orchestrator. This method must be called alongside `_apply_near_match_pattern_attention_flags()` in Step 6e.

---

### 4.11 Index Status Decision Rules

`index_status` on `HistoricalSignalEpisode` is not only a reporting field — it governs whether linkage is attempted and how results are flagged.

| `index_status` | Linkage behavior | Confidence adjustment | Analyst flag |
|---|---|---|---|
| `"indexed"` | Proceed normally | None | None (unless `linkage_outcome != "linked"`) |
| `"no_episodes_indexed"` | **Do not attempt linkage**; set `linkage_outcome = "no_data"` immediately | N/A | Flag: "No historical signal episodes indexed for this asset; cross-pattern assessment unavailable." |
| `"stale"` | Allow linkage; set `linkage_outcome` normally | Cap `link_confidence` at 0.70 regardless of formula result | Flag: "Signal episode index is stale; cross-pattern results may not reflect recent plant history." |

The staleness cap of 0.70 prevents a stale-index link from reaching a confidence level that would suggest high reliability. It is applied after the formula in §4.2 and recorded in `provenance`.

This means `index_status` has three distinct downstream effects: it gates linkage execution, it adjusts the confidence ceiling, and it controls the analyst attention flag. All three must be implemented together.

---

## 5. Expectations for the Existing `pattern_search` Draft

`pattern_search` should be treated as the producer of historical signal episodes and similarity results, not as a second causality engine. It answers one question: which historical alarm/SOE/anomaly episodes look operationally similar to the current event. It must not generate failure-mode candidates, assign causal categories, or modify ranking logic.

**1. Stable, explicit output object.** Each result must include `episode_id`, event window, similarity scores, matched event diagnostics, provenance, and `index_status`.

**2. Independence from document reasoning.** The module may later provide inputs to the cross-pattern linker but must not directly query CR/WO records or resolve failure modes.

**3. Metadata sufficient for later linkage.** Each episode must preserve `asset_id`, `window_start`, `window_end`, `source_types`, and a stable list of event-type labels. Any source references that may later map to a CR, WO, or RCA record should be carried forward even if not yet used.

**4. Metric-level transparency.** Jaccard, NLCS, and EMD scores must remain individually visible in the result object and must not be collapsed into a single combined score in the output artifact.

**5. Explicit index-status behavior.** When the index is empty or stale, the module must return an explicit `index_status` value rather than an empty result list. The downstream behavior rules in §4.11 depend on this field being populated accurately. An empty list is not an acceptable substitute.

**6. Optional integration under Step 2d.** If disabled, the existing workflow behaves exactly as it does now. If enabled, it emits `historical_signal_episodes.json` without affecting candidate generation, Step 4 scoring, or Step 5 refinement.

---

## 6. Concrete Implementation Backlog

The sequencing is critical: semantic recurrence hardening first, then signal-episode retrieval, then the cross-pattern bridge, then output surfacing.

### Phase 0 — Harden the current semantic recurrence path

#### `src/dackar/RCA/doc_extraction/schema.py`
- Add `event_time_start`, `event_time_end`, `event_time_confidence` (`"explicit"` / `"inferred"` / `"absent"`)
- Add source-reference fields: `source_cr_id`, `source_wo_id`, `source_event_id`
- Add FM-resolution fields: `fm_resolution_status`, `fm_resolution_score`, `fm_id_candidate_alt`
- Add helper: recurrence eligibility check (records with `fm_resolution_status == "ambiguous"` are ineligible without analyst promotion)

#### `src/dackar/RCA/doc_extraction/store.py`
- Add `exact_doc_ids` exclusion parameter to recurrence query method (mandatory double-counting guard)
- Fix FM resolution at threshold 0.88; route [0.80, 0.88) to `"ambiguous"` state
- Always return top-2 FM candidates with similarity scores
- Continue enforcing `EmbeddingModelVersionError` on model-version mismatch

#### `src/dackar/RCA/orchestrators/tskr_temporal_scorer.py`
- Maintain separate exact and semantic recurrence pools; pass `exact_doc_ids` to store query
- Implement and enforce semantic-recurrence tier cap; set `semantic_recurrence_capped` on pattern output
- Add `fm_resolution_ambiguous` to pattern output
- Add output fields: `effective_recurrence_count`, `semantic_match_count`, `near_match_count`, `near_match_pattern`, `exact_doc_ids_count`, `semantic_doc_ids_count`

#### `src/dackar/RCA/orchestrators/rca_reasoning_orchestrator.py`
- Keep `enable_semantic_recurrence = False` as default
- Propagate 0.88 FM-resolution threshold and `exact_doc_ids` dedup into scorer via `_apply_tskr_runtime_overrides()`
- Add `_apply_fm_resolution_ambiguity_flags()`: fires when any TSKR pattern has `fm_resolution_ambiguous = True`; called alongside `_apply_near_match_pattern_attention_flags()` at Step 6e
- Preserve current ranking behavior

---

### Phase 1 — Integrate `pattern_search` as Step 2d signal-episode retrieval

#### `src/dackar/RCA/pattern_search/config.py`
- Add or confirm feature-enable flag for signal-episode retrieval
- Add `index_staleness_window_days` parameter (default configurable)
- Define `PatternSearchConfig` dataclass; keep separate from `CrossPatternConfig`

#### `src/dackar/RCA/pattern_search/models.py`
- Define `HistoricalSignalEpisode` with all fields in §4.1 including `index_status`
- Ensure each result includes individual Jaccard, NLCS, EMD scores and provenance
- `index_status` must be `"indexed"` for an episode to be eligible for linkage

#### `src/dackar/RCA/pattern_search/indexer.py`
- Confirm persistent, reproducible `episode_id`
- Add index metadata: build timestamp, asset scope, episode count
- Keep index independent from KG and doc-extraction logic in Phase 1

#### `src/dackar/RCA/pattern_search/searcher.py`
- Return `list[HistoricalSignalEpisode]` with `index_status` populated on every result
- When index is empty or out of asset scope: return result with `index_status = "no_episodes_indexed"`, not an empty list
- When index is outside staleness window: return results with `index_status = "stale"`
- Keep Jaccard, NLCS, EMD individually visible per result

#### `src/dackar/RCA/orchestrators/rca_reasoning_orchestrator.py`
- Add dependency injection for pattern-search subsystem
- Add `enable_signal_episode_search` config flag
- Invoke pattern search in Step 2d when enabled; persist `historical_signal_episodes.json`
- Include `index_status` summary in `run_manifest.artifacts`
- Surface analyst flag when `index_status` is `"no_episodes_indexed"` or `"stale"`

---

### Phase 2 — Build the cross-pattern linkage layer

#### `src/dackar/RCA/cross_pattern/config.py`
Define `CrossPatternConfig` dataclass with:
- `temporal_compatibility_max_gap_days` (default 180)
- `temporal_compatibility_mode`: `"gate"` or `"formula"`
- `link_confidence_threshold`: minimum confidence for a link to appear in `evidence_paths`
- `fm_alignment_score_threshold`: minimum FM alignment score for level-3 linking
- `signal_similarity_floor`: minimum signal similarity for episode inclusion in linkage
- `stale_index_confidence_cap` (default 0.70): applied when `index_status == "stale"`

#### `src/dackar/RCA/cross_pattern/models.py`
- Implement `CrossPatternLink` with `linkage_precedence_level`, `temporal_link_skipped`, provenance
- Implement `CandidateCrossPatternEvidence` with `support_posture`, `reinforcement_strength`, `linkage_outcome`

#### `src/dackar/RCA/cross_pattern/rules.py`
- Implement three-level precedence evaluation
- Implement redundancy suppression for level-1 pairs
- Implement `temporal_link_skipped` assignment
- Implement episode-to-candidate mapping: link only to candidates with matching FM; build separate `CrossPatternLink` per qualifying pair
- Implement `reinforcement_strength` classification: `"single"`, `"multiple_consistent"`, `"mixed"` (§4.6)
- Implement `linkage_outcome` classification: `"linked"`, `"no_data"`, `"no_match"`, `"below_threshold"` (§4.7)
- Implement `support_posture` classification including `"weakly_supporting"` downgrade when `reinforcement_strength == "mixed"`
- Apply `stale_index_confidence_cap` when `index_status == "stale"`
- Apply `temporal_compatibility_max_gap_days` gate or formula per config mode
- Implement conflict conditions that trigger `review_required` flags

#### `src/dackar/RCA/cross_pattern/linker.py`
- Link `HistoricalSignalEpisode` objects to `HistoricalDocExtraction` records using `rules.py`
- Build `CrossPatternLink` objects with full precedence and provenance
- Build `CandidateCrossPatternEvidence`: `best_link_score` from highest-confidence path; all other paths in `evidence_paths`
- Respect cardinality rules: separate links per qualifying candidate-episode pair

#### `src/dackar/RCA/cross_pattern/summary.py`
- Summarize reinforcing and conflicting support per candidate
- Apply RCA card wording rules from §4.7 `linkage_outcome` table
- Generate run-manifest summaries with `linkage_outcome` distribution and candidate-level statuses
- Generate attention-flag triggers: `conflicting` posture on top-ranked candidates, `multiple_consistent` reinforcement, `"no_data"` and `"stale"` index conditions

#### `src/dackar/RCA/orchestrators/rca_reasoning_orchestrator.py`
- Add dependency injection for cross-pattern linker
- Add `enable_cross_pattern_linkage` feature flag
- Run linkage stage after Step 2d, before Step 5; apply §4.11 index-status decision rules before invoking linker
- Persist `cross_pattern_evidence.json`
- Pass artifact as auxiliary evidence only; must not enter scoring, hard-gate, or `score_rationale` logic

---

### Phase 3 — Surface the new artifacts in RCA outputs

#### Output schema files
- Define `cross_pattern_evidence.json` schema including `linkage_precedence_level`, `temporal_link_skipped`, `reinforcement_strength`, `linkage_outcome`
- Add `cross_pattern_summary` block to `run_manifest.artifacts`
- Add concise cross-pattern summary block to `rca_card` using wording rules from §4.7

#### `src/dackar/RCA/orchestrators/rca_reasoning_orchestrator.py`
- Inject cross-pattern summary into `rca_card`
- Add structured summary into `run_manifest.artifacts` including `linkage_outcome` distribution
- Apply attention flags: top-candidate conflicts, stale/missing index conditions
- Enforce that cross-pattern logic does not appear in `composite_score`, hard gates, or `score_rationale`

---

### Phase 4 — Testing and validation

#### Unit tests
- Double-counting exclusion via `exact_doc_ids`
- `semantic_recurrence_capped` tier-cap boundary behavior
- FM-resolution ambiguity routing and attention-flag generation
- Episode-to-candidate mapping: only FM-matching candidates linked; non-matching candidates produce no `CandidateCrossPatternEvidence`
- `reinforcement_strength` classification for `single`, `multiple_consistent`, `mixed` cases
- `linkage_outcome` assignment for all four values including `"no_data"` and `"below_threshold"`
- `linkage_outcome` wording in RCA card matches §4.7 table
- Linkage precedence and redundancy suppression
- `temporal_link_skipped` assignment when `event_time_confidence == "absent"`
- `stale_index_confidence_cap` applied to `link_confidence` when `index_status == "stale"`
- Normalization of `link_confidence` formula when one or more dimensions are missing

#### Integration tests
- Current event → historical signal episode retrieval with explicit `index_status`
- `"no_episodes_indexed"` → linkage not attempted; `linkage_outcome = "no_data"` in output
- `"stale"` → linkage attempted; `link_confidence` capped at 0.70; flag present in manifest
- Historical signal episode → document-extraction linkage with precedence tracing
- Candidate-facing cross-pattern evidence generation with `reinforcement_strength`
- RCA card and manifest surfacing of `linkage_outcome` wording and attention flags

#### Regression tests
- No candidate-ranking change when cross-pattern linkage is enabled in Phase 1–3 mode
- No hard-gate behavior change
- `review_required` flag fires for `conflicting` cross-pattern support on top-ranked candidate
- No semantic double-counting when `doc_id` overlap exists between exact and semantic pools
- `novel_pattern` semantics unchanged for failure modes with zero exact-match and zero semantic match
- TC-1 through TC-7: no change to ranking or gate outcomes with `enable_semantic_recurrence = True`

---

## 7. Final Recommendation

The semantic extraction pipeline should be implemented as the primary improvement to Step 3, with feature gating and strict data-consistency controls.

The existing `pattern_search` draft should be stabilized and integrated as an optional Step 2d producer of historical signal episodes satisfying the six expectations in §5, with `index_status` as a first-class field governing both linkage decisions and analyst communication.

Cross-pattern recognition should be implemented as a bridge artifact governed by `CrossPatternConfig`, strict precedence rules, explicit `linkage_outcome` classification, and `reinforcement_strength` for multiple-link cases. In Phase 1 it must remain non-intrusive to scoring. Phase 2 score influence may be considered only after calibration on labeled data.

---

## One-line Summary

Semantic extraction fixes documentary recurrence undercounting; `pattern_search` becomes the Step 2d signal-episode producer with explicit index-status governance; cross-pattern linkage is added as a separate evidence layer with precise null-case handling, episode-to-candidate mapping rules, and reinforcement-strength classification — before any attempt to influence RCA scoring.
