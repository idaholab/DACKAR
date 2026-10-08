# Causal Extraction Enhancement Plan

**Date:** 2026-05-08
**Last updated:** 2026-05-09
**Context:** DACKAR RCA pipeline — document pre-processing layer
**Status:** Stages 1–4 complete. Multi-dataset evaluation completed 2026-05-09 (DS2–DS5, 349 GT relations). Fixes 1–6 implemented 2026-05-09. Improvements A–F implemented 2026-05-09. Improvements G, H, I implemented 2026-05-09. Code review and schema fixes applied 2026-05-09. Improvement J (evaluation harness) and DS6a dataset in progress.

---

## 1. Where Causal Extraction Lives in the Full RCA Workflow

Causal extraction is a **document pre-processing step**, not part of the per-run orchestration (Steps 0–6). It runs once per document at ingestion time, producing records that the RCA workflow then queries.

```
INGESTION TIME (per document, runs once)
─────────────────────────────────────────────────────────────────────────────
 augment_chunks.py
  │
  ├─ [A] HybridNERPipeline.run()
  │       → mechanism_spans (G4_MECHANISM_PROCESS)
  │       → outcome_spans   (G5_FAILURE_OUTCOME)
  │
  ├─ [B] extract_stage5_causal_condition()   ← causal_condition_adapter.py
  │       → extracted_causal_statements[]
  │           {cause_text, effect_text, connector, confidence, negated, conjectural, source}
  │       → condition_state {as_found, as_left}
  │       → procedural_deviation {score}
  │
  ├─ [C] DocExtractionAdapter.extract()
  │       links [A] + [B]:
  │         cause_text ──[substring overlap]──► mechanism_spans → inferred_fm_label
  │         effect_text ────────────────────► identified_effect
  │         _assign_confidence(extractor_used, stmt_confidence, has_mechanism, ...)
  │           HIGH  : CausalSentence + gazetteer hit (≥0.85) + stmt_conf ≥ 0.60
  │           MEDIUM: CausalSentence or CausalSimple, mechanism present
  │           LOW   : dep_fallback / LLM_implicit / no mechanism match
  │       → DocExtractionRecord[] → upserted into DocExtractionStore (Chroma)
  │
  └─ [D] build_embedding_text()
          FM_LABELS: mechanisms + outcomes + causal cause/effect spans
          CAUSAL_STATEMENTS: "cause connector effect" lines
          → embedding_text → Chroma vector index

RCA RUN TIME (per-run, Steps 0–6)
─────────────────────────────────────────────────────────────────────────────
  Step 1  : DocExtractionStore.resolve_fm_candidates(fm_list_from_KG)
              embeds inferred_fm_label vs KG FM labels
              AUTO_RESOLVED if cosine ≥ 0.88; AMBIGUOUS if ≥ 0.80; else UNRESOLVED

  Step 2d : DocExtractionStore.query(fm_query_text)
              returns SemanticMatch[] with semantic_contribution:
                sim_score × confidence_weight × cause_is_symptom_factor
              these drive effective_recurrence_count for candidate ranking

  Step 5  : evidence_retriever queries Chroma text corpus
              has_explicit_causal_statement / has_condition_state flags used in
              evidence posture scoring (documentary stream)
```

### What drives record quality downstream

Every quality improvement in causal extraction flows through this chain:

```
Better cause/effect spans
  → higher overlap with G4 NER entities (has_mechanism = True)
  → _assign_confidence → MEDIUM or HIGH  (instead of LOW)
  → embed_text quality: identified_effect | assessed_cause | inferred_fm_label
  → fm_resolution_score at Step 1 (cosine vs KG FM labels)
  → SemanticMatch.semantic_contribution at Step 2d
  → effective_recurrence_count → candidate ranking at Step 5
```

---

## 2. Current State After May 8 Fixes

| Tier | extractor_used value | Confidence assigned by adapter | Gap |
|------|----------------------|-------------------------------|-----|
| CausalSentence | `"CausalSentence"` | HIGH (if gaz hit + conf≥0.60) or MEDIUM | Requires SSC entity patterns; without them always returns empty |
| CausalSimple | `"CausalSimple"` | MEDIUM (if mechanism present) | Same dependency |
| dep_fallback | `"CausalSentence"` (inherits) | LOW (no mechanism overlap) | Span quality too low for substring overlap to fire; confidence mis-labelled |
| LLM_implicit | `"LLM_implicit"` | LOW | Fires only on complete miss |

**Immediate gap:** dep_fallback statements inherit `extractor_used = "CausalSentence"` at the stage5 level but their raw string cause/effect spans rarely overlap with G4 NER entities via substring match → `has_mechanism = False` → `ConfidenceLevel.LOW`. Good dep_fallback extractions are therefore silently downgraded.

---

## 3. Enhancement Stages

---

### Stage 1 — Noun-chunk span extraction ✓ COMPLETED

**Scope:** `_head_phrase()` in `causal_condition_adapter.py`

**Problem:** The dep-tree fallback uses the subtree of the first dep-matched child token. Compound noun phrases headed by a different token are truncated. `"Bearing wear"` may come back as just `"wear"`; `"actuator spring fatigue"` may come back as `"fatigue"`. Truncated spans are less likely to overlap with the G4 entity text that the adapter's `_best_overlapping_entity` needs to produce a non-null `inferred_fm_label`.

**Fix:** Use spaCy's `doc.noun_chunks` — contiguous noun phrases with linguistically valid boundaries — as the preferred span source, with the current subtree walk as fallback.

```python
def _head_phrase(head_tok, target_deps):
    for child in head_tok.children:
        if child.dep_ in target_deps:
            for chunk in child.doc.noun_chunks:
                if chunk.start <= child.i < chunk.end:
                    return chunk.text          # full NP boundary
            # fallback: subtree (current behaviour)
            return " ".join(t.text for t in child.subtree if not t.is_punct)
    return None
```

**Pipeline impact:**
- Better `cause_text` / `effect_text` in `extracted_causal_statements`
- Better `FM_LABELS` and `CAUSAL_STATEMENTS` lines in `embedding_text`
- Higher overlap rate in `_best_overlapping_entity` → `has_mechanism = True` more often → MEDIUM confidence instead of LOW

**Testing:** Create a labelled sentence dataset (20–30 WO/CR sentences with known cause/effect spans) to measure span quality before and after. This dataset will also serve as the threshold calibration set for Stage 2.

**Effort:** Small — single function, no new dependencies.

**Implementation notes:**
- `_head_phrase` now tries `child.doc.noun_chunks` first; falls back to subtree walk if the token is not inside any chunk or the model has no noun-chunker.
- Wrapped in `try/except` so models without a `parser` component degrade safely.

---

### Stage 2 — Multi-hop causal chain detection ✓ COMPLETED

**Scope:** New post-processor `_chain_causal_statements(stmts)` in `causal_condition_adapter.py`; new `causal_chain` key in `STAGE5_OUTPUT_SCHEMA`; read in `DocExtractionAdapter`

**Problem:** RCA narratives describe propagation paths: *"Bearing wear caused vibration, which led to seal failure, resulting in coolant leakage."* The current extractor returns three independent statements with no structural link. The adapter assigns `cause_is_symptom` per individual statement via `_text_overlaps_any(cause_text, outcome_spans)`, which cannot detect that `"vibration"` is an intermediate node — it fires only if the text overlaps a G5 outcome span, which is unreliable for intermediate nodes.

With a chain structure, the adapter can:
1. Assign `cause_is_symptom = True` to any node that is not the chain root
2. Use the chain root as `assessed_cause` (proximate cause) and chain start as root cause
3. Write the full chain into `embed_text` for richer FM resolution

**Design:**

```
_chain_causal_statements(stmts) → List[Dict]

Algorithm:
  1. Build directed graph: node_text → set of successor node_texts
     Match criterion: Jaccard token overlap ≥ threshold T
     (T will be calibrated on the Stage 1 test dataset)
  2. Find all source nodes (no incoming edge)
  3. DFS from each source → collect all paths
  4. Break cycles at first repeated node
  5. Return chains sorted by length descending

Output added to stage5 result:
  "causal_chain": [
    {
      "chain_id": "DOC::0::chain::0",
      "nodes": ["Bearing wear", "excessive vibration", "seal failure", "coolant leakage"],
      "length": 4,
      "min_confidence": 0.72,
      "source_statement_ids": ["DOC::0::dep::0", "DOC::0::dep::1", "DOC::0::prep::0"]
    }
  ]
```

**DocExtractionAdapter change:** When a causal_chain is present with length ≥ 2:
- `assessed_cause` ← chain node at position [-2] (proximate cause, one step before the terminal failure)
- `identified_effect` ← chain node at position [-1] (terminal failure)
- `cause_is_symptom` flag on each non-root node (intermediate propagation steps)
- `inferred_fm_label` ← chain root node matched against G4 entities (root cause)

**Testing:** Use the Stage 1 sentence dataset extended with chained sentences. Measure: chain reconstruction accuracy, and whether `cause_is_symptom` flag is correctly set on intermediate nodes.

**Effort:** Medium. Graph construction is straightforward; the Jaccard threshold T needs calibration.

**Dependency:** Stage 1 first (better node text quality → higher chain linking accuracy).

**Implementation notes:**
- `_chain_causal_statements(stmts)` added to `causal_condition_adapter.py`; `_CHAIN_JACCARD_THRESHOLD = 0.35` (conservative; pending calibration on test dataset).
- `"causal_chain": []` added to `empty_stage5_output()`; chain building wired into all three extractor paths (CausalSentence, CausalSimple, LLM) before each return.
- `DocExtractionAdapter.extract()` reads `causal_chain` first: one record per chain (root node → `inferred_fm_label`, `nodes[-2]` → `assessed_cause`, `nodes[-1]` → `identified_effect`, `cause_is_symptom = len(nodes) > 2`). Statements not covered by any chain fall through to the existing per-statement loop.

---

### Stage 3 — Entity linking at the adapter layer ✓ COMPLETED

**Scope:** `DocExtractionAdapter.extract()` in `doc_extraction/adapter.py`

**Problem:** `_best_overlapping_entity(cause_text, mechanism_spans)` uses simple substring containment. It works when `cause_text == "bearing wear"` and the G4 entity is `"adhesive wear"` or `"bearing"` — only if one is a substring of the other. It fails for paraphrase distances (e.g., `cause_text = "corrosion of the pump impeller"` vs. G4 entity `"corrosion"`).

**Where this lives:** Entity linking belongs in `DocExtractionAdapter` (step C), not in `causal_condition_adapter`. The adapter already has both artifacts — `mechanism_spans` from step A and `causal_statements` from step B — so the link can be resolved there without coupling the two extractors.

**Design:**

```python
def _best_overlapping_entity_v2(target_text, spans):
    """
    Priority chain:
    1. Exact substring match (current behaviour) — free, O(n)
    2. Token-set Jaccard ≥ 0.4 across whitespace-split tokens
    3. Lemma match: lemmatize both sides, re-apply (1) and (2)
       (requires spaCy nlp — pass through existing self._nlp)
    Returns (entity_text, match_method) or (None, None)
    """
```

**Pipeline impact:**
- `inferred_fm_label` populated more reliably → better FM resolution scores
- `has_mechanism = True` more often → MEDIUM/HIGH confidence more often
- `cause_is_symptom` detection more accurate

**Side effect on `_assign_confidence`:** Since `dep_fallback` statements go into the stage5 result with `extractor_used = "CausalSentence"` (from the outer stage5 path), the confidence rule already fires correctly once `has_mechanism` is True. No change needed to `_assign_confidence` for this stage.

However: dep_fallback as an extractor_used value should eventually be surfaced distinctly. For now, note this as a known labelling approximation.

**Testing:** Same dataset. Measure: `inferred_fm_label` hit rate before and after.

**Effort:** Medium. Lemma matching requires spaCy nlp (already injected into `DocExtractionAdapter` via `self._nlp`); Jaccard is pure Python. Main risk is false positive matches — the priority chain and threshold need tuning.

**Dependency:** Stage 1 (better cause_text → better Jaccard overlap). Can proceed in parallel with Stage 2.

**Implementation notes:**
- `_best_overlapping_entity` signature extended with `nlp=None`; function name kept unchanged for compatibility.
- `_ENTITY_LINK_JACCARD_THRESHOLD = 0.4` added as a module-level constant.
- Priority chain: exact substring → token-set Jaccard ≥ 0.4 → spaCy lemma (substring on lemma strings, then Jaccard on lemma sets); pass 3 skipped when `nlp=None`.
- All four call sites in `adapter.py` updated to pass `nlp=self._nlp`.

---

### Stage 4 — Targeted LLM repair of incomplete statements ✓ COMPLETED

**Scope:** New `_llm_repair_weak_statements()` in `causal_condition_adapter.py`; called from `extract_stage5_causal_condition` after dep_fallback

**Problem:** After Stage 1–3, some statements will still have one empty side (`cause_text` or `effect_text`) or a `confidence < 0.60`. These are precisely the cases where a small, fast LLM call adds most value — it has the sentence context and can fill the missing field. The current LLM tier fires only on a complete extraction miss (`extracted_causal_statements == []`), so these partial records never see the LLM.

The adapter's `_assign_confidence` gives LOW to `LLM_implicit` records. This is correct for whole-document implicit extraction. But a targeted repair that fills a missing `cause_text` field, confirmed against the sentence, should be rated higher. A repaired dep_fallback statement with both sides filled and mechanism overlap could reach MEDIUM confidence.

**Trigger condition (added after dep_fallback step):**
```python
weak = [s for s in stmts if
        not s["cause_text"] or not s["effect_text"] or s["confidence"] < 0.60]
if weak and llm_cfg and llm_cfg.get("enabled"):
    stmts = _llm_repair_weak_statements(stmts, weak, chunk_text, llm_cfg)
```

**Prompt design:** Send only the weak statement and its source sentence. Do NOT re-extract everything.
```
Sentence: "<sentence_text>"
Partial extraction:
  cause: "<cause_text or UNKNOWN>"
  connector: "<connector>"
  effect: "<effect_text or UNKNOWN>"
Fill the UNKNOWN field using only text from the sentence.
Return JSON: {"cause_text": "...", "effect_text": "..."}
```

**Source tagging:** Repaired statements get `source = "dep_fallback+llm_repair"`.

**LLM cost control:** Per-doc LLM call budget can be set via `llm_cfg["max_repair_calls"]` (default 3). Calls are ordered by priority: empty sides first, then lowest confidence.

**Acceptance criteria:**
- Statements with `cause_text == ""` before repair have a non-empty value after
- Statements with `confidence ≥ 0.60` and both sides non-empty are NOT sent to LLM
- `max_repair_calls` is respected per document

**Effort:** Medium. Prompt engineering and graceful degradation on bad JSON are the main work items.

**Dependency:** Stages 1–3 complete so the LLM operates on best-quality spans and entity context.

**Implementation notes:**
- `_llm_repair_weak_statements(stmts, weak, chunk_text, llm_cfg)` added to `causal_condition_adapter.py`.
- Trigger wired into both the CausalSentence path (after dep_fallback) and the CausalSimple path, before chain building. Completely dormant when `llm_cfg` is absent or `llm_cfg["enabled"]` is falsy.
- Priority order: both sides missing → one side missing → lowest confidence. Budget gate: `llm_cfg["max_repair_calls"]` (default 3).
- Repaired statements tagged `source = "dep_fallback+llm_repair"` and re-scored via `_score_causal_statement`.
- Verified in notebook: DOC_EXPLICIT_CAUSAL first statement repaired to `cause='bearing wear' → effect='excessive vibration'` with `conf=0.95`.
- **Bonus fix (same session):** prep-connector direction bug in `_dep_causal_fallback` pass 2 — forward connectors (`led to`, `leading to`, `resulting in`) now correctly assign `cause=before_text`, `effect=after_text`; backward connectors (`due to`, `caused by`, etc.) keep the prior assignment.

---

## 4. Stage Sequencing and Dependencies

```
Stage 1 — noun-chunk spans   [DONE] unblocks all others
    │
    ├──→ Stage 2 — causal chains      [DONE] improves cause_is_symptom + chain position
    │
    └──→ Stage 3 — entity linking     [DONE] improves has_mechanism + inferred_fm_label
                │
                └──→ Stage 4 — LLM repair   [DONE] rescues remaining LOW-confidence records
                            │
                            ├──→ Fix 1 — all causal tokens / sentence  [DONE]
                            ├──→ Fix 2 — sent_index + cross_sentence    [DONE]
                            ├──→ Fix 3 — passive + participial          [DONE]
                            ├──→ Fix 4 — full subtree NP spans          [DONE]
                            ├──→ Fix 5 — annotation offset correction   [DONE]
                            └──→ Fix 6 — LLM extract_all supplement     [DONE]
```

All stages and fixes completed 2026-05-09.

---

## 5. Answers to Open Questions

**Q1 — Fuzzy match threshold calibration:**
Build a labelled dataset of 30–50 WO/CR sentences with:
- annotated cause/effect spans (ground truth)
- the expected G4 mechanism entity from the gazetteer that should match the cause span

Use this dataset to tune: the Jaccard threshold in Stage 2 chain linking (currently guessed at 0.6), and the Jaccard threshold in Stage 3 entity linking (guessed at 0.4). The same dataset validates Stage 1 span quality.

**Q2 — NERSeed availability at causal extraction time:**
Entity linking (Stage 3) belongs in `DocExtractionAdapter.extract()`, not in `causal_condition_adapter`. The adapter already runs NER in step A before calling the causal extractor in step B, so `mechanism_spans` and `outcome_spans` are available for the linking step (C) without any structural change. The causal extractor stays decoupled from NERSeed.

**Q3 — LLM cost control:**
The `_assign_confidence` function already penalises `LLM_implicit` records with `ConfidenceLevel.LOW`. Stage 4 introduces a separate `llm_cfg["max_repair_calls"]` budget per document (default 3) and a `confidence ≥ 0.60 + both sides non-empty` bypass gate. This limits LLM invocations to genuinely weak records and caps total calls per document regardless of chunk count.

**Q4 — Temporal ordering validation:**
`DocExtractionRecord` has `event_time_start`, `event_time_end`, and `event_time_confidence` fields. These are currently unpopulated by the causal extractor. Stage 2 chain construction should optionally order chain nodes using `temporal_refs` from `NERSeed` when available (spaCy annotator already extracts `lag_hours`). This is not in the Stage 2 MVP but is the natural extension once chains are stable.

---

## 6. Files Affected

| Stage | File | Change | Status |
|-------|------|--------|--------|
| 1 | `ner/causal_condition_adapter.py` | Replace `_head_phrase` body with noun-chunk + subtree fallback | ✓ |
| 2 | `ner/causal_condition_adapter.py` | Add `_chain_causal_statements`, `_CHAIN_JACCARD_THRESHOLD`, `causal_chain` key in output; wire into all three extractor paths | ✓ |
| 2 | `doc_extraction/adapter.py` | Read `causal_chain`; produce chain-based records first, then uncovered-statement records | ✓ |
| 3 | `doc_extraction/adapter.py` | Replace `_best_overlapping_entity` with v2 (substring → Jaccard → lemma); add `_ENTITY_LINK_JACCARD_THRESHOLD`; update all call sites | ✓ |
| 4 | `ner/causal_condition_adapter.py` | Add `_llm_repair_weak_statements`; wire repair trigger into CausalSentence and CausalSimple paths | ✓ |
| Bonus | `ner/causal_condition_adapter.py` | Fix prep-connector direction bug in `_dep_causal_fallback` pass 2 | ✓ |
| All | `RCA/tests/test_NER_1/test_NER.ipynb` | Verified passing after all stages | ✓ |
| Eval | `RCA/tests/test_causal/test_causal_extraction.ipynb` | Multi-dataset evaluation notebook (Sections 1–7) | ✓ |
| Eval | `RCA/tests/test_causal/data/causal_dataset.json` | 25 entries; added `expected_g4_mechanism`, `chain`, `connector_direction` fields | ✓ |
| Eval | `RCA/tests/test_causal/data/causal_dataset_2.json` | 25 entries; secondary_effect; missing direction/g4 annotations | — |
| Eval | `RCA/tests/test_causal/data/causal_dataset_3.json` | 27 entries; multi-relation single text; offsets unreliable (span text correct) | — |
| Eval | `RCA/tests/test_causal/data/causal_dataset_4.json` | 28 entries; multi-sentence cross-sentence chains; offsets unreliable | — |
| Fix 1 | `ner/causal_condition_adapter.py` | Multi-causal token extraction per sentence: `_find_all_causal_tokens`, `_dedup_span_overlap`; rewrote Pass 1 loop | ✓ |
| Fix 2 | `ner/causal_condition_adapter.py` | `sent_index` in statement dicts + IDs; `cross_sentence` flag in chain dicts | ✓ |
| Fix 3 | `ner/causal_condition_adapter.py` | Passive agent (3a), participial subject inference `_infer_participial_cause` (3b), cross-statement inheritance (3c) | ✓ |
| Fix 4 | `ner/causal_condition_adapter.py` | `_np_subtree_text` replaces noun_chunk path in `_head_phrase`; clause filtering via `_CLAUSE_DEPS_NP` | ✓ |
| Fix 5 | `data/causal_dataset_4.json`, notebook `cell-load` | Corrected 4 bad span annotations; `_resolve_span_offset` in all loaders; `unresolved_spans` column in summary | ✓ |
| Fix 6 | `ner/causal_condition_adapter.py`, `doc_extraction/adapter.py` | `_llm_extract_all_relations`, `_merge_llm_statements`, `_apply_llm_extract_all`; wired at all three return points; `LLM_extract_all` → LOW confidence | ✓ |

---

## 7. Multi-Dataset Evaluation Results (2026-05-09)

Five datasets evaluated in `test_causal_extraction.ipynb` after all A–F improvements:

| Dataset | Entries | GT relations | Format | Recall |
|---------|---------|-------------|--------|--------|
| ds1 | 25 | 25 | Single sentence; `chain`, `connector_direction`, `expected_g4_mechanism` annotated | — |
| ds2 | 25 | 25 | Single sentence + `secondary_effect`; direction/G4 not annotated | 12% |
| ds3 | 27 | 102 | Long single text, multiple explicit causal relations per entry | 11% |
| ds4 | 28 | 143 | Multi-sentence, 5 sentences, cross-sentence relations | 23% |
| ds5 | 30 | 79 | Adversarial cross-sentence, 12 challenge types | 5% |

**Total (DS2–DS5): 110 entries, 349 GT relations.**
*(DS1 file currently missing from `data/` — silently skipped; see Improvement J.)*

### DS5 structure (added 2026-05-09)

DS5 is a deliberately adversarial dataset of 30 records, 72 annotated relations, 40/72 cross-sentence. All records sourced from condition reports. Twelve challenge types (2–3 records each):

| Challenge type | Recall | TP/Total | Why hard |
|----------------|--------|----------|----------|
| confounded_causality | 20% | 1/5 | Shared connector; one of two candidate causes selected |
| recursive_feedback | 18% | 2/11 | Circular reference; dep tree catches some explicit links |
| passive_nominalisation | 14% | 1/7 | NP form of causal verb; no active connector |
| reversed_causal_order | 0% | 0/7 | Effect stated first; cause in subordinate "because" clause |
| negated_causality | 0% | 0/9 | Relation negated; dep fallback extracts but negation scope is sentence-wide |
| multi_cause_convergence | 0% | 0/10 | `cause_set` structure; multiple causes share one effect |
| operator_action_as_cause | 0% | 0/11 | Cross-sentence demonstrative reference ("This caused…") |
| implicit_no_connective | 0% | 0/4 | No lexical connector; inference required |
| latent_gap_in_chain | 0% | 0/4 | Intermediate node not stated; requires domain knowledge |
| counterfactual_conditional | 0% | 0/7 | Relation inside counterfactual ("would not have…") |
| reporting_verb_ambiguity | 0% | 0/4 | Causal verb is a reporting verb ("noted", "indicated") |
| cross_paragraph | — | — | Not present in this evaluation pass |

**DS5 by relation type:**
- explicit: 7% (4/56)
- implicit: 0% (0/20)
- ambiguous: 0% (0/3)

100% of all detected relations across DS2–DS5 came from `_dep_causal_fallback`. `CausalSentence` and `CausalSimple` require SSC entity patterns and returned empty on all raw-text inputs.

### Key metrics (DS2–DS5 combined)

| Metric | Value | Notes |
|--------|-------|-------|
| Overall recall | **15%** | 51/349 GT relations detected with ≥40% cause+effect overlap |
| DS4 recall | **23%** | Best dataset; context window + explicit connectors in adjacent sentences |
| DS5 recall | **5%** | Adversarial; structural ceiling for dep-tree approach |
| Mean cause F1 | **0.12** | Cause span quality reasonable when extractor fires |
| Mean effect F1 | **0.18** | Higher than cause F1; many causes returned empty |
| Implicit recall | **0%** | 0/20 implicit DS5 relations; structurally unreachable without LLM |
| Direction accuracy | **53%** (DS1) | Many "unknown" cases — 0 Jaccard on both sides, not inversion |
| Entity linking hit-rate | **80%** (DS1) | All 8 via substring pass-1; 2 misses (`pitting`, `oxidation`) |

### Root cause analysis of low performance

**A — Single-statement extraction limit (DS3)**
Each entry has 2–6 GT relations in one text. After Fix 1, multiple causal tokens are extracted per sentence, but recall still lags because many relations span clauses the dep tree separates.

**B — No cross-sentence linking (DS4, DS5)**
The dep fallback processes one chunk; each sentence produces independent statements. `_chain_causal_statements` rarely fires because cross-sentence cause/effect spans use different vocabulary. DS5 cross-sentence recall (5%) is the floor for the current architecture.

**C — Implicit and reversed-order relations (DS5 structural ceiling)**
54% of DS5 relations (implicit: 20, reversed: 7, counterfactual: 7) require capabilities the dep tree cannot provide: domain-knowledge inference, subordinate-clause inversion, and negation scoping. These are not addressable by further dep-tree refinement — only LLM extraction can recover them.

**D — Cross-sentence demonstrative references (DS5)**
Sentences starting with "This caused…" / "The above resulted in…" have a pronoun as the grammatical cause. The dep fallback collects `cause_text = "This"` — no semantic content, no entity link. Affects operator_action_as_cause (11 relations), recursive_feedback (11 relations), latent_gap (4 relations).

**E — Empty effect spans**
The dep fallback returns `effect_text = ''` when the effect is under `xcomp` ("caused the pump **to fail**") or `pcomp` ("led to the system **shutting down**"). Empty effects count as recall=0 and block entity linking → `ConfidenceLevel.LOW`.

**F — Annotation issues in DS3/DS4** *(historical; resolved by Fix 5)*
Character offsets were inaccurate; resolved by `_resolve_span_offset`. DS4 had 4 annotation word-order mismatches; corrected in source file.

---

## 8. Next Improvement Stages

### Fix 1 — Multi-causal token extraction per sentence ✓ COMPLETED

**Scope:** `_dep_causal_fallback` in `causal_condition_adapter.py`

**Problem:** The function calls `_find_causal_token(sent)` and extracts exactly one statement per sentence. DS3 entries have 2–6 GT causal relations in a single sentence; all but the first are silently dropped.

**Fix:** Replace the single-token search with a loop that collects *all* causal tokens in the sentence:

```python
def _dep_causal_fallback(doc, chunk_text, doc_id, chunk_index):
    statements = []
    for sent in doc.sents:
        for tok in sent:
            if tok.lower_ in CAUSAL_TOKENS or tok.lemma_.lower() in CAUSAL_LEMMAS:
                stmt = _build_stmt_from_causal_tok(tok, sent, ...)
                if stmt:
                    statements.append(stmt)
    return statements
```

**Expected impact:** DS3 extraction coverage from ~25% to 60–70% of GT relations; DS4 per-sentence coverage also improves.

**Effort:** Small–medium. Main risk: duplicate overlapping statements when two causal tokens are very close (e.g., "resulting in ... leading to ..."). De-duplicate by Jaccard overlap of extracted spans.

**Implementation notes:**
- Added `_find_all_causal_tokens(sent)` returning all causal-verb tokens in a sentence.
- Added `_dedup_span_overlap(stmts, threshold=0.8)` — removes near-duplicate statements from the same sentence by Jaccard on cause+effect spans; keeps the higher-confidence one.
- Pass 1 loop in `_dep_causal_fallback` rewrote to iterate over `_find_all_causal_tokens(sent)` per sentence; per-sentence dedup applied before extending global list.

---

### Fix 2 — Sentence-level segmentation and statement merge ✓ COMPLETED

**Scope:** `extract_stage5_causal_condition` in `causal_condition_adapter.py`

**Problem:** DS4 texts are 5 sentences joined with spaces. A single extraction pass can only see within-sentence causal tokens. Cross-sentence relations (WO-039: sentence 2 → sentence 1) are structurally invisible. `_chain_causal_statements` never has ≥2 statements with overlapping spans because statements from different sentences refer to different entities.

**Fix:** When `nlp` is available, split the input text into sentences via `doc.sents`, run `_dep_causal_fallback` on each sentence separately, merge the resulting statement lists, then run `_chain_causal_statements` across the merged list.

```
extract_stage5_causal_condition(text, nlp, ...)
  ├─ [existing] try CausalSentence / CausalSimple extractors
  ├─ [new] if dep_fallback needed AND nlp AND len(sents) > 1:
  │       for sent in doc.sents:
  │           stmts += _dep_causal_fallback(sent_doc, ...)
  │       _chain_causal_statements(merged_stmts)
  └─ [existing] LLM fallback on total miss
```

**Expected impact:** Chain detection rate from 0% to meaningful for DS4 multi-sentence entries. Validates Stage 2 chain detection properly for the first time.

**Effort:** Medium. Need to pass sentence-level `doc` slices to `_dep_causal_fallback`. Statement IDs must encode `sent_index` to remain unique.

**Dependency on Fix 1:** Fix 2 feeds multiple statements per sentence into the chain builder; Fix 1 ensures each sentence already produces multiple statements. Together they unlock Stage 2 for both DS3 and DS4.

**Implementation notes:**
- Pre-computed `sent_bounds = [(i, s.start_char, s.end_char) for i, s in enumerate(doc.sents)]` at top of `_dep_causal_fallback`.
- Each statement dict carries `"sent_index": sent_idx`; statement IDs encode `s{sent_idx}` (e.g. `doc::0::s2::dep::1`).
- Pass 2 (regex) maps match position to sentence index via `sent_bounds`.
- `_chain_causal_statements` extended with `"cross_sentence": bool` flag in each chain dict.

---

### Fix 3 — Passive voice handling ✓ COMPLETED

**Scope:** `_dep_causal_fallback` in `causal_condition_adapter.py`

**Problem:** Passive constructions ("was caused by", "was attributed to", "was linked to") parse with `nsubjpass` (effect subject) and `agent`/`pobj` (cause). These patterns are not covered by the current nsubj/dobj subject-object extraction.

**Fix:** Add a dep-tree pass for `nsubjpass + prep(by/to/from) + pobj` pattern after the existing nsubj/dobj pass. Effect = nsubjpass head phrase; Cause = pobj head phrase.

**Implementation notes:**
- **3a (passive agent):** `_extract_passive_agent(causal_tok)` looks for `agent` dep child; returns `pobj` phrase as true cause; `nsubjpass` phrase becomes the effect.
- **3b (participial subject inference):** `_infer_participial_cause(causal_tok)` walks up dep tree (up to 3 hops) when `causal_tok.dep_` is in `_PARTICIPIAL_DEPS`; returns `nsubj/nsubjpass` NP of the nearest ancestor verb. Fixes "…triggering Y" → cause = head-clause subject.
- **3c (cross-statement inheritance):** When 3b finds nothing, borrows the most recent `effect_text` from earlier same-sentence statements as the implicit cause.
- Added module constant `_PARTICIPIAL_DEPS = frozenset({"advcl", "relcl", "acl", "partmod", "xcomp"})`.

---

### Fix 4 — Full subtree cause span including prep children ✓ COMPLETED

**Scope:** `_head_phrase` in `causal_condition_adapter.py`

**Problem:** Cause spans like "Erosion of the turbine blade leading edges" have head "Erosion" with right-attached `prep` child "of the turbine blade leading edges". The current noun-chunk captures only "Erosion"; the subtree walk gets truncated by the guard against punctuation.

**Fix:** After noun-chunk lookup, extend the span right-ward to include immediately-attached `prep` subtrees up to a configurable max token length.

**Implementation notes:**
- Added `_CLAUSE_DEPS_NP = frozenset({"relcl", "acl", "advcl", "ccomp", "rcmod"})` — clausal deps excluded from NP subtrees.
- Added `_np_subtree_text(tok)` — walks `tok.subtree`, blocks sub-trees whose root is in `_CLAUSE_DEPS_NP`, returns joined token text. Captures "Erosion of the turbine blade leading edges" while stopping before "which failed last year".
- `_head_phrase` rewrote to drop the noun_chunk path entirely; `_np_subtree_text` is now the primary (and only) span extraction path.
- `_infer_participial_cause` updated to use `_np_subtree_text` instead of `noun_chunks` for the head-clause nsubj, so Fix 3b also benefits.

---

### Fix 5 — Re-derive DS3/DS4 annotation offsets ✓ COMPLETED

**Scope:** `causal_dataset_3.json`, `causal_dataset_4.json`

**Problem:** `start`/`end` (DS3) and `local_start`/`local_end` (DS4) offsets are inaccurate by inconsistent amounts. One DS4 span has a word-order mismatch (WO-040-R1). The evaluation notebook currently ignores all offsets and uses substring search.

**Fix:** Script that re-derives offsets via `text.find(span)` for every relation in DS3 and DS4, writes corrected JSON files, and flags any span not found in the text for manual review.

**Implementation notes:**
- Added `_resolve_span_offset(text, span)` helper in `cell-load` of the test notebook; returns `(start, end)` or `(-1, -1)` if not found.
- `_load_ds3` and `_load_ds4` now populate `cause_start/cause_end/effect_start/effect_end` in every relation dict via substring search, replacing DS3's systematic off-by-2 offsets and DS4's unreliable `local_start/end`.
- Dataset summary table gains `unresolved_spans` column; any remaining annotation mismatches surface at load time.
- Fixed 4 word-order/paraphrase mismatches directly in `causal_dataset_4.json`: WO-040 R1 cause, CR-043 R2 effect, CR-044 R3 effect, CR-053 R3 effect; all spans now locatable in source text.

---

### Fix 6 — LLM multi-relation extraction mode ✓ COMPLETED

**Scope:** `extract_stage5_causal_condition` in `causal_condition_adapter.py`

**Problem:** For DS3 entries with 4–6 relations, dep_fallback (even after Fix 1) will still miss some. The Stage 4 LLM repair only fixes *weak* existing statements; it doesn't add *missing* ones.

**Fix:** Add `llm_cfg["extract_all"] = True` mode: after dep_fallback, if `n_extracted < llm_cfg.get("min_expected_relations", 2)`, prompt the LLM to return the complete list of all causal relations from the text. Tag these statements `source = "LLM_extract_all"`.

**Implementation notes:**
- Added `_llm_extract_all_relations(...)` — prompt covers both explicit and implicit relations; instructs LLM to report multi-hop chains as individual (A,B) and (B,C) links. Tagged `source="LLM_extract_all"`.
- Added `_merge_llm_statements(existing, llm_stmts, threshold=0.6)` — appends only novel LLM relations not already covered by rule-based statements (Jaccard on both cause and effect >= threshold).
- Added `_apply_llm_extract_all(result, ...)` — in-place supplement: calls `_llm_extract_all_relations`, merges novel statements, re-chains, updates summary flags. No-ops when `llm_cfg["enabled"]` or `llm_cfg["extract_all"]` are falsy.
- Wired `_apply_llm_extract_all` at both CausalSentence and CausalSimple `return` points.
- Complete-miss LLM fallback: when both rule-based paths return empty, uses `_llm_extract_all_relations` (instead of `_llm_causal_fallback`) when `extract_all=True`; sets `extractor_used="LLM_extract_all"`, `status="llm_extract_all"`.
- `_assign_confidence` in `doc_extraction/adapter.py` updated: `LLM_extract_all` → `ConfidenceLevel.LOW` (same as `LLM_implicit`; supplement statements inherit the rule-based extractor's confidence path instead).

---

## 9. Remaining Improvements (Post-Fix 6)

The following improvements are identified after the completion of all six fixes. They are ordered by estimated impact/effort ratio.

---

### Improvement A — Vocabulary alignment with existing project data files ✓

**Status: Implemented 2026-05-09** — `_load_causal_verb_lemmas()` reads from `data/cause_effect_keywords_full.csv`; `_build_conjecture_pattern()` reads from `data/conjecture_keywords.csv`; `_FORWARD_PREP_CONNECTORS` frozenset replaces brittle `startswith()` direction detection in Pass 2; `_PREP_CONNECTOR_PAT` extended with 12 new patterns from the keyword files.

**Priority: High | Effort: Small**

**Scope:** `_DEP_CAUSAL_VERB_LEMMAS`, `_PREP_CONNECTOR_PAT`, `_CONJECTURE_PAT`, `_infer_condition_from_text` in `causal_condition_adapter.py`

**Problem:** The adapter defines three vocabulary constants independently of the project's existing curated keyword files in `data/`. These files are already used by `ConjectureEntity`, `CausalSentence`, and `CausalSimple` — but `causal_condition_adapter.py` does not reference them, creating vocabulary drift and missed coverage.

**Gap analysis (computed 2026-05-09):**

| Constant | Current count | Source file | Available count | Gap |
|----------|--------------|-------------|-----------------|-----|
| `_DEP_CAUSAL_VERB_LEMMAS` | 11 lemmas | `data/cause_effect_keywords_full.csv` VERB column | 79 verbs | **68 missing** |
| `_PREP_CONNECTOR_PAT` | 11 patterns | `data/cause_effect_keywords.csv` causal-relator + effect-relator columns | ~40 patterns | ~29 missing |
| `_CONJECTURE_PAT` | ~13 terms | `data/conjecture_keywords.csv` | 20 terms | **13 missing**: expected, feasible, plausible, hypothetical, hypothetically, uncertain, anticipated, foreseen, impending, upcoming, brewing, looming, forthcoming |
| `_infer_condition_from_text` | ~30 hardcoded terms | `data/health_status_keywords.csv` | 80+ terms | partial coverage |

**Notable missing causal verb lemmas** (from `cause_effect_keywords_full.csv`):
`accelerate, activate, actuate, affect, alter, damage, facilitate, generate, impede, impinge, increase, influence, initiate, originate, perturb, precipitate, prevent, promote, prompt, propagate, provoke, render, spark, stimulate, suppress, transform, upset`

**Notable missing prep connectors** (from `cause_effect_keywords.csv`):
`in consequence of, as a consequence of, give rise to, stemming from, arising from, triggered off by, in response to, responsible for, attributed to` (multi-word), `sparked by, prompted by, initiated by, determined by, owing to` (beyond current coverage)

**Fix:** Load vocabulary at import time from the data files via the same `nlpConfig` / `CreatePatterns` path that `ConjectureEntity` already uses. This avoids hard-coding and keeps the adapter in sync with the curated project vocabulary.

```python
# At module init — align with ConjectureEntity vocabulary source
from dackar.config import nlpConfig
import csv, pathlib

def _load_causal_verb_lemmas() -> frozenset:
    path = nlpConfig['files']['cause_effect_keywords_file']
    lemmas = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            v = (row.get('VERB') or '').strip().lower()
            if v and ' ' not in v:   # single-word verbs only for dep-tree matching
                lemmas.add(v)
    return frozenset(lemmas)

_DEP_CAUSAL_VERB_LEMMAS = _load_causal_verb_lemmas()
```

For `_CONJECTURE_PAT`: replace the hardcoded regex with a lookup against `conjecture_keywords.csv` (the same file `ConjectureEntity` already loads), or call `ConjectureEntity` directly on the sentence text and check for `conjecture` entity labels.

**Expected impact:** Recall improvement across all datasets, particularly for DS3/DS4 sentences containing verbs like "accelerated", "propagated", "prompted", "initiated" that currently produce zero dep-tree matches.

---

### Improvement B — Calibrated threshold constants ✓

**Status: Implemented 2026-05-09** — `calibrate_chain_threshold(annotated_chains, extracted_statements)` added to `causal_condition_adapter.py`. The function sweeps Jaccard thresholds from 0.10 to 0.90 and returns a precision/recall/F1 table plus `best_threshold`. Call from the test notebook after pipeline execution; update `_CHAIN_JACCARD_THRESHOLD` in `causal_condition_adapter.py` and `_ENTITY_LINK_JACCARD_THRESHOLD` in `doc_extraction/adapter.py` with the returned values.

**Priority: High | Effort: Trivial**

**Scope:** `_CHAIN_JACCARD_THRESHOLD` and `_ENTITY_LINK_JACCARD_THRESHOLD` in `causal_condition_adapter.py` and `doc_extraction/adapter.py`

**Problem:** Both constants were set to conservative initial guesses (0.35 and 0.40). The test notebook Section 6 calibration sweep already computes the optimal values across all four datasets — but the results have not been written back to the source constants.

**Fix:** Run the calibration notebook after Fixes 1–6 are applied (span quality is now higher, so optimal thresholds may have shifted), read the sweep output, and update the two constants. Re-run the full evaluation to confirm improvement.

**Note:** Optimal threshold may differ between chain linking (where longer, richer spans from Fix 4 now produce higher Jaccard scores) and entity linking (where the same improvement applies). Evaluate independently.

---

### Improvement C — Per-clause negation and conjecture scoping ✓

**Status: Implemented 2026-05-09** — Added `_causal_span_text(causal_tok)` helper that collects the causal verb + its argument subtrees (nsubj, dobj, pobj, obl, agent) while blocking clause expansions (`relcl`, `acl`, `advcl`, `ccomp`, `rcmod`). Pass 1 now calls `_CONJECTURE_PAT.search(_causal_span_text(causal_tok))` instead of `sent.text`. Pass 2 now computes `p2_neg`/`p2_conj` from the `before_text + after_text` window and passes them to `_score_causal_statement` and stores them in the statement dict (previously hardcoded `False`).

**Priority: Medium | Effort: Small–Medium**

**Scope:** `_dep_causal_fallback` Pass 1 in `causal_condition_adapter.py`

**Problem:** `_CONJECTURE_PAT` and `_NEGATION_PAT` are currently matched against the full `sent.text`. In DS3/DS4, a long sentence with 4 causal relations may contain one hedged clause ("possibly caused by X") and three unhedged ones. The current code flags all four extracted statements as `conjectural=True`, incorrectly downgrading the confidence of the unhedged ones.

**Fix:** Scope the negation/conjecture match to the token window around each causal token: from the nsubj head token to the dobj/pobj head token, rather than the full sentence. For pass 2 (regex), match within the `before_text`/`after_text` window already extracted.

```python
# Instead of:
conj = bool(_CONJECTURE_PAT.search(sent.text))

# Scope to the causal span window:
span_start = min(causal_tok.i, (nsubj_tok.i if nsubj_tok else causal_tok.i))
span_end   = max(causal_tok.i, (obj_tok.i   if obj_tok   else causal_tok.i)) + 1
span_text  = sent[max(0, span_start - sent.start): span_end - sent.start].text
conj = bool(_CONJECTURE_PAT.search(span_text))
```

**Expected impact:** Precision improvement — statements from unhedged clauses no longer incorrectly flagged as conjectural; confidence scores rise for those statements; chain detection improves since conjectural statements are skipped in some downstream paths.

---

### Improvement D — Embedding-based chain linking for cross-sentence relations ✓

**Status: Implemented 2026-05-09** — Added `_CHAIN_EMBED_THRESHOLD = 0.75` constant. Added `_build_embed_fn(nlp, llm_cfg)` helper: priority (1) `llm_cfg["embedding_fn"]` injected callable, (2) `nlp(a).similarity(nlp(b))` when the spaCy model has word vectors (`en_core_web_md/lg/trf`; sm is skipped automatically), (3) None → Jaccard-only. Modified `_chain_causal_statements(stmts, embed_fn=None)` to run the embedding pass when Jaccard fails. `embed_fn` is built once at the top of `extract_stage5_causal_condition` and threaded to all `_chain_causal_statements` call sites (including `_apply_llm_extract_all`). `calibrate_chain_threshold` also accepts `embed_fn` so sweep results reflect the live code path.

**Priority: Medium | Effort: Large**

**Scope:** `_chain_causal_statements` in `causal_condition_adapter.py`

**Problem:** The Jaccard linker works when adjacent cause/effect spans share tokens. For DS4 cross-sentence relations, semantically equivalent but lexically distinct phrases ("primary coolant inventory loss" ↔ "reactor coolant system leakage") produce Jaccard ≈ 0 and are never linked. This is the root cause of near-zero formal chain detection on DS4.

**Fix:** Add an optional embedding-similarity pass as a fallback after Jaccard fails to link two statements. When `nlp` is `en_core_web_md` or `en_core_web_lg` (which include word vectors), use spaCy's built-in `doc.similarity()` on the cause/effect spans. Alternatively, inject a lightweight sentence-transformer via `llm_cfg["embedding_fn"]`.

```python
# In _chain_causal_statements, after Jaccard pass:
if _jaccard(eff, cau) < _CHAIN_JACCARD_THRESHOLD and embed_fn is not None:
    sim = embed_fn(stmts[i]["effect_text"], stmts[j]["cause_text"])
    if sim >= _CHAIN_EMBED_THRESHOLD:
        successors[i].append(j)
```

**Dependency:** Requires either `en_core_web_md/lg` (already in project deps?) or an injected embedding callable. The threshold `_CHAIN_EMBED_THRESHOLD` needs calibration against DS4 annotations.

**Expected impact:** Chain detection rate on DS4 from near-0% to meaningful; the cross-sentence chains that are the primary evaluation target of DS4 become reachable.

---

### Improvement E — Per-statement `extractor_used` label for `dep_fallback` ✓

**Status: Implemented 2026-05-09** — `_assign_confidence` in `doc_extraction/adapter.py` gains a `stmt_source` parameter (default `""`). A new rule inserted before the existing rules: when `"dep_fallback"` appears in the effective source (`stmt_source or extractor_used`), the statement is capped at MEDIUM (never promoted to HIGH via the CausalSentence path). For statement-level records, `stmt.get("source","")` is passed directly. For chain-level records, a `stmt_source_by_id` lookup is built from `causal_statements` and the chain is marked `dep_fallback` when all contributing statement sources are `dep_fallback` or empty.

**Priority: Medium | Effort: Small**

**Scope:** `causal_condition_adapter.py` and `doc_extraction/adapter.py`

**Problem:** Statements produced by `_dep_causal_fallback` end up inside the CausalSentence result dict. At the stage5 level, `extractor_used = "CausalSentence"`. `_assign_confidence` in `doc_extraction/adapter.py` then applies the CausalSentence confidence rules to these statements, potentially upgrading them to MEDIUM — even though their spans and linguistic quality are significantly lower. This is a silent precision bug.

**Fix:** Add a per-statement `"extractor_used"` field (already present implicitly via `"source"`) and teach `DocExtractionAdapter` to read `stmt["source"]` instead of relying solely on the top-level `stage5["extractor"]["used"]` when assigning confidence.

```python
# In _assign_confidence, check per-statement source when available:
stmt_source = stmt.get("source", extractor_used)
if "dep_fallback" in stmt_source and not "llm_repair" in stmt_source:
    # dep_fallback statements: mechanism overlap gates MEDIUM, not CausalSentence rules
    if has_mechanism:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW
```

**Expected impact:** More accurate confidence scores for dep_fallback statements. Statements that currently reach MEDIUM via the CausalSentence path but have low span quality are correctly rated LOW until Fix 4's improved spans actually push them to MEDIUM via mechanism overlap.

---

### Improvement F — Health status vocabulary alignment ✓

**Status: Implemented 2026-05-09** — Added `_load_health_condition_terms()` loading from `data/health_status_keywords_negative.csv` (→ failed/degraded split via `_HEALTH_FAILED_FRAGMENTS`) and `data/health_status_keywords_positive.csv` (→ acceptable). Result stored in `_HEALTH_CONDITION_TERMS` (24 failed, 183 degraded, 79 acceptable terms). Added `_HEALTH_FAILED_ROOTS` / `_HEALTH_DEGRADED_ROOTS` frozensets as a root-fragment fallback to catch morphological variants absent from the CSVs (e.g. "leakage" via "leak", "worn" via "wear"). Both `_infer_condition_from_text` and `_normalize_health_state` now use a two-pass check: full-term first, root-fragment second. Falls back to hardcoded sets when files are unavailable.

**Priority: Low | Effort: Small**

**Scope:** `_infer_condition_from_text` and `_normalize_health_state` in `causal_condition_adapter.py`

**Problem:** `_infer_condition_from_text` uses ~30 hardcoded terms for failed/degraded/acceptable classification. `data/health_status_keywords.csv` contains 80+ curated terms across VERB, NOUN, ADJ, ADV columns (e.g., rupture, aggravate, cavitation, burnout, leakage, vibration, deflection, inoperative, impaired, flawed). These are not referenced by the adapter.

**Fix:** Load the health status vocabulary from `data/health_status_keywords.csv` at module init (same pattern as Improvement A), replacing the hardcoded term lists in `_infer_condition_from_text`. The failed/degraded/acceptable mapping can be derived from `health_status_keywords_negative.csv`, `health_status_keywords_positive.csv`, and the main file.

**Expected impact:** More accurate `as_found`/`as_left` condition state extraction, particularly for WO/CR text that uses domain-specific failure vocabulary not in the generic hardcoded list.

---

## 10. Post-Evaluation Improvements (G–J)

Identified from the DS2–DS5 evaluation (2026-05-09). Ordered by recommended implementation sequence.

---

### Improvement G — Effect subtree expansion

**Priority:** High
**Effort:** Small
**Status:** Implemented ✓ 2026-05-09
**Files:** `ner/causal_condition_adapter.py`

#### Problem

Pass 1 of `_dep_causal_fallback` collects effect spans from `_EFFECT_DEPS = {"dobj", "obj", "pobj", "obl", "attr"}`. Two common syntactic patterns are missed:

**Pattern A — `xcomp` infinitival complement:**
> "caused the pump **to fail**"

Parse: `caused` → `fail` (xcomp) → `pump` (nsubj of fail). The semantic effect is the `xcomp` + its subject. Neither is under `dobj`.

**Pattern B — `pcomp` after prepositional causal verb:**
> "led to the system **shutting down**"

Parse: `led` → `to` (prep) → `shutting` (pcomp) → `system` (nsubj of shutting). The effect phrase is `the system shutting down`. Current code picks up `to` as the prep connector but finds no dobj/pobj under `led`.

Both patterns produce `effect_text = ''`, which:
- Counts as recall=0 in evaluation (strict cause+effect overlap required)
- Prevents `_best_overlapping_entity` from finding any G5 outcome overlap
- Keeps `has_mechanism = False` → `ConfidenceLevel.LOW` in the adapter

Empty effects account for a significant fraction of the 15% → recall gap across all datasets. This is the highest-value structural fix that requires no LLM.

#### Design

Extend Pass 1 effect collection with two additional dep patterns, applied when `_EFFECT_DEPS` produces an empty result:

```python
# Pattern A: xcomp infinitival
if not effect_text:
    for child in causal_tok.children:
        if child.dep_ == "xcomp":
            xcomp_subj = next(
                (c for c in child.children if c.dep_ in {"nsubj", "nsubjpass"}), None
            )
            effect_text = (
                _np_subtree_text(xcomp_subj) + " " + child.text
                if xcomp_subj else _np_subtree_text(child)
            )
            break

# Pattern B: pcomp under prep child
if not effect_text:
    for child in causal_tok.children:
        if child.dep_ == "prep":
            pcomp = next((c for c in child.children if c.dep_ == "pcomp"), None)
            if pcomp:
                pcomp_subj = next(
                    (c for c in pcomp.children if c.dep_ in {"nsubj", "nsubjpass"}), None
                )
                effect_text = (
                    _np_subtree_text(pcomp_subj) + " " + pcomp.text
                    if pcomp_subj else _np_subtree_text(pcomp)
                )
                break
```

Also extend Pass 2: when the regex connector fires but the text after the boundary is a gerund phrase, collect the full gerund NP as effect rather than stopping at the first sentence boundary.

#### Expected impact

- Empty `effect_text` rate: ~30% → ~10%
- Mean effect F1 across all datasets: 0.18 → ~0.28
- `has_mechanism = True` rate increases because effect spans can now overlap G5 outcome entities
- Confidence upgrade path: dep_fallback + non-empty spans + G5 overlap → MEDIUM instead of LOW

#### Regression risk

Low. Patterns fire only when `effect_text` is empty after the existing `_EFFECT_DEPS` pass.

---

### Improvement H — LLM trigger broadening

**Priority:** High
**Effort:** Small (infrastructure already exists; one trigger condition + config flag)
**Status:** Implemented ✓ 2026-05-09
**Files:** `ner/causal_condition_adapter.py`

#### Problem

`_apply_llm_extract_all` currently fires only when **both** CausalSentence and CausalSimple return nothing. Because dep_fallback populates `extracted_causal_statements` inside the CausalSentence path, `_has_useful_stage5_signal` passes and the function returns before LLM runs — even when every dep_fallback statement has an empty effect or confidence ≤ 0.60.

Result: 0% recall on implicit relations (20 DS5 relations), 0% on reversed order (7 relations), 0% on counterfactual (7 relations). These are structurally unreachable without LLM — no dep-tree improvement will recover them.

#### Why expanding LLM use is the right call for RCA

The RCA workflow is **recall-critical**: a missed causal link in a condition report means an undetected failure mode propagates through the FM ranking at Steps 1 and 2d without evidence. The cost of a false positive (an LLM-generated statement that a reviewer discards) is far lower than the cost of a missed true positive (a latent failure mode that doesn't appear in the candidate set).

The three categories of relations where LLM adds irreplaceable value:

1. **Implicit causation (domain-knowledge inference):**
   > "A 0.3 mm longitudinal scratch was noted on the zircaloy cladding of assembly E-14. Primary coolant iodine-131 activity increased tenfold."
   No connector; the link requires knowing that cladding scratches cause fuel leakage which releases fission products into coolant. A dep tree sees two unrelated sentences. An LLM with nuclear domain knowledge recovers the causal relation.

2. **Reversed causal order (syntactic inversion):**
   > "The reactor tripped on low steam generator level because the feedwater control valve was slow to respond."
   The effect (`reactor trip`) is the syntactic main clause; the cause (`valve slow response`) is a subordinate `because` clause. The dep fallback fires on `tripped` and finds `nsubj = reactor` → cause — which is wrong. An LLM correctly inverts the direction.

3. **Multi-cause convergence:**
   > "The combined effect of coolant leakage, bearing wear, and elevated ambient temperature drove the pump below its minimum flow requirement."
   Three causes, one effect, no pairwise connectors. The dep fallback extracts at most one (cause, effect) pair. An LLM returns all three (cause_i, effect) pairs.

These three categories account for ~43% of DS5 relations and are the dominant failure modes in real nuclear condition reports.

#### Design

Add a secondary trigger gated on a new `llm_cfg` flag, executed after dep_fallback fills `cs_result`:

```python
# In extract_stage5_causal_condition, after dep_fallback branch:
_maybe_trigger_llm_on_weak_fallback(
    cs_result, text, doc_id, chunk_index, doc_type, section_role,
    llm_cfg, embed_fn,
)
```

```python
def _maybe_trigger_llm_on_weak_fallback(result, text, doc_id, chunk_index,
                                         doc_type, section_role, llm_cfg, embed_fn):
    """Fire LLM supplement when dep_fallback returned only weak results."""
    if not (llm_cfg and llm_cfg.get("extract_all") and llm_cfg.get("enabled")):
        return
    if not llm_cfg.get("trigger_on_weak_fallback", False):
        return
    stmts = result.get("extracted_causal_statements", [])
    # Trigger when all existing statements are weak (empty effect or low confidence)
    all_weak = all(
        not s.get("effect_text") or s.get("confidence", 0) <= 0.60
        for s in stmts
    )
    if stmts and not all_weak:
        return  # dep_fallback already produced good output; don't supplement
    _apply_llm_extract_all(
        result, text, doc_id, chunk_index, doc_type, section_role,
        llm_cfg, embed_fn=embed_fn,
    )
```

New `llm_cfg` parameter:

```python
llm_cfg = {
    "enabled": True,
    "extract_all": True,
    "trigger_on_weak_fallback": True,   # NEW
    "max_repair_calls": 3,
    # Optional: tighten what counts as "weak" at the call site
    "weak_confidence_threshold": 0.60,  # NEW (optional, default 0.60)
}
```

#### LLM prompt strategy for RCA

The existing `_llm_extract_all_relations` prompt covers explicit relations well. For the three categories above, augment with explicit instructions:

```
You are analyzing a nuclear power plant condition report or work order.
Extract ALL causal relations, including:
- Implicit relations (no connector word; inferred from domain knowledge)
- Relations where the effect is stated before the cause
- Multi-cause relations (list each cause separately as a (cause, effect) pair)
- Relations expressed through negation or counterfactual ("would not have ... if")

For each relation return:
  cause: the text of the causal agent or condition
  effect: the text of the resulting condition or event
  connector: the causal connector if present, or null
  relation_type: "explicit" | "implicit" | "ambiguous"
  confidence: 0.0–1.0

Return a JSON array. Return [] if no causal relations are present.
```

This prompt extension directly targets the three DS5 failure categories.

#### Cost control

- LLM fires at most once per chunk (existing `_apply_llm_extract_all` contract)
- The `trigger_on_weak_fallback` flag is `False` by default — opt-in, no regression on deployments without LLM
- The "all_weak" gate ensures LLM only supplements when dep_fallback genuinely failed; strong dep_fallback results are not re-processed
- Existing `max_repair_calls` budget is separate and unaffected

#### Expected impact

- Implicit relation recall: 0% → ~30–40% (LLM-dependent)
- Reversed order: 0% → ~50–60%
- Multi-cause convergence: 0% → ~40% (expanded into individual pairs)
- Overall DS5 recall: 5% → ~25–30%
- No regression when `trigger_on_weak_fallback=False`

#### Acceptance tests

1. Run DS5 evaluation with a mock LLM returning correct answers for 5 test records; verify statements tagged `"LLM_extract_all"`, `ConfidenceLevel.LOW` applied
2. Verify `trigger_on_weak_fallback=False` (default) produces identical output to current behaviour
3. Verify a dep_fallback result with non-empty effect text does NOT trigger the LLM supplement

---

### Improvement I — Lightweight cross-sentence coreference resolution

**Priority:** Medium
**Effort:** Medium
**Status:** Implemented ✓ 2026-05-09
**Files:** `ner/causal_condition_adapter.py`

#### Problem

DS5 cross-sentence patterns where the second sentence begins with a demonstrative subject are systematically missed:

> S1: "The operator isolated the cooling loop."
> S2: "**This** caused the heat exchanger to exceed its temperature limit."

S2 parses as `This` (nsubj) → `caused` (ROOT) → `heat exchanger` (dobj). The dep fallback extracts `cause_text = "This"` — a pronoun with no semantic content. The entity linker finds nothing. `has_mechanism = False`. `ConfidenceLevel.LOW`. The actual cause ("operator isolated the cooling loop") is invisible.

This pattern affects operator_action_as_cause (11 DS5 relations), recursive_feedback (11 relations), and latent_gap (4 relations) — 34% of DS5.

#### Design

Add `_resolve_sentence_coreference(sents)` as a pre-processing step inside `_dep_causal_fallback`, applied to the sentence list before the main Pass 1 loop:

```python
_DEMONSTRATIVES = frozenset({
    "this", "it", "these", "that", "those",
    "the above", "such conditions", "the situation",
    "the event", "this condition", "this failure",
    "this issue", "the problem",
})

def _resolve_sentence_coreference(
    sents: List[Any], nlp: Any
) -> List[str]:
    """
    Rule-based demonstrative resolution for adjacent sentences.
    Returns list of sentence text strings with demonstrative nsubj
    of S[i] replaced by the head NP of S[i-1].

    Scope: one sentence back, subject position only.
    Does not require a full coreference model.
    """
    resolved = []
    prev_head_np = None
    for sent in sents:
        root = next((t for t in sent if t.dep_ == "ROOT"), None)
        nsubj = next(
            (t for t in sent if t.dep_ in {"nsubj", "nsubjpass"}), None
        )
        # Check if subject is a demonstrative reference
        if (nsubj is not None
                and nsubj.lower_ in _DEMONSTRATIVES
                and prev_head_np is not None):
            # Substitute: replace demonstrative with previous head NP
            new_text = sent.text.replace(nsubj.text, prev_head_np, 1)
            resolved.append(new_text)
        else:
            resolved.append(sent.text)
        # Track this sentence's root NP for next iteration
        if root is not None:
            root_subj = next(
                (t for t in sent if t.dep_ in {"nsubj", "nsubjpass"}), None
            )
            prev_head_np = _np_subtree_text(root_subj) if root_subj else None
    return resolved
```

Wire into `_dep_causal_fallback` before Pass 1:

```python
# Resolve demonstrative cross-sentence references
if nlp is not None and len(list(doc.sents)) > 1:
    resolved_texts = _resolve_sentence_coreference(list(doc.sents), nlp)
    # Re-parse resolved texts for Pass 1
    resolved_docs = [nlp(t) for t in resolved_texts]
else:
    resolved_docs = [s.as_doc() for s in doc.sents]
```

Tag resolved statements with `"coref_resolved": True` for downstream tracking.

#### Scope constraint

Resolution is limited to:
- One sentence back only (S[i] references S[i-1])
- Deterministic demonstratives only — not generic pronouns (`he`/`she`/`they`)
- Subject position only — objects and obliques not resolved

This avoids requiring a full coreference model while capturing the dominant cross-sentence pattern in condition reports.

#### Expected impact

- operator_action_as_cause: 0% → ~20–25%
- recursive_feedback: 18% → ~30%
- latent_gap_in_chain: 0% → ~15%
- No regression on single-sentence cases (trigger only on demonstrative nsubj)

---

### Improvement J — Evaluation harness hardening

**Priority:** Supporting
**Effort:** Small
**Status:** Planned
**Files:** `tests/test_causal/test_causal_extraction.ipynb`

#### Problems and fixes

**J1 — DS1 path missing:**
`causal_dataset.json` not found at `data/`; loader silently skips it, leaving DS1 absent from all evaluation sections. Add a path fallback:

```python
for fname in ('causal_dataset.json', 'causal_dataset_1.json'):
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        raw = json.load(open(fpath))
        all_entries.extend(_load_ds1(raw))
        break
else:
    print('WARNING: DS1 not found — skipped')
```

**J2 — Add precision metric to Section 2:**
Currently only recall is measured. Precision (fraction of extracted statements that match a gold relation) is unknown. A high false-positive rate from the dep fallback would affect operator trust in the tool.

```python
precision_rows = []
for r in results:
    for s in r['stmts']:
        matched = any(
            best_stmt_match([s], rel['cause_span'], rel['effect_span'])[1] >= 0.5
            for rel in r['entry']['relations']
        )
        precision_rows.append({'dataset': r['entry']['dataset'], 'matched': matched})
df_prec = pd.DataFrame(precision_rows)
display(df_prec.groupby('dataset').agg(
    n_extracted=('matched', 'count'),
    precision=('matched', 'mean'),
).round(3))
```

**J3 — Document embedding model limitation:**
`en_core_web_sm` has no word vectors; Improvement D (embedding chain linking) is silently disabled. Add a warning at setup:

```python
if nlp.vocab.vectors.shape[0] == 0:
    print("WARNING: no word vectors — Improvement D (embedding chain) is disabled.")
    print("         Re-run with en_core_web_md or en_core_web_lg for embedding evaluation.")
```

**J4 — DS5 challenge-type breakdown in Section 1:**
Section 1 (per-entry extraction display) currently shows all entries uniformly. For DS5, group by challenge type and print a header before each group so that failure patterns are visually identifiable during interactive review.

---

## 11. Improvement Sequencing (G–J)

```
Improvement G — Effect subtree expansion
  │  [no deps; smallest change; directly upgrades LOW → MEDIUM confidence]
  │
  └──► Improvement H — LLM trigger broadening
         │  [depends on G so that LLM supplements already-improved dep_fallback output]
         │  [requires LLM infrastructure to be available in the test environment]
         │
         └──► Improvement I — Cross-sentence coreference
                [depends on G so coref-resolved cause_text feeds improved effect extraction]
                [independent of H — can run in parallel if LLM is not yet available]

Improvement J — Evaluation harness
  [independent; run first to get accurate baselines before G/H/I]
  [run again after each improvement to measure delta]
```

**Recommended order: J → G → H → I**

J produces accurate baselines. G is the smallest change with the most direct confidence-level impact and unblocks H. H is high-ROI but requires LLM infrastructure; it should be evaluated on real LLM output, not a mock. I adds meaningful cross-sentence coverage but is the most complex implementation and benefits from G's improved spans.

---

## 12. Expected Metrics After G–J

Estimated against DS2–DS5 (no LLM for G/H unless noted):

| Dataset | Baseline (A–F) | After G | After G+H (LLM) | After G+H+I |
|---------|---------------|---------|-----------------|-------------|
| ds2 | 12% | ~18% | ~35% | ~35% |
| ds3 | 11% | ~16% | ~30% | ~30% |
| ds4 | 23% | ~30% | ~45% | ~50% |
| ds5 | 5% | ~8% | ~25% | ~35% |
| **TOTAL** | **15%** | **~20%** | **~35%** | **~40%** |

DS5 adversarial types that remain refractory even after G–I:
- `counterfactual_conditional` — requires pragmatic reasoning beyond surface extraction
- `reporting_verb_ambiguity` — requires disambiguation of factual vs. reported events
- `latent_gap_in_chain` (partial) — missing intermediate node requires domain inference

These residual cases are the long-tail motivation for continued LLM investment.

---

## 13. Updated Files Affected

| Improvement | File | Change | Status |
|-------------|------|--------|--------|
| G | `ner/causal_condition_adapter.py` | Extend Pass 1 with `xcomp`/`pcomp`/`ccomp` patterns; Pass 3 conjunctive backward connectors (`because`/`since`/`given that`) | ✓ |
| H | `ner/causal_condition_adapter.py` | `_maybe_trigger_llm_on_weak_fallback`; `trigger_on_weak_fallback` flag; few-shot `llm_cfg["few_shot_examples"]`; extended prompt with 4 relation categories + `relation_type` field | ✓ |
| I | `ner/causal_condition_adapter.py` | `_DEMONSTRATIVES` frozenset; `_get_prev_sent_subject_np`; post-hoc coref substitution in Pass 1; `coref_resolved` tag | ✓ |
| G/H/I review | `ner/causal_condition_adapter.py`, `doc_extraction/adapter.py`, `doc_extraction/schema.py` | Code review: `ruled_out_mechanisms` end-to-end wiring; double LLM call prevention; `has_ruled_out_mechanisms` flag; `_has_useful_stage5_signal` update; statement schema consistency (`relation_type`, `coref_resolved`, `sent_index`) | ✓ |
| G/H/I review | `doc_extraction/schema.py` | Added `ruled_out_mechanisms: List[str]` field to `DocExtractionRecord`; `as_chroma_metadata()` serialization | ✓ |
| G/H/I review | `doc_extraction/adapter.py` | `ruled_out_mechanisms` extraction from stage5 output; passed to both chain and statement record constructors | ✓ |
| J0 | `tests/test_causal/test_causal_extraction.ipynb` | Section 0 smoke test cell: added `text_processing` sys.modules alias (was `ModuleNotFoundError`) | ✓ |
| J1 | `tests/test_causal/test_causal_extraction.ipynb` | `cell-load`: DS1 path fallback | ✓ (in notebook) |
| J2 | `tests/test_causal/test_causal_extraction.ipynb` | `cell-span-metrics`: precision metric | ✓ (in notebook) |
| J3 | `tests/test_causal/test_causal_extraction.ipynb` | `cell-setup`: word vector warning | ✓ (in notebook) |
| J4 | `tests/test_causal/test_causal_extraction.ipynb` | `cell-extract`: DS5 challenge-type grouping | pending |

---

## 14. Pre-Implementation Review Notes (2026-05-09)

Five open considerations identified during plan review. Items 1–4 are actionable and should be resolved before or during implementation of G–J. Item 5 is a longer-term architectural question deferred for now.

---

### 14.1 — Reversed causal order is likely a fixable dep-tree bug, not a structural ceiling

The plan currently groups `reversed_causal_order` with `implicit` and `counterfactual` and routes all three to the LLM. This conflates two distinct problems.

True implicit causation (no connector, domain inference required) cannot be recovered without LLM. But reversed order via a conjunctive `because`/`since`/`as` clause has an explicit connector — just in a syntactic position the current pass misses. `"The reactor tripped because the feedwater level dropped"` uses `because` as a subordinating conjunction introducing an `advcl` (adverbial clause), not a prepositional phrase. Pass 2's `_PREP_CONNECTOR_PAT` does not match bare conjunctions.

**Proposed fix:** Add Pass 3 to `_dep_causal_fallback` handling conjunctive backward connectors:

```python
_CONJ_BACKWARD_CONNECTORS = re.compile(
    r'\b(because|since|as|given that|seeing that|inasmuch as)\b',
    re.IGNORECASE
)
```

For each match, split the sentence at the connector boundary: text before = effect, text after = cause. Apply `_np_subtree_text` to extract the head NPs from each side.

**Impact:** DS5 `reversed_causal_order` is 7 relations; a Pass 3 fix could recover 4–5 without any LLM call. This should be implemented as part of Improvement G (same file, same pass structure) rather than deferring to the LLM path.

**Action:** Separate `reversed_causal_order` from the LLM-only bucket. Add Pass 3 in Improvement G. Re-evaluate after implementation to determine how many reversed-order cases remain for LLM.

---

### 14.2 — The evaluation metric is misaligned with RCA workflow value

The current recall metric (≥40% token overlap on both cause and effect spans) measures extraction fidelity, not downstream workflow value. What actually matters is: *does the extracted `cause_text`, when resolved through `_best_overlapping_entity`, produce the correct `inferred_fm_label`?*

A statement like `cause_text = "wear on the pump impeller"` scores low token overlap against the gold span `"impeller wear-induced cavitation"`, but both resolve to the same FM label. The current metric likely underreports real workflow value and may misdirect improvement priorities — we could be optimising for span-level fidelity at the expense of FM-resolution quality.

**Proposed addition to Improvement J — FM-resolution metric:**

```python
# For DS1 entries (which have expected_g4_mechanism annotations):
fm_rows = []
for r in results:
    e = r['entry']
    mech = e.get('expected_g4_mechanism')
    if mech is None:
        continue
    for s in r['stmts']:
        linked = _best_overlapping_entity(
            s.get('cause_text', ''), mock_spans=[mock_span(mech)], nlp=nlp
        )
        fm_rows.append({
            'id': e['id'],
            'cause_text': s.get('cause_text', ''),
            'expected_fm': mech,
            'fm_resolved': linked is not None,
        })
df_fm = pd.DataFrame(fm_rows)
print(f"FM-resolution rate: {df_fm.fm_resolved.mean():.0%}")
```

This metric — FM-resolution rate — should sit alongside span F1 in Section 2 and be the primary success criterion for Improvements G and H.

**Action:** Add FM-resolution metric to Improvement J scope. Treat it as the primary KPI for G–I, with token overlap as a secondary diagnostic.

---

### 14.3 — LLM prompt needs domain grounding (few-shot examples)

The Improvement H prompt tells the LLM it is analysing a nuclear plant condition report but provides no domain examples. For nuclear-specific implicit causation — where the link between cladding damage and coolant activity, or between bearing wear and pump cavitation, requires domain knowledge the LLM may not surface reliably — few-shot examples substantially improve precision.

**Proposed addition to the Improvement H prompt:**

```
Examples of causal relations in nuclear plant condition reports:

  cause: "a 0.3 mm longitudinal scratch on the zircaloy cladding"
  effect: "tenfold increase in primary coolant iodine-131 activity"
  relation_type: "implicit"
  connector: null

  cause: "pump wear ring clearances beyond the maintenance limit"
  effect: "pump output below the technical specification minimum flow"
  relation_type: "explicit"
  connector: "drove"

  cause: "bearing wear on the feedwater pump"
  effect: "excessive shaft vibration"
  relation_type: "explicit"
  connector: "caused"
```

Two to three examples covering implicit, explicit, and multi-cause patterns are sufficient. Examples should be drawn from actual condition reports in the project corpus rather than synthetic text, so the LLM sees realistic vocabulary and sentence structure.

**Action:** Add `llm_cfg["few_shot_examples"]` as an optional list of `{cause, effect, relation_type, connector}` dicts. `_llm_extract_all_relations` injects them into the prompt when present. Populate from 3–5 real WO/CR entries in the test dataset.

---

### 14.4 — Negated causal statements are an underused asset

`negated=True` statements are currently extracted, tagged, and assigned `ConfidenceLevel.LOW`, but are otherwise treated as low-quality outputs to be sidelined. In the RCA context this is the wrong framing: "the trip was **not** caused by sensor failure" is positively informative — it eliminates a failure mode hypothesis from the candidate set.

**Proposed routing change in `DocExtractionAdapter.extract()`:**

Instead of writing negated statements as low-confidence `DocExtractionRecord` entries that compete with positive evidence, route them to a dedicated `ruled_out_mechanisms` list in the record:

```python
# In DocExtractionAdapter.extract():
ruled_out = []
positive  = []
for s in causal_statements:
    if s.get("negated"):
        ruled_out.append({
            "cause_text":  s.get("cause_text", ""),
            "effect_text": s.get("effect_text", ""),
            "connector":   s.get("connector", ""),
            "source":      s.get("source", ""),
        })
    else:
        positive.append(s)
# ruled_out written to DocExtractionRecord.ruled_out_mechanisms
# positive processed through existing confidence/linking pipeline
```

At Step 2d, `ruled_out_mechanisms` can be used to penalise FM candidates whose label matches a ruled-out cause — a direct hypothesis elimination signal. No new extraction work is needed; this is a routing and schema change.

**Action:** Add `ruled_out_mechanisms: List[Dict]` field to `DocExtractionRecord` schema. Update `DocExtractionAdapter.extract()` to split on `negated` before the confidence assignment loop. Add Step 2d query logic to use the field when present.

---

### 14.5 — CausalSentence and CausalSimple are invisible in the evaluation

The evaluation shows 100% of detected relations sourced from `dep_fallback`. This happens because `CausalSentence` and `CausalSimple` require SSC entity annotations from the upstream NER pipeline, which are not present in raw-text evaluation runs. If those paths have regressions from the A–F improvements, the evaluation cannot detect them.

**Proposed addition to Improvement J — primary extractor smoke test:**

Add a small fixture (5–10 sentences) that injects mock SSC entity spans directly into the spaCy `Doc.ents` before calling `extract_stage5_causal_condition`. Verify that:
1. `CausalSentence` fires and `extractor_used = "CausalSentence"` in the result
2. `dep_fallback` does NOT fire when `CausalSentence` returns statements
3. Confidence is MEDIUM or HIGH (not LOW) when `has_mechanism = True`

```python
# Minimal SSC entity injection for smoke test:
from spacy.tokens import Span
doc = nlp("Bearing wear caused excessive vibration.")
doc.ents = (Span(doc, 0, 2, label="SSC"),)   # "Bearing wear" as SSC entity
result = extract_stage5_causal_condition(
    doc_id="smoke", chunk_index=0, chunk_text=doc.text,
    doc_type="CR", section_role="body", nlp=nlp,
    causal_sentence_factory=lambda text, nlp: CausalSentence(text, nlp=nlp),
)
assert result["extractor"]["used"] == "CausalSentence"
```

**Action:** Add 5-entry smoke-test fixture to `test_causal_extraction.ipynb` as a new Section 0. Treat passing this fixture as a regression gate before any merge of G–J changes.

---

## 15. Code Review Findings and Fixes (2026-05-09)

Applied during post-implementation review of Improvements G, H, I.

### 15.1 — `ruled_out_mechanisms` end-to-end wiring

**Problem:** Negated statements were routed to `ruled_out_mechanisms` in `_route_negated_statements` but the field existed nowhere downstream (`DocExtractionRecord` had no such field; `adapter.py` did not read it).

**Fix:**
- Added `ruled_out_mechanisms: List[str] = field(default_factory=list)` to `DocExtractionRecord` in `doc_extraction/schema.py`
- Added `"; ".join(self.ruled_out_mechanisms)` serialization to `as_chroma_metadata()`
- Added extraction in `doc_extraction/adapter.py`: `ruled_out = [s.get("cause_text","") for s in stage5.get("ruled_out_mechanisms",[]) if s.get("cause_text")]`
- Passed `ruled_out_mechanisms=ruled_out` to both chain and statement record constructors

### 15.2 — Double LLM call prevention

**Problem:** `_maybe_trigger_llm_on_weak_fallback` called `_apply_llm_extract_all`, then the existing return point in CausalSentence/CausalSimple paths also called `_apply_llm_extract_all` — executing LLM twice on the same chunk.

**Fix:** Added early exit at the top of `_apply_llm_extract_all`:
```python
existing = result.get("extracted_causal_statements", [])
if any(s.get("source") == "LLM_extract_all" for s in existing):
    return
```

### 15.3 — Summary flag and signal detection consistency

**Problem:** After `_route_negated_statements` moves negated items out of `extracted_causal_statements`, `has_negation` would be False even when ruled-out mechanisms were present. `_has_useful_stage5_signal` also had no awareness of `ruled_out_mechanisms`.

**Fix:**
- Added `"has_ruled_out_mechanisms": bool(out.get("ruled_out_mechanisms"))` to `_fill_summary_flags`
- Added `flags.get("has_ruled_out_mechanisms", False)` and `bool(out.get("ruled_out_mechanisms"))` to `_has_useful_stage5_signal`
- Added `"has_ruled_out_mechanisms"` to `STAGE5_OUTPUT_SCHEMA` summary_flags

### 15.4 — Statement schema consistency

**Problem:** Pass 2 and Pass 3 statement dicts did not emit `relation_type` or `coref_resolved` fields. `_build_causal_statement` emitted `sent_index: None` but did not emit `coref_resolved` or `relation_type`.

**Fix:** Added `"relation_type": "explicit"`, `"coref_resolved": False` to all three pass dicts and to `_build_causal_statement` defaults. LLM statements already emitted `relation_type` from the prompt; rule-based paths now match that schema.

### 15.5 — Notebook Section 0 smoke test

**Problem:** `CausalBase.py` imports `from text_processing.Preprocessing import Preprocessing` using a bare module name. The existing `sys.modules` aliases (`config`, `utils`) did not cover `text_processing`, causing `ModuleNotFoundError` on Section 0 import.

**Fix:** Added to the smoke test cell:
```python
import dackar.text_processing as _dackar_tp
_sys.modules.setdefault('text_processing', _dackar_tp)
```
Verified: smoke test imports `CausalSentence` successfully; 3/3 fixtures pass in the `dackar_libs` environment.

