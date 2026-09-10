# Causal Extraction Improvement Plan — Post-Evaluation Cycle

**Date:** 2026-05-09
**Context:** DACKAR RCA pipeline — follows the multi-dataset evaluation (DS2–DS5) completed 2026-05-09
**Predecessor:** `causal_extraction_enhancement_plan_may_8.md` (Stages 1–4 + Fixes 1–6 + Improvements A–F — all complete)

---

## 1. Evaluation Baseline

### 1.1 — May 8 partial baseline (DS2–DS5, prior session)

Five datasets evaluated prior to full integration audit:

| Dataset | Entries | GT relations | Recall |
|---------|---------|-------------|--------|
| ds2 | 25 | 25 | 12% |
| ds3 | 27 | 102 | 11% |
| ds4 | 28 | 143 | 23% |
| ds5 | 30 | 79 | 5% |
| **TOTAL** | **110** | **349** | **15%** |

### 1.2 — May 9 full baseline (DS1–DS6a, all 6 datasets)

Run: 2026-05-09 · Evaluation: `tests/test_causal/test_causal_extraction.ipynb`
Integration status after session: 3 bugs fixed in `augment_chunks.py` (bare import, NameError, missing NERSeed bridge).
Extractor used in all entries: `dep_fallback` (no NER seed input → CausalSentence/CausalSimple not active, as expected for raw-text evaluation).

| Dataset | Entries | GT rels | Extracted% | cause F1 | effect F1 | avg F1 |
|---------|---------|---------|-----------|----------|-----------|--------|
| ds1 | 25 | 25 | 64% | 0.387 | 0.527 | 0.457 |
| ds2 | 25 | 40 | 32% | 0.139 | 0.223 | 0.181 |
| ds3 | 27 | 102 | 97% | 0.248 | 0.329 | 0.289 |
| ds4 | 28 | 143 | 97% | 0.308 | 0.270 | 0.289 |
| ds5 | 30 | 85 | 66% | 0.083 | 0.157 | 0.120 |
| ds6a | 44 | 90 | 94% | 0.267 | 0.347 | 0.307 |
| **TOTAL** | **179** | **485** | **84%** | **0.243** | **0.293** | **0.268** |

Note: "Extracted%" = fraction of GT relations for which at least one dep_fallback statement was produced. "avg F1" uses the best-matching extracted statement per GT relation.

**Stage metrics (all datasets):**
- Direction accuracy (DS1, n=16): **56%** (9/16) — 7 failures: empty cause (pronoun extraction) or wrong assignment
- Chain detection rate (146 chain-annotated entries): **18%** (26/146); mean full-chain score=0.340
- Entity linking hit-rate (DS1, n=10 G4-annotated): **80%** (8/10) — 2 misses are semantic (cause text shares no tokens with expected FM label)

**DS5 challenge-type breakdown (85 GT relations):**

| Challenge type | n | extracted% | avg F1 |
|----------------|---|-----------|--------|
| confounded_causality | 5 | 100% | 0.147 |
| counterfactual_conditional | 7 | 43% | 0.024 |
| cross_paragraph | 6 | 100% | 0.236 |
| implicit_no_connective | 4 | 25% | 0.000 |
| latent_gap_in_chain | 4 | 50% | 0.143 |
| multi_cause_convergence | 10 | 100% | 0.148 |
| negated_causality | 9 | 0% | 0.000 |
| operator_action_as_cause | 11 | 64% | 0.071 |
| passive_nominalisation | 7 | 100% | 0.231 |
| recursive_feedback | 11 | 100% | 0.265 |
| reporting_verb_ambiguity | 4 | 50% | 0.115 |
| reversed_causal_order | 7 | 29% | 0.007 |

**DS6a challenge-type breakdown (90 GT relations, 8 challenge types, 2 tiers):**

| Challenge type | n | extracted% | avg F1 |
|----------------|---|-----------|--------|
| contrastive_discourse_reversal | 14 | 100% | 0.141 |
| elliptical_causality | 1 | 100% | 0.000 |
| epistemic_hedge_causality | 11 | 82% | 0.425 |
| misdirection_revised_claims | 12 | 100% | 0.446 |
| nested_quotation_causality | 16 | 100% | 0.468 |
| probabilistic_causal_claims | 11 | 100% | 0.503 |
| table_list_embedded_causes | 16 | 38% | 0.038 |
| temporal_causal_ambiguity | 9 | 100% | 0.224 |

**DS6a by tier:**
- Tier 1 (n=44 rels): extracted=77%, avg F1=0.310
- Tier 2 (n=46 rels): extracted=96%, avg F1=0.305
- Tiers are nearly indistinguishable — both benefit equally from current dep_fallback; adversarial challenge type matters more than tier.

### 1.3 — Threshold calibration (May 9)

Calibration sweep over `np.arange(0.10, 0.85, 0.05)` using all chain-annotated and entity-annotated entries.

**`_CHAIN_JACCARD_THRESHOLD`** (current: 0.35):
- Sweep shows monotonically decreasing score from T=0.10 (0.122) to T=0.80 (0.051).
- T=0.10 appearing "best" is an artifact: very loose thresholding creates false chain links that accidentally match GT node tokens; it does not reflect better chain reconstruction.
- Root-cause analysis of chain detection failures: the dominant failure mode is extraction quality (empty/pronoun cause text, 0–1 statements extracted per entry), not the linking threshold. CR-002 has 2 stmts but cause is '' and 'which'; WO-009/WO-011/CR-013 each have exactly 1 stmt — no chain possible regardless of threshold.
- **Decision: keep at 0.35.** Dropping to 0.10 would produce many false chain links in production.

**`_ENTITY_LINK_JACCARD_THRESHOLD`** (current: 0.40):
- Flat hit-rate of 80% (8/10) across the entire sweep range (0.10–0.80).
- The 2 misses (WO-007: cause='Degraded insulation…' vs mech='degradation'; CR-003: cause='Water intrusion…' vs mech='degradation') fail at lemma-overlap (pass 3), not at the Jaccard threshold. The cause texts share no overlapping tokens with the 1-word FM label 'degradation' at any threshold.
- **Decision: keep at 0.40.** No threshold value recovers the 2 misses; they require semantic embedding matching or labeling enrichment.

**No source-file changes required from calibration.**

---

## 2. Root Cause Analysis of Remaining Gaps

### Gap 1 — Implicit and reversed-order relations (structural ceiling)

The dep-tree + regex approach requires a lexical causal connector to fire. It has no mechanism for:
- **Implicit causation:** no connector present; relation inferred from domain knowledge ("The cladding scratch was noted. Primary coolant iodine activity increased tenfold." — implicit causation via fuel failure)
- **Reversed order:** effect stated first, cause in a subordinate clause ("The reactor tripped because the feedwater level dropped")
- **Counterfactual / negated:** "The trip would not have occurred had the valve been tested" — the relation exists but is semantically negated

These account for 43/79 DS5 relations (54%) and represent a structural limit of the current approach. No amount of dep-tree improvement will recover them. **LLM extraction is the only path.**

### Gap 2 — Cross-sentence pronoun/demonstrative references

DS5 cross-sentence pairs like:
> S1: "The pump experienced bearing wear."
> S2: "This caused excessive vibration."

Sentence 2 has a pronoun (`This`) as the grammatical cause. The dep fallback processes each sentence independently and finds no referent for `This`. Sentences that begin with a demonstrative subject and a causal verb are systematically missed. This affects operator_action_as_cause (11 relations), recursive_feedback (11 relations), and latent_gap (4 relations).

### Gap 3 — Empty effect spans

The dep fallback returns `effect_text = ''` when the causal verb's syntactic effect is under `xcomp` or `pcomp` rather than `dobj`/`pobj`:
- "caused the pump to fail" → `fail` is `xcomp` of `caused`; `pump` is `xcomp`'s nsubj — not collected
- "led to the system shutting down" → `shutting` is `pcomp` of the `to` prep

Empty effects count as recall=0 in all dataset evaluations and prevent the `_best_overlapping_entity` link from firing (no text to overlap against G5 outcome spans), which blocks `has_mechanism = True` and keeps confidence at LOW.

### Gap 4 — Evaluation harness gaps

- DS1 (`causal_dataset.json`) missing from the test data directory; silently skipped
- No precision metric; false positive rate is unknown
- The evaluation uses `en_core_web_sm` (no word vectors) so Improvement D (embedding chain linking) never fires — the embedding path is untested in this evaluation

---

## 3. Improvement Plan

Three functional improvements + one evaluation harness improvement, in priority order.

---

### Improvement G — LLM trigger broadening

**Priority:** High
**Effort:** Small (infrastructure already exists; config change + trigger condition)
**Files:** `ner/causal_condition_adapter.py`

#### Problem

`_apply_llm_extract_all` fires only when **both** CausalSentence and CausalSimple return nothing. Because dep_fallback is wired inside the CausalSentence path and populates `extracted_causal_statements`, the `_has_useful_stage5_signal` gate passes and the function returns before LLM ever runs — even when every dep_fallback statement has an empty effect or confidence = 0.6.

Result: 0% recall on implicit (20 DS5 relations), 0% on reversed order (7 relations), 0% on counterfactual (7 relations) — all structurally unreachable without LLM.

#### Design

Add a secondary trigger condition **after** dep_fallback runs, gated on a new config flag:

```python
# In extract_stage5_causal_condition, after dep_fallback fills cs_result:
if llm_cfg and llm_cfg.get("extract_all") and llm_cfg.get("enabled"):
    weak_fallback = (
        llm_cfg.get("trigger_on_weak_fallback", False)
        and all(
            not s.get("effect_text") or s.get("confidence", 0) <= 0.60
            for s in cs_result["extracted_causal_statements"]
        )
    )
    if weak_fallback:
        _apply_llm_extract_all(
            cs_result, text, doc_id, chunk_index, doc_type, section_role,
            llm_cfg, embed_fn=embed_fn,
        )
```

New `llm_cfg` parameter:

```python
llm_cfg = {
    "enabled": True,
    "extract_all": True,
    "trigger_on_weak_fallback": True,   # NEW — fires LLM when fallback is weak
    "max_repair_calls": 3,
    ...
}
```

When `trigger_on_weak_fallback` is True, the LLM supplement runs whenever:
1. dep_fallback returned statements but all have empty `effect_text` OR confidence ≤ 0.60, AND
2. `extract_all=True` and `enabled=True`

The LLM output is merged with existing dep_fallback statements via `_merge_llm_statements` (already implemented). LLM-sourced statements tagged `"LLM_extract_all"` → `ConfidenceLevel.LOW` (existing rule preserved).

#### Cost control

LLM fires at most once per chunk (already the case in `_apply_llm_extract_all`). The existing `max_repair_calls` budget is separate and unaffected. The new trigger adds at most 1 LLM call per chunk where the dep_fallback produced only weak results — which is exactly where it adds most value.

#### Expected impact

- Implicit relation recall: 0% → ~30–40% (depends on LLM quality on nuclear domain text)
- Reversed order: 0% → ~50–60% (explicit connectors in reverse syntax are understandable to LLM)
- Counterfactual / negated: partial improvement; LLM can detect these and flag `negated=True`
- No regression on cases where dep_fallback already returns good results (trigger is gated on weak-only condition)

#### Acceptance test

Run DS5 evaluation with a mock LLM that returns a fixed correct answer for 3 test records; verify:
- `extracted_causal_statements` contains LLM-sourced statements
- `extractor_used = "LLM_extract_all"`
- `trigger_on_weak_fallback=False` disables the new path entirely (no regression)

---

### Improvement H — Lightweight cross-sentence coreference resolution

**Priority:** Medium
**Effort:** Medium
**Files:** `ner/causal_condition_adapter.py`

#### Problem

The dep fallback processes sentences independently. DS5 cross-sentence patterns where the second sentence opens with a demonstrative subject ("This", "These conditions", "It") are systematically missed because there is no referent in the local sentence for the dep tree to collect as cause.

Example (operator_action_as_cause):
> S1: "The operator isolated the cooling loop."
> S2: "This caused the heat exchanger to exceed its temperature limit."

S2 parses as: `This` (nsubj) → `caused` (ROOT) → `heat exchanger` (dobj). The dep fallback will find `cause_text = "This"` — a pronoun with no semantic content — and the entity linker will fail to match it against any G4 mechanism.

#### Design

Add `_resolve_sentence_coreference(sents)` as a pre-processing step inside `_dep_causal_fallback`, applied to the sentence list before the main Pass 1 loop:

```python
def _resolve_sentence_coreference(sents: List[Any]) -> List[Any]:
    """
    Rule-based demonstrative resolution for adjacent sentences.
    When a sentence's root subject is a demonstrative pronoun or
    generic reference ("this", "it", "these conditions", "the above"),
    substitute the head NP of the immediately preceding sentence.

    Scope: one-step back only (S[i] → S[i-1] referent).
    Does not modify the original spaCy spans; returns augmented text
    strings for dep-tree re-processing.
    """
    _DEMO = frozenset({"this", "it", "these", "that", "those",
                       "the above", "such conditions", "the situation"})
    ...
```

Implementation steps:

1. Before Pass 1, call `_resolve_sentence_coreference` on the sentence list
2. For each sentence whose root nsubj is in `_DEMO`, find the preceding sentence's root NP (largest noun chunk containing the sentence root)
3. Substitute: prepend the resolved NP text to the demonstrative sentence text and re-parse with `nlp`
4. Continue Pass 1 on the substituted text; tag the extracted statement with `"coref_resolved": True`

#### Scope constraint

Resolution is limited to:
- One sentence back only (S[i] refers to S[i-1])
- Deterministic demonstratives only (no pronoun `he`/`she`/`they` — those require full coreference)
- Subjects only (not objects or obliques)

This avoids the complexity of a full coreference resolver while capturing the most common pattern in condition reports ("This caused..." / "The above resulted in...").

#### Expected impact

- operator_action_as_cause: 0% → ~25% (operator actions are typically stated then referenced in adjacent sentence)
- recursive_feedback: 18% → ~30% (some feedback chains use demonstrative references)
- latent_gap_in_chain: 0% → ~15% (gap nodes often referenced demonstratively)
- No regression on single-sentence cases (trigger is only on demonstrative nsubj)

#### Fallback

When coreference resolution produces an unparseable or empty result, fall back to the original sentence. The `try/except` wrapper already present in `_dep_causal_fallback` covers this case.

---

### Improvement I — Effect subtree expansion

**Priority:** Medium
**Effort:** Small
**Files:** `ner/causal_condition_adapter.py`

#### Problem

Pass 1 of `_dep_causal_fallback` collects effect spans from `_EFFECT_DEPS = {"dobj", "obj", "pobj", "obl", "attr"}`. This misses two common syntactic patterns:

**Pattern A — `xcomp` infinitival complement:**
> "caused the pump **to fail**"

Parse: `caused` → `fail` (xcomp) → `pump` (nsubj of fail). The effect is the xcomp's nsubj (`pump`) combined with the xcomp head (`fail`). Neither is under `dobj`.

**Pattern B — `pcomp` after prepositional causal verb:**
> "led to the system **shutting down**"

Parse: `led` → `to` (prep) → `shutting` (pcomp) → `system` (nsubj of shutting). The effect phrase is `the system shutting down`. Current code collects `to` as the prep connector but finds no dobj/pobj under `led`.

Both patterns produce `effect_text = ''`, which:
- Counts as recall=0 in all evaluations
- Prevents `_best_overlapping_entity` from finding G5 outcome overlap
- Keeps `has_mechanism = False` → `ConfidenceLevel.LOW` in the adapter

#### Design

Extend Pass 1 effect collection with two additional dep patterns:

```python
# After failing to find dobj/obj/pobj — before returning empty effect:

# Pattern A: xcomp infinitival
for child in causal_tok.children:
    if child.dep_ == "xcomp":
        # subject of xcomp is the grammatical object of the causal verb
        xcomp_subj = next(
            (c for c in child.children if c.dep_ in {"nsubj", "nsubjpass"}), None
        )
        if xcomp_subj:
            effect_text = _np_subtree_text(xcomp_subj) + " " + child.text
        else:
            effect_text = _np_subtree_text(child)
        break

# Pattern B: pcomp after prep
if not effect_text:
    for child in causal_tok.children:
        if child.dep_ == "prep":
            pcomp = next(
                (c for c in child.children if c.dep_ == "pcomp"), None
            )
            if pcomp:
                pcomp_subj = next(
                    (c for c in pcomp.children if c.dep_ in {"nsubj", "nsubjpass"}), None
                )
                if pcomp_subj:
                    effect_text = _np_subtree_text(pcomp_subj) + " " + pcomp.text
                else:
                    effect_text = _np_subtree_text(pcomp)
                break
```

Also extend Pass 2 (regex connector scan): when the prep connector fires but the text after the connector boundary is a gerund phrase, collect the full gerund NP as effect rather than stopping at the first token.

#### Expected impact

- Empty `effect_text` rate: ~30% → ~10%
- Mean effect F1 across all datasets: 0.178 → ~0.25–0.30
- `has_mechanism = True` rate: increases because effect spans can now overlap G5 outcome entities
- Direct confidence upgrade path: dep_fallback + non-empty spans + G5 overlap → MEDIUM instead of LOW

#### Regression risk

Low. The new patterns fire only when `effect_text` is empty after the existing `_EFFECT_DEPS` check. Existing good extractions are unaffected.

---

### Improvement J — Evaluation harness hardening

**Priority:** Supporting
**Effort:** Small
**Files:** `tests/test_causal/test_causal_extraction.ipynb`, `tests/test_causal/data/`

#### Problems

1. **DS1 missing:** `causal_dataset.json` is not found in `data/`; loader silently skips it. The notebook loads DS1 from `causal_dataset.json` but that file does not exist at the path (`causal_dataset_1.json` exists). Either rename or add a path alias.

2. **No precision metric:** The evaluation measures recall (did we find the gold pair?) but not precision (are extracted pairs correct?). Without precision, the false positive rate of the dep fallback is unknown.

3. **Embedding evaluation untested:** `en_core_web_sm` has no word vectors; Improvement D (embedding chain linking via `_build_embed_fn`) never fires. The evaluation should note this explicitly and provide an optional `en_core_web_md` path.

4. **DS2 G4/direction annotations missing:** DS2 was loaded correctly after the fix, but `connector_direction` and `expected_g4_mechanism` are not annotated, so DS2 entries are excluded from Section 3 (direction) and Section 5 (entity linking). Annotating 10–15 DS2 entries would extend coverage.

#### Changes

**J1 — Fix DS1 path:** Add a path fallback in `_load_ds1` call:

```python
for fname in ('causal_dataset.json', 'causal_dataset_1.json'):
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        raw = json.load(open(fpath))
        all_entries.extend(_load_ds1(raw))
        break
```

**J2 — Add precision metric to Section 2:**

```python
# For each extracted statement: does it match ANY gold relation in the entry?
precision_rows = []
for r in results:
    for s in r['stmts']:
        matched_any = any(
            best_stmt_match([s], rel['cause_span'], rel['effect_span'])[1] >= 0.5
            for rel in r['entry']['relations']
        )
        precision_rows.append({'dataset': r['entry']['dataset'], 'matched': matched_any})
df_prec = pd.DataFrame(precision_rows)
print(f"Precision: {df_prec.matched.mean():.1%}  ({df_prec.matched.sum()}/{len(df_prec)})")
```

**J3 — Document embedding limitation explicitly:**

```python
if nlp.vocab.vectors.shape[0] == 0:
    print("WARNING: no word vectors in this model — Improvement D (embedding chain) is disabled.")
    print("         Re-run with en_core_web_md or en_core_web_lg to evaluate embedding chain linking.")
```

---

## 4. Sequencing and Dependencies

```
Improvement G — LLM trigger broadening       [no code deps; config + 1 trigger condition]
     │
     └── enables meaningful evaluation of implicit/reversed DS5 categories

Improvement H — Cross-sentence coreference   [depends on dep_fallback stability]
     │
     └── feeds better cause_text into dep_fallback Pass 1
         → better Jaccard chain linking (Stage 2)
         → better entity linking (Stage 3)

Improvement I — Effect subtree expansion     [independent; no deps]
     │
     └── feeds better effect_text into entity linking (Stage 3)
         → higher has_mechanism rate → MEDIUM confidence more often

Improvement J — Evaluation harness          [independent; supporting only]
     │
     └── run after G, H, I to measure updated baselines
```

Recommended order: **I → G → H → J** (I is smallest and unblocks cleaner baseline; G is highest ROI but needs a real LLM for meaningful evaluation; H adds complexity and should build on a stable dep_fallback).

---

## 5. Expected Metrics After All Improvements

Estimated against DS2–DS5 (no LLM in baseline; with LLM for Improvement G estimates):

| Dataset | Current recall | After I | After I+G (LLM) | After I+G+H |
|---------|---------------|---------|-----------------|-------------|
| ds2 | 12% | ~18% | ~35% | ~35% |
| ds3 | 11% | ~16% | ~30% | ~30% |
| ds4 | 23% | ~30% | ~45% | ~50% |
| ds5 | 5% | ~8% | ~25% | ~35% |
| **TOTAL** | **15%** | **~20%** | **~35%** | **~40%** |

These are estimates. The LLM contribution (Improvement G) depends on the model, prompt, and domain coverage. DS5 adversarial types (counterfactual, latent_gap, confounded) will remain partially refractory even with LLM — their inference requirements go beyond surface-level extraction.

---

## 6. RCA Workflow Impact Assessment

| Improvement | Adapter-level effect | Step 1 (FM resolution) | Step 2d (ranking) |
|-------------|---------------------|----------------------|------------------|
| **G — LLM trigger** | More `LLM_extract_all` statements; `ConfidenceLevel.LOW` (correct) | More `inferred_fm_label` candidates; low-weight but increases coverage for rare FMs | Increases `semantic_contribution` coverage for documents with implicit causation |
| **H — Coreference** | `cause_text` no longer `"This"` / `"It"`; entity linker can match | Higher `inferred_fm_label` hit rate for operator-action causes | `cause_is_symptom` flag more accurate for cross-sentence chains |
| **I — Effect spans** | Fewer empty `effect_text`; `has_mechanism` more often True | `identified_effect` populated in more records | MEDIUM confidence replaces LOW; higher `confidence_weight` in ranking |
| **J — Evaluation** | No production effect | Validates I correctly | Validates G+H correctly |

The highest-value improvement for the RCA workflow is **I** (effect span expansion) because it directly upgrades confidence from LOW to MEDIUM/HIGH without requiring LLM infrastructure. The highest-coverage improvement is **G** (LLM trigger) because it is the only path to recovering implicit relations, which are the dominant failure mode in real condition reports.

---

## 7. Files to be Modified

| Improvement | File | Change |
|-------------|------|--------|
| G | `ner/causal_condition_adapter.py` | Add `trigger_on_weak_fallback` logic in `extract_stage5_causal_condition` |
| H | `ner/causal_condition_adapter.py` | Add `_resolve_sentence_coreference`; wire into `_dep_causal_fallback` |
| I | `ner/causal_condition_adapter.py` | Extend Pass 1 effect collection with `xcomp`/`pcomp` patterns; extend Pass 2 gerund collection |
| J1 | `tests/test_causal/test_causal_extraction.ipynb` | `cell-load`: DS1 path fallback |
| J2 | `tests/test_causal/test_causal_extraction.ipynb` | `cell-span-metrics`: precision metric |
| J3 | `tests/test_causal/test_causal_extraction.ipynb` | `cell-setup`: word vector warning |
