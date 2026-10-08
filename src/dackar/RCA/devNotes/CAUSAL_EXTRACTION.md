# RCA Causal Extraction — Design Decisions and Results

**Status:** Curated development record. Distilled April–May 2026 from the two causal-extraction
enhancement plans, the document-similarity extraction plan and the alarm/SOE episode-mining plan.

This document captures *why* the pipeline extracts causal relations from documents the way it
does and *what the multi-dataset evaluation found*. It is a **development-choices and
development-results** record, not a description of current code. For the live extraction code
(`ner/causal_condition_adapter.py`, `doc_extraction/`) and the TSKR scorer that consumes it, see
[`../ARCHITECTURE.md`](../ARCHITECTURE.md); where this document and the code disagree, the code
wins.

---

## 1. Where causal extraction sits

Causal extraction is a **document pre-processing step**, not part of the per-run orchestration
(Steps 0–6). It runs once per document at ingestion time and produces versioned records that the
RCA run later queries. Keeping it out of the per-run path is a deliberate choice: it makes each
RCA run deterministic and adds no extraction latency to execution. Extracted fields are stored as
metadata beside the raw record; the pipeline consumes them as structured inputs, never as live
outputs.

The records feed three places in a run:

- **Step 1** resolves each extracted `inferred_fm_label` against the KG failure-mode list by
  embedding similarity (auto-resolve above a high cosine threshold).
- **Step 2d** uses semantic matches as a ranking contribution for similar-event scoring.
- **Step 3** uses them to compute an *effective* recurrence count, so a failure mode recorded in
  different words across past CRs is still counted as a recurrence.

## 2. The problem

Step 3 (recurrence) and Step 2d (similar events) originally matched past records on structured
fields: `component_id`, `fm_id`, `event_type`, `actuation_type`. The limitation is fundamental:
**most CRs and WOs carry no structured `fm_id`.** They are free text written by operators and
maintenance staff with inconsistent terminology. A formal failure-mode ID appears only when a
prior RCA closed the record against the FMEA taxonomy, which is the exception. The result is that
`recurrence_count` understates true recurrence and `novel_pattern` fires on failure modes that
have unmatched prior occurrences. The goal of causal extraction is to enable **semantic matching
on what records actually say** — identified effect, assessed cause, inferred failure mode — rather
than on exact ID fields.

## 3. Extraction architecture decisions

**One record per causal chain, not per document.** A single CR may describe several
cause/effect pairs. Producing one record per document would blur them into a coarse embedding that
matches specific failure-mode queries poorly. Each linked (cause, effect) pair becomes its own
record referencing the same `doc_id`; query-time deduplication by `doc_id` prevents one document
counting as several recurrence hits. A document with no extractable causal language still produces
one *null* record (confidence `low`) so it is never silently absent from the store.

**A bounded document scope.** Extraction applies only to plant-specific past-event records:
`CR`, `WO`, `RCA`, `ECA`. It deliberately excludes `FMEA` (which defines the taxonomy, so it is
reference context, not a past event) and `SOP` / `MANUAL` / `BULLETIN` / `OE` (procedures and
industry experience, not plant events — recurrence is a plant-record concept). Fleet-level
recurrence from OE documents is a future option, not in this scope.

**Rule-based extraction is primary; the LLM is a bounded fallback.** The dep-tree extractor
(`causal_condition_adapter`) runs first; the LLM fires only to repair weak statements or to
supplement a weak dep-tree result. The hard constraint, stated repeatedly because it matters in a
nuclear context: **the LLM is used for extraction only and must never infer a mechanism not stated
or implied in the text.** Using it to deepen a symptom-only record into a mechanism-level cause is
prohibited — `cause_is_symptom` stays `True` and `assessed_cause` reflects what is written.

**Confidence is tiered and feeds scoring fractionally.** Records are `high` / `medium` / `low`.
A `cause_is_symptom` record (the assessed cause is an observable effect, not a mechanism) carries
half weight. Semantic matches contribute *fractional* values to the recurrence count, never
integer increments, so a single low-confidence near-match cannot by itself push a novel event over
the recurrence threshold.

**The feature is gated off by default.** `enable_semantic_recurrence = False` in both
`OrchestratorConfig` and `TSKRTemporalScorerConfig`. Semantic augmentation must be explicitly
enabled; existing pipeline behavior is unchanged until it is. Thresholds (`similarity_threshold`
0.75, `fm_id_resolution_threshold` 0.80) are documented defaults that **require empirical
calibration against labeled plant data before production use**.

**Vocabulary is loaded from the curated project keyword files, not hardcoded.** The adapter had
drifted from the `data/*.csv` keyword files that `ConjectureEntity`, `CausalSentence` and
`CausalSimple` already use (for example 68 of 79 curated causal verbs were missing). Loading causal
verbs, prep connectors, conjecture terms and health-status terms from those files at import time
keeps the adapter in sync and closes the coverage gap.

**Negated causal statements are a positive asset, not noise.** "The trip was *not* caused by
sensor failure" eliminates a hypothesis — it is informative. Rather than writing negated statements
as low-confidence records that compete with positive evidence, they are routed to a dedicated
`ruled_out_mechanisms` field (added end-to-end: `DocExtractionRecord`, Chroma metadata, adapter)
that Step 2d can use to *penalize* candidates whose label matches a ruled-out cause. This is a
routing and schema change, not new extraction work.

## 4. What the evaluation found

A six-dataset evaluation (DS1–DS6a, 179 entries, 485 ground-truth relations) was run on raw CR/WO
text. The honest results, preserved here because they justify the architecture:

- **Extraction coverage is high; span fidelity is modest.** 84% of ground-truth relations produced
  at least one statement, but mean span F1 was 0.27 (cause 0.24, effect 0.29). The dep-tree finds
  *where* a causal relation is far more often than it captures *exact* spans.
- **One extractor does all the work.** 100% of detected relations across all datasets came from the
  dep-tree fallback (`_dep_causal_fallback`). `CausalSentence` and `CausalSimple` require upstream
  SSC entity annotations that are absent on raw text, so they stayed inert in evaluation. This is
  expected for raw-text runs, but it means the evaluation cannot catch regressions in those two
  paths — a smoke test with injected SSC spans was added as a regression gate.
- **There is a structural ceiling that only an LLM can break.** Implicit causation (no connector,
  domain inference), reversed causal order, counterfactual and negated relations together account
  for roughly 54% of the adversarial DS5 set and are near-0% recall. No amount of dep-tree
  refinement recovers them; they are structurally unreachable without an LLM. This is why LLM
  supplementation (gated, opt-in) exists at all.
- **Empty effect spans were the highest-value non-LLM fix.** When the causal verb's effect sits
  under `xcomp` ("caused the pump *to fail*") or `pcomp` ("led to the system *shutting down*") the
  effect came back empty, which scored recall 0 and blocked mechanism overlap, forcing LOW
  confidence. Expanding effect collection to those two patterns was the single structural change
  with the most direct LOW-to-MEDIUM confidence impact.
- **Cross-sentence demonstratives were a systematic miss.** A sentence beginning "This caused…"
  parses with a contentless pronoun as the grammatical cause. A deliberately narrow, rule-based
  coreference step (one sentence back, demonstratives only, subject position only) recovers the
  dominant cross-sentence pattern in condition reports without a full coreference model.

### The threshold calibration decision

A sweep of both Jaccard thresholds (`_CHAIN_JACCARD_THRESHOLD` 0.35, `_ENTITY_LINK_JACCARD_THRESHOLD`
0.40) produced **no source change**. Chain-linking scored highest at a very loose 0.10, but that is
an artifact — loose thresholding creates false chain links that accidentally match ground-truth
tokens; it does not reconstruct chains better. The real failure mode is extraction quality (empty or
pronoun cause text, often only one statement per entry), not the linking threshold. Entity linking
was flat at 80% across the whole sweep; the two misses fail at lemma overlap, not at any threshold,
because the cause text shares no tokens with a one-word FM label. **Both constants were kept** —
dropping the chain threshold to 0.10 would produce many false links in production.

### A metric caveat worth preserving

The evaluation's recall metric (token overlap on cause and effect spans) measures extraction
fidelity, not downstream workflow value. What actually matters is whether the extracted
`cause_text`, resolved through entity linking, produces the *correct* `inferred_fm_label`. A cause
phrased differently from the gold span can still resolve to the same FM. The intended primary KPI
for this work is **FM-resolution rate**, with token overlap as a secondary diagnostic.

## 5. Semantic recurrence — status and the deferred join

The document-similarity path is a two-stage pipeline: structured extraction into a Chroma
`doc_extractions` collection at ingestion, then embedding similarity search at run time. Each record
embeds `identified_effect | assessed_cause | inferred_fm_label` (health states and the procedural-
deviation score are metadata filters only — they have low semantic content that would dilute the
embedding). The embedding model version is stored with the collection, and a query-time mismatch
raises an error rather than producing silently wrong similarities.

Implementation status as of this record:

- **Extraction adapter and store (Phases 1–2): implemented.** `DocExtractionAdapter`,
  `DocExtractionRecord`, `DocExtractionStore` with upsert / query / `resolve_fm_candidates` /
  model-version guard.
- **Step 3 integration (Phase 3): implemented and gated.** The TSKR scorer accepts a store, computes
  `effective_recurrence_count`, and sets a `near_match_pattern` flag when only near-threshold matches
  exist (so novelty-uncertain cases surface to the analyst instead of being silently included or
  excluded).
- **Step 2d dimension (Phase 3b): deferred on a schema gap.** Adding a semantic dimension to
  similar-event scoring needs a way to join a past event to its originating document, but
  `kg_context.past_events` records carry `fm_id` and `component_id`, not `doc_id`. Closing this
  requires a `source_doc_id` field on past events (a KG schema change). Deferred pending that
  alignment.
- **Calibration (Phase 4): pending labeled data.** The `history_score` lookup table was calibrated
  for integer exact-match counts and needs recalibration for a float effective count; the precision
  and recall targets below cannot be validated without a labeled set.

The success criteria were set deliberately high because, in a nuclear context, a false positive (an
analyst treating a novel event as a known recurrence) is as dangerous as a missed recurrence:
precision at least 0.80 (target 0.90), recall at least 0.70, false-positive rate on same-component
different-FM pairs at most 0.10, and no change to exact-match recurrence for well-tagged records.

## 6. Episode mining for alarm/SOE sequences — planned

The pipeline treats each alarm or signal anomaly as an independent flag; it does not exploit the
**ordering and co-occurrence structure** of an alarm cascade. Frequent episode mining was planned to
close this gap. This remains a **pre-implementation design** — no code was written — but the design
choices are worth preserving:

- **Two modes.** Single-event mode (match the current alarm sequence against a library) is the
  primary RCA use; fleet mode (discover recurring episodes from a historical corpus) is offline.
- **Non-overlapping occurrence-based frequency**, so sub-episode frequency is anti-monotonic and
  pruning is efficient and there is no double-counting across windows.
- **A mandatory span constraint** (default 72 hours for process-plant cascades, 60 seconds for
  SOE-level protection sequences) — without it an episode spanning weeks is not causal.
- **Serial vs parallel episodes.** A serial episode is an ordered cascade (the condenser-vacuum
  archetype); a parallel episode is co-occurring alarms on *different* trains or divisions, which is
  a lightweight, high-value common-cause-failure fingerprint feeding the CCF signal.

Open questions gating implementation: the canonical SOE schema and whether it differs from
`alarm_log.json`; whether a fleet corpus exists to seed the library or it starts empty; and how the
episode score should combine with the TSKR confidence formula.
