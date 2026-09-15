# RCA Causal-Soundness Review — Phase 1 (Causal Reasoning Core)

**Date:** 2026-08-20
**Reviewer role:** Independent causal-logic / systems-engineering assessor
**Scope (Phase 1):** the *causal reasoning core* of `src/dackar/RCA/` —
- run-time hypothesis ranking: `orchestrators/causality_engine_v32.py`, `orchestrators/tskr_temporal_scorer.py`, `orchestrators/temporal_relations.py`
- causal-depth & human-performance synthesis: `synthesis/rca_synthesizer_v31.py`
- document causal extraction: `causal/CausalBase.py`, `causal/CausalSentence.py`, `causal/CausalSimple.py`, `ner/causal_condition_adapter.py`

**Method:** design + code reading only (no execution). This review **independently verifies and extends** the June 6 systems-engineering review (`diagrams/june_5/rca_workflow_se_review_june_5.md`); it does not repeat its non-causal findings. Phase 2 (rest of the pipeline) is scoped separately.

**Yardsticks (Layer 0 — what "correct" means here):**
1. **Engineer requirements** — `diagrams/april_20/RCA_Engineer_Needs_Requirements - 2.md` (esp. §3.3 symptoms-vs-causes / CCF, §3.4 temporal causality, §3.9 uncertainty, §3.10 bias mitigation, §7 interpretability).
2. **RCA methodology** — IAEA ASSET three-question structure (what happened / why / why not prevented), AP-913 proximate→contributing→root depth, as already adopted in `diagrams/april_25/rca_metamodel.md`.
3. **Formal causality** — the discipline's minimal guardrails: temporal precedence ≠ causation (post hoc / cum hoc), confounding & common-cause must be adjusted for (not double-counted), direction of causation must be established, and any "confidence" attached to a cause must mean what it claims.

---

## ✅ Remediation status — Workstream 1 (updated 2026-08-20)

The following findings have been **fixed, tested, and merged into a green suite** (1896 + 7 slow unit tests passing). Details and acceptance tests below are updated in place; the register in §4 reflects the new status.

| ID | Status | What changed | Tests |
|---|---|---|---|
| **F-1** | ✅ Resolved (honest-labelling) | `physical_plausibility` gate now emits `check_basis: "minimum_structural_score"` + `operating_state_checked: false`, and its rationale states the operating-state/FMEA envelope was **not** evaluated. (A real operating-state check remains a separate enhancement.) | `test_ws1_review_fixes_aug20.py::test_f1_*` |
| **F-2** | ✅ Resolved (taxonomy migration) | HoP block + `rca_card` schema migrated to the v32 A–L taxonomy: included = **G** (human perf.), human facet of **I** (change control), **L** (organisational); **H/J/K** (design/surveillance/vendor) excluded and noted, no longer misattributed with §4.3 refs. Also fixed a latent bug where genuine **G** candidates were never surfaced. | `test_step6_conclusion.py` (HoP), `test_ws1_review_fixes_aug20.py` |

### Workstream 2 — Part A (chain-position eligibility) — updated 2026-08-20

Scoped deliberately to **Part A only** (keep category-based depth label; do not derive depth from `chain_position`). Implemented + tested (full suite green, 1904 tests):

| ID | Status | What changed | Tests |
|---|---|---|---|
| **F-6** | ⚠️ **Partially resolved** | Primary-cause selection is now chain-position-aware: `_select_candidates` promotes a **near-tie `initiating` candidate ahead of a top `consequence`** (margin `0.05`), and a deterministic step flags any `consequence`-as-primary (both LLM and fallback paths), pointing at the strongest upstream initiator. Uses the existing TSKR-derived `chain_position`. **P-5 completion (2026-08-20)** additionally surfaces the telemetry signal-DAG view: a primary at a `convergence_confluence` (downstream symptom) or an initiator with an unestablished lead now raises its own analyst flag. Depth *label* still category-based (by decision). | `test_ws2_chain_position_aug20.py`, `test_p5_signal_dag_position_aug20.py` |
| **N-1** | 🔷 **Deliberate design confirmed; depth label unchanged** | Decision-log review (metamodel "Causal Depth Levels" §; decision log §6) shows the category→depth mapping is **intentional**, while `chain_position` was specified as first-class but never wired to ranking — the latter gap is now closed for primary eligibility. Deriving the depth *label* from `chain_position` remains deferred by design decision. | — |
| **P-5** | ✅ **Resolved (2026-08-20)** | `chain_position` drives primary eligibility + review flag; **and** the signal-DAG `position_type` is now first-class — initiator scored by `path_score` (not flat 1.0), co-temporal (`OVERLAPS`/sub-threshold-lag) roots discounted + `initiator_lag_established` flag, `position_type`/path/lag wired onto the candidate (`signal_dag_*`), synthesizer raises an analyst flag for downstream-symptom / unestablished-lead primaries. Depth-*label* derivation still deferred (N-1 decision). | `test_p5_signal_dag_position_aug20.py` |

### N-2 — temporal honesty — Part (a) *labelling* done (2026-08-20)

| ID | Status | What changed | Tests |
|---|---|---|---|
| **N-2** | ⚠️ **Largely resolved (labelling + confidence cap)** | The co-occurrence temporal fallback (anomalies present, no matched TSKR pattern) is no longer silent: it is tagged `temporal_basis="cooccurrence_proxy"` + `temporal_support_unestablished=True` on the candidate `scores`; a deterministic analyst flag + uncertainty note fire when the **primary**'s temporal support is unestablished (both LLM and fallback paths); and the primary/executive **confidence label is capped at `medium`** (downward-only, reason `temporal_support_unestablished`) so co-occurrence can never yield a `high`-confidence causal claim. The magic `0.55`/`0.30` literals are now named constants. Ranking (composite score) is deliberately **left unchanged** → zero golden regeneration. | `test_n2_temporal_honesty_aug20.py` |

**N-2 still open (deferred by decision):** the *ranking* magnitude — reducing/removing the `0.55`/`0.30` proxy so co-occurrence no longer confers moderate temporal *score* — was declined in favour of the confidence cap (keeps ranking, blocks over-confident claims). Part (b) (gate temporal causal credit on an actual propagation path from the signal-DAG) also remains.

Still open from Phase 1: **N-2 (magnitude + part b), N-3 (ranking-discount half; surfacing done 2026-08-20)** (and the depth-label half of N-1, by decision). Resolved 2026-08-20: **F-3** (`prevention_analysis`), **N-4** (`score_interpretation`), **F-4** (`gate_disposition`), **N-5** (negation → non-evidence), **N-6** (`causal_graph` — first materialisation).

---

## 0. Executive verdict

**Does the software support system engineers doing RCA? Largely yes, at the workflow level.** The causal core is a disciplined, auditable, coverage-driven *hypothesis-ranking and screening* engine. Its stage structure, 12-category coverage model, evidence traceability, hard-gate discipline, sensitivity/near-tie surfacing, and human-in-the-loop posture are the right primitives, and they map onto recognised methodology. As **decision support that helps an engineer resist fixation and keep an audit trail, it works as intended.**

**Does the causal logic hold, in the formal sense?** **Partially — and the gaps are concentrated in exactly the places that carry the word "causal."** The engine is best described as a **weighted plausibility ranker over structural, temporal, telemetry, documentary and governance signals** — not a causal-inference engine. Three issues most affect whether an engineer can trust a *causal* conclusion:

1. **Causal *depth* is hardcoded to causal *category*** (A–F ⇒ proximate, G–K ⇒ contributing, L ⇒ root), and the `chain_position` field that could establish real depth is computed but never used to determine it. The pipeline therefore *cannot structurally conclude* that a hardware or design cause is the **root** — it will relabel it "proximate" and report root as "unresolved." **This is a conflation of cause *type* with causal *depth*.** (New — **High**)
2. **Temporal precedence is converted to causal weight with no mechanism check**, and temporal support is *manufactured* by fallback defaults whenever anomalies merely co-occur. This is the textbook post-hoc/cum-hoc risk, embedded in the score. (New/extends June-23 — **High**)
3. The **carried-over June 6 High findings are still present and unchanged** in code: the "physical plausibility" gate that checks no physics (F-1), and the human-performance block that mislabels design (H) and vendor (K) causes as human error with human-performance regulatory references (F-2).

None of these make the tool unsafe *as decision support* (the analyst remains accountable, and the design says so). They do mean **the causal claims should currently be read as "ranked plausibility with hygiene," not "established causation,"** and the card wording / gate names should say so until the checks behind them exist.

---

## 1. What the causal core actually computes (verified)

Two loosely-coupled layers, no shared causal model between them:

**Layer A — run-time hypothesis ranking (`causality_engine_v32.py`).** Each candidate failure mode gets five sub-scores combined by a weighted linear blend:

```3379:3386:src/dackar/RCA/orchestrators/causality_engine_v32.py
        raw = (
            w["structural"] * scores.get("structural", 0.0)
            + w["temporal"] * scores.get("temporal", 0.0)
            + w["telemetry"] * scores.get("telemetry", 0.0)
            + w["evidence"] * scores.get("evidence", 0.0)
            + w["governance"] * scores.get("governance", 0.0)
        )
```

Weights are SE-assigned per category (`_DEFAULT_SCORING_PROFILES`, `causality_engine_v32.py:94`), each profile summing to 1.0. Three hard gates (physical plausibility, timeline, barrier) run *after* scoring in `refine_with_evidence`. Retention is threshold-then-top-k.

**Layer B — document causal extraction (`CausalBase`/`CausalSentence`/`CausalSimple`).** spaCy dependency + keyword-lexicon extraction of `(cause → effect)` tuples from CR/WO/RCA text, feeding Chroma evidence and recurrence — **not** the primary FM ranking. Direction is assigned from connector lexicon, subject/object roles, passive-voice swaps, and (in `CausalSimple`) raw token order.

**Depth & human-performance (`rca_synthesizer_v31.py`).** Organises retained candidates into proximate/contributing/root and builds the Human & Organisational Performance block.

---

## 2. Verification of the June 6 causal-relevant findings

I re-checked each June 6 finding that bears on causal logic against the current code.

| June 6 ID | Finding | Verified status (2026-08-20) | Evidence |
|---|---|---|---|
| **F-1** | "Physical plausibility" gate only checks `structural_score < 0.20`; no operating-state/FMEA-condition test | **✅ Resolved (honest-labelling, 2026-08-20).** Gate now self-declares `check_basis="minimum_structural_score"`, `operating_state_checked=false`, and disclaims the envelope check in its rationale. Real operating-state check deferred. | `_apply_physical_plausibility_gate`, `causality_engine_v32.py` |
| **F-2** | Human-performance block mislabels H (design) and K (vendor) as human performance with AP-913 §4.3 human-performance refs | **✅ Resolved (taxonomy migration, 2026-08-20).** `HOP_CATEGORIES={G,I,L}`; H/J/K excluded and noted; schema `causal_category` enum → `[G,I,L]`; correct AP-913 refs per category. | `_build_human_performance_assessment`, `rca_synthesizer_v31.py`; `schemas/rca_card.json` |
| ~~**F-4**~~ | Gates execute *after* composite scoring, not elimination-first | **✅ Resolved (audit, 2026-08-20).** Rather than reorder the pipeline (churny, ranking-affecting), the elimination-first *semantics* are now explicit and auditable: an additive `gate_disposition` card block declares hard gates dispositive, reports `gate_order`, `primary_gate_status`, and lists every gate-eliminated candidate (sorted by descending score) so a high-scoring-but-eliminated hypothesis is visible rather than silently retaining its score. Raw per-candidate gate verdicts already existed (`hard_gates`, `primary_eligibility`, `ruleout`); this consolidates them. | `_build_gate_disposition`, `rca_synthesizer_v31.py`; `schemas/rca_card.json` |
| **F-6** | Primary-cause selection ignores `chain_position` (a consequence can outrank the initiator) | **⚠️ Partially resolved (2026-08-20, WS2 Part A).** `_select_candidates` now promotes a near-tie `initiating` candidate over a top `consequence`; `_apply_chain_position_review_flag` flags any remaining consequence-as-primary. Depth label still category-based (by decision). | `_select_candidates` / `_promote_initiator_over_consequence` / `_apply_chain_position_review_flag`, `rca_synthesizer_v31.py` |
| Allen blend | May-23 "Allen only raises temporal" | **Confirmed fixed.** `new = 0.75·old + 0.25·allen` raises or lowers; `follows` sets `temporal_contradiction`. | `_apply_allen_temporal_blend`, `causality_engine_v32.py:1913` |

**Conclusion:** the June 6 review remains accurate. Of its three High causal findings, **F-1 and F-2 are now resolved (2026-08-20, Workstream 1)**; **F-3 is now resolved too (2026-08-20)** via an additive `prevention_analysis` card block that makes *"why was it not prevented?"* a first-class output (defense-in-depth barrier assessment for the primary cause). F-3 was reinforced and partly re-explained by N-1 below.

---

## 3. Extended causal-soundness findings (new)

These go beyond June 6 and target the core question: *does the causal logic hold?*

### N-1 (High) — Causal **depth** is hardcoded to causal **category**; `chain_position` is unused

The synthesizer decides proximate/contributing/root purely from the candidate's A–L category:

```48:50:src/dackar/RCA/synthesis/rca_synthesizer_v31.py
    _PROXIMATE_CATEGORIES = {"A", "B", "C", "D", "E", "F"}
    _CONTRIBUTING_CATEGORIES = {"G", "H", "I", "J", "K"}
    _ROOT_CATEGORIES = {"L"}
```

Depth in RCA (ASSET/AP-913) is a property of **position in the causal chain** — the earliest occurrence that, if removed, prevents the event — not of the *kind* of cause. This mapping asserts, by construction, that:
- an equipment or design failure (A–H) can **never** be a root cause, only proximate/contributing;
- a Category-L "systemic/organisational" candidate is **always** the root, even when it is only a weak coverage scaffold (the metamodel itself flags L as the least-populated, hardest-to-evidence category).

Consequences that undermine causal soundness:
- The system is **structurally incapable of naming a hardware/design root cause.** A genuine design-deficiency root cause (H) is reported as "contributing," and `root_cause` comes back `"unresolved"` unless a thin L scaffold is retained (`_build_depth_summary`, `rca_synthesizer_v31.py:1900-1930`). This is the mechanism behind June 6 F-3.
- The candidate object *carries* a `chain_position` field (initiating/consequence) that is the correct signal for depth, but it is **only used to append a sentence to `why_primary`** (`rca_synthesizer_v31.py:1622-1630`) — never to order, select, or assign depth.

**This is the single most important causal-logic defect found in Phase 1**, because it makes the tool's headline causal conclusion (the depth-organised cause set) a re-labelling of category rather than a finding about the causal chain.

**Recommendation:** derive depth from `chain_position` + the ruled-in causal graph (initiating occurrence ⇒ root candidate), and treat category as an orthogonal attribute. At minimum, allow any category to occupy any depth and flag when the top-ranked candidate is a `consequence`.

### N-2 (High) — Temporal precedence → causal weight with no mechanism test, plus manufactured temporal support  — ⚠️ *labelling half resolved (2026-08-20)*

> **Update (2026-08-20):** implemented as *labelling + confidence cap* (ranking left unchanged by decision). The co-occurrence fallback below is now tagged `temporal_basis="cooccurrence_proxy"` / `temporal_support_unestablished=True` on the candidate `scores`; the synthesizer raises an analyst flag + uncertainty note when the primary relies on it; and the primary/executive `confidence_label` is **capped at `medium`** (reason `temporal_support_unestablished`) so co-occurrence can never produce a `high`-confidence causal claim. The `0.55`/`0.30` values are unchanged (now named constants). **Still open:** reducing the *score* magnitude (declined in favour of the cap) and gating temporal credit on a real propagation path (part b).


The temporal dimension rewards Allen relations by a fixed prior — OVERLAPS 0.90, CONTAINS 0.85, PRECEDES 0.75 (`temporal_relations.py:31-37`) — and folds that into the composite. There is **no independent check that a physical/functional mechanism connects the earlier interval to the event**; temporal order alone raises causal candidacy. That is precisely the post-hoc/cum-hoc fallacy the requirements warn against (`RCA_Engineer_Needs_Requirements §3.3`: "distinguishing symptoms from causes," "Granger causality … with caution").

Worse, temporal support is **fabricated by fallback** whenever any anomaly exists, even with no matched TSKR pattern:

```2842:2846:src/dackar/RCA/orchestrators/causality_engine_v32.py
        if tskr_pattern_match == 0.0 and anomaly_signals:
            tskr_pattern_match = 0.55

        if latency_consistency == 0.0 and anomaly_signals:
            latency_consistency = 0.30
```

With these defaults, a candidate for which the temporal scorer found *no* pattern still receives `temporal ≈ 0.35·0.55 + 0.25·0.30 = 0.27` purely because unrelated anomalies are present in the event window. Co-occurrence is thereby converted into positive temporal evidence — the exact confusion between correlation and causation the tool is meant to help engineers avoid.

**Mitigations that exist (credit where due):** `follows` correctly flags `temporal_contradiction` and trips the timeline gate; the Allen blend can lower as well as raise temporal. But these operate *after* the fallback has already injected baseline temporal credit.

**Recommendation:** (a) do not synthesise temporal credit from mere anomaly presence — a missing pattern should read as "no temporal evidence" (proxy/`temporal_score_quality` already exists to carry this), not 0.55; (b) gate temporal *causal* credit on the existence of a plausible propagation path (the `signal_evidence_builder` DAG already models propagation — require an actual path, not just precedence).

### N-3 (Medium-High) — Confounding / common-cause is not disentangled; structural deltas are additively stacked  — ⚠️ *partially resolved (2026-08-20)*

> **Update (2026-08-20, explain-away surfacing):** the reasoning-surfacing half is implemented. When the engine suspects a common cause (`common_cause_summary.suspected_common_cause`), it now emits `explained_away_candidate_ids` — the clustered co-symptoms other than the strongest shared-cause candidate (`top_common_cause_candidate_id`) — as additive provenance (`_build_common_cause_summary`). The synthesizer's `_apply_common_cause_explain_away_flag` raises an analyst attention flag + primary uncertainty when the **selected primary is one of those co-symptoms**, pointing at the shared-cause candidate and shared dependency ("review whether the common cause, not this symptom, is the true root"). Ranking is unchanged (additive; zero golden shift — `suspected_common_cause` requires score ≥ 0.45 **and** ≥ 2 clustered candidates, which the fixtures rarely hit). Tests: `test_n3_common_cause_explain_away_aug20.py`. **Still open (deferred, magnitude):** the actual *explaining-away discount* that down-weights a symptom's structural credit once its shared parent is hypothesised (a ranking change), plus non-additive confounder adjustment.

Structural corroboration is built by **adding independent bonuses** — symptom match, alarm signal, RPN, barrier delta, operating-point delta, CCF delta — with no adjustment for the fact that these signals are typically *not independent* (they are frequently joint symptoms of one upstream cause). Two failure modes downstream of a shared cause will each independently accumulate symptom + alarm + telemetry credit, so the engine can rank **several consequences of one common cause above the common cause itself** — the CCF-blindness pitfall called out in `RCA_Engineer_Needs_Requirements §3.3`.

Common-cause is represented, but only as a small *additive* Category-C delta and a `common_cause_index`, not as a confounder adjustment that *down-weights* correlated candidates. There is no backdoor-style correction and no notion of screening off symptoms once a shared parent is hypothesised.

**Recommendation:** when a common-cause/CCF hypothesis is present, apply an explicit *explaining-away* discount to its downstream symptom candidates rather than letting each accrue full independent structural credit.

### N-4 (Medium) — The composite "score" reads like a probability but is an uncalibrated ordinal blend

`composite_score ∈ [0,1]` is a weighted mean of heuristic sub-scores with hand-set weights and hand-set relation priors (0.90/0.85/0.75…), then surfaced with confidence labels and a `score_confidence_interval` whose width is literally `n_degraded/5` (`_apply_score_confidence_interval`, `causality_engine_v32.py:1978-1982`). None of it is calibrated against outcome frequencies. This satisfies the *form* of `§3.9` (probabilistic confidence, sensitivity, evidence-vs-insufficient-data) but not the *substance* — the number is an ordinal preference, not a probability, and the interval encodes data-availability, not statistical uncertainty.

This is acceptable for ranking **if labelled honestly**. The risk is an engineer (or auditor) reading `composite_score = 0.72` as "72% likely the cause."

**Recommendation:** label the composite explicitly as a non-probabilistic ranking score, and either calibrate against the `tests/test_case_*` ground-truth set or keep the numeric out of the analyst-facing card in favour of the ordinal confidence label.

> ✅ **Resolved (labelling, 2026-08-20).** Implemented the labelling half: the card now emits a constant, additive `score_interpretation` block (`score_type="ordinal_ranking"`, `is_probability=false`, `is_calibrated=false`, plus a note pointing analysts to `confidence_label` for likelihood and an `interval_meaning` clarifying that any interval reflects data availability, not statistical uncertainty), and the `composite_score` schema fields carry matching descriptions. Injected on both the LLM and deterministic-fallback paths; ranking magnitude untouched (zero golden regeneration beyond the additive block). *Calibration* against the ground-truth set remains the optional, separate enhancement. Tests: `test_n4_score_interpretation_aug20.py`.

### N-5 (Medium) — Extraction-layer direction is linguistically shallow and empirically weak on the causally hard cases

`CausalSimple` assigns cause/effect ordering by **raw token position** (`sorted(causalPairs, key=itemgetter(1))`, `CausalSimple.py:389`) — token order, not causal direction. The richer `CausalSentence` uses dependency roles and passive swaps, but the team's own evaluation (`diagrams/april_25/causal_extraction_enhancement_plan_may_9.md`) documents the ceiling on exactly the cases where *causal logic* matters:

- Direction accuracy **56%** (9/16).
- `negated_causality` **0%** extracted, F1 **0.000**.
- `reversed_causal_order` 29% extracted, F1 **0.007**.
- `counterfactual_conditional` F1 **0.024**; `implicit_no_connective` F1 **0.000**.

Because these extractions feed evidence/recurrence scoring (Layer B → Stage F), a **mis-directed or negation-blind extraction can raise the evidence/recurrence score of the wrong failure mode** — e.g. text saying "X did *not* cause Y" contributing support for X→Y. The coupling is indirect (it does not set the primary structural rank), which bounds the damage, but it is a live path by which document text can tilt the causal ranking in the wrong direction.

**Recommendation:** treat negated/hedged/reversed extractions as *non-evidence* (drop, don't score) until direction accuracy improves; the conjecture discount in `refine_with_evidence` is a good pattern to extend to negation.

> ✅ **Resolved (2026-08-20).** Two layers: **(1) source propagation** — `CausalSentence` already *detects* sentence negation (`isNegation`) but dropped it during stage-5 export; it is now carried through `collectExtactedCausals` → `to_stage5_dict()` as a `negated` flag (backward-compatible 8th tuple slot, default `False`). The adapter's pre-existing `_route_negated_statements` (moves negated tuples to `ruled_out_mechanisms`) and the doc-extraction recurrence skip already key off that flag, so negated causal tuples stop entering positive support/recurrence. **(2) Evidence-scoring backstop** — `_assess_hit_against_candidate` now detects a negated causal *link* ("did not cause", "not attributable to", "unrelated to") that is relevance-gated to *this* candidate, drops the lexically-driven support (caps to ≤0.10, re-routes to context) and counts it as contradiction (parity with the P-3 negated-*state* refutation, which it complements). Improving the extractor's raw direction accuracy (reversed/counterfactual) remains a separate ML enhancement. Tests: `test_n5_causal_negation_aug20.py`; extended `test_p3_*`. | `CausalSentence.py`, `CausalBase.to_stage5_dict`, `causal_condition_adapter.py`, `evidence_retriever.py` |

### N-6 (Medium) — Two causal vocabularies, no inspectable causal model

There is no single causal graph the analyst can see and contest. "Causal" reasoning is distributed across: KG topology (structure), Allen intervals (temporal priors), telemetry/evidence heuristics, and NLP tuples — stitched by embeddings and weights. This is why depth, direction, and mechanism are each approximated separately (N-1, N-2, N-5) rather than falling out of one model. `§3.2`/`§8` call for "movement from correlation analysis to causal modeling grounded in plant architecture" and a "Causal Reasoning & Hypothesis Management Layer" — the pieces exist but are not unified into an explicit, inspectable cause→effect graph with committed directionality.

**Recommendation (strategic):** consider materialising a per-run causal graph (nodes = occurrences/candidates, edges = mechanism/temporal/evidence links with direction) as a first-class artifact. It would make N-1/N-2/N-3 checkable by construction and directly serve the "interrogability" trust requirement (`§4`).

> ✅ **Resolved — first materialisation (2026-08-20).** The card now carries an additive, deterministic `causal_graph` block that consolidates the previously-scattered signals into ONE inspectable directed graph the analyst can see and contest: **nodes** = the target event + assessed candidates (each annotated with role — primary/contributing/alternative/eliminated — and *both* chain-position views, TSKR `chain_position` and telemetry `signal_dag_position`, which can differ by design); **directed edges** = chain-position precedence relative to the event (initiating/contributing → event; event → consequence) and shared-cause `explained_away` links from the common-cause summary; **undirected edges** = `near_tie` competition. `directionality_committed` reports whether any cause→effect edge could be committed. This makes N-1/N-2/N-3 checkable by construction and serves the interrogability requirement. It is ranking-neutral (reflects existing scores). A *unified upstream causal-reasoning layer* (replacing the two vocabularies rather than reconciling them in the card) remains the larger architectural follow-up. Tests: `test_n6_causal_graph_aug20.py`. | `_build_causal_graph`, `rca_synthesizer_v31.py`; `schemas/rca_card.json` |

---

## 4. Consolidated issue register (severity-ranked)

| # | Finding | Type | Severity | Origin |
|---|---|---|---|---|
| **N-1** | Causal depth hardcoded to category (A–F/ G–K/ L); `chain_position` unused → cannot name a hardware/design root cause | Soundness / methodology | **High** — ⚠️ *partial (2026-08-20):* `chain_position` now drives primary eligibility + review flag (WS2 Part A); category→depth mapping confirmed **deliberate** (metamodel), depth-label change deferred by decision | New (deepens F-3, F-6) |
| **N-2** | Temporal precedence → causal weight with no mechanism check; temporal support fabricated (0.55/0.30 fallback) from co-occurrence | Soundness (post-hoc/cum-hoc) | **High** — ⚠️ *largely resolved (2026-08-20):* co-occurrence proxy labelled `temporal_support_unestablished`, analyst-flagged, and confidence capped at `medium`; ranking magnitude left unchanged (by decision) and propagation-path gate still open | New (extends May-23) |
| ~~**F-1**~~ | "Physical plausibility" gate only checks `structural < 0.20`; no operating-state/FMEA test | Soundness / mislabel | ✅ **Resolved** (honest-labelling, 2026-08-20) | June 6 |
| ~~**F-2**~~ | HoP block mislabels H (design) & K (vendor) as human performance with §4.3 refs | Correctness / defensibility | ✅ **Resolved** (taxonomy migration, 2026-08-20) | June 6 |
| **N-3** | Confounding/common-cause not disentangled; additive structural deltas assume independence (CCF blindness) | Soundness | **Med-High** — ⚠️ *partial (2026-08-20):* explain-away **surfacing** done (`explained_away_candidate_ids` + analyst flag when primary is a co-symptom); ranking-discount deferred | New |
| ~~**F-3**~~ | "Why was it not prevented?" not a first-class output | Methodology completeness | ✅ **Resolved** (2026-08-20): additive `prevention_analysis` card block answers *which barriers failed / held and why* for the primary cause — deterministic defense-in-depth across PM/surveillance compliance, condition-monitoring detection, and the primary's barrier-logic gate; layers without inputs are `not_evaluated` (honest), never assumed-failed. Distinct from the structural `barrier_analysis` safety-function impact map. | June 6 — reinforced by N-1 |
| ~~**N-4**~~ | Composite score presented probability-like but uncalibrated ordinal blend | Transparency / §3.9 | ✅ **Resolved (labelling, 2026-08-20):** every card now carries a `score_interpretation` block (`score_type=ordinal_ranking`, `is_probability=false`, `is_calibrated=false`) plus schema descriptions on `composite_score` stating it is a non-probabilistic ranking number and that any interval encodes data availability, not statistical uncertainty. Ranking untouched; calibration against ground-truth remains a separate (optional) enhancement. | New |
| ~~**N-5**~~ | Extraction direction weak on negation/reversed/counterfactual (0% / 0.007 F1) can tilt evidence toward wrong FM | Soundness (bounded) | ✅ **Resolved (negation, 2026-08-20):** already-detected sentence negation now propagates into stage-5 (`negated` flag) so negated tuples are routed out of positive support/recurrence; plus a relevance-gated causal-link negation backstop in evidence scoring drops support for "X did not cause Y". Reversed/counterfactual *extraction accuracy* remains a separate ML enhancement. | New (team eval confirms) |
| ~~**F-4**~~ | Gates run after composite scoring, not elimination-first | Soundness / audit | ✅ **Resolved (audit, 2026-08-20):** additive `gate_disposition` card block makes hard-gate elimination dispositive and auditable (gate order, primary gate status, high-scoring gate-eliminated candidates surfaced). Full pipeline reorder to gates-first deferred as an optional refactor. | June 6 — confirmed |
| ~~**N-6**~~ | No unified, inspectable causal model; two causal vocabularies | Architecture / interrogability | ✅ **Resolved — first materialisation (2026-08-20):** additive `causal_graph` card block unifies chain-position, signal-DAG, common-cause/explain-away, near-tie and gate signals into one inspectable directed graph (nodes = event + candidates with both chain-position views; directed precedence + explain-away edges; undirected near-tie edges). Full upstream reasoning-layer unification remains a strategic follow-up. | New |
| ✓ | Allen temporal blend one-directionality | — | Resolved | Verified fixed |

---

## 5. Prioritised recommendations (Phase 1)

**Tier 1 — restore meaning to "causal" (do first):**
1. **N-1:** Derive depth from `chain_position` + ruled-in graph; allow any category at any depth; ~~flag `consequence`-as-primary~~ ✅ *done (2026-08-20, WS2 Part A):* chain-position primary eligibility + consequence flag implemented. *Remaining:* the depth-label derivation is **deferred by design decision** (category→depth mapping confirmed deliberate in the metamodel).
2. **N-2:** Stop synthesising temporal credit from anomaly presence; require a propagation path for temporal *causal* credit (reuse `signal_evidence_builder` DAG). ⚠️ *Largely resolved (2026-08-20):* co-occurrence proxy is labelled + analyst-flagged, and confidence is capped at `medium` so it cannot produce a high-confidence causal claim. *Remaining:* the ranking-magnitude reduction was declined in favour of the cap; the propagation-path gate (part b) is still open.
3. ~~**F-1 / F-2:** Either implement the operating-state/FMEA plausibility check or rename the gate; re-map H/J/K out of the human-performance block with correct references.~~ ✅ **Done (2026-08-20):** gate honestly labelled; HoP block + schema re-mapped to G/I/L.

**Tier 2 — honesty & robustness:**
4. **N-3:** Add explaining-away discount for symptom candidates once a common-cause/CCF parent is hypothesised. ⚠️ *Partial (2026-08-20):* explain-away **surfacing** done — engine emits `explained_away_candidate_ids`; synthesizer flags a co-symptom-as-primary pointing at the shared cause. *Remaining (deferred, ranking magnitude):* the actual structural-credit discount.
5. **N-4:** Label the composite as a non-probabilistic ranking score (or calibrate it against `tests/test_case_*`). ✅ *Resolved (labelling, 2026-08-20)* — `score_interpretation` card block; calibration remains optional.
6. **N-5:** Treat negated/reversed/hedged extractions as non-evidence. ✅ *Resolved (negation, 2026-08-20)* — negation propagated to stage-5 + causal-link negation backstop in evidence scoring; reversed/counterfactual extraction accuracy is a separate ML task.
7. **F-4:** Reorder to elimination-first (gates → score survivors). ✅ *Resolved (audit, 2026-08-20)* — `gate_disposition` card block makes elimination-first explicit/auditable (gates dispositive over score; eliminated candidates surfaced). Physical pipeline reorder remains an optional refactor.

**Tier 3 — strategic:**
8. **N-6:** Materialise a per-run inspectable causal graph as a first-class artifact. ✅ *Resolved — first materialisation (2026-08-20)* — additive `causal_graph` card block (nodes = event + candidates with both chain-position views; directed precedence + explain-away edges; undirected near-tie edges). Full upstream reasoning-layer unification remains the larger follow-up.

---

## 6. Open questions for the team

1. Is the category→depth mapping (N-1) a deliberate simplification, or an artifact? Was `chain_position` intended to drive depth and never wired in?
2. For N-2: is the 0.55/0.30 temporal fallback intended to keep candidates alive for review, or is it accidentally injecting causal credit? If the former, should it be carried as `review_required` rather than as temporal score?
3. Should the composite score be analyst-facing at all (N-4), or should the card lead with the ordinal confidence label?
4. Appetite for a unified per-run causal graph (N-6) as the Phase-2 target architecture?

---

## 7. Bottom line

The causal core is a **well-engineered, auditable plausibility ranker that genuinely helps an engineer run a disciplined, coverage-driven RCA** — the workflow-level answer to "does it do what it's supposed to?" is yes. But the **formal causal logic has real gaps in the components that carry the causal claim**: depth is a re-label of category (N-1), temporal precedence is treated as causation and even manufactured from co-occurrence (N-2), common causes are not disentangled from their symptoms (N-3), and two carried-over High findings (mis-named gate F-1, mis-mapped human-performance F-2) remain in code. Until N-1/N-2 and F-1/F-2 are addressed, the tool's outputs should be read and worded as **"ranked, screened plausibility with a strong audit trail,"** not as **"established root cause"** — which is consistent with its stated decision-support mandate, but needs to be reflected in gate names, the depth section, and the confidence labelling.

**Phase 2 (next):** apply the same Layer 0–5 lens to the rest of the pipeline — data-coverage/quality propagation, KG context construction, evidence retrieval & supersession, signal-evidence DAG, manifest/attention-flag completeness, and the LLM synthesis path (still `DummyLLMClient`, June 6 F-12).
