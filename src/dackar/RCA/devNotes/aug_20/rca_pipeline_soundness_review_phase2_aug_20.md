# RCA Pipeline Soundness Review — Phase 2 (Rest of the Pipeline)

**Date:** 2026-08-20
**Reviewer role:** Independent causal-logic / systems-engineering assessor
**Scope (Phase 2):** everything around the causal core reviewed in Phase 1 —
- data-coverage & quality propagation (`causality_engine_v32.py` uncertainty/coverage methods)
- KG context construction / candidate universe (`orchestrators/kg_context_builder.py`)
- evidence retrieval + supersession (`orchestrators/evidence_retriever.py`, `orchestrators/supersession.py`)
- signal-evidence propagation DAG (`signal_evidence/builder.py`, `orchestrators/signal_evidence_builder.py`, `kg/kg_query_utils.py`)
- optional-phase failure visibility / manifest (`orchestrators/rca_reasoning_orchestrator.py`)
- LLM synthesis path & validation posture (`synthesis/rca_synthesizer_v31.py`, `orchestrators/llm_clients.py`)

**Method:** design + code reading only (no execution). Independently verifies/extends the June 6 SE review (`diagrams/june_5/rca_workflow_se_review_june_5.md`) and continues the Phase-1 report (`rca_causal_soundness_review_aug_20.md`). Findings are numbered **P-#** to distinguish from Phase-1 **N-#** and June-6 **F-#**.

**Yardsticks (unchanged):** engineer requirements (`RCA_Engineer_Needs_Requirements - 2.md`), IAEA ASSET / AP-913, and formal-causality guardrails (precedence ≠ causation, confounding adjustment, evidence-gap honesty, conservative bias under sparse data).

---

## ✅ Remediation status — Workstream 1 (updated 2026-08-20)

The two **Tier-1 visibility fixes** are **fixed, tested, and merged** (full suite green: 1896 + 7 slow).

| ID | Status | What changed | Tests |
|---|---|---|---|
| **P-6** | ✅ Resolved | All four swallowed optional phases (fm-id resolution, CMMS context, signal-episode search, cross-pattern linkage) now append `{phase, artifact, error_type, error, impact}` to `optional_artifact_failures` → surfaces in `run_manifest.pipeline_warnings`. | `test_ws1_review_fixes_aug20.py::test_p6_*` |
| **P-1** | ✅ Resolved (truncation visibility) | `kg_context.provenance` now carries an `expansion` block and per-family `truncation` stats (cap, total_matched, retained, dropped_count, dropped_ids, `truncated`) + top-level `truncation_occurred`. (The `ORDER BY`/reproducibility half is now also done — see P-9.) | `test_ws1_review_fixes_aug20.py::test_p1_*` |
| **P-4** | ✅ Resolved (relevance gate) | `supersession.py` now gates authority-based supersession on relevance: a higher-authority hit only supersedes a lower-authority hit when it is at least *nearly as on-point* (relevance within `_RELEVANCE_SUPERSEDE_MARGIN = 0.15`, using `metadata.relevance_score`/`semantic_overlap` when present, else `support_score`). Off-point high-authority evidence no longer erases on-point support; retained hits carry `supersession_relevance_retained` provenance and a bundle-level count. Recency tiebreak scoped to equal-rank buckets so retained hits aren't re-erased. | `test_p4_supersession_relevance_aug20.py` |

Also resolved from Phase 1 in the same workstream: **F-1** (gate honest-labelling) and **F-2** (HoP taxonomy migration). N-2 (Phase 1) largely resolved in parallel (labelling + confidence cap). Partially addressed: **P-2, P-7**. **P-5 and P-8 resolved (2026-08-20).**

---

## 0. Executive verdict

The supporting pipeline is **defensively engineered and, in several places, more careful than typical**: zero-failure-mode runs fail loud, "no data" is cleanly separated from "evidence against," the signal DAG requires temporal ordering *and* topological reachability (not mere co-anomaly), and LLM output cannot invent a primary cause. As a data-to-decision scaffold for a system engineer, it is sound and auditable.

The Phase-2 issues cluster around **three themes**, none of which are safety defects (the analyst stays accountable), but all of which affect how far a *causal* conclusion can be trusted:

1. **Completeness is silently bounded.** The hypothesis universe is the KG neighborhood (2-hop containment + 1-hop port connectivity) intersected with populated FMEA. Causes outside that envelope, or absent from FMEA, simply never become candidates — and several truncations (past-events top-10, docs top-20) are **not recorded** anywhere the analyst can see (P-1). This is the completeness face of Phase-1's causal-depth problem.
2. **Degraded runs can look clean.** June 6 F-5 is confirmed unchanged: fm-id resolution, CMMS, signal-episode, and cross-pattern failures are swallowed with a log line and do **not** reach the manifest's `optional_artifact_failures` (P-6). Combined with a weak data-quality penalty (P-7), a data-starved run can present a confident-looking card.
3. **A few "causal" judgments are shallower than their labels.** Contradiction detection is keyword/intent-based rather than semantic negation (P-3); supersession can let a weakly-relevant high-authority document erase the most on-point evidence (P-4); common-cause detection is keyed on graph edges the builder rarely emits, so CCF under-fires (P-2, reinforcing Phase-1 N-3); and the richest causal-chain signal the pipeline computes (`position_type` in the signal DAG) is mostly discarded downstream (P-5, reinforcing N-1/F-6).

Net: **the scaffolding is trustworthy as decision support; its main weaknesses are silent incompleteness and under-surfaced degradation, plus a handful of shallow causal judgments that should be labelled honestly.**

---

## 1. Verification of the June 6 non-causal findings

| June 6 ID | Finding | Verified status (2026-08-20) | Evidence |
|---|---|---|---|
| **F-5** | Optional-phase failures partially silent | **✅ Resolved (2026-08-20).** fm-id, CMMS, signal-episode, and cross-pattern failures now also record structured entries to `optional_artifact_failures`. | see P-6 |
| **F-12** | LLM synthesis path unvalidated end-to-end | **Confirmed.** `DummyLLMClient.generate_json` always raises (`llm_clients.py:31-34`) → every dev/test run takes the deterministic fallback; `OllamaLLMClient` path has no golden-card coverage. | see P-8 |
| **F-6 / §4.6** | Coverage quality reaches engine only at refine, not initial generate | **Confirmed.** `_apply_coverage_quality_adjustment` is a refine-time step; v1 ranking is not coverage-adjusted. | `causality_engine_v32.py:2241` |

---

## 2. Phase-2 findings

### P-1 (Med-High) — Hypothesis universe is silently bounded; truncations are invisible  — ✅ *truncation-visibility half resolved (2026-08-20)*

> **Update (2026-08-20, WS1):** the *truncation-visibility* recommendation is implemented — `kg_context.provenance` now records `expansion` stats and per-family `truncation` (cap, total matched, retained, dropped count + IDs, `truncated`) plus `truncation_occurred`. The **`ORDER BY` reproducibility gap (P-9)** is now also closed (path/OE/document/past-event queries deterministically ordered; OE ordered before its `LIMIT`). Remaining: the **hop/edge-type bounding of the hypothesis universe**.


Two structural bounds jointly define what the pipeline can ever propose:

- **Neighborhood shape:** `_expand_neighborhood` expands only `has_part_usage` up to `max_hops` (default **2**) plus exactly **one** connector hop (`kg_context_builder.py:310-345`). Functional/causal edge types in the schema (`UPSTREAM_OF`, support relations, etc.) are **not** traversed. A legitimate cause >2 containment hops away, or reachable only via a non-modeled edge, cannot appear as a candidate regardless of telemetry strength.
- **FMEA population:** candidate failure modes are exactly the `failure_mode` nodes `APPLIES_TO` neighborhood components (`:445-463`), with **no synthesis/stub**. A component with no FMEA row contributes **zero** hypotheses — invisibly.

Zero-total FMEA is well guarded (governance `red` → default hard abort — a genuine strength). But **partial** losses are silent:
- past events sorted and sliced to `max_past_events=10`, docs to `max_documents=20`, OE to `max_oe_documents=10` — with **no truncation count** recorded; `kg_context.provenance` carries only `builder` + `run_id` (`:140-143`).
- neighborhood/ FM ordering for `upstream_paths` and OE docs has **no `ORDER BY`**, so ordering (and thus which items survive a downstream cap) can vary across Neo4j versions.

**Why it matters causally:** the tool's coverage model (12 categories) certifies that categories were *considered*, but says nothing about candidates dropped by hop-bounding or top-N. An engineer reading a clean card cannot tell that the true cause was 3 hops away, or that past-event #11 (the matching recurrence) was truncated. This is the completeness counterpart to Phase-1 N-1 and directly touches requirement §3.9 ("explicit evidence-gap representation") and §3.2 (topology-grounded reasoning).

**Recommendation:** record expansion stats and every truncation (counts + dropped IDs) into `kg_context.provenance` and surface a manifest attention flag when any cap binds; add `ORDER BY` to the path/OE queries for reproducibility (§6 defensibility).

### P-2 (Med) — Common-cause/CCF detection is keyed on edges the builder rarely emits (reinforces N-3)  — ⚠️ *partially addressed (2026-08-20)*

> **Update (2026-08-20):** the *index-recognition* half is implemented. `_build_common_cause_index` no longer exact-matches only `{connected_support, support_environment, support_system}` (names the expansion never emits); it now recognises shared-support / functional-coupling edges by **semantic family** (case-insensitive substring: `support*`, `connects_port`/`connector`, `power*`, `suppl*`, `cool*`, `service_water`, `instrument_air`, `lube`, `shared*`) via `_is_support_dependency_edge`, while still excluding pure containment (`has_part_usage`, which feeds the separate upstream/adjacency signal) to avoid CCF over-fire. So wherever the KG carries a support/coupling edge — however named — the strong `shared_dependency` signal (weight 0.30) can now fire directly instead of only via the cluster-proxy fallback. Additive (existing fixtures lack these edge types → zero golden shift). Tests: `test_p2_common_cause_index_aug20.py`. **Still open (decisions):** (a) making the expansion actually *emit* connectivity/support edges into `upstream_paths` (a KG-context/topology change needing live-KG validation + golden re-check), and (b) the N-3 explain-away (discount downstream symptom candidates once a CCF fires — a ranking-magnitude change).

`_build_common_cause_index` derives support dependencies from `kg_context.upstream_paths` edges typed `connected_support` / `support_environment` / `support_system` (`causality_engine_v32.py:3652-3686`). But `_expand_neighborhood` emits path edges from `has_part_usage` / port relationships (`kg_context_builder.py:386-391`) — **not** those support types. So `support_dependency_ids` is frequently empty and the explicit CCF path under-fires; the engine falls back to a symptom-convergence proxy (≥3 affected components, ≥6 anomaly signals) which is coarser.

**Why it matters:** CCF screening is a first-class requirement (§3.3, "blindness to common-cause failures"). Combined with Phase-1 N-3 (symptoms of a shared cause each accrue independent structural credit), the pipeline is weaker at the exact failure class nuclear RCA cares most about.

**Recommendation:** either populate support-edge types into `upstream_paths` during expansion, or point the common-cause index at the relationships the builder actually emits; treat a fired CCF hypothesis as a trigger to explain-away its downstream symptoms (per N-3).

### P-3 (Med) — Contradiction detection is lexical/intent-based, not semantic refutation  — ✅ *resolved (2026-08-20)*

> **Update (2026-08-20):** implemented. `evidence_retriever.py` now runs a deterministic, tightly-scoped **negation/refutation detector** (`_negation_refutation_hit`) alongside the phrase list: a negation trigger (`no`, `not`, `did not`, `showed no`, `ruled out`, `no signs of`, …) followed within a 3-token window by a **degradation/failure-state term** is treated as refutation. It is scoped to the candidate's *own* named states (`cause_label`/`hypothesis_type` ∩ state vocab) so a negation about a different hypothesis in a multi-hypothesis snippet does not refute this candidate; the semantic-relevance gate is an added guard. On a hit, the negated state no longer accrues *support* (the negated word is suppressed as a support cue) and contributes contradiction at parity with a fixed cue (0.45). The phrase list remains as a fallback. Tests: `test_p3_negation_refutation_aug20.py`.

`evidence_retriever.py` classifies a snippet as contradicting mainly via phrase-list cues (`"no evidence of"`, `"not observed"`, `"within normal limits"`, …, `:109-118`) plus a contradiction-oriented query plan (`:383-395`) and structured alternate-attribution fields (`:703-736`). It is **not** a negation/refutation model. Refutation expressed in unlisted phrasing, or requiring scope/negation parsing, is missed.

**Credit where due:** `evidence_posture` cleanly separates `no_data` (zero hits, zero scores) from `contradicted` (hits with contradiction > support) (`causality_engine_v32.py:978-990`) — this correctly implements the §3.9 "evidence-against vs insufficient-data" distinction, which many systems get wrong.

**Why it matters:** disconfirming-evidence surfacing is the core bias-mitigation mechanism (§3.10). Keyword contradiction will systematically under-detect refutation, biasing postures toward `weak`/`no_data` rather than `contradicted`, which weakens the tool's ability to talk an engineer *out* of a favored hypothesis.

**Recommendation:** add a lightweight negation/scope check (the extraction layer already detects negation; reuse it) before classing a relevant snippet as merely contextual; keep the phrase list as a fallback.

### P-4 (Med-High) — Supersession ignores relevance; high-authority-but-off-point evidence can erase on-point evidence  — ✅ *resolved (2026-08-20)*

> **Update (2026-08-20):** implemented. `resolve_supersession` now applies a **relevance gate** before authority-based supersession: a higher-authority hit supersedes a lower-authority hit only when its relevance is within `_RELEVANCE_SUPERSEDE_MARGIN` (0.15) of the hit it would erase (relevance = `metadata.relevance_score`/`semantic_overlap` when present, else `support_score`). If every higher-authority hit is materially *less* on-point, the lower-authority hit survives with `supersession_relevance_retained=True` provenance (+ a bundle-level `supersession_relevance_retained_count`). The recency tiebreak is now scoped to equal-authority buckets so a retained hit cannot be re-erased by a newer, higher-authority off-point hit. Tests: `test_p4_supersession_relevance_aug20.py`.

`supersession.py` groups `analyzes_past_degradation` hits **by `candidate_id` only** and zeroes the support of every lower-authority hit in the group (`:197-201`), authority ranked RCA(1) > ECA(2) > CR(3) > OE(4/5) (`:71-87`), recency as tiebreak among equals. There is **no `cause_label`/semantic-overlap check**.

Consequence: a plant RCA with a `formal_conclusion` that only weakly mentions the candidate's failure mode will **supersede a highly on-point CR/WO** with strong semantic overlap — the most relevant evidence gets its support zeroed and role downgraded to contextual. (Contradiction scores are left intact, which is correct, but support erasure is relevance-blind.)

**Why it matters:** authority and relevance are different axes. Superseding by authority alone can discard the very evidence that best grounds a hypothesis, silently weakening a correct candidate. This is a soundness issue in the evidence layer, not just hygiene.

**Recommendation:** gate supersession on a minimum relevance/overlap between the superseding and superseded hits (same failure-mode/cause target), not just shared `candidate_id`; record superseded IDs so the analyst can inspect what was down-weighted.

### P-5 (Med) — Signal-DAG initiator scoring is coarse and its richest output is discarded (reinforces N-1/F-6)  — ✅ *resolved (2026-08-20)*

> **Update (2026-08-20, WS2 Part A):** the *primary-eligibility* half is implemented — the candidate `chain_position` (initiating/consequence) now drives primary selection (near-tie initiator promotion) and raises a review flag on consequence-as-primary. Uses the TSKR-derived `chain_position`.
>
> **Update (2026-08-20, P-5 completion):** the remaining three sub-issues are now closed. **(b)** the initiator (`root`/`common_cause_root`) is scored by its chain `path_score` instead of a flat `1.0` (`signal_evidence/builder.py:_per_candidate_scores`); **(c)** a root whose onset lead is not established — `OVERLAPS`/co-temporal, or lag `< _MIN_INITIATOR_LAG_HOURS=0.5h` — is discounted by `_COTEMPORAL_INITIATOR_FACTOR=0.6` and marked `initiator_lag_established=False`; **(a)** the signal-DAG `position_type`, `path_score`, and lag-established flag are wired onto the candidate at refine (`causality_engine_v32.py`: additive `signal_dag_position_type` / `signal_dag_chain_position` / `signal_dag_initiator_lag_established` / `signal_dag_path_score`, TSKR `chain_position` left intact for provenance), and the synthesizer raises an analyst flag when the primary sits at a `convergence_confluence` (downstream symptom) or is an initiator with an unestablished lead (`_apply_signal_dag_position_flag`). Full suite green (1953 tests), **zero golden shifts** — the golden cases do not feed signal-DAG root candidates, so the corrected scoring is exercised by the new targeted tests. Covered by `test_p5_signal_dag_position_aug20.py`.


The signal DAG is better than naive: an edge requires Allen `PRECEDES`/`OVERLAPS` **and** KG upstream reachability (`signal_evidence/builder.py:160-177`), feedback loops are truncated, convergence nodes are labelled `concurrent_cause_candidate` not root. Good.

But:
- **Root = 1.0 by position, not strength:** the earliest node on a topology-consistent chain gets `chain_position_score = 1.0` regardless of `path_score` (`:394-400`); `OVERLAPS` (simultaneous) is treated as upstream, so co-temporal anomalies on connected components produce a "propagation" edge without a real lag.
- **`position_type` is barely consumed:** downstream, `refine_with_evidence` uses only the scalar `chain_position_score` (30% blend) and zeroes `convergence_confluence` (`causality_engine_v32.py:1117-1124`); the rich `root` / `common_cause_root` / `intermediate` semantics are **not** copied to the candidate's `chain_position`, which is instead re-derived from TSKR temporal relations in a parallel path (`:1460-1474`).

**Why it matters:** the pipeline actually computes a plausible causal-chain topology (exactly what N-1 says is missing from depth assignment) and then largely throws it away. Wiring `position_type` into depth/primary selection would address N-1, F-6, and P-5 together.

**Recommendation:** use the signal-DAG `position_type` as a first-class input to causal depth and primary-cause eligibility; weight initiator score by `path_score`, and require a non-zero lag (not pure `OVERLAPS`) before calling a node an initiator.

### P-6 (Med-High) — Degraded runs can look clean (June 6 F-5 confirmed)  — ✅ *Resolved (2026-08-20)*

> **Update (2026-08-20, WS1):** implemented. Each of the four optional phases below now appends a structured `{phase, artifact, error_type, error, impact}` record to `optional_artifact_failures`, which flows into `run_manifest.pipeline_warnings`. A run with failed CMMS + cross-pattern is now distinguishable from a healthy one. Regression: `test_ws1_review_fixes_aug20.py::test_p6_cmms_build_failure_recorded_in_pipeline_warnings`.


Only Ishikawa records a structured failure to `optional_artifact_failures` (`:658`), which flows into the manifest (`optional_artifact_failures`, `optional_artifacts_degraded`, `pipeline_warnings`, `:3022-3025`). These optional phases do **not**:

```426:427:src/dackar/RCA/orchestrators/rca_reasoning_orchestrator.py
            except Exception as exc:
                LOGGER.warning("fm_id_candidate resolution failed — pipeline continues: %s", exc)
```

…and likewise CMMS context (`:448-452`, `LOGGER.error`), signal-episode search (`:698-701`), and cross-pattern linkage (`:719-720`) — each continues without a manifest-visible record. A run where CMMS context and cross-pattern evidence both failed is **indistinguishable in the manifest** from a fully-healthy run.

**Why it matters:** §3.9/§4 require uncertainty and degradation to be *visible*; §6 requires audit logging. A confident card built on a silently-degraded evidence base is the single most consequential auditability gap.

**Recommendation:** copy the Ishikawa pattern to every optional phase (append `{phase, error, degraded:true}` to `optional_artifact_failures`). Low-effort, high-value.

### P-7 (Med) — Data-quality penalty is weak; sparse-data conservatism is only partial  — ⚠️ *largely resolved (2026-08-20)*

> **Update (2026-08-20):** the *confidence* half is implemented (consistent with the N-2 decision — cap confidence, leave ranking magnitude unchanged). `data_limited_conclusion` was previously only annotated; the synthesizer now **caps** the primary/executive `confidence_label` at `medium` (reason `data_limited_conclusion`, downward-only) and raises an analyst attention flag listing the critical stream(s) below floor, on both the LLM and fallback paths. So a data-limited primary can no longer present as `high`-confidence. Tests: `test_p7_data_limited_confidence_aug20.py`. **Still open (decision):** lowering the 0.70 *quality-multiplier* floor for `critical_streams_below_floor` candidates — that changes composite magnitude/ranking (it shifts hand-tuned refine thresholds and would need golden re-validation), so it is deferred exactly like N-2's magnitude reduction.

`_apply_uncertainty_propagation` computes a per-candidate `quality_multiplier` from stream quality but **floors it at 0.70** (`causality_engine_v32.py:1734`), so *any* candidate keeps ≥70% of its composite regardless of how little data supports it; the `oe` stream even defaults to **0.35** when absent (`:1721-1722`). Coverage-level adjustment applies only at refine (F-6/§4.6). Together with Phase-1 N-2 (temporal credit fabricated from anomaly presence), a data-poor candidate can still reach a moderate, confident-looking composite.

**Why it matters:** §3.5 and §7 explicitly require "conservative bias in sparse-data contexts." A 30%-max penalty does not deliver strong conservatism; it mainly reshuffles ranking.

**Recommendation:** lower/remove the 0.70 floor for candidates with `critical_streams_below_floor`, and let `data_limited_conclusion` cap the confidence label (not just annotate it).

### P-8 (Low-Med, validation) — LLM path unexercised, but causal structure is deterministic (F-12 nuance)  — ✅ *resolved (2026-08-20)*

`DummyLLMClient` always raises (`llm_clients.py:31-34`), so every dev/test run exercises the **deterministic fallback**, and the `OllamaLLMClient` branch (normalization, `_validate_and_repair_llm_sections`, `_validate_card_semantics`) has no golden-card regression. F-12 stands: the narrative-generation path is unvalidated before production.

**Important mitigating strength (credit):** the synthesizer hard-rejects an LLM card whose `primary_hypothesis.candidate_id` is not a real input candidate (`rca_synthesizer_v31.py:130-137`), strips hallucinated secondary IDs (`:142-144`), and applies **deterministic** post-processing for safety-significance routing, metamodel phase-2, recommended-action depth mapping, epistemics, and `ccf_summary` on **both** LLM and fallback cards (`:173-190`). So the **causal structure** (which cause is primary, its depth, safety impact, CCF) is deterministic; the LLM only writes prose around it. The F-12 risk is therefore about narrative fidelity/citation, not causal logic — lower severity than a raw "LLM unvalidated" reading suggests, but still a gate before production use.

**Recommendation:** run the `OllamaLLMClient` path once against representative events, capture golden cards, and add a semantic-validation regression; until then keep the card labelled `synthesis_quality: deterministic` (already done — good).

> **Update (2026-08-20, P-8 resolved):** `test_p8_llm_synthesis_regression_aug20.py` closes this in two layers. **(A, always runs):** a scripted (deterministic) LLM output is driven through the *full* `synthesize()` LLM path and the post-processed card is snapshotted against a committed golden (`unit_tests/goldens/p8_llm_card_golden.json`, volatile ids/timestamps masked, self-seeding) — this is the golden-card regression the finding asked for, catching drift in normalization / `_validate_and_repair_llm_sections` / post-processing without needing a live LLM. It also asserts the review's key mitigation directly: the LLM card is `full_llm` (no fallback) **and** the deterministic structure is applied on top of it — human-performance block and CCF injected, and the WS2 chain-position flag (consequence-as-primary → points at the upstream initiator) and P-5 signal-DAG convergence flag both fire on the *LLM* card. **(B, opt-in):** a live-Ollama semantic regression (`RCA_LLM_GOLDEN=1`, skipped if unreachable) drives the real `OllamaLLMClient` and requires `_validate_card_semantics(card) == []`. Full suite green (1957 passed, 1 opt-in skip), zero golden shifts. This complements the pre-existing scripted-LLM branch coverage in `test_phase3_hallucination_guard.py` (full/partial/deterministic/hallucination transitions).

---

## 3. Consolidated Phase-2 issue register (severity-ranked)

| # | Finding | Type | Severity | Origin |
|---|---|---|---|---|
| **P-1** | Hypothesis universe silently bounded (2-hop/1-port neighborhood ∩ FMEA); ~~truncations unrecorded~~ | Completeness / soundness | **Med-High** — ⚠️ *partial:* truncations now recorded + queries now deterministically ordered (2026-08-20, see P-9); hop/edge bounding still open | New (completeness face of N-1) |
| ~~**P-4**~~ | Supersession ignores relevance; off-point high-authority evidence erases on-point support | Soundness (evidence) | ✅ **Resolved** (2026-08-20) — relevance gate + equal-rank recency scoping | New |
| ~~**P-6**~~ | Optional-phase failures swallowed; degraded run looks clean in manifest | Robustness / audit | ✅ **Resolved** (2026-08-20) | June 6 F-5 confirmed |
| **P-2** | Common-cause index keyed on edges builder rarely emits → CCF under-fires | Soundness | **Medium** — ⚠️ *partial (2026-08-20):* index now recognises support/coupling edges by family (no longer brittle exact-match); **N-3 explain-away surfacing done** (co-symptom-as-primary flag); builder edge-emission + explain-away ranking-discount deferred | New (reinforces N-3) |
| ~~**P-3**~~ | Contradiction detection lexical/intent, not semantic negation | Soundness / bias-mitigation | ✅ **Resolved** (2026-08-20) — candidate-scoped negation/refutation detector + support suppression | New |
| **P-5** | Signal-DAG initiator scoring coarse; `position_type` discarded downstream | Soundness | **Medium** — ✅ *resolved (2026-08-20):* initiator scored by `path_score`; co-temporal roots discounted + lag-flagged; `position_type`/path/lag wired to candidate + analyst flag | New (reinforces N-1/F-6) |
| **P-7** | Data-quality multiplier floored at 0.70; weak sparse-data conservatism | Soundness / §3.5 | **Medium** — ⚠️ *largely resolved (2026-08-20):* data-limited primary confidence now capped at `medium` + flagged; the 0.70 floor reduction (ranking magnitude) deferred by decision | New (extends §4.6) |
| **P-8** | LLM narrative path unvalidated (but causal structure deterministic) | Validation | **Low-Med** — ✅ *resolved (2026-08-20):* scripted-LLM golden-card regression on the full `synthesize()` LLM path + deterministic-structure parity (WS2/P-5 flags applied on the LLM card), plus an opt-in live-Ollama semantic regression | June 6 F-12 confirmed |
| ~~P-9~~ | Some KG queries lack `ORDER BY` → nondeterministic ordering across Neo4j | Reproducibility | ✅ **Resolved** (2026-08-20) — deterministic `ORDER BY` added to path/connectivity/OE/document/past-event queries (OE now ordered *before* its `LIMIT`) | New |

**Verified strengths (keep):** zero-FM fail-loud governance; `no_data` vs `contradicted` separation; signal-DAG requires temporal + topology; LLM hallucinated-ID hard rejection + deterministic causal post-processing; supersession restricted to analyzes-class.

---

## 4. Prioritised recommendations (Phase 2)

**Tier 1 — make incompleteness and degradation visible (highest value, low effort):**
1. ~~**P-6:** Record every optional-phase failure to `optional_artifact_failures` (copy the Ishikawa pattern).~~ ✅ **Done (2026-08-20).**
2. ~~**P-1:** Record expansion/truncation stats + dropped IDs in `kg_context.provenance`~~; ✅ **stats/dropped-IDs done (2026-08-20).** *Remaining:* raise a manifest attention flag when a cap binds (currently recorded in provenance only).

**Tier 2 — fix shallow causal judgments:**
3. ~~**P-4:** Add a relevance gate to supersession.~~ ✅ **Done (2026-08-20).**
4. ~~**P-5:** Wire signal-DAG `position_type` into causal depth / primary eligibility (jointly closes N-1, F-6, P-5).~~ ✅ **P-5 done (2026-08-20):** initiator scored by `path_score`; co-temporal roots discounted + `initiator_lag_established` flag; `position_type`/path/lag wired onto the candidate (`signal_dag_*`) with a synthesizer analyst flag for downstream-symptom / unestablished-lead primaries. (Depth-*label* derivation from `chain_position` remains deferred by the N-1 design decision.)
5. **P-2:** Align the common-cause index with the edges the builder emits (or emit support edges), and explain-away symptoms of a fired CCF (with N-3). ⚠️ *Partial (2026-08-20):* index recognition now family-based (`_is_support_dependency_edge`); **N-3 explain-away surfacing done** (engine `explained_away_candidate_ids` + synthesizer co-symptom-as-primary flag). *Remaining (decisions):* emit connectivity/support edges from the expansion; the N-3 explain-away **ranking discount**.
6. ~~**P-3:** Add negation/scope check before classing relevant snippets as contextual.~~ ✅ **Done (2026-08-20).**

**Tier 3 — conservatism & validation:**
7. **P-7:** Strengthen sparse-data conservatism (lower the 0.70 floor for below-floor candidates; let `data_limited_conclusion` cap confidence). ⚠️ *Largely resolved (2026-08-20):* `data_limited_conclusion` now caps confidence at `medium` + flags it. *Remaining (decision):* the 0.70 floor reduction (ranking magnitude) is deferred like N-2's magnitude change.
8. ~~**P-8:** Golden-card regression for the real LLM path.~~ ✅ **P-8 done (2026-08-20):** scripted-LLM golden-card regression on the full `synthesize()` LLM path + deterministic-structure parity, plus an opt-in live-Ollama semantic regression (`test_p8_llm_synthesis_regression_aug20.py`). ~~**P-9:** add `ORDER BY` to path/OE queries.~~ ✅ **P-9 done (2026-08-20).**

---

## 5. Combined picture (Phase 1 + Phase 2)

The two phases tell one coherent story. **The pipeline is a strong, auditable plausibility-ranking scaffold, but the specifically *causal* claims are approximated by loosely-coupled heuristics rather than a single causal model, and both its incompleteness and its degradation are under-surfaced.** The highest-leverage cross-cutting fix remains the one identified in Phase 1: **materialise and use an explicit causal chain** — the signal-DAG already computes most of it (P-5), depth just needs to consume it (N-1), and temporal credit should require a real propagation path rather than mere precedence (N-2). Around that, three low-effort visibility fixes (P-6, P-1, and honest labelling of the gate names F-1 / human-performance mapping F-2 from Phase 1) would substantially raise how far an engineer — or a regulator — can trust the card.

**Suggested next step:** if you want, I can turn the combined N/F/P registers into a single prioritised remediation backlog (owner-ready, with acceptance criteria and the specific functions to change), or drill into any one finding to confirm exact call sites and draft the fix.
