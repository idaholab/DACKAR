# RCA Reference Guide — Code-to-Text Review Log

**Session date:** 2026-05-06
**Document under review:** `rca_workflow_reference_guide_may_6.md`
**Reviewer:** Diego Mandelli + Claude Code (code-to-text verification)

---

## Document audience (as stated in guide)

| Audience | Sections |
|---|---|
| Plant managers | Section 1 (overview, purpose, what it produces) |
| System engineers | Section 2 (workflow), Section 3 (data structures), Section 5 (step detail) |
| Data analysts | Section 5 (step detail), intermediate artifacts, Appendices A–C |
| Developers | Section 5 (implementation notes), Appendix B (LLM paths), schemas |

---

## Review status by section

| Section | Title | Status | Notes |
|---|---|---|---|
| §0 | Document conventions | **2 fixes — two broken anchor links corrected** | See §0 notes below |
| §1 | Problem, purpose, what it produces | **OK** | User confirmed |
| §1.1 | What the pipeline is designed to answer | **Fixed — coverage enforcement wording corrected (pass 2)** | See §1.1 notes below |
| §1.2 | What the pipeline produces | **Fixed — §8→§6 cross-reference corrected** | See §1.2 notes below |
| §1.3 | What it does not replace | **OK** | See §1.3 notes below |
| §1.4 | Main components | **OK** | All 7 components confirmed; see §1.4 notes below |
| §1.5 | Causal categories A–L | **Fixed — 4 items corrected (pass 1)** | See §1.5 notes below |
| §1.6 | Candidates vs. causal categories | **Fixed — FM source label corrected** | See §1.6 notes below |
| §2.1 | Input data inventory | **Fixed — bypass table corrected** | See §2.1 notes below |
| §2.2 | Workflow steps | **Fixed — optional extensions table added (pass 2)** | See §2.2 notes below |
| §2.3 | Allen temporal relations | **Fixed — Step 4 consumer removed; alarm field order; missing fields added** | See §2.3 notes below |
| §2.4 | Pattern recognition | **Fixed — Q5 and Q6 already present in document; log updated** | See §2.4 notes below |
| §2.5 | Analyst interaction points | **Fixed — 3 flags added; count updated to 11 (pass 2)** | See §2.5 notes below |
| §2.6 | Fleet, plant, industry OE | **§2.6.2 incomplete — semantic scoring dimension missing; rest correct** | See §2.6 notes below |
| §2.7 | Pipeline outputs | **§2.7.2 one section missing; §2.7.3 two artifacts missing, one wrong** | See §2.7 notes below |
| §3.1 | Knowledge Graph | **Three gaps fixed — FM properties, config param, edge types** | See §3.1 notes below |
| §3.2 | Chroma | **Composite score formula wrong; query table incomplete; metadata fields missing — all fixed** | See §3.2 notes below |
| §3.3 | KG and Chroma together | **Three issues fixed — two-phase evidence scoring, missing DocExtractionStore** | See §3.3 notes below |
| §5.1 | Step 0: Initialize | **2 fixes — asset_id mismatch claim wrong; scope seeding incomplete** | See §5.1 notes below |
| §5.2 | Step 1 part 1: Equipment model | **3 fixes — dual-flag abort, temporal enrichment description, missing params** | See §5.2 notes below |
| §5.3 | Step 1 part 2: Signal evidence | **3 fixes — no-params claim wrong, output structure incomplete, fallback behavior missing** | See §5.3 notes below |
| §5.4 | Step 2: Temporal analysis | **4 fixes — confidence formula (floor + penalty), novel_pattern field name, output fields, allen_map persistence** | See §5.4 notes below |
| §5.5 | Steps 3/3.5: Pattern recognition | **2 fixes — Step 3.5 output structure wrong, classification logic misrepresented** | See §5.5 notes below |
| §5.6 | Step 4: Candidate generation | **OK** | All weights, thresholds, gate names confirmed |
| §5.8 | Step 5: Evidence retrieval | **1 fix — Step 5a "two queries" wrong; actually 8 query types** | See §5.8 notes below |
| §5.12 | Step 6: Conclusion | **2 fixes — 4 attention flags missing from Step 6e; Step 6h pseudocode trigger wrong** | See §5.12 notes below |
| §6 | Output artifacts (reader-oriented) | **3 fixes — complete status missing; missing description wrong; stale §5.18 cross-ref** | See §6 notes below |
| §7 | Human and organizational contributors | **OK** | G/I/L labels and TC-6 cross-ref confirmed |
| §8 | Appropriate use and limitations | **OK** | DummyLLMClient and OllamaLLMClient confirmed; reproducibility claim correct |
| Appendix A | Pattern recognition details | **OK** | All code claims confirmed — no fixes needed; see Appendix A notes below |
| Appendix B | LLMs in the pipeline | **OK** | All class/method/file names confirmed; hallucination guard and post-processing claims correct |
| Appendix C | Similar event search | **2 fixes — semantic scoring dimension missing from C.3 table; multiply-then-cap order wrong** | See Appendix C notes below |
| Appendix D | Show-and-tell test case index | **OK** | All TC-1–TC-7 dirs exist; emphasis labels confirmed against fixtures |
| `rca_metamodel.md` | Category E note | **Fixed — stale 'currently ignored' note removed** | Category E operating-point scoring implemented in Phase 5 Finding H |

---

## Pass 1 edits applied (2026-05-06)

| File | Section | What changed |
|---|---|---|
| `rca_workflow_reference_guide_may_6.md` | §1.2 | `§8` → `§6` cross-reference |
| `rca_workflow_reference_guide_may_6.md` | §1.5 Cat B | Name restored to "Required support unavailable or degraded" |
| `rca_workflow_reference_guide_may_6.md` | §1.5 Cat C | Removed "lot or batch common-mode defects" (belongs to K); replaced with upstream topology CCF framing |
| `rca_workflow_reference_guide_may_6.md` | §1.5 Cat G | Name changed to "Human and organizational contributors"; `training_records` removed from primary driver; replaced with procedure documents and `operational_context` |
| `rca_workflow_reference_guide_may_6.md` | §1.5 Cat L | Added `training_records` (training program gaps) to primary driver data |
| `rca_workflow_reference_guide_may_6.md` | §1.6 | "(FMEA-derived)" label replaced with full KG registration description including ECA/RCA `CONFIRMED_CAUSE`/`MAY_CAUSE` paths |
| `rca_workflow_reference_guide_may_6.md` | §2.1 | `tskr_patterns` bypass: "Step 2b" → "Steps 2b + 3"; `causality_candidates` bypass: "Steps 3–4" → "Step 4 only" with note |
| `rca_workflow_reference_guide_may_6.md` | §2.3 | Removed "hypothesis ranking engine (Step 4)" from consumer list; added note that map is built after Step 4; fixed alarm end-field order (`acknowledged_at` before `cleared_at`); added `node_id` to per-node table; expanded summary to include `unknown_relation_nodes`, `dominant_causal_type`, `quality_flags.soe_nodes_capped` |
| `rca_metamodel.md` | Category E | Removed stale "currently ignored" note; replaced with correct description of `_operating_point_score()` and `op_delta` |

## Pass 2 edits applied (2026-05-06)

| File | Section | What changed |
|---|---|---|
| `rca_workflow_reference_guide_may_6.md` | §1.1 | Coverage enforcement sentence rewritten to distinguish `"no_supporting_data"` (always documented in `category_coverage`) from `"unknown"` (triggers attention flag); high-impact B/F/I/L note retained |
| `rca_workflow_reference_guide_may_6.md` | §2.2 | Added "Optional pipeline extensions" table after scope revision paragraph: semantic recurrence, evidence supersession, signal episode search, cross-pattern linkage, epistemics digests — each with config flag, insertion point, and what it adds |
| `rca_workflow_reference_guide_may_6.md` | §2.4 | Q5 (signal episode search) and Q6 (cross-pattern documentary linkage) were already present in document; review log status updated to reflect this |
| `rca_workflow_reference_guide_may_6.md` | §2.5.1 | Flag count updated 7 → 11; four flags added: near-match documentary pattern (`near_match_pattern=True`), FM resolution ambiguity (`fm_resolution_ambiguous=True`, [0.80, 0.88) band), signal episode index stale/missing, cross-pattern linkage gap (`"no_match"` / `"no_data"`) |

| `rca_workflow_reference_guide_may_6.md` | §0 | Two broken anchor links corrected: §1.1 link (stale "Central elements" title) and Appendix A link (stale "how-it-works" title) |

**All identified issues now applied. Review of full document is complete.**

---

## §0 — Code verification notes

**Verdict: 2 fixes — two broken anchor links. Everything else confirmed correct.**

### What checks out

- `check_doc_code_sync.py` exists at `diagrams/april_25/check_doc_code_sync.py` ✓
- All four annotation tags (`@code`, `@schema`, `@status`, `@reviewed`) match the script's compiled regexes (lines 38, 42, 46) ✓
- Symbol existence check: script resolves Python class, method, and `def` names per anchor — confirmed by per-anchor loop at lines 314–344 ✓
- `--stale` flag (line 402): warns when a referenced file's last git commit is after the max `@reviewed` date ✓
- `--strict-warnings` flag (line 406): exits 1 if any warning was printed ✓
- `main()` function at line 381 → `<!-- @code: diagrams/april_25/check_doc_code_sync.py | main -->` in §0 is correct ✓
- CI-safe: exits with integer codes 0/1/2, no side effects ✓

### Issues fixed

**Issue 1 — §1.1 anchor stale (broken link)**

Header link used old title "Central elements" which no longer matches the actual §1.1 heading "What the pipeline is designed to answer."

| | Before | After |
|---|---|---|
| Link text | `§1.1 — Central elements` | `§1.1 — What the pipeline is designed to answer` |
| Anchor | `#11-central-elements-what-system-engineers-should-focus-on` | `#11-what-the-pipeline-is-designed-to-answer` |

**Issue 2 — Appendix A anchor stale (broken link)**

Anchor used old section title text that no longer matches the actual Appendix A heading "Pattern recognition: limitations and improvement directions [WIP]."

| | Before | After |
|---|---|---|
| Anchor | `#appendix-a--pattern-recognition-how-it-works-limitations-and-improvements` | `#appendix-a--pattern-recognition-limitations-and-improvement-directions-wip` |

Appendix B, C, D anchors confirmed correct.

### Minor note

`@reviewed: 2026-04-25` on line 16 predates this review session (2026-05-06). Not a code claim error — bump at your discretion when the document version is formally revised.

---

## §1.1 — Code verification notes

**Verdict: OK, with one precision issue worth tracking.**

### What checks out

- **5-phase investigation structure** — conceptual, not a code construct; accurately summarizes pipeline intent.
- **Out-of-boundary signals flagged** — confirmed: `_apply_out_of_boundary_attention_flags()` injects into `rca_card.executive_summary.analyst_attention_flags` (orchestrator line ~705). Note: this runs at Step 6 finalization, not Step 0, even though §5.1 is cited in the Scoping row. The §5.1 reference is for scope *initialization*, which is correct; the flags apply later.
- **Sensitivity table in run_manifest** — confirmed: `_build_sensitivity_table()` in `causality_engine_v32.py` produces a per-source sensitivity table embedded in the manifest.
- **CCF as a scored dimension** — confirmed: CCF delta added to structural sub-score when `primary_causal_category == "C"`.
- **Three hard gates** — confirmed in v32: physical plausibility, timeline consistency, barrier logic (to be verified in detail at §5.6).

### Precision issue — "Missing coverage is an attention flag"

The document says: *"Missing coverage is an attention flag, not a silent omission."*

**Code reality:** There are two distinct coverage states in `category_coverage`:
- `"ruled_out"` — category assessed, no candidates generated; stored in `causality_candidates["category_coverage"]` with `reason_code = "no_supporting_data"`. **Does NOT trigger `analyst_attention_flags`.**
- `"unknown"` — applicability could not be determined. **Does trigger `analyst_attention_flags`** (the "Metamodel category coverage unknown" flag; high-impact categories B, F, I, L get an additional flag).

The document conflates these two cases. A category with no candidates is *documented* (in `category_coverage`), but it only rises to an *attention flag* if the pipeline also could not determine applicability. This is consistent with §2.5.1's flag description ("Metamodel category coverage unknown") but the §1.1 wording overstates it slightly for a manager audience.

**Recommendation:** Minor wording fix — distinguish "documented in the coverage record" (always) from "raised as an attention flag" (only when `unknown`). Not blocking.

---

## §1.2 — Code verification notes

**Verdict: OK, one broken cross-reference.**

Artifact names confirmed against `rca_reasoning_orchestrator.py` save calls:
`rca_card` ✓ · `run_manifest` ✓ · `causality_candidates` ✓ · `evidence_bundle` ✓ · `tskr_patterns` ✓ · `barrier_analysis` ✓ · `ishikawa_matrix` ✓

**Broken cross-reference:** §1.2 says supporting artifacts "are described by audience in **§8**." §8 in the document is "Appropriate Use and Limitations." The audience-oriented artifact table is in **§6** ("Output Artifacts — Reader-Oriented"). Fix: change `§8` → `§6`.

**Structural note:** The document has no Section 4 (structure goes §3 → §5). The gap appears intentional but is worth confirming no content is missing.

---

## §1.3 — Code verification notes

**Verdict: OK.**

Key claim: *"all candidate scoring is rule-based and deterministic; weights and thresholds are transparent and configurable"* — confirmed. `CausalityEngineConfigV32.weights` is an explicit dict (`structural` 0.30, `temporal` 0.20, `telemetry` 0.20, `evidence` 0.20, `governance` 0.10); `_combine_scores()` is a transparent weighted sum; `category_assignment_method` is tagged `"deterministic"` on every candidate.

The NER pipeline (`HybridNERPipeline`) contains a scikit-learn logistic regression, but it is used for document entity extraction (evidence pre-processing), not for candidate scoring. The claim is correctly scoped to scoring and is accurate.

The LLM used in RCA card synthesis (Step 6, `RuleValidatedRCASynthesizerV31`) is distinct from candidate scoring and is covered in Appendix B — no contradiction.

---

## §1.4 — Code verification notes

**Verdict: OK.**

All 7 components verified:

| §1.4 component | Code implementation | Status |
|---|---|---|
| Equipment and failure mode model | Neo4j KG via `Neo4jKGContextBuilder` | ✓ |
| Data coverage assessor | `_build_data_coverage_summary()` → `run_manifest` | ✓ |
| Temporal sequencer | `TSKRTemporalScorerV1` + `_build_allen_relation_map()` | ✓ |
| Hypothesis ranking engine | `RuleBasedCausalityEngineV32.generate()` | ✓ |
| Document evidence retriever | `ChromaEvidenceRetriever` | ✓ |
| Operating experience matcher | `_build_similar_event_list()` + `SimilarEventAdapter` | ✓ |
| RCA card synthesizer | `RuleValidatedRCASynthesizerV31.synthesize()` | ✓ |

Five scoring dimension names in the table ("structural fit, temporal support, telemetry match, documentary evidence, maintenance program posture") map exactly to code keys (`structural`, `temporal`, `telemetry`, `evidence`, `governance`). ✓

"Eliminated candidates are retained in a documented ruled-out list" — confirmed; ruled-out candidates carry `reason_code` in `causality_candidates`. ✓

---

## §1.5 — Code / metamodel verification notes

**Source of truth:** `rca_metamodel.md` (category definitions) and `causality_engine_v32.py` (keyword maps and scoring logic).

**Verdict: 4 issues — 2 substantive, 1 stale metamodel note, 1 minor.**

### Issue 1 — Category C: name and description conflict with metamodel (SUBSTANTIVE)

| | Reference guide | Metamodel |
|---|---|---|
| Name | "Upstream influence / common cause" | "Upstream Influence" |
| Description | "Inlet conditions outside design basis; **lot or batch common-mode defects across multiple trains**" | Inlet process conditions only (insufficient inlet flow, poor fluid quality, entrained gas, etc.) |

**Problem:** "Lot or batch common-mode defects" belongs to Category K in the metamodel ("Vendor and Supply Chain — Specification was correct but delivered item did not meet it, or a batch-level defect affects multiple installed components"). The code applies the CCF structural delta only for topology-based common-cause (when `primary_causal_category == "C"`, from KG connectivity). Lot/batch defects route through `vendor_supply_chain_records`, already mapped to "C, K" in §2.1 — correctly implying they feed K, not redefine C.

**Recommendation:** Remove "lot or batch common-mode defects across multiple trains" from Category C's description. Keep the CCF note but tie it explicitly to shared upstream topology. Category K already covers lot/batch.

### Issue 2 — Category G: `training_records` listed as primary driver data (SUBSTANTIVE)

Reference guide §1.5 Category G primary driver data: *"Work order text in Chroma, `training_records` (qualification data)"*

**Code reality:**
- `causality_engine_v32.py` `_CATEGORY_KEYWORDS` → `"L": ["systemic", "latent", "training", "safety culture", ...]` — "training" keyword mapped to **L**, not G.
- `_CATEGORY_KEYWORDS` for G: `["operator", "maintenance error", "calibration error", "procedure not followed", "human"]` — no training.
- `rca_metamodel.md` data table: `Training records → Candidates, Conclusion → **L**`

`training_records` informs Category L (systemic/organizational weakness — training program gaps), not Category G (human execution).

**Recommendation:** Move `training_records` out of Category G's primary driver data; add it to Category L alongside OE documents and CR reviews.

### Issue 3 — Category E: metamodel note is stale (METAMODEL NEEDS UPDATE, guide is correct)

`rca_metamodel.md` Category E still contains: *"`operational_context.operating_point` is collected by the pipeline but consumed by no scoring stage — this entire category is currently ignored."*

**Code reality:** Operating-point scoring was implemented in April 2026 (Phase 5 Finding H, confirmed in `rca_metamodel_decision_log_april_25.md`). `causality_engine_v32._operating_point_score()` implements a 7-mode base table; `op_delta = 0.12 × op_score` (cap +0.12) is added to the structural sub-score for Category E candidates.

The reference guide is **correct** (no such caveat). The **`rca_metamodel.md` needs its Category E note removed** to avoid confusion.

### Issue 4 — Category B and G names truncated (MINOR)

| Cat | Metamodel name | Guide name | What's dropped |
|---|---|---|---|
| B | "Required Support Not Available **or Degraded**" | "Required support unavailable" | "or Degraded" — a degraded (not absent) support system still causes failures |
| G | "Human **and Organizational** Contributors" | "Human execution" | Organizational scope; relevant for manager audience understanding G includes pre-job briefing failures and organizational enablers |

**Recommendation:** Restore "or Degraded" to B's name. For G, either restore "and Organizational Contributors" or add a parenthetical — the organizational aspect matters when explaining the G/L boundary to managers.

### What checks out in §1.5

- Categories A, D, F, H, I, J, K, L: names, descriptions, and primary driver data all consistent with metamodel and code.
- Causal depth column (A–F = proximate, G–K = contributing, L = root): consistent with AP-913 framework and how the RCA card synthesizer organizes its three-level output.
- The §1.6 boundary note between G and I (wrong execution vs. correct execution of wrong baseline) is consistent with the metamodel's explicit boundary note.

---

## §1.6 — Code verification notes

**Verdict: One inaccuracy — the "(FMEA-derived)" label on the Failure mode dimension is too narrow.**

### Background: two candidate generation paths

The engine has two distinct paths in `causality_engine_v32.py`:

**Path 1 — FMEA-path** (`_build_failure_mode_candidates`): iterates `kg_context.failure_modes`, populated by `kg_context_builder._fetch_failure_modes()` via:
```cypher
MATCH (fm:failure_mode)-[:APPLIES_TO]->(c)
```
Only KG-registered `failure_mode` nodes appear here. Typically ingested from FMEAs.

**Path 2 — Past event analog path** (`_build_past_event_candidates`): iterates `kg_context.past_events` (`abnormal_event` KG nodes) that have `CONFIRMED_CAUSE` or `MAY_CAUSE` edges to KG failure mode nodes. The `fm_id` on a past event comes from those graph edges — which can be populated from ECA/RCA findings entered into the KG, not just from FMEAs. If no FM link exists, `canonical_mechanism_id` falls back to `event_id` (degraded analog).

### What this means for ECA/RCA-derived failure modes

| What happened to the ECA/RCA finding | Candidate generated? | Path |
|---|---|---|
| Failure mode formalized into KG as a `failure_mode` node (regardless of whether source was FMEA or ECA/RCA) | Yes — full FMEA-path candidate | Path 1 |
| ECA/RCA created an `abnormal_event` KG node with `CONFIRMED_CAUSE` → existing `failure_mode` node | Yes — past event analog with matched FM | Path 2 |
| `abnormal_event` node exists in KG but no `failure_mode` link | Degraded analog only — `fm_id = None`, falls back to `event_id`; lower structural score | Path 2 |
| Finding lives only in Chroma document text, never entered into KG | No candidate — only influences evidence sub-score at Step 5 | N/A |

### Issue — "(FMEA-derived)" label in §1.6

The 4-tuple table describes the Failure mode dimension as *"Failure mode ID from the equipment model (FMEA-derived)"*. This is too narrow: FM IDs can also originate from ECA/RCA findings back-propagated into the KG via `CONFIRMED_CAUSE`/`MAY_CAUSE` edges on `abnormal_event` nodes.

**Suggested replacement:**
> Failure mode ID registered in the KG — sourced from FMEA ingestion, or from confirmed findings in prior ECAs/RCAs that have been entered into the KG as `CONFIRMED_CAUSE` or `MAY_CAUSE` links on past-event nodes.

**Corollary for plant event records staff:** a failure mode identified in an ECA or RCA that was never back-propagated to the KG cannot generate a new candidate. It can only raise or lower the evidence sub-score of existing candidates (via Chroma retrieval at Step 5). This is a data-quality dependency worth surfacing in the KG maintenance section (§3.1).

---

## §2.1 — Code verification notes

**Verdict: One error in the checkpoint-resume bypass table. Everything else correct.**

All required and optional inputs confirmed against `run()` signature in `rca_reasoning_orchestrator.py`. All schema names match code. Causal category assignments consistent with §1.5. The telemetry unavailability and SOE/PLC paired-data notes match code behavior.

### Error — `causality_candidates` bypass description

The pre-built artifacts table says:

| Artifact | Bypasses |
|---|---|
| `causality_candidates` | Steps 3–4 — candidate generation and initial ranking |

**Code reality:** The bypass checks are independent `if X is None:` guards:
```python
if tskr_patterns is None:        # builds tskr_patterns — contains Steps 2b + 3
    ...
if causality_candidates is None: # runs generate() — Step 4 only
    ...
```
Supplying `causality_candidates` alone skips only `causality_engine.generate()` (Step 4). Steps 2b and 3 (both embedded in `_build_tskr_patterns`) still run unless `tskr_patterns` is also supplied.

**Correct bypass mapping:**

| Artifact | Bypasses |
|---|---|
| `signal_evidence` | Step 2a — signal propagation chain build |
| `tskr_patterns` | Steps 2b + 3 — TSKR scoring and documentary pattern recognition |
| `causality_candidates` | Step 4 only — candidate generation (`causality_engine.generate()`) |
| `evidence_bundle` | Step 5a — Chroma evidence retrieval |

**Recommendation:** Change `causality_candidates` bypass description from "Steps 3–4" to "Step 4 only". Add a note that Step 3 is bypassed by supplying `tskr_patterns`, not `causality_candidates`.

---

## §2.2 — Code and source-document verification notes

**Verdict: Core 7-step sequence is accurate. Four implemented optional capabilities are absent from the step table.**

### Core step sequence — confirmed correct

The `run()` method in `rca_reasoning_orchestrator.py` executes in this order, consistent with the §2.2 table:

| Step | Code call | Confirmed |
|---|---|---|
| 0 — Initialize | `_stage_a_build_run_context`, `_enforce_input_guard_policy` | ✓ |
| 1 — KG + data context | `kg_context_builder.build()`, `_compute_kg_governance`, CMMS augment, `_enrich_past_events_temporal_metadata` | ✓ |
| 1 part 2 — Signal evidence | `_build_signal_evidence` | ✓ |
| 2b — TSKR | `_build_tskr_patterns` (carries Steps 2a + 2b + 3) | ✓ |
| 4 — Candidate generation | `causality_engine.generate()` + scope filter | ✓ |
| 5a — Evidence retrieval | `evidence_retriever.retrieve()` | ✓ |
| 5b — Allen map | `_build_allen_relation_map()` — after Chroma, before refine | ✓ |
| 5c — Refinement | `causality_engine.refine_with_evidence()` | ✓ |
| 5d — Auto-reentry | `_run_auto_reentry_if_needed()` | ✓ |
| 6a — Ishikawa | `ishikawa_evaluator.evaluate()` | ✓ |
| 6b — Barrier | `_compute_barrier_analysis()` | ✓ |
| 6c/2d — Similar events | `_build_similar_event_list()` | ✓ |
| 6d — Synthesis | `rca_synthesizer.synthesize()` | ✓ |
| 6e — Attention flags | All `_apply_*_attention_flags()` calls | ✓ |
| 6h — Scope expansion | `_detect_scope_expansion_signals()` | ✓ |
| 6i — Manifest | `_stage_g_finalize_manifest()` | ✓ |

### Missing from §2.2 — optional capabilities implemented but not described

These run inside `run()` gated by config flags (all default `False`). They are absent from the §2.2 step table and not mentioned as optional extensions anywhere in the section.

| Capability | Config flag | Where it runs in `run()` | What it does |
|---|---|---|---|
| **Semantic recurrence** (`resolve_fm_candidates`) | `enable_semantic_recurrence` | Between Step 1 KG build and Step 1 part 2 (signal evidence) | Resolves FM ID candidates in `DocExtractionStore` using KG failure mode embeddings; improves cross-pattern FM matching |
| **Phase C — Evidence supersession** (`_apply_supersession`) | `epistemics_policy_version` set | Between Step 5a (Chroma retrieval) and Step 5b (Allen map) | Applies epistemic classification policy to re-rank evidence bundle items before refinement |
| **Signal episode search** (`_build_historical_signal_episodes`) | `enable_signal_episode_search` | Step 6 block, before synthesis | Queries `PatternSearcher` (inverted index + NLCS/EMD) for historical signal episodes matching current anomaly fingerprint; produces `historical_signal_episodes` artifact; feeds cross-pattern linker |
| **Phase 2 — Cross-pattern linkage** (`_build_cross_pattern_evidence`) | `enable_cross_pattern_linkage` | Step 6 block, after signal episode search, before synthesis | `CrossPatternLinker` joins historical signal episodes with doc extraction records; produces `cross_pattern_evidence`; contributes attention flags to RCA card |
| **Phase D — Epistemics digests** (`_attach_epistemics_digests`) | `epistemics_classifier` injected | Step 6 block, just before synthesis | Attaches per-candidate `epistemics_digest` from classified evidence |

The last three (signal episodes, cross-pattern, epistemics digests) run in the Step 6 block before `rca_synthesizer.synthesize()`. They appear in attention flags (`_apply_signal_episode_index_attention_flags`, `_apply_cross_pattern_attention_flags`) and in the run manifest, so analysts can encounter their outputs with no explanation in the step table.

**Recommendation:** Add a paragraph or sub-section at the end of §2.2 listing these as optional pipeline extensions with their config flags. The cross-pattern capability in particular is a significant evidence enhancement that warrants a brief description for system engineers.

---

## §2.3 — Code verification notes

**Verdict: Core mechanics correct throughout. One error in the artifact lifecycle claim; minor omissions in output field tables.**

### What checks out

All the following confirmed exactly against `temporal_relations.py` and `rca_reasoning_orchestrator._build_allen_relation_map()`:

- Scores: OVERLAPS 0.90, CONTAINS 0.85, PRECEDES 0.75, DURING 0.30, FOLLOWS 0.10 ✓
- Evaluation order: FOLLOWS → PRECEDES → CONTAINS → OVERLAPS → DURING ✓
- `epsilon_hours` default 0.5 h ✓
- `interval_type` parameter with `closed`/`open`/`half_open_start`/`half_open_end` ✓
- `causal_candidate = True` for PRECEDES, OVERLAPS, CONTAINS only ✓
- Clock sync guard: `alarm_log.quality.clock_sync_ok = False` → `("unknown", 0.0)`; same for SOE ✓
- SOE records always point events (`Interval(start=ts, end=ts)`) ✓
- `max_soe_nodes` default 200; truncation applied before processing ✓
- All listed per-node output fields present in code ✓
- Summary fields `timeline_consistent`, `causal_nodes`, `contradiction_nodes`, `earliest_causal_onset` all confirmed ✓
- Blend formula `0.75 × TSKR_score + 0.25 × allen_base_score` confirmed (`ALLEN_ALPHA = 0.25`) ✓
- One-directional raise-only: `new_temporal = max(old_temporal, new_temporal)` ✓
- FOLLOWS → `temporal_contradiction = True` → candidate moved to `primary_block_reasons` ✓

### Issue 1 — `allen_relation_map` is NOT used at Step 4 (SUBSTANTIVE)

§2.3 states: *"The result — `allen_relation_map` — is a single shared artifact built once and reused by the **hypothesis ranking engine (Step 4 temporal blend)**, the evidence refinement pass (Step 5)..."*

**Code reality:** `_build_allen_relation_map()` is called **after** `causality_engine.generate()` (Step 4) and **before** `refine_with_evidence()` (Step 5c). The artifact does not exist when `generate()` runs and is therefore not available to Step 4.

What Step 4 *does* use is Allen-classified signal timing embedded inside `tskr_patterns` by the TSKR scorer's internal logic — but that is not the same artifact as `allen_relation_map`. The document conflates the two.

**Correct consumer list for `allen_relation_map`:**
- Step 5c — `refine_with_evidence()` via `_apply_allen_temporal_blend()` ✓
- Step 6h — `_detect_scope_expansion_signals()` ✓
- Run manifest — embedded in `_stage_g_finalize_manifest()` ✓

**Recommendation:** Remove "hypothesis ranking engine (Step 4 temporal blend)" from the consumer list. Replace with the corrected list above.

### Issue 2 — Summary and node fields not listed in output tables (MINOR)

The code produces these fields that the §2.3 output tables do not mention:

| Field | Location | What it is |
|---|---|---|
| `unknown_relation_nodes` | `summary` | Count of nodes where relation is `"unknown"` due to clock sync failure |
| `dominant_causal_type` | `summary` | Node type (`anomaly`/`alarm`/`soe_record`) with the most causal nodes |
| `soe_nodes_capped` | `quality_flags` | `True` when SOE log was truncated to `max_soe_nodes` |
| `node_id` | per node | Unique node key: `"anomaly::{sensor_id}"`, `"alarm::{alarm_id}"`, `"soe_record::{rec_id}"` |

Also minor: the document lists alarm end field precedence as `cleared_at` or `acknowledged_at`, but the code resolves `acknowledged_at` first (`alm.get("acknowledged_at") or alm.get("cleared_at")`). Priority order in the table is reversed from the code.

---

## §2.4 — Code verification notes

**Verdict: Incomplete — current §2.4 covers 4 pattern recognition questions correctly, but two implemented capabilities (signal episode search and cross-pattern documentary linkage) are absent.**

### What checks out in the existing 4 questions

- Question 1 (degradation trend): TSKR `onset_lag_hours`, `duration_profile`, Allen relation classification — all confirmed correct.
- Question 2 (recurrence / novelty): `novel_pattern` flag, `has_novel_patterns`, `signal_lessons_learned`, `similar_event_list` with `any_plant_match` — all confirmed. Note: `DocExtractionStore.resolve_fm_candidates()` (semantic recurrence) also enhances Question 2 when `enable_semantic_recurrence=True` but is not mentioned.
- Question 3 (CCF): Category C structural delta, `ccf_score`, `ccf_note`, `common_cause` block — confirmed correct. Cross-reference to Category K (lot/batch defects) is consistent with §1.5 Issue 1 fix.
- Question 4 (FM signature match): structural + telemetry + temporal sub-scores, FMEA-path candidates, `score_rationale` — confirmed correct.

### Missing — Question 5: signal episode search (`PatternSearcher`)

**Code:** `PatternSearcher` in `log_pattern_recognition/rca_pattern_search/searcher.py` + `metrics.py`

Three-metric coarse-to-fine retrieval against `IncidentIndex`:
1. Inverted-index lookup (event type intersection)
2. Jaccard pre-filter (discard below `min_jaccard`)
3. Full scoring: NLCS + EMD similarity → `combined_score = α·J + β·NLCS + γ·EMD`

Three complementary metrics:
- **Jaccard** (`|A∩B|/|A∪B|`): set overlap — did same event types occur? (order-agnostic)
- **NLCS** (`|LCS|/max(|A|,|B|)`, on deduplicated sequences): did they occur in similar order?
- **EMD similarity** (`1 − TV distance`, where `TV = 0.5·Σ|P(t)−Q(t)|`): did same types repeat with similar intensity?

Output fields on every `HistoricalSignalEpisode`: `jaccard_score`, `nlcs_score`, `emd_score`, `similarity_to_current`, `matched_events`, `query_only_events`, `episode_only_events`, `episode_density`, `known_rca`, `index_status` (`"indexed"` / `"stale"` / `"no_episodes_indexed"`).

Runs in Step 6 block gated by `enable_signal_episode_search`. Feeds cross-pattern linker.

### Missing — Question 6: cross-pattern documentary linkage (`CrossPatternLinker` + `DocExtractionStore`)

**Code:** `cross_pattern/rules.py`, `cross_pattern/linker.py`, `doc_extraction/store.py`

`CrossPatternLinker` joins `HistoricalSignalEpisode` records from PatternSearcher with `HistoricalDocExtraction` records from the Chroma-backed `DocExtractionStore`.

**Link confidence** (renormalized weighted formula from `compute_link_confidence()`):
| Term | Base weight | Present when |
|---|---|---|
| Signal similarity | 0.30 | Always |
| Temporal overlap | 0.20 | `event_time_confidence ≠ "absent"` |
| FM alignment | 0.20 | `fm_id_candidate` resolved |
| Document semantic similarity | 0.30 | Semantic score available |

Weights renormalized over present terms so missing dimensions do not silently deflate confidence.

**Linkage precedence levels** (`classify_linkage_precedence()`):
- Level 1: doc_id appears directly in episode source refs
- Level 2: `event_time_confidence ≠ "absent"` (temporal+asset link)
- Level 3: FM/semantic fallback

**Support posture** (`classify_support_posture()`): `"reinforcing"` (single / multiple_consistent) → `"weakly_supporting"` (mixed FM IDs) → `"conflicting"` → `"unresolved"`

**Linkage outcome** (`classify_linkage_outcome()`): `"linked"` / `"below_threshold"` / `"no_match"` / `"no_data"`

**Stale cap** (`apply_stale_confidence_cap()`): when `index_status="stale"`, link_confidence capped.

**DocExtractionStore semantics** (`SemanticMatch` fields):
- `epistemic_class`: 4-way (`affects_performance`, `monitors_performance`, `analyzes_past_degradation`, `characterizes_system`)
- `confidence_weight`: HIGH=1.0, MEDIUM=0.7, LOW=0.3
- `cause_is_symptom_factor`: 0.5 when assessed cause is itself a symptom, else 1.0
- `semantic_contribution = similarity_score × confidence_weight × cause_is_symptom_factor`
- `fm_id_candidate`: null at ingestion; resolved at run init by `resolve_fm_candidates()` when `enable_semantic_recurrence=True`

Runs in Step 6 block gated by `enable_cross_pattern_linkage`. Produces `cross_pattern_evidence`. Results feed `_apply_cross_pattern_attention_flags()`.

### Recommendation

Add two new questions (Q5 and Q6) to §2.4 covering the above capabilities. Update the "where results surface" table to include:
- `historical_signal_episodes` — ranked signal pattern matches with per-metric scores (Q5)
- `cross_pattern_evidence` — link confidence, support posture, linkage outcome per candidate (Q6)
- `SemanticMatch` epistemic fields and `semantic_contribution` formula (Q6)

---

## §2.5 — Code verification notes

**Verdict: §2.5.2, §2.5.3, §2.5.4 correct. §2.5.1 incomplete — 3 attention flags missing from the table.**

### §2.5.2, §2.5.3, §2.5.4 — confirmed correct

- All `review_hooks` fields confirmed: `next_step` ("writeback" / "analyst_review" / "validation_remediation"), `requires_human_review`, `writeback_ready`, `analyst_decisions_required`, `coverage_acknowledgement_required`, `hard_abort_required`, `degraded_reasons`.
- `writeback_ready` conditions match code: `outputs_ok AND schema_valid AND all_claims_cited AND passed_minimum_evidence_gate AND passed_severity_gate AND NOT decision_required AND writeback_recommendation == "ready_if_accepted" AND decision_status == "candidate_ready" AND NOT degraded_reasons`.
- `resolve_expansion_suggestion(run_id, run_context, signal_id, decision, rationale)` signature confirmed; `persist=True` is an optional internal parameter not needed at the caller level.
- Scope version increment confirmed: `new_version = active_version + 1` for "accepted", `new_version = active_version` for "deferred"/"rejected".
- `scope_management.scope_revisions[]`, Version 0 = initial open boundary, Step 4 boundary filter on re-run — all confirmed.

### §2.5.1 — 3 attention flags missing from table

The code has 10 `_apply_*_attention_flags` methods. The §2.5.1 table covers 7. Three are missing:

| Missing flag | Method | What it triggers on |
|---|---|---|
| **Near-match documentary pattern** | `_apply_near_match_pattern_attention_flags` | `tskr_patterns.patterns[].near_match_pattern == True` — historical documents exist below semantic similarity threshold but above the near-match window (0.10 default); warns against prematurely designating the event as novel |
| **Signal episode index stale or missing** | `_apply_signal_episode_index_attention_flags` | `index_status == "stale"` or `"no_episodes_indexed"` on `historical_signal_episodes`; link confidence may be capped or no episode matches available |
| **Cross-pattern linkage outcome** | `_apply_cross_pattern_attention_flags` | Low-confidence or missing cross-pattern links for top candidates; `linkage_outcome == "no_data"` or `"no_match"` |

Note: the near-match pattern flag is independent of the PatternSearcher (§2.4 Question 5) — it comes from TSKR pattern records in `tskr_patterns`, where `near_match_pattern=True` means the TSKR scorer found a historically similar document pattern below the primary semantic similarity threshold. The signal episode and cross-pattern flags are the new ones consistent with §2.4 Questions 5 and 6.

**Recommendation:** Add the three missing rows to the §2.5.1 table.

---

## §2.6 — Code verification notes

**Verdict: §2.6.1, §2.6.3, §2.6.4 correct. §2.6.2 missing the semantic scoring dimension.**

### What checks out

- **§2.6 intro**: 3 tiers, plant always runs in-memory, fleet/industry require `SimilarEventAdapter`, `status="partial"` without adapter — all confirmed.
- **§2.6.1**: `step2d_query_top_n_candidates=3` default, 5 query terms, stored in `similar_event_list.query_terms` ✓
- **§2.6.2**: Base weights (0.40/0.25/0.15/0.10/0.10), FM match uses `cand_list[:5]`, TIER_MULTIPLIER=1.0, top_n=5 (`step2d_plant_top_n`), `source_db="plant_kg"` ✓
- **§2.6.3**: `LLMOEAdapter` at `adapters/llm_oe_adapter.py`, confidence multipliers (fleet 0.80 / industry 0.60) confirmed in `TIER_CONFIDENCE_MULTIPLIERS`, `source_db` defaults (`"fleet_oe"` / `"inpo_epri_nrc"`) confirmed in `llm_oe_adapter.py:194`, `max_results=5`, `timeout_seconds=10.0`, degraded handling (exception catch + `adapter.degraded` flag check) ✓
- **§2.6.4**: Output schema (`status`, `query_terms`, `summary` fields, `events[]` fields, `provenance`) all confirmed ✓

### §2.6.2 — Missing: semantic scoring dimension (SUBSTANTIVE)

When `enable_semantic_recurrence=True` and `DocExtractionStore` is injected, `_build_doc_id_semantic_scores()` queries the store with FM name + expected symptoms for top-N candidates, returning a `doc_id → max_similarity` map. This activates a 6th dimension in `_query_plant_past_events()`:

| Condition | Component | FM | Event type | Actuation | Window | Semantic |
|---|---|---|---|---|---|---|
| Semantic disabled (default) | 0.40 | 0.25 | 0.15 | 0.10 | 0.10 | — |
| Semantic enabled | 0.36 | 0.225 | 0.135 | 0.09 | 0.09 | 0.10 |

The 5 base weights are renormalized × 0.90 to preserve a unit total. The semantic score is continuous `[0, 0.10]` and applies only to CMMS-sourced events (`CMMS::CR::*` or `CMMS::WO::*`); KG-native events receive 0.0.

Additional output fields when semantic enabled: `semantic_similarity_score` and `source_doc_id` on each plant event; `semantic_scoring_applied` and `semantic_doc_count` in `provenance`.

**Fix applied:** §2.6.2 scoring table expanded to show both base and semantic-enabled weight profiles.

---

## §2.7 — Code verification notes

**Verdict: §2.7.1 and §2.7.4 correct. §2.7.2 missing one manifest section. §2.7.3 two artifacts missing, one incorrectly listed as standalone.**

### §2.7.1 — `rca_card` structure: correct

Card tree and field names confirmed via manifest build code (echoes of card fields at lines 2863–2890). `decision_status` values (`"candidate_ready"` / `"review_required"` / `"insufficient_evidence"`), `fallback_used`, `writeback_recommendation`, and the three safety injections (barrier, CCF, human performance) confirmed.

### §2.7.2 — `run_manifest` sections: one section missing

All listed sections confirmed in `_stage_g_finalize_manifest()` return dict. One top-level key missing from the table:

| Missing section | Key | What it contains |
|---|---|---|
| `epistemics_summary` | `run_manifest["epistemics_summary"]` | Per-candidate epistemics digest summary: epistemic class distribution, policy version, calibration profile (Phase D; present when epistemics classifier injected) |

Also confirmed: `analyst_attention_flags[]` IS a top-level manifest key (not just in the card) — it echoes the card flags and appends a sensitivity flag when `any_ranking_change_possible=True`.

**Fix applied:** Added `epistemics_summary` row to §2.7.2 table.

### §2.7.3 — Intermediate artifacts: two missing, one wrong

| Issue | Artifact | Code reality |
|---|---|---|
| **Missing** | `historical_signal_episodes.json` | `_validate_and_persist(run_id, "historical_signal_episodes", ...)` at line 648; persisted when `enable_signal_episode_search=True` |
| **Missing** | `cross_pattern_evidence.json` | `_validate_and_persist(run_id, "cross_pattern_evidence", ...)` at line 670; persisted when `enable_cross_pattern_linkage=True` and episodes found |
| **Wrong** | `signal_lessons_learned.json` | NOT a standalone file; built inside `_stage_g_finalize_manifest()` at line 2650 and embedded in manifest at `run_manifest["signal_lessons_learned"]`; no separate `_validate_and_persist` call |

**Fix applied:** Added `historical_signal_episodes.json` and `cross_pattern_evidence.json` rows to §2.7.3 table; removed `signal_lessons_learned.json` row from the table and added a clarifying note below the table.

### §2.7.4 — `decision_status`: correct

---

## §3.1 — Code verification notes

**Verdict: Mostly correct. Three substantive gaps fixed; remainder confirmed against `kg_context_builder.py`.**

### What checks out

- Node types (`element_usage`/`element_definition`, `failure_mode`, `safety_function`, `monitored_variable`, `abnormal_event`, `oe_document`, `document`) ✓
- `element_usage` properties (`component_id`, `component_label`, `component_type`, `maximo_floc`, `sap_equipment_id`) ✓
- `kg_context` field list (`components[]`, `failure_modes[]`, `safety_functions[]`, `past_events[]`, `upstream_paths[]`, `documents[]`, `seed_context`, `kg_snapshot_version`) ✓
- `out_of_boundary_anomalies[]` noted as "populated at Step 4" — correct; not in `build()` return, added by orchestrator post-Step-4 ✓
- `kg_snapshot_version` strategy (Neo4j server version + max `last_modified` + fallback) ✓
- Config defaults `max_hops=2`, `max_past_events=10`, `max_documents=20`, `doc_window_days_before=90`, `doc_window_days_after=7` ✓
- KG population role table ✓; "pipeline does not write to KG during a run" ✓

### Issue 1 — `failure_mode` node: two properties missing (SUBSTANTIVE)

`_fetch_failure_modes` Cypher also returns `failure_mechanism` and `expected_anomaly_pattern`. `expected_anomaly_pattern` is used by `_build_doc_id_semantic_scores()` alongside `expected_symptoms` to build the semantic query text for plant-tier semantic scoring. Missing these understates what the KG failure mode encodes and what the pipeline uses.

**Fix applied:** Added `expected_anomaly_pattern` and `failure_mechanism` to the `failure_mode` key properties column.

### Issue 2 — `KGContextBuilderConfig`: `past_event_window_days` missing (SUBSTANTIVE)

Default `past_event_window_days = 3650` (≈ 10 years) controls how far back `_fetch_past_events` searches. Not mentioned in the document. Significant for configuration and for understanding recurrence depth.

`max_oe_documents = 10` also not mentioned — OE documents are fetched separately via `_fetch_oe_documents` and merged into `documents[]`.

**Fix applied:** Added both to the `KGContextBuilderConfig` description sentence.

### Issue 3 — Edge types: three edges absent (MINOR)

| Missing edge | Used in | Direction |
|---|---|---|
| `owns_port_usage` | `_expand_neighborhood` connectivity query | `element_usage → port` |
| `APPLICABLE_TO` | `_fetch_oe_documents` | `oe_document → failure_mode` |
| `DOCUMENTS` | `_fetch_documents` | `document → element_usage` |

`owns_port_usage` is the first half of the physical connectivity pattern (`element_usage -[owns_port_usage]-> port <-[connects_port]- connector -[connects_port]-> port <-[owns_port_usage]- element_usage`). The document only listed `connects_port`, which is the connector-to-port half.

**Fix applied:** Updated edge list to include all three, with brief explanation of the connectivity pattern.

---

## §3.2 — Code verification notes

**Verdict: Priority weights and document type list correct. Three substantive issues fixed: composite score formula wrong, query table incomplete, metadata fields missing.**

### What checks out

- **Document type priority weights**: exact match against `EvidenceRetrieverConfig.doc_type_priority` (CR=1.00, WO=0.95, ECA=0.92, RCA=0.90, ECR=0.85, FMEA=0.80, SOP=0.75, OE=0.70, MANUAL=0.60, BULLETIN=0.55) ✓
- **Hybrid retrieval**: dense + BM25 confirmed in `chroma_store.py`; BM25 degrades to dense-only on disk-loaded collections ✓
- **Deduplication and re-ranking** ✓
- **`candidate_evidence_summary`** ✓
- **"Pipeline does not write to Chroma during a run"** ✓

### Issue 1 — Composite score formula is wrong (SUBSTANTIVE)

Document stated: `dense_score × doc_type_priority_weight × query_weight`

Actual pipeline in `_assess_hit_against_candidate()`:
1. `semantic_relevance` = Chroma `_vector_score` → encoder cosine fallback → lexical overlap
2. `support_score`, `contradiction_score`, `context_score` from: semantic_relevance + cue matching (`support_cues`, `contradiction_cues`, `contextual_cues`) + causal attribution phrases ("caused by", "root cause is") + query-type priors
3. All sub-scores × `authority_weight` (from `authority_level`) × `extraction_quality` × `epistemic_weight` (finding_status/doc_type: confirmed ECA → up to 1.25×, preliminary CR → 0.80×, OE → 0.70×)
4. `ca_as_found`/`ca_as_left` adjustments: ±0.35 from structured inspection results, independent of text
5. spaCy conjecture discount: up to 35% on support_score when hedging language detected
6. `support_role` assigned by threshold: `contradicting` ≥ 0.35 and dominates; `supporting` ≥ 0.30 and dominates; else `contextual`
7. SOP/FMEA/MANUAL/BULLETIN capped at `contextual` regardless of score
8. Phase C: only `epistemic_class = analyzes_past_degradation` hits contribute to support_score

**Fix applied:** Replaced the formula with a correct multi-stage description.

### Issue 2 — Query types table incomplete (MODERATE)

- Weights for failure_mode, component, out_of_boundary listed as "variable" — they are fixed: 0.95, 0.85, 0.75
- "Operational context / OE" is two distinct query types: `oe` (weight 0.80, OE docs only, scoped to KG-retrieved OE doc IDs) and `operational_context` (weight 0.80, checks operating mode/parameters from `operational_context` input)
- `fallback` query type (weight 0.50, fires when no candidates exist) not mentioned

**Fix applied:** Expanded table to show all 8 query types with fixed weights and doc types targeted.

### Issue 3 — Metadata fields: five fields missing (MODERATE)

| Missing field | Used for |
|---|---|
| `eca_confidence` | Scales ECA epistemic_weight bonus (confirmed ECA → 1.0 + 0.25 × eca_confidence) |
| `eca_causal_factors_text` | Structural alignment score for contradiction detection |
| `failure_mode_refs_text` | Structural alignment score for contradiction detection |
| `extraction_quality` | Global multiplier on all sub-scores |
| `epistemic_class` | Phase C gate: only `analyzes_past_degradation` hits contribute to support_score |

**Fix applied:** Added all five fields to the metadata table.

### Issue 4 — Field name: `quality_warning` → `retrieval_quality_warning` (MINOR)

Code uses `provenance.retrieval_quality_warning`, not `provenance.quality_warning`.

**Fix applied:** Corrected field name in description.

### Issue 5 — `EvidenceRetrieverConfig`: `top_k_per_query` not mentioned (MINOR)

Default `top_k_per_query = 5` is the per-query cap applied before deduplication; `top_k_total = 10` is the global cap on the final merged result. Also `contradiction_cues`, `support_cues`, `contextual_cues` are configurable lists not mentioned.

**Fix applied:** Added `top_k_per_query` and cue lists to the pipeline engineers config description.

---

## §3.3 — Code verification notes

**Verdict: Three fixes applied — intro understated the store count, evidence scoring row was inaccurate about phase, DocExtractionStore entirely absent.**

### Issue 1 — "Two stores" intro is wrong (SUBSTANTIVE)

§3.3 opened with: *"The two stores serve complementary roles..."*

**Code reality:** There are three distinct persistence stores:
- **Neo4j KG** — structured equipment model, history, failure modes
- **Main Chroma evidence store** — unstructured document chunks; queried by `EvidenceRetriever`
- **DocExtractionStore** — a second, separate Chroma store holding `HistoricalDocExtraction` objects (extracted causal chains from historical documents); queried by `CrossPatternLinker`

The `IncidentIndex` used by `PatternSearcher` could be considered a fourth store, but it is addressed in §2.4 and not duplicated here.

**Fix applied:** Updated intro paragraph to name all three stores explicitly before describing the KG ↔ main Chroma complementarity.

### Issue 2 — Row 4 evidence scoring phase is inaccurate (SUBSTANTIVE)

Row 4 stated: *"What did prior CRs and RCA reports say about this FM? → Chroma (CR, WO, RCA, ECA chunks) → evidence_bundle — evidence sub-score"*

**Code reality:** Evidence scoring happens in two distinct phases:
- **Step 4 (candidate generation):** `_evidence_score_for_fm()` in `causality_engine_v32.py` (lines ~2748–2800) uses **`kg_context.documents[]`** — KG document references with `doc_type` and `recency_days` metadata. It does NOT query Chroma at this stage. Base score 0.30; FMEA +0.12; CR/WO +0.15×recency; ECA/RCA +0.22×recency; SOP/MANUAL/SPEC +0.08; OE +0.10×recency.
- **Step 5 (refinement):** `refine_with_evidence()` uses the Chroma-retrieved **`evidence_bundle`** to update the evidence sub-score via `_apply_allen_temporal_blend()` and scoring pipeline.

**Fix applied:** Split into two rows — one for Step 4 (KG document metadata) and one for Step 5 (main Chroma store).

### Issue 3 — DocExtractionStore row missing (SUBSTANTIVE)

The table had no row for the third persistence store.

**Fix applied:** Added row: "What causal chains have been extracted from historical documents matching this event pattern? → DocExtractionStore (second Chroma store, `HistoricalDocExtraction` objects) → `cross_pattern_evidence` — documentary linkage via `CrossPatternLinker`; feeds `link_confidence` and `support_posture`."

---

## §5.1 — Code verification notes

**Verdict: 2 substantive fixes. Overall structure (5 actions, PM modes, validation logic) confirmed correct.**

### Issue 1 — `asset_id` mismatch described as "input guard warning" — wrong mechanism (SUBSTANTIVE)

Document said asset_id cross-check produces an input guard warning. Code reality: the check is in `_semantic_checks_bundle()` in `schema_validator.py` (line 553), called via `validate_run_bundle()` → `_validate_bundle()`. It produces a validation ERROR with severity "error", not an input guard flag. Under default `stop_on_validation_error=True`, this aborts the run immediately.

**Fix applied:** Step 3 and "What can go wrong" item 1 updated to say "validation error in `input_validation`" not "input guard warning."

### Issue 2 — Scope seeding incomplete; "CMMS records already available" is wrong in normal flow (MODERATE)

Document said scope seeding sources include "any CMMS records already available." At Step 0, CMMS is not yet fetched (happens in Step 1). `cmms_context` is None when `_build_initial_scope_revision_record()` is called. Also `configuration_change_records` was not mentioned as a scope seed source (it feeds `change_control_systems`).

**Fix applied:** Step 4 description lists the three correct seeding sources with field names; notes CMMS not yet available at Step 0.

### What checks out

- 5 actions in sequence, PM modes (auto/off/force), PM lookback 730 days, input guard enforcement, abort before KG build, scope version 0, optional_artifact_failures — all confirmed.

---

## §5.2 — Code verification notes

**Verdict: 3 fixes. Core KG build, CMMS injection, governance logic confirmed correct.**

### Issue 1 — KG governance abort requires two config flags (MODERATE)

Document said "By default (`hard_abort_on_kg_red_state: True`), a red status aborts." Code: `_should_hard_abort_for_kg_governance()` requires BOTH `strict_red_state_governance=True` AND `hard_abort_on_kg_red_state=True`. Missing the second flag from all descriptions.

**Fix applied:** Updated Step 2 text and parameters table to show both flags.

### Issue 2 — Temporal enrichment description misleading (MODERATE)

Document said enrichment adds "days before the event." Actually `days_before_current_event` comes from the KG query; the enrichment step adds `in_precursor_window` and `window_tier`.

**Fix applied:** Step 4 rewritten with correct field names, tier thresholds (180/360/beyond days), and mention of `per_component_past_events` index.

### Issue 3 — `precursor_window_days` and `per_component_past_event_top_n` missing from params (MINOR)

**Fix applied:** Both added to parameters table.

---

## §5.3 — Code verification notes

**Verdict: 3 fixes. Purpose and historian adapter logic confirmed correct.**

### Issue 1 — "No analyst-configurable parameters" is wrong (MODERATE)

`signal_evidence_historian_mode` and `signal_evidence_historian_infile_path` are exposed config parameters. Also internal builder defaults (`fetch_lookback_hours=72`, `fetch_lookahead_hours=4`, `max_paths=20`, `max_chains=10`) exist but are not exposed.

**Fix applied:** Parameters table added with two configurable params; note on non-exposed defaults.

### Issue 2 — Output structure incomplete (MODERATE)

Document described "per-signal component assignment, anomaly window, propagation chain position." Actual output is: `augmented_anomaly_set[]` (per-signal), `propagation_chains[]` (scored DAG paths), `per_candidate_chain_score{}` (per-FM), `dag_topology_summary`. "Propagation chain position" per signal is not a field; it's `per_candidate_chain_score`.

**Fix applied:** Output paragraph replaced with accurate structure description.

### Issue 3 — Fallback behavior missing from "What can go wrong" (MODERATE)

If build fails, pipeline retries with NullHistorianAdapter, then emits empty artifact if retry also fails. `signal_evidence.runtime.fallback_used=True` marks this.

**Fix applied:** Added fallback behavior item.

---

## §5.4 — Code verification notes

**Verdict: 4 fixes across Step 2b and 2c.**

### Issue 1 — Step 2b confidence formula uses `max(anomaly_score, telemetry_support)` not just `anomaly_score` (SUBSTANTIVE)

The `telemetry_support` floor (`_normalized_weighted_sum` with `max(...)`) was absent. Also missing: the temporal contradiction penalty (`-0.20` when Allen FOLLOWS or latency violation type `"too_fast"`/`"too_slow"`).

**Fix applied:** Pseudocode updated with correct `max(anomaly_score, telemetry_support)` term and temporal contradiction penalty.

### Issue 2 — `novel_pattern` uses `effective_recurrence_count` not `recurrence_count` (MODERATE)

The effective count includes semantic recurrence contributions from DocExtractionStore when `enable_semantic_recurrence=True`.

**Fix applied:** Pseudocode updated.

### Issue 3 — Output fields incomplete (MINOR)

`temporal_contradiction`, `effective_recurrence_count`, `signal_support_score`, `recurrence_support_score` missing from output table.

**Fix applied:** Added to per-FM output field list.

### Issue 4 — Allen map persistence understated (MINOR)

Document said "not a standalone top-level return artifact." Correct, but the map IS embedded in `run_manifest.allen_relation_map` at Step 6.

**Fix applied:** Output description updated.

---

## §5.5 — Code verification notes

**Verdict: 2 fixes. Step 3 (documentary pattern) confirmed correct. Step 3.5 output structure and classification logic were wrong.**

### Issue 1 — Step 3.5 output is two lists, not a dict by fm_id (SUBSTANTIVE)

Document showed `signal_lessons_learned[pattern.fm_id] = {classification, ...}`. Actual structure: two lists `matched_patterns[]` and `novel_patterns[]`, plus `summary` block. Separation into matched/novel is based on the pre-existing `novel_pattern` flag from TSKR, not recomputed here.

**Fix applied:** Pseudocode replaced with correct structure showing two lists and `novel_pattern` flag read from TSKR.

### Issue 2 — Enrichment fields missing (MODERATE)

`causal_explanation` and `resolution_summary` text fields (built from `recurrence_count` + trend) not mentioned.

**Fix applied:** Output description updated.

---

## §5.6 — Code verification notes

**Verdict: OK. All dimension weights, thresholds, gate names, top_k_candidates defaults confirmed.**

- Weights 0.30/0.20/0.20/0.20/0.10 confirmed (causality_engine_v32.py line 107–113) ✓
- `minimum_composite_threshold=0.30`, `minimum_pre_evidence_threshold=0.10`, `minimum_evidence_threshold=0.35` confirmed ✓
- Gate names "physically_impossible", "timeline_inconsistent", "barrier_held" confirmed ✓
- `OrchestratorConfig.top_k_candidates=5` confirmed; causality engine internal default 10 ✓
- `event_analogs` pool exists and is independent of FM pool ✓

---

## §5.8 — Code verification notes

**Verdict: Step 5a description wrong; Step 5c formula confirmed correct; Step 5d params confirmed.**

### Issue 1 — Step 5a "two queries" is wrong (SUBSTANTIVE)

Document said retriever issues "two queries: support and contradiction" with `passage_score = dense_score × doc_type × query_weight`. Actual: 8 query types (candidate/candidate_contradiction/failure_mode/component/out_of_boundary/oe/operational_context/fallback); multi-stage scoring pipeline already corrected in §3.2.

**Fix applied:** Step 5a rewritten to say "up to 8 query types" and cross-reference §3.2 for scoring details.

### What checks out

- Step 5c formula (0.30×prior + 0.55×support×authority_weight + 0.15×contextual - 0.45×contradiction) confirmed (lines 1020-1023) ✓
- `enable_auto_reentry=True`, `auto_reentry_max_attempts=1` confirmed ✓
- Step 5c pre-refine snapshot, hard gate re-check pass, near-tie flagging all confirmed ✓

---

## §5.12 — Code verification notes

**Verdict: 2 substantive fixes in Step 6e and 6h. Steps 6a-6d, 6f-6g, 6i confirmed correct.**

### Issue 1 — Step 6e: 4 attention flags missing; total is 11 not 7 (SUBSTANTIVE)

Document said "seven flag checks." Actual: 11 flag functions called at lines 691-707.

Missing flags:
- `_apply_near_match_pattern_attention_flags` — fires when `near_match_pattern=True` in any TSKR pattern
- `_apply_fm_resolution_ambiguity_flags` — fires when `fm_resolution_ambiguous=True` (similarity in [0.80, 0.88) range)
- `_apply_signal_episode_index_attention_flags` — fires when IncidentIndex is absent or stale
- `_apply_cross_pattern_attention_flags` — fires based on cross-pattern documentary linkage posture

**Fix applied:** Updated to "eleven flag checks" and added all 4 missing flags to the pseudocode.

### Issue 2 — Step 6h: expansion trigger is `causal_candidate=True`, not just PRECEDES (SUBSTANTIVE)

Document said `allen_relation == "PRECEDES"`. Actual: `causal_candidate=True` (covers PRECEDES, OVERLAPS, CONTAINS). Signal evidence propagation chains are a second source not shown.

**Fix applied:** Pseudocode replaced with two-source structure showing `causal_candidate=True` condition and signal_evidence chain source.

### What checks out

- Step 6a–6d (Ishikawa, barrier, similar events, synthesis) confirmed correct ✓
- Step 6d: `barrier_analysis` injected after `synthesize()` confirmed ✓
- Step 6d: `similar_event_list` not in LLM prompt confirmed ✓
- Step 6d: hallucination guard (discard LLM output if candidate_id not in active set) confirmed ✓
- Step 6f: output validation same mechanism as input validation ✓
- Step 6g: `enable_chroma_archive_stage` default False, `hard_fail_on_chroma_archive_error=True` confirmed ✓
- Step 6i: return dict matches code (lines 830-847) ✓

---

## General notes

- The document correctly describes both `v31` and `v32` causality engines; `v32` is the current production engine. Confirm in later sections whether any v31-only behavior is still cited as current.
- `CrossPatternLinker` (`cross_pattern/linker.py`) and `cross_pattern_evidence` artifact are present in attention flags code but not described in §2.2 — flagged above. Expect them to appear in §5 or later sections.
- PM subsystem has 6 sub-modules (`aggregator`, `currency_checker`, `effectiveness_analyzer`, `execution_verifier`, `schedule_loader`, `scope_analyzer`); document treats PM as a single artifact input — acceptable at the target audience level.
- The §1.6 corollary about ECA/RCA failure modes not back-propagated to the KG is a data stewardship point that also belongs in §3.1 (KG maintenance responsibilities).

---

## §6 — Code verification notes

**Verdict: 3 fixes applied. §7 and §8 confirmed correct.**

### Issues fixed

**Issue 1 — `complete` status missing from data_coverage_summary table**
- Document showed only 3 statuses (`not_assessed`, `missing`, `partial`); code defines a 4th: `complete` (orchestrator `_build_data_coverage_summary` lines 4382, 4396, 4399, 4444–4449, etc.).
- Fix: added `complete` row — "All required checks passed with no degradation."

**Issue 2 — `missing` status description inaccurate**
- Document said "Passed but absent or could not be retrieved."
- Code reality: for always-assessed items (KG, Chroma, telemetry) `missing` fires when result count == 0, regardless of how input was structured. For `plc_status`, `missing` fires when PLC was NOT provided at all but SOE was present — not "passed but empty." The "passed but" qualifier was wrong for both cases.
- Fix: changed to "No usable data retrieved, or an expected source was absent (e.g. PLC absent when SOE is present)."

**Issue 3 — Stale §5.18 cross-reference**
- Line 2240 said "run_status (see §5.18)" — §5.18 does not exist as a document section.
- Workflow dispatch and run_status are covered in §5.12 (`_build_workflow_dispatch`, artifact_store.save at line 313-314).
- Fix: changed cross-reference to §5.12; also clarified that `similar_event_list`, `signal_lessons_learned`, and `workflow_dispatch` appear as **summaries** under `run_manifest.artifacts / review_hooks`, not as top-level return dict keys (confirmed by return dict at lines 830-847).

### What checks out
- §6 artifact table (10 rows): all artifacts are real top-level return keys. Missing from table (`run_context`, `kg_context`, `signal_evidence`) are intentional omissions for reader-oriented focus, not errors. `kg_governance` was never a top-level return key.
- §7 G/I/L labels: G = "Human execution" (execution error shorthand ✓), I = "Configuration and change control" (config baseline wrong shorthand ✓), L = "Systemic/organizational weakness" ✓ — all confirmed against §1.5.
- §7 TC-6 cross-reference: TC-6 exists in Appendix D (human performance, training) ✓.
- §8 DummyLLMClient: `generate_json` raises RuntimeError on purpose (llm_clients.py line 32); this forces fallback synthesis → stable deterministic text ✓. OllamaLLMClient class confirmed at line 37 ✓.
- §8 reproducibility claim: all rule-based and retrieval stages are deterministic; only LLM synthesis with a live client (OllamaLLMClient) introduces run-to-run variance ✓.

---

## Appendix A — Code verification notes

**Verdict: OK. No fixes needed. [WIP] label appropriate.**

### What checks out

- **`CausalityEngineConfigV32`** — correct class name; weights {structural:0.30, temporal:0.20, telemetry:0.20, evidence:0.20, governance:0.10} confirmed hand-tuned in config ✓
- **"CCF/OP deltas"** — both are real capped additive terms on the structural sub-score: `CCF_DELTA_CAP = 0.10` (line 364, Category C only) and `OP_DELTA_CAP = 0.12` (line 411, operating-point score). Document's "weights and caps (e.g. CCF/OP deltas)" is an accurate shorthand ✓
- **Null TSKR path** — "If `tskr_temporal_scorer` is `None`, patterns are synthetic / empty — downstream still runs" confirmed at orchestrator lines 877–888: returns `{patterns: [], summary: {mode: "absent", has_temporal_support: False}, provenance: {generated_by: "orchestrator_null_temporal_stage"}}`. No abort. ✓
- **`step2d_similar_event_plan_april_25.md`** — file exists in `diagrams/april_25/` ✓
- **TC-7 cross-reference** in A.2 — TC-7 exists in Appendix D ("Scope expansion, TSKR, degraded SOE/alarm") ✓
- **Debugging note four-way distinction** — TSKR (`tskr_patterns`), Allen (`allen_relation_map`), causality engine (`causality_candidates`), similar events (`similar_event_list`) are confirmed as separate artifacts with separate code paths ✓

### Notes

- A.2 is clearly framed as future directions ("not promises") — no code claims to verify.
- Minor imprecision: "synthetic" for the null TSKR output is defensible given `generated_by: "orchestrator_null_temporal_stage"` in the provenance field; not a factual error.
- A.1 FM-resolution bullet uses "recurrence_count" as a conceptual term (data quality discussion), not as a field reference — consistent with context; `effective_recurrence_count` precision not required here.

---

## Appendix B — Code verification notes

**Verdict: OK. No fixes needed.**

### What checks out

- **`RuleValidatedRCASynthesizerV31`** — correct class name; lives in `synthesis/rca_synthesizer_v31.py` line 36 ✓
- **`llm_client.generate_json(...)`** — correct method name; called at line 100 ✓
- **`_fallback_card`** — method exists at line 2586; triggered at line 142 ✓
- **`allow_fallback_template_fill`** — config field at line 32, default `True` ✓
- **Hallucination guard logic** — when `llm_primary_id not in _all_input_candidate_ids`, `card = None` (line 132) and a validation error is appended; line 140 then triggers `_fallback_card`. Three-trigger description (generation fails / validation fails / invented candidate_id) is accurate ✓
- **`_all_input_candidate_ids`** — built from the *full* candidate list (not just prompt-truncated set, per comment at lines 108-111), so a legitimately low-ranked candidate doesn't falsely trigger the hallucination guard — document does not mention this subtlety but doesn't claim otherwise
- **`ccf_summary` post-processing** — injected at lines 175-183 after both LLM and fallback paths ✓
- **`human_performance_assessment` post-processing** — injected at lines 192-194 when absent, applies to both paths; "may inject" qualifier in document is correct (conditioned on absence) ✓
- **`LLMOEAdapter`** — class exists in `adapters/llm_oe_adapter.py` line 36 ✓
- **`RuleBasedCausalityEngineV32`** — correct class name; `causality_engine_v32.py` line 126 ✓
- **B.4 "Possible futures"** — design-space discussion, no code claims to verify

---

## Appendix D — Code verification notes

**Verdict: OK. No fixes needed.**

### What checks out

All seven `tests/test_case_N/` directories exist under `src/dackar/RCA/tests/`. Emphasis labels verified against fixture files and `description.md`:

| TC | Document emphasis | Evidence |
|----|------------------|----------|
| TC-1 | Minimal plumbing, `build_dev_orchestrator` | `build_dev_orchestrator` used in `test_rca_orchestrator_notebook.ipynb`; description.md says "minimum viable input smoke test" ✓ |
| TC-2 | SOE, alarms, PLC pairing, vacuum loss | Fixtures include `soe_log.json`, `alarm_log.json`, `protection_logic_context.json`; scenario is condenser vacuum loss ✓ |
| TC-3 | Condenser / U2 scenario, configuration + environmental | Fixtures include `configuration_change_records.json` + `environmental_monitoring.json`; description.md confirms Unit 2 condenser ✓ |
| TC-4 | Reactor trip, timeline gate, RPS / PLC | Asset `U1-RPS-CORE`; fixtures include `protection_logic_context.json`, `soe_log.json`; description title confirms reactor trip / timeline gate ✓ |
| TC-5 | ECCS CCF, vendor supply chain | Asset `U3-HPCI-SYSTEM`; `vendor_supply_chain_records.json` in fixtures ✓ |
| TC-6 | Human performance, training | description.md: "Human Performance / Procedure Gap During Startup" ✓ |
| TC-7 | Scope expansion, TSKR, degraded SOE/alarm (absent) | description.md: "Scope Expansion in a Degraded-Data Environment" ✓ |

---

## Appendix C — Code verification notes

**Verdict: 2 fixes applied.**

### Issues fixed

**Issue 1 — Semantic scoring dimension missing from C.3 table (substantive)**
- Document showed a 5-dimension table with weights summing to 1.0. Code (`_query_plant_past_events`, lines 3168–3181) has two modes:
  - **Base mode** (no semantic scores): COMPONENT=0.40, FM=0.25, EVENT_TYPE=0.15, ACTUATION=0.10, WIN_BOOST=0.10 — matches document.
  - **Semantic mode** (when `_build_doc_id_semantic_scores` returns results, Phase 3b): all five base weights renormalized ×0.90, plus SEMANTIC=0.10 (continuous, capped at 0.10×sim). Applies only to CMMS-sourced events (`CMMS::CR::` or `CMMS::WO::` prefixed event_id); KG-native events receive 0.0.
- The "not a semantic text search" sentence at end of C.3 was accurate only for KG-native events. For CMMS-sourced events with Phase 3b active, embedding similarity IS a real scoring input.
- Fix: replaced 5-column table with 6-column dual-mode table; updated trailing sentence to distinguish CMMS vs. KG-native event handling; removed misleading "not a semantic text search" framing.

**Issue 2 — Multiply-then-cap order wrong**
- Document said: "The raw sum is **capped** at 1.0, **then** multiplied by the plant tier factor 1.0."
- Code (line 3215): `confidence_weight = round(min(1.0, raw_score * TIER_MULTIPLIER), 6)` — multiplication first, then cap.
- For plant tier (TIER_MULTIPLIER=1.0) this is numerically equivalent, but the description was backwards and would mislead for any non-unity multiplier. Fix: changed to "multiplied by … and then capped at 1.0."

### What checks out

- **C.1 timing** — "After `barrier_analysis` and before `rca_synthesizer.synthesize`" confirmed: orchestrator lines 619–626 (barrier), 630 (similar event list), 677 (synthesize) ✓
- **C.1 structure** — `{status, query_terms, summary: {plant_count, fleet_count, industry_count, total_count, degraded_tiers, any_plant_match}, events[], provenance}` confirmed at lines 3370–3399 ✓
- **C.1 not a top-level return key** — confirmed: not in run() return dict at lines 830–847 ✓
- **C.2 default values** — `step2d_query_top_n_candidates=3` (line 3269), `step2d_plant_top_n=5` (line 3268); both in `config.extra` ✓
- **C.2 query_terms fields** — `component_id`, `failure_mode_id` (or `canonical_tuple.failure_mode` fallback), `event_type`, `actuation_type`, `asset_id` confirmed at lines 3276–3299 ✓
- **C.3 tier multipliers** — `TIER_CONFIDENCE_MULTIPLIERS = {"plant": 1.00, "fleet": 0.80, "industry": 0.60}` confirmed in `adapters/similar_event_adapter.py` lines 22–26 ✓
- **C.4 adapter contract** — `SimilarEventAdapter.query(level="fleet"|"industry", ...)` confirmed at lines 49–60 of similar_event_adapter.py ✓
- **C.4 errors → degraded_tiers** — orchestrator wraps adapter.query() in try/except → `degraded_tiers.append(level)` (lines 3353–3354); also catches adapter.degraded flag (line 3338) ✓
- **C.5 status logic** — `complete` only when adapter is not None AND degraded_tiers is empty; confirmed at lines 3362–3368 ✓
- **C.6 cross-references** — `step2d_similar_event_plan_april_25.md` exists ✓; TC-7 and TC-3 cross-references valid per Appendix D review ✓
