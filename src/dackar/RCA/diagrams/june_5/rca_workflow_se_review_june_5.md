# RCA Workflow — Systems-Engineering Completeness & Soundness Review

**Date:** 2026-06-06
**Reviewer role:** System engineer (RCA practitioner perspective)
**Scope:** `src/dackar/RCA/` — workflow architecture, logical soundness of the RCA process, and component functional relations
**Reference material (this folder):**
- IAEA-TECDOC-1112 — *Root Cause Analysis for Fire Events at NPPs* (ASSET methodology)
- IAEA-TECDOC-1756 — *Root Cause Analysis Following an Event at a Nuclear Installation: Reference Manual*

**Internal references consulted:** `rca_metamodel.md`, `rca_workflow_reference_guide_*.md`, `rca_pipeline_review_may_23.md`, `rca_reasoning_orchestrator.py`, `causality_engine_v32.py`, `rca_synthesizer_v31.py`.

> This review takes a top-to-bottom approach: it first checks the *initial assumptions* of the workflow against the authoritative RCA methodology, then the high-level workflow architecture, then the functional relations between components, then specific code-level issues. It deliberately re-derives the logic from the IAEA sources rather than re-stating the prior May-23 review, though it confirms/updates several of its findings.

---

## 0. Executive verdict

The pipeline is an unusually mature, well-documented, deterministic RCA decision-support system. The **core workflow assumptions are sound** and map cleanly onto recognised methodology (IAEA ASSET three-question structure; AP-913 proximate/contributing/root depth). The 12-category coverage model, hard-gate-then-score discipline, traceable evidence posture, near-tie/sensitivity surfacing, and human-in-the-loop checkpoints are all the right primitives for nuclear RCA.

However, several **logical-soundness gaps** remain that a system engineer must understand before trusting the output:

1. **The "physical plausibility" gate does not check physical plausibility** — it is a minimum-structural-score filter. (High)
2. **The human-performance assessment mislabels categories H and K** (design deficiency and vendor/supply-chain) as human-performance findings with human-performance regulatory references. (High — defensibility)
3. **The ASSET defining root-cause question — "why was it not prevented/detected?" — is not a first-class output.** Detection/surveillance failure is only implicitly reachable through Categories J and L, which are the weakest-populated in the pipeline. (High — methodology)
4. **Gates run after composite scoring, not before** — the metamodel's "elimination first, facts not evidence" sequencing is not honoured in execution order (the outcome is mostly equivalent, but the ordering matters for auditability and for gate inputs). (Medium)
5. The known coverage gaps (F, G partial, K, L) and the unvalidated LLM synthesis path persist as the dominant *completeness* risks.

**Fixed since May 23 (verified):** the Allen temporal blend is no longer one-directional — it now both raises and lowers the temporal score (`causality_engine_v32._apply_allen_temporal_blend`, α = 0.25). Prior finding 1.1 is resolved.

---

## 1. Are the initial workflow assumptions correct?

I checked the metamodel's foundational assumptions against IAEA ASSET (TECDOC-1112 §2) and the Reference Manual (TECDOC-1756).

| Assumption in the pipeline | Methodology basis | Verdict |
|---|---|---|
| An event decomposes into occurrences / candidate causes along a temporal+logical chain | ASSET "logic tree of occurrences"; event = chain of occurrences, each an element failing to perform as expected | **Correct** |
| Causes have depth: proximate → contributing → root | ASSET 3 questions ("what happened / why / why not prevented"); AP-913 | **Correct in structure** (but see §3.1 on the third question) |
| Investigation must be coverage-driven, not fixation-driven | ASSET stresses analysing *all* occurrences; TECDOC-1756 §2 prerequisites | **Correct and well-implemented** (12-category coverage enforcement) |
| The workflow reasons over *pre-processed* anomalies; it does not do signal processing | Sound separation of concerns | **Correct** (but propagates upstream anomaly-detection quality risk — explicitly flagged in metamodel, good) |
| Temporal precedence discriminates cause vs consequence (Allen algebra) | ASSET chronological sequence + logic tree; TECDOC-1756 Event & Causal Factor Charting (§3.5) | **Correct** |
| Human-in-the-loop decision points are functional elements, not boundaries | TECDOC-1756 §2.3 (team, review/approve); analyst remains accountable | **Correct** |

**Conclusion:** the founding assumptions are valid and faithfully derived from the IAEA methodology. The problems are in *realisation*, not in the conceptual model — with one important methodology exception in §3.1.

---

## 2. High-level workflow architecture

The orchestrator (`RCAReasoningOrchestrator.run()`) executes the metamodel Steps 0–6 in this realised order:

```
Step 0  Scoping ............... _stage_a_build_run_context (+ versioned scope-revision lifecycle)
Step 1  Data management ....... input validation + guards + _build_data_coverage_summary (8 families)
        KG init/expansion ..... kg_context_builder.build → CMMS augmentation → past-event temporal enrich
Step 2  KG expansion .......... signal_evidence, Allen relation map (2c), similar events (2d)
Step 3/3.5 Pattern recog ...... tskr_patterns (+ signal_lessons_learned)
Step 4  Candidate generation .. causality_engine.generate (4-tuple, A–L coverage)
        Scope boundary filter . _apply_scope_boundary_filter (when scope version > 0)
Step 5  Evidence assessment ... evidence_retriever.retrieve → _apply_supersession
                                 → refine_with_evidence (scoring + 3 gates + sensitivity)
                                 → auto-reentry if needed
        Cross-cutting ......... ishikawa, barrier analysis, cross-pattern linkage
Step 6  Conclusion ............ synthesizer.synthesize → rca_card (+ depth, HoP, monitoring)
        Finalize .............. _stage_g_finalize_manifest, archive, manifest assembly
```

**Architecture-level observations:**

- **AO-1 — Step sequencing is faithful** to the metamodel and to ASSET's investigation → analysis → recommendation flow. The data-to-decision poles are respected: each stage validates and persists an artifact.
- **AO-2 — The orchestrator is a god-object.** `run()` is ~550 lines and the class is ~6,800 lines spanning orchestration, CMMS, scope lifecycle, manifest assembly, attention flags, Chroma archiving, reentry. This is a maintainability/robustness liability (consistent with May-23 §3.1) but not a logic defect.
- **AO-3 — Gate placement vs metamodel.** The metamodel (Step 5 / Phase 1) is explicit: *"Apply binary gates to eliminate physically or logically impossible candidates **before any scoring**. These are facts, not evidence."* In `refine_with_evidence`, candidates are **fully composite-scored first** (loop 1), and the three hard gates are applied **after** (loop 2, lines ~1268–1278). Functionally the gated candidates are still excluded (they get `meets_evidence_threshold=False` → filtered out), so rankings are usually unaffected — but the ordering means (a) the audit trail shows a composite score for a "physically impossible" candidate, which is conceptually contradictory, and (b) the gates cannot be used to *avoid* scoring cost or to *gate scoring inputs*. Recommend reordering to elimination-first to match the stated design and improve defensibility. **Severity: Medium.**

---

## 3. Logical soundness of the RCA process

### 3.1 The "why was it not prevented?" question is not a first-class output — *methodology gap*

This is the most important conceptual finding. In ASSET (TECDOC-1112 §2.3) the **root cause is, by definition, the failure of detection/prevention**: *"The root cause is either the reason for which the latent weakness was not discovered before an in-service failure, i.e. a failure of the surveillance programme OR stems from the inadequate restoration of a previously recognized latent weakness."* The third question — *why was it not prevented?* — is the whole point of RCA and what separates it from failure analysis.

In the pipeline this dimension is **split and diluted**:
- The "detection/surveillance failure" concept lives in **Category J** (Inspection & Testing Program Inadequacy), classified as *contributing*.
- The "systemic/restoration" concept lives in **Category L**, classified as *root*.
- Both J and L are documented as the **weakest-populated, hardest-to-automate** categories (metamodel §F/K/L notes; `_CATEGORY_REQUIRED_STREAMS` requires `documentary`/`oe` which are often absent).

Consequently, a typical run concludes with a strong **proximate** mechanism (Category A failure mode), `root_cause = "unresolved"`, and the *defining* RCA question effectively unanswered. The pipeline correctly flags this via `depth_incomplete_reason` and `unresolved_gaps` (good), but there is **no explicit "detection/prevention failure" assessment** that asks, for the selected primary cause: *which surveillance, test, PM, or feedback barrier should have caught this and why didn't it?*

**Recommendation:** add an explicit *prevention-failure / detection-escape* assessment to the Step 6 card, seeded from PM-compliance, surveillance interval data, and prior-CA effectiveness (the J/S-13/S-11 hooks already named in the metamodel). This converts the pipeline from "what most likely failed" to "what failed *and why our defences missed it*", which is the regulatory deliverable. **Severity: High (methodology completeness).**

### 3.2 The physical-plausibility gate does not test physical plausibility — *soundness defect*

`_apply_physical_plausibility_gate` (causality_engine_v32 ~L2277) fails a candidate **only** when `structural_score < 0.20`, plus an informational PLC-presence note. The metamodel's Gate 1 explicitly requires testing the failure mode against **the operating state at event time** (power level, flow, pressure, temperature, mode) using **FMEA failure-mode condition parameters** and **design-basis envelope limits**.

None of those inputs are consulted by the gate. Implications:
- A candidate that is **physically impossible for the recorded operating state** but has a high structural score (e.g., a high-cycle fatigue or thermal-transient failure mode proposed during cold shutdown with no cyclic load) will **PASS** this gate.
- The gate is effectively a *minimum-structural-score* screen wearing the name "physical plausibility." That mislabel is itself a defensibility hazard — an auditor reading "physical_plausibility: passed" will assume an operating-state check occurred.

**Recommendation:** either (a) implement the operating-state-vs-FMEA-condition check the metamodel specifies, or (b) rename the gate to `minimum_structural_score` and downgrade its claims until the real check exists. Operating-state data already flows in (`operational_context.operating_point`, used by `_operating_point_score`), so a first version is feasible. **Severity: High.**

### 3.3 Timeline-consistency and barrier-logic gates — *sound, with documented degraded modes*

- **Gate 2 (timeline):** correctly hard-fails on `latency_violation_type ∈ {too_fast, too_slow}` or `temporal_contradiction` (set by the Allen `follows` detection). When FMEA latency params are absent it degrades to Allen-only — which matches the metamodel's stated and expected degraded behaviour. **Sound.**
- **Gate 3 (barrier logic):** fails when PLC `sf_state ∈ {failed, degraded}` for an affected safety function, or when a prior `barrier_held` ruleout exists; otherwise passes (degraded) without PLC context. **Sound**, and `degraded_mode` is recorded rather than silently skipped — good.

### 3.4 Allen temporal blend — *now correct (fixed since May 23)*

`_apply_allen_temporal_blend` uses `new_temporal = 0.75·old + 0.25·allen`, which can **raise or lower** the temporal score, and sets `temporal_contradiction = True` when the component has a `follows` node (which then trips Gate 2). The May-23 finding that "Allen only raises temporal, never lowers" is **resolved**. One nuance worth documenting: only **anomaly (affects-class)** nodes raise the causal score; alarm/SOE (monitors-class) nodes contribute only to contradiction detection — a defensible Phase-C design choice that should stay documented in the metamodel.

### 3.5 Primary-cause selection is purely score-rank — *acceptable but worth a guardrail*

The synthesizer's `_select_candidates` picks the top-N by `composite_score` (plus `review_required` rows); the "primary hypothesis" is simply the highest composite. Gate-blocked candidates are excluded upstream (they are moved to `filtered_out_candidates` and never reach the synthesizer), so the conclusion cannot be a physically-impossible candidate — good. However, there is **no guardrail that the primary be the chain-initiating candidate** rather than a high-scoring consequence; the chain-position 4-tuple field exists but is not used to prefer `initiating` over `consequence` at conclusion time. In ASSET the logic-tree root (earliest initiating occurrence) is what matters. Consider weighting or at least flagging when the top-ranked candidate has `chain_position = consequence`. **Severity: Medium.**

---

## 4. Component functional-relations review

### 4.1 Human-performance assessment mis-maps categories H and K — *defensibility bug*

`RuleValidatedRCASynthesizerV31._build_human_performance_assessment` (L2125) uses:

```python
HOP_CATEGORIES = {"G", "H", "I", "J", "K"}
PERFORMANCE_MODE = {"G": execution_error, "H": execution_error,
                    "I": procedure_gap, "J": knowledge_gap, "K": supervisory_gap}
REGULATORY_REF  = {"G"/"H": "AP-913 §4.3 Human Performance", ...,
                   "K": "AP-913 §4.6 Supervisory/Organisational"}
```

But the **engine's own taxonomy** (`causality_engine_v32._CATEGORY_PROFILE_NAMES`, L204) and the metamodel define:

| Cat | Engine/metamodel meaning | Synthesizer HoP label | Mismatch |
|---|---|---|---|
| G | human_performance | execution_error | OK |
| **H** | **design_deficiency** | execution_error / AP-913 §4.3 Human Performance | **Wrong** |
| I | change_control | procedure_gap | partial |
| J | surveillance (inspection/test program) | knowledge_gap | **Wrong** |
| **K** | **vendor_procurement** | supervisory_gap / §4.6 | **Wrong** |

So a **Category H design deficiency** or a **Category K vendor/supply-chain defect** retained in the candidate set is reported in the card's *Human and Organisational Performance Assessment* as a human-execution or supervisory finding with a human-performance AP-913 citation. This is a category error that misattributes a hardware/design/procurement cause to human performance — exactly the kind of misclassification a regulator would challenge. The inline comment ("legacy synthesizer scheme") confirms this is a stale mapping that predates the v32 A–L taxonomy.

**Recommendation:** realign to the v32 taxonomy — human/organisational performance is **G** (and the human-action facets of **I**); design (**H**), inspection-program (**J**), and vendor (**K**) belong in separate card sections (design adequacy, surveillance-program adequacy, supply-chain), each with the correct AP-913 reference. **Severity: High (defensibility/correctness).**

### 4.2 Category-applicability heuristics are coarse (text-keyword based)

`_assess_category_applicability` decides applicability from `doc_type` substring matching (`"fmea"`, `"surveillance"`, `"config"`, etc.) and a few `operational_context` booleans. C and D are hard-coded `unknown`; L is always `applicable`. This is acceptable as a coverage *scaffold*, but it means "applicable vs unknown" carries little evidential weight and the keyword fallback (`_infer_category_from_text`) is brittle (May-23 §1.4 still stands — when the text-inference fallback fires it is not surfaced as an attention flag). **Severity: Medium.**

### 4.3 Coverage scaffolds satisfy the *form* of coverage, not the *substance*

`_build_metamodel_scaffolds` guarantees every category gets a `category_coverage` entry (`candidate_scored` / `not_applicable` / `ruled_out`). This correctly prevents silent gaps. But for F/K/L the practical outcome is `ruled_out: no_supporting_data` on nearly every run because the data model has no representation for them (metamodel is candid about this). Coverage enforcement is therefore **a completeness *checklist*, not completeness *evidence*** — which is fine as long as the card never implies these categories were genuinely investigated. Verify the card wording for ruled-out categories is "not assessed — data absent" rather than "ruled out." **Severity: Medium (completeness/wording).**

### 4.4 Optional-phase failures are partially silent (confirms May-23 §3.2)

Several optional phases swallow exceptions with `LOGGER.warning` and continue **without** recording to a manifest-visible failure list:
- `fm_id_candidate` resolution (run L426)
- CMMS context build (L448)
- signal-episode search (L698)
- (May-23 also listed epistemics digest, supersession, cross-pattern, KG topology, NER disambiguation)
 
Ishikawa is the **good pattern** to copy: on failure it appends a structured record to `optional_artifact_failures` (L658). The others should do the same so a degraded run is visibly distinguishable from a clean run in the manifest. **Severity: Medium-High (a degraded run can look clean — the single most consequential robustness gap).**

### 4.5 Runtime duck-typing of the engine interface (confirms May-23 §3.3)

`run()` still inspects `refine_with_evidence`'s signature at runtime (`inspect.signature`, L583) to decide whether to pass `coverage_summary`, `allen_relation_map`, `protection_logic_context`. The `CausalityEngine` Protocol (L103) does not declare `refine_with_evidence` at all. This is fragile; promote the full interface (including the optional kwargs) to the Protocol. **Severity: Low-Medium.**

### 4.6 Data-quality flag propagation is present but indirect

The metamodel requires data-limited flags from Step 1 to systematically degrade Step 3/4/5 confidence. This *is* implemented (`_coverage_quality_profile` → `_apply_coverage_quality_adjustment`; `quality_multiplier`; `_apply_score_confidence_interval`). Confirmed present and reasonable. One caveat: `coverage_summary` reaches the engine **only via `refine_with_evidence`** (not `generate`), so the *initial* v1 ranking is not quality-adjusted — acceptable by design but worth keeping documented.

---

## 5. Consolidated issue register

| # | Finding | Type | Severity | Status vs May-23 |
|---|---|---|---|---|
| F-1 | "Physical plausibility" gate only checks `structural < 0.20`; ignores operating-state/FMEA conditions | Soundness | **High** | New (deepens "gates are structural proxies") |
| F-2 | HoP assessment mislabels H (design) and K (vendor) as human-performance with wrong AP-913 refs | Correctness/defensibility | **High** | New |
| F-3 | "Why was it not prevented?" (detection/surveillance escape) not a first-class output | Methodology completeness | **High** | New (grounded in ASSET) |
| F-4 | Gates execute after composite scoring, not before (metamodel says elimination-first) | Soundness/audit | Medium | New |
| F-5 | Optional-phase failures partially silent (not all write to a manifest-visible list) | Robustness | Medium-High | Confirms May-23 §3.2 |
| F-6 | Primary-cause selection ignores chain position (consequence can outrank initiating) | Soundness | Medium | New |
| F-7 | Category F/K/L coverage is checklist-only; no data representation | Completeness | Medium | Confirms May-23 §1.2 |
| F-8 | Category-G human-performance scoring still keyword-only; no WO-date proximity | Completeness | Medium | Confirms May-23 §1.3 |
| F-9 | Applicability/category inference is coarse keyword matching; fallback not flagged | Robustness | Medium | Confirms May-23 §1.4 |
| F-10 | Runtime `inspect.signature` duck-typing; Protocol under-specified | Robustness | Low-Med | Confirms May-23 §3.3 |
| F-11 | Orchestrator god-object (~6.8k lines); `run()` ~550 lines | Maintainability | Medium | Confirms May-23 §3.1 |
| F-12 | LLM synthesis path unvalidated end-to-end (`DummyLLMClient` everywhere) | Validation | High | Confirms May-23 §2.3 |
| F-13 | Pattern-recognition conclusions fragmented across ≥5 artifacts | Usability | Medium | Confirms May-23 §2.1 |
| ✓ | Allen blend one-directionality | — | Resolved | **Fixed** (verified §3.4) |

---

## 6. Prioritised recommendations

**Tier 1 — correctness & defensibility (do first):**
1. **F-2:** Re-map the human-performance block to the v32 A–L taxonomy; move H/J/K to their own card sections with correct AP-913 references.
2. **F-1:** Implement (or honestly rename) the physical-plausibility gate so its name matches what it checks.
3. **F-3:** Add an explicit detection/prevention-failure assessment for the selected primary cause (ASSET third question).

**Tier 2 — robustness & auditability:**
4. **F-5:** Route every optional-phase failure to a manifest-visible `pipeline_warnings`/`optional_artifact_failures` entry (copy the Ishikawa pattern).
5. **F-4:** Reorder `refine_with_evidence` to elimination-first (gates → score surviving candidates).
6. **F-6:** Use `chain_position` to prefer/flag initiating over consequence at conclusion time.

**Tier 3 — completeness program (longer horizon):**
7. **F-7/F-8:** Define a data-acquisition plan for F/G(WO-date)/K/L; until then ensure the card says "not assessed — data absent," never "ruled out."
8. **F-12:** Run at least one end-to-end synthesis against a real LLM and capture a golden RCA card for regression.
9. **F-13:** Add a consolidated *Pattern Recognition Summary* block to the RCA card.

---

## 7. Open questions for the team

1. Was the gate-after-scoring order (F-4) a deliberate trade-off, or an artifact? If deliberate, document why in the metamodel.
2. Is the human-performance block (F-2) intended to be a *human-and-organisational* section (then H/K do not belong) or a generic *contributing-cause* section (then it should be renamed and its references generalised)?
3. For F-3: is there appetite to make "detection/prevention failure" a required Step-6 field gated by PM-compliance + surveillance-interval data?
4. What is the validation plan for the LLM synthesis path before any production use (F-12)?
5. v31/v32 engine coexistence — confirm v32 is the sole production path and schedule v31 retirement.

---

## 8. Bottom line

The workflow's **conceptual foundation is correct** and faithfully derived from IAEA RCA methodology; the engineering is impressively thorough (coverage enforcement, traceability, near-tie/sensitivity, scope lifecycle, ~1,000+ tests). The pipeline can genuinely help a system engineer resist fixation and keep an auditable trail.

The gaps that most affect *trust in the conclusion* are: a mis-named/under-powered physical-plausibility gate (F-1), a category mislabel that misattributes design/vendor causes to human performance (F-2), and the absence of an explicit answer to RCA's defining question — *why was it not prevented?* (F-3). Closing these three would move the system from "ranks plausible mechanisms with good hygiene" to "produces a defensible nuclear root-cause conclusion."
