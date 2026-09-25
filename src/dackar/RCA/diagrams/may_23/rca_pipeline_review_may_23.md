# RCA Pipeline Review Notes

**Date:** 2026-05-23  
**Reviewer:** Architecture review session  
**Scope:** `src/dackar/RCA/` — full pipeline; code robustness, reasoning logic, SE usability  
**Source materials:** `rca_metamodel.md`, `rca_workflow_reference_guide_april_25.md`, orchestrator, causality engine v32, synthesizer, all sub-packages, unit tests, show-and-tell notebooks

---

## 0. What the pipeline is

A deterministic, rule-based nuclear RCA pipeline that takes plant event data and produces:

- **`rca_card`** — ranked hypotheses with evidence citations, corrective actions organized at three causal depths (proximate → contributing → root), barrier assessment, and human performance findings
- **`run_manifest`** — audit record: data sources present/missing, sensitivity table, scope state, analyst review actions

Every inference is traceable to the data that drove it. The analyst remains accountable for the final safety determination.

---

## 1. Does the reasoning logic hold?

### What is sound

- The **12-category causal taxonomy (A–L)** is well-grounded and maps cleanly to AP-913 (proximate → contributing → root). The 4-tuple candidate representation *(component, failure mode, causal category, chain position)* anchors every hypothesis to a specific physical item in the equipment model.
- **Allen's interval algebra** (5-relation subset: FOLLOWS, PRECEDES, CONTAINS, OVERLAPS, DURING) is appropriate for this domain and correctly separates causal precursors from consequence followers.
- **TSKR scoring** (onset lag, duration profile, recurrence count per failure mode) is a sound approach to quantifying temporal support.
- **Three binary hard gates** (physical plausibility, timeline consistency, barrier logic) applied before composite scoring is correct — these are pass/fail conditions that should eliminate candidates, not weight them.
- **Coverage enforcement**: requiring every run to either produce at least one candidate per category or explicitly document why the category is ruled out is exactly the right way to prevent silent investigation gaps.
- **Near-tie flagging** and **sensitivity table** are strong: the pipeline doesn't silently pick a winner and it tells the analyst which missing data sources would flip the ranking.

### Reasoning issues

#### 1.1 Allen blend is one-directional (asymmetric boost)

The blend formula in `causality_engine_v32._apply_allen_temporal_blend()`:

```python
new_temporal = max(old_temporal, 0.75 * TSKR_score + 0.25 * allen_score)
# "Allen only raises temporal — never lowers it"
```

Allen acts only as a **confirmation booster**, never as a discriminator. Two candidates with identical TSKR scores but different Allen relations (OVERLAPS vs. PRECEDES) receive identical final scores unless the Allen score exceeds the TSKR score. This is a conservative design choice to avoid false negatives, but it reduces Allen's discriminating power. **If this was intentional, it should be documented in the metamodel.**

#### 1.2 Categories F, K, and L have thin-to-no data representation

| Category | Gap |
|----------|-----|
| **F — External Hazards** | Metamodel explicitly states "No representation in current pipeline data model." The engine defines the category and will generate scaffold candidates, but without `environmental_monitoring` data they score near zero. |
| **K — Vendor/Supply Chain** | Schema coverage exists (`vendor_supply_chain_records.json`) but metamodel states "no current representation in pipeline." Coverage enforcement will flag this, but a flag does not replace evidence. |
| **L — Systemic/Organizational** | Described as "hardest to automate." Only hooks are the Ishikawa `organizational` branch and pattern matching keywords. Scores for this category are the weakest of the 12. |

For regulatory defensibility, "scaffold candidate + attention flag" is a workaround, not a solution. The plan for closing these gaps should be documented.

#### 1.3 Category G: WO-date proximity check not yet implemented

The metamodel explicitly notes: *"proposed WO-date proximity check (not yet implemented)."* Human performance scoring is currently keyword-driven only (`_CATEGORY_KEYWORDS["G"]`). A work order filed 48 hours before an event is treated identically to one filed 6 months prior. This is a meaningful gap for human-performance contributors.

#### 1.4 `_infer_category_from_text()` as a fallback is brittle

When structured data is absent, the causality engine infers causal category from free text via keyword matching (`_CATEGORY_KEYWORDS` table). Text-based category inference is inherently fragile. When this fallback fires, analysts should be notified — it is not currently surfaced as an attention flag.

#### 1.5 Metamodel scaffold candidates for uncovered categories

When a causal category has no natural candidates from the equipment model, `_build_metamodel_scaffolds()` generates placeholder candidates. These ensure coverage enforcement passes but score low and can clutter the ranked list. The attention flag mechanism partially mitigates this, but the behavior is not visible to the analyst in the current RCA card layout.

---

## 2. Can this help system engineers perform RCA?

**Yes — significantly — but with friction points.**

### Where it genuinely helps

| Capability | How it helps |
|------------|-------------|
| **Prevents fixation** | Coverage of all 12 categories is required before a run can close — structurally prevents zeroing in on the first plausible explanation |
| **Traceable scoring** | Allen map, TSKR patterns, and score rationale all trace to specific data fields; every score can be challenged |
| **Near-tie detection** | Two candidates within `review_alternative_gap = 0.10` are explicitly flagged |
| **Sensitivity table** | Tells the SE which missing data sources would flip the ranking — prioritizes investigation effort |
| **Scope expansion signals** | Allen map detects out-of-boundary anomalies and suggests scope revision proactively |
| **Checkpoint/resume** | Pre-built artifacts can be injected; re-run only the scoring step with updated evidence |
| **Ruled-out list** | Eliminated candidates carry documented reason codes — regulatory defensibility |
| **Three causal depths** | RCA card organizes findings at proximate → contributing → root level — matches AP-913 expectations |

### Where there is friction

#### 2.1 Pattern recognition results are fragmented across 5+ artifacts

The four pattern recognition questions (degradation trend? first occurrence or recurrence? common-cause? signature match?) are answered across:

- `tskr_patterns` — degradation trend and signature match
- `allen_relation_map` in `run_manifest` — signal ordering
- `signal_lessons_learned` in `run_manifest.artifacts` — novelty vs. recurrence
- `similar_event_list` — past event matches
- `causality_candidates` — CCF scores and composite scores
- `rca_card.attention_flags` — gaps

An SE without prior familiarity with the artifact structure cannot easily find pattern recognition conclusions. A consolidated **pattern recognition summary block** in the RCA card would materially improve usability.

#### 2.2 The `run()` return dict has 15 keys

`rca_card` and `run_manifest` are the two artifacts the SE reads. They are returned alongside 13 intermediate artifacts. For SE-facing tooling, a wrapper that surfaces only the two primary artifacts (with links to the evidence trail) would reduce cognitive overhead.

#### 2.3 LLM dependency in the synthesizer is a silent degradation risk

`RuleValidatedRCASynthesizerV31` uses an LLM for final synthesis. `DummyLLMClient` is the default in development (and in all show-and-tell tests — `fallback_used: true` in every fixture run). In production, if the LLM call fails or returns poor-quality output, the synthesizer falls back to deterministic rule-based normalization — silently. **The RCA card narrative quality in production has never been validated against a real LLM in an end-to-end scenario.** This is the largest unvalidated production path.

---

## 3. Code robustness

The unit test suite is impressively comprehensive (~70 files, ~900+ test functions). The JSON schema validation layer (34 schemas, `RCAArtifactValidator`) catches structural errors. The issues below are about robustness of the running system, not test absence.

### 3.1 File size / single responsibility

| File | Size | Problem |
|------|------|---------|
| `rca_reasoning_orchestrator.py` | 6,783 lines | Single class handles orchestration, CMMS augmentation, scope management, reentry, manifest assembly, workflow dispatch, attention flags, Chroma archiving |
| `causality_engine_v32.py` | ~3,100 lines / 60+ methods | Candidate generation, scoring, gate logic, sensitivity table, operating point, all in one class |
| `rca_synthesizer_v31.py` | ~3,400 lines | Synthesis, normalization, fallback, safety priority, barrier context, all inlined |
| `causal_condition_adapter.py` | ~2,300 lines | CSV fallbacks, multiple extraction strategies |
| `schema_validator.py` | ~3,190 lines | Draft-7 validation + 20+ semantic check handlers |
| `tskr_temporal_scorer.py` | ~59 KB | — |

The `run()` method alone is 550+ lines. This makes isolated testing, code review, and maintenance expensive.

### 3.2 Silent exception swallowing in optional phases

The pattern `try: <optional_phase>() except Exception: pass` (or `LOGGER.warning` and continue) appears in multiple places. Failures in these phases do not write to `optional_artifact_failures` or appear in the run manifest:

| Location | Phase that can silently fail |
|----------|------------------------------|
| `_attach_epistemics_digests()` | Phase D epistemic annotation |
| `_apply_supersession()` | Phase C supersession |
| FM ID resolution block | `doc_extraction_store.resolve_fm_candidates()` |
| CMMS context build | Live CMMS augmentation |
| Cross-pattern linkage | `_build_cross_pattern_evidence()` |
| `kg_query_utils.is_upstream()` | Topology queries during signal evidence build |
| Signal episode search | `_build_historical_signal_episodes()` |
| NER / LLM disambiguator in hybrid NER pipeline | Entity disambiguation |

**Recommended:** optional phase failures should write a structured entry to `optional_artifact_failures` (or a new `pipeline_warnings` list on the run manifest) so the analyst and auditor can see what ran degraded.

### 3.3 Runtime duck-typing via `inspect.signature()`

In `run()` at lines 571–583, the orchestrator checks whether `refine_with_evidence` accepts certain keyword arguments at runtime via `inspect.signature()`:

```python
sig = inspect.signature(self.causality_engine.refine_with_evidence)
accepts_var_kw = any(
    p.kind == inspect.Parameter.VAR_KEYWORD
    for p in sig.parameters.values()
)
if accepts_var_kw or "coverage_summary" in sig.parameters:
    refine_kwargs["coverage_summary"] = coverage_summary_for_refine
```

This is a fragile pattern. The `CausalityEngine` Protocol should declare the full interface including optional kwargs, eliminating the need for runtime inspection.

### 3.4 `OrchestratorConfig.extra` buries load-bearing settings

Critical production settings are stored in an untyped catch-all dict rather than as proper `OrchestratorConfig` fields:

```python
extra={
    "enable_auto_reentry": True,
    "hard_abort_on_kg_red_state": True,
    "enable_chroma_archive_stage": True,
    "hard_fail_on_chroma_archive_error": True,
    ...
}
```

These settings have real policy consequences. They should be promoted to typed, documented fields on `OrchestratorConfig`.

### 3.5 Coexistence of v31 and v32 causality engines

Both engines are imported in the orchestrator. `v32` is the production default; `v31` is still wired in `build_dev_orchestrator()` as an alternate path. There is no documented sunset for v31. The coexistence adds dead code risk and makes it harder to reason about which scoring logic is active.

### 3.6 All inter-component data is `Dict[str, Any]`

All data flowing between components is `JsonDict = Dict[str, Any]`. Schema validation catches structural errors after the fact but not before. Key paths:

- `signal_evidence/builder.py` falls back to `datetime.now()` when event timestamps are missing — this silently corrupts Allen interval ordering
- `kg_query_utils.is_upstream()` uses `except Exception: return False` — a KG connectivity failure returns "not connected" without distinguishing error from topology
- `parse_dt()` is duplicated in `signal_evidence/builder.py`, `signal_evidence/historian_adapter.py`, and the orchestrator

### 3.7 Cross-pattern document similarity is dead code

In `cross_pattern/rules.py`, `document_similarity_score` is always `None` (Phase 2 placeholder) and `component_overlap` is always `[]`. The document weight (0.30) and component overlap in the confidence formula never contribute — but the weights are presented as active. This should be flagged in the cross-pattern summary output until the placeholder is implemented.

### 3.8 Empty package `__init__.py`

`RCA/__init__.py` is empty. The package has no public API surface. All imports assume a specific path layout (`orchestrators.*` rather than `dackar.RCA.orchestrators.*`). This creates fragility when the package is installed or run from different working directories. `kg_schema_builder_workflow.py` uses `from kg.py2neo_workflow` while other modules use `dackar.RCA.kg.*` fallbacks — inconsistent.

---

## 4. Additional findings from deep module exploration

### 4.1 `signal_evidence` propagation scoring uses magic numbers

Scoring heuristics in `signal_evidence/builder.py` — 0.3 position penalty, hub boost — are unexplained constants. These affect the signal propagation chain scores that feed TSKR, which in turn feeds candidate temporal scores. Their rationale should be documented or made configurable.

### 4.2 OSIsoft PI historian adapter is a placeholder

`signal_evidence/historian_adapter.py` contains `OSIsoftPIHistorianAdapter` which always reports "unavailable." If PI is the production historian, this path is entirely untested.

### 4.3 `EntityRuler` accumulation in NER

In `doc_extraction/adapter.py`, `_make_ner_cs_factory()` adds patterns to a shared spaCy `nlp` object per document. The code notes this and warns it "can grow with corpus." Over a large corpus run this could cause memory growth and unpredictable NER behavior.

### 4.4 No automated regression for notebooks

Show-and-tell assertions live in notebook cells (manual, pre-executed). There is no CI harness that runs the notebooks and checks output fields. If scoring logic changes, the notebooks will silently produce wrong values without a failing test.

### 4.5 TC-7 two-run scope state transfer

The scope expansion workflow (TC-7) documents uncertainty about how `run_context` from Run 1 is passed to Run 2 to activate the scope filter. If the analyst's acceptance is not correctly serialized and re-injected, Run 2 will not apply the boundary filter — silently producing an unscoped result.

---

## 5. Open questions to resolve

| # | Question | Why it matters |
|---|----------|----------------|
| 1 | **Allen blend asymmetry** — was the one-directional design intentional as a conservative anti-false-negative choice, or an artifact? | If intentional, document it in the metamodel; if not, correct it |
| 2 | **Silent exception strategy** — should optional phase failures write to `optional_artifact_failures` / a new `pipeline_warnings` field in the manifest? | Without this, degraded runs look identical to clean runs in the manifest |
| 3 | **Categories F, G (partial), K, L coverage gap** — what is the implementation plan to close these? | Scaffold + flag is not sufficient for regulatory defensibility |
| 4 | **`OrchestratorConfig.extra`** — which settings should be promoted to typed fields? | Untyped settings cannot be validated, documented, or diffed between runs |
| 5 | **LLM role in synthesizer** — how much of the RCA card content comes from deterministic normalization vs. LLM generation in production? | `DummyLLMClient` in every test means LLM synthesis quality is entirely unvalidated |
| 6 | **Cross-pattern document weight** — when will `document_similarity_score` and `component_overlap` be implemented? | Until then, the confidence formula is presenting phantom weights |
| 7 | **v31 sunset** — is v31 being maintained alongside v32 or should it be retired? | Dead code adds maintenance surface and can cause confusion about which scoring is active |

---

## 6. Summary verdict

| Dimension | Assessment |
|-----------|-----------|
| **Reasoning logic** | Sound for categories A–E; meaningful gaps for F, G (partial), K, L; Allen blend asymmetry warrants explicit design decision |
| **SE usability** | Yes, this pipeline can materially help SEs — prevents fixation, surfaces near-ties, maintains audit trail; friction is in fragmented pattern recognition outputs and the 15-key result dict |
| **Code robustness** | Strong unit test coverage but runtime fragility from silent exception swallowing, untyped inter-component data flow, and unvalidated production paths (LLM, PI historian, live CMMS, fleet OE) |
| **Documentation** | Excellent — metamodel and workflow reference guide are high quality; doc/code sync checker (`check_doc_code_sync.py`) is a good practice |
| **Priority fix** | Silent exception handling strategy in optional phases — this is the gap most likely to cause a degraded run to look like a clean run |
