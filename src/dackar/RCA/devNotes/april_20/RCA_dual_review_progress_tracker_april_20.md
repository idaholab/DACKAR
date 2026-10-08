# RCA workflow — dual review and progress tracker

**Date:** April 20, 2026  
**Scope:** `DACKAR/src/dackar/RCA` (orchestrators, synthesis, evidence retrieval, validation, unit tests)  
**Related docs (this folder):** `RCA_workflow_april_2.md`, `RCA_Engineer_Needs_Requirements - 2.md`, `RCA Data Elements.md`, `RCA_Data_Management_Strategy.md`, `Architecture_Assessment.md`, `RCA_Systems_Engineering_Review_April_20.md`, **`RCA_TSKR_orchestrator_causality_deep_pass_april_20.md`** (second pass: TSKR + orchestrator + v31/v32 causality)

**Purpose of this file:** Single place to track **systems-engineering value**, **functional (data → decision) logic**, **code-level risks**, and **test coverage** as the RCA capability evolves. Update it when gaps close or new risks appear.

---

## 1. Executive summary

The RCA pipeline is a **serious decision-support architecture**, not a black-box “root cause finder.” Typed stage artifacts, schema validation, a **pre- then post-evidence** candidate lifecycle, deterministic synthesis fallback, and explicit human-review hooks align well with nuclear engineering expectations for traceability and conservative inference.

The main limitations are **not missing enthusiasm for ML** but **known product risks**: closed-world search bounded by the KG, heuristic temporal and retrieval layers, configuration and schema drift, and a few places where **card-level narrative can diverge from what a regulator would expect** unless the analyst cross-reads all artifacts.

**Companion review:** `RCA_Systems_Engineering_Review_April_20.md` already captures many methodology-level gaps (KG closed world, Ishikawa ordering, past-event-as-primary, safety-function surfacing, scoring evolution artifact, and more). This tracker **does not replace** that document; it adds **code-verified** notes and a **test / progress** section.

---

## 2. Systems engineer lens — value and decision support

### 2.1 Where the approach creates value

| Need (from engineer-needs / data-elements docs) | How the current workflow supports it |
|-------------------------------------------------|--------------------------------------|
| Fragmented plant data | KG context + ranked documents + past events + optional CMMS injection concentrate **relevant** context per event. |
| Traceability | Per-artifact provenance, validation sidecars, run manifest, candidate `score_rationale`, evidence summaries. |
| Uncertainty and alternatives | Dual thresholds, review-alternative rescue, evidence posture, synthesizer gates, `analyst_review` questions. |
| Recurrence and common-cause thinking | Explicit recurrence / common-cause summaries and scoring dimensions in v32. |
| Conservative posture when automation is weak | **Dummy LLM → deterministic fallback** is the right default for a regulated environment while LLM paths mature. |

### 2.2 Where the engineer must still “own” the analysis

- **Search space** is largely **KG-defined**; novel mechanisms absent from the graph may never appear as candidates (see closed-world discussion in `RCA_Systems_Engineering_Review_April_20.md` §2.1).
- **TSKR / latency** uses heuristics; telemetry quality and FMEA currency are assumptions, not guarantees.
- **Evidence role** is still partly lexical / metadata-driven; domain synonyms and ambiguous WO narratives remain edge cases despite structured `condition_assessment` boosts in `evidence_retriever.py`.
- **Ishikawa** (when enabled) is a **structuring** stage after causal ranking, not a full INPO-style investigation driver (same companion doc §3.2).

**Bottom line:** The workflow **does support** system engineer decisions by **compressing retrieval, surfacing hypotheses, and forcing explicit review artifacts** — provided leadership treats outputs as **hypothesis sets + audit package**, not signed conclusions.

---

## 3. Code reviewer lens — data to decisions (functional trace)

Canonical order in `RCAReasoningOrchestrator.run()`:

1. **Inputs** → `run_context`, validation bundle.  
2. **KG context** (`Neo4jKGContextBuilder`) → causal search space, documents, failure modes, optional live CMMS context.  
3. **TSKR** (`TSKRTemporalScorerV1` or null stub) → `tskr_patterns`.  
4. **Candidates v1** (`RuleBasedCausalityEngineV32.generate`) → screened, ranked failure-mode set (+ event analogs).  
5. **Evidence** (`ChromaEvidenceRetriever.retrieve`) → merged hits, dedupe, per-candidate summaries, BM25 availability signal in bundle metadata.  
6. **Candidates v2** (`refine_with_evidence`) → evidence-adjusted scores and postures.  
7. **Ishikawa** (optional) → `ishikawa_matrix`.  
8. **RCA card** (`RuleValidatedRCASynthesizerV31.synthesize`) → LLM attempt, then **fallback** on failure / semantic errors.  
9. **Manifest** → review hooks, writeback flags.

### 3.1 Verified code behaviors (April 20, 2026)

| Topic | Observation |
|-------|-------------|
| **LLM hallucinated primary `candidate_id`** | Treated as a **hard validation error** pathing to fallback (`rca_synthesizer_v31.py` ~119–132). Aligns with “must not invent candidates” intent. |
| **Minimum evidence gate** | `_passes_minimum_evidence_gate` uses `(ev.get("support_role") or "").strip().lower() == "supporting"` — the spec’s “strip / explicit string handling” concern is **addressed** in current code. |
| **Fallback evidence excerpts** | `_fallback_card` sets `"excerpt": e.get("snippet", "")` from selected evidence rows — **not** only synthetic summary text. (If `RCA_Systems_Engineering_Review_April_20.md` §4.5 still claims summary-only excerpts, **update that section** to match code or document cases where `snippet` is empty.) |
| **Component scoping vs Chroma** | `chroma_store.py` documents post-filtering for `component_ids` (cannot live in Chroma `where` for list-shaped metadata). This is **not** a silent no-op at the store layer when using that adapter; behavior depends on which `EvidenceStore` implementation is wired. |
| **Confidence when `fallback_used`** | `_calibrate_primary_confidence` **caps at `medium` whenever `fallback_used` is true** (`rca_synthesizer_v31.py` ~1209–1210). With default `DummyLLMClient`, **every** successful run uses fallback → **“high” is unreachable**, matching the companion review §3.4. This is a **product semantics** issue: either accept “medium max” as policy for template-only cards, or decouple “LLM narrative risk” from “deterministic template.” |

### 3.2 Functional pitfalls (implementation-facing)

- **Orchestrator size** — `rca_reasoning_orchestrator.py` remains a **god module** (validation adapters, CMMS, CAP export, overrides). Harder to test in isolation and easier to regress.  
- **Validation mode** — `RCAArtifactValidator` defaults to `compat`; strict runs may still fail on legacy/schema drift (documented in `RCA_workflow_april_2.md` for `kg_context`).  
- **Stage ordering** — No second-pass KG expansion after evidence changes the leader; companion doc §3.1 remains valid.  
- **Past event analogs** — Still able to rank highly in pools that feed synthesis if not filtered upstream; methodology concern in companion §3.3 remains.  
- **Py2Neo / Neo4j** — Live graph dependency for full runs; notebooks and fixtures must stay aligned with schema versions.

---

## 4. Unit tests — appropriateness and coverage

### 4.1 Inventory (under `RCA/unit_tests/`)

Rough `def test_` counts by file (grep-based, April 20, 2026):

| File | Approx. tests | Primary focus |
|------|----------------|---------------|
| `test_tskr_temporal_scorer.py` | 117 | Temporal scoring, edge cases |
| `test_past_event_scoring.py` | 25 | Past-event / recurrence scoring |
| `test_telemetry_scoring.py` | 18 | Telemetry dimension |
| `test_causality_scoring.py` | 19 | Shared scoring helpers |
| `test_symptom_match.py` | 15 | Symptom / anomaly alignment |
| `test_evidence_scorer.py` | 17 | Hit assessment paths |
| `test_synthesizer_validation.py` | 14 | Card semantics |
| `test_refine_with_evidence.py` | 14 | Post-evidence refinement |
| `test_synthesizer_gates.py` | 14 | Gates including minimum evidence |
| `test_review_hooks.py` | 13 | Manifest / review hooks |
| `test_fallback_decision.py` | 13 | Fallback decision status |
| `test_query_builder.py` | 12 | Retrieval query plans |
| `test_review_alternative.py` | 10 | Review alternative rescue |
| `test_evidence_summary.py` | 9 | Candidate evidence summaries |
| `test_evidence_dedup.py` | 7 | Dedup / rank |
| `test_synthesizer_fallback.py` | 7 | Fallback scoping / A10-style guards |
| **Plus** | — | `test_analyst_override.py`, `test_cap_adapter.py`, `test_cap_export_serializer.py`, `test_cmms_context_builder.py`, `test_equipment_similarity_resolver.py` |

**Total:** on the order of **230+** focused tests (including large TSKR suite).

### 4.2 Strengths

- **Heavy concentration** on the highest-risk areas: **evidence scoring**, **dedupe**, **refinement**, **synthesizer semantics**, **TSKR**.  
- Tests are mostly **small fixtures + one behavioral assert**, which aids regression diagnosis.  
- **Fixture JSON** under `RCA/tests/test_case_*` supports integration-style replay.

### 4.3 Gaps / recommendations

| Gap | Why it matters |
|-----|----------------|
| **No lightweight `RCAReasoningOrchestrator.run()` unit test** in `unit_tests/` (with in-memory stores) | Stage wiring, `optional` Ishikawa failures, and manifest fields are only indirectly covered. |
| **`RCA_workflow_april_2.md` mentions** `test_causality_engine_generate.py`, `test_stage_de_contract.py`, `test_stage_ef_contract.py` — **not present** in `unit_tests/` as of this review | Stage-boundary contracts would catch handoff bugs between engine ↔ retriever ↔ refine. |
| **Synthesizer / confidence policy** | Add explicit tests for “fallback_used forces medium cap” and for a **future** flag that allows high confidence on deterministic fallback when product policy changes. |
| **Environment** | Local `pytest` failed with `ImportError: HookimplOpts` from `pluggy` — fix versions (`pytest` / `pluggy`) in the project env so CI and devs get a green run. |

---

## 5. Progress tracker (living checklist)

Use this table to record closure of issues. Dates are optional; status is the source of truth.

| ID | Item | Type | Status | Notes / owner |
|----|------|------|--------|----------------|
| P1 | Confidence cap when `fallback_used` (policy vs engineering) | Product / code | Open | See §3.1 — decide whether deterministic path may yield `high`. |
| P2 | Scoring v1→v2 delta as named artifact or manifest section | Schema / UX | Open | Companion doc §4.2 |
| P3 | `contributing_causes[]` on `rca_card` | Schema / synth | Open | Companion doc §2.5 |
| P4 | Past-event analog as non-primary mechanism | Logic | Open | Companion doc §3.3 |
| P5 | Ishikawa-driven retrieval completeness | Architecture | Open | Companion doc §3.2 |
| P6 | Stage DE / EF contract tests | Tests | Open | Files absent vs workflow spec |
| P7 | Orchestrator thin-slice test (in-memory) | Tests | Open | §4.3 |
| P8 | Strict schema: `kg_context` single canonical shape | Schema | Open | `RCA_workflow_april_2.md` §4 |
| P9 | `pytest` / `pluggy` compatibility in dev env | Tooling | Open | §4.3 |
| P10 | Refresh `RCA_Systems_Engineering_Review_April_20.md` §4.5 if excerpts are now snippet-backed | Docs | Open | Code shows `excerpt` from `snippet` |

---

## 6. Changelog (this document)

| Date | Change |
|------|--------|
| 2026-04-20 | Initial dual review + progress tracker created. |
