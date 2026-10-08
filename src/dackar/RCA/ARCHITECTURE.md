# DACKAR RCA — Architecture

This is the canonical, code-grounded description of the `dackar.RCA` subsystem. Every
claim below was checked against the source in this package; where a line number is given
it points at the current code. The curated topic docs under
[`devNotes/`](devNotes/) record *why* choices were made and *what* reviews found; they
may contradict the current code — this file is authoritative.

---

## 1. Overview

`dackar.RCA` turns a single abnormal **event** (plus a **telemetry summary** and a
knowledge-graph context) into a ranked, evidence-backed, schema-validated **RCA card**.

The reasoning is **deterministic and rule-based**: a rule-based causality engine scores
candidate causes against structural, temporal, telemetry, evidence and governance
signals. A final synthesis step can use an LLM to write the narrative, but it degrades
to a deterministic template when no real LLM client is supplied (the output then carries
`fallback_used: true`).

A run is driven end to end by one object, `RCAReasoningOrchestrator`, which calls a set
of injected collaborators (KG builder, scorer, causality engine, evidence retriever,
synthesizer, validator, artifact store) in a fixed order, validating and persisting each
artifact as it goes.

---

## 2. Entry points

- **Orchestrator** — `orchestrators/rca_reasoning_orchestrator.py`:
  - `RCAReasoningOrchestrator` is a `@dataclass` (`:412`).
  - `RCAReasoningOrchestrator.run(event, telemetry_summary, ...)` (`:560`) runs the whole
    pipeline and returns the run bundle (the validated `rca_card` plus every intermediate
    artifact and the `run_manifest`; the returned dict is assembled at `:1242`).
- **Factories**:
  - `build_dev_orchestrator(...)` (`:7106`) wires the live/production collaborators (Neo4j
    KG builder, Chroma evidence retriever, file artifact store). It selects the causality
    engine by `causality_engine_version` (`"v32"` default, `"v31"` supported) at `:7206`.
  - `build_fixture_orchestrator(...)` (`tests/RCA/scenario/shared/run_helpers.py:54`) wires
    an **offline** orchestrator for fixtures and tests — see §9.
- **Package root** — `src/dackar/RCA/__init__.py` is **empty**. There is no curated
  façade; import subpackages by their dotted path (e.g.
  `from dackar.RCA.orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator`).

---

## 3. Pipeline stages

The real order of work inside `run()` is below. Each line names what the stage reads, the
artifact it writes, and the method or collaborator that implements it. Artifacts are
validated and persisted as they are produced (see §5).

1. **PM compliance (optional input)** — reuse the provided `pm_compliance`, or build it
   via `_build_pm_compliance_if_needed` (`run()` `:636`).
2. **Input validation + guards** — `_validate_bundle(stage="inputs")` (`:651`) and
   `build_input_guards` (`:659`).
3. **Stage A — run context** — `_stage_a_build_run_context` (`:666`, method at `:2382`)
   writes `run_context`.
4. **Stage B — KG context** — `kg_context_builder.build(...)` (`:695`), with KG governance
   enforcement (`:704`) and optional live-CMMS augmentation of `kg_context` (`:733`). Writes
   `kg_context`.
5. **Stage B.5 — signal evidence** — `_build_signal_evidence` (`:769`) writes
   `signal_evidence`.
6. **Stage C — TSKR temporal patterns** — `_build_tskr_patterns` → `tskr_temporal_scorer.score`
   (`:778`) writes `tskr_patterns`.
7. **Stage D — causality candidates** — `causality_engine.generate(...)` (`:792`), followed
   by the scope-boundary filter `_apply_scope_boundary_filter` (`:811`). Writes
   `causality_candidates`.
8. **Stage E — evidence bundle** — `evidence_retriever.retrieve(...)` (`:835`), then Phase C
   supersession via `_apply_supersession` (`:842`). Writes `evidence_bundle`.
9. **Evidence refinement** — if the engine exposes `refine_with_evidence` (guarded at
   `:856`), the pre-refine candidates are snapshotted to
   `causality_candidates_pre_refine` and the candidates are re-scored
   (`refine_with_evidence(...)` `:905`). An Allen relation map is pre-computed at `:850`.
10. **Optional auto re-entry** — `_run_auto_reentry_if_needed` (`:908`) can regenerate the
    candidate set; the scope boundary is reapplied afterwards (`:940`). Writes
    `reentry_execution`.
11. **Optional Ishikawa** — when `config.enable_ishikawa`, `ishikawa_evaluator.evaluate(...)`
    (`:964`) writes `ishikawa_matrix`.
12. **Barrier analysis** — `_compute_barrier_analysis` (`:1001`) writes `barrier_analysis`.
13. **Similar-event / signal-episode / cross-pattern (optional)** —
    `_build_similar_event_list` (`:1012`), `_build_historical_signal_episodes` (`:1023`,
    gated), `_build_cross_pattern_evidence` (`:1053`, gated).
14. **Phase D epistemics** — `_attach_epistemics_digests(...)` (`:1072`) attaches an
    epistemics digest to each candidate before synthesis.
15. **Synthesis** — `rca_synthesizer.synthesize(...)` (`:1074`) writes the `rca_card`,
    followed by a long series of `_apply_*_attention_flag(s)` passes (`:1088`–`:1120`).
16. **Output validation** — `_validate_bundle(stage="outputs")` (`:1123`).
17. **Stage I — Chroma archive** — `_stage_i_archive_chroma` (`:1140`, method at `:1696`).
18. **Stage G — run manifest** — `_stage_g_finalize_manifest` (`:1156`, method at `:2953`)
    writes `run_manifest`, then workflow dispatch (`:1189`) and final `run_status` (`:1236`).

**Pre-computed inputs skip stages.** `run()` accepts `kg_context`, `signal_evidence`,
`tskr_patterns`, `causality_candidates` and `evidence_bundle` as optional keyword
arguments; when supplied, the matching stage reuses them instead of recomputing (the
`if <artifact> is None:` guards at `:694`, `:768`, `:777`, `:791`, `:834`). This is how
fixture runs short-circuit stages they do not exercise.

### Stage labels: letters vs numbers

Two parallel labelings exist, and they are **not** the same scheme — do not conflate them:

- **Lettered `stage_health`** (`_compute_stage_health` `:3463`): keys such as
  `stage_b_kg_context`, `stage_c_temporal`, `stage_d_causality`, `stage_e_evidence`,
  `stage_g_structuring`, each with a green/yellow/red status.
- **Numbered `analyst_checkpoints`** (`_build_analyst_checkpoints` `:6387`): steps
  `0 scoping`, `1 data_management`, `2 kg_expansion`, `3 pattern_recognition_documentary`,
  `3.5 pattern_recognition_signal`, `4 candidate_generation`,
  `5 ranking_and_evidence_assessment`, `6 conclusion`. Steps 5 and 6 can be analyst
  decision gates.

There is **no "Stage F" or "Stage H"** in the code — only `_stage_a_*`, `_stage_g_*` and
`_stage_i_*` methods exist. B through E are inline collaborator calls surfaced through the
`stage_health` keys above.

---

## 4. Collaborators: protocols and concrete implementations

The orchestrator depends on `Protocol` interfaces (defined at the top of
`rca_reasoning_orchestrator.py`) and is given concrete implementations by injection — the
required ones as dataclass fields, the optional ones as dataclass fields with `None`
defaults or via `set_*` methods.

| Protocol (`:line`) | Concrete impl wired by `build_dev_orchestrator` |
| --- | --- |
| `KGContextBuilder` (`:77`) | `Neo4jKGContextBuilder` (`kg_context_builder.py`) |
| `TSKRTemporalScorer` (`:114`) | `TSKRTemporalScorerV1` |
| `CausalityEngine` (`:151`) | `RuleBasedCausalityEngineV32` (v31 selectable, see §6) |
| `EvidenceRetriever` (`:190`) | `ChromaEvidenceRetriever` over an `EvidenceStore` |
| `RCASynthesizer` (`:225`) | `RuleValidatedRCASynthesizerV31` |
| `IshikawaEvaluator` (`:264`) | `HeuristicIshikawaEvaluatorV1` (only when Ishikawa enabled) |
| `SchemaValidator` (`:301`) | `RCAArtifactValidator` (`validation/schema_validator.py`) |
| `ArtifactStore` (`:325`) | `FileArtifactStore` (`orchestrators/artifact_store.py`) |

**Optional, injected collaborators** (dataclass fields defaulting to `None`, or `set_*`
injectors at `:454`–`:470`): `cap_adapter`, `cmms_adapter`, `workflow_dispatch_adapter`,
`similar_event_adapter`, `doc_extraction_store`, `pattern_searcher`,
`cross_pattern_linker`, `epistemics_classifier`. When one is absent, its stage is skipped
or degrades (see §7); the pipeline still completes.

The dev and fixture factories differ only in which concretes they inject: the fixture
factory swaps the live KG builder for `_StubKGContextBuilder`, Chroma for an in-memory
store, and the LLM for a dummy client (see §9).

---

## 5. Data model, artifacts and validation

**Artifact bundle.** `run()` returns a dict keyed by artifact name (`:1242`): `run_context`,
`pm_compliance`, `kg_context`, `signal_evidence`, `tskr_patterns`, `causality_candidates`
(and `causality_candidates_pre_refine`), `evidence_bundle`, `ishikawa_matrix`,
`barrier_analysis`, `reentry_execution`, `cmms_context`, `rca_card`, `input_validation`,
`output_validation`, `run_manifest`. The same artifacts are persisted through the
`ArtifactStore` during the run.

**Central artifacts:**
- `causality_candidates` — the ranked cause hypotheses with category coverage (§6) and
  the hard gates applied by the engine.
- `evidence_bundle` — retrieved evidence snippets plus per-candidate supporting-evidence
  summaries, after supersession.
- `rca_card` — the final analyst-facing artifact (narrative, ranked causes, attention
  flags, barrier summary, analyst-review block).

Artifacts are joined by shared identifiers (e.g. `event_id`, `asset_id`), which the
cross-artifact semantic checks rely on.

**Schemas.** `src/dackar/RCA/schemas/` holds **32** Draft-7 JSON schemas (one per artifact
type: `event.json`, `telemetry_summary.json`, `causality_candidates.json`,
`evidence_bundle.json`, `rca_card.json`, `run_manifest.json`, and so on). The validator
registers **26** of these as `CORE_ARTIFACTS` (`validation/schema_validator.py:111`).

**Validator.** `RCAArtifactValidator` (`validation/schema_validator.py:99`) works in two
layers:
1. per-artifact Draft-7 validation — `validate_artifact` (`:180`), with per-type semantic
   checks (`_semantic_checks_single` `:399`, dispatching to card/manifest/run-context/
   causality-candidate checks);
2. cross-artifact consistency over a whole bundle — `_semantic_checks_bundle` (`:640`),
   invoked from the bundle validation path (`:315`).

It has three **modes** (`:144`): `strict` (validate as-is), `compat` (default; normalize
legacy field aliases first) and `warn_only` (downgrade every failure to a warning). The
orchestrator calls it through `_validate_and_persist` (`:2826`), `_validate_bundle`
(`:6596`) and `_validate_artifact` (`:6573`). Whether a failure raises depends on
`config.stop_on_validation_error` and whether the artifact is required or optional (§8).

> Naming caution: the `rca_card` has a top-level `contributing_causes` **and** a nested
> `executive_summary.causal_depth_summary.contributing_causes`; they are different fields.

---

## 6. Causality engine

`RuleBasedCausalityEngineV32` (`orchestrators/causality_engine_v32.py:172`) is
deterministic and rule-based. `RuleBasedCausalityEngineV31` is retained as a structural
baseline and is selectable via `causality_engine_version` in `build_dev_orchestrator`
(`:7206`).

- `generate(...)` (`:227`) builds candidates, assigns each a primary **cause category**,
  scores the structural / temporal / telemetry / evidence / governance streams, then
  applies thresholds and keeps the top-k.
- `refine_with_evidence(...)` (`:1046`) re-scores candidates with the retrieved evidence
  (including Allen temporal relations and protection-logic context). It **deep-copies its
  input** rather than mutating it, which is why the orchestrator keeps both
  `causality_candidates_pre_refine` and the refined set.

**Cause categories.** The engine uses a 12-category metamodel, **A–L**
(`_CATEGORY_PROFILE_NAMES` `:198`): A–F map to `equipment_origin` variants, G
`human_performance`, H `design_deficiency`, I `change_control`, J `surveillance`, K
`vendor_procurement`, L `organizational`. Category A is the default intrinsic category;
B–L are keyword-mapped (`_CATEGORY_KEYWORDS` `:185`).

---

## 7. Package map (core vs optional vs feeder)

**Core reasoning path** (always exercised by a normal run):
- `orchestrators/` — the orchestrator, the causality engines, the KG context builder
  (`kg_context_builder.py`), the TSKR scorer, the artifact store, the Ishikawa evaluator.
- `signal_evidence/` — signal-evidence construction (Stage B.5).
- `synthesis/` — the RCA synthesizer that writes the card.
- `validation/` — the schema + semantic validator.
- `storage/` — the evidence store backing evidence retrieval.
- `pm_compliance/` — PM-compliance artifact builder (feeds Stage D governance).
- `ner/`, `doc_extraction/` — entity and causal-condition extraction backbone (causal
  logic lives in `ner/causal_condition_adapter.py`).

**Optional / pluggable** (adapter- or Protocol-gated; degrade to skipped/empty when their
collaborator is not injected, see §4): `cmms_integration/`, `cap_integration/`,
`adapters/`, `cross_pattern/`, `log_pattern_recognition/`, `equipment_similarity/`.

**Offline feeders** (not part of a live `run()`): `doc_parsers/` (e.g. FMEA → KG
ingestion), `summarizers/`.

**Not a Python package:** `viz/` is a **standalone Streamlit viewer** (`viz/app.py`,
`viz/loader.py`; no `__init__.py`). It only *loads and displays* artifact JSON already
produced by a run — it does not call the orchestrator. See
[`viz/RCA_VIZ_ARCHITECTURE.md`](viz/RCA_VIZ_ARCHITECTURE.md).

> There is **no `kg/` package and no `causal/` package** under `src/dackar/RCA/`.
> KG-context building is `orchestrators/kg_context_builder.py`; causal extraction is in
> `ner/` plus the top-level `src/dackar/causal/` package.

---

## 8. Configuration

Behavior is tuned through `OrchestratorConfig` (`:339`). The knobs that change what runs:

- `enable_ishikawa` (default `False`) — build the optional Ishikawa matrix.
- `persist_intermediate_artifacts` (default `True`) — persist per-stage artifacts, not
  just the card.
- `stop_on_validation_error` (default `True`) — when `True`, a required-artifact failure,
  or a genuine failure in an optional stage (supersession / epistemics), raises; when
  `False`, optional-stage failures are recorded in the run's `optional_artifact_failures`
  and logged instead.
- `top_k_candidates` (5), `top_k_evidence` (10) — caps carried into generation / retrieval.
- Semantic-recurrence, signal-episode and cross-pattern flags
  (`enable_semantic_recurrence`, `enable_signal_episode_search`,
  `enable_cross_pattern_linkage`) gate the optional stages in §3 step 13.
- `extra` (free-form dict) — consulted by optional stages; keys include
  `causality_engine_version`, `enable_auto_reentry`, `enable_chroma_archive_stage`,
  `strict_red_state_governance`, `hard_abort_on_kg_red_state`.

---

## 9. Fixture-only runs and testing

`build_fixture_orchestrator(...)` (`tests/RCA/scenario/shared/run_helpers.py:54`) builds an
orchestrator that needs no live services:

- **KG** — `_StubKGContextBuilder` (`:39`); `kg_context` is supplied as a fixture rather
  than queried from Neo4j.
- **Evidence store** — `InMemoryEvidenceStore` (no Chroma).
- **LLM** — `DummyLLMClient` by default, so the synthesizer takes its deterministic
  template path and the card reports `fallback_used: true`.
- **Config** — `stop_on_validation_error=False`, and the `extra` flags disable auto
  re-entry, Chroma archive and strict red-state governance (`:185`–`:200`).

`load_fixtures(fixture_dir)` (`:228`) loads the recognised fixture files for a scenario
(an `event`, a `telemetry_summary` and a `kg_context` are the usual minimum).

Tests live under the repository-level `tests/` directory:
- `tests/RCA/unit_tests/` — unit and component tests.
- `tests/RCA/scenario/` — fixture-only end-to-end scenarios plus the shared
  `run_helpers`.

Run from the repo root (`pytest.ini` sets `pythonpath = src`):

```
python -m pytest tests/RCA/unit_tests -q
```

---

## 10. Historical design notes

[`devNotes/`](devNotes/) holds seven curated topic docs distilled from the April–August 2026
working notes — the design rationale and development results that are not derivable from the
code itself:

- [`METAMODEL.md`](devNotes/METAMODEL.md) — the 12-category causal taxonomy and its regulatory mapping.
- [`DATA_MANAGEMENT.md`](devNotes/DATA_MANAGEMENT.md) — the data families and the degrade-don't-fail stance.
- [`CAUSAL_EXTRACTION.md`](devNotes/CAUSAL_EXTRACTION.md) — document causal extraction and its evaluation.
- [`EPISTEMICS.md`](devNotes/EPISTEMICS.md) — the epistemic-role classification and evidence routing.
- [`PM_COMPLIANCE.md`](devNotes/PM_COMPLIANCE.md) — the PM-compliance module design and review.
- [`PIPELINE_STAGES.md`](devNotes/PIPELINE_STAGES.md) — the staged workflow and the scoring arithmetic.
- [`REVIEWS.md`](devNotes/REVIEWS.md) — the review chronology, findings and resolutions.

They record choices and measurements as of their dates and are **not** maintained in lockstep
with the code. Where they disagree with this file or with the source, the code and this
`ARCHITECTURE.md` win. The two IAEA TECDOC PDFs under `devNotes/june_5/` (TECDOC-1112 / ASSET
and TECDOC-1756) are the external standards the reviews are grounded in.
