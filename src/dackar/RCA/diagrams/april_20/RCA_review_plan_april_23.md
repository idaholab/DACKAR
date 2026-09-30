# RCA Pipeline Review Plan — April 23, 2026

## Purpose

Structured review of completeness and robustness for the v32 RCA pipeline.
Based on: `RCA_pipeline_stages.md` §3.4–§3.6, `Architecture_Assessment.md`, and `rca_reasoning_oriented_schema.md`.

Two review lenses:
- **Lens 1 — Scenario coverage**: walk each S-1–S-14 scenario against the code; verify pipeline behavior matches §3.4 description; identify missing paths.
- **Lens 2 — Stage contracts**: verify each stage reads only its declared inputs, produces only its declared outputs, and respects the Authoritative / Observed / Inferred / Derived / Decision authority boundary defined in the reasoning schema.

---

## How to Use This Plan

Work items are grouped by priority tier. Within each tier, sequence respects dependencies (items that must be done first are listed first). Each item names the target files and the specific function or field to inspect or change.

**Scope legend**: S = targeted change to 1–2 functions, single session. M = 2–4 files + schema change, 1–2 sessions. L = architectural or data-dependent, multi-sprint.

---

## Dependency Graph

The following dependencies must be respected within P1/P2 work:

```
T-1 (telemetry baseline) ──────────────────────► S-8 (no-anomaly conclusion)
T-2 (NameError fix) ────────────────────────────► [unblocks authoritative-doc testing]
T-4 (governance normalization) ─────────────────► [scoring comparability]
T-5 (Stage C weight normalization) ─────────────► [scoring comparability]
S-2 (observation_role schema) ──────────────────► S-7 (per-pair epsilon for actuation signals)
G-1 (operating_point in scoring) ───────────────► S-12 (condition-dependent latency)
                                                 └─► G-2 (impossible FM pre-filter)
S-8 (conclusion_type) ──────────────────────────── also covers G-4 (no adequate hypothesis)
S-3 (human performance candidate) ──────────────► S-13 (prior CA ineffective — shares WO data path)
```

---

## P1 — Fix Before Review Can Be Called Complete

These items affect regulatory adequacy or produce demonstrably wrong outputs today.

| ID | Title | Scope | Primary files | Function / field | Dependency |
|----|-------|-------|---------------|-----------------|------------|
| **T-2** | ✅ **Fixed Apr 23 (doc bug)** — `recency_fmea`/`recency_eca_rca` were undefined in §3.2 pseudo-code only; actual `_evidence_score_for_fm()` always correct. Pseudo-code rewritten to define all variables and match weights (0.12 FMEA, 0.15 CR/WO, 0.22 ECA/RCA, 0.08 SOP, 0.10 OE). | S | `diagrams/april_20/RCA_pipeline_stages.md` | Stage D evidence pseudo-code | none |
| **T-1** | ✅ **Fixed Apr 23** — baseline was 0.20 (not 0.35). Both `return 0.20` cases in `_telemetry_score_for_fm()` changed to `return 0.0`. Zero-anomaly candidates no longer receive telemetry credit. Prerequisite for S-8. | S | `orchestrators/causality_engine_v32.py` | `_telemetry_score_for_fm()` | none; prerequisite for S-8 |
| **T-4** | ✅ **Already fixed in code** — `_combine_scores()` divides raw sum by `sum(weights)`, normalizing regardless of governance weight. Doc updated Apr 23: §3.2 pseudo-code now shows `gw` variable and `(0.90 + gw)` denominator. | S | `diagrams/april_20/RCA_pipeline_stages.md` | Stage D composite formula | none |
| **T-5** | ✅ **Already fixed in code** — `_normalized_weighted_sum()` in `tskr_temporal_scorer.py` divides by `total_weight`; convex by construction. No change needed. | — | — | — | — |
| **S-8** | ✅ **Fixed Apr 23** — `rca_card.executive_summary.conclusion_type` enum added (`hypothesis_supported / hypothesis_speculative / no_adequate_hypothesis`); `event.actuation_type` enum added (`anomalous / design_signal / spurious_suspected / unknown`); `_compute_conclusion_type()` added to synthesizer; `design_signal` attention flag injected in both branches of `_fallback_card()`; Stage D FM candidate suppression for `design_signal` deferred to Session 3. | M | `synthesis/rca_synthesizer_v31.py`, `schemas/rca_card.json`, `schemas/event.json` | Add `rca_card.conclusion_type` enum; add `event.actuation_type`; Stage D suppress anomaly-based FMs when `actuation_type=design_signal` | T-1 first |
| **S-3** | Human performance root cause invisible to Stage D candidate generation | M | `orchestrators/causality_engine_v32.py`, `orchestrators/kg_context_builder.py`, `schemas/causality_candidates.v3_2.schema.json` | Add `human_performance_candidate` path in `_generate_candidates()`: WO close-date within precursor window on neighborhood component → speculative candidate with `human_factors_flag` | none |
| **S-4** | ✅ **Fixed Apr 23** — `ccf_summary` optional block added to `rca_card.json` schema; `_build_ccf_summary()` added to synthesizer; injected deterministically in both `_fallback_card()` and `synthesize()` (LLM path) when `candidate_count_with_common_cause > 0`; populates `suspected_ccf`, `ccf_confidence`, `shared_mechanism_ids`, `affected_trains` (from per-candidate `train_id_in_oos`), `affected_candidate_ids`, `rationale`. **Not covered**: (1) No new KG traversal for `SupportingSystem` nodes across trains — `ccf_summary` relies entirely on the CCF signal already computed by the causality engine; cross-train topology expansion requires `kg_context_builder.py` changes (see P3 S-10). (2) `sister_components` is not an explicit field on candidates; A/B-series membership is derived from score thresholds in `_build_ccf_summary()`, not from a labeled field. (3) Stage D FM candidate suppression for `actuation_type=design_signal` (originally part of S-8) is still open — candidates are labeled speculative via `conclusion_type` in the card but not filtered upstream in `_generate_candidates()`. | M | `synthesis/rca_synthesizer_v31.py`, `schemas/rca_card.json` | Add `ccf_summary` block to `rca_card` when ≥2 A-series candidates share FM across trains; populate from `common_cause_features.sister_components` + `SupportingSystem` KG path | none |

---

## P2 — Fix Before Next Validation Run

These items produce systematically biased outputs or are scoring consistency issues. None require architectural changes.

| ID | Title | Scope | Primary files | Function / field | Dependency |
|----|-------|-------|---------------|-----------------|------------|
| **T-6** | ✅ **Fixed Apr 23** — added `signal_evidence_chain_score_fm_not_in_kg` warning in `validate_run_bundle()` immediately before the existing TSKR→KG check. Emits warning for each `per_candidate_chain_score` key not in `kg_context.failure_modes[].fm_id`. | S | `validation/schema_validator.py` | `validate_run_bundle()` cross-artifact section | none |
| **S-9** | Documentation-density bias; raw count advantages well-documented FMs | M | `orchestrators/causality_engine_v32.py`, `orchestrators/evidence_retriever.py` | Add `document_density_adjustment` to Stage D `_evidence_prior_for_fm()` (divide by FM corpus density); apply same normalization in `_build_candidate_evidence_summary()` | none |
| **S-5** | Long-latency degradation silently incomplete; historian window vs. FM latency never compared | S | `orchestrators/signal_evidence_builder.py`, `schemas/signal_evidence.json` | Add `window_adequacy_check()`: compare `FailureMode.expected_latency_hours_max` vs. lookback window; emit `fetch_gaps[{fm_id, reason, required_hours, available_hours}]`; propagate to `analyst_attention_flags[]` | none |
| **S-2** | Instrument fault self-referential anomaly lacks `observation_role` distinction | M | `schemas/telemetry_summary.json`, `orchestrators/signal_evidence_builder.py`, `orchestrators/causality_engine_v32.py` | Add `observation_role: process_symptom\|instrument_health\|actuation_signal` to telemetry anomaly schema; Stage B.5 generates instrument-fault candidate when anomaly tag = trip signal tag and `observation_role=instrument_health` | prerequisite for S-7 actuation epsilon |
| **S-7** | Global epsilon = 0.5h wrong for protection signals; source-pair epsilon needed | M | `orchestrators/tskr_temporal_scorer.py`, `schemas/telemetry_summary.json` | Add `sample_interval_seconds` to signal metadata; compute per-pair epsilon = `max(interval_A, interval_B) × 3`; millisecond floor for `actuation_signal` | S-2 (for actuation_role field) |
| **G-1** | `operational_context.operating_point` not used in any scoring dimension | M | `orchestrators/causality_engine_v32.py` | Add operating-state factor to `_structural_score_for_fm()`: condition-specific FM plausibility (cavitation below min-flow, FIV scales with power) | none; prerequisite for S-12, G-2 |
| **S-12** | Static FM latency bounds penalize correct FM at changed operating condition | M | `orchestrators/tskr_temporal_scorer.py`, `orchestrators/kg_context_builder.py` | `_latency_alignment_score()`: accept `operating_point` context; apply scale factor from `FailureMode.latency_operating_point_sensitivity` if present; emit `operating_point_mismatch` flag otherwise | G-1 first |
| **S-13** | No prior-CA ineffectiveness check on recurrence against closed CAP | M | `orchestrators/causality_engine_v32.py`, `synthesis/rca_synthesizer_v31.py`, `schemas/rca_card.json` | `_recurrence_features_for_candidate()`: add `prior_ca_closed_recurred` flag when past event has `resolved=True` and same FM recurs; Stage H injects `prior_ca_ineffective` into `analyst_review.questions_to_resolve[]` | shares WO data path with S-3 |
| **G-4** | No "no adequate hypothesis" output state (partially covered by S-8 `conclusion_type`) | S | `synthesis/rca_synthesizer_v31.py` | Add `conclusion_type=no_adequate_hypothesis` when all candidates B-series and no strong evidence; covered by S-8 fix — verify it applies to evidence-gap case as well | S-8 first |

---

## P3 — Backlog (Important but Bounded or Data-Dependent)

| ID | Title | Scope | Primary files | Notes |
|----|-------|-------|---------------|-------|
| **S-1** | First-occurrence FM: no candidate generated | L | KG content + `orchestrators/causality_engine_v32.py` | Requires KG expansion policy (generic FM entries per equipment/material class) |
| **S-6** | Concurrent unrelated event contaminates telemetry | S | `schemas/telemetry_summary.json` | Add optional `unrelated_event_id` field; analyst-supplied; Stage B.5 suppresses tagged anomalies; add attention flag for topologically unrelated clusters |
| **S-10** | Multi-unit shared-system event outside Unit 1 neighborhood | L | `orchestrators/kg_context_builder.py` | Cross-unit topology expansion for `SupportingSystem` nodes; requires KG multi-unit model |
| **S-11** | Programmatic/AMP root cause invisible to KG search space | L | — | Out of scope for current KG model; requires programmatic knowledge layer; track as roadmap item |
| **S-14** | SPF outside two-hop neighborhood missed | M | `orchestrators/kg_context_builder.py` | Promote `SupportingSystem` nodes that are SPFs for redundant trains regardless of hop count; requires topology annotation `is_spf: true` |
| **G-2** | Physically impossible FMs not pre-filtered | M | `orchestrators/causality_engine_v32.py` | Add operating-point plausibility pre-filter in `_generate_candidates()`; depends on `OperatingCondition` constraint vocabulary in KG | after G-1 |
| **G-3** | Analyst override not fed back to KG | M | `synthesis/analyst_override_processor.py`, KG writer | Override-to-KG writeback path; requires KG write interface |
| **T-3** | `symptom_match()` called but not specified in doc | S | `diagrams/april_20/RCA_pipeline_stages.md` | Add spec of `_symptom_match_score()` to Stage D section; already implemented in `causality_engine_v32.py:1194–1271` |
| **T-7** | Evergreen documents re-embedded on every run | L | `storage/chroma_store.py` | Pre-embedded evergreen layer; scope depends on EDMS/FMEA ingestion implementation |
| **§3.2** | `event.json` / `telemetry_summary.json` production interface unspecified | M | `schemas/event.json`, `schemas/telemetry_summary.json` | Define CMMS/historian producer interface contract; blocking for production deployment |

---

## Lens 2 — Stage Contract Checklist

For each stage, verify: (a) only declared inputs are read, (b) only declared outputs are written, (c) authority categories are respected (no Observed artifact treated as Authoritative, no Inferred used as Derived without uncertainty propagation).

| Stage | Contract check | Status | Notes |
|-------|---------------|--------|-------|
| **A** | Input validation covers JSON schema, not just key presence; `event.severity` flows to severity gates | Open | `orchestrators/rca_reasoning_orchestrator.py` + `schemas/event.json` |
| **B** | `SupportingSystem` nodes included in neighborhood expansion; alias resolution hard-fails on miss; hop limit is documented | Open | `orchestrators/kg_context_builder.py` |
| **5B** | Document authority tags (`mandatory/guidance/informational`) present on all ingested records; EDMS/FMEA stubs flagged in manifest | Open | `orchestrators/kg_context_builder.py`, `storage/` |
| **B.5** | Window adequacy check emitted (S-5 fix); chain acyclicity enforced; `per_candidate_chain_score` keys match KG FM ids (T-6 prereq) | Open | `orchestrators/signal_evidence_builder.py` |
| **C** | ✅ Weight vector convex (T-5 already fixed); ✅ severity weighting applied. Open: epsilon per-pair (S-7); `TemporalPattern` read-only after production | Partial | `orchestrators/tskr_temporal_scorer.py` |
| **D** | ✅ Telemetry baseline 0.0 (T-1 fixed Apr 23); ✅ evidence variables defined (T-2 doc fixed); ✅ governance normalized (T-4 already fixed). Open: human_performance path (S-3); operating_point factor (G-1) | Partial | `orchestrators/causality_engine_v32.py` |
| **E** | `component_ids` filter functional (✅ fixed); BM25 status propagated (✅ fixed); role classification content-only (✅ fixed Apr 23) | Mostly done | `orchestrators/evidence_retriever.py`, `storage/chroma_store.py` |
| **F** | `evidence_posture` 6-value enum used consistently; density adjustment applied (S-9); `rank_delta` computed and non-zero for meaningful re-ranks | Open | `orchestrators/causality_engine_v32.py` `refine_with_evidence()` |
| **G** | `maintenance_human_factors` Ishikawa branch populated when `human_performance_candidate` present (S-3 downstream); skip reason recorded in manifest | Open | `orchestrators/ishikawa_evaluator.py` |
| **H** | ✅ `conclusion_type` emitted (S-8); ✅ `ccf_summary` emitted (S-4); both fixed Apr 23. Open: `prior_ca_ineffective` (S-13); fallback confidence calibration ignores `fallback_used` (✅ verified Apr 23) | Partial | `synthesis/rca_synthesizer_v31.py` |
| **I** | Atomic writes (✅ fixed Sprint 7); `run_status.json` sentinel written (✅ fixed); `output_dir` writability checked at Stage A pre-flight | Mostly done | `orchestrators/artifact_store.py` |
| **J** | ✅ `signal_evidence` chain score FM ids cross-validated (T-6 fixed Apr 23). Open: per-artifact validation still late-binding; `stop_on_validation_error` doesn't distinguish optional vs. required artifacts | Partial | `validation/schema_validator.py` |

---

## Session Sequencing Recommendation

**Session 1 — Code bugs (all S-scope P1/P2 items)**
T-2, T-1, T-4, T-5, T-6. All are targeted single-function changes; no schema impact. Unblock ECA/RCA testing and fix scoring comparability before anything else.

**Session 2 — S-8 + G-4 (no-anomaly conclusion)**
Requires T-1 already done. Changes: `rca_card.json` schema (`conclusion_type`), `event.json` schema (`actuation_type`), `rca_synthesizer_v31.py` (emit `conclusion_type`, suppress FM candidates for `design_signal`). Also closes G-4.

**Session 3 — S-4 (CCF summary)**
Self-contained synthesizer + schema change. No upstream dependencies.

**Session 4 — S-3 + S-13 (human performance + prior CA ineffectiveness)**
Both touch WO data path — do together. `causality_engine_v32.py` + `rca_card.json` schema extension.

**Session 5 — S-9 + S-5 (density bias + window adequacy)**
Both are evidence-layer improvements with no cross-dependencies. Validates the evidence scoring path end to end.

**Session 6 — S-2 + S-7 (observation_role + per-pair epsilon)**
S-2 is prerequisite for S-7; do sequentially in the same session.

**Session 7 — G-1 + S-12 (operating point scoring + condition-dependent latency)**
G-1 is prerequisite for S-12. Both touch `tskr_temporal_scorer.py` and `causality_engine_v32.py`.

**Session 8 — Lens 2 contract sweep**
After all P1/P2 items are done, walk the Stage Contract Checklist above top-to-bottom. Each open item maps to a specific file and function — no analysis required, just verification.

**Backlog — P3 items**
Sequence based on sprint capacity; S-14 and S-6 are smallest. S-1 and S-10 require KG content decisions outside the pipeline code.

---

## Success Criteria

The review is complete when:
- All P1 items pass a targeted unit test on the condensate polisher test case
- All P2 scoring items show non-trivial rank delta on at least one scenario from §3.4
- The Stage Contract Checklist has no Open items remaining
- `run_manifest.pipeline_config.causality_engine_version` is confirmed written on every run
- `Architecture_Assessment.md` has no remaining `[OPEN]` items at P1 or P2 priority

---

*Document owner: Diego Mandelli (diego.mandelli@inl.gov) — created April 23, 2026*
