# Second pass — TSKR module, orchestrator, and dual causality engines (v31 / v32)

**Date:** April 20, 2026  
**Code baseline:** `orchestrators/tskr_temporal_scorer.py`, `orchestrators/temporal_relations.py`, `orchestrators/rca_reasoning_orchestrator.py`, `orchestrators/causality_engine_v31.py`, `orchestrators/causality_engine_v32.py`

This note is a **lineage-aware** review: how telemetry intervals become `tskr_patterns`, how each causality engine consumes them, and how the orchestrator chooses engines and follow-on stages.

---

## 1. End-to-end wiring (orchestrator)

### 1.1 Stage order (relevant slice)

From `RCAReasoningOrchestrator.run()`:

1. Inputs validated → `run_context` persisted.  
2. `kg_context` built (or injected).  
3. **`tskr_patterns`**: if caller did not supply it, `self.tskr_temporal_scorer.score(...)` runs when the scorer is non-`None`; otherwise a **null stub** (empty patterns, `mode: absent`) is emitted.  
4. **`causality_candidates`**: `self.causality_engine.generate(..., tskr_patterns=tskr_patterns, ...)`.  
5. `evidence_bundle` from `evidence_retriever.retrieve`.  
6. **Refinement:** `if hasattr(self.causality_engine, "refine_with_evidence"):` then `refine_with_evidence(...)` and re-persist `causality_candidates`.  
7. Synthesis, manifest, etc.

### 1.2 Engine selection — **only one engine per run**

The orchestrator holds **one** `causality_engine: CausalityEngine`. There is **no** built-in dual execution (A/B run v31 and v32 in one `run()`).

`build_dev_orchestrator(..., causality_engine_version: str = "v31", ...)` selects:

- `"v32"` → `RuleBasedCausalityEngineV32` with `CausalityEngineConfigV32(top_k_candidates=...)`.  
- `"v31"` → `RuleBasedCausalityEngineV31` with `CausalityEngineConfig(...)`.  
- Anything else → `ValueError`.

**Always** attaches `TSKRTemporalScorerV1()` as `tskr_temporal_scorer` (not configurable in this factory).

### 1.3 Manifest metadata caveat

`run_manifest["pipeline_config"]["causality_engine_version"]` is taken from:

`(self.config.extra or {}).get("causality_engine_version", "v31")`.

`build_dev_orchestrator` sets `extra["causality_engine_version"]` consistently with the factory argument. If a caller constructs `RCAReasoningOrchestrator` **manually** with `RuleBasedCausalityEngineV32` but **forgets** to set `OrchestratorConfig.extra["causality_engine_version"] = "v32"`, the manifest will **mis-report** `v31` while v32 logic ran.

`evidence_refinement_applied` is inferred from `causality_candidates.provenance.evidence_refinement_applied` — **false for v31** paths (v31 has no `refine_with_evidence`).

---

## 2. TSKR temporal scorer (`TSKRTemporalScorerV1` + `temporal_relations`)

### 2.1 Role

For **each** `kg_context.failure_modes[]` row, emit one **pattern** object keyed by `target_id == fm_id`, with Allen `relation`, lag stats, latency alignment to `expected_latency_min_hours` / `max_hours`, recurrence fields, and scalar `confidence` / `support`.

### 2.2 Allen layer (`temporal_relations.py`)

- Intervals **A** = anomaly window, **B** = event interval (`timestamp_start` / `timestamp_end`).  
- Classification order: **FOLLOWS** → **PRECEDES** → **CONTAINS** → **OVERLAPS** → default **DURING**.  
- `CAUSAL_PRIORITY` for picking a **single dominant relation** across many windows: **OVERLAPS > CONTAINS > PRECEDES > DURING > FOLLOWS**.  
- `onset_lag_hours`: `b.start - a.start` (positive ⇒ anomaly onset before event onset).

This is a **sound, documented** discretization for “degradation vs event” reasoning.

### 2.3 Important modeling property (cross-FM coupling)

`_score_against_anomalies` uses the **same** sorted anomaly list for **every** failure mode. The **dominant Allen relation** and **severity-weighted mean/std lag** therefore reflect **global** telemetry vs the event — not “this FM’s expected sensors only.”

Per-FM differentiation in TSKR comes mainly from:

- **`_latency_alignment_details`** (expected min/max **from that FM**), and  
- **`_score_history_support`** (past events matched to that FM’s `fm_id` **or** `component_id`).

So two failure modes on the same component can get **identical** `relation` / `mean_lag_hours` / `anomaly_score` from the Allen stage but **different** `latency_alignment_score` / `temporal_contradiction` if their FMEA latency windows differ.

### 2.4 Confidence composition (TSKR)

```text
confidence = clamp01(
    anomaly_weight   * max(anomaly_score, telemetry_support)
  + latency_weight   * latency_score
  + history_weight   * history_score
  + anomaly_count_weight  * anomaly_count_score
  + lag_consistency_weight * lag_consistency_score
  - (0.20 if temporal_contradiction else 0.0)
)
```

Weights default to **0.55 / 0.30 / 0.15 / 0.20 / 0.15** — they **sum to 1.35** before the contradiction penalty. That is **not** a normalized convex combination; it is a **weighted sum with headroom** then clamped to [0,1]. Intentional or not, **tuning one weight without rebalancing others** changes total scale in a non-obvious way (also noted at workflow-spec level for the causality engines’ five dimensions).

### 2.5 Recurrence / past events (TSKR)

`_build_recurrence_profile` includes a past event if:

- `fm_id in pe.matched_failure_mode_ids`, **or**  
- `pe.component_id == component_id` **without** FM match.

The **OR second clause** can admit events that share the component but **not** the failure mode, inflating `recurrence_count` and history_score for that FM. Whether that is desirable depends on KG hygiene; it is a **review-worthy** assumption.

### 2.6 Provenance timestamp

`provenance.generated_at` uses `datetime.utcnow().replace(microsecond=0).isoformat() + "Z"`. That is **naive UTC** labeled with `Z`. Prefer `datetime.now(timezone.utc)` for clarity and Python 3.12+ alignment.

### 2.7 Severity weighting (TSKR vs older specs)

The scorer applies **`_severity_weight`** to Allen relation aggregation, lag mean/variance, and **`_effective_anomaly_count`** for `anomaly_count_score`. This addresses “all timestamps equal” at the **TSKR** layer. (Causality engines still have their own telemetry dimensions.)

---

## 3. TSKR pattern index — **v31 and v32 are aligned**

Both engines implement:

- `_index_tskr_patterns` → `Dict[target_id, List[pattern]]` (append all patterns per target).  
- `_lookup_tskr_pattern` → `max(patterns, key=lambda p: p.get("confidence") or 0.0)`.

So the historical bug “only first pattern per `target_id` retained” is **resolved in both** engines: multiple patterns per FM are stored; the **highest-confidence** pattern is selected for scoring.

---

## 4. Causality engine v31 (`RuleBasedCausalityEngineV31`)

### 4.1 Temporal scoring for failure modes (`_temporal_score_for_fm`)

- Loads **one** pattern via `_lookup_tskr_pattern(tskr_index, fm_id)`.  
- **`tskr_pattern_match` fallback:** if pattern confidence is 0 but anomalies exist → **`0.85`** (aggressive bump).  
- **`latency_consistency`:** computed from FM’s `expected_latency_min_hours` / `max_hours` and `inferred_delay_hours = mean_lag_hours` from the pattern, else **`1.0` hours** if anomalies exist else `None` → low consistency branch. This is **not** the same as reading `latency_alignment_score` off the TSKR pattern.  
- **No explicit `temporal_contradiction` penalty** inside `_temporal_score_for_fm` (unlike v32).  
- Returned `temporal_evidence` on the candidate **omits** `latency_violation_type`, `observed_lag_hours`, `expected_latency_*` from TSKR pattern fields (v32 fills these from the pattern).

### 4.2 Evidence prior (`_evidence_score_for_fm`)

v31 applies **recency** to all doc types (no “timeless” bucket). v32 introduces `_TIMELESS_DOC_TYPES` and reweights FMEA / OE / etc. This is a **major behavioral delta** unrelated to TSKR but affects **the same composite** alongside temporal.

### 4.3 Symptom and governance

v31 **does not** include v32’s symptom/governance refinements (v32 module is much larger). For “two engines” comparisons, document whether the run uses **matching** `top_k_candidates` and whether **symptom_match** and **governance** deltas are understood as confounders, not only TSKR.

### 4.4 **No `refine_with_evidence`**

`RuleBasedCausalityEngineV31` defines **no** `refine_with_evidence`. The orchestrator **skips** post-retrieval candidate refinement for v31. Evidence retrieval still runs and feeds the synthesizer, but **candidate scores and ordering are not re-done** from the evidence bundle inside the engine.

**Operational consequence:** The workflow spec’s “Stage D vs Stage F ranking delta” diagnostic **does not apply** to v31-only runs the same way it does to v32.

---

## 5. Causality engine v32 (`RuleBasedCausalityEngineV32`)

### 5.1 Temporal scoring for failure modes (`_temporal_score_for_fm`)

- Same pattern lookup (`fm_id`).  
- **`tskr_pattern_match` fallback** when zero but anomalies exist → **`0.55`** (less aggressive than v31’s **0.85**).  
- **`latency_consistency`** comes from **`_pattern_latency_alignment(pattern)`** → uses TSKR’s `latency_alignment_score` when present.  
- **`temporal_contradiction`** from pattern; applies **`temporal -= 0.25`** when true.  
- Weights in the temporal blend: **0.35 / 0.25 / 0.25 / 0.15** on `tskr_pattern_match`, `temporal_precedence`, `latency_consistency`, `support` (v31 used **0.35 / 0.30 / 0.20 / 0.15** and baked latency partly from FM fields).

### 5.2 `temporal_posture` and review-alternative logic

v32 computes **`temporal_posture`** (`supported` / `partial` / `weak` / `contradicted`) from temporal scores and contradiction flags, and uses temporal contradiction in **review-alternative rescue** gating — v31 candidate payloads are **lighter** on these fields.

### 5.3 Past events

v32’s `_temporal_score_for_past_event` enriches early-return branches with pattern fields (`latency_violation_type`, etc.) and uses `_pattern_latency_alignment` where v31 used flat constants in some branches.

### 5.4 `refine_with_evidence`

Present on v32; orchestrator calls it when the engine implements it. This is the **production** path for evidence-driven re-ranking described in the formal workflow doc.

---

## 6. Side-by-side summary (v31 vs v32, TSKR-adjacent and pipeline)

| Topic | v31 | v32 |
|--------|-----|-----|
| TSKR index / multi-pattern | List + max confidence | Same |
| FM `tskr_pattern_match` anomaly fallback | **0.85** | **0.55** |
| FM latency consistency | FM `expected_latency_*` + pattern lag / default 1h | TSKR `latency_alignment_score` + pattern fields |
| FM `temporal_contradiction` penalty in temporal score | No | Yes (−0.25) |
| `temporal_evidence` richness | Smaller | Includes violation, observed/expected lag from pattern |
| Document / evidence prior | Recency on all types | Timeless types (SOP, FMEA, OE, …), reweighted |
| Symptom / governance / common-cause / safety | Baseline | Extended |
| **`refine_with_evidence`** | **Absent** | **Present** |

---

## 7. Risks, pitfalls, and test implications

1. **Comparing v31 vs v32 on one run:** instantiate **two** orchestrators (or call `generate` twice with injected `tskr_patterns`) — the default pipeline does **not** produce both.  
2. **Manifest `causality_engine_version`:** must match manual wiring (§1.3).  
3. **v31 + “evidence delta” KPIs:** refinement flag stays false; do not interpret manifests as v32-style refinement.  
4. **TSKR confidence weights** sum > 1 before clamp — calibration documentation should state that explicitly.  
5. **Recurrence OR-by-component** in TSKR may inflate history for crowded components.  
6. **Global Allen relation per FM:** reviewers should not assume relation differs per FM without checking latency/recurrence slices.

**Tests:** `unit_tests/test_tskr_temporal_scorer.py` is the right home for scorer edge cases; engine-level differences between v31/v32 temporal fallback (0.85 vs 0.55) deserve **paired** unit tests if regression sensitivity is high.

---

## 8. References (code)

```257:311:c:\Users\mande\projects\DACKAR\src\dackar\RCA\orchestrators\rca_reasoning_orchestrator.py
        if tskr_patterns is None:
            if self.tskr_temporal_scorer is not None:
                tskr_patterns = self.tskr_temporal_scorer.score(
                    ...
                )
            else:
                tskr_patterns = {
                    ...
                    "summary": {
                        "has_temporal_support": False,
                        "mode": "absent",
                    },
                    ...
                }
        ...
        if causality_candidates is None:
            causality_candidates = self.causality_engine.generate(
                ...
                tskr_patterns=tskr_patterns,
                ...
            )
        ...
        if hasattr(self.causality_engine, "refine_with_evidence"):
            causality_candidates = self.causality_engine.refine_with_evidence(
                causality_candidates=causality_candidates,
                evidence_bundle=evidence_bundle,
            )
```

```2157:2176:c:\Users\mande\projects\DACKAR\src\dackar\RCA\orchestrators\causality_engine_v32.py
    def _index_tskr_patterns(self, tskr_patterns):
        index: Dict[str, List[JsonDict]] = {}
        ...
                index.setdefault(target_id, []).append(p)
        return index

    def _lookup_tskr_pattern(self, tskr_index, target_id):
        """Return the highest-confidence pattern for *target_id*, or None."""
        ...
        return max(patterns, key=lambda p: p.get("confidence") or 0.0)
```

```401:427:c:\Users\mande\projects\DACKAR\src\dackar\RCA\orchestrators\causality_engine_v31.py
    def _temporal_score_for_fm(self, fm, telemetry_summary, event_time, tskr_index):
        ...
        if tskr_pattern_match == 0.0 and anomaly_signals:
            tskr_pattern_match = 0.85
        ...
        latency_consistency = self._latency_consistency(
            min_h,
            max_h,
            inferred_delay_hours=mean_lag_hours if mean_lag_hours is not None else (1.0 if anomaly_signals else None),
        )
        temporal = min(
            1.0,
            0.35 * tskr_pattern_match
            + 0.30 * temporal_precedence
            + 0.20 * latency_consistency
            + 0.15 * support,
        )
```

```1028:1057:c:\Users\mande\projects\DACKAR\src\dackar\RCA\orchestrators\causality_engine_v32.py
    def _temporal_score_for_fm(self, fm, telemetry_summary, event_time, tskr_index):
        ...
        if tskr_pattern_match == 0.0 and anomaly_signals:
            tskr_pattern_match = 0.55
        ...
        temporal = min(
            1.0,
            0.35 * tskr_pattern_match
            + 0.25 * temporal_precedence
            + 0.25 * latency_consistency
            + 0.15 * support,
        )

        if temporal_contradiction:
            temporal = max(0.0, temporal - 0.25)
```

---

## 9. Changelog (this document)

| Date | Change |
|------|--------|
| 2026-04-20 | Initial second-pass technical note (TSKR + orchestrator + v31/v32). |
