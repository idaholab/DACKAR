# Finding H — Category E Operating Point in Scoring Plan
**Date:** 2026-04-25  
**Status:** Planning  
**Audit origin:** Phase 5 audit, row H: "Category E `operating_point` not in scoring — Declared limitation: noted in metamodel"

---

## 1. Problem Statement

Category E covers **operating envelope / process-state anomalies**: overload, off-design transient, cycling, standby runout, and power-level-dependent degradation.  Every `operational_context` carries:

| Field | Schema type | What it encodes |
|---|---|---|
| `percent_rated_power` | number 0–100 | Thermal-hydraulic stress level |
| `mode` | enum | `startup`, `power_ramp`, `steady`, `power_ramp_down`, `shutdown`, `post_maintenance_test`, `maintenance`, `unknown` |
| `train_configuration.in_service` | boolean | Whether the affected train was in service |

None of these fields currently contribute to any candidate's score. A Category E candidate (e.g. "pump cavitation during power ramp") gets the same `structural` score at 20% power as at 100% power, and the same score on `startup` mode as `steady` mode. This is a material plausibility gap.

### What "scoring blind" means concretely

- At `percent_rated_power ≥ 90` and `mode = power_ramp`, transient-induced failures (Category E keywords: `overload`, `cycling`, `off-design`, `runout`) are more plausible → should score higher.
- At `mode = shutdown` or `mode = maintenance`, those same failure modes are less plausible unless specifically initiated by the shutdown sequence.
- At `mode = startup`, equipment that has been in standby is at elevated risk of runout / self-heat / seal stagnation → `standby` keyword candidates should score higher.
- When `train_configuration.in_service = False`, standby-degradation mechanisms on that train are highly plausible → should boost relevant candidates.

---

## 2. Current Architecture

The `structural` score for a failure-mode candidate is computed in `_build_failure_mode_candidates`:

```
structural = topology
           + 0.40 × (symptom_score – 0.5)   [-0.20, +0.20]
           + 0.15 × alarm_signal             [0.00, +0.15]
           + 0.08 × rpn_prior                [0.00, +0.08]
           + 0.10 × barrier_signal           [0.00, +0.10]
```

`operational_context` is passed into `_build_failure_mode_candidates` but is only used for alarm corroboration (`_alarm_signal_for_candidate`).  The `operating_point` block (`percent_rated_power`, `mode`) is never consulted.

---

## 3. Design

### 3.1 New static helper: `_operating_point_score`

```python
@staticmethod
def _operating_point_score(
    *,
    operational_context: Optional[JsonDict],
    primary_causal_category: str,
    fm_superclass: Optional[str],
    fm_name: Optional[str],
) -> Tuple[float, str]:
    """Return (score 0–1, rationale_note) for the operating-point dimension.

    Returns (0.0, "not_assessed") when operational_context is None or
    percent_rated_power is absent — never penalises candidates for missing data.
    """
```

#### Score logic

The helper produces a raw signal in [0.0, 1.0].

**Step 1 — Mode-based base contribution (applies to all categories)**

| `mode` | Base |
|---|---|
| `power_ramp` | 0.70 |
| `startup` | 0.60 |
| `power_ramp_down` | 0.50 |
| `steady` | 0.30 |
| `post_maintenance_test` | 0.40 |
| `maintenance` | 0.35 |
| `shutdown` | 0.20 |
| `unknown` / absent | 0.0 (not assessed) |

**Step 2 — Power-level modifier (Category E candidates only)**

- Only applied when `primary_causal_category == "E"` and `percent_rated_power` is present.
- Normalise power level: `p_norm = percent_rated_power / 100.0`.
- If category is E and fm keyword matches `overload | off-design | runout | cycling | transient`: power_modifier = `p_norm × 0.30` (high power → up to +0.30 boost).
- If category is E and fm keyword matches `standby | stagnation | idle`: power_modifier = `(1.0 – p_norm) × 0.25` (low power / standby → up to +0.25 boost).

**Step 3 — Train OOS contribution**

- If `train_configuration.in_service == False` and fm keyword matches `standby | idle | stagnation | cycling`: train_bonus = `0.15`.
- Else: `0.0`.

**Final score** = `min(1.0, mode_base + power_modifier + train_bonus)`.

**No-data default**: if `operational_context` is None or `mode` is absent → return `(0.0, "not_assessed")`.  This is critical: missing operational context must never penalise candidates.

### 3.2 Integration into `_build_failure_mode_candidates`

The `operating_point_score` is added as a new named sub-score field — it does **not** replace an existing weight dimension. Instead it acts as an additive delta on `structural`, capped so the total still obeys [0, 1]:

```python
op_score, op_note = self._operating_point_score(
    operational_context=operational_context,
    primary_causal_category=primary_causal_category,
    fm_superclass=fm.get("superclass"),
    fm_name=fm.get("name"),
)
OP_DELTA_CAP = 0.12   # max structural contribution from operating point
op_delta = OP_DELTA_CAP * op_score   # [0, +0.12]
structural = max(0.0, min(1.0, topology + symptom_delta + alarm_delta + rpn_delta + barrier_delta + op_delta))
```

The `scores` dict is extended:
```python
"operating_point_score": round(op_score, 6),
"operating_point_note": op_note,
```

### 3.3 Score rationale update

`_update_score_rationale_for_refinement` already builds `score_rationale["structural"]`.  After generation, `_build_score_rationale` (initial rationale builder) will append the operating-point note when `operating_point_score > 0`.

### 3.4 Category coverage applicability update

`_assess_category_applicability` currently marks Category E as `"applicable"` when `has_ops` is True (operating_point present OR alarms present).  No change needed — the gate already fires correctly.  But the rationale string can be enriched to mention `percent_rated_power`.

### 3.5 Thread through `generate()`

`_build_failure_mode_candidates` already accepts `operational_context`. The only required change is calling `_operating_point_score` inside that method using the already-available `primary_causal_category` (inferred just before scores dict assembly).

Wait — there is a sequencing issue: `primary_causal_category` is inferred **after** `structural` is computed (line 417, after line 374).  

**Fix:** infer category first (it depends only on `fm` and `event`, not on scores), then compute `op_score`, then assemble `structural`.

This reorder is safe — `_infer_primary_category_for_failure_mode` has no side-effects and depends only on fm text + event type.

---

## 4. Workstreams

| WS | Deliverable |
|---|---|
| **WS1** | New `_operating_point_score` static helper in `causality_engine_v32.py` |
| **WS2** | Reorder category inference before structural assembly in `_build_failure_mode_candidates`; integrate `op_delta` into `structural`; add `operating_point_score` / `operating_point_note` to `scores` dict |
| **WS3** | Extend `_build_score_rationale` / `_update_score_rationale_for_refinement` to include operating-point note in `score_rationale["structural"]` |
| **WS4** | Tests — `test_finding_h_operating_point.py` (~18 tests) |
| **WS5** | Docs — backlog + metamodel update |

---

## 5. Design Decisions

### D1 — Additive delta on `structural`, not a separate weight dimension
**Rationale:** Adding a new weight key (e.g. `"operating_point": 0.10`) would require rebalancing all five weights (structural 0.30, temporal 0.20, telemetry 0.20, evidence 0.20, governance 0.10) and would break every existing test that sums to 1.0.  An additive delta on `structural` (capped at 0.12) achieves the same plausibility signal while leaving the weighting contract intact and preserving backward compatibility.

### D2 — Delta cap 0.12
**Rationale:** The operating point should be a plausibility modifier, not a dominant signal. 0.12 is smaller than the alarm delta cap (0.15) and the symptom delta cap (0.20), reflecting the hierarchy: measured signals > observed alarms > state-inferred context.

### D3 — No-data returns 0.0, not penalty
**Rationale:** Many existing tests do not supply `operational_context`. Returning `0.0` for missing data means no existing test is affected.

### D4 — Power modifier only for Category E candidates
**Rationale:** Applying `percent_rated_power` to Category A (equipment internal) or Category G (human performance) candidates would introduce spurious correlations. The power level is physically meaningful only for process-envelope (E) candidates.

### D5 — Category inference reordered before structural assembly
**Rationale:** The current code infers `primary_causal_category` after computing `structural`. Since `_operating_point_score` needs the category, the reorder is necessary. The inference is a pure function of `fm` text + `event.event_type`, so it has no scoring side-effects. No other code in the method depends on the old ordering.

### D6 — Train OOS bonus applies only to standby-mechanism keywords
**Rationale:** If the train was OOS, only standby-degradation mechanisms (stagnation, idle, cycling) are directly enabled by that state. Applying the bonus to all Category E candidates on an OOS train would over-boost unrelated process anomalies.

---

## 6. Test Plan (`test_finding_h_operating_point.py` — target: 18 tests)

### `_operating_point_score` unit tests (8 tests)
- `test_no_operational_context_returns_zero` — None → (0.0, "not_assessed").
- `test_power_ramp_mode_high_base` — `mode=power_ramp` → mode_base = 0.70.
- `test_steady_mode_lower_base` — `mode=steady` → mode_base = 0.30.
- `test_shutdown_mode_lowest_base` — `mode=shutdown` → mode_base = 0.20.
- `test_category_e_overload_high_power_boost` — Cat. E, `overload` keyword, `percent_rated_power=95` → op_score raised by power_modifier.
- `test_category_e_standby_low_power_boost` — Cat. E, `standby` keyword, `percent_rated_power=5` → boosted by (1-p_norm).
- `test_non_category_e_no_power_modifier` — Cat. A candidate at 95% power → no power_modifier.
- `test_train_oos_standby_mechanism_bonus` — `in_service=False`, `standby` keyword → +0.15 bonus.

### Integration — `_build_failure_mode_candidates` (6 tests)
- `test_category_e_candidate_higher_structural_at_high_power` — same fm, high power vs low power → higher `structural` at high power.
- `test_category_e_standby_scores_higher_in_startup_mode` — `startup` mode, `standby` fm → structural boosted vs `steady` mode.
- `test_non_category_e_structural_unchanged_by_operating_point` — Cat. A, high power → `operating_point_score == 0.0`.
- `test_missing_operational_context_no_regression` — `operational_context=None` → no change to structural vs existing baseline.
- `test_operating_point_score_field_stored_in_scores` — `scores["operating_point_score"]` present on candidate.
- `test_structural_capped_at_one` — extreme inputs do not push structural > 1.0.

### Score rationale (2 tests)
- `test_score_rationale_includes_operating_point_note` — when op_score > 0, rationale["structural"] contains operating-point mention.
- `test_score_rationale_absent_when_not_assessed` — None operational_context → no operating-point note in rationale.

### Backward-compatibility regression (2 tests)
- `test_existing_candidates_unaffected_without_operational_context` — run `generate()` without `operational_context`; verify scores are identical to current baseline.
- `test_composite_score_still_in_range` — with operating point inputs, all composite scores remain in [0, 1].

---

## 7. Definition of Done

| # | Criterion |
|---|---|
| 1 | `_operating_point_score` returns `(0.0, "not_assessed")` when `operational_context` is None |
| 2 | Mode-based base and power-level modifier applied to Category E candidates |
| 3 | Train OOS bonus applied to standby-mechanism keywords |
| 4 | `op_delta` capped at 0.12; `structural` remains in [0, 1] |
| 5 | `scores["operating_point_score"]` and `scores["operating_point_note"]` present on every candidate |
| 6 | Category inference reordered before structural assembly (no functional regression) |
| 7 | `score_rationale["structural"]` mentions operating point when `op_score > 0` |
| 8 | No existing test broken (backward-compatible zero delta when no operational context) |
| 9 | 18 tests in `test_finding_h_operating_point.py` pass |
| 10 | Full suite ≥ 1003 tests pass, zero regressions |

---

## 8. Step Readiness Matrix (Target State Post-WS5)

| Dimension | Before | After |
|---|---|---|
| Category E scoring source | KG structural proxy only | KG structural + operating-point mode + power-level modifier + train OOS |
| `percent_rated_power` consumption | Never read by scoring | Power modifier for Cat. E overload/standby mechanisms |
| `mode` consumption | Never read by scoring | Mode-based base contribution for all Cat. E candidates |
| `train_configuration.in_service` | Only used by CCF scoring (common-cause) | Also feeds Cat. E standby bonus |
| Score traceability | No operating-point field | `operating_point_score`, `operating_point_note` in `scores` dict |
| Phase 5 audit row H | "Declared limitation" | Closed |
