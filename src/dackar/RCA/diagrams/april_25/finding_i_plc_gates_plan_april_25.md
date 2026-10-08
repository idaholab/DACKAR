# Finding I — Direct `protection_logic_context` Read in Hard Gates

**Date:** 2026-04-25  
**Status:** Planning  
**Author:** RCA workflow systems engineering session

---

## 1. Problem Statement

Both hard gates that involve physical protection evidence currently operate in **structural proxy mode**:

| Gate | Current input | Gap |
| --- | --- | --- |
| Physical plausibility | `scores["structural"]` < 0.20 threshold only | No awareness of actual protection logic signals |
| Barrier logic | KG-derived `affected_safety_functions` + `barrier_signal` score | Circular dependency on `ruleout.reason_code = "barrier_held"` pre-set; always passes in degraded mode |

`protection_logic_context` (PLC) carries exactly the missing information:

- `logic_sets[]` — trip, permissive, interlock, and actuation logic definitions with `input_signals`, `output_signals`, and setpoints.
- `barrier_states[]` — per-barrier direct state observation: `held` / `failed` / `degraded` / `unknown`, with `sf_id` cross-reference and `evidence_refs`.

When PLC is available, both gates should switch from proxy reasoning to **direct evidence reading**.

---

## 2. Current Logic (Both Gates)

### 2.1 Physical plausibility gate

```
FAIL if:  scores["structural"] < 0.20
PASS else (structural is the only check)
```

`degraded_mode` is not tracked. PLC is never consulted.

### 2.2 Barrier logic gate

```
has_barrier_inputs = bool(affected_safety_functions) or barrier_signal is not None
degraded = not has_barrier_inputs     ← always True when KG has no safety-function nodes

blocked_by_barrier_held = (ruleout.reason_code == "barrier_held")   ← never pre-set; always False

passed = not blocked_by_barrier_held   ← always True
```

The gate is effectively **always passing**. The circular dependency on `ruleout` was never wired to actually be set.

---

## 3. Design

### 3.1 New helper — `_build_plc_barrier_index`

```python
@staticmethod
def _build_plc_barrier_index(
    protection_logic_context: Optional[JsonDict],
) -> Tuple[Dict[str, str], Set[str]]:
    """
    Returns:
        sf_state_index   {sf_id → barrier_state}   from barrier_states[]
        logic_signal_ids set of component-id strings appearing in any
                         logic_set input_signals / output_signals
    """
```

### 3.2 Physical plausibility gate enhancement

**New check (in addition to structural < 0.20):**

When PLC is provided:
- Collect `logic_signal_ids` (all `input_signals` and `output_signals` across all `logic_sets`).
- If candidate's `component_id` appears in `logic_signal_ids` and the barrier for the associated sf_id is **`held`** → add note to rationale: "PLC: component monitored in trip/actuation logic; barrier held — protection system responded."
- This is a **positive confirmation**, not a failure. It does NOT fail the gate.
- Sets `hard_gates["physical_plausibility"]["plc_consulted"] = True` and `degraded_mode = False`.

Physical plausibility gate still only FAILs on structural < 0.20, but rationale is now enriched by PLC evidence when available.

### 3.3 Barrier logic gate — direct barrier-held check

**New failure condition:**

When PLC is provided and `sf_state_index` is non-empty:

1. Build `plc_sf_state = {sf_id → state}` from PLC `barrier_states`.
2. For each `sf` in candidate's `affected_safety_functions`, look up `sf["sf_id"]` in `plc_sf_state`.
3. If **`state == "held"`** AND **`barrier_signal >= 0.80`**:
   - Gate **FAILS**: barrier was held while the candidate is a high-barrier-signal hypothesis.
   - Rationale: "PLC confirms barrier {sf_id} held (state=held). Candidate barrier_signal={x:.2f} ≥ 0.80 indicates this failure mode is about a protection-critical function that demonstrably held."
   - `ruleout.reason_code = "barrier_held_by_plc"`
4. If barriers are `failed` or `degraded` for matched sf_ids:
   - Gate **PASSES**; evidence of barrier compromise is CONSISTENT with the candidate hypothesis.
   - Add note to rationale.
5. If PLC provided but no sf_id matches → `degraded_mode = False`; gate passes with "PLC consulted, no sf_id match."

**`barrier_signal >= 0.80` threshold rationale:** A `barrier_signal` of 1.0 is assigned only when the failure mode directly involves a critical barrier (`CRITICAL_BARRIER_KEYWORDS`). The 0.80 threshold ensures we only block candidates whose failure mechanism is fundamentally about a safety-critical function — not ordinary components that happen to be associated with safety functions.

**`degraded_mode` clearing:**
- If PLC is provided → `degraded_mode = False` (we have real data, not a proxy).
- New field `hard_gates["barrier_logic"]["plc_consulted"] = True/False`.

---

## 4. Data Flow

```
run()
  ├─ protection_logic_context (already accepted as parameter)
  │
  ├─ refine_kwargs["protection_logic_context"] = protection_logic_context   ← NEW
  │    (guarded by inspect.signature, backward-compatible)
  │
  └─ refine_with_evidence(protection_logic_context=...)                      ← NEW param
       │
       ├─ plc_sf_state, plc_logic_signal_ids = _build_plc_barrier_index(plc) ← NEW helper
       │
       └─ for each candidate:
            ├─ _apply_physical_plausibility_gate(candidate, plc_logic_signal_ids, plc_sf_state)
            ├─ _apply_timeline_consistency_gate(candidate)           [unchanged]
            └─ _apply_barrier_logic_gate(candidate, plc_sf_state)
```

---

## 5. Workstreams

### WS1 — New helper `_build_plc_barrier_index` (causality engine)

**File:** `causality_engine_v32.py`

```python
@staticmethod
def _build_plc_barrier_index(
    protection_logic_context: Optional[JsonDict],
) -> Tuple[Dict[str, str], Set[str]]:
    """
    Parse protection_logic_context into:
      sf_state_index    {sf_id → state}   (barrier_states array)
      logic_signal_ids  set of signal/component IDs from logic_set input/output signals
    Returns empty structures when PLC is None or malformed.
    """
```

Steps:
1. Guard `isinstance(protection_logic_context, dict)`.
2. Iterate `barrier_states[]`: `sf_state_index[bs["sf_id"]] = bs["state"]` (skip null sf_id).
3. Iterate `logic_sets[]`, flatten `input_signals` and `output_signals` into `logic_signal_ids`.
4. Return both.

---

### WS2 — Update `_apply_physical_plausibility_gate`

**Signature change:**
```python
def _apply_physical_plausibility_gate(
    self,
    candidate: JsonDict,
    plc_logic_signal_ids: Optional[Set[str]] = None,
    plc_sf_state: Optional[Dict[str, str]] = None,
) -> None:
```

**New logic block (after structural check):**
```
plc_consulted = False
if plc_logic_signal_ids:
    cid = candidate["component_id"]
    if cid in plc_logic_signal_ids:
        # Check if any affected sf has its barrier held
        barrier_held_sf = [
            sf["sf_id"] for sf in affected_safety_functions
            if plc_sf_state and plc_sf_state.get(sf["sf_id"]) == "held"
        ]
        plc_note = (
            f"PLC: component_id='{cid}' appears in trip/actuation logic signals. "
            + (f"Barrier held for sf_ids={barrier_held_sf} — protection responded." if barrier_held_sf else "")
        )
        plc_consulted = True

hard_gates["physical_plausibility"]["plc_consulted"] = plc_consulted
hard_gates["physical_plausibility"]["degraded_mode"] = not (bool(structural_raw) or plc_consulted)
```

---

### WS3 — Update `_apply_barrier_logic_gate`

**Signature change:**
```python
def _apply_barrier_logic_gate(
    self,
    candidate: JsonDict,
    plc_sf_state: Optional[Dict[str, str]] = None,
) -> None:
```

**Replace current logic:**

```python
affected_safety_functions = candidate.get("affected_safety_functions") or []
barrier_signal_raw = scores.get("barrier_signal")
barrier_signal = float(barrier_signal_raw) if isinstance(barrier_signal_raw, (int, float)) else 0.0
PLC_BARRIER_HELD_THRESHOLD = 0.80

# Step 1 — PLC direct check (when available)
plc_consulted = False
plc_held_sf_ids = []
plc_failed_sf_ids = []
plc_matched = False

if plc_sf_state:
    for sf in affected_safety_functions:
        sf_id = sf.get("sf_id")
        if sf_id and sf_id in plc_sf_state:
            plc_matched = True
            plc_consulted = True
            state = plc_sf_state[sf_id]
            if state == "held":
                plc_held_sf_ids.append(sf_id)
            elif state in {"failed", "degraded"}:
                plc_failed_sf_ids.append(sf_id)

# Step 2 — Determine degraded_mode
has_barrier_inputs = bool(affected_safety_functions) or barrier_signal > 0
degraded = (not has_barrier_inputs) and (not plc_consulted)

# Step 3 — Gate decision
blocked_by_plc_barrier_held = (
    bool(plc_held_sf_ids)
    and barrier_signal >= PLC_BARRIER_HELD_THRESHOLD
)
# Fallback: legacy ruleout path (no change)
blocked_by_legacy = (
    isinstance(candidate.get("ruleout"), dict)
    and str((candidate.get("ruleout") or {}).get("reason_code") or "") == "barrier_held"
)
passed = not (blocked_by_plc_barrier_held or blocked_by_legacy)

# Step 4 — Rationale
if blocked_by_plc_barrier_held:
    rationale = (
        f"FAIL (PLC): barrier held confirmed for sf_ids={plc_held_sf_ids}; "
        f"barrier_signal={barrier_signal:.3f} ≥ {PLC_BARRIER_HELD_THRESHOLD}. "
        "Protection system demonstrably held — candidate hypothesis contradicted."
    )
    ruleout["reason_code"] = "barrier_held_by_plc"
elif plc_failed_sf_ids:
    rationale = (
        f"PASS (PLC): barrier compromised for sf_ids={plc_failed_sf_ids} — "
        "consistent with candidate hypothesis."
    )
elif plc_consulted and not plc_matched:
    rationale = "PASS (PLC consulted): no matching sf_id in PLC barrier_states."
# ... existing rationale for non-PLC paths

hard_gates["barrier_logic"]["plc_consulted"] = plc_consulted
hard_gates["barrier_logic"]["degraded_mode"] = bool(degraded)
```

If failed: `ruleout.reason_code = "barrier_held_by_plc"`.

---

### WS4 — Update `refine_with_evidence`

**New parameter:**
```python
def refine_with_evidence(
    self,
    ...
    allen_relation_map: Optional[JsonDict] = None,
    protection_logic_context: Optional[JsonDict] = None,   # NEW
) -> JsonDict:
```

**Build PLC index at top of method:**
```python
plc_sf_state, plc_logic_signal_ids = self._build_plc_barrier_index(protection_logic_context)
```

**Pass to gates (post-loop section):**
```python
for candidate in candidates:
    self._apply_physical_plausibility_gate(candidate, plc_logic_signal_ids, plc_sf_state)
    self._apply_timeline_consistency_gate(candidate)
    self._apply_barrier_logic_gate(candidate, plc_sf_state)
```

---

### WS5 — Thread through orchestrator

**File:** `rca_reasoning_orchestrator.py`

In the `inspect.signature` guard block:
```python
if accepts_var_kw or "protection_logic_context" in sig.parameters:
    refine_kwargs["protection_logic_context"] = protection_logic_context
```

---

### WS6 — Tests (`test_finding_i_plc_gates.py`, ~20 tests)

| Test | What it checks |
| --- | --- |
| `test_plc_barrier_index_none_returns_empty` | `None` → `({}, set())` |
| `test_plc_barrier_index_builds_sf_state` | `barrier_states` → correct `{sf_id → state}` |
| `test_plc_barrier_index_null_sf_id_skipped` | null sf_id entries skipped |
| `test_plc_barrier_index_logic_signals_extracted` | `input_signals` + `output_signals` flattened |
| `test_pp_gate_plc_consulted_flag_set` | `hard_gates["physical_plausibility"]["plc_consulted"]` when component in logic signals |
| `test_pp_gate_plc_not_consulted_when_plc_none` | flag False when no PLC |
| `test_pp_gate_plc_component_not_in_signals_no_flag` | component absent from signals → not consulted |
| `test_barrier_gate_plc_held_high_signal_fails` | `barrier_signal=1.0` + `state=held` → FAIL |
| `test_barrier_gate_plc_held_low_signal_passes` | `barrier_signal=0.5` + `state=held` → PASS (below threshold) |
| `test_barrier_gate_plc_failed_barrier_passes` | `state=failed` → PASS with evidence note |
| `test_barrier_gate_plc_degraded_barrier_passes` | `state=degraded` → PASS with evidence note |
| `test_barrier_gate_plc_no_sf_match_passes` | PLC provided but no sf_id overlap → PASS |
| `test_barrier_gate_plc_consulted_clears_degraded_mode` | `degraded_mode=False` when PLC provided |
| `test_barrier_gate_no_plc_remains_degraded` | `degraded_mode=True` when PLC absent and no KG safety functions |
| `test_barrier_gate_ruleout_reason_code_barrier_held_by_plc` | `ruleout.reason_code = "barrier_held_by_plc"` on PLC-fail |
| `test_barrier_gate_plc_consulted_flag` | `hard_gates["barrier_logic"]["plc_consulted"]` set |
| `test_refine_with_evidence_threads_plc` | full `refine_with_evidence` call with PLC arg |
| `test_orchestrator_refine_kwargs_includes_plc` | orchestrator passes PLC to refine when param accepted |
| `test_barrier_gate_plc_held_multiple_sfs_one_matches` | only one sf in plc, others absent → only matched one counts |
| `test_backward_compat_no_plc_param` | old callers without PLC param unchanged |

---

### WS7 — Documentation

- Update `rca_workflow_development_backlog_april_25.md` with Finding I section, DoD, and Readiness Matrix.
- Update `rca_metamodel.md` — Phase 5 Finding I entry in Execution Sequencing.

---

## 6. Schema Changes

None. `protection_logic_context.json` already defines `barrier_states[].sf_id`, `barrier_states[].state`, `logic_sets[].input_signals`, `logic_sets[].output_signals`.

`hard_gates` entries gain two new fields (`plc_consulted`, `degraded_mode` is already present for timeline gate):
- `hard_gates["physical_plausibility"]["plc_consulted"]`: boolean
- `hard_gates["physical_plausibility"]["degraded_mode"]`: boolean (new for this gate)
- `hard_gates["barrier_logic"]["plc_consulted"]`: boolean (new)

These are additive to the existing hard_gates structure.

---

## 7. Definition of Done

| # | Criterion | Target |
| --- | --- | --- |
| 1 | `_build_plc_barrier_index` static helper implemented | Yes |
| 2 | `_apply_physical_plausibility_gate` accepts PLC signals; `plc_consulted` flag set | Yes |
| 3 | `_apply_barrier_logic_gate` uses `plc_sf_state`; FAIL when `barrier_signal ≥ 0.80` AND `state=held` | Yes |
| 4 | Barrier gate `degraded_mode = False` when PLC is provided | Yes |
| 5 | `ruleout.reason_code = "barrier_held_by_plc"` on PLC-based fail | Yes |
| 6 | `refine_with_evidence` accepts `protection_logic_context` | Yes |
| 7 | Orchestrator threads PLC through `refine_kwargs` | Yes |
| 8 | 20 targeted tests pass | Yes |
| 9 | Full suite 958+ tests pass, zero regressions | Yes |
| 10 | `None` PLC is fully backward-compatible | Yes |

---

## 8. Step Readiness Matrix (pre-implementation)

| Dimension | Before | After |
| --- | --- | --- |
| Physical plausibility gate inputs | `structural` score only | `structural` + PLC logic signal presence |
| Barrier logic gate inputs | KG safety-function proxy only | KG proxy + PLC `barrier_states` direct read |
| Barrier gate degraded mode | Always True when KG has no safety functions | False when PLC provided |
| Barrier gate failure trigger | Circular `ruleout` pre-set (never fires) | Direct PLC `held` + `barrier_signal ≥ 0.80` |
| PLC data utilisation | None (coverage report only) | Hard-gate decisions |
| Backward compatibility | N/A | `Optional[JsonDict] = None`; no change when absent |
