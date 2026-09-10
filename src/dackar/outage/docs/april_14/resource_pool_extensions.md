# Resource Pool Extensions — Design Notes
*Date: 2026-04-15*

## Background

The LOGOS RCPSP scheduler currently models three resource pool types, all sharing the same
capacity-over-time contract: a finite quantity available at any hour, consumed during
`[start, end)`, replenished when the activity finishes.

| Pool | Unit | Replenishes? | Current status |
|------|------|-------------|----------------|
| `ResourcePool` | workers (by skill) | Yes | Implemented |
| `EquipmentPool` | units (by equipment id) | Yes | Implemented |
| `LocationPool` | task-slots + worker-density | Yes | Implemented |

Radiation dose (`dose_trackers`) is already present as a special case: it is **irrevocable**
and never replenishes. It was added as an ad-hoc mechanism rather than a first-class pool.

---

## Candidate Pool Types

### 1. ConsumablePool — Highest Priority (addressed)

**What it models:** Inventory items that are permanently depleted when an activity starts —
anti-contamination (AC) suits, filter media, resin bags, gaskets, bolts, sampling kits.
Unlike equipment, consumables are not returned after use.

**Scheduling contract:** `deduct-on-start` — the inventory is reduced by the activity's
declared consumption when the activity is admitted to the schedule. Feasibility is checked
at the candidate start time: "is there sufficient inventory left?". There is no
`restore-on-end` phase.

**Relationship to dose:** `dose_trackers` is a per-skill consumable. `ConsumablePool` would
generalise that pattern into a first-class, named-item inventory, making dose a special case
rather than a hardcoded mechanism.

**Scheduling impact:** Changes the feasibility check in `_fits_with_tentative` and the
commitment step in `_apply_tentative` / `_update_activity_sets`. No capacity-over-time grid
needed — a single scalar per item suffices.

**Activity schema extension:** Activities declare consumption as a list of
`{item_id, quantity}` dicts, analogous to `required_resources`.

---

### 2. PermitPool — High Priority (addressed)

**What it models:** Radiation Work Permits (RWPs), Confined Space Entry permits, and
Operations Work Orders that have a hard limit on the number simultaneously open within a
work zone. This is a first-class operational constraint at nuclear plants.

**Scheduling contract:** Same `occupy-while-active` contract as `LocationPool`, but scoped
to a permit zone rather than a physical room. Structurally this is a task-slot–only variant
of `LocationPool` (no worker-density dimension needed).

**Implementation path:** Could be a thin subclass of `LocationPool` with
`max_workers = None` (unbounded) and only `max_tasks` enforced. Alternatively, a standalone
`PermitPool` class for clarity.

**Activity schema extension:** Activities declare the `permit_zone_id` they require, similar
to `location_id`.

---

### 3. SystemStatePool — Medium Priority (addressed)

**What it models:** Shared plant systems that must be in a specific isolation state for a
task to proceed. Two activities that require the same system to be in *different* states
cannot run concurrently (e.g., valve A must be closed for task X and open for task Y).

**Scheduling contract:** Fundamentally different — this is **mutual exclusion by state**,
not a numeric capacity limit. It cannot be expressed with the current pool model. It requires
a new abstraction: activities declare `{system_id, required_state}` pairs, and the scheduler
checks that no concurrently running activity claims a conflicting state on the same system.

**Implementation path:** Requires a new constraint-checking hook in `_fits_with_tentative`
and a new `SystemStatePool` class (a dict of `system_id → current_state`). The conflict
check is: for each system the candidate requires, is any ongoing activity already holding
that system in a different state?

**Note:** This cannot be absorbed into the existing numeric-capacity framework; it is a
separate constraint contract.

---

### 4. Equipment Zone-Affinity — Low Priority (addressed)

**What it models:** Temporary power drops, compressed air manifolds, cooling water hookups,
and any piece of equipment that is physically installed in, or dedicated to, a specific
plant zone. A zone-assigned piece of equipment may only be used by activities that are
operating in that same zone.

**Assessment:** Two orthogonal constraints apply to utility connections:

* **Count** — how many ports/units are physically available → `EquipmentPool` quantity
* **Zone-affinity** — whether the equipment is restricted to a specific location zone

The count constraint was already covered.  Zone-affinity (Option B) was added as a
lightweight extension to `EquipmentPool`: an optional `zone_id` field on each equipment
entry.  No new pool class is needed.

**Scheduling contract:** Static feasibility check in `_fits_with_tentative`:
if an equipment item has a `zone_id`, the activity's `zone_ids` list must contain it.
Two backward-compat guards prevent regressions:
- `zone_id is None` → equipment is unconstrained (unzoned equipment never blocks)
- `act_zones` empty → activity has no declared zone, skip the affinity check

**Activity schema:** No change — activities already declare `zone_ids` (Option C).

**Implementation path:** Single field `zone_id` on `EquipmentAvailability`; one
`get_zone_id()` helper on `EquipmentPool`; thirteen-line check in
`_fits_with_tentative` after the equipment-count block.

---

## Implementation Priority

| Priority | Pool type | Key design decision |
|----------|-----------|---------------------|
| 1 | `ConsumablePool` | `deduct-on-start` contract; generalises `dose_trackers` |
| 2 | `PermitPool` | `occupy-while-active`; subclass or sibling of `LocationPool` |
| 3 | `SystemStatePool` | Mutual-exclusion contract; new constraint type |
| 4 | Equipment zone-affinity | `zone_id` field on `EquipmentAvailability`; static check in `_fits_with_tentative` |

---

## Implementation Status

| Pool type | Status | Date | Notes |
|-----------|--------|------|-------|
| `ConsumablePool` | **Implemented** | 2026-04-14 | `deduct-on-start`; mid-outage restocking; 48 tests |
| PermitPool (Option C) | **Implemented** | 2026-04-15 | Merged into `LocationPool` — see below |
| `SystemStatePool` | **Implemented** | 2026-04-15 | Shared-state lock with refcounting; 54 tests |
| Equipment zone-affinity | **Implemented** | 2026-04-15 | `zone_id` field on `EquipmentAvailability`; 28 tests |

---

## PermitPool — Decision Record (Option C, 2026-04-15)

### Decision

After evaluating four implementation paths, **Option C** was chosen: unify physical rooms
and permit zones into a single `LocationPool` by extending each entry with a `zone_type`
field (`'physical'` | `'permit'`), and allowing activities to declare a list of zone IDs
(`zone_ids`) rather than a single `location_id`.

### Rationale

| Criterion | Option C advantage |
|---|---|
| Single source of truth | All capacity constraints live in one pool |
| Backward compatibility | `getZoneIds()` falls back to `[location_id]` — no existing tests break |
| Expressive | A single activity can occupy multiple zones simultaneously (e.g. a physical room AND a permit zone) — correctly modelling real plant operations |
| Low migration cost | No new class; `zone_type` is optional and defaults to `'physical'` |

### What was implemented

**`outage_data.py` — `LocationAvailability`**

- New `zone_type: str = 'physical'` constructor parameter stored as `self.zone_type`.
- `__repr__` updated to include zone_type.
- `LocationPool.from_json` parses `zone_type = loc_data.get('zone_type', 'physical')` and
  passes it to the constructor.
- New `LocationPool.get_zone_type(location_id) -> str` helper (returns `'physical'` for
  unknown IDs).

**`activity.py`**

- New `self.zone_ids: list = []` field (after `self.location_id`).
- New `getZoneIds() -> list` method: returns `self.zone_ids` when set, otherwise
  `[self.location_id]` (or `[]` when both are absent).
- `from_json`: parses `zone_ids = list(task_dict.get('zone_ids', []))`.
- `to_json_dict`: emits `zone_ids` only when non-empty.

**`pert.py` — 12 call sites updated**

All `getLocation()` / single `loc_id` checks replaced with `getZoneIds()` loops:

| Method | Change |
|--------|--------|
| `calculate_greatest_resource_demand` | Uses `len(zone_ids)` for GRD contribution |
| `calculate_resource_requirement` | Loops over `zone_ids` when setting `rr[zone_id] = 1.0` |
| `_build_capacity_snapshots` | Decrements `loc_tasks_rem` / `loc_workers_rem` for every zone |
| `_fits_with_tentative` | Checks all zones; any zone over-capacity → infeasible |
| `_apply_tentative` | Decrements all zones on commit |
| `explain_delay` (diagnostic) | Iterates `act_zone_ids` for capacity diagnostics |
| `_get_tasks_at_location` | Uses `location_id in act.getZoneIds()` |
| `_get_workers_at_location` | Uses `location_id in act.getZoneIds()` |
| Binding arcs | Uses `loc_id in a.getZoneIds()` |
| `_serial_check_feasibility` | Loops over `activity.getZoneIds()` |
| `_compute_resource_consumption` | Emits `LOC_TASKS_` / `LOC_WORKERS_` keys for each zone |

**`outage_schema.json`**

- `zone_ids` added to task `properties` (array of strings, optional).
- `zone_type` added to location `properties` (enum `["physical", "permit"]`,
  default `"physical"`).

**`unit_tests/test_zone_ids.py`** — 34 new tests across 6 test classes:

| Class | What it covers |
|-------|---------------|
| `TestGetZoneIds` | `getZoneIds()` unit — backward compat, fallback, precedence, copy safety |
| `TestZoneIdsRoundtrip` | `from_json` / `to_json_dict` round-trip |
| `TestLocationAvailabilityZoneType` | zone_type field, default, repr |
| `TestLocationPoolZoneType` | `from_json`, `get_zone_type()`, mixed pools |
| `TestSchedulerZoneIds` | Full-scheduler integration: backward compat, single zone, multi-zone permit, multi-zone room, zone isolation, worker density |
| `TestZoneIdsCapacitySnapshots` | `_build_capacity_snapshots` multi-zone decrement; `_fits_with_tentative` blocked / unblocked |
| `TestZoneIdsSchema` | JSON schema fields present and correctly typed |

Total test count after Option C: **530** (496 pre-implementation + 34 new).

---

## SystemStatePool — Decision Record (2026-04-15)

### What it models

Shared plant systems (valves, pumps, circuit breakers, temporary power drops,
compressed air manifolds) that must be in a specific **isolation state** for an
activity to proceed.  Two activities requiring *different* states on the same
system cannot run concurrently.  Activities requiring the *same* state can coexist.

### Contract: shared-state lock

This is fundamentally different from all other pools — it is not a numeric
capacity limit but a **state-based mutual-exclusion lock**:

| Situation | Result |
|---|---|
| System free (no holder) | `fits()` → True for any state |
| System held in state S, candidate requests S | `fits()` → True (shared lock) |
| System held in state S, candidate requests T ≠ S | `fits()` → False (blocked) |

Reference counting ensures that the lock is released only when the *last*
holder completes.

### Relationship to EquipmentPool and utility connections

For utility connections (power drops, compressed air manifolds) two orthogonal
constraints apply:

* **Count** — how many ports / units are physically available → `EquipmentPool`
* **Isolation state** — whether the connection must be ENERGIZED, DE-ENERGIZED,
  PRESSURIZED, DRAINED, etc. → `SystemStatePool`

A task that uses a power drop typically needs both: a port (EquipmentPool count)
**and** the correct isolation state (SystemStatePool).  `system_id` can therefore
refer to any physical entity with isolation states — valves, breakers, and utility
connections alike.  This means `SystemStatePool` closes the gap that `EquipmentPool`
alone could not cover for utility connection isolation scenarios.

### What was implemented

**`outage_data.py` — `SystemStatePool`**

- `__init__`: `systems` dict (id → metadata) + `_held` dict (id → {state: refcount})
- `from_json(plant_systems_list)`: parses `system_id`, `description`, `valid_states`
- `has_system(system_id)`, `get_all_system_ids()`, `get_held_state(system_id)`
- `fits(system_id, required_state)` — permissive for unknown systems
- `acquire(system_id, state)` — increment refcount
- `release(system_id, state)` — decrement refcount, remove when zero
- `reset()` — clears `_held` entirely
- Wired into `OutageData.__init__` (optional param), `from_dict` (parses
  `data.get('plant_systems', [])`), `validate_data_consistency` (checks system_id
  references and valid_states), `print_summary`, `__repr__`

**`activity.py`**

- `self.required_system_states: list = []` field (structural; not cleared by `reset()`)
- `getRequiredSystemStates()` getter
- `from_json`: parses `required_system_states`
- `to_json_dict`: emits `required_system_states` when non-empty
- `set_mode`: optional per-mode override of `required_system_states`

**`pert.py` — 5 touch points**

| Where | What |
|---|---|
| `__init__` | `self.system_state_pool = getattr(outage_data, 'system_state_pool', None)` |
| `_reset_scheduling_state` | `system_state_pool.reset()` |
| `_partial_reset` | `reset()` then re-`acquire()` for in-progress activities |
| `_fits_with_tentative` | `fits()` check for each required state (after consumable check) |
| `_apply_tentative` | `acquire()` for each required state (tentative lock during SGS selection pass) |
| `_update_ongoing_list` | `release()` for each completing activity |

The key design insight: `acquire()` is called in `_apply_tentative` (not
`_update_activity_sets`) so that later candidates at the **same time-step** see
the tentative lock and are correctly blocked.  This is stronger than the
ConsumablePool pattern (which tolerates slight over-consumption within one
time-step) because conflicting isolation states are physically impossible
regardless of timing.

**`outage_schema.json`**

- `required_system_states` added to task `properties` (array of
  `{system_id, required_state}` objects).
- Top-level `plant_systems` array added (after `consumables`).

**`unit_tests/test_system_state_pool.py`** — 54 new tests across 7 test classes:

| Class | Coverage |
|---|---|
| `TestSystemStatePoolInit` | `from_json`, `has_system`, `repr`, valid_states |
| `TestSystemStatePoolFitsAcquireRelease` | `fits/acquire/release` semantics, refcounting, independent systems |
| `TestSystemStatePoolReset` | `reset()` clears all locks |
| `TestActivitySystemStates` | field defaults, `from_json`, `to_json_dict`, roundtrip, `set_mode` |
| `TestSchedulerSystemState` | same-state parallel; different-state serialised; unrelated activity unblocked; three-way exclusion; lock released after first finishes; multi-system requirement; locks zero after schedule |
| `TestReplanSystemState` | `_partial_reset` re-acquires for in-progress; skips completed; clears stale locks |
| `TestOutageDataSystemState` | `from_dict` pool construction; `validate_data_consistency` unknown system / invalid state / valid pass |
| `TestSchemaSystemState` | JSON schema fields present and correctly typed |

Total test count after SystemStatePool: **584** (530 pre-implementation + 54 new).

---

## Equipment Zone-Affinity — Decision Record (Option B, 2026-04-15)

### What it models

Equipment items that are physically installed in, or dedicated to, a specific plant zone
(temporary power drops, compressed air manifolds, test skids, local vacuum sources).
An activity must operate in the equipment's assigned zone to use it.

### Contract: static zone-affinity check

Unlike count-based constraints, this is a **static, time-independent** feasibility
check: if a piece of equipment declares a `zone_id`, the candidate activity's
`zone_ids` set must contain it.

| Situation | Result |
|---|---|
| `zone_id` is None (equipment not zone-locked) | `fits()` → True for any activity |
| `act_zones` is empty (activity has no zone declaration) | `fits()` → True (backward compat) |
| `zone_id` set, `act_zones` non-empty, `zone_id in act_zones` | `fits()` → True |
| `zone_id` set, `act_zones` non-empty, `zone_id not in act_zones` | `fits()` → False |

### Relationship to SystemStatePool and EquipmentPool (count)

For utility connections three independent constraints may apply simultaneously:

* **Count** — how many ports/units are available → `EquipmentPool` quantity check
* **Zone-affinity** — which zone the equipment is physically in → Option B field
* **Isolation state** — whether the connection is ENERGIZED, DRAINED, etc. → `SystemStatePool`

All three can coexist on the same activity with no interaction between them.

### What was implemented

**`outage_data.py` — `EquipmentAvailability`**

- New `zone_id: Optional[str] = None` constructor parameter stored as `self.zone_id`.

**`outage_data.py` — `EquipmentPool`**

- `from_json`: parses `zone_id = eq_data.get('zone_id')` (None when absent) and passes to
  `EquipmentAvailability` constructor.
- New `get_zone_id(equipment_id) -> Optional[str]` helper.

**`outage_data.py` — `validate_data_consistency`**

- Added check: for each equipment item with a `zone_id`, verifies it references a known
  `location_id` in the location pool.

**`pert.py` — `_fits_with_tentative`**

- Added zone-affinity block after the equipment-count block:
  ```python
  act_zones = set(activity.getZoneIds())
  for eq in activity.getRequiredEquipment():
      eq_zone = self.equipment_pool.get_zone_id(eq['equipment_id'])
      if eq_zone is not None and act_zones and eq_zone not in act_zones:
          return False
  ```

**`outage_schema.json`**

- `zone_id` added to equipment item `properties` (optional string).

**`unit_tests/test_equipment_zone.py`** — 28 new tests across 5 test classes:

| Class | Coverage |
|---|---|
| `TestEquipmentAvailabilityZoneId` | field default, setter, availability unaffected, None explicit |
| `TestEquipmentPoolFromJson` | absent, present, mixed, empty pool |
| `TestEquipmentPoolGetZoneId` | unknown equipment, no zone, with zone |
| `TestSchedulerEquipmentZone` | unzoned equipment allowed; no-zone activity allowed; correct zone allowed; wrong zone blocked; multi-zone match; mixed equipment; two-activity discrimination |
| `TestValidateEquipmentZone` | valid pass; unknown zone error; no zone passes; mixed valid/invalid |
| `TestSchemaEquipmentZone` | field present, string type, not required |

Total test count after equipment zone-affinity: **612** (584 pre-implementation + 28 new).

---

## Open Design Questions

1. **ConsumablePool replenishment:** Resolved — mid-outage restocking is supported via
   a sorted `(delivery_hour, quantity)` list with an idempotency cursor (`_restock_cursor`).

2. **PermitPool vs LocationPool unification:** Resolved by Option C — `zone_type='permit'`
   entries in `LocationPool` serve as permit zones. No separate class needed.

3. **Dose migration:** `dose_trackers` remains as-is (per-skill consumable). Its structure
   (budget shared across workers of the same skill) differs from `ConsumablePool`'s
   per-item scalar model. Both coexist cleanly. Document as "built-in consumable".

4. **`SystemStatePool` design:** Resolved — shared-state lock with reference counting,
   five scheduler touch points, 54 tests.  See decision record above.

5. **Equipment zone-affinity design:** Resolved — Option B, `zone_id` field on
   `EquipmentAvailability`, static check in `_fits_with_tentative`, 28 tests.
   See decision record above.

6. **`SystemStatePool` — same-activity multi-state conflict:** Resolved — Option A
   implemented 2026-04-15.  `validate_data_consistency` now detects when any task
   lists the same `system_id` twice with different `required_state` values and emits
   a load-time error (physically impossible: a system cannot be in two states
   simultaneously).

   **Safety Function Mutual Exclusion — Option A (implemented):**
   No new pool class needed.  Each safety function (ECCS, EDG, AFW …) is declared as
   an abstract `plant_systems` entry whose `valid_states` are train-level OOS tokens
   (`TRAIN_A_OOS`, `TRAIN_B_OOS`, …).  Activities working on Train A request
   `required_state: 'TRAIN_A_OOS'`; activities on Train B request `'TRAIN_B_OOS'`.
   `SystemStatePool`'s existing different-state mutual exclusion enforces "never both
   trains OOS simultaneously" with zero new scheduling code.
   A `safety_functions` metadata array was added to `outage_schema.json` to document
   the encoding pattern; the scheduler ignores it.

   **Future Option B — first-class `SafetyFunctionPool`:**
   If any of the following become necessary, implement Option B:
   - `max_trains_oos_simultaneously > 1` (e.g. a 4-train plant allowing 2 trains OOS)
   - Per-safety-function utilisation reporting (dashboard / audit trail)
   - Train grouping for surveillance windows that reference the safety function directly
   Option B adds `SafetyFunctionPool` with `{safety_function_id, trains, max_trains_oos}`
   semantics, a new feasibility check in `_fits_with_tentative`, and a commit step in
   `_apply_tentative`.  It supersedes the `SystemStatePool` encoding for safety functions
   but can coexist with it during migration.
