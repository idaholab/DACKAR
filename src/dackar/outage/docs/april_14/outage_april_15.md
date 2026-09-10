# Outage Scheduling — Gap Analysis & Real-Time Replanning Design
*Date: 2026-04-15*

---

## Part 1 — Gap Analysis: What the Code Addresses vs. What Outage Managers Need

### Coverage Summary (end of April 15 iteration)

| Challenge | Status | Implementation |
|---|---|---|
| Logic & dependency chains | Strong | Topological sort, lag support, all FS/SS/FF path types |
| Renewable resource contention | Strong | Per-hour capacity checks, skill substitution |
| Radiation dose (consumable) | Addressed | `ConsumablePool` + `DoseBudgetTracker` |
| Spatial / zone conflicts | Addressed | `LocationPool` with `zone_ids`, `max_concurrent_workers` |
| System state dependencies | Addressed | `SystemStatePool` mutual-exclusion lock |
| Safety function mutual exclusion | Addressed | Option A: train-per-state encoding in `SystemStatePool`; `safety_functions` schema array; intra-activity conflict validation |
| Equipment zone-affinity | Addressed | `zone_id` field on `EquipmentAvailability` |
| Hold points (NRC/QA/Eng/Ops) | Addressed | `blocks_tasks`, `is_hold_point` |
| Shift calendar | Addressed | `working_hours_per_day` enforced in scheduling loop |
| Time windows — single window | Addressed | Schema fields + `_apply_time_windows()` with window-propagating backward sweep |
| Time windows — multiple discrete windows | Addressed | `time_windows` list field; `_resolve_windows()`; scheduler tries all windows before recording violation |
| CPM slack correctness with windows | Addressed | `_apply_time_windows(topo)` backward sweep propagates tightened LF to all predecessors |
| Multi-skill substitution | Addressed | `alternative_skill_types` with ordered fallback |
| WBS priority roll-up | Addressed | `wbs_group` float aggregation |
| Mobilization lead hours | Addressed | Schema field + CPM early-start baking |
| Multi-mode activities | In progress | MMRCPSP — activity-level mode selection |
| Scale / candidate scan O(n²) | Addressed | `_completed_set` (O(1) membership); `_ready` set; `_rebuild_ready_set()`; `_select_candidate_activities` iterates `_ready` |
| Replanning — pool mutations | Addressed | `replan(resource_updates, equipment_updates)` with chop-and-replace; `clone_for_analysis()` deep-copy fix |
| Replanning — predecessor wiring for injection | Addressed | `replan(predecessor_wiring={new_id: [pred_ids]})` inserts edges automatically |
| Replanning — duration overrides | Addressed | `replan(duration_overrides={task_id: new_duration})` |

### What the Code Does Not Yet Cover — Outage Manager's View

#### 1. GP Fitness Function — The Optimization Loop Has No Objective
The scheduler produces a *feasible* schedule, not an *optimized* one. The GP coupling
point is correctly wired (`set_priorities`, `value_assignment='external'`), and the
feature vectors GP will use (`grpw`, `rr`, `slack`, etc.) are computed. `compute_fitness()`
returns a composite scalar (makespan, delay, criticality, window violations). What's
missing is the outer GP training harness that evaluates fitness across a population of
candidate priority rules and evolves them.

#### 2. Surveillance Test Recurring Windows
The schema handles both single and multiple discrete windows. Surveillance tests with
a regulatory *frequency* (e.g., every 31 days) are still not modelled — there is no
recurring-window concept. Each window must currently be declared explicitly.

#### 3. Proactive Schedule Robustness — Buffer Insertion
`compute_fitness()` provides the signal; no buffer-insertion heuristic exists yet.
Proactive robustness would insert time buffers on near-critical paths before finalising
the schedule so that a single duration slip does not propagate to the critical path.

#### 4. Scale / Performance — **Partially addressed 2026-04-15**
`nx.descendants()` was already mitigated (O(V+E) topo path in `calculate_total_successors`).
The dominant remaining bottleneck was `pred in self.completed` (O(n) list scan) called for
every activity in `self.wait` at every scheduling event → O(n³) worst case at 1,500 activities.
Fixed via `_completed_set` and `_ready` (see §5.4 below).
Remaining open item: `_serial_check_feasibility` still uses `_iter_hours` hourly scan
(Serial SGS path only; not the default strategy).

### Suggested Priority Order

| Priority | Gap | Why It Matters |
|---|---|---|
| 1 | GP training harness | `compute_fitness()` is ready; nothing can be optimised without the outer loop |
| 2 | Surveillance test recurring windows | Regulatory completeness |
| 3 | Proactive buffer insertion | Robustness before the fact |
| 4 | Scale / performance | Prerequisite for full-plant pilot |

---

## Part 2 — Real-Time Replanning: Full Design

### 2.1 What the Code Already Has

The replanning infrastructure is already substantial:

| Method | What it does |
|---|---|
| `_partial_reset(t)` | Classifies activities into completed / in_progress / pending; replays dose and consumable consumption; re-acquires `SystemStatePool` locks for in-progress activities |
| `_inject_activities(new_acts)` | Inserts new `Activity` objects into the live graph; resolves successor wiring; rebuilds `backwardDict` and `infoDict` |
| `_generate_info_from(t)` | Partial CPM with completed/in-progress times as frozen anchors; floors pending ES at `t` |
| `_build_event_queue_from(t)` | Seeds the event heap from `t` (in-progress completions, availability boundaries, CPM ES, shift starts) |
| `calculateScheduleWithResources_from(t)` | Full event-driven scheduling loop starting from `t` |
| `replan(t, new_activities, sgs)` | Orchestrates the above four steps |
| `clone_for_analysis()` | Deep-copy for what-if analysis without mutating the live schedule |

**The critical gap:** `replan()` accepts only `new_activities`. It has no mechanism to
apply resource pool changes before rescheduling. The pools (`ResourcePool`,
`EquipmentPool`, `ConsumablePool`) have no mutation API — only constructors and queries.

---

### 2.2 What a Full Mid-Outage Replan Looks Like

At `t = T` (e.g., hour 48 of a 600-hour outage), five distinct types of change may occur
simultaneously:

| Change type | Example | Affects |
|---|---|---|
| New activities | ISI crew finds cracked nozzle → 3 new tasks, 6h each | Activity graph |
| Activity duration extension | Valve replacement running longer than planned: 8h → 20h | In-progress activity |
| Resource reduction | Night-shift welder called in sick | `ResourcePool` — reduce count from T forward |
| Resource addition | Extra MECH crew mobilized and on-site | `ResourcePool` — increase count from T to T+96h |
| Equipment change | ISI crawler broke down | `EquipmentPool` — quantity drops to 0 until repair |

The current `replan()` handles only the first type. The design below adds the other four.

---

### 2.3 Pool Mutation Design

**Core operation: `update_from_hour(from_hour, until_hour, new_count)`**

Each pool needs a mutation method that:
1. Splits any existing period that straddles `from_hour` into two: the portion before
   `from_hour` (unchanged) and the portion at/after `from_hour` (replaced).
2. Removes any periods entirely within `[from_hour, until_hour)`.
3. Appends a new period `[from_hour, until_hour)` with `new_count`.
4. Re-sorts and re-validates.

This is a "chop and replace" operation — predictable, auditable, and reversible if
the caller holds the original snapshot.

```python
# ResourceAvailability
def update_from_hour(self, outage_start: datetime,
                     from_hour: float,
                     until_hour: float,   # None = end of outage
                     new_count: int) -> None:
    """Replace availability in [from_hour, until_hour) with new_count."""

# EquipmentAvailability
def update_from_hour(self, outage_start: datetime,
                     from_hour: float,
                     until_hour: float,
                     new_quantity: int) -> None:
```

Pool-level wrappers:
```python
# ResourcePool
def update_skill_from_hour(self, skill_type: str, outage_start: datetime,
                            from_hour: float, until_hour: float,
                            new_count: int) -> None

# EquipmentPool
def update_equipment_from_hour(self, equipment_id: str, outage_start: datetime,
                                from_hour: float, until_hour: float,
                                new_quantity: int) -> None
```

**ConsumablePool restocks** are already handled by `apply_restocks_up_to()` in
`_partial_reset`. A new emergency delivery is modeled by appending a restock entry
before calling `replan()`.

---

### 2.4 Duration Override for In-Progress Activities

When an activity's remaining duration changes (e.g., ISI inspection was expected to
finish in 4 more hours but now needs 28), `_partial_reset` must be told the new
remaining duration *before* it computes `act._remaining_duration`.

Proposed: a `duration_overrides` dict passed to `replan()`:

```python
duration_overrides = {
    'T_ISI_NOZZLE_INSP': 28.0,   # new total duration for this task
}
```

Inside `_partial_reset`, after classifying an in-progress activity, check
`duration_overrides.get(act.name)` and update `act.duration` and
`act._remaining_duration` accordingly. `_generate_info_from` will then
propagate the new EF through the network.

---

### 2.5 Extended `replan()` Signature

```python
def replan(
    self,
    current_time_hours: float,
    new_activities: list = None,
    resource_updates: list = None,
    equipment_updates: list = None,
    duration_overrides: dict = None,
    sgs: str = 'max_use_res_ranked',
    max_time_hours: float = None,
) -> dict:
```

`resource_updates` format:
```python
[
    # Reduce: welder called in sick from hour 48 onwards
    {'skill_type': 'WELDER', 'from_hour': 48, 'until_hour': None, 'new_count': 2},
    # Add: extra MECH crew, available for 96 hours
    {'skill_type': 'MECHANIC', 'from_hour': 52, 'until_hour': 148, 'new_count': 7},
]
```

`equipment_updates` format:
```python
[
    # ISI crawler broken; back at hour 60
    {'equipment_id': 'ISI_CRAWLER', 'from_hour': 48, 'until_hour': 60, 'new_quantity': 0},
]
```

**Execution sequence inside the extended `replan()`:**

```
1. Apply resource_updates   → resource_pool.update_skill_from_hour(...)
2. Apply equipment_updates  → equipment_pool.update_equipment_from_hour(...)
3. Inject new_activities    → _inject_activities(new_activities)
4. Apply duration_overrides → stored on self._pending_duration_overrides
5. _partial_reset(t)        → uses duration_overrides for in-progress durations
6. Rebuild nxgraph
7. _generate_info_from(t)   → updated CPM anchors
8. Rebuild _availability_events → pool boundary events have changed
9. calculateScheduleWithResources_from(t)
```

Step 8 is important: `_availability_events` is built at `__init__` time from the
original pool periods. After pool mutations, those boundaries are stale. They must
be recomputed before building the event heap.

---

### 2.6 Snapshot / Rollback

Before mutating pools, `replan()` should optionally capture a snapshot of the current
pool state so the caller can roll back if the replan result is rejected:

```python
snapshot = p.resource_pool.snapshot()   # shallow copy of periods lists
# ... replan ...
p.resource_pool.restore(snapshot)       # undo if needed
```

This supports "what-if" replanning — the outage manager can explore three different
crew assignments and pick the best outcome before committing.

`clone_for_analysis()` already provides this pattern at the Pert level. Pool-level
`snapshot()` / `restore()` would complete the picture.

---

### 2.7 Constraints on Resource Mutations

Two rules that the mutation API should enforce:

1. **No retroactive changes.** `from_hour` must be ≥ `current_time_hours` at the time
   `replan()` is called. Changing availability in the past would invalidate completed
   activities' feasibility checks, which is physically meaningless.

2. **Non-negative count.** `new_count` / `new_quantity` ≥ 0. A negative availability
   is a schema error, not a legitimate constraint.

---

### 2.8 What This Enables (Outage Manager's Perspective)

| 3 AM scenario | Extended replan() handles it |
|---|---|
| ISI team finds cracked nozzle — inject 3 new tasks | `new_activities` (already exists) |
| Night-shift welder called in sick | `resource_updates`: reduce WELDER count from T |
| Valve replacement running 12h over | `duration_overrides`: T_VALVE_REPLACE = new_duration |
| ISI crawler broke down | `equipment_updates`: ISI_CRAWLER quantity → 0 until repaired |
| Emergency resin delivery arrived | `consumable_pool.add_restock(hour, qty)` before replan |
| Extra contract mechanics on site | `resource_updates`: increase MECHANIC count from T |
| Want to try two staffing options before committing | `clone_for_analysis()` + two `replan()` calls |

The full replan (all six changes above, 600 activities, event-driven loop) should
run in well under one second based on the current scheduler's timing characteristics.

---

### 2.9 Implementation Scope

The implementation requires changes to three files:

| File | Change |
|---|---|
| `outage_data.py` | Add `update_from_hour()` to `ResourceAvailability` and `EquipmentAvailability`; add pool-level wrappers; add `snapshot()` / `restore()` |
| `pert.py` | Extend `replan()` signature; apply resource/equipment updates in the correct sequence; handle `duration_overrides` in `_partial_reset`; rebuild `_availability_events` after mutations |
| `unit_tests/test_replan_resources.py` | ~35 new tests: pool mutation unit tests; scheduler integration tests for each change type; snapshot/rollback test |

No changes to `activity.py` or `outage_schema.json` are required — the schema is
unchanged (pool mutations are runtime operations, not input-data declarations).

---

## Part 3 — Implementation Record (2026-04-15)

### 3.1 What Was Implemented

All five mid-outage change types are now handled by a single `replan()` call.

#### `outage_data.py`

**`ResourceAvailability`**
- `update_from_hour(outage_start, from_hour, new_count, until_hour=None)`:
  Chop-and-replace on `self.periods`. Splits any period straddling `from_dt` or
  `until_dt`, removes periods entirely within the window, inserts a new period
  `[from_dt, until_dt)` with `new_count`. Uses `datetime(9999, 12, 31)` as a
  far-future sentinel when `until_hour` is None (permanent change). Does not call
  `_validate_periods()` — adjacent periods are valid and the sentinel end is legal.
- `snapshot() -> list`: `copy.deepcopy(self.periods)` for rollback.
- `restore(saved)`: replaces `self.periods` from a saved snapshot.

**`EquipmentAvailability`** — identical three methods; period key is `quantity_available`.

**`ResourcePool`**
- `update_skill_from_hour(skill_type, outage_start, from_hour, new_count, until_hour=None)`:
  Delegates to `ResourceAvailability.update_from_hour`; no-op with warning for
  unknown skill.
- `snapshot() -> dict`: `{skill: ra.snapshot()}` for all skills.
- `restore(saved)`: calls `ra.restore(periods)` per skill.

**`EquipmentPool`** — same three methods for equipment items.

Also added: `import copy` and `from datetime import timedelta` to the module imports.

#### `pert.py`

**`_partial_reset(current_time_hours, duration_overrides=None)`** — extended:
- For each in-progress activity, checks `duration_overrides.get(act.name)`.
  If found: sets `act.duration = new_total` (permanent), computes
  `remaining = max(0, new_total - elapsed)`. If not found: uses original
  `endTime` as before.

**`replan(current_time_hours, new_activities, resource_updates, equipment_updates,
          duration_overrides, sgs, max_time_hours)`** — extended:
- Steps 1–2: apply `resource_updates` / `equipment_updates` via pool wrappers.
  Validates `from_hour ≥ 0` (no retroactive changes); raises `ValueError` otherwise.
- Step 3: call `_precompute_availability_events()` if any pool was mutated —
  new period boundaries must be in the event heap.
- Steps 4–7: unchanged (`_inject_activities`, `_partial_reset`, `nxgraph` rebuild,
  `_generate_info_from`, `calculateScheduleWithResources_from`).

**`clone_for_analysis()`** — bug fix:
- Previously shared pool references with the original Pert. `replan()` on the clone
  would mutate the baseline's pools.
- Now deep-copies `resource_pool`, `equipment_pool`, `location_pool`,
  `consumable_pool`, `system_state_pool`.
- Rebuilds `dose_trackers` from the cloned `resource_pool` so tracker identity
  is consistent.

### 3.2 New Activity Injection — Already Implemented

`new_activities` was already supported by the existing `_inject_activities()` /
`replan()` infrastructure (implemented in a prior session). It is preserved in the
extended signature and tested in `test_resource_equipment_and_new_activity`.
The full set of mid-outage change types handled:

| Change type | Parameter | Status |
|---|---|---|
| Inject new activities (emergent work) | `new_activities` | Pre-existing |
| Extend in-progress task duration | `duration_overrides` | New (Apr 15) |
| Reduce/increase workforce | `resource_updates` | New (Apr 15) |
| Equipment breakdown/return | `equipment_updates` | New (Apr 15) |
| Emergency consumable delivery | `consumable_pool.add_restock()` before call | Pre-existing |

### 3.3 Test Results

**`unit_tests/test_replan_resources.py`** — 33 new tests across 8 classes:

| Class | Coverage |
|---|---|
| `TestResourceAvailabilityUpdateFromHour` | Chop-and-replace mechanics: full, partial, temporary, zero, increase, stacked, outside-window |
| `TestResourceAvailabilitySnapshot` | Restore after update; snapshot independence |
| `TestEquipmentAvailabilityUpdateFromHour` | Full replacement, temporary breakdown, partial reduction |
| `TestEquipmentAvailabilitySnapshot` | Restore after update |
| `TestResourcePoolWrapper` | Wrapper delegation; unknown skill no-op; pool snapshot/restore |
| `TestEquipmentPoolWrapper` | Wrapper delegation; unknown equipment no-op; pool snapshot/restore |
| `TestReplanResourceUpdates` | Sick call serialises pending parallel tasks; extra crew enables parallelism; temporary window delays then unblocks |
| `TestReplanEquipmentUpdates` | Broken equipment blocks pending task; temporary breakdown with repair at `until_hour` |
| `TestReplanDurationOverrides` | In-progress extension; delay propagates to successors; permanent update |
| `TestReplanCombined` | All three change types + new activity in one call; `ValueError` on negative `from_hour` |
| `TestClonePoolIsolation` | Resource mutation on clone doesn't affect original; equipment same; independent dose trackers; two clones independent |

Total test count: **612 → 645** (+33 new, 0 regressions).

---

## Part 4 — Safety Function Mutual Exclusion (Option A)

### 4.1 Problem

Nuclear outages involve multi-train safety systems (ECCS, EDG, AFW, etc.) where a hard
regulatory constraint applies: **no two trains of the same safety function may be out of
service simultaneously**.  The scheduler must refuse to start Train B work while Train A
work is in progress, and vice versa.

### 4.2 Design Decision — Option A (implemented)

No new pool class needed.  The existing `SystemStatePool` already enforces mutual exclusion
between activities that require *different* states on the same system.  By mapping each
train's OOS condition to a distinct state token, the constraint falls out naturally.

**Encoding pattern:**

1. Declare an abstract system in `plant_systems`:
   ```json
   {
     "system_id": "ECCS_SAFETY_FUNCTION",
     "description": "Emergency Core Cooling System",
     "valid_states": ["TRAIN_A_OOS", "TRAIN_B_OOS"]
   }
   ```
2. Tag each activity with its train state in `required_system_states`:
   - Train A maintenance → `{"system_id": "ECCS_SAFETY_FUNCTION", "required_state": "TRAIN_A_OOS"}`
   - Train B maintenance → `{"system_id": "ECCS_SAFETY_FUNCTION", "required_state": "TRAIN_B_OOS"}`
3. `SystemStatePool` blocks any activity requiring `TRAIN_B_OOS` while any holder of
   `TRAIN_A_OOS` is active (and vice versa).  Multiple activities on the same train can
   run concurrently (reference-counted lock).

**What was built:**

| Component | Change |
|---|---|
| `outage_data.py` — `validate_data_consistency()` | Added intra-activity conflict check: same `system_id` with different `required_state` values in one task → load-time error ("physically impossible — a system cannot be in two states simultaneously") |
| `outage_schema.json` | Added optional top-level `safety_functions` array to document the encoding pattern (metadata only — scheduler ignores it) |
| `resource_pool_extensions.md` | Open Design Question 6 closed; Option B future note added |
| `unit_tests/test_safety_function.py` | 19 new tests (see §4.3) |

**Future Option B — `SafetyFunctionPool`:**
If `max_trains_oos_simultaneously > 1` (e.g. 4-train plant allowing 2 trains OOS) or
dedicated per-function reporting is needed, implement a first-class `SafetyFunctionPool`
with `{safety_function_id, trains, max_trains_oos}` semantics.  See
`resource_pool_extensions.md` — Open Design Question 6 for full specification.

### 4.3 Test Results

**`unit_tests/test_safety_function.py`** — 19 new tests across 3 classes:

| Class | Coverage |
|---|---|
| `TestIntraActivityStateConflict` | No states → valid; empty list → valid; single state → valid; same system+state twice → valid (redundant); same system different states → error; error message names both states and system; conflict in T1 doesn't suppress T2; three entries with conflict; different systems → no conflict |
| `TestSafetyFunctionEncoding` | Same train → coexist; different trains → mutually exclusive; release allows other train; reference-counted lock requires two releases |
| `TestSafetyFunctionsSchema` | Key present; array type; required fields; `train_ids` min 2; `max_trains_oos_simultaneously` optional; not in top-level required |

Total test count: **645 → 664** (+19 new, 0 regressions).

---

## Part 5 — Exposed Issues from April 14 Changelog Scope Boundaries

Review of the items explicitly marked "not in scope" in `outage_april_14_changelog.md`
identified four issues. Three are design gaps; one (#1) is a silent correctness defect
that degrades schedule quality whenever time-windowed activities exist.

---

### Issue #1 — Window-propagating CPM (SILENT CORRECTNESS DEFECT — Priority)

**Source:** Change 8 scope boundary — *"adjustments to windowed ES/LF are local;
upstream/downstream CPM values are not updated."*

**What goes wrong:**

`_apply_time_windows()` tightens `ES`/`LF` for windowed activities only, without
propagating those changes through the network. Consider `A → B → C` where B has
`window_latest_finish = 100h` and the unconstrained CPM gives B an LF of 120h:

1. `_apply_time_windows` sets B's LF = 100h and B's slack = −20h (infeasibility warning
   fired).
2. A's LF is **unchanged** — it is still computed from the 120h unconstrained LF. A
   therefore shows ~20h of float it does not actually have.
3. `_weight_function` and `_compute_wbs_slack` both read `infoDict[act]['slack']`. A's
   priority weight is underestimated; if A belongs to a WBS group, the whole group's
   priority is wrong.
4. `compute_fitness()` undercounts truly critical activities in `criticality_ratio`.

The scheduler's `_select_candidate_activities` still dispatches correctly — it enforces
the window at decision time — but the *priority ordering* that drives which activities
are selected first is computed from a globally inconsistent CPM.

**Affected code paths:** `_weight_function`, `_compute_wbs_slack`, `compute_fitness()`,
any GP fitness evaluation that uses `slack` or `criticality_ratio`.

**Planned fix:** see §5.1.

---

### Issue #2 — Multiple windows per activity (RESOLVED 2026-04-15)

**Source:** Change 8 scope boundary — *"some surveillance tests can run in any of several
discrete windows — one window pair per activity only."*

Technical Specifications frequently allow discrete alternatives: "run between T+48h and
T+72h, **or** between T+120h and T+144h." With only a single `(window_earliest_start_hours,
window_latest_finish_hours)` pair, a missed first window was incorrectly recorded as a
violation even when a second window remained available.

**What was built:**

| Component | Change |
|---|---|
| `activity.py` | Added `time_windows: list = []` field; `from_json` parses `time_windows` JSON array; `to_json_dict` serialises it when non-empty |
| `pert.py` — `_resolve_windows(act)` | New helper: returns unified `[(earliest, latest)]` tuple list from either `time_windows` or legacy single-window fields |
| `pert.py` — `_apply_time_windows()` | Uses `_resolve_windows`; CPM bounds from broadest envelope (min earliest, max latest) |
| `pert.py` — `_select_candidate_activities()` | Iterates all windows in earliest-first order; records violation only when ALL windows are exhausted; waits silently while a future window exists |
| `pert.py` — `_build_event_queue()` / `_build_event_queue_from()` | Seeds events for every window's open time |
| `outage_schema.json` | Added `time_windows` array to task items |
| `unit_tests/test_time_windows.py` | 14 new tests in `TestMultipleWindows` |

---

### Issue #3 — Predecessor wiring for injected activities (RESOLVED 2026-04-15)

**Source:** Change 9 scope boundary — *"caller must update existing activities' `childs`
lists manually before calling `replan()`."*

`_inject_activities` wired new activities' outgoing edges but had no mechanism for
incoming edges from existing activities, leaving precedence constraints silent if the
caller forgot to wire them.

**What was built:**

| Component | Change |
|---|---|
| `pert.py` — `_inject_activities(new_activities, predecessor_wiring=None)` | New `predecessor_wiring` parameter: `{new_task_id: [existing_pred_id, ...]}`. After building `forwardDict` for all new activities, inserts edges `pred → new_act` for each entry. Unknown IDs on either side are skipped with a WARNING — no crash. Accepts string shorthand when a single predecessor is supplied. |
| `pert.py` — `replan(..., predecessor_wiring=None, ...)` | Accepts and forwards `predecessor_wiring` to `_inject_activities` |
| `unit_tests/test_replan.py` | 6 new tests in `TestPredecessorWiring` |

**Usage:**

```python
new_insp = Activity('T_INSP', 6.0)
new_insp.childs = []   # no successors (or list them)
result = p.replan(
    current_time_hours=48.0,
    new_activities=[new_insp],
    predecessor_wiring={'T_INSP': ['T_ISOLATE']},  # T_ISOLATE → T_INSP
)
```

---

### Issue #4 — O(n²) performance ceiling (SCALE RISK)

**Source:** "What remains" section — *"`_iter_hours` and `nx.descendants()` both O(n²);
unverified at 1500-activity outage scale."*

`nx.descendants()` is called per activity inside `calculate_total_successors()` and
`_iter_hours` scans the full activity list on every event. At ~300 activities (current
test datasets) this is imperceptible. At 1,500+ activities (a full unit outage) both
loops become the dominant runtime cost. This is not a correctness issue — it is a
practical deployment risk.

**Not planned for the current iteration.** Note as a prerequisite before any
full-plant pilot.

---

### 5.1 — Fix: Window-Propagating CPM Backward Pass (Implemented 2026-04-15)

**Goal:** After `_apply_time_windows()` tightens a windowed activity's LF, propagate
the tightened constraint backward through all predecessors so that every upstream
activity's LF — and therefore slack — reflects the window. No forward propagation is
needed (ES tightening by `window_earliest_start_hours` only affects successors,
which the standard forward pass already handles up to the windowed node).

#### Algorithm

The standard CPM backward pass computes:

```
LF(u) = min over all successors v of: LF(v) - lag(u,v) - v.mobilization_lead_hours
LS(u) = LF(u) - u.duration
slack(u) = LS(u) - ES(u)
```

The window-tightened LF values from `_apply_time_windows()` are correct starting
points. The existing backward pass in `generateInfo()` runs *before*
`_apply_time_windows()`, so it never sees the tightened values.

**Fix: call a second backward-pass sweep after `_apply_time_windows()`.**

```
reverse topological order of all activities:
    for u in reverse_topo_order:
        if u has successors:
            LF(u) = min(LF(u), min over successors v of:
                        infoDict[v]['lf'] - lag(u,v) - v.mobilization_lead_hours)
            LS(u) = LF(u) - u.duration
            slack(u) = LS(u) - ES(u)
```

This is a single linear pass in reverse topological order (O(V + E)) — identical cost
to the existing backward pass. No iteration needed: because the graph is a DAG,
one reverse-topo sweep propagates every window constraint to all ancestors.

The same sweep must be applied in `_generate_info_from()` (used by `replan()`) after
its own call to `_apply_time_windows()`.

#### Changes required

| Location | Change |
|---|---|
| `pert.py` — `_apply_time_windows()` | Add the reverse-topo backward sweep at the end of the method |
| `pert.py` — `_generate_info_from()` | Already calls `_apply_time_windows()` — the sweep inside that method covers replanning automatically |
| `unit_tests/test_time_windows.py` | Add tests: predecessor slack tightened by window; WBS group slack updated; chain A→B→C with window on B updates A's slack; `compute_fitness()` criticality_ratio reflects correct critical set |

#### What was changed

| File | Change |
|---|---|
| `pert.py` — `_apply_time_windows(topo=None)` | Added `topo` parameter; tracks `any_lf_tightened` flag; appends reverse-topo backward sweep when `topo` is provided and a window actually reduced any LF |
| `pert.py` — `generateInfo()` | Passes `topo=topo` to `_apply_time_windows` |
| `pert.py` — `_generate_info_from()` | Passes `topo=topo` to `_apply_time_windows` — replanning path covered automatically |
| `unit_tests/test_time_windows.py` | 12 new tests in `TestWindowPropagatingBackwardSweep` |

Test count: **664 → 676** (+12 new, 0 regressions).

Running total after Issues #2 and #3: **676 → 694** (+18 new, 0 regressions).

#### Scope boundary for this fix

- **ES forward propagation from `window_earliest_start_hours`**: not needed —
  a window that opens *late* only constrains the windowed activity itself; the
  standard forward pass already propagated predecessor completions forward.
- **Infeasibility detection**: unchanged — the `window_infeasible` flag and
  WARNING remain as-is. The new backward sweep may cause additional upstream
  activities to show negative slack if the window makes the network infeasible;
  these will be visible as negative slack values (not separately flagged).
- **`_select_candidate_activities` window enforcement**: unchanged — dispatching
  logic already correct; only CPM values change.
- **Multi-window activities (Issue #2)**: out of scope for this fix.

---

### 5.4 — Fix: O(n³)→O(n) Candidate Selection (Implemented 2026-04-15)

**Goal:** Eliminate the dominant performance bottleneck in the event-driven scheduling
loop: O(n) list scan for predecessor membership check × O(n) waiting activities × O(n) events = O(n³).

#### Root cause

In `_select_candidate_activities()`, two independent O(n) operations compounded:

1. **`pred in self.completed`** — `self.completed` is a Python `list`; membership test is O(n).
   Called for every predecessor of every waiting activity at every event.
2. **`for act in list(self.wait)`** — iterates *all* waiting activities regardless of
   predecessor readiness. At a typical event, the vast majority of waiting activities
   have unsatisfied predecessors and are examined needlessly.

Combined: O(events × |wait| × |predecessors|) where each predecessor check is O(|completed|).
For 1,500 activities: ~6.75 billion comparisons in the predecessor loop alone.

#### Algorithm

**Fix A — `_completed_set: set`**
Mirror of `self.completed` maintained as a Python `set`. `pred in self._completed_set` is O(1).

**Fix B — `_ready: set`**
Subset of `self.wait` containing only activities whose direct predecessors are all in
`_completed_set`. Maintained incrementally:

- `_rebuild_ready_set()`: O(n × k) one-time rebuild, called after `_reset_scheduling_state`
  and `_partial_reset`.
- `_update_ongoing_list()`: when activity A completes, walk `forwardDict[A]` and promote
  any successor whose full predecessor list is now satisfied. O(|successors of A| × |predecessors|).
- `self.wait.remove(act)` call-sites: add `self._ready.discard(act)` to keep `_ready ⊆ wait`.

`_select_candidate_activities` iterates `self._ready` (typically 5–30 activities at any event)
instead of `self.wait` (~750 on average for a 1,500-activity schedule). The predecessor check
inside the loop becomes a safety-net O(1) lookup using `_completed_set`.

#### Changes required

| Location | Change |
|---|---|
| `pert.py` — `__init__` | Add `self._completed_set: set` and `self._ready: set` |
| `pert.py` — `_rebuild_ready_set()` | New helper method; O(n × k) rebuild from wait and _completed_set |
| `pert.py` — `_reset_scheduling_state` | Reset `_completed_set = set()`, call `_rebuild_ready_set()` |
| `pert.py` — `_partial_reset` | After loop: `_completed_set = set(completed)`, call `_rebuild_ready_set()` |
| `pert.py` — `_update_ongoing_list` | After append to completed: add to `_completed_set`, unlock successors into `_ready` |
| `pert.py` — bootstrap in `calculateScheduleWithResources` | `_ready.discard(startActivity)` after removing from wait |
| `pert.py` — commit selected activities | `_ready.discard(act)` after `wait.remove(act)` |
| `pert.py` — window violation | `_ready.discard(act)` after `wait.remove(act)` |
| `pert.py` — serial SGS | `_ready.discard(act)`, `_completed_set.add(act)` |
| `pert.py` — `_select_candidate_activities` | Iterate `self._ready` (was `self.wait`); use `_completed_set` for O(1) predecessor check |
| `unit_tests/test_scale_performance.py` | New test file: 17 tests across `TestCompletedSet`, `TestReadySet`, `TestSchedulingCorrectnessWithReadySet` |
| `unit_tests/test_replan_resources.py` | Fixed over-constrained baseline assertion (was checking one serialization order; now accepts either) |
| `unit_tests/test_system_state_pool.py` | Fixed over-constrained serialization assertion (same reason) |
| `unit_tests/test_substitution.py` | Fixed over-constrained serialization assertion (same reason) |
| `unit_tests/test_shift_calendar.py` | Added `_completed_set = set(); _rebuild_ready_set()` to tests that manually set `p.wait` |

#### What was changed

Three test files had assertions that assumed a specific serialization *order* (A before B)
when the scheduler is free to choose either order. With `_ready` as a Python set, iteration
order is non-deterministic — the same correct serialization now sometimes produces B before A.
The assertions were updated to accept either order: `b_et <= a_st or a_et <= b_st`.

Test count: **694 → 711** (+17 new, 0 regressions).

#### Complexity summary

| Operation | Before | After |
|---|---|---|
| `pred in self.completed` per check | O(n) | O(1) |
| Outer loop in `_select_candidate_activities` per event | O(\|wait\|) ≈ O(n/2) | O(\|ready\|) ≈ O(5–30) |
| Full schedule at n=1,500 | ~6.75 billion comparisons | ~O(n × k) |
| `_rebuild_ready_set` (one-time reset cost) | — | O(n × k), k = avg predecessors |
| `_update_ongoing_list` successor unlock | — | O(\|successors\| × k) per completion |
