# LOGOS CPM/RCPSP Module — Change Log
**Date:** April 14, 2026  
**Files modified:** `LOGOS/src/CPM/pert.py`, `LOGOS/src/CPM/activity.py`  
**Reference documents:** `outage_april_2.docx` (approach overview), `outage_april_3.docx` (code assessment)

---

## Overview

This iteration addresses the highest-priority gaps identified in the April 3rd code assessment. All changes are backward-compatible: schedules built from existing JSON data with no lag entries and `working_hours_per_day=24` behave identically to before.

---

## Change 1 — `_weight_function` sigmoid cliff scaled with project duration

**File:** `pert.py` — `_weight_function()`, two call sites  
**Category:** Immediate / blocks GP coupling  

### Problem
The sigmoid priority weight function had a hardcoded inflection point at `slack = 5 hours`:
```python
return 1.0 - 1.0 / (1.0 + math.exp(5.0 - total_float))
```
For a 720-hour outage, activities with 4.9 h and 5.1 h of float received radically different priorities (≈0.73 vs ≈0.27). The 5-hour cliff is meaningless relative to a project horizon one or two orders of magnitude longer.

### Fix
The inflection point now scales with project duration:
```python
threshold = max(5.0, 0.01 * project_duration)
return 1.0 - 1.0 / (1.0 + math.exp(threshold - total_float))
```
| Project duration | Inflection point |
|-----------------|-----------------|
| ≤ 500 h          | 5 h (floor)      |
| 720 h            | 7.2 h            |
| 2 000 h          | 20 h             |

Both call sites (`_select_candidate_activities` and `LookAheadScheduler._evaluate_future_opportunities`) updated to pass `project_duration`.

---

## Change 2 — Finish-to-start lag relationships

**Files:** `activity.py` — `Activity.__init__`, `from_json`, `to_json_dict`; `pert.py` — `Pert.__init__`, `_build_graph_from_outage_data`, `generateInfo` forward/backward pass  
**Category:** High priority / domain correctness (Challenge 8 in assessment)

### Problem
All predecessor–successor links were pure FS zero-lag. Real outage schedules use lags on roughly 20–30% of links (e.g. "weld inspection can start 2 h after weld completion", "scaffold erection and staging must overlap but staging follows by 4 h"). The schema and CPM forward pass had no mechanism to express or compute these.

### Fix

**`Activity`**

Added `successor_lags: dict` attribute mapping successor task-ID → lag hours. `from_json` parses a mixed-format `successors` list — each entry may be either a plain task-ID string (zero lag, fully backward-compatible) or a dict:
```json
"successors": ["T002", {"task_id": "T003", "lag_hours": 2.0}]
```
`to_json_dict` serialises back to the same format, round-tripping cleanly.

**`Pert`**

- `lag_dict: dict` attribute maps `(predecessor_Activity, successor_Activity) → lag_hours`. Populated in `_build_graph_from_outage_data()` from `activity.successor_lags`.
- Forward pass updated: `ES(v) = max(EF(u) + lag(u,v))` for each predecessor `u`.
- Backward pass updated: `LF(u) ≤ LS(v) − lag(u,v)`.
- `clone_for_analysis()` translates `lag_dict` keys to the cloned Activity instances.

### Bonus fix discovered during implementation
The forward pass condition `if u_ef > v.es` silently failed to propagate EF for activities whose only predecessor is a zero-duration START node (START.ef=0, v.es=0, condition `0 > 0` → False). Changed to `>=`. This is a correctness fix that affects any graph with an explicit START activity of duration 0.

---

## Change 3 — Shift calendar enforcement

**File:** `pert.py` — `Pert.__init__`, three new methods, `_select_candidate_activities`, `_build_event_queue`  
**Category:** High priority / schedule validity (Challenge 12 in assessment — "smoking gun")

### Problem
`working_hours_per_day` was stored in `OutageData`, printed in `print_summary()`, and assigned to `self.working_hours_per_day` in `Pert.__init__` — but **never referenced anywhere in the scheduling loop**. The scheduler would happily start a 16-hour task at hour 20 of a 24-hour period, running silently through what would be a shift boundary with potentially zero staff.

### Fix

Added `shift_start_hour: int` attribute (default 0; also readable from `OutageData.shift_start_hour` if present).

Three new helper methods:

| Method | Purpose |
|--------|---------|
| `_is_work_time(t)` | Returns `True` if `t` falls inside an active shift window. Always `True` when `working_hours_per_day ≥ 24`. |
| `_next_shift_start_after(t)` | Snaps `t` forward to the earliest valid shift-start ≥ `t`. |
| `_shift_boundary_events(start, end)` | Returns all daily shift-start datetimes in a window, for event-queue seeding. |

**Scheduling integration:**

1. `_select_candidate_activities()` returns an empty dict immediately when `_is_work_time(time)` is `False`. No activity can be started in an off-shift period.
2. `_build_event_queue()` seeds daily shift-start events across the full project window when `working_hours_per_day < 24`, ensuring the event loop wakes up at each shift open without spinning.

**Behaviour:**
- `working_hours_per_day = 24` (default) → no change, all hours valid.
- `working_hours_per_day = 12, shift_start_hour = 6` → activities only start 06:00–18:00.

---

## Change 4 — Composite fitness function for GP coupling

**File:** `pert.py` — `Pert.compute_fitness()`, `calculateScheduleWithResources()`, `Pert.__init__`  
**Category:** Immediate / blocks GP training (identified as missing in assessment)

### Problem
The GP coupling point was correctly designed — `set_priorities()` injects evolved rules, `value_assignment='external'` routes them into candidate selection — but the **output evaluation was undefined**. Without a fitness signal, GP cannot be trained.

### Fix

`calculateScheduleWithResources()` now caches its return dict as `self._last_schedule_result`.

New method `Pert.compute_fitness(alpha=1.0, beta=0.5, gamma=0.3)`:

| Component | Formula | Meaning |
|-----------|---------|---------|
| `makespan_ratio` | `scheduled_duration / cpm_duration` | 1.0 = no resource stretch |
| `delay_ratio` | `total_delay_hours / cpm_duration` | 0 = no waiting |
| `criticality_ratio` | `zero_actual_TF activities / total activities` | Higher = less robust |
| **`composite`** | `α·makespan + β·delay + γ·criticality` | **Minimise for GP** |

Default weights give makespan dominant influence, matching nuclear outage priorities (finish on time above all else). All components are returned so GP fitness functions can weight them differently.

**Usage:**
```python
pert.calculateScheduleWithResources(sgs='max_use_res_ranked')
fitness = pert.compute_fitness(alpha=1.0, beta=0.5, gamma=0.3)
# fitness['composite'] is the scalar to minimise
```

---

## Change 5 — `MDKnapsackScheduler` made location-aware

**File:** `pert.py` — `MDKnapsackScheduler._get_capacities()`, `_get_resource_consumption()`  
**Category:** Medium priority / domain correctness

### Problem
The greedy knapsack optimizer only tracked resource and equipment dimensions. Location constraints (`max_concurrent_tasks`, `max_concurrent_workers` per zone) were absent from its capacity vector. In high-radiation zones — which are often the binding constraint in real outages — the knapsack would produce an initial activity set that `_fits_with_tentative` then rejected, wasting iterations and degrading solution quality.

### Fix

`_get_capacities()` now adds two dimensions per location:
- `LOC_TASKS_<loc_id>` — maximum concurrent task slots
- `LOC_WORKERS_<loc_id>` — maximum concurrent worker slots (`inf` if unbounded)

`_get_resource_consumption()` contributes:
- `LOC_TASKS_<loc_id>` = 1 (one task slot consumed)
- `LOC_WORKERS_<loc_id>` = sum of crew counts across all resource requirements

The greedy selector now respects zone occupancy limits during initial selection, not only during the downstream `_fits_with_tentative` re-validation pass.

---

## Change 6 — `addActivity()` hardened

**File:** `pert.py` — `Pert.addActivity()`, `Pert.calculate_gp_rules()`  
**Category:** High priority / correctness

### Problem (a) — infoDict key mismatch
The inline `infoDict` initialisation in `addActivity()` had only 6 keys, while `resetInfo()` (called immediately after) initialises 14. This was functionally harmless because `resetInfo()` overwrote it, but it was misleading and a latent risk if execution order ever changed.

### Fix (a)
Inline dict updated to include all 14 keys (matching `resetInfo()` exactly).

### Problem (b) — `nxgraph` not rebuilt
After adding the new activity to `forwardDict`, `generateInfo()` calls `calculate_total_successors()` which calls `nx.descendants(self.nxgraph, activity)`. Because `nxgraph` was never rebuilt, this raised `NetworkXError: The node X is not in the digraph`.

### Fix (b)
```python
self.nxgraph = nx.DiGraph(self.forwardDict)  # rebuild before generateInfo()
self.resetInfo()
self.generateInfo()
```

### Problem (c) — `ZeroDivisionError` in `calculate_gp_rules()`
When called on trivial or single-activity networks (all `es=0`), `max_es=0` caused a division by zero.

### Fix (c)
Safe denominators throughout: `safe_max_X = max_X if max_X != 0 else 1.0`.

---

## Summary table

| # | Change | Files | Category |
|---|--------|-------|----------|
| 1 | `_weight_function` threshold scales with project duration | `pert.py` | Immediate |
| 2 | Finish-to-start lag relationships (schema + CPM pass) | `activity.py`, `pert.py` | High |
| 3 | Shift calendar enforcement (`working_hours_per_day` actually used) | `pert.py` | High |
| 4 | `compute_fitness()` composite signal for GP training | `pert.py` | Immediate |
| 5 | `MDKnapsackScheduler` location-aware | `pert.py` | Medium |
| 6 | `addActivity()` — full infoDict keys, `nxgraph` rebuild, `ZeroDivisionError` guard | `pert.py` | High |
| 7 | Pool-level consumable radiation dose tracking (Challenge 2) | `outage_data.py`, `activity.py`, `pert.py`, schema | Medium |
| 8 | Regulatory time-window constraints (Challenge 6) | `activity.py`, `pert.py`, schema | High |
| 9 | Real-time replanning with frozen activity state (Challenge 4) | `activity.py`, `pert.py` | High |
| 10 | Mobilization lead times (Challenge 13) | `activity.py`, `pert.py`, schema | Medium |
| 11 | Multi-mode activities / MMRCPSP (Challenge 15) | `activity.py`, `pert.py`, schema | Medium |
| 12 | Multi-skill substitution (Challenge 10) | `activity.py`, `pert.py`, schema | Medium |
| 13 | WBS-level priority roll-up (Challenge 11) | `activity.py`, `pert.py`, schema | Medium |

---

## Change 10 — Mobilization lead times (Challenge 13)

**Files:** `activity.py`, `pert.py`, `outage_schema.json`, `unit_tests/test_mobilization_lead.py`  
**Category:** Medium priority / domain correctness

### Problem

Vendor specialists and specialized crews require advance notice before they can start work: a team may need 24 hours to travel to the site, or a specialty inspection tool must be ordered 12 hours ahead. The schema and Activity model had no mechanism to express this constraint. Every activity could theoretically start the instant its predecessors finished, even those requiring specialist mobilization.

### Design

`mobilization_lead_hours: float` is added to `Activity` (default 0.0). It represents the mandatory preparation window between the last predecessor finishing and this activity starting. The constraint is expressed as hours, consistent with all other time fields.

The value is **baked into the CPM ES** during `generateInfo()`, so all downstream metrics — slack, GRPW, priority weights, the scheduler's candidate-selection check — automatically respect it without any additional change.

**Forward pass change:**

| Scenario | Old formula | New formula |
|---|---|---|
| Source activity | `ES = 0` | `ES = mobilization_lead_hours` |
| Non-source activity | `ES = max(EF(u) + lag)` | `ES = max(EF(u) + lag + mobilization_lead_hours)` |

**Backward pass change:**

`LF(u) ≤ LS(v) − lag − v.mobilization_lead_hours`

This ensures the predecessor's latest finish accounts for the time the successor needs to mobilize, so slack values correctly identify the critical path.

**`_generate_info_from()` (replanning):**

For pending activities:
- Seed ES is `max(current_time_hours, mobilization_lead_hours)` — if the mobilization window has not yet closed (e.g. source with lead=8h and current_time=5h), the activity cannot start before the lead elapses even during a replan.
- EF is seeded as `ES + duration` (fixing a latent bug where source EF was seeded as `current_time` only, missing the duration component).

The forward and backward propagation loops in `_generate_info_from` apply the same `+ mobilization_lead_hours` extension as `generateInfo`.

**Scheduler enforcement:** implicit — the scheduler checks `abs_es > time`, where `abs_es = startTime + infoDict[act]['es']` and `infoDict[act]['es']` already includes the lead. No additional scheduler logic needed.

### What is NOT in scope

- **Resource-level mobilization lead**: the current implementation is activity-level. If the same vendor skill is used by multiple activities, each specifies its own lead independently. A future iteration could add `mobilization_lead_hours` to `ResourceAvailability` for a skill-type-level default.
- **Partial mobilization (call-in-advance)**: the model assumes the mobilization order is placed at the moment the last predecessor finishes. Proactive early-ordering (before predecessors complete) is not modelled.

### Backward compatibility

`mobilization_lead_hours = 0.0` by default. Existing JSON files, Pert constructions, and test networks produce identical CPM values.

### Unit tests added

26 new tests in `unit_tests/test_mobilization_lead.py` covering:
- `Activity` field default, `from_json` parse, `to_json_dict` round-trip (field present when nonzero, omitted when zero), `reset()` preserves the value
- CPM forward pass: source ES shifted by lead; successor ES = predecessor EF + lag + lead; lag + lead both applied; project duration extended
- CPM backward pass: predecessor LF tightened; critical path activities have zero slack; non-critical predecessor has positive slack
- Scheduler: activity starts only after predecessor EF + lead; schedule completes; zero lead is unchanged; lead adds to scheduled duration
- `_generate_info_from` with lead: pending source ES uses `max(current, lead)`; pending non-source ES includes lead from frozen predecessor; replan respects lead
- Schema: `mobilization_lead_hours` field present and typed correctly

---

## What remains (next iteration candidates)

From the assessment roadmap, the following items were **not** addressed in this iteration:

- ~~**Consumable resource tracking**~~ — resolved in Change 7 (pool-level dose budgets)
- ~~**Mid-run replanning with frozen activities**~~ — resolved in Change 9
- ~~**Regulatory time-window constraints**~~ — resolved in Change 8
- ~~**Skill substitution rules**~~ — resolved in Change 12 (`alternative_skill_types` per resource requirement; actual skill breakdown tracked for correct capacity accounting).
- **Scale / performance** — `_iter_hours` and `nx.descendants()` both O(n²); unverified at 1500-activity outage scale.
- ~~**Multi-mode activities**~~ — resolved in Change 11 (MMRCPSP with pre-scheduling mode injection).
- ~~**WBS-level priority roll-up**~~ — resolved in Change 13 (`wbs_group` label; `_compute_wbs_slack` propagates group minimum; weight function uses `min(individual_slack, wbs_slack)`).
- ~~**Mobilisation lead-time constraints**~~ — resolved in Change 10
- **Proactive robustness buffering** — `compute_fitness()` provides the signal; no buffer-insertion heuristic yet.
- **GP fitness evaluation loop** — `compute_fitness()` provides the signal; the outer GP training harness still needs to be built.

---

## Post-April-14 Fixes (April 15 iteration — see `outage_april_15.md` Part 5)

The following scope-boundary items from this changelog were addressed in the April 15 iteration:

| Item | Was | Resolution |
|---|---|---|
| Window-propagating CPM (Change 8 scope boundary) | Local ES/LF tightening only — upstream slack silently wrong | `_apply_time_windows(topo)` runs a second backward sweep in reverse-topo order after tightening; propagates tightened LF to all predecessors. `generateInfo()` and `_generate_info_from()` pass `topo`. |
| Multiple windows per activity (Change 8 scope boundary) | Single `(west, wlf)` pair only — missed first window = violation | `time_windows: list` field on `Activity`; `_resolve_windows()` helper in `Pert`; scheduler tries all windows in order, records violation only when all are exhausted; event queue seeded for every window open. Schema: `time_windows` array on task items. |
| Predecessor wiring for injected activities (Change 9 scope boundary) | Caller must mutate `forwardDict` manually before `replan()` | `_inject_activities(predecessor_wiring=dict)` and `replan(predecessor_wiring=dict)` accept `{new_task_id: [existing_pred_id, ...]}` and insert the required edges automatically. |

Test count after April 15 (Issues #1–#3): **694** (676 pre-fix + 18 new, 0 regressions).

| Item | Was | Resolution |
|---|---|---|
| O(n³) candidate scan (Issue #4) | `pred in self.completed` (list, O(n)) × all `self.wait` × all events | `_completed_set: set` (O(1) membership); `_ready: set` of predecessor-complete activities; `_rebuild_ready_set()` helper; `_select_candidate_activities` iterates `_ready` (~5–30 entries) instead of `self.wait` (~n/2). |

Test count after Issue #4: **711** (694 + 17 new, 0 regressions).

---

---

## Change 7 — Pool-level consumable radiation dose tracking (Challenge 2)

**Files:** `outage_data.py`, `activity.py`, `pert.py`, `outage_schema.json`, `unit_tests/test_dose_budget.py`  
**Category:** Medium priority / domain completeness

### Problem

Radiation dose is a consumable resource governed by 10 CFR 20 and plant ALARA goals: once a worker absorbs dose it cannot be returned. The scheduler had no mechanism to represent this — every skill type was implicitly renewable (capacity resets each period). A worker who had absorbed 1,800 mRem in a 2,000 mRem ALARA outage budget could still be scheduled on high-dose tasks with no constraint.

### Scope (this iteration)

Pool-level consumable budget tracking. The entire skill pool shares a single aggregate outage budget:

```
total_budget_mrem = dose_budget_per_worker_mrem × peak_available_count
```

Per-worker identity and individual dose ledgers are deferred — they require a worker-roster model and are a separate iteration.

### Design

**`DoseBudgetTracker`** (new class in `outage_data.py`):

| Method | Behaviour |
|--------|-----------|
| `fits(dose_rate, crew_count, duration_hours)` | Returns `True` if the task fits within remaining budget |
| `consume(dose_rate, crew_count, duration_hours)` | Permanently records dose drawn (irrevocable) |
| `reset()` | Clears consumed dose — called at the start of each scheduling run |
| `remaining_mrem` (property) | Budget remaining, clamped to 0 |

**`ResourceAvailability`** — two new optional fields:
- `resource_type: str` — `'renewable'` (default) or `'consumable'`
- `dose_budget_per_worker_mrem: float` — per-worker outage budget (mRem); only meaningful for consumable resources

**`ResourcePool`** — two new methods:
- `get_consumable_skills()` — list of skill types typed as consumable
- `build_dose_trackers()` — builds `{skill_type: DoseBudgetTracker}` for all consumable skills; empty dict if none

**`Activity`** — new field `dose_rate_mrem_per_hour: float` (default 0.0); parsed from JSON and round-tripped via `to_json_dict()` (key omitted when zero)

**`Pert`** — three integration points:
1. `__init__`: `self.dose_trackers = resource_pool.build_dose_trackers()` — zero cost when no consumable resources
2. `_reset_scheduling_state()`: `tracker.reset()` for each tracker — ensures clean slate between runs
3. `_fits_with_tentative()`: dose budget check added after location check — activity blocked if budget insufficient
4. `_update_activity_sets()`: `tracker.consume()` called when activity starts — dose is committed at task start, not completion

**`outage_schema.json`** — new fields:
- Resource items: `resource_type` (enum), `dose_budget_per_worker_mrem` (number ≥ 0)
- Task items: `dose_rate_mrem_per_hour` (number ≥ 0, optional)
- Also fixed `successors` items schema to allow both plain strings and `{task_id, lag_hours}` dicts (lag-format successors were accepted by the parser but rejected by the schema validator)

### What this does NOT cover (explicit scope boundary)
Individual worker identity / per-worker dose ledger — out of scope for this iteration
Dose accumulation across multiple outages (year-to-date tracking) — not in scope; would require external HR/dose system integration
ALARA optimization as a secondary objective in GP — compute_fitness() could gain a dose_utilization_ratio component in a later pass

### Backward compatibility

All new fields are optional with safe defaults (`resource_type='renewable'`, `dose_rate_mrem_per_hour=0.0`). Existing JSON files and Pert constructions produce `dose_trackers = {}` — the checks in `_fits_with_tentative` and `_update_activity_sets` are no-ops when the dict is empty.

### Collateral fix

`calculate_resource_requirement()` in `pert.py` lacked pool-existence guards on the inner resource/equipment loops (lines 661–669). If activities carry `required_resources` but `resource_pool` is `None` (valid for graph-only Pert objects), the method would crash with `AttributeError`. Added `if self.resource_pool:` and `if self.equipment_pool:` guards — the `rr` metrics default to 0.0 in that case.

### JSON usage example

```json
// resource with dose budget
{
  "skill_type": "RADIATION_WORKER",
  "resource_type": "consumable",
  "dose_budget_per_worker_mrem": 2000.0,
  "availability_periods": [...]
}

// task in a radiological zone
{
  "task_id": "T_ISI_NOZZLE",
  "duration": 8.0,
  "dose_rate_mrem_per_hour": 75.0,
  ...
}
```

### Unit tests added

32 new tests in `unit_tests/test_dose_budget.py` covering:
- `DoseBudgetTracker` arithmetic (`fits`, `consume`, `reset`, `remaining`)
- `ResourceAvailability` field storage and defaults
- `ResourcePool.from_json()` parsing + `build_dose_trackers()` budget calculation
- `Activity.from_json()` / `to_json_dict()` round-trip
- `Pert` tracker initialisation and no-pool / renewable-only cases
- Scheduler integration: dose consumed after run, reset between runs, zero-rate activities always pass, renewable pools unaffected

---

---

## Change 8 — Regulatory time-window constraints (Challenge 6)

**Files:** `activity.py`, `pert.py`, `outage_schema.json`, `unit_tests/test_time_windows.py`  
**Category:** High priority / domain correctness

### Problem

Surveillance tests and Technical Specification activities have hard calendar constraints of the form: "this task cannot start before T+72h and must be completed by T+120h from outage start." These are not precedence constraints (which shift when upstream activities shift) — they are absolute regulatory windows that do not move. The schema, Activity model, CPM pass, and scheduler had no mechanism to represent or enforce them.

### Design scope

Two new optional task fields, expressed as hours from outage start:
- `window_earliest_start_hours` — activity cannot start before this offset
- `window_latest_finish_hours` — activity must complete by this offset

Using hours-from-start rather than absolute datetimes keeps files portable if the outage start date changes (all windows shift together).

### Changes made

**`Activity`** — two new optional fields (both `None` by default):
- `window_earliest_start_hours: float | None`
- `window_latest_finish_hours: float | None`
- Parsed in `from_json()`, serialised in `to_json_dict()` (keys omitted when `None`)

**`Pert._apply_time_windows()`** (new method, called from `generateInfo()` after the CPM passes):
- For each activity with a window, tightens CPM ES/LF:
  - `ES = max(CPM_ES, window_earliest_start_hours)`
  - `LF = min(CPM_LF, window_latest_finish_hours)`
  - Slack recomputed as `LS − ES`
- Sets `infoDict[a]['window_infeasible'] = True` and logs a WARNING when the window is narrower than the activity duration (slack < 0)
- Non-windowed activities are untouched; `window_infeasible = False` is set for completeness

**`Pert._build_event_queue()`** — seeds an event at `startTime + window_earliest_start_hours` for each windowed activity so the scheduler wakes up exactly when the window opens

**`Pert._select_candidate_activities()`** — two new checks per activity:
- Too early (`current_hours < window_earliest_start_hours`): activity stays in `wait`, no delay accrued
- Window missed (`current_hours + duration > window_latest_finish_hours`): activity is removed from `wait` and a violation dict is appended to `self._window_violations`

**`Pert.calculateScheduleWithResources()`** — result dict gains `'window_violations'` key: a list of dicts with `{activity, reason, window_earliest_start_hours, window_latest_finish_hours, current_hours, duration_hours}` for each missed window

**`Pert.compute_fitness()`** — fourth component added:
- `window_violation_ratio = n_violations / n_real_activities`
- `composite += delta * window_violation_ratio` (default `delta=2.0` — high penalty)
- New return keys: `window_violation_ratio`, `n_window_violations`

**`outage_schema.json`** — two new optional fields on task items: `window_earliest_start_hours`, `window_latest_finish_hours`

### Scope boundaries (not implemented in this iteration)

- **Window-propagating CPM**: adjustments to windowed ES/LF are local; upstream/downstream CPM values are not updated. Full propagation requires an iterative backward pass with window clamps and is deferred.
- **Preemption**: activities run to completion once started; the scheduler cannot interrupt a task to respect a window.
- **Start-to-start lag interactions with windows**: not propagated.
- **Multiple windows per activity**: some surveillance tests can run in any of several discrete windows — one window pair per activity only.

### Backward compatibility

All new fields are optional with `None` defaults. Existing JSON files and Pert constructions produce no window constraints; `_apply_time_windows()`, `_select_candidate_activities()` window checks, and `compute_fitness()` delta term are all no-ops.

### Unit tests added

30 new tests in `unit_tests/test_time_windows.py` covering:
- Activity field storage, `from_json()` parse, `to_json_dict()` round-trip
- `_apply_time_windows()`: ES/LF tightening, non-loosening, slack recomputation, infeasibility flag
- Event queue seeded at window-open time
- Scheduler holds activity until window opens; activity starts no earlier than `west`
- Missed window recorded as violation; violation list reset between runs
- `compute_fitness()`: `window_violation_ratio`, `n_window_violations`, composite increases with violations

---

## Change 9 — Real-time replanning with frozen activity state (Challenge 4)

**Files:** `activity.py`, `pert.py`, `unit_tests/test_replan.py`  
**Category:** High priority / operational necessity

### Problem

Nuclear outage schedules encounter mid-execution disruptions that require immediate rescheduling: an inspection reveals unexpected damage at t=48h, a specialist crew is delayed, or a regulatory hold point reveals additional work. The scheduler had no concept of "current outage time" — calling `calculateScheduleWithResources()` at any point wiped all activity state and rescheduled from scratch. There was no way to:
- Freeze activities that had already started or completed
- Inject newly discovered tasks into the live graph
- Re-run CPM and scheduling only on the remaining work

### Design

Six new components added in sequence:

**`Activity.status: str`** (new field, default `'pending'`)

| Value | Set by |
|---|---|
| `'pending'` | `Activity.__init__()`, `Activity.reset()` |
| `'in_progress'` | `_update_activity_sets()` when activity starts |
| `'completed'` | `_update_ongoing_list()` when activity finishes |

**`Pert._partial_reset(current_time_hours)`** — classifies every activity by actual timing relative to `current_time_hours`:
- `endTime ≤ current_abs` → **completed**: preserved entirely; dose re-consumed from tracker
- `startTime ≤ current_abs < endTime` → **in-progress**: start frozen; `act._remaining_duration` computed; dose re-consumed
- not yet started → **pending**: `act.reset()` called; added to `self.wait`

Window violations accumulated before the replan are **not cleared** — they are historical facts.

**`Pert._inject_activities(new_activities)`** — inserts new Activity objects into the live graph:
- Resolves successor names via `task_to_activity` (auto-synced from `forwardDict` for graph-built Pert objects)
- Extends `forwardDict` and `lag_dict`; rebuilds `backwardDict` / `infoDict` / `nxgraph`
- Skips duplicates and unknown successors with WARNINGs

**`Pert._generate_info_from(current_time_hours)`** — partial CPM pass that respects frozen state:
- Completed / in-progress: ES/EF fixed from actual times (or `current_time + remaining`)
- Pending: ES floored at `current_time_hours`; forward pass propagates normally from frozen anchors
- `_apply_time_windows()` runs afterward

**`Pert._build_event_queue_from(current_time_hours)`** — event heap anchored at `current_time_hours`, seeding in-progress completions, availability boundaries, pending ES times, window-open times, and shift boundaries

**`Pert.calculateScheduleWithResources_from(current_time_hours, sgs, ...)`** — identical event-driven loop to `calculateScheduleWithResources` but without `_reset_scheduling_state()` and with pre-populated `completed` / `ongoing` lists; adds `'replan_time_hours'` to the result dict

**`Pert.replan(current_time_hours, new_activities=None, sgs=...)`** — public entry point orchestrating inject → partial_reset → generate_info_from → calculateScheduleWithResources_from.  Raises `RuntimeError` if called before any scheduling run.

### Collateral fix

`_inject_activities` now pre-syncs `task_to_activity` from `forwardDict.keys()` via `setdefault()` before resolving successor names, so injection works correctly for graph-built Pert objects (which don't populate `task_to_activity`).

### What is NOT in scope for this iteration

- **Saving / loading mid-run state to disk** — replanning is in-memory only
- **Re-optimising already-started activity resource assignments** — in-progress activities are frozen in their entirety
- **Cascading window constraint propagation for injected activities** — same deferred limitation as Challenge 6
- **Undo / rollback** — caller must snapshot the Pert before calling `replan()` if needed
- **Predecessor wiring from existing → new activities** — caller must update existing activities' `childs` lists manually before calling `replan()`

### Backward compatibility

`calculateScheduleWithResources()` and `compute_fitness()` unchanged. `Activity.status` defaults to `'pending'` and `reset()` always restores it.

### Unit tests added

33 new tests in `unit_tests/test_replan.py` covering `Activity.status` lifecycle, `_partial_reset` classification and dose replay, `_inject_activities` graph wiring, `_generate_info_from` CPM anchoring, and end-to-end `replan()` scenarios including injection.

---

## Document Assessment — `outage_april_3.docx` vs current code (April 14)

Cross-referenced each claim in the document against the current code after the April 14 iteration.

### Claims resolved by April 14 iteration

| Document item | Was accurate when written | Resolved by |
|---|---|---|
| Challenge 1 / 8 — lag relationships | Yes | Change 2 above |
| Challenge 12 — shift calendar never used | Yes | Change 3 above |
| Challenge 5 — fitness function absent | Yes | Change 4 above |
| Additional Finding — `_weight_function` cliff | Yes | Change 1 above |
| Additional Finding — `addActivity()` infoDict gap | Yes (real `nxgraph` issue) | Change 6 above |
| Additional Finding — MDKnapsack ignores location | Yes | Change 5 above |

### Claims that are factually incorrect against the current code

**1. `irsm`/`wcs`/`acs` raise `IOError("Not yet implemented!")`**

This is incorrect. All three rules are fully implemented in `priority_calculation()` following Kolisch (1996) formulations:
- `wcs` (Worst Case Slack, eq. 19) — implemented at `pert.py:3033`
- `acs` (Average Case Slack, eq. 23) — implemented at `pert.py:3053`
- `irsm` (Improved Resource Scheduling Method, eq. 14) — implemented at `pert.py:3072`

The only `IOError` in the function is the `else` catch-all for truly unknown rule names. The document appears to have confused this with method stubs.

**2. `print()` calls in `calculateScheduleWithResources()` at lines 1264–1272**

No bare `print()` calls exist in that function. The entire scheduling loop uses `logger.debug()` throughout. This finding does not apply to the current code.

### Challenges that remain valid and unresolved

| Document challenge | Status | Summary |
|---|---|---|
| Challenge 2 | **Resolved** (Change 7) | Pool-level consumable dose tracking |
| Challenge 4 | **Resolved** (Change 9) | Real-time replanning with frozen-activity state |
| Challenge 6 | **Resolved** (Change 8) | Regulatory time-window constraints |
| Challenge 7 | Remaining | Scale — `_iter_hours` and `nx.descendants()` both O(n²), unverified at 1500-activity outage |
| Challenge 9 | **Resolved** (Change 14) | CCPM proactive robustness — Project Buffer + Feeding Buffers; `_size_buffer`, `insert_project_buffer`, `insert_feeding_buffers`, `get_buffer_status` |
| Challenge 10 | **Resolved** (Change 12) | Multi-skill substitution — `alternative_skill_types` per requirement; actual breakdown tracked for capacity and dose |
| Challenge 11 | **Resolved** (Change 13) | WBS priority roll-up — `wbs_group` + `_compute_wbs_slack`; hierarchical CPM deferred |
| Challenge 13 | **Resolved** (Change 10) | Mobilization lead time baked into CPM ES; backward pass tightened |
| Challenge 15 | **Resolved** (Change 11) | Multi-mode activities — `Activity.set_mode()` + `Pert.set_modes()` with CPM recompute |

---

## Change 11 — Multi-mode activities / MMRCPSP (Challenge 15)

**Files:** `activity.py`, `pert.py`, `outage_schema.json`, `unit_tests/test_multimode.py`  
**Category:** Medium priority / domain completeness

### Problem

The scheduler assumed every activity had exactly one duration, one resource requirement, and one equipment requirement. Real outage work packages have alternative execution modes: a weld inspection can be done in 8 h with 2 inspectors (normal) or in 4 h with 4 inspectors (crash). A reactor head lift can use a polar crane (fast, requires dedicated crane outage) or a hydraulic jack (slow, no crane dependency). The RCPSP became MMRCPSP — Multi-Mode — without any mechanism to express or select modes.

### Design

**Pre-scheduling mode injection** — modes are selected before `calculateScheduleWithResources()` runs, not during the scheduling loop. This keeps the RCPSP loop unchanged and lets the GP evolve mode assignment vectors as a separate combinatorial layer above the scheduler.

The design parallels `set_priorities()` / `set_durations()`: the GP calls `set_modes()` → the scheduler sees a fully configured single-mode network.

### Changes made

**`Activity`** — three new fields:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `modes` | `list[dict]` | `[]` | Available execution modes; empty = single implicit mode |
| `selected_mode_id` | `str \| None` | `None` | Mode currently active |

New methods:

| Method | Behaviour |
|---|---|
| `set_mode(mode_id)` | Writes the named mode's `duration`, `required_resources`, `required_equipment`, and (if present) `dose_rate_mrem_per_hour` / `mobilization_lead_hours` into the activity's live fields. Raises `ValueError` if activity has no modes or mode_id is unknown. |
| `get_available_modes()` | Returns the list of mode IDs defined for this activity. |

Each mode dict carries:

```json
{
  "mode_id": "crash",
  "duration": 4.0,
  "required_resources": [{"skill_type": "MECHANIC", "crew_count": 4}],
  "required_equipment": [],
  "dose_rate_mrem_per_hour": 120.0,    // optional — inherit task-level if absent
  "mobilization_lead_hours": 12.0      // optional — inherit task-level if absent
}
```

`from_json()` parses the `modes` array from the task dict (empty list if absent).  
`to_json_dict()` serialises `modes` when non-empty (omitted when empty, backward-compatible).  
`reset()` does **not** clear `modes` or `selected_mode_id` — they are structural data.

**`Pert.set_modes(mode_assignments: dict)`** — batch mode setter:

1. Pre-syncs `task_to_activity` from `forwardDict.keys()` (handles graph-built Pert objects that don't populate `task_to_activity`).
2. Calls `activity.set_mode(mode_id)` for each entry.
3. Calls `_sync_infodict_durations()` to push updated durations into `infoDict`.
4. Calls `generateInfo()` to recompute ES, EF, LS, LF, slack for the full network.

**`outage_schema.json`** — new optional `modes` array on task items. Each mode item requires `mode_id`, `duration`, `required_resources`, `required_equipment`; optional `dose_rate_mrem_per_hour` and `mobilization_lead_hours` override the task-level values when present.

### GP usage pattern

```python
# GP evolves a mode assignment vector for each iteration
mode_assignment = {'T_WELD_INSPECT': 'crash', 'T_HEAD_LIFT': 'normal', ...}

pert.set_modes(mode_assignment)          # apply + recompute CPM
pert.calculateScheduleWithResources(sgs='max_use_res_ranked')
fitness = pert.compute_fitness()         # evaluate
```

### What is NOT in scope

- **Mode-dependent precedence constraints**: some crash modes may introduce or remove predecessor links (e.g. adding a staging task). This would require re-wiring the graph, not just field overrides. Deferred.
- **Continuous-mode activities**: the current model supports a discrete set of named modes. Parametric duration/cost tradeoffs (e.g. duration = f(crew_count)) are not modelled.
- **In-flight mode switching**: modes are selected before `calculateScheduleWithResources()`. Switching a mode mid-schedule (e.g. after a delay makes the normal mode infeasible within its window) is not supported — use `replan()` with a new mode assignment.
- **Mode-dependent lag values**: lags are stored on the predecessor activity, not per-mode. If a crash mode requires a different lag, the caller must update `activity.successor_lags` manually.
- **Automatic mode selection by the scheduler**: the scheduler does not search over modes — it executes whatever mode is currently active. Mode selection is entirely the caller's (GP's) responsibility.

### Backward compatibility

`Activity.modes = []` by default. Existing JSON files, graph-built Pert objects, and test networks that never call `set_mode()` or `set_modes()` are completely unaffected. All new fields are absent from `to_json_dict()` output when empty/None.

### Unit tests added

53 new tests in `unit_tests/test_multimode.py` covering:

- `Activity` field defaults, `set_mode()` application (duration, resources, equipment, optional dose/lead), error paths (no modes, unknown mode_id), `get_available_modes()`, `reset()` preservation
- `from_json()` parse, empty/absent modes, `to_json_dict()` serialisation, round-trip
- `Pert.set_modes()`: valid application, CPM recomputation (crash shorter than normal), partial assignment, empty dict, infoDict ES update, resource requirements updated
- Error paths: non-dict input, unknown task_id, unknown mode_id
- CPM correctness: zero slack on critical path after crash / normal mode; slack changes correctly after mode switch
- Scheduler: crash finishes faster than normal; resource contention under crash mode correctly deadlocks (confirms resource requirements are applied)
- Schema: `modes` field present, array type, mode item has all required and optional fields, not in task `required` list

---

## Change 12 — Multi-skill substitution (Challenge 10)

**Files:** `activity.py`, `pert.py`, `outage_schema.json`, `unit_tests/test_substitution.py`  
**Category:** Medium priority / domain correctness

### Problem

The scheduler produced artificial infeasibilities when cross-trained workers could have covered a requirement. A task needing 2 `WELDER`s would stall even when 3 `WELDER_SENIOR` workers — who are qualified to perform the same work — were idle. Plant outage schedules rely heavily on cross-training; ignoring it causes the optimizer to under-utilize available labour and inflate the critical path.

### Design

Substitution is expressed **per resource requirement**, not globally. Each requirement dict gains an optional `alternative_skill_types: list[str]` field whose entries are tried in declared order when the primary pool is exhausted:

```json
"required_resources": [
  {
    "skill_type": "WELDER",
    "crew_count": 2,
    "alternative_skill_types": ["WELDER_SENIOR", "MECHANIC_CERTIFIED"]
  }
]
```

**Worker assignment is resolved once at activity start time** (first hour of the activity window) and held constant for the full duration. This is physically correct: a worker assigned to a task does not switch roles mid-activity.

The resolved assignment is stored on the activity as `_actual_resources: dict` (maps `skill_type → workers_actually_assigned`) so that subsequent capacity checks account for which skills are truly in use, not just which skills were declared.

### Changes made

**`Activity`** — one new field:

| Field | Type | Default | Reset? |
|---|---|---|---|
| `_actual_resources` | `dict \| None` | `None` | Yes (`reset()`) |

**`Pert._fits_with_tentative`** — resource block extended: when primary capacity is insufficient for a requirement at any hour, alternatives are checked in order; if primary + alternatives can collectively satisfy `crew_count`, the check passes.

**`Pert._apply_tentative`** — resource block replaced: the worker assignment is resolved at the first hour (primary first, alternatives fill any shortfall), stored as `activity._actual_resources_for_start`, then the same breakdown is subtracted from `res_rem` for every hour of the activity window (no per-hour re-resolution — workers don't change roles mid-task).

**`Pert._update_activity_sets`** — after the activity starts:
1. `_actual_resources_for_start` is committed to `act._actual_resources`.
2. Dose consumption uses `_actual_resources` when set, so dose is charged to the skills that are *actually* working, not the declared primary skill.

**`Pert._get_consumed_resources`** — uses `act._actual_resources` when set. This ensures that an ongoing activity using WELDER_SENIOR as a substitute for WELDER correctly reduces the available WELDER_SENIOR count (not the WELDER count) for subsequent scheduling steps.

**`outage_schema.json`** — `alternative_skill_types` added as an optional property on resource requirement items. `additionalProperties: false` is preserved.

### Example (JSON)

```json
// Two WELDER workers needed; if fewer than 2 are available, draw from WELDER_SENIOR first
{
  "task_id": "T_WELD_NOZZLE",
  "required_resources": [
    {
      "skill_type": "WELDER",
      "crew_count": 2,
      "alternative_skill_types": ["WELDER_SENIOR"]
    }
  ],
  ...
}
```

### What is NOT in scope

- **`MDKnapsackScheduler` substitution**: the greedy knapsack pre-selector uses the declared primary `skill_type` dimension only. Activities that would only fit via substitution may be wrongly filtered out by the knapsack pass; they are then re-evaluated by `_fits_with_tentative` during the subsequent validation sweep, so correctness is maintained — but knapsack solution quality is slightly degraded when substitution is common. Full knapsack integration would require a merged-capacity dimension per equivalence group.
- **Global skill equivalence groups**: alternatives are declared per-requirement. There is no pool-level "these skills are equivalent" table. If 50 tasks all substitute WELDER_SENIOR for WELDER, each must list it individually.
- **Dose budget for alternative consumable skills in the feasibility check**: `_fits_with_tentative`'s dose check verifies the primary skill's tracker. If the primary skill has no dose budget issue but the alternative does (or vice versa), the fit check may be optimistic. Dose *consumption* at start time IS charged to the actual skill via `_actual_resources`, so the budget is correctly depleted — but a future task could be scheduled against an over-committed alternative dose budget. This corner case requires a substitution-aware dose fit check.
- **Partial substitution across shifts**: if worker availability changes between hours within the activity window, the assignment computed at the first hour may assign more alternative workers than are available in a later hour. The capacity snapshot will protect against this being scheduled if capacity is insufficient at any hour — but the `_actual_resources` record uses first-hour counts.

### Backward compatibility

`alternative_skill_types` is absent from all existing requirement dicts. `req.get('alternative_skill_types', [])` returns `[]` — the old behaviour is fully preserved. `_actual_resources = None` by default; `_get_consumed_resources` falls back to the original `required_resources` loop when it is `None`.

### Unit tests added

30 new tests in `unit_tests/test_substitution.py` covering:

- `Activity._actual_resources` default, `reset()` clears it, independence across instances
- Primary sufficient: no substitution triggered; activity scheduled normally
- Primary exhausted, no alternative: activity stalls (correctly)
- Alternative fills gap: activity scheduled using substitute workers
- Partial primary + alternative: correct split when partial primary available
- Multiple alternatives tried in order: first alternative tried first
- Two alternatives split a shortfall
- Combined primary + alternatives still insufficient: correctly stalls
- No `alternative_skill_types` field: backward-compatible
- Per-hour check: every hour in the window must be coverable
- `_actual_resources` set after scheduling
- Full primary case: `_actual_resources` reflects primary only
- Substitution case: `_actual_resources` reflects the split
- `_get_consumed_resources` returns correct primary count post-start
- `_get_consumed_resources` returns correct alternative count post-start
- `_actual_resources` reset between runs
- Dose charged to alternative skill after substitution
- Dose charged proportionally when split across primary and alternative
- Zero dose rate with substitution: no dose charged
- Schema: `alternative_skill_types` present, array type, string items, unique items, not required, `additionalProperties: false` intact, description mentions substitute

---

## Change 13 — WBS-level priority roll-up (Challenge 11)

**Files:** `activity.py`, `pert.py`, `outage_schema.json`, `unit_tests/test_wbs_priority.py`  
**Category:** Medium priority / scheduling quality

### Problem

The assessment identified that when a WBS package's aggregate float reaches zero, only the tasks individually on the critical path received elevated priority; other tasks in the same package continued to be scheduled at their own (higher) float. This caused the scheduler to idle cross-trained workers on routine low-urgency tasks while the package's tightest tasks were stalling.

Concrete example:

| Task | Duration | Individual slack | Reality |
|---|---|---|---|
| A — Remove seal | 4 h | **8 h** | Must finish before system restart |
| B — Install seal | 6 h | 2 h | On package critical path |
| C — Leak test | 2 h | **8 h** | Must follow A, precedes B |

A and C show 8 h of individual float and would be deprioritised. But the package must complete before the system restart deadline — the aggregate package float is already 0. Without WBS roll-up, the scheduler spends 8 h doing other work before touching A and C, pushing B (and the whole package) late.

### Design

**`wbs_group: str | None`** — a flat package identifier added to `Activity`. All activities with the same label form one scheduling unit.

**`Pert._compute_wbs_slack()`** — called at the end of both `generateInfo()` and `_generate_info_from()` (replanning). For each `wbs_group`:

```
group_min_slack = min(slack for all members)
```

Every member receives `infoDict[act]['wbs_slack'] = group_min_slack`. Activities with no group get `wbs_slack = slack`.

**`_select_candidate_activities`** — the `TF_based` weight function line changes from:

```python
_weight_function(act.slack, proj_dur)
```

to:

```python
_weight_function(min(act.slack, act.wbs_slack), proj_dur)
```

When any one member hits zero float, `group_min_slack = 0`, and every member's effective scheduling priority is elevated to maximum simultaneously.

### Collateral fix

`set_durations()` was missing the `task_to_activity` pre-sync that `set_modes()` and `_inject_activities()` already perform. Graph-built Pert objects don't populate `task_to_activity` during construction, so calling `set_durations({'D': 8.0})` on a graph-built Pert raised `KeyError`. Added `setdefault()` sync at the entry of `set_durations()`.

### What is NOT in scope

- **Hierarchical CPM** — re-computing a parent WBS element's duration from its sub-network's critical path is a separate architectural change. `addSubActivities()` still uses serial composition. The WBS roll-up here is purely a priority mechanism: it does not change ES/EF/LS/LF values.
- **Multi-level WBS** — only flat group membership is tracked. A `PKG1` that is itself a member of `SYSTEM_A` would need a second level of roll-up, which is not implemented.
- **WBS reporting in the output dict** — `calculateScheduleWithResources()` result does not include a per-group float summary. The `infoDict['wbs_slack']` value is accessible for post-processing if needed.
- **Dynamic group formation** — `wbs_group` is static (set at construction / from JSON). Groups cannot be defined or modified during a scheduling run.

### Backward compatibility

`wbs_group = None` by default. `_compute_wbs_slack()` is a no-op for activities with no group (they get `wbs_slack = slack`). The weight function `min(slack, wbs_slack)` degrades to `slack` when `wbs_slack = slack`. All existing tests pass unchanged.

### JSON usage

```json
{
  "task_id": "T_REMOVE_SEAL",
  "wbs_group": "PKG_RCP_SEAL_REPLACEMENT",
  ...
},
{
  "task_id": "T_INSTALL_SEAL",
  "wbs_group": "PKG_RCP_SEAL_REPLACEMENT",
  ...
}
```

### Unit tests added

26 new tests in `unit_tests/test_wbs_priority.py` covering:

- `Activity.wbs_group` default, parse, serialise, round-trip, null, `reset()` preserves
- `_compute_wbs_slack`: ungrouped activities retain individual slack; group minimum propagates to all members; multiple groups are independent; single-member group works; all-critical group stays zero; `set_durations` triggers recomputation
- Priority elevation: grouped member with high individual slack gets weight based on group min (near 1.0 when group is critical); ungrouped weight is based on individual slack only; group collapse lifts all members in scheduler run
- Replanning: `_generate_info_from` computes `wbs_slack`; group minimum respected after partial freeze
- Schema: `wbs_group` present, allows string and null, not required

---

## Change 14 — CCPM proactive robustness buffering (Challenge 9)

**Files:** `activity.py`, `pert.py`, `unit_tests/test_buffer.py`  
**Category:** Medium priority / schedule robustness

### Problem

The scheduler produced a schedule with zero explicit schedule reserve. Any single activity delay on the critical chain immediately extended the project finish date. Real nuclear outage schedules are built with explicit time buffers at the end of the critical path and at points where non-critical paths merge into the critical path. Without buffers, the planner had no early-warning signal for chain degradation — a 2-hour slip on one chain activity looked identical to a 2-hour slip anywhere else.

### Design

Implemented the Critical Chain Project Management (CCPM) buffer approach:

**Project Buffer (PB)** — one buffer placed at the end of the resource-constrained critical chain. It absorbs accumulated delay from anywhere on the chain so that individual task slippages do not automatically push the project finish. The chain terminal's original successors become PB's successors.

**Feeding Buffers (FB)** — one buffer per merge point where a non-chain "feeding" path joins the critical chain. Each FB is inserted between all non-chain predecessors of the merge-point activity and the merge activity itself. This protects the critical chain from delays originating on feeding paths.

**Buffer sizing** supports two methods controlled by the caller:
- `'half'` (cut-and-paste method): `fraction × Σ(durations)` — simple, conservative
- `'ssq'` (default, statistically grounded): `√(Σ((dᵢ × fraction)²))` — approximates the half-normal distribution of duration uncertainty; produces a smaller buffer than `'half'` for networks with many activities (the uncertainties partially cancel)

**Buffer status reporting** — `get_buffer_status()` measures how much of each buffer has been consumed by upstream delays: `consumed = max(0, actual_start − CPM_ES)`. Returns `consumed_hours`, `size_hours`, and `utilization_pct` for monitoring.

### Changes made

**`activity.py`**

Added `buffer_type: str | None = None` attribute. Values: `'project'`, `'feeding'`, or `None` (real task). The attribute is structural — `reset()` preserves it so graph-surgery methods can be called repeatedly without losing buffer identity.

**`pert.py`**

Five new methods added in the `PROACTIVE ROBUSTNESS BUFFERING (CCPM)` section:

| Method | Description |
|---|---|
| `_size_buffer(durations, method, fraction)` | Static helper; computes buffer size from a list of activity durations using `'half'` or `'ssq'` method |
| `_splice_buffer_activity(buffer_act, predecessors, successors)` | Inserts a buffer Activity into the graph: removes direct pred→succ edges, wires pred→buffer and buffer→succ, rebuilds `nxgraph` + CPM |
| `insert_project_buffer(method, fraction)` | Creates and splices a PB after the critical chain terminal; idempotent; requires `calculateScheduleWithResources()` to have run |
| `insert_feeding_buffers(method, fraction)` | Creates and splices an FB at every merge point on the critical chain; returns list of inserted FB activities; idempotent |
| `get_buffer_status()` | Returns a dict keyed by buffer name with `buffer_type`, `size_hours`, `cpm_start_hours`, `actual_start_hours`, `consumed_hours`, `utilization_pct` |

`compute_fitness()` updated to exclude buffer activities (`getattr(a, 'buffer_type', None) is None`) from the real-task count so that buffer insertion does not inflate `criticality_ratio`.

`insert_project_buffer` and `insert_feeding_buffers` use `getattr(self, 'constrained_chain_list', None)` guard so they raise `RuntimeError` correctly even on Pert objects constructed without scheduling.

### What is NOT in scope

- **Dynamic buffer replenishment** — buffers are sized once at insertion time, not re-sized as the schedule evolves.
- **Buffer fever acceleration** — CCPM "fever chart" milestones are not implemented; `get_buffer_status()` gives raw utilization only.
- **Multi-project shared buffers** — each Pert instance manages its own buffers independently.
- **Buffer activities in JSON round-trip** — buffers are runtime-generated and not serialised to the outage JSON schema; they must be re-inserted after loading from JSON.
- **Automatic chain computation without scheduling** — `insert_project_buffer` and `insert_feeding_buffers` still require a prior `calculateScheduleWithResources()` call to populate `constrained_chain_list`.
- **Buffer size optimisation** — sizing is heuristic (`'half'` or `'ssq'`); no solver-based minimisation of buffer size subject to a reliability target.

### Backward compatibility

All existing schedules are unaffected. Buffer insertion is opt-in: nothing calls `insert_project_buffer` or `insert_feeding_buffers` unless the caller does so explicitly. `compute_fitness()` changes are invisible for schedules with no buffers (the filter matches no activities).

### Unit tests added

49 new tests in `unit_tests/test_buffer.py` covering:

- `_size_buffer`: `'half'` and `'ssq'` methods, single and multiple durations, empty input, unknown method raises
- `Activity.buffer_type`: default `None`, explicit `'project'` and `'feeding'`, `reset()` preserves value
- `insert_project_buffer`: raises before scheduling, PB activity created, `buffer_type='project'`, correct size with both methods, positive duration, present in `forwardDict` and `infoDict`, predecessor is chain terminal, idempotent
- `insert_feeding_buffers`: raises before scheduling, empty list on linear chain, FB created at merge point, `buffer_type='feeding'`, present in `forwardDict` and `infoDict`, successor is merge activity, predecessor is non-chain terminal, positive duration, idempotent
- `get_buffer_status`: empty when `startTime` not set, empty before buffers inserted, status dict has required keys, `buffer_type` correct, `size_hours` matches duration, `consumed_hours ≥ 0`, `utilization_pct` in [0, 100], both PB and FBs appear in status
- `compute_fitness`: `criticality_ratio` unchanged after PB insertion, fitness returns finite values after insertion
