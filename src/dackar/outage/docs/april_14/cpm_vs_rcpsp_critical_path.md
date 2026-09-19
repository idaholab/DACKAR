# CPM Critical Path vs RCPSP Critical Chain

## The distinction

The **CPM critical path** is purely graph-structural. It is the longest path
through the dependency DAG where every activity has zero total float and
consecutive EF/ES values align. It only sees precedence constraints. Remove all
resource limits and the CPM path gives the theoretical minimum project duration.

The **RCPSP critical chain** is different. When resources are constrained, an
activity can be delayed not because a predecessor has not finished, but because
the required crew or equipment is unavailable. That delay propagates forward in
time even though there is no dependency edge connecting the activities involved.
Two activities in completely separate branches of the DAG can form a de-facto
critical sequence purely because they compete for the same pool. The critical
chain therefore spans activities that the dependency graph says are independent.

Concrete example: activity A (MECH, branch 1) and activity B (MECH, branch 2)
are graph-independent. If the MECH pool is tight, B waits while A runs, and
everything downstream of B is pushed out. A → B becomes a critical sequence in
the actual schedule even though no edge exists between them. The resource-
constrained critical chain is longer than the CPM path and passes through
activities across disconnected subgraphs.

Practical implication for GP/GA optimisation: optimising mode assignments or
activity ordering based purely on the CPM path can miss the actual makespan
driver, which may be resource contention between graph-independent activities.
The fitness function must be evaluated against the full RCPSP schedule.

---

## What the Pert class computes

The Pert class maintains two distinct critical-path concepts.

### 1. CPM critical path — `getCriticalPath()`

Pure graph analysis, run after `generateInfo()`. Finds all paths through the
dependency DAG where every activity has zero slack and consecutive EF/ES values
are aligned. Returns `List[Activity]` (longest path) or all paths. Stored per
activity as `act.belongsToCP`. This is the **pre-schedule, resource-free** bound.

### 2. Resource-constrained critical chain — `constrained_chain_list`

Computed automatically at the end of each `calculateScheduleWithResources()` call
via two post-schedule steps:

**Step 1 — `_build_augmented_graph()`**

Starts from the original precedence edges (`forwardDict`) and adds
*resource-flow arcs* induced by capacity constraints:

- **Crew / equipment binding arcs**: for every pair of overlapping scheduled
  activities that share a skill or equipment type AND whose combined demand
  saturates availability during the overlap, an arc is added from the
  earlier-start activity to the later-start activity.
- **Location serialisation arcs**: for locations with `max_concurrent_tasks == 1`,
  consecutive overlapping activities in that location get a binding arc.

These arcs encode resource contention as explicit directed edges between
activities that have no precedence relationship, producing an augmented DAG that
captures both logical and resource-induced sequencing.

**Step 2 — `_longest_path_in_augmented()`**

Runs a topological-sort DP (Kahn + longest-path) over the augmented DAG.
The longest path is stored as `pert.constrained_chain_list` (List[Activity])
and `pert.constrained_chain_set` (Set[Activity]) for O(1) membership tests.

**Related: `_compute_actual_tf_proxy()`**

Computes `TF_actual(a) = s_min(a) - EF_actual(a)` for every activity using
augmented-graph successors. Activities with `TF_actual ≈ 0` form
`actual_zero_tf_set` — the set of all activities with no slack in the actual
schedule. The constrained chain is the single longest path through this zero-
float subgraph; `actual_zero_tf_set` is the broader set that may include
multiple parallel zero-float paths.

---

## Limitations of the augmented-graph approach

The `constrained_chain_list` is a meaningful and useful approximation of the
true RCPSP critical chain, but it is not exact:

1. **Binding arc criterion requires full saturation.** An arc is only added when
   `combined_demand >= availability`. Partial saturation (e.g., 3 MECH needed
   but only 2 free, causing a queue) can delay activities without triggering an
   arc. Binding arcs are conservative — they may miss some resource-induced
   dependencies.

2. **Location arcs only for `max_tasks == 1`.** Locations with
   `max_concurrent_tasks >= 2` can still create contention but produce no binding
   arcs in the current implementation.

3. **Post-hoc, not causal.** The augmented graph is built from the schedule that
   was already produced. It identifies what drove the makespan in that particular
   schedule, but a different SGS or mode assignment could produce a different
   augmented graph and a different critical chain. The chain is descriptive, not
   prescriptive.

4. **True RCPSP critical path is NP-hard to compute exactly.** A provably exact
   resource-constrained critical path requires enumerating all resource
   feasibility conditions — computationally intractable at outage scale. The
   augmented-graph approximation is the practical alternative.

---

## Summary

| | CPM critical path | RCPSP critical chain |
|---|---|---|
| Method | `getCriticalPath()` | `constrained_chain_list` |
| When computed | After `generateInfo()` | After `calculateScheduleWithResources()` |
| Edges used | Precedence only | Precedence + resource-flow arcs |
| Activities | Always graph-connected | May span disconnected subgraphs |
| Nature | Exact, graph-structural | Approximation via augmented DAG |
| Use | CPM duration bound, slack analysis | Schedule quality, GP/GA fitness insight |
