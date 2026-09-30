# Outage Planning — Gap Analysis

Assessment of the LOGOS CPM/RCPSP engine against real nuclear outage planning
requirements. Written April 2026.

---

## What is genuinely strong

The constraint coverage already exceeds most commercial tools: crew pools,
equipment, locations, dose budgets, system-state locking, hold points,
consumables, lag constraints, multi-window time constraints, shift calendars,
and real-time replanning. O(n) scheduling performance at 15K activities is
production-ready. Multi-mode RCPSP gives the GP/GA optimizer its decision
variable (mode assignment per activity). This is a serious foundation.

---

## Gaps that matter most to a planner

### 1. Probabilistic durations — the biggest missing piece

Every real outage plan is built around uncertainty. Activities have optimistic /
most-likely / pessimistic durations (PERT distributions). Discovery scope —
finding unexpected degradation when a valve is opened or a component pulled —
can add 20–200 hours to a critical path with no warning. The engine currently
treats durations as deterministic. Without Monte Carlo over duration
distributions it is impossible to answer the questions planners actually ask:

- "What is the P80 outage duration?"
- "Which activities carry the most schedule risk?"
- "How much float do we really have on this chain?"

### 2. Crew continuity and fatigue rules

Real outage crews work 12-hour shifts with regulatory limits on consecutive
hours and weekly totals. Beyond compliance, a crew that starts a job should
finish it — breaking continuity on a complex valve overhaul causes rework and
quality issues. The shift calendar handles time windows but does not model
individual crew rotation, fatigue accumulation, or continuity constraints. The
resource pool treats crew as a fungible count, not as named individuals with
work history.

### 3. Cost dimension

Crashing — adding crew to compress a critical activity's duration — is the
central trade-off in outage management. Multi-mode RCPSP has the mechanism
(crash mode = more crew, shorter duration) but the cost of each mode is absent
from the model. A planner cannot answer "what does it cost to recover 8 hours?"
without a cost per mode per crew-hour.

### 4. Baseline vs. actual tracking

Real outage management requires a frozen baseline schedule that variance is
measured against: schedule performance index, earned value, float consumption
rate. The replan capability updates the live schedule but there is no concept of
a locked baseline. A planner has no way to know whether the current schedule is
ahead or behind the plan approved at outage entry.

### 5. Work authorization and permit lead times

In a nuclear plant, before any activity can start it needs a radiation work
permit, a work order release, and sometimes a confined space or hot work permit.
Each has a preparation and approval lead time (typically 2–24 hours). These
constraints routinely delay activities that are otherwise resource-ready. The
system-state pool partially captures some of this but the permit-queue workflow
— preparation time, approval chain, re-issue on scope change — is not modeled.

### 6. Scope change management

Discovery scope is routine, not exceptional. When a technician opens a heat
exchanger and finds unexpected fouling, new activities need to be injected with
proper predecessor/successor wiring, resource assignment, and immediate CPM
recalculation. `_inject_activities` exists as a low-level primitive but there
is no workflow for scope change approval, impact assessment, or communication to
crews already working downstream.

---

## Gaps that matter for GP/GA specifically

The optimizer has a fitness function and the mode-assignment API. What is not
yet present:

- A **chromosome representation** and population management layer
- **Constraint repair operators** — a random mode assignment may violate dose
  budgets or system-state conflicts; the optimizer needs repair heuristics, not
  just rejection
- **Warm-start from the CPM/RCPSP solution** — initialising the population from
  the deterministic schedule rather than random assignments would dramatically
  reduce convergence time
- The **approximate RCPSP critical chain** (augmented-graph binding arcs) may
  not be sharp enough as a fitness signal; activities near resource contention
  but below the binding-arc threshold are invisible to the chain

---

## Priority order

| Priority | Gap | Why |
|---|---|---|
| 1 | Probabilistic durations + Monte Carlo | Planners live in probability space |
| 2 | Cost per mode | Crashing decisions need a cost axis |
| 3 | Baseline locking + variance tracking | Management requires schedule adherence metrics |
| 4 | Crew continuity + fatigue | Quality and regulatory requirement |
| 5 | Permit lead times | Routinely the actual delay cause |
| 6 | GP/GA chromosome + repair layer | Needed before the optimizer is usable |

---

## Overall assessment

The engine is ready for a GP/GA research prototype. It is not yet ready to hand
to an outage planner as a decision-support tool.
