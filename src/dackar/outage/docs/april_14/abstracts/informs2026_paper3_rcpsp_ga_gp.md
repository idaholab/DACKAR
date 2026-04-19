# INFORMS 2026 — Abstract and Paper Plan
## Paper 3: Resource-Constrained Outage Scheduling with GA and GP-Evolved Priority Rules

---

## INFORMS SUBMISSION ABSTRACT
*(250 words maximum — INFORMS Annual 2026 limit)*

Nuclear refueling outages present one of the most constrained instances of the resource-constrained project scheduling problem encountered in practice. Unlike classical formulations, outage scheduling must simultaneously enforce crew availability by shift and skill type, radiation dose accumulation limits per worker, physical zone occupancy constraints, equipment availability windows, plant system isolation states, consumable inventory depletion, regulatory hold points, and mobilization lead times — with regulatory time-window violations treated as schedule-invalidating rather than merely costly. This paper presents an event-driven genetic algorithm framework for multi-mode resource-constrained project scheduling of nuclear refueling outages, with three primary contributions. First, we formalize the nuclear outage scheduling problem as an eight-constraint-type multi-mode formulation and describe an event-driven parallel schedule generation scheme that advances time only to the next meaningful decision point, achieving sub-quadratic dispatch complexity for outage-scale instances. Second, we couple the scheduler to a genetic algorithm using the activity list representation, with a composite fitness function that penalizes makespan extension, total activity delay, critical-path exposure, and regulatory window violations with differentiated weights reflecting nuclear operational priorities. Third, we apply genetic programming to evolve composite priority dispatch rules operating on schedule network features — float, resource demand, successor count, and worst-case slack — replacing hand-crafted heuristics with learned rules that generalize across outage configurations. Results on standard project scheduling benchmark instances demonstrate competitive gap-to-optimal performance, while experiments on nuclear outage instances show that evolved dispatch rules reduce regulatory window violations and makespan extension relative to classical priority heuristics.

---

**Title:**
*An Event-Driven Genetic Algorithm with Genetic-Programming-Evolved Priority Rules for Nuclear Outage Resource-Constrained Project Scheduling*

**Authors:** [TBD]
**Conference:** INFORMS Annual 2026 — San Francisco, November 1–4, 2026
**Primary cluster:** Optimization — Metaheuristics / Discrete Optimization
**Secondary cluster:** Energy, Natural Resources, and the Environment
**Keywords:** resource-constrained project scheduling, genetic algorithm, genetic programming, hyper-heuristics, nuclear outage management, schedule optimization

---

## Submission Deadline

| Milestone | Date |
|---|---|
| Abstract submission deadline | Wednesday, 20 May 2026 |
| Speaker registration deadline | Wednesday, 22 July 2026 |
| Conference | Sunday 1 – Wednesday 4 November 2026 |

**Note:** INFORMS is an abstract + presentation conference — no full paper submission required.

---

## Paper Contributions

### Contribution 1 — Nuclear RCPSP Problem Formulation
The nuclear refueling outage scheduling problem requires simultaneously enforcing constraint types that do not co-occur in standard RCPSP benchmarks:

| Constraint type | Standard RCPSP | Nuclear outage |
|---|---|---|
| Renewable resources (crew by skill) | ✓ | ✓ |
| Non-renewable resources (consumables) | Sometimes | ✓ (AC suits, gaskets, seals) |
| Multi-mode activities | Sometimes | ✓ (crash vs. normal execution) |
| Shift calendars | Rarely | ✓ (mandatory) |
| Radiation dose budgets | No | ✓ (per-worker cumulative) |
| Physical zone occupancy | No | ✓ (multi-zone concurrent) |
| Plant system isolation states | No | ✓ (mutex per system/state) |
| Regulatory hold points | No | ✓ (NRC/QA gates) |
| Mobilization lead times | Rarely | ✓ |
| Multiple discrete time windows | Rarely | ✓ (surveillance test windows) |

This combination makes the outage RCPSP a novel, harder problem class than the PSPLIB benchmarks typically used for algorithm comparison.

### Contribution 2 — Event-Driven GA Evaluation Engine
Standard RCPSP solvers use an hour-by-hour time-stepping simulation to evaluate each candidate schedule, which scales as O(T × n) where T is the project horizon in hours. For a 21-day outage with 1,500 activities this is prohibitive inside a GA population loop.

The event-driven parallel schedule generation scheme advances time only to the next meaningful decision point (activity completion, shift start, equipment availability boundary, time-window opening), reducing the evaluation complexity to O(n log n) per candidate via a heap-based event queue with lazy deletion. This enables population sizes and generation counts that are impractical with time-stepped simulation.

### Contribution 3 — GP-Evolved Composite Priority Dispatch Rules
The serial schedule generation scheme used as the GA decoder requires a priority rule to order activities at each scheduling decision point. Classical rules (latest finish, greatest rank position weight, worst-case slack) are single-feature hand-crafted heuristics that were derived for standard RCPSP and do not account for nuclear-specific features such as dose budget pressure or zone occupancy.

Genetic programming evolves composite priority rules as expression trees over the feature set:
- Schedule network features: total float, early finish, latest start, rank position weight, successor count
- Resource features: resource demand, average/max resource requirement
- Dynamic features: worst-case slack, improved resource scheduling metric
- Nuclear domain features: dose budget consumed, zone occupancy load, time-to-window-close

The evolved rules are decoded through the `set_priorities(external)` interface, which injects per-activity priority values into the serial schedule generation scheme before each fitness evaluation. This creates a clean separation between the evolutionary search (GP) and the scheduling engine (serial SGS), allowing either component to be replaced independently.

---

## Evaluation Plan

### Benchmark validation (PSPLIB)
- Instances: j30 (30 activities), j60 (60), j120 (120 activities)
- Metric: percentage gap to optimal / best-known solution
- Comparison: GA with classical rules (LF, GRPW, WCS) vs. GA with GP-evolved rules
- Target: GP-evolved rules match or improve on best classical rule within the same wall-clock budget

### Nuclear outage case study
- Instance: synthetic 150–300 activity outage schedule with all 8 constraint types active
- Metric: makespan, number of regulatory window violations, total delay hours, critical-path exposure
- Comparison: classical dispatch heuristics vs. GP-evolved rules vs. unconstrained CPM baseline
- Visualisation: Gantt chart, resource utilisation profiles, critical chain analysis

---

## Methodology Summary

```
OUTER LOOP — Genetic Programming
  Population of priority rule expression trees
  Each individual: composite rule over feature set
  Fitness: schedule quality across k training outage instances
  Operators: subtree crossover, point mutation, constant perturbation
  Terminal set: schedule/resource/domain feature scalars
  Function set: +, −, ×, ÷(protected), min, max, if-then-else

INNER LOOP — Genetic Algorithm (Activity List)
  Chromosome: precedence-feasible permutation of activity indices
  Decoder: serial schedule generation scheme with external priorities
  Fitness: composite score (makespan + delay + criticality + window violations)
  Crossover: one-point / two-point / uniform order (UOX)
  Mutation: adjacent-swap with topological repair

SCHEDULE GENERATION
  Parallel SGS: event-driven, O(n log n), used for final evaluation
  Serial SGS: lightweight, used as GA decoder
  Constraint checks: all 8 nuclear constraint types at each decision point
```

---

## Differentiation from PHM Papers 1 and 2

This paper is purely an **optimization** contribution. The connection to Papers 1 and 2 is a data interface, not shared methodology:

- Papers 1 and 2 produce: per-activity duration distributions (p50/p80), component risk scores, confidence tiers
- Paper 3 consumes: p80 durations as multi-mode duration inputs; component criticality as a fitness weight
- This connection is described as a **design motivation** in the paper introduction and as **future integration work** in the conclusions — it is not a result claimed in this paper

The scheduling and optimization content stands entirely independently of the DACKAR PHM work.

---

## Code Status

| Component | Status |
|---|---|
| Event-driven parallel SGS (O(n log n)) | Complete |
| Serial SGS for GA decoding | Complete |
| GA — Activity List, crossover, mutation, topological repair | Complete |
| 21 priority rules including GH/GP seeded variants | Complete |
| Multi-mode RCPSP (modes, set_mode, CPM recompute) | Complete |
| Composite fitness function (4 components) | Complete |
| All 8 nuclear constraint types | Complete |
| External priority injection (set_priorities) | Complete |
| PSPLIB benchmark runner (ga_test.py) | Complete |
| **GP training outer loop** | **In development — expected days** |
| GP feature set definition for nuclear domain | To define |
| Nuclear outage benchmark instance | To build |

---

*Part of the DACKAR / LOGOS Outage Analytics Framework.*
*See also: [Paper 1 — RCA](phm2026_paper1_rca.md) · [Paper 2 — Outage Analytics](phm2026_paper2_outage_analytics.md)*
