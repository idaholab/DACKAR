# Plant Outages Risk Management — Data Analytics Framework
## Overview for Outage Management

---

## Slide 1 — The Problem We Are Solving

**Refueling outages are the most complex, high-stakes operations in plant life.**

- Hundreds of interdependent tasks, dozens of specialized crews, shared equipment, strict regulatory gates
- Each unplanned extension costs **hundreds of thousands of dollars per day**
- Emergent work — discovered mid-outage — is unavoidable, yet current tools offer no structured way to quantify its impact before committing
- Today's practice: deterministic critical-path methods, manual judgment, tribal knowledge

> **The gap:** We know _what_ needs to be done. We lack tools to know _when it will finish_, _what could go wrong_, and _what we should do next_ — with evidence to back the decision.

> **Figure 1:** Gantt chart of a representative outage schedule — a dense, resource-loaded network with many overlapping tasks and visible delay bars — to convey the sheer scheduling complexity before the analytics are applied.  
> _Source: `LOGOS/src/CPM/test_pert_res_full.ipynb` — "Gantt chart (HTML interactive with delay visualization)" cell. Export as static PNG at a resolution where individual task bars are readable._

---

## Slide 2 — Our Answer: An Integrated Analytics Framework

**Three capabilities. Two phases of the outage lifecycle. One coherent system.**

```
┌──────────────────────────────┐  ┌──────────────────────────────┐  ┌──────────────────────────────┐
│   1. RCPSP SCHEDULING        │  │   2. UNCERTAINTY              │  │   3. EMERGENT WORK           │
│                              │  │      QUANTIFICATION           │  │      ANALYSIS                │
│  Planning:                   │  │  Planning:                    │  │  Planning:                   │
│  "Does this schedule hold     │  │  "How much contingency        │  │  "Which components are       │
│  under real crew, dose,       │  │   should we reserve for       │  │   most likely to generate    │
│  and equipment limits?"       │  │   the critical path?"         │  │   unplanned work?"           │
│                              │  │                              │  │                              │
│  Execution:                  │  │  Execution:                  │  │  Execution:                  │
│  "A task slipped 6 hours.    │  │  "What is our probability    │  │  "We found a leak. Fix now   │
│   What is the new critical   │  │   of finishing on time       │  │   or defer — and what is     │
│   chain right now?"          │  │   given work remaining?"     │  │   the schedule impact?"      │
└──────────────────────────────┘  └──────────────────────────────┘  └──────────────────────────────┘
```

> **Figure 2:** Three equal-width boxes arranged horizontally, each with a header, a "Planning:" question, and an "Execution:" question. Use a two-tone layout per box (darker header band, lighter body). No arrows — the boxes are independent capabilities, not a pipeline. Suggested colors: blue for RCPSP, teal for UQ, amber for Emergent Work.  
> _No notebook source — design in draw.io, PowerPoint SmartArt, or Lucidchart based on the layout above._

---

## Slide 3 — Capability 1: Realistic Outage Scheduling (RCPSP Engine)

**Why standard CPM falls short — and what we do instead.**

Standard critical-path methods assume unlimited resources and ignore operational reality.  
Our Resource-Constrained Project Scheduling (RCPSP) engine models what actually governs outage execution:

| Constraint | What We Model |
|---|---|
| **Workforce** | Skill types, crew counts, shift calendars, overtime limits |
| **Equipment** | Asset availability windows, zone-locking to physical locations |
| **Radiation** | Cumulative dose budgets per craft; ALARA scheduling |
| **Plant Systems** | Mutual-exclusion states (valve alignments, system configurations) |
| **Consumables** | AC suits, gaskets, specialty seals — finite inventory tracked |
| **Regulatory** | Technical Specification windows, hold points, mode-change gates |
| **Spatial** | Zone occupancy limits, concurrent access restrictions |
| **Dependencies** | Finish-to-start lags, mobilization lead times |

**Result:** A schedule that reflects outage physics, not just network logic.

> **Figure 3:** Resource utilization stacked area chart showing crew availability (colored bands per skill type) against actual scheduled demand across the outage duration — illustrating how the engine fills crew windows and respects skill-specific limits.  
> _Source: `LOGOS/src/CPM/test_pert_res_full.ipynb` — "Resource utilization plots (per skill type, HTML Plotly)" cells. Combine two or three skill types into one figure; export as PNG._

---

## Slide 4 — Under the Hood: How the Scheduler Works

**Fast, event-driven simulation — not hour-by-hour stepping.**

- The scheduler advances time only to the **next meaningful decision point** (shift opening, equipment available, activity completes) — orders of magnitude faster for long outages
- At each decision point it evaluates every waiting task against **all constraints simultaneously** — crew, equipment, zone, dose, system state, regulatory window
- Assigns tasks using configurable **priority rules** (20+ available): total float, latest finish, resource demand, weighted criticality indices
- Supports **genetic algorithm optimization** for multi-mode RCPSP (each task can have alternative execution modes — fast/expensive vs. slow/less crew)

**Key output metrics every outage manager cares about:**

- Projected finish time and CPM vs. constrained-schedule gap
- Resource-constrained critical chain (may differ from CPM critical path!)
- Delay attribution: which activities were held up by resource conflicts vs. dependencies
- Regulatory window violations: which tasks risk missing their compliance window

> **Figure 4:** Activity network DAG with topological layout — nodes colored by total float (green = slack, red = critical chain) and edges labeled with FS lags where present. Shows the structural complexity of the scheduling problem in one view.  
> _Source: `LOGOS/src/CPM/test_pert_res_full.ipynb` — "Interactive Plotly DAG (activity graph with topological layout highlighting constrained chains)" cell. Export static PNG; annotate critical chain nodes in red._

---

## Slide 5 — Capability 1 in Action: What Changes

**Before → After**

| Today | With RCPSP Engine |
|---|---|
| CPM assumes crews are always available | Schedule accounts for skill availability by shift and day |
| Equipment conflicts discovered in the field | Zone-locked equipment conflicts flagged before mobilization |
| Dose-driven delays surface mid-job | ALARA constraints shape the schedule from the start |
| "Critical path" is fixed at outage start | Constrained critical chain updates as resources shift |
| Replanning is manual | Mid-outage replan in seconds: freeze completed work, reschedule everything else |

**Mid-outage replanning:** When actual progress diverges from plan, the engine reclassifies completed, in-progress, and pending tasks, re-applies resource state, and produces a fresh schedule — in the same computational framework, no manual data re-entry.

> **Figure 5:** Side-by-side Gantt comparison — left panel: CPM (unconstrained, tasks run as early as possible with no crew limits), right panel: RCPSP result (same tasks shifted due to resource constraints, critical chain highlighted). The visual gap between the two finish times makes the value of realistic scheduling immediately apparent.  
> _Source: `LOGOS/src/CPM/test_pert_res_full.ipynb` — run `calculateScheduleWithResources()` twice, once with empty pools (CPM proxy) and once with real pools; export both Gantt charts and place side by side._

---

## Slide 6 — Capability 2: Uncertainty Quantification

**"When will we finish?" needs a distribution, not a date.**

Deterministic schedules give one number. Reality produces a range.

**How we build the distribution:**

1. Query **historical work order data** — semantic similarity search retrieves analogous activities from past outages at this and comparable plants
2. Fit an **empirical duration distribution** (typically lognormal) to retrieved execution times
3. Separate **routine execution** (typical cases) from **disruption-driven outliers** (scope expansions, rework, parts delays)
4. Propagate uncertainty through the RCPSP engine via **Monte Carlo simulation** (1,000 schedule runs per analysis)

**Output: actionable risk metrics**

```
Activity: Replace SG-A Feedwater Nozzle

  Baseline (CPM):      12.0 h
  p50 finish:          14.5 h   ← median outcome
  p80 finish:          22.0 h   ← 80% confidence bound
  p90 finish:          28.5 h   ← conservative contingency anchor

  Critical path sensitivity:  65% of runs on CP
  Confidence tier:            DATA_SUPPORTED (18 analogs found)
```

**Contingency is no longer a gut call.** The p80−p50 gap tells you exactly how much buffer to add.

> **Figure 6:** Two-panel figure. _Left:_ histogram of sampled activity durations from historical analogs (bars) with fitted lognormal curve overlaid; p50/p80/p90 marked as vertical dashed lines with labels. _Right:_ histogram of Monte Carlo project finish times (1,000 runs) showing the same percentile markers — connecting duration uncertainty to schedule completion risk.  
> _Source: `DACKAR/src/dackar/outage/demos/activity_duration variance/outage_uncertainty_demo.ipynb` — "Duration estimation histogram" and "Schedule risk propagation chart" cells. Export as a combined two-panel PNG._

---

## Slide 7 — Capability 2: Criticality Index and Path Sensitivity

**Not all critical-path activities are equally risky.**

Two activities can both sit on the critical path yet have very different risk profiles.

**Criticality Index** = fraction of Monte Carlo runs in which an activity is on the critical path  
→ An activity with CI = 95% is nearly always critical. One with CI = 30% is critical only under adverse conditions.

**CP Drag** = how many hours sooner the project would finish if this activity's duration were halved  
→ Prioritizes where acceleration efforts (additional crew, parallel tasking) have the most leverage.

**Path Sensitivity** = variance in project finish time attributable to this activity's duration variability  
→ Identifies which activities need the most contingency protection.

These metrics let you answer:  
_"If I can only get one extra crew, where do I put them?"_

> **Figure 7:** Horizontal bar chart of activities ranked by Criticality Index (fraction of Monte Carlo runs on CP), bars colored by CP Drag magnitude (darker = more drag). A second small chart below shows the cumulative distribution of project finish times. Together they answer both "which task is most likely critical?" and "which task delays the project most if it slips?"  
> _Source: `DACKAR/src/dackar/outage/demos/activity_duration variance/outage_uncertainty_demo.ipynb` — "Risk metrics comparison (3 subplots)" and "Cumulative delay analysis" cells. Extract or adapt the relevant subplots._

---

## Slide 8 — Unexpected Activity Analysis: Planning Phase

**Before the outage window opens, identify which components are most likely to surprise you.**

The framework analyzes years of condition report and work order history to produce a ranked risk register of components likely to generate unplanned work:

- **Recurrence patterns:** Repeated corrective maintenance on the same failure mode signals an unresolved root cause
- **Degradation trends:** Escalating CR severity or frequency over time indicates a component approaching failure
- **PM compliance gaps:** Overdue preventive maintenance combined with prior degradation history
- **Outage correlation:** Components that historically generate emergent work specifically during refueling, when systems are opened and inspected for the first time in years

**Output delivered to planners before the outage starts:**

```
RISK REGISTER — Pre-Outage Assessment

  Component       Risk Score  Recurrences  Last PM    Recommended Action
  ──────────────  ──────────  ───────────  ─────────  ─────────────────────────
  1SJ-MOV-101     HIGH 0.84       2         8 mo ago  Add to work scope; pre-order packing kit
  2RHR-PUMP-A     MEDIUM 0.61     1        14 mo ago  Stage spare mechanical seal
  3AFW-CV-022     MEDIUM 0.57     1        11 mo ago  Monitor; inspector walkdown on day 1
  1CCW-HX-001     LOW 0.31        0         6 mo ago  No action required
```

Pre-staging resources and parts for high-risk components converts reactive emergent-work management into planned contingency — before a single wrench is turned.

> **Figure 8:** Two-panel figure. _Left:_ temporal degradation trend lines for the top flagged components (escalation score on Y-axis, time on X-axis), with the outage start date marked as a vertical dashed line. _Right:_ risk register table as shown above, rows color-coded by risk band (red / amber / green).  
> _Source: `DACKAR/src/dackar/outage/demos/unexpected_act_workflow_2/dackar_v2_demo_executed.ipynb` — "Temporal degradation trends" and "Risk register summary" cells. Pre-executed outputs available; export as PNG._

---

## Slide 9 — Unexpected Activity Analysis: Execution Phase

**When an unplanned activity is discovered mid-outage, a structured recommendation is produced in minutes.**

A freeform condition report description is the only input required. The pipeline automatically answers the four questions an outage planner needs right now:

| Question | How it is answered |
|---|---|
| **What is this activity?** | NLP intake: component identified, regulatory flags detected, emergence type classified |
| **Has this happened before?** | Knowledge graph query: 5-year component history; temporal causal chain scored |
| **How long will it take?** | Semantic retrieval of historical analogs; lognormal distribution fitted; p50/p80/p90 reported |
| **What should we do?** | Monte Carlo schedule impact; insertion options ranked by risk; PROCEED / DEFER / ESCALATE |

**The output is a scored, evidence-backed recommendation — not a gut call.**

```
Input:  "Through-leakage on 1SJ-MOV-101. TS 3.5.7 applicable."

        ┌─────────────────────────────────────────────────────┐
        │  PROCEED — Insert now with 7.5 h contingency buffer │
        │                                                     │
        │  • TS 3.5.7 prohibits deferral                      │
        │  • p80 duration 22 h; buffer covers uncertainty     │
        │  • Protects 3 downstream regulatory tasks           │
        │  ⚑ Analyst review: recurrent pattern, 28-month cycle│
        └─────────────────────────────────────────────────────┘
```

> **Figure 9a — Pipeline diagram:** The 7-stage A–G flow with stage names and one-line descriptions; already generated in `dackar_workflow_demo.ipynb` — export as PNG.

> **Figure 9b — Options risk chart:** Horizontal bar chart of insertion options ranked by risk score; blocked options hatched with lock icon; recommended option starred.  
> _Source: `DACKAR/src/dackar/outage/demos/unexpected_act_workflow_1/dackar_workflow_demo.ipynb` — "Stage F option risk scores" cell._

---

## Slide 10 — An Example Across Both Phases

**The same component, seen twice: first as a planning signal, then as an execution decision.**

```
  PLANNING PHASE  (6 weeks before outage)
  ─────────────────────────────────────────────────────────────
  Risk register flags 1SJ-MOV-101 as HIGH risk (score 0.84)
    → 2 prior packing failures, 28-month recurrence cycle
    → Recommended action: add to work scope; pre-order packing kit

  Action taken: packing kit pre-ordered; 2 mechanics allocated
                to contingency pool for first week of outage


  EXECUTION PHASE  (Day 4 of outage)
  ─────────────────────────────────────────────────────────────
  CR raised: "Through-leakage on 1SJ-MOV-101 during PMT"
  Pipeline runs in < 2 minutes:

    History:   prior weep CR in 2020 — recurrence confirmed
    Duration:  p50 = 14.5 h · p80 = 22.0 h  (18 analogs)
    Impact:    +14.5 h delay · on critical path · 3 regulatory
               tasks displaced
    Regulatory: TS 3.5.7 — deferral blocked

    → PROCEED with contingency buffer
      Pre-ordered kit available → mobilization time: 0 h
      Pre-allocated mechanics available → no resource conflict
      Effective CP impact reduced from +14.5 h to +6 h
```

**The planning-phase flag cut the execution-phase impact in half.** Pre-staged resources eliminated mobilization delay; pre-ordered parts avoided a parts-wait that would have added 8+ hours in a reactive scenario.

> **Figure 10:** Two-column timeline graphic. Left column (Planning): risk register entry for the component with trend line. Right column (Execution): the recommendation card for the same component. A horizontal arrow between them labeled "Pre-staged kit and crew" with the impact reduction (−8 h) highlighted. This figure will need to be designed from scratch compositing outputs from both notebooks.  
> _Source panels: `dackar_v2_demo_executed.ipynb` (risk register row) + `dackar_workflow_demo.ipynb` (recommendation card). Composite in draw.io or PowerPoint._

---

## Slide 11 — Evidence Traceability

**Every recommendation is documented, sourced, and reusable — across both phases.**

Decisions made verbally today leave no record. When the same component fails next cycle, the reasoning is gone.

The framework stores a structured evidence chain for every planning-phase flag and every execution-phase recommendation:

```
EVIDENCE CHAIN — 1SJ-MOV-101

  Planning flag    Degradation score 0.84 · 2 recurrences · trend slope +0.12/month
  (pre-outage)     Recommended: add to scope, pre-order packing kit

  Execution rec.   CR raised Day 4 · TS 3.5.7 defer-prohibited
  (mid-outage)     18 analogs · p80 = 22.0 h · CP impact +14.5 h
                   Decision: PROCEED with 7.5 h buffer
                   Actual duration: 16.2 h  ← recorded post-completion

  Feedback loop    Actual 16.2 h added to analog index
                   Next query for "MOV packing corrective" will include this job
```

- Searchable across outage cycles: _"Show all high-risk flags that materialized as emergent work"_
- Closes the loop: actual durations feed back into the analog index, improving future estimates
- Supports post-outage review, root cause analysis, and regulatory inquiries with a full audit trail

> **Figure 11:** Evidence chain visualization showing both the planning flag node and execution recommendation node for the same component, connected by a timeline arrow, with the feedback loop back into the analog index shown as a return arc.  
> _Source: `DACKAR/src/dackar/outage/demos/unexpected_act_workflow_1/dackar_workflow_demo.ipynb` — "Evidence chain visualization" cell as the base; annotate with the planning-phase flag node and feedback arc in post-processing._

---

## Slide 12 — What Outage Managers Gain

**Concrete improvements across both phases of outage management.**

| Phase | Challenge Today | With This Framework |
|---|---|---|
| Planning | "We're surprised by the same failures every outage" | Proactive risk register flags high-risk components weeks in advance; resources and parts pre-staged before work starts |
| Planning | "How much contingency should we reserve on the critical path?" | p80−p50 duration gap from historical analogs gives a data-driven buffer; no more gut-feel padding |
| Planning | "Our schedule looks fine on paper but ignores real crew and equipment limits" | RCPSP engine enforces shift calendars, dose budgets, zone constraints, and hold points before the outage window opens |
| Planning | "We don't know which activities are truly critical once resources are constrained" | Resource-constrained critical chain replaces unconstrained CPM as the planning baseline |
| Execution | "We don't know when we'll finish given what has slipped so far" | p50/p80/p90 project finish times updated in real time as actual progress is recorded |
| Execution | "A task slipped — I need a new schedule now, not tomorrow morning" | Mid-outage replan runs in seconds: completed work frozen, remaining work rescheduled against current resource state |
| Execution | "This emergent job — should we fix it now or defer?" | Structured PROCEED / DEFER / ESCALATE recommendation in minutes, with regulatory constraints and schedule impact quantified |
| Execution | "If I can get one extra crew, where do I put them?" | Criticality index and CP drag identify which activities have the most schedule leverage at this moment |
| Both | "We lose the reasoning behind past decisions" | Every planning flag and execution recommendation documented with sources, searchable across outage cycles |
| Both | "We keep rediscovering the same component problems" | Actual durations feed back into the analog index; each outage makes the next estimate more accurate |

---

## Slide 13 — Architecture at a Glance

```
                    DATA SOURCES
         ┌──────────────────────────────┐
         │  Outage Schedule (P6 / JSON) │
         │  Maintenance History (Neo4j) │
         │  Condition Reports           │
         │  Work Orders                 │
         │  Resource Pools              │
         └──────────────┬───────────────┘
                        │
          ┌─────────────▼─────────────┐
          │    RCPSP SCHEDULING       │
          │    ENGINE  (LOGOS/CPM)    │
          │  ┌─────────────────────┐  │
          │  │ Event-driven loop   │  │
          │  │ 8 constraint types  │  │
          │  │ 20+ priority rules  │  │
          │  │ GA optimization     │  │
          │  │ Mid-outage replan   │  │
          │  └─────────────────────┘  │
          └─────────────┬─────────────┘
                        │  Scheduled baseline + metrics
          ┌─────────────▼─────────────┐
          │  DECISION SUPPORT         │
          │  PIPELINE  (DACKAR)       │
          │  ┌─────────────────────┐  │
          │  │ NLP intake (A)      │  │
          │  │ KG timeline (B)     │  │
          │  │ Causal chain (C)    │  │
          │  │ Analog retrieval (D)│  │
          │  │ Monte Carlo (E)     │  │
          │  │ Option scoring (F)  │  │
          │  │ Recommendation (G)  │  │
          │  └─────────────────────┘  │
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │  OUTAGE MANAGER INTERFACE │
          │  Decision + Evidence Chain│
          │  PROCEED / DEFER /        │
          │  ESCALATE / MONITOR       │
          └───────────────────────────┘
```

---

## Slide 14 — Current Status and Maturity

**Where we are today.**

| Capability | What it delivers to outage managers | Status |
|---|---|---|
| Constraint-aware scheduling | Generate a schedule that accounts for crew skills, shift calendars, dose limits, equipment availability, and regulatory hold points — before the outage starts | Ready |
| Resource-constrained critical chain | Identify which activities will actually drive the outage finish date once real workforce and equipment limits are applied | Ready |
| Mid-outage replanning | When work slips or scope changes, produce an updated schedule in seconds without manual re-entry | Ready |
| Probabilistic finish times | Report p50 / p80 / p90 project completion times based on historical work durations — replace a single deterministic date with a confidence range | Ready |
| Data-driven contingency sizing | Quantify how much schedule buffer to add to each critical activity based on historical variability — not engineering judgment | Ready |
| Proactive component risk register | Deliver a ranked list of components likely to generate unplanned work, weeks before the outage window opens | In development |
| Emergent work recommendation | When an unexpected activity is discovered, produce a PROCEED / DEFER / ESCALATE recommendation with schedule impact and regulatory constraints in minutes | Integration testing |
| Outage manager dashboard | Single interface for schedule view, risk register, active recommendations, and evidence chains | Scoped |

---

## Slide 15 — The Path Forward

**From prototype to production support tool.**

**Near-term (next 3–6 months)**
- Integrate pipeline with live outage schedule data feeds (P6 / SAP)
- Calibrate duration distributions from plant-specific historical work orders
- Pilot decision-support pipeline on representative emergent activities from recent outages (retrospective validation)
- Develop outage manager dashboard (schedule view + active recommendations)

**Medium-term (6–18 months)**
- Deploy proactive risk identification module pre-outage
- Extend to multi-unit coordination (shared resources across simultaneous outages)
- Automated schedule optimization: GA-driven mode selection for outage-shortening opportunities
- Feedback loop: record actual durations → refine historical analog index

**Long-term vision**
- Continuous probabilistic schedule tracking during outage execution
- Automated replanning triggered by field-reported progress
- Fleet-level pattern learning across multiple plants and outage cycles

---

## Slide 16 — Summary

**A step-change in how outage risk is understood and managed.**

Three integrated capabilities:

> **1. Realistic Scheduling**  
> Models every constraint that governs outage execution. Produces a schedule that holds up in the field, not just on paper.

> **2. Quantified Uncertainty**  
> Replaces a single finish date with a probability distribution. Contingency is grounded in data, not intuition.

> **3. Evidence-Based Decisions**  
> When emergent work arrives, produces a structured recommendation in minutes — with the plant history, schedule impact, and regulatory constraints that support it, all documented and traceable.

**The goal:** Move outage management from _"we do our best with the schedule we have"_ to _"we know our risk, we understand our options, and we have the evidence to act."_

> **Figure 12:** Radar / spider chart comparing the two scenarios from the workflow demo (RCP seal leak vs. snubber inspection) across five axes: schedule impact, confidence level, causal evidence strength, regulatory risk, and resource readiness. Shows how different types of emergent work produce different risk profiles and different recommended responses.  
> _Source: `DACKAR/src/dackar/outage/demos/unexpected_act_workflow_1/dackar_workflow_demo.ipynb` — "Scenario comparison radar chart" cell. This figure is already generated; export as PNG._

---

_Framework developed by the Data Analytics for Critical Knowledge Acquisition and Reasoning (DACKAR) team._  
_Scheduling engine: LOGOS RCPSP module. Decision support: DACKAR outage pipeline._
