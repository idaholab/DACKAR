# Workflow Reference — Demo 3: Cross-Phase Integrated Story

**Notebook:** `dackar_cross_phase_demo.ipynb`  
**Purpose:** Management briefing demo — shows the same component seen first as a planning signal, then as an execution decision, with quantified impact reduction from pre-staging.

---

## The Anchor Story

```
Component:  1SJ-MOV-101  (Motor-Operated Valve, Safety Injection system)
                          (mapped to 1RHS-P-001A in the Millbrook synthetic dataset)

PLANNING PHASE  (~6 weeks before outage)
────────────────────────────────────────────────────────────────────────
  Risk register output:
    Tier:    DATA-SUPPORTED  (causal_score = 1.80)
    Trend:   ESCALATING  (2 prior packing failures, 28-month recurrence cycle)
    Action:  Pre-order packing kit; allocate 2 mechanics to contingency pool

  Action taken by outage planning team:
    ✓ Packing kit ordered and confirmed on-site
    ✓ Mechanics assigned to contingency pool (week 1 of outage)

EXECUTION PHASE  (Day 4 of outage)
────────────────────────────────────────────────────────────────────────
  Trigger:  CR raised — "Through-leakage on 1SJ-MOV-101 during PMT"
  Pipeline runtime: < 2 minutes

  Pipeline outputs:
    Regulatory:   TS 3.5.7 — deferral to post-outage blocked
    Duration:     p50 = 14.5 h · p80 = 22.0 h  (18 analogs, DATA-SUPPORTED)
    CP impact:    14.5 h  (activity itself)
    Decision:     PROCEED with 7.5 h contingency buffer

  Without pre-staging:   mobilization = 8.5 h  →  total CP extension = 23.0 h
  With pre-staging:      mobilization = 0.0 h  →  total CP extension =  6.0 h

  SAVING:  17.0 h  (59% reduction in critical-path extension)
```

---

## Who this demo is for

This notebook is the **management briefing demo**. It answers the question:
> *"What is this framework worth — in hours and dollars?"*

Both personas are served:

**Outage Managers:** The two-column planning → execution timeline and the
impact-delta bar chart tell the story in a single slide.  The quantified saving
(17 h CP reduction) translates directly to avoided extension costs.

**Data Scientists:** The demo shows how two pipelines with independent data
sources (historical CR/WO database and live condition report text) connect through
the same knowledge graph.  The cross-module import pattern (`importlib.util`) is
reusable for any future integration.

---

## Notebook sections

| Section | What it shows |
|---|---|
| §1 — Run Both Pipelines | Sets up pre-outage and in-outage results side by side |
| §2 — Planning Phase: Risk Register | Ranked risk register with recommended pre-staging actions |
| §2.1 — Recommendation Card (planning) | Per-component card with evidence chain |
| §3 — Execution Phase: Triage Decision | In-outage PROCEED recommendation with TS constraint |
| §3.1 — Option Risk Scores | Stage F bar chart showing why PROCEED wins |
| §3.2 — Recommendation Card (execution) | Full recommendation card |
| §3.3 — Evidence Chain | Source-typed evidence items for the execution decision |
| §4 — Impact Delta | Stacked bar chart: reactive vs. actual CP extension |
| §5 — Cross-Phase Timeline | Two-column illustrated timeline with savings annotation |
| §6 — Pipelines Side by Side | Both pipeline architecture diagrams stacked |
| §7 — Closed-Loop Feedback | Feedback arc showing how actual duration improves future estimates |
| §8 — Summary | Before/after table for management |

---

## Key numbers (synthetic demo)

| Metric | Value |
|---|---|
| Causal score — anchor component | 1.80 (DATA-SUPPORTED threshold = 1.5) |
| Historical analogs retrieved | 18 |
| p50 duration estimate | 14.5 h |
| p80 duration estimate | 22.0 h |
| Contingency buffer recommended | 7.5 h (= p80 − p50) |
| CP drag from activity itself | 14.5 h |
| Mobilization — reactive (no pre-staging) | 8.5 h |
| Mobilization — actual (pre-staged) | 0.0 h |
| Total CP extension — reactive | 23.0 h |
| Total CP extension — actual | 6.0 h |
| CP impact reduction | 17.0 h (59%) |

---

## Import architecture

Demo 3 imports from both demo modules without name collisions using `importlib.util`:

```python
def _load_demo_plots(folder: Path):
    spec = importlib.util.spec_from_file_location("demo_plots", folder / "demo_plots.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_dp1 = _load_demo_plots(DEMO1_DIR)   # unexpected_act_workflow_1/demo_plots.py
_dp2 = _load_demo_plots(DEMO2_DIR)   # unexpected_act_workflow_2/demo_plots.py
```

This pattern is also how a future GUI would load plot modules at runtime without
requiring them to be installed as packages.

---

## Adapting to a real plant

1. Replace `SCENARIO_RCP_SEAL` in Demo 1 with a real condition report
2. Replace `demo_data.py` in Demo 2 with real plant CR/WO history
3. Update the mobilization constants (`REACTIVE_MOBI_H`, `PRESTAGED_MOBI_H`) with
   plant-specific data
4. The impact-delta and cross-phase timeline will update automatically

---

*Part of the DACKAR Outage Analytics Framework.  
See also: [Demo 1 — In-Outage Triage](../unexpected_act_workflow_1/workflow_reference.md) · [Demo 2 — Pre-Outage Risk Register](../unexpected_act_workflow_2/workflow_reference.md)*
