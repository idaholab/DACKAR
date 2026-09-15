# Demo 1 — In-Outage Activity Triage (Workflow 1)

**Question answered:** *An unexpected activity has just been discovered during the outage. What should we do with it — and why?*

The pipeline runs stages A–G on a single emergent activity and produces an
**ESCALATE / PROCEED / DEFER** recommendation backed by a traceable evidence chain.

---

## Business context

During a nuclear outage, unexpected work is discovered continuously. Each item
requires a rapid decision: insert it now, defer it, or escalate to management.
Getting this wrong is costly in either direction — inserting unnecessarily extends
the outage; deferring something safety-critical creates compliance risk.

This demo shows how DACKAR reasons through that decision automatically, using:
- the component's historical event record (condition reports, prior work orders)
- the current schedule float at the proposed insertion point
- regulatory constraints (Technical Specification holds)
- historical analogues (how long has similar work taken before?)

---

## The two scenarios

| | Scenario 1 — RCP Seal Leak | Scenario 2 — Snubber Scope Expansion |
|---|---|---|
| **Emergence type** | `regulatory_driven` | `scope_expansion` |
| **CP impact** | 48 h drag on critical path | 0 h drag, 28 h float remaining |
| **Regulatory constraint** | TS 3.4.6 — deferral prohibited | None |
| **Expected decision** | **ESCALATE** | **PROCEED** |

These two scenarios are designed to produce opposite outcomes from the same pipeline,
making it easy to trace exactly which inputs drove each conclusion.

---

## How to run

Open `dackar_workflow_demo.ipynb` and run all cells.
No external services are required — see "Stub backends" below.

```
conda activate base
cd .../outage
jupyter lab demos/unexpected_act_workflow_1/dackar_workflow_demo.ipynb
```

---

## What are "stub backends"?

Stages B and D are marked **Live (stub backend)** in the pipeline architecture diagram.
This means they run real production logic but read from pre-built in-memory fixtures
rather than live external services:

| Stage | Production service | Demo substitute |
|---|---|---|
| B — KG Timeline Builder | Neo4j knowledge graph query | `_StubKGDriver` in `demo_scenarios.py` — hardcoded condition reports and prior work orders for each scenario |
| D — Historical Analog Retriever | Vector embedding server (Ollama) | `_StubRetrievalIndex` in `demo_scenarios.py` — pre-selected analogues with similarity scores |
| E — Schedule Impact | LOGOS CPM scheduling engine | Pre-built float/drag numbers in `demo_scenarios.py` |

The stub data is realistic (it reflects real failure modes and timescales for RCP pump
seals and mechanical snubbers) but it is synthetic — not sourced from a real plant.

All stub definitions live in `demo_scenarios.py` in this folder.
To understand exactly what historical data the pipeline sees, read that file.

---

## Stage outputs — what to look for

| Stage | Key output | What it means |
|---|---|---|
| A — Intake | `emergence_type`, `has_regulatory_constraint` | How this activity entered scope; whether deferral is legally possible |
| B — KG Timeline | `events` list | Prior condition reports and work orders for this component |
| C — Temporal Chain | `causal_posture` | `supported` / `partial` / `contradicted` — did past events causally precede this one? |
| D — Analog Retrieval | `confidence_tier`, `p50_hours`, `p80_hours` | How similar the history is; duration planning estimates |
| E — Schedule Impact | `cp_drag_hours`, `remaining_float_after_hours` | Hours added to outage end date; negative = critical path extended |
| F — Insertion Options | `recommended_option_id`, option risk scores | Best available way to handle the activity |
| G — Recommendation | `decision_status`, `evidence_chain` | Final ESCALATE / PROCEED / DEFER + full audit trail |

---

## Confidence tiers

The same tier taxonomy is used across all three DACKAR demos.

| Tier | What it means | Analogue count |
|---|---|---|
| `data_supported` | Strong historical evidence — act on this confidently | ≥ 5 analogues |
| `sme_informed` | Partial history — use SME judgment alongside the estimate | 1–4 analogues |
| `low_confidence` | No history found — treat all estimates cautiously | 0 analogues |

A `low_confidence` recommendation does **not** mean the pipeline failed.
It means there is insufficient historical data and a human reviewer must weigh in.

---

## Evidence chain and trust architecture

Every DACKAR recommendation cites its sources. The evidence chain (Stage G output)
lists every data point that contributed to the decision:
- historical work orders and condition reports used as analogues
- schedule float values and their source
- temporal event relations (Allen interval algebra results)
- regulatory constraint records

This is the audit trail. Any stakeholder can trace the recommendation back to
specific source records and understand exactly why the system reached its conclusion.

---

## How this fits the other demos

| Demo | When in the outage lifecycle | What it answers |
|---|---|---|
| **This demo (Demo 1)** | During the outage — reactive | "This activity just appeared. What do we do?" |
| Demo 2 (workflow 2) | Before the outage — proactive | "Which components are likely to generate emergent work?" |
| Demo 3 (duration) | Planning phase | "How long will this activity take, and what is the schedule risk?" |

Demo 2 tells you *which* components to watch before the outage starts.
This demo tells you *what to do* when one of those components generates unexpected work.
Demo 3 provides the duration estimates that feed Stage D of this pipeline.
