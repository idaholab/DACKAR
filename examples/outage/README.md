# DACKAR Outage — Demos

Three self-contained demonstrations of the outage activity analysis pipeline.
All demos run without external services (no Neo4j, no embedding server, no LOGOS required).

---

## How the three demos connect

The three demos cover different points in the outage lifecycle and answer different questions.
They are designed to be used together, not independently.

```
  ── Planning phase ──────────────────────────────────────────────────────────────────────────
  Demo 3 (duration)
  "How long will each planned activity take? What is the probability the outage overruns?"
  → Produces per-activity P50/P80/P90 duration distributions and a schedule risk ranking.
  → These estimates feed Stage D of Demo 1 whenever a similar activity is encountered.

  ── Pre-outage phase ─────────────────────────────────────────────────────────────────────────
  Demo 2 (workflow 2)
  "Which components are most likely to generate unexpected work in the upcoming outage?"
  → Analyses condition report and work order history across prior outages.
  → Produces a risk register with confidence-tiered recommendations: pre-stage resources,
    expand planned scope, or monitor.

  ── In-outage phase ──────────────────────────────────────────────────────────────────────────
  Demo 1 (workflow 1)
  "An unexpected activity just appeared. Should we insert it now, defer it, or escalate?"
  → Takes a single emergent activity and the component's KG timeline as input.
  → Produces an ESCALATE / PROCEED / DEFER recommendation with a traceable evidence chain.
```

**Recommended reading order for new users:**
Demo 2 → Demo 3 → Demo 1.
Start with Demo 2 to understand the pre-outage prediction problem, then Demo 3 for
duration uncertainty, then Demo 1 to see how a discovered activity is triaged in real time.

---

## Demo overview

|  | Demo 1 (workflow 1) | Demo 2 (workflow 2) | Duration variance |
|--|---------------------|---------------------|-------------------|
| **When** | Activity already discovered, in-outage | Before outage start | Planning phase |
| **Question** | "What should we do with this activity?" | "Which components will generate emergent work?" | "How long will this activity take?" |
| **Input** | Single emergent activity + KG timeline | CR/WO history across ≥2 prior outages | Historical activity completions |
| **Output** | ESCALATE / PROCEED / DEFER recommendation | Risk register + evidence chains | Duration distribution + schedule risk |
| **Pipeline** | Stages A–G | Ingestion → NLP → KG → Prediction → Recommendation | Duration service + Monte Carlo |
| **Notebook** | `dackar_workflow_demo.ipynb` | `dackar_v2_demo_executed.ipynb` | `outage_uncertainty_demo.ipynb` |
| **Folder** | `unexpected_act_workflow_1/` | `unexpected_act_workflow_2/` | `activity_duration variance/` |

---

## Folder contents

### `unexpected_act_workflow_1/`
In-outage triage: given a newly discovered emergent activity, run the full A–G pipeline
and produce a structured ESCALATE / PROCEED / DEFER recommendation.

| File | Purpose |
|------|---------|
| `README.md` | Start here — business context, stage outputs, confidence tiers, stub backend explanation |
| `dackar_workflow_demo.ipynb` | Interactive notebook — run this for the full demo |
| `demo_scenarios.py` | Stub backends + two pre-built scenarios (RCP Seal, Snubber inspection) |
| `embedding_benchmark.py` | Standalone benchmark comparing embedding models on nuclear text |
| `outage_cleaning_benchmark.csv` | Benchmark data — corruption robustness test set |
| `outage_cleaning_benchmark_severity.csv` | Benchmark data — category retrieval test set |

### `unexpected_act_workflow_2/`
Pre-outage risk prediction: given condition report and work order history from prior
outages, predict which components will generate emergent work in the upcoming outage.

| File | Purpose |
|------|---------|
| `dackar_v2_demo_executed.ipynb` | Main notebook — pre-executed with all outputs. Clear outputs (`Edit → Clear All Outputs`) and re-run to execute fresh. |
| `pipeline.py` | 7-stage pipeline (all in-memory, no external services) |
| `demo_data.py` | Synthetic Millbrook Nuclear Station dataset (RF-20/21 train, RF-22 holdout) |
| `demo_build_guide.md` | Step-by-step guide to how the pipeline was built |
| `test_case_spec.md` | Acceptance test cases for the demo pipeline |

### `activity_duration variance/`
Duration uncertainty: given historical activity completions, estimate a duration
distribution for a planned activity and propagate uncertainty through a schedule network.

| File | Purpose |
|------|---------|
| `README.md` | Start here — business context, how to read P50/P80/P90 estimates, confidence tiers, schedule risk outputs |
| `outage_uncertainty_demo.ipynb` | Interactive notebook — run this for the full demo |
| `activity_duration_demo.py` | CLI version of the same demo |

---

## Quick start

Each demo is self-contained. Open the notebook for the workflow you want and run all cells.
See `../docs/TESTING.md` for environment setup.
