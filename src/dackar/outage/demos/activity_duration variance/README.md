# Demo 3 — Activity Duration Estimation & Schedule Risk

**Question answered:** *How long will this planned activity take — and what does that
uncertainty do to the outage finish date?*

The demo estimates a duration distribution for each planned activity in an upcoming
refuelling outage (P50 / P80 / P90 percentiles), then propagates that uncertainty
through the full schedule network using a 2,000-iteration Monte Carlo simulation.

---

## Business context

Outage planners assign a single planned duration to each activity.
In reality, durations vary — sometimes by 50% or more — depending on what is found
during execution. This variance accumulates across the schedule: activities on the
critical path compound their overruns; activities with float absorb them.

Two questions matter for a planner:

1. **Per-activity:** "I planned 20 hours for this pump seal replacement.
   Based on similar work, should I expect 20 hours, 30 hours, or 40 hours?"

2. **Schedule-level:** "If all six activities on my critical path each run a bit long,
   what is the probability my outage overruns its target finish date?"

This demo answers both.

---

## How to run

### Notebook (recommended)

Open `outage_uncertainty_demo.ipynb` and run all cells.

```
conda activate base
cd .../outage
jupyter lab "demos/activity_duration variance/outage_uncertainty_demo.ipynb"
```

### CLI script

```
conda activate base
cd .../outage
python "demos/activity_duration variance/activity_duration_demo.py"
```

No external services required — all steps run offline.

---

## What the demo covers

| Section | What it does |
|---|---|
| 1 — Text pre-processing | Cleans noisy activity descriptions: expands abbreviations (`MOV` → motor operated valve), corrects typos, strips component tags |
| 2 — Pre-processing quality benchmark | Measures how much cleaning improves text similarity (normalised edit distance, token F1) |
| 3 — Duration estimation | Retrieves historical analogues, fits a distribution, returns P50/P80/P90 per planned activity |
| 4 — Monte Carlo schedule risk | Propagates per-activity distributions through the schedule network; outputs outage-level finish risk |

---

## Reading the duration estimates

For each planned activity the service returns:

| Output | What it means | How a planner uses it |
|---|---|---|
| `p50_hours` | Median expected duration — 50% of similar work finishes by this time | Use as the baseline plan duration |
| `p80_hours` | 80th-percentile duration — only 20% of similar work takes longer | Use when contingency matters; typical planning buffer target |
| `p90_hours` | 90th-percentile duration | Use for critical-path activities where overrun is costly |
| `confidence_tier` | How well-supported the estimate is by history (see table below) | Low confidence → plan more buffer; consult SME |
| `uncertainty_type` | Whether the variance is reducible or inherent (see table below) | Drives the right risk response |

**Rule of thumb:** For critical-path activities, plan to `p80`. For non-critical activities
with adequate float, `p50` is usually sufficient.

---

## Confidence tiers

The same tier taxonomy is used across all three DACKAR demos.

| Tier | What it means | How many analogues found |
|---|---|---|
| `data_supported` | Strong historical evidence — estimate is reliable | ≥ 5 similar past activities |
| `sme_informed` | Some history — estimate is directional, SME input recommended | 1–4 similar past activities |
| `low_confidence` | No similar history found — estimate is a prior, treat cautiously | 0 similar past activities |

---

## Uncertainty types

| Type | What it means | Risk response |
|---|---|---|
| `aleatory` | Natural variability — even with more data, duration will vary | Plan buffer; cannot be eliminated |
| `epistemic` | Uncertainty from lack of data — more history would narrow the estimate | Prioritise collecting better records; treat estimate conservatively |
| `mixed` | Both sources contribute | Use p80 as the planning target; flag for SME review |

---

## Reading the schedule risk output

The Monte Carlo simulation outputs:

| Output | What it means |
|---|---|
| **P80 / P90 finish time** | The outage finish time that 80% / 90% of simulated runs complete within |
| **Robustness** | Fraction of simulated runs that finish on or before the baseline target — e.g., 0.62 means 62% on-time probability |
| **Criticality index** | Fraction of simulation iterations in which each activity was on the critical path — high values mean the activity frequently determines outage length |
| **Expected drag** | Average hours each activity adds to the outage finish when it is on the critical path |
| **CP sensitivity** | Correlation between an activity's duration and the final overrun — the single most useful ranking for a planner |

**What to do with this:** Rank activities by CP sensitivity. Activities at the top of
that list are where schedule buffer and contingency resources have the highest return.

---

## The schedule network used in the demo

```
Q-INIT ──┬──► Q-MCP-2B ──► Q-REPL ──────────────────────────► Q-END
         ├──► Q-MSIV-2 ──────────────────────────────────────► Q-END
         ├──► Q-HX-CCW ──► Q-LT-CALIB ──► Q-BKR-4A ─────────► Q-END
         └──► Q-DCS-UPGRDE ──────────────────────────────────► Q-END
```

6 planned activities plus start/end and one fixed-duration auxiliary task.
Baseline critical path: 60 hours.
The dataset is synthetic — Plant Millbrook, a fictional PWR.

---

## How this fits the other demos

| Demo | When in the outage lifecycle | What it answers |
|---|---|---|
| Demo 2 (workflow 2) | Before the outage — proactive | "Which components are likely to generate emergent work?" |
| **This demo (Demo 3)** | Planning phase | "How long will each activity take, and what is the schedule risk?" |
| Demo 1 (workflow 1) | During the outage — reactive | "This activity just appeared. What do we do?" |

Duration estimates from this service feed **Stage D** of Demo 1's pipeline —
when Demo 1 asks "how long does this type of work typically take?", it draws on
the same analogue retrieval and distribution fitting shown here.
