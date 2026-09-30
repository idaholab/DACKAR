# Workflow Reference — Demo 1: In-Outage Activity Triage

**Notebook:** `dackar_workflow_demo.ipynb`  
**Pipeline path:** Reactive (mid-outage discovery → recommendation)  
**Decision outputs:** ESCALATE / DEFER / PROCEED / MONITOR

---

## Who this demo is for

### Outage Managers
The seven questions this demo answers in real time:

| Question | Stage | Answer format |
|---|---|---|
| What is this activity? | A | Emergence type, regulatory flags, entity list |
| Has this happened before on this component? | B + C | Allen-relation timeline, causal posture |
| How long will it take? | D | p50 / p80 / p90 duration, analog count |
| How much schedule risk does this add? | E | Float consumed, CP drag, criticality label |
| What are my options? | F | Risk-scored options, feasibility flags, regulatory blocks |
| What should I do? | G | ESCALATE / DEFER / PROCEED / MONITOR with rationale |
| Why should I trust this recommendation? | G | Evidence chain with confidence scores per source |

### Data Scientists
The demo exposes every stage artifact as a Python dict and every plot as a reusable function. Points of interest:
- Confidence tier logic: `low_confidence` when analog count = 0, `sme_informed` for 1–4, `data_supported` for ≥5
- MONITOR branch trigger conditions (see §Decision Taxonomy below)
- All plot functions return `(fig, ax)` — embed them in your own figures or export via `fig.savefig()`

---

## Pipeline stages (A–G)

### Stage A — Activity Intake & Classification
**Input:** Free-text condition report description  
**What it does:** NLP entity extraction (component IDs, regulatory keywords, CR/WO cross-references), regulatory driver detection, emergence type classification (`SCOPE_EXPANSION`, `DISCOVERY`, `REGULATORY`, `OPTIMIZATION`), data quality scoring  
**Key output fields:**
- `emergence_type` — category of unexpected work
- `regulatory_drivers[]` — list of TS clauses; each has `defer_prohibited: bool`
- `unknown_abbreviation_rate` — drives analyst review flag if > 0.25
- `data_quality_score` — [0,1] composite NLP quality metric
- `safety_related` — determines whether `defer_to_post_outage` is feasible in Stage F

### Stage C — Temporal Event Chain (Allen Interval Algebra)
**Input:** Stage A artifact + KG component history  
**What it does:** Retrieves prior events for the component; classifies each event's temporal relation to the current activity (PRECEDES, OVERLAPS, CONTAINS, DURING, FOLLOWS, SIMULTANEOUS); scores causal strength  
**Key output fields:**
- `chain_links[]` — each link: `prior_event_type`, `prior_event_timestamp`, `allen_relation`, `relation_score`, `confidence`
- `summary.causal_posture` — `established` / `partial` / `insufficient_data`
- `summary.strong_link_count` — number of links with confidence ≥ 0.7

### Stage D — Historical Analog Retrieval
**Input:** Stage A activity description  
**What it does:** Semantic similarity search over historical work orders; fits a duration distribution (typically lognormal) to retrieved analogs; reports confidence tier based on analog count  
**Key output fields:**
- `analogs[]` — each analog: `outage_id`, `description`, `actual_duration_hours`, `similarity_score`
- `duration_distribution.p50_hours / p80_hours / p90_hours`
- `duration_distribution.confidence_tier` — `data_supported` / `sme_informed` / `low_confidence`
- `retrieval_summary.analog_count`

### Stage E — Schedule Impact Assessment
**Input:** Stage A activity + LOGOS schedule data  
**What it does:** Computes how inserting the activity consumes available float; estimates CP drag; identifies displaced tasks and resource conflicts  
**Key output fields:**
- `float_analysis.available_float_before_hours` / `float_consumed_hours` / `remaining_float_after_hours`
- `float_analysis.criticality_label` — `critical` / `near_critical` / `non_critical`
- `cp_impact.cp_drag_hours` — how many hours the outage finish date is extended
- `cp_impact.baseline_cp_hours`
- `displaced_tasks[]` — tasks pushed out; each has `has_regulatory_constraint: bool`
- `resource_conflicts[]`

### Stage F — Insertion Option Generation & Scoring
**Input:** Stages A + C + D + E artifacts  
**What it does:** Generates all viable insertion options (insert now, defer to later in outage, defer to post-outage, scope reduction, contingency buffer, escalate to management); scores each option's risk; applies regulatory clearance check  
**Key output fields:**
- `options[]` — each: `option_id`, `option_type`, `risk_score` [0,1], `feasible`, `regulatory_cleared`, `confidence`, `rationale`
- `recommended_option_id` — lowest risk score among feasible + cleared options
- `ranking_summary.best_risk_score`

### Stage G — Recommendation Synthesis
**Input:** All upstream stage artifacts  
**What it does:** Synthesises primary recommendation, computes attention flags, performs analyst review check, builds executive summary and evidence chain  
**Key output fields:**
- `decision_status` — `ESCALATE` / `PROCEED` / `DEFER` / `MONITOR` / `INCONCLUSIVE`
- `executive_summary.confidence_tier` — inherited from Stage D
- `executive_summary.primary_conclusion` — one-paragraph plain-English summary
- `executive_summary.analyst_attention_flags[]` — flags like `low_confidence_recommendation`, `low_analog_count`, `high_unknown_abbreviation_rate`, `regulatory_constraint_present`, `displaced_regulatory_tasks`
- `analyst_review.required` / `analyst_review.reason`
- `primary_recommendation` — the winning option dict from Stage F
- `evidence_chain[]` — each item: `source_type`, `snippet`, `confidence`

---

## Decision taxonomy

| Decision | When it fires | What it means for the outage manager |
|---|---|---|
| **ESCALATE** | CP drag > 24 h, or regulatory constraint + escalate option has lowest risk score | Stop other work; escalate to shift supervisor and outage manager immediately |
| **PROCEED** | Insert now option has lowest risk, regulatory constraint present (defer blocked), schedule impact manageable | Begin mobilization; apply contingency buffer if available |
| **DEFER** | Non-critical schedule, no defer prohibition, defer option has lower risk than insert now | Schedule the work at next available window; monitor float consumption |
| **MONITOR** | Zero analogs + low confidence + non-critical + insufficient causal evidence | Cannot recommend confidently — surface to domain SME before committing |

---

## Scenario inventory

| Scenario | File key | Expected decision | Key driver |
|---|---|---|---|
| Scenario 1 — RCP Train-A Seal Leak | `SCENARIO_RCP_SEAL` | ESCALATE | TS 3.1.4 defer-prohibited; 18 analogs; CP drag 14 h |
| Scenario 2 — Snubber Scope Expansion | `SCENARIO_SNUBBER_EXT` | DEFER | Non-critical; 5 analogs; 28 h float remaining |
| Scenario 3 — Unknown Component | `SCENARIO_UNKNOWN_COMPONENT` | MONITOR | Zero analogs; empty KG; safety_related blocks defer; no causal evidence |

---

## Adding a new scenario

1. Open `demo_scenarios.py`
2. Define an `_ACTIVITY` dict (see `_RCP_SEAL_ACTIVITY` as a template) with fields:
   - `activity_id`, `description`, `component_id`, `safety_related`, `regulatory_keywords`
3. Define a `_KG_DRIVER` dict (component history events) or use `{}` for zero history
4. Define an `_ANALOG_ACTIVITIES` list (historical work orders) or use `[]`
5. Define a `_SCHEDULE_IMPACT` dict — see `_UNKNOWN_COMPONENT_SCHEDULE_IMPACT` for the required schema fields (`float_analysis`, `cp_impact`, `duration_estimate`, etc.)
6. Assemble into a `SCENARIO_*` dict and add it to the module's `__all__`
7. Import in the notebook setup cell alongside the existing scenarios
8. Add a `run_pipeline(SCENARIO_*)` call and a new section in the notebook

---

## Plot function reference

All functions live in `demo_plots.py` in this folder.

| Function | Returns | Primary data argument |
|---|---|---|
| `draw_pipeline_architecture(highlight_stage=None)` | `(fig, ax)` | — |
| `plot_stage_a_summary(intake)` | `(fig, axes[3])` | `r['intake']` |
| `plot_allen_timeline(temporal)` | `(fig, ax)` | `r['temporal']` |
| `plot_analog_distribution(analogs_result)` | `(fig, (ax1, ax2))` | `r['analogs']` |
| `plot_schedule_impact(schedule)` | `(fig, (ax1, ax2))` | `r['schedule']` |
| `plot_option_risk_scores(options_result)` | `(fig, ax)` | `r['options']` |
| `plot_recommendation_card(result)` | `(fig, ax)` | full result dict |
| `plot_scenario_comparison(*results)` | `(fig, (radar, bar, table))` | 2–3 result dicts |
| `plot_evidence_chain(result)` | `(fig, ax)` | full result dict |

**Composing into a report figure:**
```python
from demo_plots import plot_option_risk_scores, plot_recommendation_card

fig, axes = plt.subplots(1, 2, figsize=(22, 6))
plot_option_risk_scores(result['options'], ax=axes[0])
plot_recommendation_card(result, ax=axes[1])
fig.savefig('report_figure.pdf', bbox_inches='tight')
```

---

## Trust architecture fields (§8 requirements)

Every pipeline result must contain these four top-level fields:

| Field | Purpose |
|---|---|
| `run_id` | Unique identifier linking all stage artifacts for this recommendation |
| `recommendation.evidence_chain` | Sourced evidence items with confidence scores |
| `recommendation.executive_summary.analyst_attention_flags` | Machine-detectable conditions requiring human review |
| `recommendation.analyst_review.required` | Boolean; triggers SME escalation workflow |

The notebook's §7 cell verifies these fields are present before any recommendation is used.

---

*Part of the DACKAR Outage Analytics Framework. See also: [Demo 2 — Pre-Outage Risk Prediction](../unexpected_act_workflow_2/workflow_reference.md) and [Demo 3 — Cross-Phase Integrated Story](../unexpected_act_cross_phase/workflow_reference.md).*
