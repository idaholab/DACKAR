# Workflow Reference — Demo 2: Pre-Outage Risk Prediction

**Notebook:** `dackar_v2_demo_executed.ipynb`  
**Pipeline path:** Proactive (pre-outage history analysis → risk register)  
**Output:** Ranked risk register with confidence tiers and recommended pre-staging actions

---

## Who this demo is for

### Outage Managers
The pre-outage questions this demo answers **before the outage window opens**:

| Question | Stage | Answer format |
|---|---|---|
| Which components are most likely to generate unplanned work? | G | Risk register ranked by tier + causal score |
| What should I pre-stage, pre-order, or add to scope? | G | Recommended action per component (tier-derived) |
| Which components have a history of consuming critical-path time? | F | Historical CP float consumed per outage |
| Is this component showing a worsening trend? | D | Trend score + escalation flag |
| Which components are safety-regulated — and does that change priority? | A | Regulatory constraint flag on each component |
| What evidence supports a HIGH risk flag? | G | Per-component evidence chain (CR/WO/activity records) |
| How accurate was the prediction last cycle? | G | RF-22 confusion matrix + actual vs predicted CP hours |

### Data Scientists
- Stage D (★ new) is the analytical differentiator: it converts raw CR/WO history into a scalar trend score that catches escalating components even when there is no prior emergent work record
- The `confidence_tier` logic combines causal score + trend escalation; components with `sme_informed` tier come from trend signal alone (`tier_reason = "escalating_trend_no_emergent_precedent"`)
- Ground truth holdout (RF-22) demonstrates the methodology; replace `demo_data.py` with real plant data to produce validated findings
- All plot functions return `(fig, ax)` — composable for reports and dashboard widgets

---

## Pipeline stages (A–G)

### Stage A — Data Ingestion & Normalization
**Input:** Component list, condition reports, work orders, activities, schedule  
**What it does:** Loads and validates all datasets; tags each activity with its emergence category; identifies regulated components; produces a quality gate result  
**Key output fields:**
- `quality_summary.data_record_counts` — component / CR / WO / activity / schedule counts
- `quality_summary.emergence_category_counts` — breakdown by `DISCOVERY`, `SCOPE_EXPANSION`, `REGULATORY`, `OPTIMIZATION`
- `quality_summary.quality_gate_passed`
- `regulatory_component_ids[]` — components with `regulatory_constraint_flag = True`

### Stage B — NLP Extraction
**Input:** Stage A normalized records  
**What it does:** Resolves abbreviations in all CR and WO descriptions; extracts named entities (plant IDs, CR/WO cross-references, nuclear vocabulary); computes unknown token rate  
**Key output fields:**
- `nlp_quality.unknown_token_rate` — target ≤ 15% on anchor component, ≤ 25% overall
- `nlp_quality.quality_gate_passed`

### Stage C — Knowledge Graph Construction
**Input:** Stage B extracted records  
**What it does:** Builds an in-memory knowledge graph linking components → CRs → WOs → activities  
**Key output fields:**
- `nodes{}` — all entities; each has `type`, `id`, and attributes
- `edges[]` — directed relations (CR_FOR, WO_FOR, PRECEDES, etc.)

### Stage D — Temporal Trend Analysis ★ New Stage
**Input:** Stage C KG per component  
**What it does:** Analyses degradation CR frequency, severity escalation, and WO duration overruns across training outage cycles; computes a composite trend score and escalation flag  
**Key output fields (per component):**
- `trend_score` — [0,1]; ≥ 0.5 = escalating, ≥ 0.2 = moderate
- `trend_label` — `escalating` / `moderate` / `stable` / `no_signal`
- `deg_counts_by_cycle{}` — degradation CR count per outage cycle
- `cycle_detail{}` — per-cycle `degradation_crs`, `observation_crs` counts
- `category_escalation` — boolean; True when severity stepped up from observation → degradation
- `overrun_ratios[]` — actual/planned duration ratios across WO history
- `overrun_mean` — mean overrun ratio (> 1.0 signals systematic underestimation)

**Why Stage D matters:** A component can have zero prior emergent activities yet show a clear escalation pattern (e.g., 1CSP-P-001B in the demo). Without Stage D, the causal score formula (Stage E) would assign it zero risk. Stage D catches these components before their first failure.

### Stage E — Causal Chain Scoring
**Input:** Stage C KG + Stage D trends  
**What it does:** Applies the causal evidence formula from `demo_build_guide.md`; produces a scalar risk index per component  
**Formula:**
```
causal_score = (n_outages_with_degradation_cr / n_training_outages)
             × (n_outages_with_emergent_activity / n_outages_with_degradation_cr)
             × criticality_weight
```
**Key output fields (per component):**
- `causal_score` — [0, 2]; ≥ 1.5 → DATA-SUPPORTED, ≥ 0.5 → SME-INFORMED
- `n_training_outages`, `n_outages_with_degradation_cr`, `n_outages_with_emergent_activity`
- `criticality_weight` — 2.0 if `safety_related` else 1.0

### Stage F — Schedule Risk Contextualization
**Input:** Stage E causal scores + historical activity records  
**What it does:** For each flagged component, retrieves its historical critical-path float consumption in prior outages  
**Key output fields (per component):**
- `historical_cp_impacts[]` — per-outage impact: `outage_id`, `float_consumed_hrs`, `on_critical_path`
- `mean_cp_float_consumed`
- `cp_impact_frequency` — fraction of training outages where the component consumed CP float

### Stage G — Risk Register & Recommendations
**Input:** All upstream stage artifacts  
**What it does:** Assigns confidence tier, generates finding + recommendation text, builds evidence chain, produces ranked risk register  
**Key output fields:**
- `risk_register[]` — ranked list; each entry: `component_id`, `confidence_tier`, `tier_reason`, `rank`
- `recommendations{}` — keyed by component_id; each has `finding`, `recommendation`, `category`, `evidence_chain[]`, `confidence_tier`
- `flagged_components[]` — IDs of components with non-null tier
- `true_negatives[]` — IDs of components correctly assessed as low-risk

---

## Confidence tier → recommended action

| Tier | Trigger | Recommended action |
|---|---|---|
| **DATA-SUPPORTED** | causal_score ≥ 1.5 (emergent history + degradation evidence) | Pre-order parts; allocate contingency crew to pool |
| **SME-INFORMED** | 0.5 ≤ causal_score < 1.5 (degradation history, no prior emergent) | Stage spare components; SME walkdown inspection on day 1 |
| **WATCH** | Escalating trend only; no emergent precedent | Monitor closely; inspector attention required at system opening |
| **— (not flagged)** | causal_score < 0.5, no escalation signal | No action required |

---

## Ground truth validation (RF-22 holdout)

The RF-22 holdout ground truth (`demo_data.RF22_GROUND_TRUTH`) is loaded **after** the pipeline runs — it is never seen during prediction. The confusion matrix and score-vs-actual chart in §7 of the notebook quantify prediction quality on the synthetic dataset.

With the Millbrook synthetic data:
- True positives: `1RHS-P-001A` (DATA-SUPPORTED), `1RHS-E-001A` (SME-INFORMED)
- False positive: `1CSP-P-001B` (SME-INFORMED — trend signal without prior emergent; this is a false positive in RF-22 but the recommendation to increase inspection vigilance is still defensible)
- True negatives: `1CCW-P-002A`, `1RHS-V-001A`

**To replace with real plant data:** swap out the five CSV-like dicts in `demo_data.py` with actual plant extracts; keep the same schema fields.

---

## Dataset schema (Millbrook synthetic)

**Components** — one row per component:
- `component_id`, `description`, `system`, `regulatory_constraint_flag`, `safety_related`, `notes`

**Condition Reports** — one row per CR:
- `cr_id`, `component_id`, `outage_id`, `date`, `description`, `severity_category` (`observation` / `degradation`)

**Work Orders** — one row per WO:
- `wo_id`, `component_id`, `outage_id`, `description`, `planned_hours`, `actual_hours`

**Activities** — one row per activity:
- `activity_id`, `component_id`, `outage_id`, `activity_type` (`planned` / emergent category), `planned_hours`, `actual_hours`, `on_critical_path`

**Schedule** — one row per component:
- `component_id`, `outage_id`, `available_float_hours`, `criticality_label`

---

## Plot function reference

All functions live in `demo_plots.py` in this folder.

| Function | Returns | Primary arguments |
|---|---|---|
| `draw_pipeline_architecture_v2(highlight_stage=None)` | `(fig, ax)` | — |
| `plot_stage_a_summary(sa, components)` | `(fig, axes[3])` | `results['stage_a']`, `COMPONENTS` |
| `plot_stage_d_trends(sd, comp_ids=None)` | `(fig, (ax1,ax2,ax3,ax4))` | `results['stage_d']` |
| `plot_stage_e_causal_scores(se, sg, comp_ids=None)` | `(fig, axes[2])` | `results['stage_e']`, `results['stage_g']` |
| `plot_stage_f_float_history(sf, flagged_ids=None)` | `(fig, ax)` | `results['stage_f']` |
| `plot_risk_register(sg, sd, se, components)` | `(fig, ax)` | `results['stage_g']`, `results['stage_d']`, `results['stage_e']`, `COMPONENTS` |
| `plot_anchor_evidence_chain(comp_id="1RHS-P-001A")` | `(fig, ax)` | — (hardcoded diagram) |
| `plot_recommendation_card_v2(comp_id, sg, sf, sd, se, components)` | `(fig, ax)` | all stage dicts + component list |
| `plot_ground_truth_validation(gt, sg, se, rf22_ground_truth)` | `(fig, axes[2])` | `results['ground_truth_comparison']` + stage dicts |

**Example — export the risk register for a management briefing:**
```python
from demo_plots import plot_risk_register
fig, ax = plot_risk_register(sg, sd, se, COMPONENTS,
                              outage_label="RF-23",
                              plant_label="Millbrook Unit 1")
fig.savefig("rf23_risk_register.pdf", bbox_inches="tight", dpi=150)
```

---

## Extending to real plant data

1. **Replace `demo_data.py`** with loaders that read from your plant's CMMS / EAM export (Maximo, SAP PM, etc.). Keep the same dict structure for each record type.
2. **Update `RF22_GROUND_TRUTH`** with the actual RF-22 (or most recent) emergent work records after the outage completes.
3. **Tune thresholds** in `pipeline.py` — specifically `CAUSAL_DATA_SUPPORTED_THRESHOLD` (default 1.5) and `TREND_ESCALATING_THRESHOLD` (default 0.5) — using precision/recall curves on historical data.
4. **Add fleet-level components** by extending the KG constructor (`_stage_c`) to include cross-unit data when your plant has multiple units.

---

*Part of the DACKAR Outage Analytics Framework. See also: [Demo 1 — In-Outage Activity Triage](../unexpected_act_workflow_1/workflow_reference.md) and [Demo 3 — Cross-Phase Integrated Story](../unexpected_act_cross_phase/workflow_reference.md).*
