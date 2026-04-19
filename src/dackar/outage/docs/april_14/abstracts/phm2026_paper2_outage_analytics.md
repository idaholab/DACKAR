# PHM 2026 — Extended Abstract
## Paper 2: Outage Analytics — Emergent Work Decision Support

---

## PHM SUBMISSION ABSTRACT
*(Single paragraph, compliant with PHM guidelines: 300+ words, no abbreviations, no equations, no references)*

Nuclear refueling outages are among the most operationally complex and cost-sensitive activities in plant life, and their schedule performance is governed by two persistent challenges that operate across both the planning and execution phases. The first is the inherent variability in activity completion times: durations depend on component health state, crew availability, and field findings that deviate from plan, yet current practice sizes contingency buffers through engineering judgment rather than data-driven uncertainty quantification. The second is the occurrence of unplanned maintenance activities discovered during execution, where a defect found during a post-maintenance test or system opening demands an immediate decision — whether to proceed, defer, or escalate — with consequences for the critical path, regulatory compliance, and resource availability that must be assessed within minutes. This paper presents a cross-phase analytical framework that addresses both challenges using a unified architecture built on historical condition report records, work order histories, and outage schedule data as primary inputs. In the planning phase, a proactive pipeline processes multi-cycle condition report and work order history through a sequence of analytical stages that construct a component knowledge graph, assess component health state trajectories by analysing degradation trends across successive outage cycles, and compute a causal score combining degradation frequency, prior emergent work precedent, and component criticality. Components exhibiting escalating health deterioration are identified even when no prior unplanned maintenance record exists — a key prognostic capability for early detection of first-failure components that would otherwise receive zero risk score under conventional evidence-based methods. The pipeline outputs a confidence-tiered risk register supporting pre-outage decisions on parts pre-ordering, contingency crew allocation, and scope additions weeks before the outage window opens. In the execution phase, a reactive pipeline accepts a free-text condition report as its sole input and produces a structured recommendation — proceed, defer, escalate, or monitor. The recommendation is supported by a duration estimate derived from semantic retrieval of analogous historical work orders fitted to a lognormal distribution and reported at the expected and eightieth-percentile levels alongside an analog count as a calibrated confidence signal. Schedule impact is assessed through Monte Carlo simulation of the resource-constrained outage schedule, and applicable regulatory constraints are evaluated automatically to identify options that cannot be deferred. The two pipelines share the same knowledge graph, so a component pre-staged in the planning phase enters the execution phase with resources already in position, directly reducing mobilization delay when an unexpected activity materializes. In a demonstration case, this cross-phase coordination reduced the total critical-path extension of an emergent repair activity by fifty-nine percent relative to a fully reactive response. Beyond nuclear outage management, the framework's architecture — combining a component health state prognostic layer with a probabilistic schedule impact model — is applicable to any asset-intensive operation where scheduled maintenance windows create tensions between planned scope, emergent findings, and constrained resources, offering a generalizable approach to evidence-based maintenance decision support under uncertainty.

---

**Title:**  
*A Cross-Phase Prognostic Framework for Emergent Work Risk Assessment and Schedule Decision Support in Nuclear Refueling Outages*

**Authors:** [TBD]  
**Track:** Prognostics and Decision Support — Maintenance and Operations  
**Keywords:** nuclear outage management, uncertainty quantification, emergent work, risk register, schedule impact, analog retrieval, prognostics

---

## Abstract

Unplanned activities discovered during nuclear refueling outages — emergent work — are a primary driver of schedule extension and cost overrun. Current practice offers no structured method to predict which components are likely to generate emergent work before the outage begins, nor to quantify the schedule impact of an unexpected activity once it is discovered. Decisions are made under time pressure, drawing on individual expertise and without a documented evidence chain.

This paper presents a cross-phase analytical framework that supports outage managers at both phases of the problem. In the planning phase, a proactive pipeline analyzes years of condition report and work order history to produce a ranked risk register of components likely to generate unplanned work, with recommended pre-staging actions. In the execution phase, a reactive pipeline accepts a free-text condition report description as input and produces — within two minutes — a structured recommendation (PROCEED / DEFER / ESCALATE / MONITOR) with a quantified schedule impact, regulatory constraint assessment, and confidence-tiered duration estimate based on historical analogs.

The two pipelines share a common knowledge graph and evidence architecture, enabling a key organizational capability: a component flagged in the planning phase automatically informs the execution-phase recommendation for the same component. In a cross-phase demonstration using a synthetic nuclear plant dataset, pre-staging resources for a flagged component reduced the effective critical-path impact of the resulting emergent activity from 23.0 hours to 6.0 hours — a 59% reduction — by eliminating mobilization delay and parts-wait time.

The framework is grounded in uncertainty quantification: duration estimates are drawn from semantic retrieval of historical analogs and fitted to a lognormal distribution, reporting p50 (expected duration), p80 (contingency anchor), and analog count as a confidence signal. Schedule impact is propagated through Monte Carlo simulation of the resource-constrained project schedule, producing probabilistic finish-time distributions rather than single-point estimates.

---

## 1. Introduction and Motivation

Refueling outages in light-water reactors run on critical-path schedules where every unplanned hour of extension costs hundreds of thousands of dollars. Emergent work — maintenance discovered mid-outage when systems are opened for inspection — is structurally unavoidable: wear, packing leakage, and instrumentation drift appear only after the outage window opens. The industry challenge is not eliminating emergent work but managing it with speed, evidence, and confidence.

Two gaps motivate this framework:

1. **Reactive-only practice.** Pre-outage planning today lacks a data-driven mechanism to predict which components will generate unplanned work. Resources — parts, specialized crew, contingency schedule float — are not pre-positioned for high-probability failures.

2. **Unquantified execution decisions.** When an unexpected activity is discovered, the outage manager must decide within minutes whether to begin the work now, defer it within the outage window, defer it post-outage, or escalate. This decision is made without a structured assessment of likely duration, schedule impact, regulatory constraints, or precedent from prior outages.

The framework described here addresses both gaps with a unified architecture.

---

## 2. Framework Architecture

The framework consists of two pipelines operating on a shared knowledge graph.

### 2.1 Planning Pipeline — Pre-Outage Risk Register

The planning pipeline processes historical condition reports, work orders, and maintenance activity records through seven stages (A–G):

**Stage A — Data Ingestion:** Loads and validates component lists, CR records, WO history, and outage schedule data. Tags each historical activity with an emergence category (DISCOVERY, SCOPE_EXPANSION, REGULATORY, OPTIMIZATION). Identifies regulated components.

**Stage B — NLP Extraction:** Resolves plant-specific abbreviations; extracts named entities (equipment IDs, failure mode keywords, CR/WO cross-references). Computes an unknown-token rate to flag data quality issues before analysis.

**Stage C — Knowledge Graph Construction:** Builds an in-memory knowledge graph linking components → CRs → WOs → activities across training outage cycles. The graph is the shared data substrate between the planning and execution pipelines.

**Stage D — Temporal Trend Analysis** *(new contribution)*: Analyzes degradation CR frequency, severity escalation (observation → degradation), and WO duration overruns per component across training cycles. Produces a composite trend score and escalation flag. This stage is the key differentiator: a component with zero prior emergent work but a clear escalating degradation trend is flagged for proactive attention — without Stage D it would be assigned zero risk.

**Stage E — Causal Chain Scoring:** Applies an evidence-based causal formula:

```
causal_score = (outages_with_degradation_CR / training_outages)
             × (outages_with_emergent_activity / outages_with_degradation_CR)
             × criticality_weight
```

Components with `causal_score ≥ 1.5` are DATA-SUPPORTED; `≥ 0.5` are SME-INFORMED.

**Stage F — Schedule Risk Contextualization:** Retrieves each flagged component's historical critical-path float consumption across prior outages, quantifying how often the component has consumed critical-path time.

**Stage G — Risk Register and Recommendations:** Assigns a confidence tier, generates a finding and recommended pre-staging action per component, and produces a ranked risk register for delivery to the outage planning team weeks before the outage window opens.

### 2.2 Execution Pipeline — In-Outage Activity Triage

The execution pipeline accepts a free-text condition report as the sole input and runs through a parallel seven-stage structure:

**Stage A — Activity Intake:** NLP entity extraction; regulatory keyword detection (Technical Specification identifiers); emergence type classification; data quality scoring.

**Stage C — Temporal Event Chain:** Retrieves component history from the shared knowledge graph; classifies prior events using Allen interval algebra (PRECEDES, OVERLAPS, CONTAINS, DURING, FOLLOWS); scores causal strength. Produces a causal posture: *established*, *partial*, or *insufficient_data*.

**Stage D — Historical Analog Retrieval:** Semantic similarity search over historical work orders retrieves analogous activities. Fits a lognormal duration distribution to retrieved execution times. Reports p50 / p80 / p90 and a confidence tier based on analog count (DATA-SUPPORTED: ≥5 analogs; SME-INFORMED: 1–4; LOW-CONFIDENCE: 0).

**Stage E — Schedule Impact Assessment:** Computes available float consumption, critical-path drag, displaced tasks, and regulatory task conflicts. Identifies the criticality label (critical / near-critical / non-critical).

**Stage F — Insertion Option Generation and Scoring:** Generates all viable options (insert now, defer in-outage, defer post-outage, scope reduction, escalate). Scores each option's risk on [0,1]. Applies regulatory clearance checks — options that violate Technical Specification defer-prohibition constraints are blocked.

**Stage G — Recommendation Synthesis:** Selects the lowest-risk feasible option. Maps to a decision status (PROCEED / DEFER / ESCALATE / MONITOR). Produces an executive summary, analyst attention flags, and a full evidence chain linking the recommendation to its supporting source records.

---

## 3. Cross-Phase Integration

The two pipelines are coupled through the shared knowledge graph. When the execution pipeline processes a condition report for a component previously flagged in the planning-phase risk register, the planning-phase evidence (causal score, trend label, recommended pre-staging action taken) is incorporated into the Stage G evidence chain and executive summary. This connection enables a key capability: **pre-staged resources directly reduce the execution-phase critical-path impact**.

The impact delta is quantified:

| Scenario | Mobilization time | Total CP extension |
|---|---|---|
| Reactive (no pre-staging) | 8.5 h | 23.0 h |
| Pre-staged (planning-phase flag acted on) | 0.0 h | 6.0 h |
| **Saving** | **8.5 h** | **17.0 h (59%)** |

This quantified saving translates directly to avoided extension costs and provides a measurable return-on-investment signal for the planning-phase analytics investment.

---

## 4. Uncertainty Quantification Design

Duration estimates are not point values — they are distributions with calibrated confidence signals:

- **Analog count** is the primary confidence signal. Zero analogs triggers the MONITOR decision rather than a fabricated estimate. This prevents false precision in novel failure scenarios.
- **p80 as the contingency anchor.** Consistent with DOE Order 413.3B and PMBOK schedule risk guidance, the p80 (not the p50) is used to set contingency buffers. The p80−p50 gap provides a quantified buffer size rather than an engineering-judgment padding factor.
- **Confidence tier propagation.** The duration confidence tier (Stage D) propagates to the final recommendation confidence tier (Stage G). A DATA-SUPPORTED schedule recommendation rests on ≥5 historical analogs; an SME-INFORMED one signals a need for domain-expert validation before commitment.

---

## 5. Preliminary Results

Evaluated on three synthetic outage scenarios across the Millbrook synthetic plant dataset:

| Scenario | Condition | Expected Decision | Pipeline Decision | Analog Count | p80 Duration |
|---|---|---|---|---|---|
| RCP Train-A Seal Leak | TS defer-prohibited; CP drag 14 h | ESCALATE | ESCALATE | 18 | 22.0 h |
| Snubber Scope Expansion | Non-critical; 28 h float remaining | DEFER | DEFER | 5 | 9.5 h |
| Unknown Component Leak | Zero analogs; empty KG; safety-related | MONITOR | MONITOR | 0 | — |

Pre-outage risk register precision on RF-22 holdout: 2 true positives (DATA-SUPPORTED, SME-INFORMED), 1 defensible false positive (escalating trend flagged without prior emergent record), 2 true negatives — consistent with a conservative risk-identification strategy appropriate for safety-critical maintenance planning.

---

## 6. Contribution to PHM Practice

This work contributes: (1) a cross-phase framework that connects proactive risk identification to reactive decision support through a shared knowledge graph; (2) a temporal trend analysis stage (Stage D) that detects components approaching failure before their first emergent work record appears; (3) a calibrated UQ architecture — analog retrieval → lognormal fitting → p50/p80 reporting — that provides actionable confidence signals rather than binary risk flags; (4) a quantified cross-phase impact model demonstrating that planning-phase pre-staging reduces execution-phase critical-path extension.

The full paper will include a complete description of both pipelines, extended evaluation on a larger synthetic dataset, sensitivity analysis on confidence tier thresholds, and a discussion of integration with plant CMMS systems (Maximo, SAP PM) for real-world deployment.

---

*Submitted to PHM 2026 — Annual Conference on Prognostics and Health Management*  
*Track: Prognostics and Decision Support — Maintenance and Operations*
