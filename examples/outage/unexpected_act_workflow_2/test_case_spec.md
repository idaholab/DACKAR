# Test Case Specification: MVP Outage Risk Prediction
**Version:** 3.0 (Revised — Scope Narrowed for First Meeting)
**Status:** Draft for internal review

---

## Preface: Current Demo Scope and Deferred Items

### What This Demo Is
This document describes an MVP test case with two distinct phases of maturity. The current phase — suitable for a first meeting with plant managers — is a **workflow and methodology demonstration** using a small, transparent, explicitly synthetic dataset. Its purpose is to show how the system reasons, not to report validated results. It ends with a concrete data request that enables the real analysis to begin.

### What Is Currently in Scope
- End-to-end workflow demonstration on a small synthetic illustrative dataset (≈ 5 components, ≈ 15 CRs, ≈ 3 WOs, ≈ 2 emergent activities)
- Demonstration of pipeline stages: ingestion → NLP extraction → graph construction → candidate causal chain → recommendation output
- Mockup or wireframe of user-facing outputs: risk register, evidence drilldown, confidence tier display
- Architecture and data source overview
- Explicit data request to plant for real dataset access

### What Is Explicitly Deferred Until Real Plant Data Is Available
The following items appear in this spec as internal planning targets. They must not be presented to stakeholders as current capabilities or validated findings until a real dataset has been obtained under an appropriate data sharing agreement:

| Deferred Item | Reason |
|---|---|
| Quantitative success metrics (recall, precision@10, chain coverage) | Meaningless without real ground truth |
| Retrospective holdout test and blind evaluation | Requires real multi-outage dataset |
| Delay estimation metric | No real baseline to compare against |
| SME usefulness rating session | Synthetic scenarios cannot produce valid usefulness ratings |
| Pattern frequency claims | Synthetic frequencies are constructed, not discovered |
| Cross-outage generalization claims | Requires ≥ 3 real outages on same unit |

> ⚠️ If any of the deferred items above appear in materials shown to plant managers before real data is available, they must be explicitly labeled as illustrative targets, not measured results. Presenting synthetic demo outputs as validated findings is the single most damaging thing that could happen to stakeholder trust at this stage.

### Framing for the First Meeting
The first meeting should be positioned as: *"We are here to show you how this system works and what it will produce. Your data is what makes it real. We would like to leave today with an agreement on what data we need and how to access it."*

This framing is stronger than a results demo because it makes plant managers participants rather than evaluators, and it sets honest expectations for what comes next.

---

## 0. What This Test Case Is Not

This test case does not claim to replace planner judgment. It does not model regulatory hold points, Technical Specification surveillance requirements, or contractor resource constraints. It does not account for craft availability or crew fatigue limits. It demonstrates **pattern recognition, evidence traceability, and schedule-aware risk flagging** on historical outage data.

These boundaries should be stated explicitly at the start of any stakeholder presentation.

---

## 0.1 Expected Imperfections in MVP Output

The following failure modes are expected and should be stated proactively in any stakeholder presentation. Naming them first disarms criticism and demonstrates analytical honesty.

- Some high-risk components will not be flagged due to missing or incomplete CR/WO linkage in the historical record
- Some flagged components will not result in emergent work — false positives are expected and their rate is part of what the pilot measures
- Candidate causal chains may be incomplete where CR or WO history is sparse, poorly linked, or stored in disconnected systems
- Schedule impact scores will be directionally informative but not numerically precise in the MVP — they should be treated as relative risk rankings, not point estimates
- NLP extraction quality will vary across activity descriptions depending on abbreviation density and text structure

> These are not bugs to be fixed before the demo. They are honest characterizations of MVP scope that build rather than undermine credibility when stated plainly.

---

## 1. Goal of the Test Case

### Dual Purpose
This test case serves two distinct purposes that must be kept operationally separate:

- **Workflow validation:** Confirm that each pipeline stage (ingestion, NLP extraction, graph construction, prediction, recommendation) produces correct, logged, and reproducible outputs against defined exit criteria.
- **Showcase demonstration:** Provide a single, clean, end-to-end narrative scenario suitable for presentation to plant managers at a first meeting.

> ⚠️ These two purposes should never be conflated in a live presentation. Workflow stress-testing (intentional error injection, incomplete data handling, null graph results) must not be visible during the showcase walkthrough. Build one system with two presentation modes.

### Primary Capability Goals
The system must demonstrate ability to:
- Identify high-risk components and systems before an outage begins
- Quantify probability of emergent work and expected schedule impact relative to critical path
- Provide fully traceable evidence chains (CR → WO → Activity → Schedule impact)
- Generate actionable, tiered recommendations with explicit confidence levels

### Anchor Demonstration Scenario — Current Scope
For the first meeting, the anchor scenario is a **small, fully transparent synthetic illustration** with the following properties:

- Approximately 5 components, 15 condition reports, 3 work orders, 2 emergent activities
- Small enough that the full dataset can be shown in a single table if asked
- Explicitly introduced in the presentation as a constructed example: *"This is an illustrative dataset we built to show how the system reasons. Your plant's data is what produces real findings."*
- Designed to exercise the full reasoning chain — CR history → candidate causal link → recommendation — without claiming any pattern is real

The narrative arc below applies to the **Phase 2 real-data demo** once plant data is available:

> *"This component had [N] condition reports over [X] months. Historically, that pattern preceded emergent work in [Y]% of similar cases across the training outages. The system would have flagged it [Z] weeks before the outage. Here is the recommendation it would have generated. The emergent work did occur. It consumed [H] hours of schedule float."*

---

## 2. Success Metrics

### Framing Principle
Metrics must be framed in operational terms, not classifier performance terms. A "40% false positive rate" is technically reasonable but will be received as *"four out of ten things you flag are wrong"* by a plant manager. Reframe every metric around planner-actionable value.

### Quantitative Targets

> ⚠️ The targets below are internal planning thresholds for validation against real plant data. They must not appear in stakeholder-facing materials until the retrospective test in Phase 3 has been completed on a real dataset.

| Metric | Target | Framing for Stakeholders |
|---|---|---|
| Emergent activity detection recall | ≥ 60% | Of emergent activities that occurred, the system flagged the component pre-outage in at least 6 of 10 cases |
| Top-10 risk list quality | ≥ 6 of 10 correspond to actual emergent or expanded scope work | At least 6 of our top 10 pre-outage flags corresponded to real work scope changes |
| Delay estimation error | ±30% vs. actuals — **only report if current planner baseline is known** | System estimates are within [X]% of actual delay, compared to current manual baseline of [Y]% |
| Evidence trace coverage | 100% | Every recommendation links to specific source records — no black-box outputs |
| SME recommendation usefulness | ≥ 3 of 5 reviewing SMEs rate top-5 recommendations as actionable or worth investigating | Defined by structured SME review session with explicit rating rubric |

> ⚠️ **On delay estimation:** Do not report this metric in the showcase unless you have established what experienced planners currently achieve by hand. If the system's ±30% is worse than the human baseline, the metric is counterproductive. If the baseline is unknown, drop this metric from the first meeting and note it as a benchmarking objective for the pilot phase.

---

## 3. Data Requirements

### Minimum Dataset for Workflow Validation

| Requirement | Minimum | Notes |
|---|---|---|
| Outages | ≥ 3 on same unit | Single-outage validation is insufficient for any pattern frequency claim |
| Activities | ≥ 5,000 | Across all outages |
| Emergent activities | ≥ 200 | With categorical tags (see Section 4) |
| Condition Reports | ≥ 1,000 | Linked to components where possible |
| Work Orders | ≥ 1,000 | With component and system references |
| Components | ≥ 500 | With canonical plant tag IDs |
| Sister unit / comparable plant | ≥ 1 additional outage | Required before any cross-plant generalization claim |

### On Synthetic Data
If real data is insufficient to meet the minimums above, synthetic records may be used to complete the dataset. The following rules apply:

- Synthetic records must conform to real plant naming conventions, work order numbering formats, and activity description style (including realistic abbreviations and alphanumeric IDs)
- The proportion of synthetic vs. real records must be documented and disclosed proactively in any stakeholder presentation
- Synthetic condition report descriptions must be reviewed by at least one plant SME for realism before use
- The showcase anchor scenario must be traceable to **real records only**, even if the broader dataset contains synthetic fill-in
- **Synthetic data must not influence model learning in a misleading way.** Synthetic records are permitted only for pipeline completion and volume balancing. They must not be used as the basis for any pattern frequency claim, causal chain identification, or key analytical finding presented to stakeholders. All such claims must be traceable exclusively to real records.

### Core Data Tables

#### Activity
- `activity_id` — alphanumeric, unique
- `activity_name` — free text, raw plant format
- `role_id` — ELEC / MECH / OPS / etc.
- `planned_start`, `planned_end`
- `actual_start`, `actual_end`
- `emergent_flag` — boolean
- `emergence_category` — see Section 4 *(new field)*

#### Schedule
- `task_id`
- `critical_path_flag`
- `total_float_hours`
- `float_consumed_hours` *(actual vs. planned)*

#### Condition Reports
- `cr_id`
- `description` — free text
- `created_date`
- `component_id`
- `cr_category` — degradation / corrective / observation

#### Work Orders
- `wo_id`
- `description` — free text
- `component_id`
- `system_id`
- `wo_type` — preventive / corrective / emergent

#### Asset / Component
- `component_id`
- `plant_tag` — canonical plant ID
- `system`
- `regulatory_constraint_flag` — boolean *(new field — see Section 4)*

---

## 4. Data Quality and Normalization Requirements

### Emergent Activity Categorization *(Critical — must precede any causal analysis)*
The binary planned/emergent split used in the current pipeline is insufficient. Before causal analysis or pattern mining, every emergent activity must be tagged with one of the following categories:

| Category | Description |
|---|---|
| `DISCOVERY` | Unexpected degradation or condition found during inspection |
| `SCOPE_EXPANSION` | A planned activity expanded due to findings (e.g., gasket replacement becomes valve body repair) |
| `REGULATORY` | NRC commitment, Tech Spec action level, or regulatory-driven addition surfacing mid-outage |
| `OPTIMIZATION` | Formally approved schedule move or scope addition that does not reflect a failure |

Conflating these categories will contaminate causal inference. `REGULATORY` and `OPTIMIZATION` activities should be excluded from emergent pattern modeling in the MVP.

### Regulatory Constraint Flagging *(Critical — must be visible in recommendation outputs)*
Any component or activity linked to a Technical Specification surveillance requirement, NRC commitment, or regulatory hold point must be flagged. The recommendation engine must not suggest deferral or reprioritization of flagged items without an explicit override acknowledgment. This flag does not need to capture the full regulatory context in the MVP — a boolean is sufficient — but its absence from an output must be visible to the user.

### NLP Extraction Quality Gate
After abbreviation expansion and NER, the following must be measured and reported as a pipeline health metric:

- **Unknown token rate:** The proportion of non-stopword tokens in activity descriptions that match neither a dictionary entity nor an alphanumeric ID pattern
- **Exit criterion:** If unknown token rate exceeds 25% across the dataset, the NLP extraction results are unreliable and downstream graph construction must be flagged as low-confidence
- **Minimum acceptable:** Unknown token rate ≤ 15% for the showcase anchor scenario specifically

---

## 5. Expected System Behavior

### Prediction
- Output: probability of emergent work, confidence tier
- Confidence tiers: `DATA-SUPPORTED` / `SME-INFORMED` / `LOW-CONFIDENCE WATCH`
- Higher risk scores with repeated CRs on same component across multiple outages
- All predictions must degrade gracefully with missing data — no silent failures
- **Delay estimation is exploratory in the MVP and is not a primary output.** Primary value is risk identification and evidence traceability. Delay estimation may be noted as a future capability but should not appear in evaluation metrics or stakeholder-facing outputs at this stage.

### Causality
- Identify and score **candidate causal chains** based on temporal ordering and contextual co-occurrence across CR → WO → Activity sequences
- Causal strength = evidence-weighted likelihood, not proof. This must be stated explicitly in any output label or UI element that references causality
- Assign causal strength with explicit evidence count (e.g., "this pattern observed in 4 of 6 prior outages")
- Surface the specific source records behind each causal link

### Schedule Integration
Risk index = P(emergent event) × criticality multiplier

Where criticality multiplier is derived from the historical distribution of float consumption across activity types and schedule positions in the training outages, and calibrated per dataset. Fixed reference values (e.g., non-critical = 1.0, critical path = 5.0) are used only as initialization priors before calibration and should not be presented as validated thresholds.

> Note: frequency of entity occurrence (as shown in paper Figures 2–3) is a starting point only. The primary planning metric is **float consumption on the critical path**, not activity count. Features and outputs must be oriented around this.

### Recommendations
Each recommendation must include:
- Category: Preventive / Mitigation / Operational / Investigative
- Confidence tier (see below) with interpretation guide
- Specific source records it is drawn from
- Number of historical outages supporting the pattern
- Regulatory constraint flag if applicable
- "Reject with reason" field for SME feedback (feeds learning loop)

### Confidence Tier Interpretation Guide

| Tier | Meaning | Appropriate Stakeholder Action |
|---|---|---|
| `DATA-SUPPORTED` | Pattern observed across multiple outages with strong CR/WO-to-activity linkage | Treat as a planning input; consider pre-staging resources or inspection |
| `SME-INFORMED` | Partial data linkage; pattern is consistent with domain knowledge but not fully evidenced in records | Flag for SME review before acting; do not ignore |
| `LOW-CONFIDENCE WATCH` | Weak data linkage or single-outage observation; included for awareness only | Monitor but do not drive planning decisions; revisit as more data is available |

> Low-confidence items must never be highlighted as primary findings in a stakeholder presentation. They belong in a watchlist appendix, not the lead narrative.

---

## 6. Pitfalls and Known Risks

### Data Issues
- Identifier ambiguity across outages or units
- Poor free-text quality (realistic plant activity names are highly compressed and inconsistent)
- Missing or misaligned timestamps between schedule and activity systems
- Unknown synthetic data contamination if not carefully tracked

### Modeling Issues
- Overfitting to a single outage's character (fleet bulletin-driven patterns, one-time regulatory items)
- Frequency bias — high-frequency entity mentions ≠ high schedule impact
- Ignoring schedule context — emergent work off the critical path is often irrelevant to outage duration
- Causal conflation across emergence categories (see Section 4)

### System Issues
- Knowledge graph visualizations are not self-explanatory to non-technical audiences — never show a raw graph screenshot without a guided walkthrough
- Overconfident outputs without visible uncertainty will destroy trust with experienced planners
- Recommendation outputs that ignore regulatory constraints are a compliance liability

### Organizational Issues
- SME availability is constrained by the outage planning calendar — all SME-dependent tasks must be scheduled during the operating cycle, not within 6 months of outage execution
- Plant manager audiences will ask whether the demo data is real — have a clear, honest answer prepared in advance
- **Mismatch with existing planning workflow tools (e.g., Primavera P6):** If system outputs cannot be consumed within the tools planners actually use day-to-day, adoption will fail regardless of analytical quality. Risk identification and recommendations must be exportable in a format compatible with the plant's scheduling environment. This integration path should be scoped explicitly before pilot deployment, not after.

---

## 7. Test Execution Plan

### Holdout Structure *(must be explicitly documented for stakeholder credibility)*
- **Training set:** Outages N-1 and N-2 (minimum)
- **Test set:** Outage N, held out completely during model development
- The test must be genuinely blind — no parameter tuning or threshold adjustment using outage N data
- Ground truth for outage N must be established from actual emergent activity records before model outputs are reviewed

### Execution Steps
1. Select pilot plant and confirm data availability across ≥ 3 outages
2. Tag all emergent activities with emergence categories (Section 4)
3. Flag all regulatory-constrained components and activities
4. Build pre-outage dataset for holdout outage N (cut off at T-0 of outage start)
5. Run full pipeline: ingestion → normalization → NLP extraction → graph construction → prediction → recommendation
6. Log pipeline health metrics at each stage (including unknown token rate)
7. Capture all outputs before comparing against ground truth
8. Run structured SME review session using rating rubric
9. Document false positives and false negatives with root cause

### Showcase Walkthrough Path *(separate from validation run)*
Script the anchor scenario completely before the meeting. Walk through:
- **Slide / screen 1:** The component and its condition report history
- **Slide / screen 2:** The graph evidence chain (simplified flow diagram, not raw Neo4j output)
- **Slide / screen 3:** The pre-outage recommendation the system would have generated
- **Slide / screen 4:** What actually happened and the schedule impact

Consider embedding a 60-second screen recording of the live graph query in the PPT rather than static screenshots.

### Demo Constraints *(must be followed by anyone presenting)*

- **Do not show raw graph visualizations** — a Neo4j node-edge diagram is not interpretable by a non-technical audience and will derail the narrative
- **Do not discuss model internals unless directly asked** — if asked, give a one-sentence answer and redirect to the evidence chain
- **Do not present unvalidated causal claims** — use "candidate causal link" or "evidence-supported pattern" language at all times
- **Do not highlight low-confidence predictions as primary findings** — these belong in a watchlist appendix only
- **Do not improvise on data provenance questions** — if asked which records are real vs. synthetic, give the prepared, documented answer from Section 3

---

## 8. Evaluation Metrics

| Metric | Method | Threshold |
|---|---|---|
| Precision@10 | Top-10 risk flags vs. actual emergent work | ≥ 6 of 10 |
| Recall | Flagged components / total emergent components | ≥ 60% |
| MAE (delay hours) | Only if planner baseline is known | Report relative to baseline |
| Evidence trace coverage | Fraction of recommendations with ≥ 1 source record | 100% |
| SME usefulness | Structured rating: actionable / worth investigating / not useful | ≥ 3 of 5 SMEs rate top-5 as actionable or worth investigating |
| NLP unknown token rate | Pipeline health metric | ≤ 15% on anchor scenario, ≤ 25% overall |
| Precursor chain coverage | % of emergent activities linked to ≥ 1 upstream CR or WO | Target ≥ 20–30%; coverage gaps must be explicitly reported and root-caused |
| Pipeline runtime | Wall-clock time from data ingestion to recommendation output | To be measured on pilot dataset — do not report to stakeholders until benchmarked |
| Time to insight (user-facing) | Time for a planner to locate, interpret, and act on a top-5 recommendation | Target ≤ 10 minutes from login to decision; to be measured in SME review session |

---

## 9. Acceptance Criteria

### Technical
- System identifies ≥ 1 real high-impact emergent risk in the holdout outage
- Every recommendation links to specific, named source records
- Pipeline completes without silent failures on incomplete or noisy input data
- Regulatory constraint flags are visible on all relevant outputs

### Operational
- At least one SME rates the anchor scenario recommendation as something they would have acted on pre-outage
- Plant manager audience can follow the anchor scenario narrative without technical background
- Synthetic data proportion is documented and disclosed without prompting

### Boundaries Confirmed
- Deferral recommendations are blocked for regulatory-flagged items
- Confidence tiers are visible on all outputs
- Reject/feedback workflow is functional for SME review session
