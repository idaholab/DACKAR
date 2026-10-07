# Title: Condenser Vacuum Loss with Turbine Load Runback — PWR Unit 2

# Scope of this Scenario
It has four properties that make it analytically hard and instructive:
1) Multi-timescale causality. The true root cause develops over 14 days as a slow precursor before producing an acute event. This tests TSKR's ability to distinguish gradual drift from initiating cause, and the synthesizer's ability to surface the precursor as causal rather than symptomatic.
2) Three plausible competing hypotheses with different evidence postures. Air in-leakage, condenser tube fouling, and circulating water temperature elevation all produce overlapping telemetry signatures. Documents support different hypotheses depending on which you privilege. This tests the ranking and discrimination logic under mixed evidence.
3) Maintenance change analysis. A condenser waterbox cleaning 21 days prior creates a legitimate maintenance-induced hypothesis. Combined with a surveillance overdue condition, this tests whether governance scoring meaningfully differentiates candidates rather than uniformly boosting all of them.
4) Recurrence with a twist. A similar event occurred 18 months ago on the same unit, but its confirmed root cause was condenser tube fouling — the opposite of this event's true cause. This tests whether the recurrence analysis strengthens or misleads the ranking.

# Scenario narrative
Unit 2 is at 97% rated power during steady-state operation, mid-summer. Over a 14-day period, the plant computer shows a slow monotonic rise in condenser backpressure from a baseline of 1.8 inHg to 2.9 inHg. Turbine heat rate degrades proportionally. Operators initially attribute the trend to seasonal circulating water temperature rise and increase circulating water flow as a compensatory action.
On Day 14, an automated turbine load runback initiates when backpressure reaches the 3.0 inHg setpoint. Reactor power reduces to 85% under automatic control. A Condition Report is generated. Post-event investigation reveals the backpressure rise had two contributors: a degraded expansion joint at the turbine exhaust duct-to-condenser neck interface allowing air in-leakage, compounded by mildly elevated circulating water temperature. The air in-leakage is the dominant cause. The HVAC system serving the turbine building condenser pit has been operating at reduced capacity since a fan motor bearing failure 10 days prior, elevating local ambient temperature and accelerating expansion joint seal degradation.
The waterbox cleaning 21 days prior is a red herring — inspection records confirm tube cleanliness and chemistry data shows no evidence of tube-side fouling. The 18-month-ago analog event was fouling, which makes the recurrence signal misleading if not carefully interpreted.

### Telemetry Summary

This is where the multi-timescale character of the event must be captured. You need signals at three timescales: the 14-day precursor drift, the short-term acceleration in the 48 hours before the event, and the acute runback transient.

**Signals to include and their anomaly structure:**

**U2-PT-1847A — Condenser backpressure (primary signal)**
- Baseline mean: 1.82 inHg, std: 0.04 over prior 30 days
- Window mean: 2.41 inHg, std: 0.31 (massive sigma deviation)
- Anomaly 1: `gradual_drift`, timestamp Day 1, severity `low`, severity_score 0.35
- Anomaly 2: `sustained_exceedance`, timestamp Day 12, severity `high`, severity_score 0.88
- Changepoint: Day 1 of precursor period, confidence 0.91

**U2-AIT-0341 — Hotwell dissolved oxygen**
- Baseline mean: 6.2 ppb. Window mean: 89.4 ppb
- This is the most diagnostically important signal. DO elevation in the hotwell is a classic indicator of air in-leakage, not tube fouling. The synthesizer should treat this as strong discriminating evidence.
- Anomaly: `gradual_drift` → `sustained_exceedance`, onset Day 3, severity `high`, severity_score 0.94
- A good system should recognize that condenser tube fouling does NOT produce elevated DO — this contradicts the fouling hypothesis.

**U2-TE-2201 — Turbine exhaust temperature**
- Correlated with backpressure rise. Gradual drift, onset Day 2.

**U2-FT-3301A/B — Condensate flow (both trains)**
- Slight reduction consistent with reduced steam flow, not diagnostic by itself.

**U2-TE-4401 — Circulating water inlet temperature**
- Seasonal rise of 4.2°F over the window. This is real but insufficient alone to explain a 1.2 inHg backpressure rise. The synthesizer must recognize this as a contributing factor, not root cause.
- No anomaly flag — it's within seasonal normal range.

**U2-TE-5501 — Condenser pit ambient temperature**
- Rises 8°F over Day 4 to Day 10 of the precursor window, corresponding to when the HVAC fan motor bearing began degrading. This links the HVAC degradation to the expansion joint environment.
- Anomaly: `gradual_drift`, severity `medium`, onset Day 4.

**U2-VIB-7701 — HVAC fan motor bearing vibration (turbine building bay)**
- Step change upward on Day 4, followed by sustained high vibration until Day 10 when the fan trips on high vibration.
- Anomaly: `step_change` then `sustained_exceedance`. Onset Day 4, severity `high`.

**Signals with NO anomalies (important for hypothesis discrimination):**
- Condenser tube outlet temperatures (both waterboxes): within normal limits — contradicts fouling hypothesis
- Circulating water outlet temperature: normal delta-T — contradicts fouling
- Condensate specific conductivity: normal — no tube leakage
- Feedwater flow: stable until the runback

The `overall_assessment` should flag the DO elevation as the most anomalous signal (earliest meaningful onset, highest diagnostic specificity) and note that condenser chemistry signals are inconsistent with tube fouling.

---

### KG Context Elements

**Components to include:**
- `U2-CND-EXPANSION-JOINT-EXHAUST` — primary component, seed match
- `U2-CND-HOTWELL` — downstream
- `U2-CND-WATERBOX-A`, `U2-CND-WATERBOX-B` — alternate causal path (fouling hypothesis)
- `U2-AIR-EJECTOR-A`, `U2-AIR-EJECTOR-B` — upstream of backpressure; if air ejectors are working normally this contradicts air ejector failure as cause
- `U2-HVAC-TURBINE-BAY-FAN-A` — supporting system, component whose degradation contributes
- `U2-SG-1`, `U2-SG-2`, `U2-SG-3`, `U2-SG-4` — upstream; included to show system boundary

**Failure modes to include (this is where the discrimination happens):**

| fm_id | name | component | expected latency | expected symptoms | expected anomaly pattern |
|---|---|---|---|---|---|
| FM-CND-AIR-INLEAK | Air in-leakage through boundary | expansion joint | 48–336 hrs | pressure, temperature | gradual_drift |
| FM-CND-TUBE-FOUL | Condenser tube fouling | waterbox | 168–720 hrs | pressure, flow, temperature | gradual_drift |
| FM-CND-TUBE-LEAK | Condenser tube leakage | waterbox | 2–48 hrs | electrical (conductivity) | step_change |
| FM-CW-TEMP-RISE | CW inlet temperature elevation | CW system | seasonal | pressure, temperature | gradual_drift |
| FM-HVAC-DEGRAD | HVAC cooling capacity reduction | HVAC fan | 24–120 hrs | temperature | gradual_drift |

Note that `FM-CND-AIR-INLEAK` and `FM-CND-TUBE-FOUL` have overlapping latency windows and symptom types — both look like `gradual_drift` in backpressure. The discriminating signal is the DO sensor, which only the air in-leakage mode predicts.

## Past events — the recurrence trap

Past Event 1: EVT-U2-2022-1103
  asset_id: U2-CONDENSER-MAIN
  timestamp: 18 months prior
  matched_failure_mode_ids: [FM-CND-TUBE-FOUL]   ← confirmed fouling
  resolved: true
  severity: MEDIUM

Past Event 2: EVT-U2-2021-0612
  asset_id: U2-CONDENSER-MAIN
  timestamp: 36 months prior
  matched_failure_mode_ids: [FM-CND-AIR-INLEAK]  ← confirmed air in-leakage
  resolved: true
  severity: HIGH

Past Event 3: EVT-U2-2019-1847
  asset_id: U2-CONDENSER-MAIN
  timestamp: 60 months prior
  matched_failure_mode_ids: [FM-CND-AIR-INLEAK]
  resolved: true
  severity: HIGH


## Operational Context
The alarm sequence is analytically important: the HVAC vibration alarm on Day -10 precedes the accelerated backpressure rise, and an operator documented that increasing CW flow had "minimal effect" — which is evidence against CW temperature being the dominant cause.

## PM Compliance
This is where the governance scoring should become candidate-specific rather than generic.
The key discriminating design here: the failed PM items are specific to air in-leakage precursors — expansion joint inspection overdue, air ejector surveillance overdue, HVAC PM overdue. The tube inspection passed. A properly designed governance scorer should link these failed items to the air in-leakage and HVAC degradation candidates specifically, and the passed tube inspection should weakly contradict the fouling hypothesis. Your current uniform governance scorer won't do this — and this test case will expose that gap explicitly.

## Document corpus
CR-2024-04799 (operator trend CR, filed Day 9 of precursor)
    * Doc type: CR
    * Narrative: "Backpressure trend observed over past 9 days. Initial evaluation suggests possible CW temperature effect or condenser fouling. Recommending condenser performance test. No chemistry anomalies noted at time of writing."
    * Causal statement: "Backpressure rise may be due to reduced heat transfer from condenser fouling or elevated CW inlet temperature."
    * Note: Written before chemistry data showed DO elevation. This CR supports the fouling hypothesis but predates the discriminating evidence.

CR-2024-04821 (post-event CR, filed Day 14)
    * Doc type: CR
    * Narrative: "Turbine runback on 3.0 inHg backpressure. Hotwell DO found elevated at 142 ppb at time of event. Chemistry notified. Initial RCA direction: air in-leakage vs. tube fouling. WO issued for helium leak test of condenser boundaries."
    * Causal statement: "Elevated dissolved oxygen in hotwell is consistent with air in-leakage. Tube fouling contradicted by recent tube inspection results and normal tube outlet temperatures."
    * This CR contains the explicit discriminating statement. A good retriever should surface this for the air in-leakage candidate and treat it as supporting evidence.

WO-2024-11847 (waterbox cleaning work order)
    * Doc type: WO
    * As-found condition: acceptable
    * As-left condition: acceptable
    * Measurements: tube cleanliness score 0.94, zero tubes plugged
    * This WO actively contradicts the fouling hypothesis — a good evidence scorer should classify it as contradicting for fouling candidates.

WO-2024-12001 (helium leak test, filed after event)
    * Doc type: WO
    * As-found condition: failed — "Helium detected at expansion joint weld, north face, indicating active air in-leakage pathway"
    * This is the confirmation WO. It strongly supports the air in-leakage hypothesis. Its timestamp is after the event, so temporal scoring should treat it as confirmatory rather than precursor evidence.

SOP-U2-CND-001 — Condenser Performance Monitoring Procedure
    * Contains the guidance that backpressure trending above 2.5 inHg requires engineering evaluation
    * Contains the checklist for distinguishing air in-leakage from fouling including: "Check hotwell dissolved oxygen. DO above 20 ppb is indicative of air in-leakage, not tube fouling."
    * This SOP provides the explicit causal rule. A good retriever should surface this as supporting context for the air in-leakage hypothesis.

SOP-U2-CHE-041 — Secondary Chemistry Surveillance
    * Contains the acceptance criterion: hotwell DO < 10 ppb during normal operation
    * Supporting context for interpreting the DO anomaly

ECA-2022-1103 — Engineering Cause Analysis from 18-month-ago event
    * Confirmed root cause: tube fouling from biological growth in CW system
    * Corrective actions included: biocide treatment, tube cleaning, enhanced monitoring
    * This document will be retrieved by semantic search and could mislead a system that doesn't distinguish confirmed historical cause from current candidate

Industry OE document (INPO format)
    * Describes a fleet event at another PWR where condenser backpressure rise was initially attributed to CW temperature but was confirmed as expansion joint air in-leakage
    * Discriminating detail: "Hotwell dissolved oxygen elevation is the most reliable discriminator between thermal degradation and air in-leakage mechanisms"

## Expected Pipeline Behavior — Stage by Stage
Design the expected outputs explicitly so you can write automated assertions:
  * Stage B (KG context): Should return all 7 components listed above. Should return 5 failure modes. Should return 3 past events. Should return all 6 documents.

  * Stage C (TSKR): Should produce patterns showing gradual_drift relation for FM-CND-AIR-INLEAK and FM-CND-TUBE-FOUL with similar confidence. Should produce a step_change pattern for FM-HVAC-DEGRAD (fan trip on Day 10). Should NOT produce a contradiction for air in-leakage despite the long precursor window — the latency window for FM-CND-AIR-INLEAK includes 336 hours. Should produce a latency violation for FM-CND-TUBE-LEAK (too fast — tube leaks don't produce 14-day gradual drift).

  * Stage D (candidate generation, pre-refinement): Air in-leakage and tube fouling should be closely ranked because before evidence refinement their telemetry signatures are similar. The DO sensor anomaly is not yet feeding into the structural score at this stage. This is intentional — it tests that evidence refinement actually changes the ranking.
  
  * Stage E (evidence retrieval): The retrieved snippets should include the SOP DO threshold statement, CR-2024-04821's discriminating language, and WO-2024-11847's tube inspection results. The WO should be classified as contradicting for fouling candidates. CR-2024-04799 should be contextual (it predates the discriminating information).
  
  * Stage D refined: After evidence refinement, air in-leakage should clearly lead. Tube fouling's evidence score should drop due to contradiction from the WO. The score gap should exceed 0.10 — enough to trigger a clear primary selection.
  
  * Stage F (Ishikawa): Should populate equipment_hardware with both air in-leakage and tube fouling; maintenance_human_factors with HVAC PM overdue and expansion joint inspection overdue; process_procedure with the SOP deviation (surveillance overdue); environment_operating_context with the CW temperature seasonal rise and the HVAC degradation contribution.
  
  * Stage G (RCA card): Primary hypothesis: air in-leakage. Why_primary should explicitly reference DO elevation as discriminating. Alternatives should list tube fouling with weakness citing WO-2024-11847. Uncertainties should note that helium leak test WO was post-event and requires analyst confirmation. Analyst review questions should include asking whether the expansion joint inspection deferral constitutes a programmatic issue requiring a separate corrective action.

## Discriminating Test Assertions
Write these as explicit pass/fail checks on pipeline output:
  * Assertion 1 — Primary candidate correct: rca_card.primary_hypothesis.cause_label contains "air in-leakage" or maps to FM-CND-AIR-INLEAK. Failure means the ranking logic is not properly weighting DO evidence.
  * Assertion 2 — Fouling is alternative, not primary: FM-CND-TUBE-FOUL appears in alternatives, not as primary. Failure means evidence refinement is not working.
  * Assertion 3 — WO-2024-11847 classified as contradicting for fouling: In evidence_bundle.candidate_evidence_summary, the fouling candidate should have contradicting_count >= 1. Failure means the keyword-based role classifier is not detecting "within acceptance criteria" as contradicting language for a fouling hypothesis.
  * Assertion 4 — DO anomaly drives ranking change: causality_candidates before refinement should have score gap between air in-leakage and fouling < 0.08. After refinement the gap should be > 0.12. Failure means evidence refinement is having no real discriminating effect.
  * Assertion 5 — Recurrence doesn't mislead: Air in-leakage recurrence score should exceed fouling recurrence score despite the most recent analog being fouling (2 prior air in-leakage events vs 1 fouling). Failure exposes recency bias in the recurrence scorer.
  * Assertion 6 — Governance links to correct candidates: The failed expansion joint inspection PM should be associated with air in-leakage candidates, not with tube fouling. This will fail with the current uniform governance scorer and document that gap explicitly.
  * Assertion 7 — CW temperature is contributor, not root cause: FM-CW-TEMP-RISE should appear in filtered candidates or as a low-ranked alternative, not as primary. The operator note that increasing CW flow had "minimal effect" should be surfaced as contradicting evidence for this hypothesis.
  * Assertion 8 — HVAC degradation is contributing factor: FM-HVAC-DEGRAD should appear in the Ishikawa matrix under environment_operating_context or maintenance_human_factors with a temporal relation noting onset on Day 4, preceding the acceleration in backpressure on Days 5-14.
  * Assertion 9 — Temporal contradiction for tube leak: FM-CND-TUBE-LEAK should be either filtered out or marked temporal_contradiction: true because its expected latency (2-48 hours) is inconsistent with a 14-day gradual drift pattern.
  * Assertion 10 — Analyst review questions are substantive: analyst_review.questions_to_resolve should contain at least one question about the expansion joint inspection deferral as a programmatic contributor and one about whether the HVAC fan bearing failure is independently correctable.

## Why This Test Case Is Hard Enough
Most pipeline tests use scenarios where the true cause has the highest signal strength across all dimensions. This scenario is designed so the true cause (air in-leakage) is not the obvious winner before evidence refinement. It specifically tests whether your pipeline's evidence refinement stage actually changes the ranking rather than just confirming what the structural stage already selected. If your evidence refinement produces no rank change in this scenario, that's a definitive finding that the retriever and role classifier aren't doing useful work.
The recurrence trap tests an analytically important failure mode: using the most recent historical analog as a shortcut to the current diagnosis. Nuclear engineers know this failure mode by name — it's one of the primary cognitive biases in RCA. Your system should resist it through proper recurrence weighting, and this scenario will show whether it does.
The governance gap — where the failed PM items are cause-specific — will definitively demonstrate the uniform governance scoring issue flagged in the earlier review. The test case makes that gap visible in the output rather than just in a code review.