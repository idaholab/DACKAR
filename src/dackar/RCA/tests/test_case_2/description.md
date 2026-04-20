# PWR secondary side test case

## Context (Set the scene)
“We’re looking at a pressurized water reactor operating at high load, around 96% power. The event occurs on the secondary side, specifically involving the condenser, feedwater system, and some supporting systems like HVAC.”

## Timeline
13:10 — Early precursor
    Cabinet temperature in instrumentation cabinet (AC12) begins to rise
    No immediate impact on plant performance

13:20 — Event starts
    Unit is stable at high load
    No alarms yet

13:22 — First anomaly
    Air removal train differential pressure shows a step change
    👉 Potential degradation in vacuum support system

13:24 – 13:26 — Core symptom develops
    Condenser vacuum begins to degrade
    Turbine backpressure starts rising
    👉 This is the primary plant-level symptom

13:31 – 13:34 — Secondary effects appear
    Hotwell level begins oscillating
    Feedwater flow becomes unstable
    👉 These are likely effects, not causes

    ~14:00 — Operators respond
    Performance degradation is clearly visible
    Investigation begins

## Observations 
* main symptom: condenser vacuum degradation with unit power reduction
* primary systems involved: condenser, circulating water / air removal, condensate and feedwater
* supporting system context: loss or degradation of an AC/instrumentation support function
* competing hypotheses:
    1) condenser tube fouling / reduced heat transfer
    2) air in-leakage degrading vacuum
    3) feedwater control valve / condensate control instability
    4) instrumentation bias or intermittent sensor issue
    5) support-system contribution via degraded AC to instrumentation cabinet or vacuum equipment support
   
## Event narrative

During steady operation at high load, condenser backpressure begins to rise over about 40 minutes. Operators observe gradual loss of condenser vacuum, increasing turbine exhaust pressure, feedwater flow oscillations, and elevated hotwell level variability. A recent CR mentions intermittent performance degradation in one air removal train, while a WO notes reduced cooling effectiveness in an instrumentation/relay cabinet HVAC unit serving condenser vacuum instrumentation and part of the feedwater control cabinet area. A maintenance history item also mentions trending fouling on condenser waterboxes.

The intended outcome is not completely trivial:
* the top hypothesis should likely be air in-leakage / degraded air removal performance
* condenser fouling should remain a strong alternative
* feedwater control instability should appear as a secondary or contributing factor
* the AC/HVAC support issue should appear mainly in Ishikawa and possibly as a contextual contributor, not necessarily the top direct root cause

## What the System Sees (Translate to pipeline stages)

“This is what our RCA pipeline is doing behind the scenes.”

Step A — Structure (KG Context)
* Identify relevant components:
* condenser
* air removal system
* hotwell
* feedwater control
* HVAC support system

👉 This defines where to look

Step B — Temporal Reasoning (TSKR)
* The system detects patterns like:
* Air removal anomaly precedes vacuum degradation
* Vacuum degradation and backpressure rise track together
* Feedwater oscillations follow the main event

👉 This is critical: it distinguishes cause vs effect

Step C — Candidate Generation
The system evaluates multiple hypotheses:
* Air in-leakage / air removal degradation
* Condenser fouling
* Feedwater control instability
* HVAC/support degradation



## Expected behavior 

What I would expect the orchestrator to produce:
* top candidate: FM_AIR_INLEAK
* strong alternative: FM_COND_FOULING
* secondary/contributing: FM_HVAC_SUPPORT_DEGRAD
* consequential effect candidate: FM_FWCV_INSTAB

And in the Ishikawa matrix:
* strong rows under equipment_hardware
* useful rows under measurement_instrumentation
* at least one row under maintenance_human_factors
* one support/context row under environment_operating_context
* a process/procedure row from SOP evidence