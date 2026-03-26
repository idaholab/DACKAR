# PWR secondary side test case

* main symptom: condenser vacuum degradation with unit power reduction
* primary systems involved: condenser, circulating water / air removal, condensate and feedwater
* supporting system c*ontext: loss or degradation of an AC/instrumentation support function
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