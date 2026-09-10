# TC-6 — Human Performance / Procedure Gap During Startup

## Scenario Description

**Reactor type:** PWR Unit 2 (Northbrook Nuclear Station)  
**Event ID:** `E2026-01-22-001`  
**Asset ID:** `U2-MFP-B`  
**Operational mode:** Power ascension (startup, ~30% rated power)  
**CR Numbers:** `CR-2026-00741`, `CR-2026-00742`

During power ascension from a refueling outage, main feedwater pump B trips on high bearing temperature 18 seconds after restart. Investigation reveals the lube oil venting step (§4.3.2 of MNT-FW-022 Rev. 14) was skipped — the outage coordinator compressed the maintenance window 2 hours to meet startup schedule, and the maintenance supervisor signed the WO completion checklist without witnessing the step. The lube oil flow transmitter was left in maintenance bypass, so no alarm annunciated for inadequate flow.

**Why this scenario is compelling for show-and-tell:**  
The pipeline must distinguish two independent programmatic findings from a single event: an *execution error* (the skipped step, Category H) and a *procedure inadequacy* (no quantitative acceptance criterion for the venting step, Category I). A standard checklist-based RCA tool requires the analyst to prompt for each explicitly; the pipeline should identify both from the evidence patterns independently. Training records show the technician is fully qualified — training currency is emphatically NOT the root issue, which the pipeline should confirm by deprioritising the training-gap hypothesis.

---

## Data Elements Used

| Fixture | Required | Contents |
|---|---|---|
| `event.json` | Yes | MFP-B trip descriptor, 18s after pump start |
| `telemetry_summary.json` | Yes | Bearing temp — immediate rise from first rotation; LO flow — in bypass |
| `kg_context.json` | Yes | Four failure modes: execution error (H), LO cooler fouling (A), bearing wear (A), procedure criterion gap (I); past MFP-A event |
| `operational_context.json` | No | Startup mode, 30% power, two maintenance WOs nearby |
| `pm_compliance.json` | No | Bearing PM overdue 92 days (contributing, not causal) |
| `soe_log.json` | No | Pump start T+0; bearing temp alarm T+2s; trip T+18s; LO bypass active |
| `alarm_log.json` | No | Bearing temp alarms + MFP-B trip + LO bypass status |
| `training_records.json` | No | TECH-4421 qualified; SUPV-1147 qualified; schedule pressure from COORD-0891 |
| `configuration_change_records.json` | No | WO narrative documenting deferred venting step and coordinator direction |
| `evidence_bundle.json` | No | WO narrative, procedure text, initial CR causal assessment |

---

## Analysis Performed in Notebook

1. **Fixture loading** — all 10 data elements.
2. **Orchestrator** — `build_fixture_orchestrator()` from shared helpers.
3. **Full pipeline run** — `run_rca()` with all fixtures.
4. **Human performance assessment block** — inspect `rca_card.human_performance_assessment`.
5. **Ishikawa matrix** — show `process_procedure` row for acceptance criterion gap.
6. **Training records** — verify `training_records` consumed and training-gap candidate NOT primary.
7. **Telemetry onset pattern** — show that immediate-on-start rise rules out fouling/wear.
8. **Operating point scoring** — startup mode should boost execution-error candidate (active maintenance transition).
9. **Assertions** — A6-1 through A6-10.

### Show-stopper demonstration

Two distinct programmatic findings from one event emerge independently:
- `human_performance_assessment` → `performance_mode = "execution_error"` (skipped step)
- `ishikawa_matrix["process_procedure"]` → procedure acceptance criterion gap

The training records data confirms TECH-4421 is qualified — the pipeline deprioritises training as a root cause. A manager can see immediately: the problem is not training, it is supervision under schedule pressure combined with a procedure that lacks a testable acceptance criterion.

---

## Expected Outputs (Assertions)

| ID | Claim | Path / Condition |
|---|---|---|
| A6-1 | Execution error candidate ranked primary | `rca_card["primary_cause_label"]` references lube oil / venting or maps to `FM-MFPB-LUBE-OIL-OMISSION` |
| A6-2 | Human performance assessment applicable | `rca_card["human_performance_assessment"]["applicable"] == True` |
| A6-3 | Execution error finding present | At least one finding with `performance_mode == "execution_error"` |
| A6-4 | Procedure gap in Ishikawa | `ishikawa_matrix["process_procedure"]` has ≥1 entry |
| A6-5 | Training records consumed | `data_coverage_summary["training_records"]["status"] == "complete"` |
| A6-6 | Training not causal | `FM-MFPB-LUBE-OIL-OMISSION` is primary; no training-gap candidate is top-ranked |
| A6-7 | Startup operating point score > 0 | `cand["scores"]["operating_point_score"] > 0` for primary candidate |
| A6-8 | Fouling and wear deprioritised | Fouling and wear candidate composites < execution-error candidate |
| A6-9 | AP-913 completeness present | `run_manifest["ap913_completeness"]` is not None |
| A6-10 | Bearing PM gap is contributing, not primary | PM-overdue candidate does not appear as primary hypothesis |
