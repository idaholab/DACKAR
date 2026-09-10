# Show-and-Tell Test Suite — Planning Document
**Created:** 2026-04-25  
**Revised:** 2026-04-25 (post code-review pass)  
**Audience:** Managers, system engineers, technical reviewers  
**Purpose:** Demonstrate the RCA v32 workflow mechanics across realistic nuclear plant operational
situations. Tests are scenario-driven, structured as pre-executed Jupyter notebooks, and
designed so an SE can follow the reasoning chain without reading source code.

---

## Guiding Principles

1. **No latency-based discrimination.** Failure mode latency windows are a knowledge-engineering
   artifact that plant SEs will immediately challenge. All candidate discrimination must come from
   observable evidence: telemetry anomaly *patterns* (type, onset, correlation), document corpus
   (CR/WO/SOP language), SOE/alarm *sequence*, operational context, PLC/barrier state,
   PM compliance records, and prior event history.

2. **Breadth of input coverage.** The v32 pipeline accepts a richer input set than the old tests
   exercised. The suite exercises every non-trivial input type at least once. Some cases are
   data-rich (all sources populated); others are deliberately data-degraded so the pipeline's
   uncertainty surfacing is visibly demonstrated.

3. **One clear "show-stopper" per case.** Each scenario has one feature that would be impossible
   to replicate with a keyword-search tool — the thing a manager can point to and say
   "the system did something a checklist wouldn't."

4. **Pseudo-plant naming convention — consistent across all cases.**
   - Unit 1 scenarios use `U1-` prefix; Unit 2 use `U2-`
   - Signal tags follow ISA-5.1 pattern: `U1-<instrument_type>-<loop>-<suffix>` (e.g., `U1-PT-4201A`)
   - Component IDs: `U1-<system_abbrev>-<descriptor>` (e.g., `U1-CND-EXPANSION-JOINT-EXH`)
   - Failure mode IDs: `FM-<component_abbrev>-<mechanism>` (e.g., `FM-CND-AIR-INLEAK`)
   - Document refs: `CR-YYYY-NNNNN`, `WO-YYYY-NNNNN`, `SOP-U1-SYS-NNN`
   - Past event IDs: `EVT-U1-YYYY-NNNN`
   - Units 1 and 2 are on the same fictional site "Northbrook Nuclear"; Unit 3 = BWR (TC-5)

5. **Explicit assertions grounded in actual output field paths.** Assertion tables reference
   the true JSON path in the result dict returned by `orchestrator.run()`. The top-level keys of
   that dict are: `rca_card`, `run_manifest`, `run_context`, `causality_candidates`,
   `causality_candidates_pre_refine`, `evidence_bundle`, `tskr_patterns`, `ishikawa_matrix`,
   `signal_evidence`, `kg_context`, `barrier_analysis`, `reentry_execution`, `cmms_context`,
   `input_validation`, `output_validation`.

6. **Pre-executed notebooks, multi-cell structure.** Each notebook is pre-executed and saved.
   Structure: Setup → Scenario context (markdown) → Run cell(s) → Key output cells →
   Assertion cell → Interpretation cell (markdown "what did the system do?").

---

## Input Data Element Coverage Matrix

| Input type (run() parameter name) | TC-1 | TC-2 | TC-3 | TC-4 | TC-5 | TC-6 | TC-7 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `event` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `telemetry_summary` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `kg_context` (components, FMs, past events) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `tskr_patterns` (pre-built or generated) | — | ✓ | ✓ | ✓ | — | — | ✓ |
| `evidence_bundle` (`{"results":[...]}`) | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `operational_context` | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `pm_compliance` | — | ✓ | ✓ | — | ✓ | ✓ | — |
| `soe_log` | — | ✓ | — | ✓ | — | — | — |
| `alarm_log` | — | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| `protection_logic_context` | — | ✓* | — | ✓ | ✓ | — | — |
| `configuration_change_records` | — | — | ✓ | — | ✓ | ✓ | — |
| `environmental_monitoring` | — | — | ✓ | ✓ | — | — | — |
| `vendor_supply_chain_records` | — | — | — | — | ✓ | — | — |
| `training_records` | — | — | — | — | — | ✓ | — |

*TC-2: PLC provided with benign content (barrier `not_tripped`) specifically to satisfy the
SOE/PLC pairing requirement — if SOE is present but PLC is absent, the pipeline emits
`soe_plc_pairing = "violated"` and creates a blocking `analyst_decisions_required` entry.

---

## Evidence Bundle Format

The `evidence_bundle` parameter must be a Python dict with a `"results"` key:
```json
{
  "event_id": "...",
  "results": [
    {
      "doc_id": "CR-2026-04799",
      "doc_type": "CR",
      "snippet": "...",
      "relevance_score": 0.88,
      "evidence_role": "supporting|contradicting|contextual",
      "candidate_id_hint": "...",
      "timestamp": "2026-04-15T09:00:00Z"
    }
  ],
  "metadata": { "retrieval_method": "fixture", "total_docs_searched": 6 }
}
```
When `results` count < 3, `chroma_corpus` status is `"partial"`; 0 results → `"missing"`.
This affects the coverage multiplier and sensitivity table.

---

## Data Coverage Status Reference

The `source_families` dict inside `data_coverage_summary` uses these status values:

| Situation | Status | Notes |
|---|---|---|
| Source not passed to `run()` | `"not_assessed"` | Used for: `soe_log`, `alarm_log`, `protection_logic_context`, `configuration_change_records`, `environmental_monitoring`, `vendor_supply_chain_records`, `training_records` |
| Source passed and good quality | `"complete"` | |
| Source passed but degraded | `"partial"` | High missing fraction, clock issues, etc. |
| SOE present + PLC absent | PLC → `"missing"` (violated) | Triggers `analyst_decisions_required` entry |
| Required source absent (kg_context, tskr) | `"missing"` | Different from optional sources |

`"not_assessed"` sources appear in `sensitivity_table.summary.missing_sources_checked` —
the sensitivity table will estimate how much score could improve if they were provided.

---

## Assertion Path Reference

Common paths used in assertion tables:

```python
r = orchestrator.run(...)  # r is the returned result dict

# RCA card primary hypothesis
r["rca_card"]["primary_hypothesis"]["cause_label"]

# RCA card executive summary (causal depth, unresolved gaps)
r["rca_card"]["executive_summary"]["causal_depth_summary"]["depth_complete"]
r["rca_card"]["executive_summary"]["unresolved_gaps"]      # list of gap entries

# Human performance assessment
r["rca_card"]["human_performance_assessment"]["applicable"]       # bool
r["rca_card"]["human_performance_assessment"]["findings"][0]["performance_mode"]

# Scope filter (from run_manifest artifacts)
r["run_manifest"]["artifacts"]["scope_filter"]["applied"]         # bool
r["run_manifest"]["artifacts"]["scope_filter"]["filtered_count"]  # int

# Scope expansion suggestions (pending)
r["run_context"]["scope_management"]["expansion_suggestions"]     # list

# Data coverage summary
r["run_manifest"]["artifacts"]["data_coverage_summary"]["source_families"]["soe_log"]["status"]

# Sensitivity table
r["run_manifest"]["artifacts"]["sensitivity_table"]["any_ranking_change_possible"]  # bool
r["run_manifest"]["artifacts"]["sensitivity_table"]["missing_sources_checked"]       # list

# Candidate scores (post-refine)
candidates = r["causality_candidates"]["candidates"]
cand = next(c for c in candidates if c["candidate_id"] == "...")
cand["scores"]["ccf_score"]
cand["scores"]["operating_point_score"]
cand["hard_gates"]["timeline_consistency"]["passed"]
cand["hard_gates"]["barrier_logic"]["plc_consulted"]

# Similar event list
r["run_manifest"]["artifacts"]["similar_event_list"]["any_plant_match"]  # bool
r["run_manifest"]["artifacts"]["similar_event_list"]["total_count"]       # int
```

---

## Part 1 — Updating Existing Tests

### test_case_1: Rotating Equipment Bearing Wear (minor update)

**Status:** Notebook-only, no operational context, outdated `run()` signature.

**Remediation:**
- Add `operational_context.json`: `mode=steady_state`, `percent_rated_power=100`,
  `train_configuration.in_service=True` for both trains.
- Update `event.json`: `symptom_signature` must be a dict `{"anomaly_pattern": ...,
  "symptom_types": [...]}`, not a bare string.
- Update notebook cell to use current `run()` keyword-arg signature.
- Regenerate output (no new fixtures otherwise — this stays the "minimum viable input" baseline).

---

### test_case_2: PWR Secondary Side Condenser Vacuum Loss (update + enrich)

**Reactor type:** PWR Unit 1 (Northbrook Nuclear)  
**Operational mode:** Full power (~96% rated)

**Status:** Missing `soe_log`, `alarm_log`, `protection_logic_context`; notebook uses old
signatures; no phase-5 output saved.

**New fixtures to add:**

`soe_log.json`
```json
{ "event_id": "E2026-04-15-001", "quality": {"clock_sync_ok": true, "dropped_record_count": 0},
  "records": [
    {"timestamp": "2026-04-15T13:22:00Z", "signal_id": "U1-DPT-AR-2201B", "state": "HIGH", "description": "Air removal train B differential pressure high"},
    {"timestamp": "2026-04-15T13:24:00Z", "signal_id": "U1-PT-CND-4201", "state": "ALARM", "description": "Condenser backpressure first threshold alarm"},
    {"timestamp": "2026-04-15T13:26:00Z", "signal_id": "U1-PT-CND-4201", "state": "HIGH", "description": "Condenser backpressure high alarm"},
    {"timestamp": "2026-04-15T13:31:00Z", "signal_id": "U1-LT-HW-3301", "state": "HIGH", "description": "Hotwell level high"},
    {"timestamp": "2026-04-15T13:34:00Z", "signal_id": "U1-FT-FW-2101A", "state": "DEVIATION", "description": "Feedwater flow deviation"},
    {"timestamp": "2026-04-15T14:00:00Z", "signal_id": "OPERATOR-ACTION", "state": "LOGGED", "description": "Operator initiated 5% load reduction"}
  ]
}
```

`alarm_log.json` — mirrors SOE with alarm IDs `ALM-U1-AR-001`, `ALM-U1-CND-004`, etc.;
includes setpoint values and priority tiers (2 or 3).

`protection_logic_context.json` — barrier `not_tripped`; turbine trip logic set references
`U1-PT-CND-4201` as a trip initiator signal at a lower-low setpoint (not reached in this
event); barrier state for condenser vacuum-low trip function = `armed` (not actuated). This
is benign but must be provided to satisfy the SOE/PLC pairing check.

**Show-stopper:** The SOE places the air removal differential pressure anomaly 2 minutes before
condenser vacuum begins to degrade — the Allen `precedes` relation is built automatically
across two separate systems and surfaced in the score rationale. A human analyst would need to
manually correlate these across multiple alarm response procedures; the pipeline does it in
the temporal stage.

**Bonus demo cell (TC-2 notebook only):** Show what the analyst_decisions_required block looks
like when `protection_logic_context` is omitted despite SOE being present. One notebook cell
runs a minimal "no-PLC" call; the next cell shows the `soe_plc_pairing = "violated"` flag
and the blocking `analyst_decisions_required` entry; the final cell runs the full call with
PLC provided and shows the flag is gone. This makes the pairing requirement tangible without
adding a separate test case.

---

### test_case_3: PWR Condenser Backpressure / Turbine Runback (update)

**Reactor type:** PWR Unit 2 (Northbrook Nuclear)  
**Status:** `data_generator.py` generates fixtures against old schema; missing
`configuration_change_records`, `alarm_log`, and `environmental_monitoring`.

**Updates to data_generator.py:**
- Remove `expected_latency_min/max` fields from all failure mode defs in `kg_context`.
- Add builder for `alarm_log.json`: Day 1 backpressure trending advisory; Day 9 operator
  trend CR alarm; Day 12 backpressure above 2.5 inHg engineering evaluation threshold;
  Day 14 turbine runback actuation.
- Add builder for `configuration_change_records.json`: Waterbox cleaning WO-2024-11847 (21
  days prior; as-found tube cleanliness score 0.94; as-left acceptable; zero tubes plugged).
  This lets the pipeline pick up tube-clean confirmation as a structured record *without*
  requiring document retrieval, providing a second independent contradicting path for fouling.
- Add builder for `environmental_monitoring.json`: CW inlet temperature seasonal rise
  (4.2°F over 14-day window; within seasonal normal, no anomaly flag); condenser pit ambient
  temperature `gradual_drift` onset Day 4 (linked to HVAC fan degradation).

**Discrimination mechanism (post-update):** Fouling hypothesis is contradicted by:
1. `configuration_change_records` (structured): tube cleanliness score 0.94 → contradicting
2. Evidence bundle WO-2024-11847 snippet (document): "as-found acceptable" → contradicting  
3. Normal condenser tube outlet temperatures (telemetry) → no supporting anomaly

Air in-leakage is supported by: hotwell DO elevation in telemetry + CR-2024-04821 language +
SOP DO threshold rule. Three independent evidence paths converging on the same hypothesis is
the kind of multi-source convergence an SE would find credible.

**Assertions:**
| ID | Claim | Path and condition |
|---|---|---|
| A3-1 | Air in-leakage is primary post-refine | `r["run_manifest"]["artifacts"]["rca_card"]["primary_cause_label"]` maps to `FM-CND-AIR-INLEAK` |
| A3-2 | Fouling is alternative, not primary | Fouling candidate appears in `rca_card.alternatives`, not as primary |
| A3-3 | Evidence refinement changed ranking | Fouling rank in `causality_candidates_pre_refine` < air in-leakage rank; order flips after refine |
| A3-4 | WO contradicts fouling | Fouling candidate `primary_contradicting_evidence_count >= 1` in `run_manifest.artifacts.rca_card` |
| A3-5 | Recurrence favours air in-leakage | Air in-leakage candidate `scores.recurrence` ≥ fouling candidate `scores.recurrence` (2 prior events vs 1) |
| A3-6 | Similar event matched | `r["run_manifest"]["artifacts"]["similar_event_list"]["any_plant_match"] == True` |
| A3-7 | Env monitoring consumed | `r["run_manifest"]["artifacts"]["data_coverage_summary"]["source_families"]["environmental_monitoring"]["status"] == "complete"` |
| A3-8 | CCR consumed | `r["run_manifest"]["artifacts"]["data_coverage_summary"]["source_families"]["configuration_change_records"]["status"] == "complete"` |

---

## Part 2 — New Test Cases

---

### test_case_4: Reactor Trip — PLC Gates and Timeline Gate

**Reactor type:** PWR Unit 1 (Northbrook Nuclear)  
**Operational mode:** Full power (99% rated)  
**Event ID:** `E2026-03-10-001`  
**Event:** Automatic reactor trip (SCRAM) on high neutron flux from NI channel 4 at 14:22.

**Scenario narrative:**  
Unit 1 is at full power, mid-cycle. At 14:22:03, RPS channel 4 initiates an automatic trip on
high neutron flux (`U1-NI-RPS-4A`). All four control rod banks insert within 2.4 seconds.
Post-trip review must determine the initiating cause. Three hypotheses are evaluated:

**Hypothesis A — Spurious NI instrument signal (Category D: instrumentation drift/noise).**
NI channel 4 (`U1-NI-RPS-4A`) had a calibration zero-offset correction 72 hours prior; the
as-left reading was within tolerance but at the high end of its acceptance band. The channel
has a documented noise history from an EMI source in the local cabinet area.

**Hypothesis B — Real power excursion from SG-3 feedwater transient (Category A: process).**
FW control valve `U1-FCV-FW-3301` showed a 4% position anomaly at 14:21:18 — 45 seconds
before the NI trip signal. A genuine power excursion would produce correlated flux signals
across all four NI channels simultaneously.

**Hypothesis C — CRD mechanism fault (Category A: mechanical).**
CRD position indicator `U1-ZI-CRD-2214` showed an unexplained 2-step position change at
14:22:13 — 10 seconds *after* the NI trip initiator. This places it as a consequence of the
trip insertion, not a cause. The timeline gate should eliminate this hypothesis automatically.

**Discriminating evidence — no latency required:**
- NI channels 1, 2, 3 (`U1-NI-RPS-1A/2A/3A`) show no anomaly in the 60-second window before
  the trip → if Hypothesis B were correct (real flux excursion), all four channels would respond.
  Single-channel response strongly supports a spurious signal.
- SOE record for `U1-ZI-CRD-2214` is timestamped T+10s after the NI trip signal → Allen
  `follows` relation; timeline gate fails Hypothesis C.
- Calibration record in evidence bundle (WO-2026-03-07-044): NI-4 zero offset noted as
  "at upper bound of acceptance; EMI shielding check recommended." Supports Hypothesis A.

**Fixtures:**

`kg_context.json` — components: `U1-NI-RPS-4A`, `U1-NI-RPS-1A/2A/3A` (reference channels),
`U1-FCV-FW-3301`, `U1-CRD-2214`, `U1-RPS-CORE`; failure modes: `FM-NI-SPURIOUS`,
`FM-FW-TRANSIENT`, `FM-CRD-MECHANICAL`.

`telemetry_summary.json` — `U1-NI-RPS-4A`: single-channel exceedance, no anomaly on
NI-1/2/3; `U1-FCV-FW-3301`: position deviation anomaly T-45s; `U1-ZI-CRD-2214`: step
change T+10s.

`soe_log.json` — 6 records: FW valve deviation at T-45s; NI-4 trip at T+0; rod bank A
insertion at T+0.8s; rod bank B-D insertion T+1.2s–T+2.4s; CRD-2214 position step at T+10s;
operator procedure entry at T+90s.

`alarm_log.json` — NI-4 high flux alarm; FW control valve position deviation alarm (priority 2).

`protection_logic_context.json` — RPS logic set: trip signals are `U1-NI-RPS-1A`,
`U1-NI-RPS-2A`, `U1-NI-RPS-3A`, `U1-NI-RPS-4A`; barrier state for reactor trip function
= `held` (rods inserted, core subcritical); turbine trip interlock = `held`; main steam
isolation = `armed`.

`environmental_monitoring.json` — containment radiation at background; no anomaly.

`evidence_bundle.json` — WO-2026-03-07-044 (NI-4 calibration, EMI note), calibration
surveillance procedure SOP-U1-NI-001, prior CR referencing EMI sensitivity.

**Show-stopper:** Hypothesis C is eliminated automatically by the timeline consistency gate
because the SOE places the CRD anomaly 10 seconds *after* the trip initiator — without any
analyst input. During post-trip review under time pressure, this kind of temporal error (assuming
the CRD anomaly was a cause rather than an effect) is a well-known cognitive trap in nuclear
event investigation. The pipeline sidesteps it.

**Assertions:**
| ID | Claim | Path and condition |
|---|---|---|
| A4-1 | PLC consulted for NI candidates | `cand["hard_gates"]["barrier_logic"]["plc_consulted"] == True` for NI-4 candidate |
| A4-2 | CRD ruled out by timeline | `cand["hard_gates"]["timeline_consistency"]["passed"] == False` for CRD candidate |
| A4-3 | CRD reason code | `cand["decision_trail"][-1]["reason_code"] == "timeline_inconsistent"` for CRD candidate |
| A4-4 | Multi-channel contradiction | Hypothesis B composite score < Hypothesis A composite score |
| A4-5 | Spurious NI signal is primary | `r["run_manifest"]["artifacts"]["rca_card"]["primary_cause_label"]` contains NI/spurious |
| A4-6 | Allen relation in rationale | NI candidate `score_rationale["temporal"]` contains an Allen relation keyword |
| A4-7 | RPS barrier `held` noted | NI candidate hard gate rationale note references `held` state |
| A4-8 | Operating point score > 0 | `cand["scores"]["operating_point_score"] > 0` for full-power candidates |
| A4-9 | Environmental monitoring consumed | `r["run_manifest"]["artifacts"]["data_coverage_summary"]["source_families"]["environmental_monitoring"]["status"] == "complete"` |
| A4-10 | Signal lessons learned present | `r["run_manifest"]["artifacts"]["signal_lessons_learned"]["total_matched"] >= 1` (NI-4 single-channel pattern matched) |

---

### test_case_5: ECCS Train Degradation — Common-Cause Failure

**Reactor type:** BWR Unit 3 (Northbrook Nuclear)  
**Operational mode:** Cold shutdown (maintenance outage, mode 4)  
**Event ID:** `E2026-02-14-001`  
**Event:** Both HPCI trains fail post-maintenance surveillance flow acceptance criterion.

**Scenario narrative:**  
During refueling outage, both HPCI trains (`U3-HPCI-TRAIN-A` and `U3-HPCI-TRAIN-B`) were
overhauled by the same maintenance crew over a 5-day window. Pump coupling assemblies were
replaced using parts from vendor lot `VEN-2026-Q1-HC7`. Post-maintenance surveillance testing
(SURV-U3-HPCI-001) shows both trains fail to reach required injection flow within the required
time. Failure mode identical on both: coupling slippage under load from incorrect torque.

A vendor service advisory for lot `VEN-2026-Q1-HC7` regarding torque sensitivity was received
3 months prior but was not dispositioned into a work order. The station surveillance procedure
does not include vendor-specified coupling torque acceptance criteria — the criteria exist only
in the vendor technical manual (not translated into station procedures).

**Competing hypotheses:**
- **Hypothesis A — CCF: common vendor lot (Category C).** Both failures traced to same lot,
  same assembly type, same crew, same outage window.
- **Hypothesis B — Independent wear, Train A (Category A).** Pump coupling wear from accumulated
  service hours; coincidental failure.
- **Hypothesis C — Independent wear, Train B (Category A).** Same argument, Train B.
- **Hypothesis D — Procedure gap (Category I).** Coupling torque acceptance criterion absent
  from surveillance procedure. Programmatic contributor independent of A.

**Discriminating evidence:**
- Vendor supply chain records: Both assemblies from lot VEN-2026-Q1-HC7; vendor advisory
  received 3 months prior; advisory non-disposition documented (no WO issued). This structural
  link — same lot, same failure, undispositioned advisory — is the CCF signature.
- PM compliance: HPCI Train A coupling inspection overdue by 6 months; Train B was current.
  This *undermines* the independent-wear hypothesis for Train A specifically: if it were pure
  wear, Train B (with current PM) should not fail at the same test.
- Alarm log: Both trains produce identical alarm signatures during the surveillance test —
  same alarm sequence, same timing relative to pump start. Pattern-match between two
  independent trains is characteristic of CCF, not independent wear.
- Past event in kg_context: `EVT-U3-2023-0021` — prior HPCI flow degradation from incorrect
  coupling orientation, same component family. Category C framing is not novel to this unit.

**Fixtures:**

`vendor_supply_chain_records.json`
```json
{
  "event_id": "E2026-02-14-001",
  "records": [
    { "lot_id": "VEN-2026-Q1-HC7", "part_type": "pump_coupling",
      "component_ids": ["U3-HPCI-TRAIN-A-PUMP", "U3-HPCI-TRAIN-B-PUMP"],
      "vendor_advisory_id": "VA-2025-HC7-001",
      "advisory_subject": "Torque sensitivity: insufficient preload may cause slip under load",
      "advisory_received": "2025-11-14T00:00:00Z",
      "disposition_status": "not_dispositioned",
      "disposition_wo": null }
  ],
  "quality": {"coverage_status": "complete"}
}
```

`protection_logic_context.json` — HPCI system logic set; both trains show
`barrier_state = "failed"` for HPCI injection function (condition that triggered event
classification); ECCS actuation signal logic set included.

`alarm_log.json` — Train A and Train B identical alarm sequences during surveillance test
(same alarm IDs, same time-from-start-of-test ordering).

`configuration_change_records.json` — Work order records for coupling replacement on both
trains; crew ID, outage dates, lot number reference.

`pm_compliance.json` — Train A coupling inspection overdue; Train B current.

**Show-stopper:** The CCF structural delta (Item 3) gives the CCF candidate a scoring boost from
`common_cause_score`. The sensitivity table then shows that if the vendor advisory had been
dispositioned into a WO, CCF confidence would increase further — a direct pointer to a
programmatic gap with board-level significance.

**Assertions:**
| ID | Claim | Path and condition |
|---|---|---|
| A5-1 | CCF candidate in top 2 pre-refine | CCF candidate rank ≤ 2 in `causality_candidates_pre_refine` by composite score |
| A5-2 | CCF delta applied | `cand["scores"]["ccf_score"] > 0` and `cand["scores"]["ccf_note"] != "not_applied"` |
| A5-3 | Vendor records in coverage | `r["run_manifest"]["artifacts"]["data_coverage_summary"]["source_families"]["vendor_supply_chain"]["status"] == "complete"` |
| A5-4 | Procedure gap in Ishikawa | `r["ishikawa_matrix"]["process_procedure"]` has ≥1 entry referencing procedure gap |
| A5-5 | Sensitivity table flags vendor | `r["run_manifest"]["artifacts"]["sensitivity_table"]["missing_sources_checked"]` includes `vendor_supply_chain` if advisory not dispositioned |
| A5-6 | PLC barrier failed noted | `cand["hard_gates"]["barrier_logic"]["plc_consulted"] == True`; rationale includes `failed` state |
| A5-7 | Independent wear ranks below CCF | Both single-train wear candidates composite score < CCF candidate |
| A5-8 | Human perf assessment: procedure gap | `r["rca_card"]["human_performance_assessment"]["applicable"] == True`; finding with `performance_mode == "procedure_gap"` |
| A5-9 | Barrier analysis present | `r["barrier_analysis"] is not None` and `r["barrier_analysis"]["summary"]["degraded_barrier_count"] >= 2` (both HPCI trains failed) |

---

### test_case_6: Human Performance / Procedure Gap During Startup

**Reactor type:** PWR Unit 2 (Northbrook Nuclear)  
**Operational mode:** Power ascension (startup, ~30% rated power)  
**Event ID:** `E2026-01-22-001`  
**Event:** Main feedwater pump B trips on high bearing temperature 18 minutes after restart.

**Scenario narrative:**  
Unit 2 is ascending to power following a scheduled outage. At approximately 30% power,
feedwater pump B (`U2-MFP-B`) trips on high bearing temperature (`U2-TE-BRG-2201B` reading
148°F against trip setpoint 145°F). Investigation reveals the lube oil system for the pump was
not properly vented after maintenance — a required step in procedure MNT-FW-022 Rev. 14 that
was skipped. The maintenance supervisor signed the completion checklist without witnessing the
step. A note in the work order from the outage coordinator indicates the maintenance window was
cut short by 2 hours to meet startup schedule.

**Competing hypotheses:**
- **Hypothesis A — Lube oil venting omission (Category H: execution error).** Required step
  skipped. Pump ran without adequate lube oil flow from first rotation; bearing temperature rose
  immediately (not after thermal equilibration).
- **Hypothesis B — Lube oil cooler fouling (Category A).** Would show gradual temperature rise
  over multiple start cycles, not a first-start-after-maintenance step.
- **Hypothesis C — Bearing wear/end-of-life (Category A).** Bearing inspection 6 months prior:
  acceptable. Not consistent with a first-start-after-maintenance failure pattern.
- **Hypothesis D — Procedure acceptance criterion gap (Category I: procedure inadequacy).**
  The venting step exists in the procedure but has no measurable acceptance criterion for
  "vent complete" — no flow rate, no time, no indicator. Programmatic contributor independent
  of H.

**Discriminating evidence:**
- Telemetry: `U2-TE-BRG-2201B` rises *immediately* from first pump rotation (onset within
  2 seconds of startup in alarm log). This pattern — no thermal warmup period — is inconsistent
  with fouling or wear (both require extended operation to manifest). It is consistent with
  no lube oil flow from the first revolution.
- Alarm log: Lube oil flow transmitter `U2-FT-LO-2201B` was in bypass for maintenance and
  not restored prior to startup. No lube oil flow alarm fired despite inadequate flow — a
  separate finding (Category H: incomplete post-maintenance restoration).
- Training records: Technician qualifications current; last task performance on identical
  procedure 7 months prior on a different pump. Training currency is not the root issue — the
  problem is supervision and schedule pressure (Category K / H).
- Evidence bundle: WO narrative "venting step deferred per coordinator direction"; completion
  note "start-up acceptable" signed by supervisor; CR post-event references 2-hour schedule
  compression.
- PM compliance: Bearing inspection overdue 3 months — contributing factor only (as-found was
  within limits).

**Fixtures:**

`training_records.json`
```json
{
  "event_id": "E2026-01-22-001",
  "records": [
    { "technician_id": "TECH-4421", "qualification": "MNT-FW-LUBE-OIL",
      "last_task_date": "2025-06-10T00:00:00Z", "currency_status": "current",
      "notes": "Last performance on U2-MFP-A (different unit). Qualification maintained." }
  ],
  "quality": {"coverage_status": "complete"}
}
```

`alarm_log.json` — startup sequence: pump start T+0; bearing temperature first threshold T+12s;
bearing temperature trip T+18s; lube oil flow bypass status tag active throughout.

`configuration_change_records.json` — WO MNT-FW-2026-0114 for lube oil maintenance; completion
records; coordinator schedule note; supervisor sign-off.

`operational_context.json` — `mode=startup`, `percent_rated_power=30`, train B `in_service=False`
(tripped).

**Show-stopper:** The pipeline surfaces two distinct programmatic findings from one event:
`human_performance_assessment` with `performance_mode=execution_error` (the skipped step)
AND a separate `process_procedure` row in the Ishikawa matrix for the inadequate acceptance
criterion. A standard checklist-based RCA tool would require explicit prompting to separate
these; the pipeline identifies them independently from the evidence patterns.

**Assertions:**
| ID | Claim | Path and condition |
|---|---|---|
| A6-1 | Execution error hypothesis primary | `r["run_manifest"]["artifacts"]["rca_card"]["primary_cause_label"]` contains lube oil / venting or maps to Category H candidate |
| A6-2 | Human perf block applicable | `r["rca_card"]["human_performance_assessment"]["applicable"] == True` |
| A6-3 | Execution error finding present | At least one finding with `performance_mode == "execution_error"` in findings list |
| A6-4 | Procedure gap in Ishikawa | `r["ishikawa_matrix"]["process_procedure"]` has ≥1 entry |
| A6-5 | Training records consumed | `r["run_manifest"]["artifacts"]["data_coverage_summary"]["source_families"]["training_records"]["status"] == "complete"` |
| A6-6 | Training not causal | Training candidate does NOT appear as primary hypothesis |
| A6-7 | Startup mode operating point > 0 | `cand["scores"]["operating_point_score"] > 0` for primary candidate |
| A6-8 | Fouling/wear deprioritised | Hypothesis B and C composite scores < Hypothesis A |
| A6-9 | Bearing inspection gap contributing | PM compliance failure linked to bearing candidate, not primary |
| A6-10 | AP-913 completeness block present | `r["run_manifest"].get("ap913_completeness") is not None` (human performance event requires AP-913 completeness tracking) |

---

### test_case_7: Scope Expansion in a Degraded-Data Environment

**Reactor type:** PWR Unit 1 (Northbrook Nuclear)  
**Operational mode:** Full power (100% rated)  
**Event ID:** `E2026-04-20-001`  
**Event:** RCP-C No. 1 seal leakoff flow exceeds administrative action level.

**Scenario narrative:**  
During routine parameter surveillance, operators identify that RCP-C No. 1 seal leakoff flow
(`U1-FT-RCP-C-SEAL1`) is 2.4 gpm — above the administrative action level of 2.0 gpm. The
plant is stable; no automatic trip. A condition report is initiated.

**Why this scenario is compelling for show-and-tell:**  
RCP seal degradation is a primary LOCA precursor. In the early stage it looks almost benign —
a single parameter slightly above an administrative level. This tests whether the pipeline can
surface the significance and identify the causal path from minimal, degraded data.

**Data degradation (deliberate, explained):**  
The plant process computer's data historian was in a scheduled backup maintenance window during
the initial parameter deviation. As a result, `soe_log` and `alarm_log` are not available for
this event. `tskr_patterns` are available because the TSKR subsystem runs on a separate server
with independent data retention.

**Initial scope:** `run_context` is initialised with `active_scope_version=0` (discovery mode)
and scope boundary covering only `U1-RCP-C` and its seal package components.

**Scope expansion:**  
The Allen map identifies a preceding `gradual_drift` anomaly in seal water heat exchanger
inlet temperature (`U1-TE-SW-HX-4C-IN`) — 4°F above normal, onset 6 hours before the seal
flow increase. `U1-SW-HX-4C` is outside the initial scope. The pipeline generates a scope
expansion suggestion with `trigger_type=temporal_predecessor` for this component.

**Two-run demonstration:**

*Run 1 — discovery mode (`active_scope_version=0`):*
- `scope_filter.applied = False`
- `scope_expansion_summary.pending_analyst_decision ≥ 1`
- `data_coverage_summary`: `soe_log` and `alarm_log` both show `"not_assessed"`
- `sensitivity_table.missing_sources_checked` includes `soe_log` and `alarm_log`
- `sensitivity_table.any_ranking_change_possible = True` (missing SOE could affect ranking)
- `analyst_decisions_required` includes scope expansion and SOE data recovery

*Analyst action (in notebook between Run 1 and Run 2):*
```python
orchestrator.resolve_expansion_suggestion(
    signal_id="U1-SW-HX-4C",
    decision="accepted",
    analyst_rationale="HX inlet temperature precursor is plausible thermal path to seal degradation"
)
# active_scope_version advances to 1
```

*Run 2 — scope active (`active_scope_version=1`):*
- `scope_filter.applied = True`, `scope_filter.version = 1`
- `scope_filter.filtered_count ≥ 1` (narrow-scope candidates from Run 1 now filtered out)
- New candidate for `U1-SW-HX-4C` thermal degradation path is now in `candidates`
- Primary hypothesis: SW-HX thermal degradation → reduced seal cooling → seal degradation
- `similar_event_list.any_plant_match = True` (past event EVT-U1-2024-0308 from kg_context)

**Competing hypotheses (within expanded scope):**
- **Hypothesis A — SW-HX-4C fouling / reduced cooling (Category A + chain_position=root).**
  Consistent with 6-hour lead signal. Thermal path: HX fouling → reduced seal cooling →
  elevated seal temperature → increased seal face wear → leakoff increase.
- **Hypothesis B — RCP-C No. 1 seal face wear (Category A + chain_position=proximate).**
  Intrinsic wear at service hours. No thermal precursor predicted; the 6-hour HX temperature
  signal contradicts an independent seal-wear-only hypothesis.
- **Hypothesis C — Seal control valve drift (Category D).** Would produce a step change in
  leakoff flow, not a gradual 6-hour thermal development.

**Fixtures:**

`tskr_patterns.json` — two patterns: `U1-TE-SW-HX-4C-IN` `gradual_drift` onset T-6h (outside
initial scope); `U1-FT-RCP-C-SEAL1` `sustained_exceedance` onset T-0.

`kg_context.json` — initial scope components: `U1-RCP-C`, `U1-RCP-C-SEAL1-PKG`,
`U1-SWP-SEAL-WATER-HX-4C` (the HX, included in KG but outside initial scope boundary);
past event `EVT-U1-2024-0308` (RCP-A seal degradation, confirmed root cause: SW-HX biofouling).

`operational_context.json` — `mode=steady_state`, `percent_rated_power=100`, all four RCPs
`in_service=True`.

`evidence_bundle.json` — RCP seal leakoff administrative limit procedure (SOP-U1-RCP-004);
prior CR from EVT-U1-2024-0308 confirming HX fouling mechanism; SW HX maintenance history.

**Show-stopper:** The before/after comparison between Run 1 and Run 2 makes the scope revision
workflow tangible. A manager can see: Run 1 says "we found something outside your original
scope, here it is"; the analyst approves it in one line of code; Run 2 now includes it and
it becomes the primary hypothesis. That feedback loop — in a live notebook — is impossible
to replicate with a static checklist tool.

**Assertions:**
| ID | Claim | Path and condition |
|---|---|---|
| A7-1 | Run 1: no filter | `r1["run_manifest"]["artifacts"]["scope_filter"]["applied"] == False` |
| A7-2 | Run 1: expansion suggestion generated | `len(r1["run_context"]["scope_management"]["expansion_suggestions"]) >= 1` |
| A7-3 | Run 1: SOE not assessed | `r1["run_manifest"]["artifacts"]["data_coverage_summary"]["source_families"]["soe_log"]["status"] == "not_assessed"` |
| A7-4 | Run 1: alarm not assessed | same path for `alarm_log` |
| A7-5 | Run 1: SOE in sensitivity check | `"soe_log" in r1["run_manifest"]["artifacts"]["sensitivity_table"]["missing_sources_checked"]` |
| A7-6 | Run 2: scope filter active | `r2["run_manifest"]["artifacts"]["scope_filter"]["applied"] == True` and `r2["run_manifest"]["artifacts"]["scope_filter"]["approved_scope_version"] == 1` |
| A7-7 | Run 2: filtered candidates | `r2["run_manifest"]["artifacts"]["scope_filter"]["filtered_count"] >= 1` |
| A7-8 | Run 2: HX candidate present | At least one candidate in `r2["causality_candidates"]["candidates"]` with component `U1-SW-HX-4C` or `U1-SWP-SEAL-WATER-HX-4C` |
| A7-9 | Run 2: HX path is primary | `r2["run_manifest"]["artifacts"]["rca_card"]["primary_cause_label"]` references HX or thermal path |
| A7-10 | Prior event match | `r2["run_manifest"]["artifacts"]["similar_event_list"]["any_plant_match"] == True` |
| A7-11 | Seal wear deprioritised in Run 2 | Seal intrinsic wear candidate composite score < HX thermal path candidate |
| A7-12 | Unresolved gap for missing SOE | `r1["rca_card"]["executive_summary"]["unresolved_gaps"]` contains ≥1 entry referencing `soe_log` or data absence |
| A7-13 | Causal depth complete in Run 2 | `r2["rca_card"]["executive_summary"]["causal_depth_summary"]["depth_complete"] == True` (root cause resolved once HX is in scope) |
| A7-14 | Chain positions in Run 2 | HX fouling candidate has `chain_position == "root"`; seal leakoff candidate has `chain_position == "proximate"` |

---

## Part 3 — Shared Infrastructure

### Directory layout (all cases)

```
test_case_N/
  description.md                     ← scenario narrative (one file per case, standalone)
  fixtures/
    event.json
    kg_context.json
    telemetry_summary.json
    evidence_bundle.json              ← {"results": [...], "metadata": {...}}
    operational_context.json          ← if applicable
    pm_compliance.json                ← if applicable
    soe_log.json                      ← if applicable
    alarm_log.json                    ← if applicable
    protection_logic_context.json     ← if applicable
    tskr_patterns.json                ← if applicable
    configuration_change_records.json ← if applicable
    environmental_monitoring.json     ← if applicable
    vendor_supply_chain_records.json  ← if applicable
    training_records.json             ← if applicable
  data_generator.py                   ← reproduces fixtures deterministically
  run_test_case_N.ipynb               ← pre-executed notebook (show-and-tell artifact)
  rca_runs_case_00N/
    v32/                              ← saved run output (artifact store output)
    v32_full_result.json              ← top-level result dict serialised
```

### Shared helper module

`tests/shared/`
- `run_helpers.py` — `load_fixtures(case_dir)`, `build_orchestrator(run_dir)`,
  `print_score_table(candidates)`, `print_rca_card_summary(rca_card)`,
  `print_ishikawa_summary(ishikawa_matrix)`, `print_scope_filter_summary(run_manifest)`
- `assertion_helpers.py` — `check_primary(result, expected_label_fragment)`,
  `check_scope_filter(result, expected_applied)`, `check_human_perf(result, expected_mode)`,
  `check_coverage_status(result, family, expected_status)`, `run_assertions(result, checks)`

### Notebook standard structure (all cases)

| Cell | Type | Content |
|---|---|---|
| 1 | Code | Imports; `from shared.run_helpers import *`; `from shared.assertion_helpers import *` |
| 2 | Markdown | **Scenario context** — 1–2 paragraph narrative (what happened, why it matters) |
| 3 | Code | `fixtures = load_fixtures(case_dir)`; `orc = build_orchestrator(run_dir)` |
| 4 | Code | `result = orc.run(**fixtures)` (one run, or Run 1 for TC-7) |
| 5 | Code | `print_score_table(result["causality_candidates"])` — ranked candidate table |
| 6 | Code | `print_rca_card_summary(result["rca_card"])` — primary, alternatives, gaps |
| 7 | Code | `print_ishikawa_summary(result["ishikawa_matrix"])` |
| 8 | Code | (TC-7 only) Analyst action + Run 2 |
| 9 | Code | `run_assertions(result, ASSERTIONS)` — pass/fail table |
| 10 | Markdown | **What did the system do?** — 1 sentence per assertion, plain English |

---

## Execution Sequence

| Phase | Scope | Notes |
|---|---|---|
| **A** | `tests/shared/` infrastructure | Do first — unblocks all other phases |
| **B** | Update TC-2 (fixtures + notebook) | High audience value; most complete existing case |
| **C** | Update TC-3 (data_generator.py + new fixtures) | Highest analytical depth |
| **D** | TC-4 (Reactor Trip, PLC gates) | First new case; directly demos Finding I |
| **E** | TC-5 (ECCS CCF, vendor supply chain) | Demos Item 3 (CCF scoring) and vendor path |
| **F** | TC-6 (Human Performance) | Demos H/I categories and training records path |
| **G** | TC-7 (Scope Expansion, two runs) | Largest; most visually compelling for managers |
| **H** | Update TC-1 (minor polish) | Low priority; baseline case |

---

## Design Decisions Locked

| Decision | Choice | Rationale |
|---|---|---|
| Pseudo-plant naming | `U1-`/`U2-`/`U3-` prefix, ISA-5.1 tags | Authentic feel for SE audience |
| Evidence bundle format | `{"results": [...]}` JSON dict | Matches `evidence_bundle.get("results")` in orchestrator |
| Notebook execution | Pre-executed, saved outputs | Enables offline presentation |
| Multi-run structure | Multi-cell (Run 1 cell → analyst action cell → Run 2 cell) | Clearest narrative flow for TC-7 |
| Failure mode latency | Not included in any fixture | Avoids KG-parameter dependency; all discrimination through observable evidence |
| `tskr_patterns` | Pre-built fixture for cases where temporal reasoning is the show-stopper | Ensures temporal assertions are deterministic |
| `causality_candidates` | NOT pre-supplied (always generated by engine) | Ensures scoring logic is exercised, not bypassed |

---

## Workflow Mechanics Coverage Summary

Cross-reference of which pipeline mechanics are exercised and where:

| Mechanic | Where tested |
|---|---|
| Hard gate: physical plausibility | TC-4, TC-5 (PLC-based), TC-7 (scope boundary) |
| Hard gate: timeline consistency | TC-4 (CRD ruled out by SOE T+10s) |
| Hard gate: barrier logic + PLC | TC-4 (`held`), TC-5 (`failed`), TC-2 (`armed`) |
| Allen temporal scoring | TC-2 (air removal precedes vacuum), TC-3 (multi-timescale), TC-4 (FW valve → NI), TC-7 (HX precedes seal) |
| Operating point score (Category E) | TC-4 (full power), TC-6 (startup mode), TC-5 (cold shutdown / train OOS) |
| CCF structural delta (Category C) | TC-5 (common vendor lot — CCF scored and boosted) |
| Human performance block (H/I/J/K) | TC-5 (procedure gap / Category I), TC-6 (execution error / Category H) |
| Evidence refinement rank change | TC-3 (fouling → air in-leakage flip explicitly asserted) |
| Similar event list (plant tier) | TC-3 (3 past events, recurrence trap), TC-7 (EVT-U1-2024-0308 match) |
| Sensitivity table | TC-7 (SOE/alarm not assessed → ranking change possible) |
| Signal lessons learned | TC-4 (NI single-channel pattern matched) |
| Data coverage degraded path | TC-7 (soe_log + alarm_log both `not_assessed`); TC-5 (vendor advisory non-dispositioned) |
| soe_plc_pairing = "violated" | TC-2 bonus demo cell (deliberately omit PLC, show the flag, then fix it) |
| Scope revision + downstream filter | TC-7 (two-run demo: version 0 → version 1, filter active, candidates differ) |
| Scope expansion suggestion | TC-7 (HX outside initial scope, expansion suggestion generated) |
| Barrier analysis artifact | TC-5 (two `failed` HPCI barriers in barrier_analysis) |
| Ishikawa matrix population | TC-2 (multi-branch), TC-5 (process_procedure), TC-6 (maintenance + process_procedure) |
| Unresolved gaps | TC-7 (SOE absence → gap logged in rca_card) |
| Causal chain depth (proximate/root) | TC-7 Run 2 (HX=root, seal=proximate, depth_complete=True) |
| AP-913 completeness block | TC-6 (human performance event) |
| Decision trail | TC-4 (CRD candidate: `timeline_inconsistent` reason code in trail) |
| Recurrence scoring | TC-3 (recurrence trap: 2 air in-leakage events vs 1 fouling) |
| Environmental monitoring (Category F) | TC-3 (CW temp seasonal, condenser pit ambient), TC-4 (containment radiation benign) |
| Configuration change records | TC-3 (waterbox cleaning WO contradicts fouling), TC-5 (coupling WO traces to lot), TC-6 (maintenance WO schedule pressure) |
| Vendor supply chain records | TC-5 (vendor advisory non-disposition as CCF evidence) |
| Training records | TC-6 (technician currency current → not causal) |
| PM compliance | TC-2, TC-3, TC-5 (train A overdue), TC-6 (bearing overdue contributing) |

**Causal categories exercised across the suite (metamodel A–L):**

| Category | Description | Test cases |
|---|---|---|
| A | Equipment-internal degradation | TC-1, TC-2, TC-3, TC-4, TC-5, TC-6, TC-7 (baseline hypothesis in all) |
| B | Required support not available/degraded | TC-2 (HVAC → AC cabinet), TC-7 (SW-HX → seal cooling) |
| C | CCF / upstream common influence | TC-5 (common vendor lot — primary focus) |
| D | Downstream influence / instrumentation | TC-4 (NI spurious signal) |
| E | Operating context / mission demand | TC-4 (full power), TC-5 (cold shutdown), TC-6 (startup) |
| F | External hazards and disturbances | TC-4 (EMI in evidence bundle for NI spurious), TC-3 (seasonal CW temperature) |
| G | Human and organizational contributors | TC-6 (execution error: venting step skipped, supervisor sign-off) |
| H | Design and specification deficiencies | Not explicitly exercised — acceptable gap for this suite |
| I | Configuration and change control | TC-5 (procedure gap: torque criterion absent), TC-6 (procedure acceptance criterion gap) |
| J | Inspection and testing program inadequacy | TC-5 (surveillance procedure doesn't include vendor torque criterion) |
| K | Vendor and supply chain | TC-5 (lot VEN-2026-Q1-HC7 defect — primary focus) |
| L | Systemic and latent organizational | Surfaced implicitly in TC-5 (undispositioned advisory → CAP gap) and TC-6 (schedule pressure) but not as a primary ranked candidate — acceptable gap |

---

## Accepted Gaps

These mechanics exist in the codebase but are not the primary focus of any test case.
They are either low-priority for the show-and-tell audience or too fragile to assert on
without knowing LLM-synthesized card content:

| Gap | Why accepted |
|---|---|
| `writeback_recommendation` value | Driven by LLM card synthesis; content too variable to assert reliably |
| `effectiveness_monitoring_plan` | Same reason; presence is implicitly tested, content is not |
| Category H (Design deficiency) | Requires domain-specific KG nodes; low audience priority |
| Category L (Systemic weakness) | Qualitative category; surfaced narratively in TC-5/TC-6 but not as a ranked candidate |
| Quality multiplier exact value | Implicitly tested through sensitivity table; no direct assertion needed |
| `reentry_execution` block | Auto-reentry is tested in unit tests; scenario tests focus on primary path |
| `cmms_context` (CMMS adapter) | Requires external CMMS adapter; not applicable to fixture-based tests |

