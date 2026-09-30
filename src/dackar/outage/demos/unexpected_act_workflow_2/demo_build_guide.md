# Demo Build Guide: Synthetic MVP Scenario
**Version:** 1.0
**Purpose:** Step-by-step instructions for constructing the synthetic illustrative dataset and running the full pipeline for the first stakeholder meeting.

---

## Scenario Overview

> **Dataset scope note:** This is a reduced synthetic dataset designed for transparency and walkthrough clarity. The same pipeline architecture scales to full plant datasets (500+ components, 5,000+ activities, multiple outages) without structural changes. The five-component scope is a deliberate demo choice, not a system limitation.

> **Date note:** All dates are illustrative and are not intended to reflect real outage cycles or actual plant schedules.

**Fictional plant:** Millbrook Nuclear Station, Unit 1 (generic PWR)
**System in focus:** Residual Heat Removal (RHR) — chosen because it is safety-significant, well-known across the industry, and commonly generates outage work
**Training outages:** RF-20, RF-21 (two refueling outages)
**Test outage:** RF-22 (holdout — what the system predicts pre-outage)
**Anchor story:** RHR Pump 1A has a seal degradation pattern across RF-20 and RF-21 that the system should flag before RF-22. Emergent seal replacement work did occur in RF-22.

### The Five Components

| Component ID | Description | System | Role in Demo |
|---|---|---|---|
| `1RHS-P-001A` | RHR Pump 1A | Residual Heat Removal | Anchor — clean causal chain, DATA-SUPPORTED tier |
| `1RHS-E-001A` | RHR Heat Exchanger 1A | Residual Heat Removal | Supporting — partial linkage, SME-INFORMED tier |
| `1CSP-P-001B` | Containment Spray Pump 1B | Containment Spray | Weak linkage — LOW-CONFIDENCE WATCH tier |
| `1CCW-P-002A` | Component Cooling Water Pump 2A | Component Cooling Water | No prior emergent history — true negative |
| `1RHS-V-001A` | RHR Suction Isolation Valve 1A | Residual Heat Removal | No prior history — true negative |

### Confidence Tier Coverage
The five components are deliberately chosen to exercise all three confidence tiers and both true positive and true negative outcomes, so the demo can show the full range of system behavior.

---

## Step 1 — Generate the Asset Registry

This is the canonical component table. Every other dataset links back to it.

```csv
component_id,description,system,system_code,plant_tag,component_type,regulatory_constraint_flag,notes
1RHS-P-001A,RHR Pump 1A,Residual Heat Removal,RHS,1RHS-P-001A,Pump,TRUE,Tech Spec 3.5.2 — operability required
1RHS-E-001A,RHR Heat Exchanger 1A,Residual Heat Removal,RHS,1RHS-E-001A,Heat Exchanger,TRUE,Tech Spec 3.5.2
1CSP-P-001B,Containment Spray Pump 1B,Containment Spray,CSP,1CSP-P-001B,Pump,TRUE,Tech Spec 3.6.6
1CCW-P-002A,Component Cooling Water Pump 2A,Component Cooling Water,CCW,1CCW-P-002A,Pump,FALSE,No active Tech Spec action level
1RHS-V-001A,RHR Suction Isolation Valve 1A,Residual Heat Removal,RHS,1RHS-V-001A,Valve,TRUE,Tech Spec 3.5.2
```

**Notes for implementation:**
- `regulatory_constraint_flag = TRUE` means the recommendation engine must not suggest deferral without an override acknowledgment
- Plant tag format follows standard PWR convention: `[unit][system code]-[type]-[sequence][train]`
- All five components should be loadable as nodes of type `component` in the knowledge graph

---

## Step 2 — Generate Condition Reports

Generate 15 CRs across the three outage cycles. Distribution should be:
- RF-20 prep period: 4 CRs
- RF-21 prep period: 6 CRs
- RF-22 prep period (pre-outage, used as prediction input): 5 CRs

**CR ID format:** `CR-[YEAR]-[5-digit sequence]`
**Date format:** ISO 8601 (`YYYY-MM-DD`)

```csv
cr_id,component_id,created_date,outage_cycle,description_raw,cr_category,linked_wo_id
CR-2019-04412,1RHS-P-001A,2019-08-14,RF-20 prep,"1RHS-P-001A noted vibration above baseline during quarterly surveillance. Vib reading 0.42 in/s vs 0.30 in/s baseline. Maint notified. Monitor.",observation,
CR-2019-06891,1RHS-P-001A,2019-11-22,RF-20 prep,"Ops noted minor slt lkg at pump mech seal during walkdown. No active drip. Vlv pkgs ok. Recommend insp during next OT.",degradation,WO-2019-52341
CR-2020-01203,1RHS-E-001A,2020-01-08,RF-20 prep,"HX 1A outlet temp trending 2.3 deg F below design basis. Possibly biofouling or partial tube plugging. Schedule tube inspection at RF-20.",observation,WO-2020-10042
CR-2020-02871,1CSP-P-001B,2020-02-17,RF-20 prep,"1CSP-P-001B motor current trending high — 43A vs 39A baseline. Bearings within spec. No immediate action. Monitor.",observation,
CR-2021-00892,1RHS-P-001A,2021-03-03,RF-21 prep,"Repeat slt lkg noted at mech seal 1RHS-P-001A. Lkg rate approx 1-2 drops/min. Elevated from prior CR-2019-06891. Recommend seal insp & possible repl at RF-21.",degradation,WO-2021-38471
CR-2021-02234,1RHS-P-001A,2021-05-19,RF-21 prep,"Vib trending continued — now 0.51 in/s. Correl with seal lkg — possible seal face wear contributing to shaft movement. Priority elevated.",degradation,WO-2021-38471
CR-2021-03301,1RHS-E-001A,2021-06-30,RF-21 prep,"Tube inspection from RF-20 showed 3 tubes with wall loss >20%. Plugged per procedure. Thermal performance marginally acceptable. Re-inspect RF-21.",degradation,WO-2021-44892
CR-2021-05512,1CSP-P-001B,2021-08-11,RF-21 prep,"Motor current now 45A. Bearing temps within limits. Possible impeller wear. Schedule inspection at RF-21.",observation,WO-2021-47201
CR-2021-07743,1RHS-P-001A,2021-10-04,RF-21 prep,"Pre-outage walkdown: mech seal lkg now 3-4 drops/min. WO-2021-38471 confirmed for RF-21. Parts on order — seal kit P/N RHS-SK-0042.",degradation,WO-2021-38471
CR-2021-09981,1CCW-P-002A,2021-11-28,RF-21 prep,"Minor oil seepage noted at 1CCW-P-002A bearing housing. Within acceptable limits. No corrective action at this time.",observation,
CR-2022-01142,1RHS-P-001A,2022-02-08,RF-22 prep,"Post RF-21 seal replacement: new seal installed. However vib reading still 0.44 in/s — not fully resolved. Monitor for recurrence.",observation,WO-2022-20341
CR-2022-03387,1RHS-P-001A,2022-05-14,RF-22 prep,"Vib now 0.49 in/s. Seal replaced at RF-21 but vibration pattern suggests possible impeller wear or bearing degradation contributing. Recommend enhanced insp at RF-22.",degradation,WO-2022-31102
CR-2022-04901,1RHS-E-001A,2022-07-02,RF-22 prep,"Thermal performance trending down again — 3.1 deg F below design basis. Tube plugging from RF-21 may be insufficient. Schedule full tube inspection RF-22.",degradation,WO-2022-33891
CR-2022-06234,1CSP-P-001B,2022-08-19,RF-22 prep,"Motor current 46A — marginal. Bearing temp now slightly elevated (162F vs 155F baseline). Schedule bearing replacement RF-22.",degradation,WO-2022-35102
CR-2022-08801,1RHS-V-001A,2022-10-11,RF-22 prep,"Valve position indication verified during surveillance. No anomalies noted. Routine observation — no action required.",observation,
```

**Key design decisions in this dataset:**
- `1RHS-P-001A` has a clear escalating pattern across all three cycles — vibration and seal leakage both trending upward — that a planner should have acted on before RF-22
- The raw descriptions are written in realistic compressed plant style: abbreviations (`slt lkg`, `mech seal`, `vib`, `repl`, `insp`, `OT` for outage), alphanumeric part numbers, cross-references to other CRs
- `1CCW-P-002A` and `1RHS-V-001A` have only minor or clean observations — they are the true negatives the system should not flag
- Some CRs have no linked WO (observation only) — this exercises the incomplete linkage scenario

---

## Step 3 — Generate Work Orders

Generate 9 WOs linked to the CRs above. Not every CR generates a WO — this is realistic and exercises the sparse linkage scenario.

**WO ID format:** `WO-[YEAR]-[5-digit sequence]`

```csv
wo_id,component_id,created_date,outage_cycle,description_raw,wo_type,planned_duration_hrs,actual_duration_hrs,completed_flag,emergent_flag
WO-2019-52341,1RHS-P-001A,2019-12-01,RF-20,"1RHS-P-001A MECH SEAL INSP — REMOVE PUMP CASING COVER, INSP SEAL FACES, MEASURE CLEARANCES, RPT FINDINGS. REF CR-2019-06891.",corrective,8.0,9.5,TRUE,FALSE
WO-2020-10042,1RHS-E-001A,2020-01-15,RF-20,"1RHS-E-001A TUBE INSP — EDDY CURRENT TEST ALL TUBES, PLUG DEGRADED TUBES PER PROC ENG-HX-042, RPT RESULTS.",corrective,24.0,31.0,TRUE,FALSE
WO-2021-38471,1RHS-P-001A,2021-03-10,RF-21,"1RHS-P-001A MECH SEAL INSP & REPL — REMOVE & INSP EXISTING SEAL, INSTALL NEW SEAL KIT P/N RHS-SK-0042, ALIGN PUMP, PERFORM POST-MAINT TEST. REF CR-2021-00892 CR-2021-02234 CR-2021-07743.",corrective,16.0,24.0,TRUE,FALSE
WO-2021-44892,1RHS-E-001A,2021-07-08,RF-21,"1RHS-E-001A TUBE INSP RF-21 — EDDY CURRENT TEST, ASSESS TUBES FLAGGED RF-20, PLUG AS REQUIRED, UPDATE TUBE MAP.",corrective,20.0,22.0,TRUE,FALSE
WO-2021-47201,1CSP-P-001B,2021-08-20,RF-21,"1CSP-P-001B MOTOR & IMPELLER INSP — PULL MOTOR, INSP BEARINGS, INSP IMPELLER FOR WEAR, RPT. REF CR-2021-05512.",corrective,12.0,10.5,TRUE,FALSE
WO-2022-20341,1RHS-P-001A,2022-02-15,RF-22 prep,"1RHS-P-001A POST-MAINT MONITORING — VIB TRENDING FOLLOWING RF-21 SEAL REPL. DOCUMENT READINGS MONTHLY. REF CR-2022-01142.",observation,2.0,,FALSE,FALSE
WO-2022-31102,1RHS-P-001A,2022-05-20,RF-22,"1RHS-P-001A ENHANCED INSP RF-22 — INSP MECH SEAL, BEARINGS, IMPELLER. ASSESS VIB ROOT CAUSE. REPL AS REQUIRED. REF CR-2022-03387.",corrective,20.0,,FALSE,FALSE
WO-2022-33891,1RHS-E-001A,2022-07-10,RF-22,"1RHS-E-001A FULL TUBE INSP RF-22 — EDDY CURRENT ALL TUBES, ASSESS PLUGGING STRATEGY, RPT.",corrective,24.0,,FALSE,FALSE
WO-2022-35102,1CSP-P-001B,2022-08-25,RF-22,"1CSP-P-001B BEARING REPL RF-22 — PULL MOTOR, REPLACE BEARINGS, INSPECT IMPELLER, REINSTALL, PMT. REF CR-2022-06234.",corrective,14.0,,FALSE,FALSE
```

**Key design decisions:**
- WOs in RF-22 prep have no `actual_duration_hrs` and `completed_flag = FALSE` — they are planned but not yet executed, simulating the pre-outage prediction window
- `WO-2021-38471` references three CRs in its description — this exercises the multi-CR linkage scenario and the NLP's ability to extract CR IDs from free text
- Actual durations in RF-20 and RF-21 are longer than planned — this provides the training signal for schedule impact estimation
- `WO-2022-20341` is an observation/monitoring WO — not a corrective action — which adds realism and exercises WO type classification

---

## Step 4 — Generate Outage Activity Records

Generate 15 activities across RF-20 and RF-21 (7 in RF-20, 8 in RF-21) plus 5 planned activities for RF-22 pre-outage state, giving 20 total. Include 2 emergent activities in each training outage to provide ground truth pattern data. RF-22 ground truth emergent records are kept in a separate subsection below and must not be loaded into the pipeline until after the prediction step has been captured — see "RF-22 Ground Truth Records" below.

**Activity ID format:** `[OUTAGE]-[ROLE]-[4-digit sequence]`
**Emergent flag:** `TRUE` for activities not in the original plan

```csv
activity_id,outage_id,component_id,linked_wo_id,activity_name,role_id,planned_start,planned_end,actual_start,actual_end,planned_duration_hrs,actual_duration_hrs,emergent_flag,emergence_category,on_critical_path,float_hrs
RF20-MECH-0042,RF-20,1RHS-P-001A,WO-2019-52341,"1RHS-P-001A MECH SEAL INSP & RPT",MECH,2020-03-02 08:00,2020-03-02 16:00,2020-03-02 08:30,2020-03-03 09:30,8.0,9.5,FALSE,,FALSE,18.0
RF20-MECH-0071,RF-20,1RHS-E-001A,WO-2020-10042,"1RHS-E-001A EDDY CURR TUBE INSP",MECH,2020-03-04 07:00,2020-03-05 07:00,2020-03-04 07:00,2020-03-05 14:00,24.0,31.0,FALSE,,TRUE,0.0
RF20-MECH-0089,RF-20,1RHS-P-001A,,"1RHS-P-001A SEAL FACE REPL — EMERGENT — INSP FOUND WEAR BEYOND ACCEPTABLE LIMITS REF WO-2019-52341",MECH,,,2020-03-03 10:00,2020-03-04 02:00,,16.0,TRUE,DISCOVERY,TRUE,0.0
RF20-MECH-0094,RF-20,1RHS-E-001A,,"1RHS-E-001A ADDL TUBE PLUGGING — EMERGENT — 3 TUBES ABOVE PLUGGING LIMIT FOUND DURING INSP",MECH,,,2020-03-05 15:00,2020-03-05 22:00,,7.0,TRUE,DISCOVERY,TRUE,0.0
RF20-ELEC-0021,RF-20,1CSP-P-001B,,"1CSP-P-001B MTR MEGGR & ELEC INSP",ELEC,2020-03-06 08:00,2020-03-06 12:00,2020-03-06 08:00,2020-03-06 11:30,4.0,3.5,FALSE,,FALSE,24.0
RF20-MECH-0103,RF-20,1CCW-P-002A,,"1CCW-P-002A BEARING INSP & LUBE",MECH,2020-03-07 08:00,2020-03-07 12:00,2020-03-07 08:00,2020-03-07 11:00,4.0,3.0,FALSE,,FALSE,32.0
RF20-OPS-0011,RF-20,1RHS-V-001A,,"1RHS-V-001A VALVE STROKE TEST & INSP",OPS,2020-03-08 06:00,2020-03-08 10:00,2020-03-08 06:00,2020-03-08 09:30,4.0,3.5,FALSE,,FALSE,40.0
RF21-MECH-0038,RF-21,1RHS-P-001A,WO-2021-38471,"1RHS-P-001A MECH SEAL REPL & ALIGN",MECH,2022-03-01 07:00,2022-03-01 23:00,2022-03-01 07:30,2022-03-02 07:30,16.0,24.0,FALSE,,TRUE,0.0
RF21-MECH-0052,RF-21,1RHS-E-001A,WO-2021-44892,"1RHS-E-001A TUBE INSP & PLUGGING",MECH,2022-03-03 07:00,2022-03-04 03:00,2022-03-03 07:00,2022-03-04 05:00,20.0,22.0,FALSE,,TRUE,0.0
RF21-MECH-0061,RF-21,1CSP-P-001B,WO-2021-47201,"1CSP-P-001B MTR PULL BEARING INSP",MECH,2022-03-05 08:00,2022-03-05 20:00,2022-03-05 08:00,2022-03-05 18:30,12.0,10.5,FALSE,,FALSE,16.0
RF21-MECH-0079,RF-21,1RHS-P-001A,,"1RHS-P-001A IMPELLER INSP — EMERGENT — VIB ROOT CAUSE INVEST IDENTIFIED IMPELLER WEAR REF WO-2021-38471",MECH,,,2022-03-02 08:00,2022-03-02 20:00,,12.0,TRUE,DISCOVERY,TRUE,0.0
RF21-MECH-0083,RF-21,1RHS-E-001A,,"1RHS-E-001A ADDL TUBE PLUGGING — EMERGENT — 2 ADDL TUBES FOUND DEGRADED BEYOND LIMIT DURING INSP",MECH,,,2022-03-04 06:00,2022-03-04 12:00,,6.0,TRUE,DISCOVERY,TRUE,0.0
RF21-ELEC-0019,RF-21,1CSP-P-001B,,"1CSP-P-001B MTR WINDING INSP & RPT",ELEC,2022-03-06 08:00,2022-03-06 14:00,2022-03-06 08:00,2022-03-06 13:00,6.0,5.0,FALSE,,FALSE,20.0
RF21-MECH-0091,RF-21,1CCW-P-002A,,"1CCW-P-002A BEARING INSP & LUBE",MECH,2022-03-07 08:00,2022-03-07 12:00,2022-03-07 08:00,2022-03-07 11:30,4.0,3.5,FALSE,,FALSE,28.0
RF21-OPS-0008,RF-21,1RHS-V-001A,,"1RHS-V-001A VALVE STROKE TEST",OPS,2022-03-08 06:00,2022-03-08 10:00,2022-03-08 06:00,2022-03-08 09:00,4.0,3.0,FALSE,,FALSE,36.0
RF22-MECH-0041,RF-22,1RHS-P-001A,WO-2022-31102,"1RHS-P-001A ENHANCED INSP — SEAL BEARING IMPELLER",MECH,2024-03-01 07:00,2024-03-02 03:00,,,20.0,,FALSE,,TRUE,0.0
RF22-MECH-0055,RF-22,1RHS-E-001A,WO-2022-33891,"1RHS-E-001A FULL TUBE INSP & PLUGGING",MECH,2024-03-03 07:00,2024-03-04 03:00,,,20.0,,FALSE,,TRUE,0.0
RF22-MECH-0063,RF-22,1CSP-P-001B,WO-2022-35102,"1CSP-P-001B BEARING REPL & PMT",MECH,2024-03-05 07:00,2024-03-05 19:00,,,12.0,,FALSE,,FALSE,14.0
RF22-MECH-0072,RF-22,1CCW-P-002A,,"1CCW-P-002A BEARING INSP & LUBE",MECH,2024-03-06 08:00,2024-03-06 12:00,,,4.0,,FALSE,,FALSE,30.0
RF22-OPS-0009,RF-22,1RHS-V-001A,,"1RHS-V-001A VALVE STROKE TEST & INSP",OPS,2024-03-07 06:00,2024-03-07 10:00,,,4.0,,FALSE,,FALSE,38.0
```

### RF-22 Ground Truth Records *(holdout reveal — load only after prediction outputs are captured)*

> ⚠️ These records represent what actually happened during RF-22. They must not be loaded into the graph or visible to the pipeline during the prediction step. In the demo, they are revealed only on the final showcase slide ("what actually happened"). Loading them prematurely invalidates the blind evaluation structure.

```csv
activity_id,outage_id,component_id,linked_wo_id,activity_name,role_id,planned_start,planned_end,actual_start,actual_end,planned_duration_hrs,actual_duration_hrs,emergent_flag,emergence_category,on_critical_path,float_hrs
RF22-MECH-0089,RF-22,1RHS-P-001A,,"1RHS-P-001A BEARING & IMPELLER REPL — EMERGENT — ENHANCED INSP FOUND BEARING WEAR BEYOND LIMITS & IMPELLER EROSION. REPL BOTH. REF WO-2022-31102 CR-2022-03387",MECH,,,2024-03-02 06:00,2024-03-03 02:00,,20.0,TRUE,DISCOVERY,TRUE,0.0
RF22-MECH-0094,RF-22,1RHS-E-001A,,"1RHS-E-001A ADDL TUBE PLUGGING — EMERGENT — 4 TUBES FOUND ABOVE PLUGGING LIMIT DURING INSP — PLUGGED PER ENG-HX-042",MECH,,,2024-03-04 06:00,2024-03-04 14:00,,8.0,TRUE,DISCOVERY,TRUE,0.0
```

**RF-22 ground truth schedule impact for showcase Slide 4:**

| Component | Emergent activity | Critical path | Actual duration | Float consumed |
|---|---|---|---|---|
| `1RHS-P-001A` | RF22-MECH-0089 — Bearing & impeller replacement | Yes | 20 hrs | 20 hrs |
| `1RHS-E-001A` | RF22-MECH-0094 — Additional tube plugging | Yes | 8 hrs | 8 hrs |

**Slide 4 narrative (anchor scenario punchline):**
> *"The system flagged 1RHS-P-001A as DATA-SUPPORTED risk before RF-22. During the outage, enhanced inspection confirmed bearing wear and impeller erosion beyond acceptable limits. Emergent replacement consumed 20 hours of critical path float. This is consistent with the 16- and 12-hour critical path impacts observed in RF-20 and RF-21 respectively. The heat exchanger also generated emergent tube plugging as predicted — 8 additional hours on critical path."*

**Key design decisions:**
- RF-22 activities have no actual times — this is the pre-outage prediction state
- Both emergent activities in RF-20 and RF-21 are on the critical path (`on_critical_path = TRUE`, `float_hrs = 0.0`) — this provides a strong training signal that `1RHS-P-001A` emergent work consumes critical path float
- Planned vs. actual duration gaps in training outages provide the delay estimation training signal
- `emergence_category = DISCOVERY` for all emergent activities — consistent with the taxonomy in the test case spec

---

## Step 5 — Generate the Schedule Table

A simplified schedule providing float and critical path data for each activity.

```csv
activity_id,outage_id,critical_path_flag,total_float_hrs,float_consumed_hrs,predecessor_activity_ids,successor_activity_ids
RF20-MECH-0042,RF-20,FALSE,18.0,0.0,,RF20-MECH-0089
RF20-MECH-0071,RF-20,TRUE,0.0,7.0,,RF20-MECH-0094
RF20-MECH-0089,RF-20,TRUE,0.0,16.0,RF20-MECH-0042,
RF20-MECH-0094,RF-20,TRUE,0.0,7.0,RF20-MECH-0071,
RF20-ELEC-0021,RF-20,FALSE,24.0,0.0,,
RF20-MECH-0103,RF-20,FALSE,32.0,0.0,,
RF20-OPS-0011,RF-20,FALSE,40.0,0.0,,
RF21-MECH-0038,RF-21,TRUE,0.0,8.0,,RF21-MECH-0079
RF21-MECH-0052,RF-21,TRUE,0.0,2.0,,RF21-MECH-0083
RF21-MECH-0061,RF-21,FALSE,16.0,0.0,,
RF21-MECH-0079,RF-21,TRUE,0.0,12.0,RF21-MECH-0038,
RF21-MECH-0083,RF-21,TRUE,0.0,6.0,RF21-MECH-0052,
RF21-ELEC-0019,RF-21,FALSE,20.0,0.0,,
RF21-MECH-0091,RF-21,FALSE,28.0,0.0,,
RF21-OPS-0008,RF-21,FALSE,36.0,0.0,,
RF22-MECH-0041,RF-22,TRUE,0.0,,,
RF22-MECH-0055,RF-22,TRUE,0.0,,,
RF22-MECH-0063,RF-22,FALSE,14.0,,,
RF22-MECH-0072,RF-22,FALSE,30.0,,,
RF22-OPS-0009,RF-22,FALSE,38.0,,,
```

---

## Step 6 — NLP Extraction

Run the NLP pipeline on the `activity_name`, `description_raw` (CRs), and `description_raw` (WOs) fields. For the synthetic demo, the following extractions should be validated manually and used as ground truth for pipeline testing.

### Expected NLP Outputs

#### Abbreviation Expansion (must fire on these cases)
| Raw token | Expanded form |
|---|---|
| `slt lkg` | slight leakage |
| `mech seal` | mechanical seal |
| `vib` | vibration |
| `repl` | replace / replacement |
| `insp` | inspect / inspection |
| `OT` | outage |
| `HX` | heat exchanger |
| `MTR` | motor |
| `PMT` | post-maintenance test |
| `RPT` | report |
| `CURR` | current |
| `ADDL` | additional |
| `INVEST` | investigation |
| `ELEC` | electrical |

#### Named Entity Extraction (nuclear entity classes)
| Entity | Class | Source records |
|---|---|---|
| `pump` | Asset — Mechanical | Multiple |
| `seal` / `mechanical seal` | Mechanical component — Purpose specific | CR-2019-06891, WO-2021-38471 |
| `heat exchanger` | Asset — Mechanical | Multiple |
| `valve` | Asset — Hydraulic/Pneumatic | RF20-OPS-0011 |
| `bearing` | Mechanical component — Rotary elements | WO-2021-47201 |
| `impeller` | Mechanical component — Rotary elements | RF21-MECH-0079 |
| `vibration` | Surveillance operation | CR-2019-04412 |
| `leakage` | Failure type (Reactions) | CR-2019-06891 |
| `wear` | Degradation mechanism | RF20-MECH-0089 |
| `tube` | Nonmechanical component | WO-2020-10042 |
| `motor` | Asset — Electrical | WO-2021-47201 |

#### Plant ID Extraction (alphanumeric pattern matching)
The following IDs should be extracted as `plant_element_id` nodes:
`1RHS-P-001A`, `1RHS-E-001A`, `1CSP-P-001B`, `1CCW-P-002A`, `1RHS-V-001A`, `RHS-SK-0042`, `ENG-HX-042`

#### Work Order Reference Extraction
The following WO IDs appear in CR or activity description text and should be extracted as `work_order` node references:
`WO-2019-52341`, `WO-2021-38471` (appears in RF21-MECH-0079 description)

#### CR Reference Extraction
The following CR IDs appear in WO description text:
`CR-2019-06891` (in WO-2019-52341), `CR-2021-00892`, `CR-2021-02234`, `CR-2021-07743` (all in WO-2021-38471), `CR-2022-03387` (in WO-2022-31102)

#### NLP Quality Gate
After extraction, measure unknown token rate on all text fields. Expected rate on this synthetic dataset: < 8% (all abbreviations are in the expansion dictionary). If rate exceeds 15%, review the dictionary coverage before proceeding.

---

## Step 7 — Knowledge Graph Construction

### Node Types to Create

| Node type | Count in demo | Key attributes |
|---|---|---|
| `component` | 5 | component_id, system, regulatory_constraint_flag |
| `activity` | 20 | activity_id, outage_id, emergent_flag, emergence_category, on_critical_path, float_hrs |
| `condition_report` | 15 | cr_id, created_date, cr_category, outage_cycle |
| `work_order` | 9 | wo_id, wo_type, planned_duration_hrs, actual_duration_hrs |
| `nuclear_entity` | 11 | entity text, entity class |
| `plant_element_id` | 7 | id string |
| `outage` | 3 | outage_id, type (training/holdout) |

### Edge Types to Create

| Edge | From | To | Properties |
|---|---|---|---|
| `has_cr` | component | condition_report | — |
| `has_wo` | component | work_order | — |
| `linked_to` | condition_report | work_order | — |
| `generated` | work_order | activity | — |
| `emergent_from` | activity (emergent) | activity (planned predecessor) | — |
| `part_of` | activity | outage | — |
| `mention` | activity / cr / wo | nuclear_entity | — |
| `refer` | activity / cr / wo | plant_element_id | — |
| `similar` | plant_element_id | plant_element_id | similarity score |

### Neo4j Cypher — Load Components (example)
```cypher
CREATE (:component {
  component_id: '1RHS-P-001A',
  description: 'RHR Pump 1A',
  system: 'Residual Heat Removal',
  regulatory_constraint_flag: true
})
```

### Neo4j Cypher — Candidate Causal Chain Query (anchor scenario)
```cypher
MATCH (comp:component {component_id: '1RHS-P-001A'})
-[:has_cr]->(cr:condition_report {cr_category: 'degradation'})
-[:linked_to]->(wo:work_order)
-[:generated]->(act:activity {emergent_flag: true})
RETURN comp, cr, wo, act
ORDER BY cr.created_date ASC
```
This query should return the chain for RF-20 and RF-21, which forms the causal evidence for the RF-22 prediction.

---

## Step 8 — Candidate Causal Chain Analysis

For each component, compute the following causal evidence score across training outages:

```
causal_score = (N_outages_with_degradation_cr / N_training_outages)
             × (N_outages_with_emergent_activity / N_outages_with_degradation_cr)
             × criticality_weight
```

Where `criticality_weight = 2.0` if emergent activities were on critical path in majority of training outages, `1.0` otherwise.

### Expected Scores for Demo Components

> ⚠️ **Important framing for any presentation of this table:** The formula below is a simplified scoring heuristic for demonstration purposes only. It is designed to be explainable in one sentence to a non-technical audience. A real implementation would use probabilistic models, larger multi-outage datasets, and cross-validated thresholds. Do not present this formula as a validated model.

| Component | Degradation CRs | Outages with emergent work | Causal score | Confidence tier |
|---|---|---|---|---|
| `1RHS-P-001A` | RF-20, RF-21 (2/2) | RF-20, RF-21 (2/2) | 1.0 × 1.0 × 2.0 = **2.0** | DATA-SUPPORTED |
| `1RHS-E-001A` | RF-20, RF-21 (2/2) | RF-20, RF-21 (2/2) | 1.0 × 1.0 × 2.0 = **2.0** → downgraded — see note | SME-INFORMED — see note |
| `1CSP-P-001B` | 0/2 (all CRs are observation) | 0/2 (planned WO only) | 0.0 (n_with_deg_cr = 0, formula guard) → flag on escalating CR trend | SME-INFORMED |
| `1CCW-P-002A` | 0/2 | 0/2 | **0.0** | Not flagged |
| `1RHS-V-001A` | 0/2 | 0/2 | **0.0** | Not flagged |

> **Note on 1RHS-E-001A confidence tier:** The heat exchanger scores DATA-SUPPORTED because emergent tube plugging occurred in both training outages on the critical path — the evidence pattern meets the same threshold as the pump. However, the degradation mechanism is progressive and cumulative (tube wall loss) rather than a single escalating failure mode, which makes the pattern somewhat more predictable and arguably more reliable as a pre-outage signal. If an SME reviewer disagrees with DATA-SUPPORTED and prefers borderline DATA-SUPPORTED / SME-INFORMED, that judgment should be recorded and the tier adjusted before the demo. Do not override SME judgment on tier assignment.

> **No-signal behavior — explicit rule:** If no components meet the DATA-SUPPORTED or SME-INFORMED thresholds after running the full pipeline, the system must explicitly report: *"No strong pre-outage risk signals detected in the available data."* This is a valid and honest outcome. Do not adjust thresholds or add components to the flagged list in order to produce a result. A no-signal output on a clean dataset is itself a meaningful finding that demonstrates the system is not over-flagging.

---

## Step 9 — Recommendation Generation

Generate one recommendation per flagged component. Each recommendation must conform to the format defined in the test case spec.

### Recommendation 1 — 1RHS-P-001A (anchor scenario)

```
Component:        1RHS-P-001A — RHR Pump 1A
Confidence tier:  DATA-SUPPORTED
Category:         Preventive

Finding:
Mechanical seal degradation and vibration above baseline observed across
RF-20 and RF-21 preparation periods. Seal replacement performed at RF-21
(WO-2021-38471) did not fully resolve vibration (CR-2022-01142, CR-2022-03387).
Pattern suggests compound degradation — possible impeller wear or bearing
deterioration contributing alongside seal wear.

Historical observed impact (not a predicted value): emergent work in RF-20
consumed 16 hours of critical path float; emergent work in RF-21 consumed
12 hours of critical path float. Combined historical observed impact: 28 hours
across two outages. This figure reflects what actually occurred in training
outages and is provided as context for planning contingency, not as a forecast
for RF-22.

Recommendation:
Expand planned WO-2022-31102 to include bearing and impeller assessment
in addition to seal inspection. Pre-stage impeller replacement parts before
outage start. Allocate contingency in critical path schedule for this work
scope.

Evidence chain:
  CR-2019-04412 → CR-2019-06891 → WO-2019-52341 → RF20-MECH-0089 (emergent, 16 hrs, critical path)
  CR-2021-00892 → CR-2021-02234 → CR-2021-07743 → WO-2021-38471 → RF21-MECH-0079 (emergent, 12 hrs, critical path)
  CR-2022-01142 → CR-2022-03387 → WO-2022-31102 (planned RF-22 — risk of further scope expansion)

Supporting outages:     RF-20, RF-21 (2 of 2 training outages)
Regulatory constraint:  YES — Tech Spec 3.5.2. Do not defer without licensing review.
Reject / feedback:      [ Accept ] [ Reject — reason: __________________ ]
```

### Recommendation 2 — 1RHS-E-001A

```
Component:        1RHS-E-001A — RHR Heat Exchanger 1A
Confidence tier:  SME-INFORMED
Category:         Preventive

Finding:
Progressive tube degradation across RF-20 and RF-21 with additional emergent
tube plugging in both outages. Thermal performance trending below design
basis each cycle. Planned full tube inspection (WO-2022-33891) is appropriate;
recommend pre-authorizing extended plugging scope.

Note on confidence tier: The raw causal score formula produces the same
value as 1RHS-P-001A, but the tier has been set to SME-INFORMED rather than
DATA-SUPPORTED. The degradation mechanism — progressive cumulative tube wall
loss — is different in character from an escalating single-component failure
mode. The work scope is more bounded and predictable, but no SME has reviewed
and confirmed the DATA-SUPPORTED classification for this component. Per the
build guide rule, tier assignments require SME sign-off before being elevated.
SME-INFORMED is the conservative and defensible position until that review
occurs. This distinction also provides a cleaner demo narrative: two
DATA-SUPPORTED components with identical scores would require explanation,
whereas DATA-SUPPORTED (pump) vs. SME-INFORMED (heat exchanger) naturally
illustrates the tiering system in action.

Evidence chain:
  CR-2020-01203 → WO-2020-10042 → RF20-MECH-0094 (emergent, 7 hrs, critical path)
  CR-2021-03301 → WO-2021-44892 → RF21-MECH-0083 (emergent, 6 hrs, critical path)
  CR-2022-04901 → WO-2022-33891 (planned RF-22)

Supporting outages:     RF-20, RF-21 (2 of 2 training outages)
Regulatory constraint:  YES — Tech Spec 3.5.2.
Reject / feedback:      [ Accept ] [ Reject — reason: __________________ ]
```

### Recommendation 3 — 1CSP-P-001B

```
Component:        1CSP-P-001B — Containment Spray Pump 1B
Confidence tier:  SME-INFORMED
Category:         Investigative

Finding:
No prior emergent work history on this component. However, motor current
has trended upward across three operating cycles (43A → 45A → 46A) and
bearing temperature has recently become slightly elevated. Planned bearing
replacement (WO-2022-35102) addresses the immediate finding. Pattern warrants
SME review to confirm planned scope is sufficient.

Evidence chain:
  CR-2020-02871 (observation, RF-20 prep)
  CR-2021-05512 → WO-2021-47201 (corrective, RF-21 — no emergent outcome)
  CR-2022-06234 → WO-2022-35102 (planned RF-22)

Supporting outages:     Trend across RF-20, RF-21, RF-22 prep — no emergent precedent
Regulatory constraint:  YES — Tech Spec 3.6.6.
Note:                   This is a watchlist item. Do not treat as primary planning risk
                        without SME confirmation.
Reject / feedback:      [ Accept ] [ Reject — reason: __________________ ]
```

---

## Step 10 — User-Facing Output Mockups

> ⚠️ **Before building or presenting any mockup, review the Demo Constraints section in the test case spec (§7).** Key rules: do not show raw graph visualizations, do not discuss model internals unless directly asked, do not present unvalidated causal claims, do not improvise on data provenance questions. Anyone running the demo without having read the spec should be briefed on these constraints explicitly before the meeting.

Build three mockup views for the PPT. These can be HTML wireframes, styled slide layouts, or screen recordings of the live system.

### View 1 — Pre-Outage Risk Register
A ranked table showing flagged components, confidence tier (color-coded), recommended action, and regulatory flag. Columns: Rank | Component | System | Confidence | Category | Regulatory | Action.

### View 2 — Evidence Drilldown (anchor scenario)
A simplified flow diagram — not a raw graph — showing the CR → WO → Activity chain for `1RHS-P-001A` across RF-20 and RF-21, with dates, durations, and the critical path impact callout. This is the single slide most likely to land with a plant manager audience.

### View 3 — Recommendation Card
A structured single-component view matching the recommendation format in Step 9. Shows finding, evidence chain, confidence tier badge, regulatory flag, and the accept/reject feedback widget.

---

## Step 11 — Pipeline Validation Checklist

Before the demo, confirm each of the following:

- [ ] All 5 components load as graph nodes with correct attributes
- [ ] All 15 CRs load and link to correct components
- [ ] All 9 WOs load and link to correct components and CRs
- [ ] All 20 activities load with correct emergent flags and critical path flags
- [ ] NLP unknown token rate < 8% on all text fields
- [ ] All plant IDs extracted correctly from text
- [ ] WO references in CR text extracted correctly
- [ ] CR references in WO text extracted correctly
- [ ] Causal chain query returns correct RF-20 and RF-21 chains for 1RHS-P-001A
- [ ] Confidence tiers assigned correctly for all 5 components
- [ ] Recommendations generated for 1RHS-P-001A, 1RHS-E-001A, 1CSP-P-001B only
- [ ] 1CCW-P-002A and 1RHS-V-001A correctly not flagged (true negatives)
- [ ] Regulatory constraint flag visible on all three recommendations
- [ ] Evidence trace links present on all recommendations (100% coverage)
- [ ] Accept/reject feedback widget functional
- [ ] Synthetic data disclosure label visible in UI or slide
