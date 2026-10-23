"""
Synthetic dataset for the Millbrook Nuclear Station Unit 1 pre-outage risk
prediction demo.

Plant: Millbrook Nuclear Station, Unit 1 (fictional generic PWR)
System: Residual Heat Removal (RHR)
Training outages: RF-20, RF-21
Holdout/test outage: RF-22 (pre-outage prediction state)

Source: demo_build_guide.md Steps 1-5 + RF-22 ground truth subsection.
All dates are illustrative and do not reflect real outage cycles.
"""

import re

# ---------------------------------------------------------------------------
# COMPONENTS
# 5 records — RHR system + related support systems
# ---------------------------------------------------------------------------
COMPONENTS = [
    {
        "component_id": "1RHS-P-001A",
        "description": "RHR Pump 1A",
        "system": "Residual Heat Removal",
        "system_code": "RHS",
        "plant_tag": "1RHS-P-001A",
        "component_type": "Pump",
        "regulatory_constraint_flag": True,
        "notes": "Tech Spec 3.5.2 — operability required",
    },
    {
        "component_id": "1RHS-E-001A",
        "description": "RHR Heat Exchanger 1A",
        "system": "Residual Heat Removal",
        "system_code": "RHS",
        "plant_tag": "1RHS-E-001A",
        "component_type": "Heat Exchanger",
        "regulatory_constraint_flag": True,
        "notes": "Tech Spec 3.5.2",
    },
    {
        "component_id": "1CSP-P-001B",
        "description": "Containment Spray Pump 1B",
        "system": "Containment Spray",
        "system_code": "CSP",
        "plant_tag": "1CSP-P-001B",
        "component_type": "Pump",
        "regulatory_constraint_flag": True,
        "notes": "Tech Spec 3.6.6",
    },
    {
        "component_id": "1CCW-P-002A",
        "description": "Component Cooling Water Pump 2A",
        "system": "Component Cooling Water",
        "system_code": "CCW",
        "plant_tag": "1CCW-P-002A",
        "component_type": "Pump",
        "regulatory_constraint_flag": False,
        "notes": "No active Tech Spec action level",
    },
    {
        "component_id": "1RHS-V-001A",
        "description": "RHR Suction Isolation Valve 1A",
        "system": "Residual Heat Removal",
        "system_code": "RHS",
        "plant_tag": "1RHS-V-001A",
        "component_type": "Valve",
        "regulatory_constraint_flag": True,
        "notes": "Tech Spec 3.5.2",
    },
]

# ---------------------------------------------------------------------------
# CONDITION_REPORTS
# 15 records spanning RF-20 prep → RF-21 prep → RF-22 prep
# ---------------------------------------------------------------------------
CONDITION_REPORTS = [
    {
        "cr_id": "CR-2019-04412",
        "component_id": "1RHS-P-001A",
        "created_date": "2019-08-14",
        "outage_cycle": "RF-20 prep",
        "description_raw": (
            "1RHS-P-001A noted vibration above baseline during quarterly surveillance. "
            "Vib reading 0.42 in/s vs 0.30 in/s baseline. Maint notified. Monitor."
        ),
        "cr_category": "observation",
        "linked_wo_id": None,
    },
    {
        "cr_id": "CR-2019-06891",
        "component_id": "1RHS-P-001A",
        "created_date": "2019-11-22",
        "outage_cycle": "RF-20 prep",
        "description_raw": (
            "Ops noted minor slt lkg at pump mech seal during walkdown. "
            "No active drip. Vlv pkgs ok. Recommend insp during next OT."
        ),
        "cr_category": "degradation",
        "linked_wo_id": "WO-2019-52341",
    },
    {
        "cr_id": "CR-2020-01203",
        "component_id": "1RHS-E-001A",
        "created_date": "2020-01-08",
        "outage_cycle": "RF-20 prep",
        "description_raw": (
            "HX 1A outlet temp trending 2.3 deg F below design basis. "
            "Possibly biofouling or partial tube plugging. Schedule tube inspection at RF-20."
        ),
        "cr_category": "observation",
        "linked_wo_id": "WO-2020-10042",
    },
    {
        "cr_id": "CR-2020-02871",
        "component_id": "1CSP-P-001B",
        "created_date": "2020-02-17",
        "outage_cycle": "RF-20 prep",
        "description_raw": (
            "1CSP-P-001B motor current trending high — 43A vs 39A baseline. "
            "Bearings within spec. No immediate action. Monitor."
        ),
        "cr_category": "observation",
        "linked_wo_id": None,
    },
    {
        "cr_id": "CR-2021-00892",
        "component_id": "1RHS-P-001A",
        "created_date": "2021-03-03",
        "outage_cycle": "RF-21 prep",
        "description_raw": (
            "Repeat slt lkg noted at mech seal 1RHS-P-001A. Lkg rate approx 1-2 drops/min. "
            "Elevated from prior CR-2019-06891. Recommend seal insp & possible repl at RF-21."
        ),
        "cr_category": "degradation",
        "linked_wo_id": "WO-2021-38471",
    },
    {
        "cr_id": "CR-2021-02234",
        "component_id": "1RHS-P-001A",
        "created_date": "2021-05-19",
        "outage_cycle": "RF-21 prep",
        "description_raw": (
            "Vib trending continued — now 0.51 in/s. Correl with seal lkg — "
            "possible seal face wear contributing to shaft movement. Priority elevated."
        ),
        "cr_category": "degradation",
        "linked_wo_id": "WO-2021-38471",
    },
    {
        "cr_id": "CR-2021-03301",
        "component_id": "1RHS-E-001A",
        "created_date": "2021-06-30",
        "outage_cycle": "RF-21 prep",
        "description_raw": (
            "Tube inspection from RF-20 showed 3 tubes with wall loss >20%. "
            "Plugged per procedure. Thermal performance marginally acceptable. Re-inspect RF-21."
        ),
        "cr_category": "degradation",
        "linked_wo_id": "WO-2021-44892",
    },
    {
        "cr_id": "CR-2021-05512",
        "component_id": "1CSP-P-001B",
        "created_date": "2021-08-11",
        "outage_cycle": "RF-21 prep",
        "description_raw": (
            "Motor current now 45A. Bearing temps within limits. "
            "Possible impeller wear. Schedule inspection at RF-21."
        ),
        "cr_category": "observation",
        "linked_wo_id": "WO-2021-47201",
    },
    {
        "cr_id": "CR-2021-07743",
        "component_id": "1RHS-P-001A",
        "created_date": "2021-10-04",
        "outage_cycle": "RF-21 prep",
        "description_raw": (
            "Pre-outage walkdown: mech seal lkg now 3-4 drops/min. "
            "WO-2021-38471 confirmed for RF-21. Parts on order — seal kit P/N RHS-SK-0042."
        ),
        "cr_category": "degradation",
        "linked_wo_id": "WO-2021-38471",
    },
    {
        "cr_id": "CR-2021-09981",
        "component_id": "1CCW-P-002A",
        "created_date": "2021-11-28",
        "outage_cycle": "RF-21 prep",
        "description_raw": (
            "Minor oil seepage noted at 1CCW-P-002A bearing housing. "
            "Within acceptable limits. No corrective action at this time."
        ),
        "cr_category": "observation",
        "linked_wo_id": None,
    },
    {
        "cr_id": "CR-2022-01142",
        "component_id": "1RHS-P-001A",
        "created_date": "2022-02-08",
        "outage_cycle": "RF-22 prep",
        "description_raw": (
            "Post RF-21 seal replacement: new seal installed. However vib reading still 0.44 in/s — "
            "not fully resolved. Monitor for recurrence."
        ),
        "cr_category": "observation",
        "linked_wo_id": "WO-2022-20341",
    },
    {
        "cr_id": "CR-2022-03387",
        "component_id": "1RHS-P-001A",
        "created_date": "2022-05-14",
        "outage_cycle": "RF-22 prep",
        "description_raw": (
            "Vib now 0.49 in/s. Seal replaced at RF-21 but vibration pattern suggests "
            "possible impeller wear or bearing degradation contributing. Recommend enhanced insp at RF-22."
        ),
        "cr_category": "degradation",
        "linked_wo_id": "WO-2022-31102",
    },
    {
        "cr_id": "CR-2022-04901",
        "component_id": "1RHS-E-001A",
        "created_date": "2022-07-02",
        "outage_cycle": "RF-22 prep",
        "description_raw": (
            "Thermal performance trending down again — 3.1 deg F below design basis. "
            "Tube plugging from RF-21 may be insufficient. Schedule full tube inspection RF-22."
        ),
        "cr_category": "degradation",
        "linked_wo_id": "WO-2022-33891",
    },
    {
        "cr_id": "CR-2022-06234",
        "component_id": "1CSP-P-001B",
        "created_date": "2022-08-19",
        "outage_cycle": "RF-22 prep",
        "description_raw": (
            "Motor current 46A — marginal. Bearing temp now slightly elevated "
            "(162F vs 155F baseline). Schedule bearing replacement RF-22."
        ),
        "cr_category": "degradation",
        "linked_wo_id": "WO-2022-35102",
    },
    {
        "cr_id": "CR-2022-08801",
        "component_id": "1RHS-V-001A",
        "created_date": "2022-10-11",
        "outage_cycle": "RF-22 prep",
        "description_raw": (
            "Valve position indication verified during surveillance. "
            "No anomalies noted. Routine observation — no action required."
        ),
        "cr_category": "observation",
        "linked_wo_id": None,
    },
]

# ---------------------------------------------------------------------------
# WORK_ORDERS
# 9 records — corrective and observation WOs across RF-20, RF-21, RF-22
# ---------------------------------------------------------------------------
WORK_ORDERS = [
    {
        "wo_id": "WO-2019-52341",
        "component_id": "1RHS-P-001A",
        "created_date": "2019-12-01",
        "outage_cycle": "RF-20",
        "description_raw": (
            "1RHS-P-001A MECH SEAL INSP — REMOVE PUMP CASING COVER, INSP SEAL FACES, "
            "MEASURE CLEARANCES, RPT FINDINGS. REF CR-2019-06891."
        ),
        "wo_type": "corrective",
        "planned_duration_hrs": 8.0,
        "actual_duration_hrs": 9.5,
        "completed_flag": True,
        "emergent_flag": False,
    },
    {
        "wo_id": "WO-2020-10042",
        "component_id": "1RHS-E-001A",
        "created_date": "2020-01-15",
        "outage_cycle": "RF-20",
        "description_raw": (
            "1RHS-E-001A TUBE INSP — EDDY CURRENT TEST ALL TUBES, "
            "PLUG DEGRADED TUBES PER PROC ENG-HX-042, RPT RESULTS."
        ),
        "wo_type": "corrective",
        "planned_duration_hrs": 24.0,
        "actual_duration_hrs": 31.0,
        "completed_flag": True,
        "emergent_flag": False,
    },
    {
        "wo_id": "WO-2021-38471",
        "component_id": "1RHS-P-001A",
        "created_date": "2021-03-10",
        "outage_cycle": "RF-21",
        "description_raw": (
            "1RHS-P-001A MECH SEAL INSP & REPL — REMOVE & INSP EXISTING SEAL, "
            "INSTALL NEW SEAL KIT P/N RHS-SK-0042, ALIGN PUMP, PERFORM POST-MAINT TEST. "
            "REF CR-2021-00892 CR-2021-02234 CR-2021-07743."
        ),
        "wo_type": "corrective",
        "planned_duration_hrs": 16.0,
        "actual_duration_hrs": 24.0,
        "completed_flag": True,
        "emergent_flag": False,
    },
    {
        "wo_id": "WO-2021-44892",
        "component_id": "1RHS-E-001A",
        "created_date": "2021-07-08",
        "outage_cycle": "RF-21",
        "description_raw": (
            "1RHS-E-001A TUBE INSP RF-21 — EDDY CURRENT TEST, ASSESS TUBES FLAGGED RF-20, "
            "PLUG AS REQUIRED, UPDATE TUBE MAP."
        ),
        "wo_type": "corrective",
        "planned_duration_hrs": 20.0,
        "actual_duration_hrs": 22.0,
        "completed_flag": True,
        "emergent_flag": False,
    },
    {
        "wo_id": "WO-2021-47201",
        "component_id": "1CSP-P-001B",
        "created_date": "2021-08-20",
        "outage_cycle": "RF-21",
        "description_raw": (
            "1CSP-P-001B MOTOR & IMPELLER INSP — PULL MOTOR, INSP BEARINGS, "
            "INSP IMPELLER FOR WEAR, RPT. REF CR-2021-05512."
        ),
        "wo_type": "corrective",
        "planned_duration_hrs": 12.0,
        "actual_duration_hrs": 10.5,
        "completed_flag": True,
        "emergent_flag": False,
    },
    {
        "wo_id": "WO-2022-20341",
        "component_id": "1RHS-P-001A",
        "created_date": "2022-02-15",
        "outage_cycle": "RF-22 prep",
        "description_raw": (
            "1RHS-P-001A POST-MAINT MONITORING — VIB TRENDING FOLLOWING RF-21 SEAL REPL. "
            "DOCUMENT READINGS MONTHLY. REF CR-2022-01142."
        ),
        "wo_type": "observation",
        "planned_duration_hrs": 2.0,
        "actual_duration_hrs": None,
        "completed_flag": False,
        "emergent_flag": False,
    },
    {
        "wo_id": "WO-2022-31102",
        "component_id": "1RHS-P-001A",
        "created_date": "2022-05-20",
        "outage_cycle": "RF-22",
        "description_raw": (
            "1RHS-P-001A ENHANCED INSP RF-22 — INSP MECH SEAL, BEARINGS, IMPELLER. "
            "ASSESS VIB ROOT CAUSE. REPL AS REQUIRED. REF CR-2022-03387."
        ),
        "wo_type": "corrective",
        "planned_duration_hrs": 20.0,
        "actual_duration_hrs": None,
        "completed_flag": False,
        "emergent_flag": False,
    },
    {
        "wo_id": "WO-2022-33891",
        "component_id": "1RHS-E-001A",
        "created_date": "2022-07-10",
        "outage_cycle": "RF-22",
        "description_raw": (
            "1RHS-E-001A FULL TUBE INSP RF-22 — EDDY CURRENT ALL TUBES, "
            "ASSESS PLUGGING STRATEGY, RPT."
        ),
        "wo_type": "corrective",
        "planned_duration_hrs": 24.0,
        "actual_duration_hrs": None,
        "completed_flag": False,
        "emergent_flag": False,
    },
    {
        "wo_id": "WO-2022-35102",
        "component_id": "1CSP-P-001B",
        "created_date": "2022-08-25",
        "outage_cycle": "RF-22",
        "description_raw": (
            "1CSP-P-001B BEARING REPL RF-22 — PULL MOTOR, REPLACE BEARINGS, "
            "INSPECT IMPELLER, REINSTALL, PMT. REF CR-2022-06234."
        ),
        "wo_type": "corrective",
        "planned_duration_hrs": 14.0,
        "actual_duration_hrs": None,
        "completed_flag": False,
        "emergent_flag": False,
    },
]

# ---------------------------------------------------------------------------
# ACTIVITIES
# 20 records — training outages RF-20 & RF-21 (completed) + RF-22 planned
# Columns: activity_id, outage_id, component_id, linked_wo_id, description_raw,
#          discipline, planned_start, planned_end, actual_start, actual_end,
#          planned_duration_hrs, actual_duration_hrs, emergent_flag,
#          emergence_category, on_critical_path, float_available_hrs
# ---------------------------------------------------------------------------
ACTIVITIES = [
    # --- RF-20 planned activities ---
    {
        "activity_id": "RF20-MECH-0042",
        "outage_id": "RF-20",
        "component_id": "1RHS-P-001A",
        "linked_wo_id": "WO-2019-52341",
        "description_raw": "1RHS-P-001A MECH SEAL INSP & RPT",
        "discipline": "MECH",
        "planned_start": "2020-03-02 08:00",
        "planned_end": "2020-03-02 16:00",
        "actual_start": "2020-03-02 08:30",
        "actual_end": "2020-03-03 09:30",
        "planned_duration_hrs": 8.0,
        "actual_duration_hrs": 9.5,  # actual > planned: overrun
        "emergent_flag": False,
        "emergence_category": None,
        "on_critical_path": False,
        "float_available_hrs": 18.0,
    },
    {
        "activity_id": "RF20-MECH-0071",
        "outage_id": "RF-20",
        "component_id": "1RHS-E-001A",
        "linked_wo_id": "WO-2020-10042",
        "description_raw": "1RHS-E-001A EDDY CURR TUBE INSP",
        "discipline": "MECH",
        "planned_start": "2020-03-04 07:00",
        "planned_end": "2020-03-05 07:00",
        "actual_start": "2020-03-04 07:00",
        "actual_end": "2020-03-05 14:00",
        "planned_duration_hrs": 24.0,
        "actual_duration_hrs": 31.0,
        "emergent_flag": False,
        "emergence_category": None,
        "on_critical_path": True,
        "float_available_hrs": 0.0,
    },
    # --- RF-20 emergent activities ---
    {
        "activity_id": "RF20-MECH-0089",
        "outage_id": "RF-20",
        "component_id": "1RHS-P-001A",
        "linked_wo_id": None,
        "description_raw": (
            "1RHS-P-001A SEAL FACE REPL — EMERGENT — "
            "INSP FOUND WEAR BEYOND ACCEPTABLE LIMITS REF WO-2019-52341"
        ),
        "discipline": "MECH",
        "planned_start": None,
        "planned_end": None,
        "actual_start": "2020-03-03 10:00",
        "actual_end": "2020-03-04 02:00",
        "planned_duration_hrs": None,
        "actual_duration_hrs": 16.0,
        "emergent_flag": True,
        "emergence_category": "DISCOVERY",
        "on_critical_path": True,
        "float_available_hrs": 0.0,
    },
    {
        "activity_id": "RF20-MECH-0094",
        "outage_id": "RF-20",
        "component_id": "1RHS-E-001A",
        "linked_wo_id": None,
        "description_raw": (
            "1RHS-E-001A ADDL TUBE PLUGGING — EMERGENT — "
            "3 TUBES ABOVE PLUGGING LIMIT FOUND DURING INSP"
        ),
        "discipline": "MECH",
        "planned_start": None,
        "planned_end": None,
        "actual_start": "2020-03-05 15:00",
        "actual_end": "2020-03-05 22:00",
        "planned_duration_hrs": None,
        "actual_duration_hrs": 7.0,
        "emergent_flag": True,
        "emergence_category": "DISCOVERY",
        "on_critical_path": True,
        "float_available_hrs": 0.0,
    },
    # --- RF-20 remaining planned ---
    {
        "activity_id": "RF20-ELEC-0021",
        "outage_id": "RF-20",
        "component_id": "1CSP-P-001B",
        "linked_wo_id": None,
        "description_raw": "1CSP-P-001B MTR MEGGR & ELEC INSP",
        "discipline": "ELEC",
        "planned_start": "2020-03-06 08:00",
        "planned_end": "2020-03-06 12:00",
        "actual_start": "2020-03-06 08:00",
        "actual_end": "2020-03-06 11:30",
        "planned_duration_hrs": 4.0,
        "actual_duration_hrs": 3.5,
        "emergent_flag": False,
        "emergence_category": None,
        "on_critical_path": False,
        "float_available_hrs": 24.0,
    },
    {
        "activity_id": "RF20-MECH-0103",
        "outage_id": "RF-20",
        "component_id": "1CCW-P-002A",
        "linked_wo_id": None,
        "description_raw": "1CCW-P-002A BEARING INSP & LUBE",
        "discipline": "MECH",
        "planned_start": "2020-03-07 08:00",
        "planned_end": "2020-03-07 12:00",
        "actual_start": "2020-03-07 08:00",
        "actual_end": "2020-03-07 11:00",
        "planned_duration_hrs": 4.0,
        "actual_duration_hrs": 3.0,
        "emergent_flag": False,
        "emergence_category": None,
        "on_critical_path": False,
        "float_available_hrs": 32.0,
    },
    {
        "activity_id": "RF20-OPS-0011",
        "outage_id": "RF-20",
        "component_id": "1RHS-V-001A",
        "linked_wo_id": None,
        "description_raw": "1RHS-V-001A VALVE STROKE TEST & INSP",
        "discipline": "OPS",
        "planned_start": "2020-03-08 06:00",
        "planned_end": "2020-03-08 10:00",
        "actual_start": "2020-03-08 06:00",
        "actual_end": "2020-03-08 09:30",
        "planned_duration_hrs": 4.0,
        "actual_duration_hrs": 3.5,
        "emergent_flag": False,
        "emergence_category": None,
        "on_critical_path": False,
        "float_available_hrs": 40.0,
    },
    # --- RF-21 planned activities ---
    {
        "activity_id": "RF21-MECH-0038",
        "outage_id": "RF-21",
        "component_id": "1RHS-P-001A",
        "linked_wo_id": "WO-2021-38471",
        "description_raw": "1RHS-P-001A MECH SEAL REPL & ALIGN",
        "discipline": "MECH",
        "planned_start": "2022-03-01 07:00",
        "planned_end": "2022-03-01 23:00",
        "actual_start": "2022-03-01 07:30",
        "actual_end": "2022-03-02 07:30",
        "planned_duration_hrs": 16.0,
        "actual_duration_hrs": 24.0,
        "emergent_flag": False,
        "emergence_category": None,
        "on_critical_path": True,
        "float_available_hrs": 0.0,
    },
    {
        "activity_id": "RF21-MECH-0052",
        "outage_id": "RF-21",
        "component_id": "1RHS-E-001A",
        "linked_wo_id": "WO-2021-44892",
        "description_raw": "1RHS-E-001A TUBE INSP & PLUGGING",
        "discipline": "MECH",
        "planned_start": "2022-03-03 07:00",
        "planned_end": "2022-03-04 03:00",
        "actual_start": "2022-03-03 07:00",
        "actual_end": "2022-03-04 05:00",
        "planned_duration_hrs": 20.0,
        "actual_duration_hrs": 22.0,
        "emergent_flag": False,
        "emergence_category": None,
        "on_critical_path": True,
        "float_available_hrs": 0.0,
    },
    {
        "activity_id": "RF21-MECH-0061",
        "outage_id": "RF-21",
        "component_id": "1CSP-P-001B",
        "linked_wo_id": "WO-2021-47201",
        "description_raw": "1CSP-P-001B MTR PULL BEARING INSP",
        "discipline": "MECH",
        "planned_start": "2022-03-05 08:00",
        "planned_end": "2022-03-05 20:00",
        "actual_start": "2022-03-05 08:00",
        "actual_end": "2022-03-05 18:30",
        "planned_duration_hrs": 12.0,
        "actual_duration_hrs": 10.5,
        "emergent_flag": False,
        "emergence_category": None,
        "on_critical_path": False,
        "float_available_hrs": 16.0,
    },
    # --- RF-21 emergent activities ---
    {
        "activity_id": "RF21-MECH-0079",
        "outage_id": "RF-21",
        "component_id": "1RHS-P-001A",
        "linked_wo_id": None,
        "description_raw": (
            "1RHS-P-001A IMPELLER INSP — EMERGENT — "
            "VIB ROOT CAUSE INVEST IDENTIFIED IMPELLER WEAR REF WO-2021-38471"
        ),
        "discipline": "MECH",
        "planned_start": None,
        "planned_end": None,
        "actual_start": "2022-03-02 08:00",
        "actual_end": "2022-03-02 20:00",
        "planned_duration_hrs": None,
        "actual_duration_hrs": 12.0,
        "emergent_flag": True,
        "emergence_category": "DISCOVERY",
        "on_critical_path": True,
        "float_available_hrs": 0.0,
    },
    {
        "activity_id": "RF21-MECH-0083",
        "outage_id": "RF-21",
        "component_id": "1RHS-E-001A",
        "linked_wo_id": None,
        "description_raw": (
            "1RHS-E-001A ADDL TUBE PLUGGING — EMERGENT — "
            "2 ADDL TUBES FOUND DEGRADED BEYOND LIMIT DURING INSP"
        ),
        "discipline": "MECH",
        "planned_start": None,
        "planned_end": None,
        "actual_start": "2022-03-04 06:00",
        "actual_end": "2022-03-04 12:00",
        "planned_duration_hrs": None,
        "actual_duration_hrs": 6.0,
        "emergent_flag": True,
        "emergence_category": "DISCOVERY",
        "on_critical_path": True,
        "float_available_hrs": 0.0,
    },
    # --- RF-21 remaining planned ---
    {
        "activity_id": "RF21-ELEC-0019",
        "outage_id": "RF-21",
        "component_id": "1CSP-P-001B",
        "linked_wo_id": None,
        "description_raw": "1CSP-P-001B MTR WINDING INSP & RPT",
        "discipline": "ELEC",
        "planned_start": "2022-03-06 08:00",
        "planned_end": "2022-03-06 14:00",
        "actual_start": "2022-03-06 08:00",
        "actual_end": "2022-03-06 13:00",
        "planned_duration_hrs": 6.0,
        "actual_duration_hrs": 5.0,
        "emergent_flag": False,
        "emergence_category": None,
        "on_critical_path": False,
        "float_available_hrs": 20.0,
    },
    {
        "activity_id": "RF21-MECH-0091",
        "outage_id": "RF-21",
        "component_id": "1CCW-P-002A",
        "linked_wo_id": None,
        "description_raw": "1CCW-P-002A BEARING INSP & LUBE",
        "discipline": "MECH",
        "planned_start": "2022-03-07 08:00",
        "planned_end": "2022-03-07 12:00",
        "actual_start": "2022-03-07 08:00",
        "actual_end": "2022-03-07 11:30",
        "planned_duration_hrs": 4.0,
        "actual_duration_hrs": 3.5,
        "emergent_flag": False,
        "emergence_category": None,
        "on_critical_path": False,
        "float_available_hrs": 28.0,
    },
    {
        "activity_id": "RF21-OPS-0008",
        "outage_id": "RF-21",
        "component_id": "1RHS-V-001A",
        "linked_wo_id": None,
        "description_raw": "1RHS-V-001A VALVE STROKE TEST",
        "discipline": "OPS",
        "planned_start": "2022-03-08 06:00",
        "planned_end": "2022-03-08 10:00",
        "actual_start": "2022-03-08 06:00",
        "actual_end": "2022-03-08 09:00",
        "planned_duration_hrs": 4.0,
        "actual_duration_hrs": 3.0,
        "emergent_flag": False,
        "emergence_category": None,
        "on_critical_path": False,
        "float_available_hrs": 36.0,
    },
    # --- RF-22 planned activities (not yet executed — prediction state) ---
    {
        "activity_id": "RF22-MECH-0041",
        "outage_id": "RF-22",
        "component_id": "1RHS-P-001A",
        "linked_wo_id": "WO-2022-31102",
        "description_raw": "1RHS-P-001A ENHANCED INSP — SEAL BEARING IMPELLER",
        "discipline": "MECH",
        "planned_start": "2024-03-01 07:00",
        "planned_end": "2024-03-02 03:00",
        "actual_start": None,
        "actual_end": None,
        "planned_duration_hrs": 20.0,
        "actual_duration_hrs": None,
        "emergent_flag": False,
        "emergence_category": None,
        "on_critical_path": True,
        "float_available_hrs": 0.0,
    },
    {
        "activity_id": "RF22-MECH-0055",
        "outage_id": "RF-22",
        "component_id": "1RHS-E-001A",
        "linked_wo_id": "WO-2022-33891",
        "description_raw": "1RHS-E-001A FULL TUBE INSP & PLUGGING",
        "discipline": "MECH",
        "planned_start": "2024-03-03 07:00",
        "planned_end": "2024-03-04 03:00",
        "actual_start": None,
        "actual_end": None,
        "planned_duration_hrs": 20.0,
        "actual_duration_hrs": None,
        "emergent_flag": False,
        "emergence_category": None,
        "on_critical_path": True,
        "float_available_hrs": 0.0,
    },
    {
        "activity_id": "RF22-MECH-0063",
        "outage_id": "RF-22",
        "component_id": "1CSP-P-001B",
        "linked_wo_id": "WO-2022-35102",
        "description_raw": "1CSP-P-001B BEARING REPL & PMT",
        "discipline": "MECH",
        "planned_start": "2024-03-05 07:00",
        "planned_end": "2024-03-05 19:00",
        "actual_start": None,
        "actual_end": None,
        "planned_duration_hrs": 12.0,
        "actual_duration_hrs": None,
        "emergent_flag": False,
        "emergence_category": None,
        "on_critical_path": False,
        "float_available_hrs": 14.0,
    },
    {
        "activity_id": "RF22-MECH-0072",
        "outage_id": "RF-22",
        "component_id": "1CCW-P-002A",
        "linked_wo_id": None,
        "description_raw": "1CCW-P-002A BEARING INSP & LUBE",
        "discipline": "MECH",
        "planned_start": "2024-03-06 08:00",
        "planned_end": "2024-03-06 12:00",
        "actual_start": None,
        "actual_end": None,
        "planned_duration_hrs": 4.0,
        "actual_duration_hrs": None,
        "emergent_flag": False,
        "emergence_category": None,
        "on_critical_path": False,
        "float_available_hrs": 30.0,
    },
    {
        "activity_id": "RF22-OPS-0009",
        "outage_id": "RF-22",
        "component_id": "1RHS-V-001A",
        "linked_wo_id": None,
        "description_raw": "1RHS-V-001A VALVE STROKE TEST & INSP",
        "discipline": "OPS",
        "planned_start": "2024-03-07 06:00",
        "planned_end": "2024-03-07 10:00",
        "actual_start": None,
        "actual_end": None,
        "planned_duration_hrs": 4.0,
        "actual_duration_hrs": None,
        "emergent_flag": False,
        "emergence_category": None,
        "on_critical_path": False,
        "float_available_hrs": 38.0,
    },
]

# ---------------------------------------------------------------------------
# RF22_GROUND_TRUTH
# Kept separate — not loaded into the pipeline until after prediction completes.
# These are the emergent activities that actually occurred in RF-22.
# ---------------------------------------------------------------------------
RF22_GROUND_TRUTH = [
    {
        "activity_id": "RF22-MECH-0089",
        "outage_id": "RF-22",
        "component_id": "1RHS-P-001A",
        "linked_wo_id": None,
        "description_raw": (
            "1RHS-P-001A BEARING & IMPELLER REPL — EMERGENT — "
            "ENHANCED INSP FOUND BEARING WEAR BEYOND LIMITS & IMPELLER EROSION. "
            "REPL BOTH. REF WO-2022-31102 CR-2022-03387"
        ),
        "discipline": "MECH",
        "planned_start": None,
        "planned_end": None,
        "actual_start": "2024-03-02 06:00",
        "actual_end": "2024-03-03 02:00",
        "planned_duration_hrs": None,
        "actual_duration_hrs": 20.0,
        "emergent_flag": True,
        "emergence_category": "DISCOVERY",
        "on_critical_path": True,
        "float_available_hrs": 0.0,
    },
    {
        "activity_id": "RF22-MECH-0094",
        "outage_id": "RF-22",
        "component_id": "1RHS-E-001A",
        "linked_wo_id": None,
        "description_raw": (
            "1RHS-E-001A ADDL TUBE PLUGGING — EMERGENT — "
            "4 TUBES FOUND ABOVE PLUGGING LIMIT DURING INSP — "
            "PLUGGED PER ENG-HX-042"
        ),
        "discipline": "MECH",
        "planned_start": None,
        "planned_end": None,
        "actual_start": "2024-03-04 06:00",
        "actual_end": "2024-03-04 14:00",
        "planned_duration_hrs": None,
        "actual_duration_hrs": 8.0,
        "emergent_flag": True,
        "emergence_category": "DISCOVERY",
        "on_critical_path": True,
        "float_available_hrs": 0.0,
    },
]

# ---------------------------------------------------------------------------
# SCHEDULE
# 20 records — one per activity, capturing critical path and float information.
# Columns: activity_id, outage_id, on_critical_path, float_available_hrs,
#          float_consumed_hrs, predecessor_activity_id, successor_activity_id
# ---------------------------------------------------------------------------
SCHEDULE = [
    # RF-20 schedule
    {
        "activity_id": "RF20-MECH-0042",
        "outage_id": "RF-20",
        "on_critical_path": False,
        "float_available_hrs": 18.0,
        "float_consumed_hrs": 0.0,
        "predecessor_activity_id": None,
        "successor_activity_id": "RF20-MECH-0089",
    },
    {
        "activity_id": "RF20-MECH-0071",
        "outage_id": "RF-20",
        "on_critical_path": True,
        "float_available_hrs": 0.0,
        "float_consumed_hrs": 7.0,
        "predecessor_activity_id": None,
        "successor_activity_id": "RF20-MECH-0094",
    },
    {
        "activity_id": "RF20-MECH-0089",
        "outage_id": "RF-20",
        "on_critical_path": True,
        "float_available_hrs": 0.0,
        "float_consumed_hrs": 16.0,
        "predecessor_activity_id": "RF20-MECH-0042",
        "successor_activity_id": None,
    },
    {
        "activity_id": "RF20-MECH-0094",
        "outage_id": "RF-20",
        "on_critical_path": True,
        "float_available_hrs": 0.0,
        "float_consumed_hrs": 7.0,
        "predecessor_activity_id": "RF20-MECH-0071",
        "successor_activity_id": None,
    },
    {
        "activity_id": "RF20-ELEC-0021",
        "outage_id": "RF-20",
        "on_critical_path": False,
        "float_available_hrs": 24.0,
        "float_consumed_hrs": 0.0,
        "predecessor_activity_id": None,
        "successor_activity_id": None,
    },
    {
        "activity_id": "RF20-MECH-0103",
        "outage_id": "RF-20",
        "on_critical_path": False,
        "float_available_hrs": 32.0,
        "float_consumed_hrs": 0.0,
        "predecessor_activity_id": None,
        "successor_activity_id": None,
    },
    {
        "activity_id": "RF20-OPS-0011",
        "outage_id": "RF-20",
        "on_critical_path": False,
        "float_available_hrs": 40.0,
        "float_consumed_hrs": 0.0,
        "predecessor_activity_id": None,
        "successor_activity_id": None,
    },
    # RF-21 schedule
    {
        "activity_id": "RF21-MECH-0038",
        "outage_id": "RF-21",
        "on_critical_path": True,
        "float_available_hrs": 0.0,
        "float_consumed_hrs": 8.0,
        "predecessor_activity_id": None,
        "successor_activity_id": "RF21-MECH-0079",
    },
    {
        "activity_id": "RF21-MECH-0052",
        "outage_id": "RF-21",
        "on_critical_path": True,
        "float_available_hrs": 0.0,
        "float_consumed_hrs": 2.0,
        "predecessor_activity_id": None,
        "successor_activity_id": "RF21-MECH-0083",
    },
    {
        "activity_id": "RF21-MECH-0061",
        "outage_id": "RF-21",
        "on_critical_path": False,
        "float_available_hrs": 16.0,
        "float_consumed_hrs": 0.0,
        "predecessor_activity_id": None,
        "successor_activity_id": None,
    },
    {
        "activity_id": "RF21-MECH-0079",
        "outage_id": "RF-21",
        "on_critical_path": True,
        "float_available_hrs": 0.0,
        "float_consumed_hrs": 12.0,
        "predecessor_activity_id": "RF21-MECH-0038",
        "successor_activity_id": None,
    },
    {
        "activity_id": "RF21-MECH-0083",
        "outage_id": "RF-21",
        "on_critical_path": True,
        "float_available_hrs": 0.0,
        "float_consumed_hrs": 6.0,
        "predecessor_activity_id": "RF21-MECH-0052",
        "successor_activity_id": None,
    },
    {
        "activity_id": "RF21-ELEC-0019",
        "outage_id": "RF-21",
        "on_critical_path": False,
        "float_available_hrs": 20.0,
        "float_consumed_hrs": 0.0,
        "predecessor_activity_id": None,
        "successor_activity_id": None,
    },
    {
        "activity_id": "RF21-MECH-0091",
        "outage_id": "RF-21",
        "on_critical_path": False,
        "float_available_hrs": 28.0,
        "float_consumed_hrs": 0.0,
        "predecessor_activity_id": None,
        "successor_activity_id": None,
    },
    {
        "activity_id": "RF21-OPS-0008",
        "outage_id": "RF-21",
        "on_critical_path": False,
        "float_available_hrs": 36.0,
        "float_consumed_hrs": 0.0,
        "predecessor_activity_id": None,
        "successor_activity_id": None,
    },
    # RF-22 schedule (planned only — no actuals yet)
    {
        "activity_id": "RF22-MECH-0041",
        "outage_id": "RF-22",
        "on_critical_path": True,
        "float_available_hrs": 0.0,
        "float_consumed_hrs": None,
        "predecessor_activity_id": None,
        "successor_activity_id": None,
    },
    {
        "activity_id": "RF22-MECH-0055",
        "outage_id": "RF-22",
        "on_critical_path": True,
        "float_available_hrs": 0.0,
        "float_consumed_hrs": None,
        "predecessor_activity_id": None,
        "successor_activity_id": None,
    },
    {
        "activity_id": "RF22-MECH-0063",
        "outage_id": "RF-22",
        "on_critical_path": False,
        "float_available_hrs": 14.0,
        "float_consumed_hrs": None,
        "predecessor_activity_id": None,
        "successor_activity_id": None,
    },
    {
        "activity_id": "RF22-MECH-0072",
        "outage_id": "RF-22",
        "on_critical_path": False,
        "float_available_hrs": 30.0,
        "float_consumed_hrs": None,
        "predecessor_activity_id": None,
        "successor_activity_id": None,
    },
    {
        "activity_id": "RF22-OPS-0009",
        "outage_id": "RF-22",
        "on_critical_path": False,
        "float_available_hrs": 38.0,
        "float_consumed_hrs": None,
        "predecessor_activity_id": None,
        "successor_activity_id": None,
    },
]

# ---------------------------------------------------------------------------
# NER_GROUND_TRUTH
# Entity text → entity class mapping used by the lightweight rule NER in Stage B.
# Keys may be single terms or short phrases; matching is case-insensitive.
# ---------------------------------------------------------------------------
NER_GROUND_TRUTH = {
    "pump": "Asset_Mechanical",
    "seal": "Mechanical_Component",
    "mechanical seal": "Mechanical_Component",
    "heat exchanger": "Asset_Mechanical",
    "valve": "Asset_Hydraulic",
    "bearing": "Mechanical_Component_Rotary",
    "impeller": "Mechanical_Component_Rotary",
    "vibration": "Surveillance_Operation",
    "leakage": "Failure_Type",
    "wear": "Degradation_Mechanism",
    "tube": "Nonmechanical_Component",
    "motor": "Asset_Electrical",
}

# Regex pattern for plant element IDs and part/procedure numbers:
#   Standard plant tag:  1RHS-P-001A  (<digit><2-4 letters>-<letter>-<3 digits><letter>)
#   Part/proc number:    RHS-SK-0042, ENG-HX-042  (<2-4 letters>-<2 letters>-<3-4 digits>)
PLANT_ID_PATTERN = re.compile(
    r"\d[A-Z]{2,4}-[A-Z]-\d{3}[A-Z]"   # standard plant tag
    r"|[A-Z]{2,4}-[A-Z]{2}-\d{3,4}"    # part/proc number
)

# Regex pattern for condition report and work order reference numbers.
# Format: CR-YYYY-NNNNN or WO-YYYY-NNNNN
CR_WO_PATTERN = re.compile(r"(?:CR|WO)-\d{4}-\d{5}")
