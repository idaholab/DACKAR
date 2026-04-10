"""
demo_scenarios.py
=================
Stub backends and pipeline runner for DACKAR show-and-tell demos.

Two pre-built scenarios are provided:

  SCENARIO_RCP_SEAL    — Reactor Coolant Pump Train-A mechanical seal leak.
                         Regulatory constraint (TS 3.4.6), critical-path impact,
                         crew mobilisation required → decision: ESCALATE.

  SCENARIO_SNUBBER_EXT — Snubber inspection scope expansion (opportunistic).
                         Non-regulatory, non-critical path, good float.
                         → decision: PROCEED.

Usage (plain Python or Jupyter)::

    from demos.demo_scenarios import run_pipeline, SCENARIO_RCP_SEAL, SCENARIO_SNUBBER_EXT

    result_rcp     = run_pipeline(SCENARIO_RCP_SEAL)
    result_snubber = run_pipeline(SCENARIO_SNUBBER_EXT)

``run_pipeline`` returns a dict with keys::

    intake, timeline, temporal, analogs, schedule, options, recommendation

Five of the seven stages execute their real production logic.
Stage B and Stage E use pre-built stub data (KG driver / LOGOS CPM not required).
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — add outage/ root so ``stages`` package is importable
# ---------------------------------------------------------------------------
_OUTAGE_DIR = Path(__file__).parent.parent
if str(_OUTAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_OUTAGE_DIR))

from stages.stage_a_intake import ActivityIntakeProcessor
from stages.stage_b_kg_timeline import KGTimelineBuilder
from stages.stage_c_temporal_chain import TemporalChainScorer
from stages.stage_d_analogs import HistoricalAnalogRetriever, _DictActivityCase
from stages.stage_f_options import InsertionOptionGenerator
from stages.stage_g_recommendation import RecommendationSynthesizer

JsonDict = Dict[str, Any]

# ============================================================================
# Stub backends
# ============================================================================

class _StubKGDriver:
    """Minimal KG driver that returns pre-loaded timeline event records.

    Each query method inspects the Cypher string to decide which fixture
    dataset to return. The record format matches what KGTimelineBuilder
    expects: ``record[alias]`` returns a dict of node properties.
    """

    def __init__(
        self,
        cr_events: List[JsonDict],
        pm_events: List[JsonDict],
        cm_events: Optional[List[JsonDict]] = None,
        component_meta: Optional[JsonDict] = None,
    ) -> None:
        self._cr = cr_events
        self._pm = pm_events
        self._cm = cm_events or []
        self._meta = component_meta or {}

    def query(self, cypher: str, parameters=None, db=None) -> List[Any]:
        cypher_l = cypher.lower()

        # Component metadata lookup (SELECT c.name, s.id … OPTIONAL MATCH)
        if "optional match" in cypher_l and "mbse_entity" in cypher_l:
            return [self._meta]

        # PM query  (work_type = $pm_code)
        if "pm_code" in cypher_l:
            return [{"wo": e} for e in self._pm]

        # CM query  (work_type = $cm_code)
        if "cm_code" in cypher_l:
            return [{"wo": e} for e in self._cm]

        # Condition-report query
        if "condition_report" in cypher_l:
            return [{"cr": e} for e in self._cr]

        # General work-order query (not PM or CM)
        if "work_order" in cypher_l:
            return []

        # Abnormal events / inspections — not used in these demos
        return []


class _StubRetrievalIndex:
    """Minimal retrieval index that returns pre-loaded ActivityCase-like objects.

    ``search()`` returns synthetic IDs; ``get()`` hydrates them from the
    pre-loaded list.  No actual embedding or BM25 logic is performed.
    """

    def __init__(self, activities: List[Any]) -> None:
        self._activities = activities
        self._ids = [f"analog_{i}" for i in range(len(activities))]

    def search(self, query_activity, top_k: int = 100) -> List[str]:
        return self._ids

    def get(self, activity_id: str) -> Optional[Any]:
        try:
            idx = int(activity_id.split("_")[1])
            return self._activities[idx]
        except (IndexError, ValueError):
            return None


# ============================================================================
# Scenario 1 — RCP Seal Leak (ESCALATE path)
# ============================================================================

_RCP_ACTIVITY: JsonDict = {
    "activity_id": "ACT-RCP-001",
    "outage_id": "RF-24",
    "plant_id": "PLANT-ALPHA",
    "source_system": "maximo",
    "raw_description": (
        "RCP Train-A mechanical seal leaking approximately 2 GPM. "
        "TS 3.4.6 entry required. "
        "Hold point placed on WO-44821 pending operability determination. "
        "Seal replacement estimated 48 hours. "
        "Activity identified during pre-restart walkdown."
    ),
    "known_component_id": "RCP-1A",
    "known_system_id": "RCS",
    "work_order_id": "WO-44821",
    "detection_timestamp": "2026-04-10T06:00:00",
    "planned_duration_hours": 48.0,
    "safety_related": True,
    "discipline": "mechanical",
}

_RCP_KG_DRIVER = _StubKGDriver(
    cr_events=[
        {
            "id": "CR-2022-3301",
            "description": "RCP 1A mechanical seal minor drip observed — outage walkdown",
            "initiated_date": "2022-04-18T08:00:00",
            "source_system": "maximo",
            "outage_id": "RF-21",
        },
        {
            "id": "CR-2023-4421",
            "description": "RCP 1A seal leakoff flow elevated above normal band (0.8 GPM)",
            "initiated_date": "2023-09-12T10:30:00",
            "source_system": "maximo",
            "outage_id": "RF-22",
        },
        {
            "id": "CR-2025-1178",
            "description": "RCP 1A primary seal degradation confirmed — leakoff 1.5 GPM",
            "initiated_date": "2025-03-20T14:15:00",
            "source_system": "maximo",
            "outage_id": "RF-23",
        },
    ],
    pm_events=[
        {
            "id": "PM-2024-RCP1A",
            "description": "RCP 1A quarterly seal package inspection PM",
            "work_type": "PM",
            "completion_date": "2024-03-01T12:00:00",
            "source_system": "maximo",
            "outage_id": "RF-23",
        },
    ],
    component_meta={
        "name": "RCP Train A",
        "system_id": "RCS",
        "system_name": "Reactor Coolant System",
        "asset_id": "RX-UNIT1",
    },
)

_RCP_ANALOG_ACTIVITIES = [
    _DictActivityCase({
        "activity_id": "H-RCP-001",
        "raw_description": "RCP 1B mechanical seal replacement unit 1 RF-19",
        "actual_duration_hours": 40.0,
        "planned_duration_hours": 48.0,
        "component_id": "RCP-1B",
        "component_family": "pump_mechanical_seal",
        "task_family": "seal_replacement",
        "discipline": "mechanical",
        "outage_id": "RF-19",
        "plant_id": "PLANT-ALPHA",
    }),
    _DictActivityCase({
        "activity_id": "H-RCP-002",
        "raw_description": "RCP primary seal replacement — unexpected leakage RF-20",
        "actual_duration_hours": 44.0,
        "planned_duration_hours": 48.0,
        "component_id": "RCP-1A",
        "component_family": "pump_mechanical_seal",
        "task_family": "seal_replacement",
        "discipline": "mechanical",
        "outage_id": "RF-20",
        "plant_id": "PLANT-ALPHA",
    }),
    _DictActivityCase({
        "activity_id": "H-RCP-003",
        "raw_description": "RCP 1A seal replacement with crew mobilisation RF-21",
        "actual_duration_hours": 52.0,
        "planned_duration_hours": 48.0,
        "component_id": "RCP-1A",
        "component_family": "pump_mechanical_seal",
        "task_family": "seal_replacement",
        "discipline": "mechanical",
        "outage_id": "RF-21",
        "plant_id": "PLANT-ALPHA",
    }),
    _DictActivityCase({
        "activity_id": "H-RCP-004",
        "raw_description": "RCP mechanical seal replacement and post-work inspection",
        "actual_duration_hours": 48.0,
        "planned_duration_hours": 48.0,
        "component_id": "RCP-1A",
        "component_family": "pump_mechanical_seal",
        "task_family": "seal_replacement",
        "discipline": "mechanical",
        "outage_id": "RF-22",
        "plant_id": "PLANT-ALPHA",
    }),
    _DictActivityCase({
        "activity_id": "H-RCP-005",
        "raw_description": "RCP 1A seal replacement — TS 3.4.6 entry, 2.1 GPM leak",
        "actual_duration_hours": 56.0,
        "planned_duration_hours": 48.0,
        "component_id": "RCP-1A",
        "component_family": "pump_mechanical_seal",
        "task_family": "seal_replacement",
        "discipline": "mechanical",
        "outage_id": "RF-23",
        "plant_id": "PLANT-ALPHA",
    }),
]

# Stage E pre-built artifact (requires LOGOS CPM — bypassed in demo)
# baseline_cp_hours = 480h  (20-day outage at 24h/day)
# cp_drag = 48h (exceeds 24h threshold → escalate option generated)
# crew_unavailable conflict → insert_now infeasible
# remaining_float_after < 0 → contingency_buffer infeasible
# criticality = critical, no non-critical displaced tasks → no parallel option
# Result: only escalate is feasible + regulatory-cleared → ESCALATE decision
_RCP_SCHEDULE_IMPACT: JsonDict = {
    "activity_id": "ACT-RCP-001",
    "run_id": "run-rcp-001",
    "generated_at": "2026-04-10T06:05:00",
    "schedule_version_id": "RF-24-WRK-003",
    "insertion_point": {
        "task_id": "T-RCS-FLUSH",
        "task_name": "RCS System Flush and Sample",
        "phase": "mode_transition",
        "after_task_id": "T-DRAIN-COMPLETE",
    },
    "duration_estimate": {
        "p50_hours": 44.0,
        "p80_hours": 56.0,
        "p90_hours": 64.0,
        "confidence_tier": "data_supported",
    },
    "float_analysis": {
        "available_float_before": 0.0,
        "float_consumed_hours": 44.0,
        "remaining_float_after": -44.0,
        "criticality_label": "critical",
        "is_critical_path_impact": True,
    },
    "cp_impact": {
        "cp_drag_hours": 48.0,
        "cp_sensitivity_score": 0.82,
        "baseline_cp_hours": 480.0,
        "expected_delay_hours": 44.0,
        "p50_delay_hours": 44.0,
        "p80_delay_hours": 56.0,
    },
    "displaced_tasks": [
        {
            "task_id": "T-MODE3-ENTRY",
            "task_name": "Mode 3 Entry Sequence",
            "criticality_label": "critical",
            "has_regulatory_constraint": True,
            "float_hours": 0.0,
            "discipline": "operations",
        },
        {
            "task_id": "T-HPSI-FLOW",
            "task_name": "HPSI Flow Verification",
            "criticality_label": "critical",
            "has_regulatory_constraint": False,
            "float_hours": 0.0,
            "discipline": "i_and_c",
        },
    ],
    "resource_conflicts": [
        {
            "conflict_type": "crew_unavailable",
            "skill_required": "nuclear_pump_seal_technician",
            "description": (
                "Specialized RCP seal installation crew not on-site. "
                "Mobilization required (estimated 8–12 h lead time)."
            ),
        },
    ],
    "confidence": 0.78,
}

SCENARIO_RCP_SEAL: JsonDict = {
    "label": "Scenario 1 — RCP Train-A Seal Leak",
    "activity": _RCP_ACTIVITY,
    "kg_driver": _RCP_KG_DRIVER,
    "analog_activities": _RCP_ANALOG_ACTIVITIES,
    "schedule_impact": _RCP_SCHEDULE_IMPACT,
}


# ============================================================================
# Scenario 2 — Snubber Inspection Scope Expansion (PROCEED path)
# ============================================================================

_SNUBBER_ACTIVITY: JsonDict = {
    "activity_id": "ACT-SNB-002",
    "outage_id": "RF-24",
    "plant_id": "PLANT-ALPHA",
    "source_system": "maximo",
    "raw_description": (
        "While performing scheduled snubber inspection on line RCS-SN-204, "
        "technician identified two additional snubbers requiring functional testing. "
        "Opportunistic scope addition to existing WO-44610. "
        "Additional work estimated 8 hours."
    ),
    "known_component_id": "SNB-RCS-204",
    "known_system_id": "RCS",
    "work_order_id": "WO-44610",
    "detection_timestamp": "2026-04-10T09:00:00",
    "planned_duration_hours": 8.0,
    "safety_related": False,
    "discipline": "mechanical",
}

_SNUBBER_KG_DRIVER = _StubKGDriver(
    cr_events=[
        {
            "id": "CR-2023-7701",
            "description": "Snubber RCS-SN-204 functional test — no failure, slight drag noted",
            "initiated_date": "2023-10-05T09:00:00",
            "source_system": "maximo",
            "outage_id": "RF-22",
        },
    ],
    pm_events=[
        {
            "id": "PM-2024-SNB204",
            "description": "Snubber inspection PM RCS line 204",
            "work_type": "PM",
            "completion_date": "2024-04-20T10:00:00",
            "source_system": "maximo",
            "outage_id": "RF-23",
        },
    ],
    component_meta={
        "name": "Snubber RCS-SN-204",
        "system_id": "RCS",
        "system_name": "Reactor Coolant System",
        "asset_id": "RX-UNIT1",
    },
)

_SNUBBER_ANALOG_ACTIVITIES = [
    _DictActivityCase({
        "activity_id": "H-SNB-001",
        "raw_description": "Snubber functional test 2 units RCS line expansion scope",
        "actual_duration_hours": 6.0,
        "planned_duration_hours": 8.0,
        "component_id": "SNB-RCS-204",
        "component_family": "snubber",
        "task_family": "functional_test",
        "discipline": "mechanical",
        "outage_id": "RF-19",
        "plant_id": "PLANT-ALPHA",
    }),
    _DictActivityCase({
        "activity_id": "H-SNB-002",
        "raw_description": "Snubber inspection and functional test while in area RF-20",
        "actual_duration_hours": 7.0,
        "planned_duration_hours": 8.0,
        "component_id": "SNB-RCS-205",
        "component_family": "snubber",
        "task_family": "functional_test",
        "discipline": "mechanical",
        "outage_id": "RF-20",
        "plant_id": "PLANT-ALPHA",
    }),
    _DictActivityCase({
        "activity_id": "H-SNB-003",
        "raw_description": "Snubber inspection scope expansion, 2 additional units RF-21",
        "actual_duration_hours": 8.0,
        "planned_duration_hours": 8.0,
        "component_id": "SNB-RCS-204",
        "component_family": "snubber",
        "task_family": "functional_test",
        "discipline": "mechanical",
        "outage_id": "RF-21",
        "plant_id": "PLANT-ALPHA",
    }),
    _DictActivityCase({
        "activity_id": "H-SNB-004",
        "raw_description": "Opportunistic snubber test while performing RCS maintenance RF-22",
        "actual_duration_hours": 9.0,
        "planned_duration_hours": 8.0,
        "component_id": "SNB-RCS-206",
        "component_family": "snubber",
        "task_family": "functional_test",
        "discipline": "mechanical",
        "outage_id": "RF-22",
        "plant_id": "PLANT-ALPHA",
    }),
    _DictActivityCase({
        "activity_id": "H-SNB-005",
        "raw_description": "Additional snubbers opportunistic test RF-23",
        "actual_duration_hours": 10.0,
        "planned_duration_hours": 8.0,
        "component_id": "SNB-RCS-204",
        "component_family": "snubber",
        "task_family": "functional_test",
        "discipline": "mechanical",
        "outage_id": "RF-23",
        "plant_id": "PLANT-ALPHA",
    }),
]

# Stage E pre-built: non-critical path, good float, no conflicts → PROCEED
_SNUBBER_SCHEDULE_IMPACT: JsonDict = {
    "activity_id": "ACT-SNB-002",
    "run_id": "run-snb-002",
    "generated_at": "2026-04-10T09:05:00",
    "schedule_version_id": "RF-24-WRK-003",
    "insertion_point": {
        "task_id": "T-SNB-SCHED",
        "task_name": "Scheduled Snubber Inspection RCS-204",
        "phase": "maintenance_window",
        "after_task_id": "T-DRAIN-COMPLETE",
    },
    "duration_estimate": {
        "p50_hours": 8.0,
        "p80_hours": 10.0,
        "p90_hours": 12.0,
        "confidence_tier": "sme_informed",
    },
    "float_analysis": {
        "available_float_before": 36.0,
        "float_consumed_hours": 8.0,
        "remaining_float_after": 28.0,
        "criticality_label": "non_critical",
        "is_critical_path_impact": False,
    },
    "cp_impact": {
        "cp_drag_hours": 0.0,
        "cp_sensitivity_score": 0.08,
        "baseline_cp_hours": 480.0,
        "expected_delay_hours": 0.0,
        "p50_delay_hours": 0.0,
        "p80_delay_hours": 0.0,
    },
    "displaced_tasks": [],
    "resource_conflicts": [],
    "confidence": 0.85,
}

SCENARIO_SNUBBER_EXT: JsonDict = {
    "label": "Scenario 2 — Snubber Scope Expansion",
    "activity": _SNUBBER_ACTIVITY,
    "kg_driver": _SNUBBER_KG_DRIVER,
    "analog_activities": _SNUBBER_ANALOG_ACTIVITIES,
    "schedule_impact": _SNUBBER_SCHEDULE_IMPACT,
}


# ============================================================================
# Pipeline runner
# ============================================================================

def run_pipeline(scenario: JsonDict, run_id: Optional[str] = None) -> JsonDict:
    """Run the full A → B → C → D → E(stub) → F → G pipeline for one scenario.

    Stages A, C, F, G execute their real production logic.
    Stage B runs the real KGTimelineBuilder with a stub KG driver.
    Stage D runs the real HistoricalAnalogRetriever with a stub index.
    Stage E uses the pre-built schedule impact artifact (LOGOS not required).

    Returns a dict with one key per stage::

        {
            "intake":         Stage A artifact,
            "timeline":       Stage B artifact,
            "temporal":       Stage C artifact,
            "analogs":        Stage D artifact,
            "schedule":       Stage E artifact (pre-built),
            "options":        Stage F artifact,
            "recommendation": Stage G artifact,
        }
    """
    run_id = run_id or f"DEMO::{uuid.uuid4().hex[:8]}"
    activity = scenario["activity"]
    run_ctx: JsonDict = {
        "run_id": run_id,
        "started_at": activity.get("detection_timestamp", ""),
    }

    # ── Stage A: Activity Intake ──────────────────────────────────────────────
    stage_a = ActivityIntakeProcessor()
    intake = stage_a.process(activity, run_ctx)

    # ── Stage B: KG Timeline (real builder, stub KG driver) ───────────────────
    stage_b = KGTimelineBuilder(kg_driver=scenario["kg_driver"])
    timeline = stage_b.build(activity, intake, run_ctx)

    # ── Stage C: Temporal Chain Scoring ──────────────────────────────────────
    stage_c = TemporalChainScorer()
    temporal = stage_c.score(activity, timeline, run_ctx)

    # ── Stage D: Historical Analog Retrieval (real retriever, stub index) ─────
    stage_d = HistoricalAnalogRetriever(
        retrieval_index=_StubRetrievalIndex(scenario["analog_activities"])
    )
    analogs = stage_d.retrieve(activity, intake, run_ctx)

    # ── Stage E: Schedule Impact (pre-built stub — LOGOS CPM not required) ────
    schedule = scenario["schedule_impact"].copy()
    schedule["run_id"] = run_id

    # ── Stage F: Insertion Options ────────────────────────────────────────────
    stage_f = InsertionOptionGenerator()
    options = stage_f.generate(activity, intake, temporal, schedule, analogs, run_ctx)

    # ── Stage G: Recommendation Synthesis ────────────────────────────────────
    stage_g = RecommendationSynthesizer()
    recommendation = stage_g.synthesize(
        activity, intake, timeline, temporal, analogs, schedule, options, run_ctx
    )

    return {
        "scenario_label": scenario["label"],
        "run_id": run_id,
        "intake": intake,
        "timeline": timeline,
        "temporal": temporal,
        "analogs": analogs,
        "schedule": schedule,
        "options": options,
        "recommendation": recommendation,
    }
