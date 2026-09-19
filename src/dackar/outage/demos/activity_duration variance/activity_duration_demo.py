"""
Activity Completion Time Variance — Show & Tell Demo
=====================================================
Audience: Outage managers / planning leads

What this script demonstrates
------------------------------
1. Build a synthetic historical database of completed outage activities.
2. Estimate the duration distribution for 6 planned activities in an upcoming
   refuelling outage, including confidence tier, uncertainty type, and
   recommended planner action.
3. Show which historical analogues drove each estimate (top-3 deep-dive).
4. Assemble a 10-activity schedule network for the upcoming outage.
5. Run a 2,000-iteration Monte Carlo simulation to propagate individual
   activity uncertainty through to outage-level finish risk.
6. Rank activities by schedule impact: criticality index, expected drag,
   CP sensitivity.

Run from the outage/ directory:
    python "demos/activity_duration variance/activity_duration_demo.py"

No external dependencies beyond the outage_uncertainty package itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — works regardless of whether the package is pip-installed.
# File lives two levels below outage/ (demos/activity_duration variance/),
# so we need parents[2]: file → activity_duration variance/ → demos/ → outage/
# ---------------------------------------------------------------------------
_OUTAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_OUTAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTAGE_ROOT))

from outage_uncertainty.api.facade import build_duration_uncertainty_service
from outage_uncertainty.domain.schedule import ScheduleActivity
from outage_uncertainty.domain.duration import DurationDistribution
from outage_uncertainty.schedule_risk.schedule_graph import ScheduleNetwork
from outage_uncertainty.schedule_risk.scenario_runner import ScenarioRunner


# ===========================================================================
# Formatting helpers
# ===========================================================================

_W = 72   # line width

def _banner(title: str) -> None:
    print()
    print("=" * _W)
    print(f"  {title}")
    print("=" * _W)

def _section(title: str) -> None:
    print()
    print(f"── {title} " + "─" * max(0, _W - len(title) - 4))

def _row(label: str, value, width: int = 28) -> str:
    return f"  {label:<{width}} {value}"

def _bar(value: float, max_val: float, width: int = 30) -> str:
    """ASCII progress bar scaled to max_val."""
    if max_val <= 0:
        return ""
    filled = int(round(value / max_val * width))
    return "█" * filled + "░" * (width - filled)

def _ascii_dist(samples: list[float], n_bins: int = 12, width: int = 40) -> str:
    """Compact ASCII histogram for a list of samples."""
    if not samples:
        return "  (no samples)"
    lo, hi = min(samples), max(samples)
    span = hi - lo
    if span <= 0:
        return f"  All values = {lo:.1f} h"
    bin_w = span / n_bins
    counts = [0] * n_bins
    for v in samples:
        idx = min(int((v - lo) / bin_w), n_bins - 1)
        counts[idx] += 1
    max_c = max(counts)
    lines = []
    for i, c in enumerate(counts):
        label = f"{lo + i * bin_w:6.1f}h"
        bar   = "█" * int(round(c / max_c * width)) if max_c > 0 else ""
        lines.append(f"  {label} │{bar}")
    return "\n".join(lines)


# ===========================================================================
# Section 1 — Synthetic historical database
# ===========================================================================

def build_historical_rows() -> list[dict]:
    """
    35 completed activities from three previous outages at Plant Millbrook
    (fictional PWR, two-unit site).

    Each row is a dict matching PandasActivityRepository's expected column names.
    Actual durations include realistic variance: RP holds, unexpected findings,
    and vendor delays inflate some jobs above planned.
    """
    base = [
        # ── Outage OH-001  (Refuelling, 2022, Unit 1) ─────────────────────
        {"activity_id": "H001", "outage_id": "OH-001", "plant_id": "Millbrook-U1",
         "raw_description": "Main coolant pump MCP-1A mechanical seal replacement",
         "discipline": "mechanical", "task_family": "replacement", "component_family": "pump",
         "planned_duration_hours": 20.0, "actual_duration_hours": 23.5,
         "has_rp_hold": True, "outage_phase": "forced outage"},

        {"activity_id": "H002", "outage_id": "OH-001", "plant_id": "Millbrook-U1",
         "raw_description": "RHR pump 1A discharge valve packing replacement",
         "discipline": "mechanical", "task_family": "replacement", "component_family": "valve",
         "planned_duration_hours": 6.0, "actual_duration_hours": 7.5,
         "outage_phase": "planned outage"},

        {"activity_id": "H003", "outage_id": "OH-001", "plant_id": "Millbrook-U1",
         "raw_description": "MSIV valve stem inspection unit 1 A train",
         "discipline": "mechanical", "task_family": "inspection", "component_family": "valve",
         "planned_duration_hours": 10.0, "actual_duration_hours": 11.0,
         "outage_phase": "planned outage"},

        {"activity_id": "H004", "outage_id": "OH-001", "plant_id": "Millbrook-U1",
         "raw_description": "Charging pump 1B complete overhaul impeller and bearings",
         "discipline": "mechanical", "task_family": "refurbishment", "component_family": "pump",
         "planned_duration_hours": 40.0, "actual_duration_hours": 47.0,
         "outage_phase": "planned outage"},

        {"activity_id": "H005", "outage_id": "OH-001", "plant_id": "Millbrook-U1",
         "raw_description": "Level transmitter LT-101 channel calibration loop check",
         "discipline": "I&C", "task_family": "calibration", "component_family": "instrument",
         "planned_duration_hours": 3.0, "actual_duration_hours": 3.0,
         "outage_phase": "planned outage"},

        {"activity_id": "H006", "outage_id": "OH-001", "plant_id": "Millbrook-U1",
         "raw_description": "Flow transmitter FT-203 calibration set point verification",
         "discipline": "I&C", "task_family": "calibration", "component_family": "instrument",
         "planned_duration_hours": 3.0, "actual_duration_hours": 4.0,
         "outage_phase": "planned outage"},

        {"activity_id": "H007", "outage_id": "OH-001", "plant_id": "Millbrook-U1",
         "raw_description": "ECCS injection valve MOV actuator replacement",
         "discipline": "mechanical", "task_family": "replacement", "component_family": "valve",
         "planned_duration_hours": 24.0, "actual_duration_hours": 38.0,
         "has_rp_hold": True, "has_clearance": True, "outage_phase": "planned outage"},

        {"activity_id": "H008", "outage_id": "OH-001", "plant_id": "Millbrook-U1",
         "raw_description": "Main transformer A bushing inspection and oil sampling",
         "discipline": "electrical", "task_family": "inspection", "component_family": "transformer",
         "planned_duration_hours": 6.0, "actual_duration_hours": 6.0,
         "outage_phase": "planned outage"},

        {"activity_id": "H009", "outage_id": "OH-001", "plant_id": "Millbrook-U1",
         "raw_description": "Bus 2A medium voltage breaker preventive maintenance",
         "discipline": "electrical", "task_family": "maintenance", "component_family": "breaker",
         "planned_duration_hours": 4.0, "actual_duration_hours": 4.5,
         "outage_phase": "planned outage"},

        {"activity_id": "H010", "outage_id": "OH-001", "plant_id": "Millbrook-U1",
         "raw_description": "Condenser tube bundle cleaning and inspection",
         "discipline": "mechanical", "task_family": "cleaning", "component_family": "condenser",
         "planned_duration_hours": 12.0, "actual_duration_hours": 14.0,
         "outage_phase": "planned outage"},

        # ── Outage OH-002  (Refuelling, 2023, Unit 1) ─────────────────────
        {"activity_id": "H011", "outage_id": "OH-002", "plant_id": "Millbrook-U1",
         "raw_description": "Main coolant pump MCP-1B mechanical seal replacement",
         "discipline": "mechanical", "task_family": "replacement", "component_family": "pump",
         "planned_duration_hours": 20.0, "actual_duration_hours": 26.0,
         "has_rp_hold": True, "outage_phase": "planned outage"},

        {"activity_id": "H012", "outage_id": "OH-002", "plant_id": "Millbrook-U1",
         "raw_description": "RHR pump 1B suction valve packing replacement",
         "discipline": "mechanical", "task_family": "replacement", "component_family": "valve",
         "planned_duration_hours": 6.0, "actual_duration_hours": 6.5,
         "outage_phase": "planned outage"},

        {"activity_id": "H013", "outage_id": "OH-002", "plant_id": "Millbrook-U1",
         "raw_description": "MSIV valve B train full inspection disassembly",
         "discipline": "mechanical", "task_family": "inspection", "component_family": "valve",
         "planned_duration_hours": 10.0, "actual_duration_hours": 13.5,
         "outage_phase": "planned outage"},

        {"activity_id": "H014", "outage_id": "OH-002", "plant_id": "Millbrook-U1",
         "raw_description": "Charging pump 1A overhaul casing and mechanical seal",
         "discipline": "mechanical", "task_family": "refurbishment", "component_family": "pump",
         "planned_duration_hours": 40.0, "actual_duration_hours": 53.0,
         "outage_phase": "planned outage"},

        {"activity_id": "H015", "outage_id": "OH-002", "plant_id": "Millbrook-U1",
         "raw_description": "Pressure transmitter PT-305 calibration set point",
         "discipline": "I&C", "task_family": "calibration", "component_family": "instrument",
         "planned_duration_hours": 3.0, "actual_duration_hours": 3.0,
         "outage_phase": "planned outage"},

        {"activity_id": "H016", "outage_id": "OH-002", "plant_id": "Millbrook-U1",
         "raw_description": "Temperature element TE-401 replacement loop verification",
         "discipline": "I&C", "task_family": "replacement", "component_family": "instrument",
         "planned_duration_hours": 4.0, "actual_duration_hours": 5.5,
         "outage_phase": "planned outage"},

        {"activity_id": "H017", "outage_id": "OH-002", "plant_id": "Millbrook-U1",
         "raw_description": "RCS isolation valve MOV overhaul with actuator rebuild",
         "discipline": "mechanical", "task_family": "refurbishment", "component_family": "valve",
         "planned_duration_hours": 36.0, "actual_duration_hours": 48.0,
         "has_rp_hold": True, "has_clearance": True, "outage_phase": "planned outage"},

        {"activity_id": "H018", "outage_id": "OH-002", "plant_id": "Millbrook-U1",
         "raw_description": "4kV bus 2B switchgear breaker inspection and lubrication",
         "discipline": "electrical", "task_family": "inspection", "component_family": "breaker",
         "planned_duration_hours": 5.0, "actual_duration_hours": 5.0,
         "outage_phase": "planned outage"},

        {"activity_id": "H019", "outage_id": "OH-002", "plant_id": "Millbrook-U1",
         "raw_description": "Service water pump 2A complete overhaul mechanical seal bearings",
         "discipline": "mechanical", "task_family": "refurbishment", "component_family": "pump",
         "planned_duration_hours": 36.0, "actual_duration_hours": 43.0,
         "outage_phase": "planned outage"},

        {"activity_id": "H020", "outage_id": "OH-002", "plant_id": "Millbrook-U1",
         "raw_description": "Lube oil cooler cleaning and inspection tube bundle",
         "discipline": "mechanical", "task_family": "cleaning", "component_family": "heat_exchanger",
         "planned_duration_hours": 10.0, "actual_duration_hours": 11.5,
         "outage_phase": "planned outage"},

        # ── Outage OH-003  (Forced, 2024, Unit 1) ─────────────────────────
        {"activity_id": "H021", "outage_id": "OH-003", "plant_id": "Millbrook-U1",
         "raw_description": "MCP-2A mechanical seal emergency replacement RCP loop 2",
         "discipline": "mechanical", "task_family": "replacement", "component_family": "pump",
         "planned_duration_hours": 20.0, "actual_duration_hours": 31.0,
         "is_emergent": True, "has_rp_hold": True, "outage_phase": "forced outage"},

        {"activity_id": "H022", "outage_id": "OH-003", "plant_id": "Millbrook-U1",
         "raw_description": "Safety relief valve SRV-2 replacement and functional test",
         "discipline": "mechanical", "task_family": "replacement", "component_family": "valve",
         "planned_duration_hours": 14.0, "actual_duration_hours": 17.0,
         "outage_phase": "forced outage"},

        {"activity_id": "H023", "outage_id": "OH-003", "plant_id": "Millbrook-U1",
         "raw_description": "Containment building structural visual inspection scaffolding",
         "discipline": "civil", "task_family": "inspection", "component_family": "structure",
         "planned_duration_hours": 60.0, "actual_duration_hours": 74.0,
         "requires_scaffold": True, "outage_phase": "forced outage"},

        {"activity_id": "H024", "outage_id": "OH-003", "plant_id": "Millbrook-U1",
         "raw_description": "EDG emergency diesel generator functional test surveillance",
         "discipline": "mechanical", "task_family": "testing", "component_family": "generator",
         "planned_duration_hours": 8.0, "actual_duration_hours": 8.0,
         "outage_phase": "forced outage"},

        {"activity_id": "H025", "outage_id": "OH-003", "plant_id": "Millbrook-U1",
         "raw_description": "Service water pump 2B overhaul impeller and mechanical seal",
         "discipline": "mechanical", "task_family": "refurbishment", "component_family": "pump",
         "planned_duration_hours": 36.0, "actual_duration_hours": 41.5,
         "outage_phase": "forced outage"},

        {"activity_id": "H026", "outage_id": "OH-003", "plant_id": "Millbrook-U1",
         "raw_description": "Radiation area monitor calibration setpoint verification",
         "discipline": "I&C", "task_family": "calibration", "component_family": "sensor",
         "planned_duration_hours": 4.0, "actual_duration_hours": 4.5,
         "outage_phase": "forced outage"},

        {"activity_id": "H027", "outage_id": "OH-003", "plant_id": "Millbrook-U1",
         "raw_description": "HP turbine first stage blade visual inspection",
         "discipline": "mechanical", "task_family": "inspection", "component_family": "turbine",
         "planned_duration_hours": 20.0, "actual_duration_hours": 22.0,
         "outage_phase": "forced outage"},

        # ── Outage OH-004  (Unit 2, 2023) — cross-unit data ───────────────
        {"activity_id": "H028", "outage_id": "OH-004", "plant_id": "Millbrook-U2",
         "raw_description": "MCP-2A mechanical seal replacement unit 2 loop A",
         "discipline": "mechanical", "task_family": "replacement", "component_family": "pump",
         "planned_duration_hours": 20.0, "actual_duration_hours": 24.5,
         "has_rp_hold": True, "outage_phase": "planned outage"},

        {"activity_id": "H029", "outage_id": "OH-004", "plant_id": "Millbrook-U2",
         "raw_description": "Charging pump 1C overhaul vendor supported inspection",
         "discipline": "mechanical", "task_family": "refurbishment", "component_family": "pump",
         "planned_duration_hours": 40.0, "actual_duration_hours": 57.0,
         "is_vendor_supported": True, "outage_phase": "planned outage"},

        {"activity_id": "H030", "outage_id": "OH-004", "plant_id": "Millbrook-U2",
         "raw_description": "Differential pressure transmitter DP-501 calibration",
         "discipline": "I&C", "task_family": "calibration", "component_family": "instrument",
         "planned_duration_hours": 3.0, "actual_duration_hours": 3.5,
         "outage_phase": "planned outage"},

        {"activity_id": "H031", "outage_id": "OH-004", "plant_id": "Millbrook-U2",
         "raw_description": "MSIV valve unit 2 full stroke test inspection disassembly",
         "discipline": "mechanical", "task_family": "inspection", "component_family": "valve",
         "planned_duration_hours": 10.0, "actual_duration_hours": 10.5,
         "outage_phase": "planned outage"},

        {"activity_id": "H032", "outage_id": "OH-004", "plant_id": "Millbrook-U2",
         "raw_description": "Main coolant pump MCP-2B seal inspection unit 2 loop B",
         "discipline": "mechanical", "task_family": "inspection", "component_family": "pump",
         "planned_duration_hours": 8.0, "actual_duration_hours": 9.5,
         "has_rp_hold": True, "outage_phase": "planned outage"},

        {"activity_id": "H033", "outage_id": "OH-004", "plant_id": "Millbrook-U2",
         "raw_description": "RHR pump 2A bearing replacement mechanical seal check",
         "discipline": "mechanical", "task_family": "replacement", "component_family": "pump",
         "planned_duration_hours": 12.0, "actual_duration_hours": 13.0,
         "outage_phase": "planned outage"},

        {"activity_id": "H034", "outage_id": "OH-004", "plant_id": "Millbrook-U2",
         "raw_description": "4kV bus 3A medium voltage switchgear maintenance lubrication",
         "discipline": "electrical", "task_family": "maintenance", "component_family": "switchgear",
         "planned_duration_hours": 8.0, "actual_duration_hours": 9.0,
         "outage_phase": "planned outage"},

        {"activity_id": "H035", "outage_id": "OH-004", "plant_id": "Millbrook-U2",
         "raw_description": "EDG battery charger replacement unit 2 division 2",
         "discipline": "electrical", "task_family": "replacement", "component_family": "transformer",
         "planned_duration_hours": 4.0, "actual_duration_hours": 5.5,
         "outage_phase": "planned outage"},
    ]
    return base


# ===========================================================================
# Section 2 — Planned activities for the upcoming outage (query set)
# ===========================================================================

PLANNED_ACTIVITIES = [
    # Q1: Well-supported — pump seal replacement, lots of analogues
    {"activity_id": "Q-MCP-2B", "outage_id": "OH-005", "plant_id": "Millbrook-U1",
     "raw_description": "Main coolant pump MCP-2B mechanical seal replacement",
     "planned_duration_hours": 20.0,
     "has_rp_hold": True, "outage_phase": "planned outage",
     "predecessor_ids": ["Q-INIT"], "successor_ids": ["Q-REPL"]},

    # Q2: Well-supported — transmitter calibration, many exact analogues
    {"activity_id": "Q-PT-410", "outage_id": "OH-005", "plant_id": "Millbrook-U1",
     "raw_description": "Pressure transmitter PT-410 calibration set point check",
     "planned_duration_hours": 3.0,
     "outage_phase": "planned outage",
     "predecessor_ids": ["Q-INIT"], "successor_ids": ["Q-LOOPS"]},

    # Q3: Moderately supported — MSIV valve inspection, some analogues
    {"activity_id": "Q-MSIV-C", "outage_id": "OH-005", "plant_id": "Millbrook-U1",
     "raw_description": "MSIV valve C train full inspection disassembly and seat check",
     "planned_duration_hours": 10.0,
     "outage_phase": "planned outage",
     "predecessor_ids": ["Q-INIT"], "successor_ids": ["Q-REASS"]},

    # Q4: Some support — pump overhaul, longer duration with known disruption risk
    {"activity_id": "Q-SWP-2B", "outage_id": "OH-005", "plant_id": "Millbrook-U1",
     "raw_description": "Service water pump 2B overhaul impeller seal and bearing replacement",
     "planned_duration_hours": 36.0,
     "outage_phase": "planned outage",
     "predecessor_ids": ["Q-INIT"], "successor_ids": ["Q-LOOPS"]},

    # Q5: Scaffold + civil — moderate support, higher epistemic contribution
    {"activity_id": "Q-CONT-INSP", "outage_id": "OH-005", "plant_id": "Millbrook-U1",
     "raw_description": "Containment building structural inspection with scaffolding",
     "planned_duration_hours": 60.0,
     "requires_scaffold": True, "outage_phase": "planned outage",
     "predecessor_ids": ["Q-INIT"], "successor_ids": ["Q-LOOPS"]},

    # Q6: Poor support — novel I&C upgrade, nothing close in history (epistemic)
    {"activity_id": "Q-DCS-UPGRDE", "outage_id": "OH-005", "plant_id": "Millbrook-U1",
     "raw_description": "Digital reactor protection system upgrade first of kind controller replacement",
     "planned_duration_hours": 48.0,
     "is_vendor_supported": True, "outage_phase": "planned outage",
     "predecessor_ids": ["Q-INIT"], "successor_ids": ["Q-LOOPS"]},
]

# Activities used in the schedule network but not needing estimation
# (deterministic or simple enough for baseline)
SCHEDULE_BACKBONE = {
    "Q-INIT":  {"name": "Outage initiation & cooldown",    "baseline_h": 8.0},
    "Q-REPL":  {"name": "RCP seal replacement execution",  "baseline_h": 0.0},  # filled by estimate
    "Q-REASS": {"name": "MSIV reassembly & leak check",    "baseline_h": 8.0},
    "Q-LOOPS": {"name": "Final I&C loop checks",           "baseline_h": 8.0},
    "Q-START": {"name": "Reactor startup sequence",        "baseline_h": 4.0},
}


# ===========================================================================
# Printing helpers for estimates
# ===========================================================================

def _tier_badge(tier: str) -> str:
    return {"high": "● HIGH  ", "medium": "◑ MEDIUM", "low": "○ LOW   "}.get(tier, tier)

def _utype_badge(utype: str) -> str:
    return {
        "epistemic": "⚠  EPISTEMIC",
        "aleatory":  "〜 ALEATORY ",
        "mixed":     "⊕ MIXED    ",
        "unknown":   "? UNKNOWN  ",
    }.get(utype, utype)

def print_estimate_table(estimates: list) -> None:
    hdr = f"  {'Activity ID':<18} {'Plan h':>6} {'P50 h':>6} {'P80 h':>6} {'P90 h':>6}  {'Conf':>8}  {'Unc type':<14} {'#Cases':>6}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for act_row, est in estimates:
        d   = est.estimated_distribution
        p50 = f"{d.p50:.1f}" if d.p50 is not None else "  ─"
        p80 = f"{d.p80:.1f}" if d.p80 is not None else "  ─"
        p90 = f"{d.p90:.1f}" if d.p90 is not None else "  ─"
        mix_p90 = d.parameters.get("mixture_p90")
        p90_str = f"{mix_p90:.1f}*" if mix_p90 else p90
        planned = act_row.get("planned_duration_hours", 0)
        tier    = _tier_badge(est.confidence_tier)
        utype   = _utype_badge(est.uncertainty_type)
        print(f"  {act_row['activity_id']:<18} {planned:>6.1f} {p50:>6} {p80:>6} {p90_str:>6}  {tier}  {utype}  {est.support_count:>6}")
    print()
    print("  * P90 with mixture model (disruption-inclusive tail)")

def print_warnings(act_id: str, estimate) -> None:
    if estimate.warnings:
        for w in estimate.warnings:
            print(f"  ⚠  {act_id}: {w}")

def print_recommended_actions(estimates: list) -> None:
    _section("Recommended planner actions")
    for act_row, est in estimates:
        if est.recommended_action:
            badge = _utype_badge(est.uncertainty_type)
            print(f"\n  [{act_row['activity_id']}]  {badge}")
            print(f"  {est.recommended_action}")

def print_top_analogues(act_id: str, estimate, historical_rows: list, top_n: int = 3) -> None:
    _section(f"Top historical analogues — {act_id}")
    hist_by_id = {r["activity_id"]: r for r in historical_rows}
    matches = sorted(
        [m for m in estimate.matched_cases if m.candidate_duration_hours is not None],
        key=lambda m: m.total_score,
        reverse=True,
    )[:top_n]

    if not matches:
        print("  (no analogue data available)")
        return

    for rank, m in enumerate(matches, 1):
        h = hist_by_id.get(m.candidate_activity_id, {})
        bar = _bar(m.total_score, 1.0, 20)
        print(f"\n  #{rank}  {m.candidate_activity_id}  ──  {h.get('raw_description','')[:55]}")
        print(f"       Similarity │{bar}│ {m.total_score:.2f}  "
              f"(lex={m.lexical_score:.2f}, ctx={m.context_score:.2f}, dep={m.dependency_score:.2f})")
        print(f"       Actual: {m.candidate_duration_hours:.1f} h   "
              f"Relevance weight: {m.relevance_weight:.3f}   "
              f"Plant: {h.get('plant_id','?')}")


# ===========================================================================
# Section 3 — Schedule network + Monte Carlo
# ===========================================================================

def build_schedule_network(
    estimates: list,
) -> tuple[ScheduleNetwork, float]:
    """
    10-activity schedule network for Outage OH-005.

    Network topology
    ─────────────────
    Q-INIT ──┬──► Q-MCP-2B ──► Q-REPL ──────────┐
             ├──► Q-MSIV-C ──► Q-REASS ──────────┤
             ├──► Q-SWP-2B ──────────────────────┤
             ├──► Q-CONT-INSP ──────────────────►─┤
             └──► Q-PT-410 ──────────────────────►─┴──► Q-LOOPS ──► Q-START

    Baseline critical path:
       Q-INIT(8) → Q-MCP-2B(20) → Q-REPL(20) → Q-LOOPS(8) → Q-START(4) = 60 h
    """
    est_by_id = {act_row["activity_id"]: est for act_row, est in estimates}

    def _dist(act_id: str, fallback_h: float) -> DurationDistribution | None:
        if act_id in est_by_id:
            return est_by_id[act_id].estimated_distribution
        return None

    # Build activities
    activities: list[ScheduleActivity] = [
        ScheduleActivity(
            activity_id="Q-INIT",
            name="Outage initiation & cooldown",
            successors=["Q-MCP-2B", "Q-MSIV-C", "Q-SWP-2B", "Q-CONT-INSP", "Q-PT-410"],
            baseline_duration_hours=8.0,
        ),
        ScheduleActivity(
            activity_id="Q-MCP-2B",
            name="MCP-2B mechanical seal replacement",
            predecessors=["Q-INIT"], successors=["Q-REPL"],
            baseline_duration_hours=20.0,
            duration_distribution=_dist("Q-MCP-2B", 20.0),
        ),
        ScheduleActivity(
            activity_id="Q-REPL",
            name="RCP seal reassembly & pressure test",
            predecessors=["Q-MCP-2B"], successors=["Q-LOOPS"],
            baseline_duration_hours=8.0,
        ),
        ScheduleActivity(
            activity_id="Q-MSIV-C",
            name="MSIV valve C inspection",
            predecessors=["Q-INIT"], successors=["Q-REASS"],
            baseline_duration_hours=10.0,
            duration_distribution=_dist("Q-MSIV-C", 10.0),
        ),
        ScheduleActivity(
            activity_id="Q-REASS",
            name="MSIV reassembly & leak check",
            predecessors=["Q-MSIV-C"], successors=["Q-LOOPS"],
            baseline_duration_hours=8.0,
        ),
        ScheduleActivity(
            activity_id="Q-SWP-2B",
            name="Service water pump 2B overhaul",
            predecessors=["Q-INIT"], successors=["Q-LOOPS"],
            baseline_duration_hours=36.0,
            duration_distribution=_dist("Q-SWP-2B", 36.0),
        ),
        ScheduleActivity(
            activity_id="Q-CONT-INSP",
            name="Containment structural inspection",
            predecessors=["Q-INIT"], successors=["Q-LOOPS"],
            baseline_duration_hours=60.0,
            duration_distribution=_dist("Q-CONT-INSP", 60.0),
        ),
        ScheduleActivity(
            activity_id="Q-PT-410",
            name="PT-410 calibration",
            predecessors=["Q-INIT"], successors=["Q-LOOPS"],
            baseline_duration_hours=3.0,
            duration_distribution=_dist("Q-PT-410", 3.0),
        ),
        ScheduleActivity(
            activity_id="Q-LOOPS",
            name="Final I&C loop checks",
            predecessors=["Q-REPL", "Q-REASS", "Q-SWP-2B", "Q-CONT-INSP", "Q-PT-410"],
            successors=["Q-START"],
            baseline_duration_hours=8.0,
        ),
        ScheduleActivity(
            activity_id="Q-START",
            name="Reactor startup sequence",
            predecessors=["Q-LOOPS"],
            baseline_duration_hours=4.0,
        ),
    ]

    # Baseline CP: Q-INIT → Q-CONT-INSP → Q-LOOPS → Q-START = 8+60+8+4 = 80 h
    # (containment inspection is the longest parallel path at 60 h baseline)
    baseline_cp_time = 8.0 + 60.0 + 8.0 + 4.0   # = 80 h

    return ScheduleNetwork(activities), baseline_cp_time


def print_risk_rankings(results: dict, baseline_cp_time: float) -> None:
    """Print a planner-readable schedule risk summary."""
    _section("Schedule risk summary")
    robustness = results["robustness"]
    p80 = results["p80_finish"]
    p90 = results["p90_finish"]
    std = results["schedule_std_dev"]
    delay = results["expected_delay"]

    print(f"\n  Baseline CP duration    :  {baseline_cp_time:.0f} h")
    print(f"  P80 projected finish    :  {p80:.1f} h   ({p80 - baseline_cp_time:+.1f} h vs baseline)")
    print(f"  P90 projected finish    :  {p90:.1f} h   ({p90 - baseline_cp_time:+.1f} h vs baseline)")
    print(f"  Schedule std deviation  :  ± {std:.1f} h")
    print(f"  Expected delay          :  {delay:.1f} h")
    print(f"  Robustness              :  {robustness * 100:.0f}%  (P[finish ≤ baseline])")

    _section("Activity risk ranking — top drivers of schedule overrun")
    print(f"\n  {'Activity':<16} {'CP Sensitivity':>16} {'Criticality %':>14} {'Exp. Drag h':>12}  {'CP Sensitivity bar'}")
    print("  " + "─" * 80)

    ci  = results["criticality_index"]
    sen = results["cp_sensitivity"]
    drg = results["expected_drag"]

    # Rank by CP sensitivity (most actionable metric for planners)
    ranked = sorted(sen.items(), key=lambda kv: kv[1], reverse=True)
    max_abs_sen = max((abs(v) for v in sen.values()), default=1.0) or 1.0

    for act_id, sensitivity in ranked:
        crit_pct = ci.get(act_id, 0.0) * 100
        drag     = drg.get(act_id, 0.0)
        bar      = _bar(max(sensitivity, 0.0), max_abs_sen, 25)
        print(f"  {act_id:<16} {sensitivity:>16.3f} {crit_pct:>13.0f}% {drag:>11.1f}h  │{bar}│")

    _section("Mitigation priority — activities warranting pre-outage action")
    top3 = ranked[:3]
    priority_labels = ["HIGH PRIORITY", "MEDIUM PRIORITY", "WATCH"]
    for (act_id, sen_val), label in zip(top3, priority_labels):
        print(f"\n  [{label}]  {act_id}")
        drag = drg.get(act_id, 0.0)
        crit = ci.get(act_id, 0.0) * 100
        print(f"    On critical path in {crit:.0f}% of simulations")
        print(f"    When critical, adds on average {drag:.1f} h to outage duration")
        print(f"    CP sensitivity: {sen_val:.3f}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    _banner("OUTAGE DURATION UNCERTAINTY DEMO — Plant Millbrook, Refuelling OH-005")
    print(f"\n  Tool:  outage_uncertainty duration estimation + schedule risk")
    print(f"  Data:  {35} historical completed activities (3 prior outages)")
    print(f"  Query: {len(PLANNED_ACTIVITIES)} planned activities for upcoming outage")

    # ── Build service ──────────────────────────────────────────────────────
    _section("Initialising duration uncertainty service")
    service = build_duration_uncertainty_service()
    print("  Service ready (abbreviation resolver, taxonomy mapper, similarity engine)")

    historical_rows = build_historical_rows()
    print(f"  Historical database: {len(historical_rows)} activities loaded")

    # ── Estimate durations ─────────────────────────────────────────────────
    _banner("SECTION 1 — Duration Estimates for Planned Activities")

    estimates: list[tuple[dict, object]] = []
    for act_row in PLANNED_ACTIVITIES:
        est = service.estimate_activity(
            query_row=act_row,
            historical_rows=historical_rows,
        )
        estimates.append((act_row, est))
        print(f"  ✓  {act_row['activity_id']:<20}  "
              f"tier={est.confidence_tier:<6}  "
              f"type={est.uncertainty_type:<10}  "
              f"n={est.support_count}")

    _section("Duration estimate summary table")
    print()
    print_estimate_table(estimates)

    # Warnings
    _section("Estimate warnings")
    any_warning = False
    for act_row, est in estimates:
        for w in est.warnings:
            print(f"\n  {act_row['activity_id']}: {w}")
            any_warning = True
    if not any_warning:
        print("  (no warnings)")

    # Recommended actions
    print_recommended_actions(estimates)

    # ── Analogue deep-dive ─────────────────────────────────────────────────
    _banner("SECTION 2 — Analogue Deep-Dive")

    # Show deep-dive for the pump replacement (well-supported) and the
    # novel I&C upgrade (epistemic)
    for target_id in ("Q-MCP-2B", "Q-DCS-UPGRDE"):
        for act_row, est in estimates:
            if act_row["activity_id"] == target_id:
                print_top_analogues(target_id, est, historical_rows, top_n=3)

                # ASCII distribution of routine duration samples
                d = est.estimated_distribution
                if d.samples:
                    _section(f"Routine duration distribution — {target_id}")
                    print(_ascii_dist(d.samples))
                    print(f"\n  Routine pool: P50={d.p50:.1f}h  P80={d.p80:.1f}h  P90={d.p90:.1f}h")
                    if d.mixture_weight > 0:
                        mix_p90 = d.parameters.get("mixture_p90")
                        print(f"  Disruption rate: {d.mixture_weight*100:.0f}%  "
                              f"Mixture P90={mix_p90:.1f}h" if mix_p90 else
                              f"  Disruption rate: {d.mixture_weight*100:.0f}%")
                break

    # ── Schedule network ───────────────────────────────────────────────────
    _banner("SECTION 3 — Schedule Network & Monte Carlo Risk Analysis")

    _section("Network topology (OH-005, 10 activities)")
    print("""
  Q-INIT ──┬──► Q-MCP-2B ──► Q-REPL ─────────────────────────┐
           ├──► Q-MSIV-C ──► Q-REASS ──────────────────────────┤
           ├──► Q-SWP-2B ─────────────────────────────────────►─┤
           ├──► Q-CONT-INSP ───────────────────────────────────►─┴──► Q-LOOPS ──► Q-START
           └──► Q-PT-410  ────────────────────────────────────►─┘

  Activities with uncertainty:  Q-MCP-2B  Q-MSIV-C  Q-SWP-2B  Q-CONT-INSP  Q-PT-410
  Deterministic activities:     Q-INIT  Q-REPL  Q-REASS  Q-LOOPS  Q-START
    """)

    _section("Running Monte Carlo simulation (2,000 iterations)")
    network, baseline_cp_time = build_schedule_network(estimates)
    runner = ScenarioRunner()
    results = runner.run(network, baseline_cp_time=baseline_cp_time, n_samples=2000)
    print(f"  Simulation complete — baseline CP = {baseline_cp_time:.0f} h")

    print_risk_rankings(results, baseline_cp_time)

    _banner("END OF DEMO")
    print()
    # Derive top driver dynamically from simulation results
    sen = results.get("cp_sensitivity", {})
    top_driver = max(sen, key=lambda a: sen[a]) if sen else "Q-MSIV-C"
    top_crit = results.get("criticality_index", {}).get(top_driver, 0.0) * 100

    print("  Key takeaways for outage planning:")
    print(f"  1. {top_driver} is the top schedule risk driver ({top_crit:.0f}% criticality index) —")
    print("     monitor its execution window closely and pre-stage resources.")
    print("  2. MCP seal replacement has well-characterised aleatory variability —")
    print("     a schedule float of ~6 h above the P50 is historically justified.")
    print("  3. The digital I&C upgrade has no close analogues (epistemic) —")
    print("     SME review and a pre-job risk assessment are recommended.")
    print()


if __name__ == "__main__":
    main()
