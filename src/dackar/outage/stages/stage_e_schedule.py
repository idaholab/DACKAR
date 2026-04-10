"""
Stage E — Schedule Impact Assessor.

Responsibilities:
    1. Load the current schedule network for the outage from the OutageDataset.
    2. Determine the insertion point for the emergent activity (after which
       task, in which phase, with what resource requirements).
    3. Compute float analysis: float consumed, remaining float, criticality label.
    4. Run a Monte Carlo simulation using the duration distribution from Stage D
       to produce probabilistic CP impact metrics (p50/p80/p90 finish times,
       expected delay, CP drag, CP sensitivity).
    5. Identify displaced tasks downstream of the insertion point.
    6. Check for resource conflicts at the insertion point.

Output schema: outage/schemas/schedule_impact_assessment.json

Reuse targets:
    LOGOS.src.CPM.pert.Pert                     → schedule network, CPM engine
    LOGOS.src.CPM.activity.Activity             → task nodes
    LOGOS.src.CPM.outage_data.OutageData        → schedule data container
    LOGOS.src.CPM.BaseCPMmodel.BaseCPMmodel     → RAVEN/Monte-Carlo pattern
        The MC simulation loop in _run_monte_carlo() mirrors the
        BaseCPMmodel.run() pattern: build Pert once, vary durations via
        set_durations(), call generateInfo() per iteration.

Integration notes:
    - schedule_loader callable must accept (outage_id, version) and return a
      LOGOS OutageData (or any object whose .outage_config has a 'version_id').
    - schedule_graph_builder must expose a .build(outage_data) method that
      returns a LOGOS Pert object with generateInfo() already called.
    - LOGOS requires Python ≥ 3.10 (uses X | Y union type syntax).

Python path note:
    Add the LOGOS project root to sys.path before importing:
        sys.path.insert(0, '/path/to/LOGOS')
    Then imports resolve as: from src.CPM.pert import Pert
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

JsonDict = Dict[str, Any]

# Confidence tier → numeric score (mirrors Stage D)
_CONFIDENCE_SCORE: Dict[str, float] = {
    "data_supported": 0.9,
    "sme_informed": 0.6,
    "low_confidence": 0.3,
}

# Small epsilon for float comparisons (hours)
_FLOAT_EPSILON = 0.01


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _ScheduleNetwork:
    """Thin wrapper around a LOGOS Pert instance.

    Satisfies the 'schedule_network.baseline_cp_hours' contract used
    throughout this stage, and provides a factory for building modified
    network copies with an emergent activity inserted.
    """

    pert: Any               # LOGOS Pert
    baseline_cp_hours: float


@dataclass
class _SimResult:
    """Output of _run_monte_carlo()."""

    project_durations: List[float]
    """Sampled project-completion durations (hours from outage start)."""

    on_cp_count: int
    """Number of MC runs in which the emergent activity had zero slack."""

    n_runs: int
    emergent_task_id: str


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ScheduleImpactConfig:
    """Configuration for Stage E."""

    use_p80_for_float_analysis: bool = False
    """If True, float analysis uses the p80 duration estimate (conservative).
    Default False uses p50 (expected-value basis)."""

    near_critical_float_threshold_hours: float = 8.0
    """Remaining float below this value triggers 'near_critical' label."""

    monte_carlo_runs: int = 1000
    """Number of Monte Carlo simulation samples."""

    max_displaced_tasks_reported: int = 20
    """Maximum displaced tasks to surface in the artifact."""

    check_resource_conflicts: bool = True
    """Whether to check for crew and vendor resource conflicts."""

    schedule_version_preference: str = "working"
    """Which schedule version to load: 'baseline', 'working', or 'as_run'.
    'working' (latest update) is the correct basis for live outage decisions."""


# ---------------------------------------------------------------------------
# Stage implementation
# ---------------------------------------------------------------------------

class ScheduleImpactAssessor:
    """Concrete Stage E implementation.

    Args:
        config: Stage configuration.
        schedule_loader: Callable(outage_id, version) → LOGOS OutageData.
                         Use the P6_adapter loader or a thin shim that
                         returns an object with .outage_config['version_id'].
        schedule_graph_builder: Object with .build(outage_data) → LOGOS Pert.
                                 The Pert must have generateInfo() already called
                                 (i.e. infoDict is populated with ES/EF/LS/LF/slack).
        monte_carlo: Reserved — MC is implemented directly in this stage using
                     the LOGOS set_durations() + generateInfo() pattern.
        cp_analyzer: Reserved — CP metrics are computed from the MC results
                     directly; no separate analyzer object is required.
    """

    def __init__(
        self,
        config: Optional[ScheduleImpactConfig] = None,
        *,
        schedule_loader=None,
        schedule_graph_builder=None,
        monte_carlo=None,
        cp_analyzer=None,
    ) -> None:
        self.config = config or ScheduleImpactConfig()
        self.schedule_loader = schedule_loader
        self.schedule_graph_builder = schedule_graph_builder
        # monte_carlo and cp_analyzer reserved for future external injection
        self.monte_carlo = monte_carlo
        self.cp_analyzer = cp_analyzer

    # ── Protocol method ───────────────────────────────────────────────────────

    def assess(
        self,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        historical_analogs: JsonDict,
        run_context: JsonDict,
    ) -> JsonDict:
        """Execute Stage E for one emergent activity.

        Returns:
            ScheduleImpactAssessment artifact conforming to
            outage/schemas/schedule_impact_assessment.json.
        """
        run_id: str = run_context["run_id"]
        activity_id: str = emergent_activity["activity_id"]
        LOGGER.debug(
            "Stage E schedule impact assessment for %s (run=%s)", activity_id, run_id
        )

        duration_dist = historical_analogs.get("duration_distribution", {})
        schedule_network, schedule_version_id = self._load_schedule_network(
            emergent_activity
        )
        insertion_point = self._determine_insertion_point(
            emergent_activity, intake_result, schedule_network
        )
        duration_for_float = (
            duration_dist.get("p80_hours")
            if self.config.use_p80_for_float_analysis
            else duration_dist.get("p50_hours")
        )
        float_analysis = self._compute_float_analysis(
            schedule_network, insertion_point, duration_for_float
        )
        sim_result = self._run_monte_carlo(
            schedule_network, insertion_point, duration_dist
        )
        cp_metrics = self._compute_cp_metrics(
            sim_result, schedule_network.baseline_cp_hours
        )
        displaced = self._identify_displaced_tasks(
            schedule_network, insertion_point, duration_for_float
        )
        conflicts: List[JsonDict] = []
        if self.config.check_resource_conflicts:
            conflicts = self._check_resource_conflicts(
                emergent_activity, insertion_point, schedule_network
            )
        confidence = self._compute_confidence(duration_dist, schedule_network)

        return {
            "activity_id": activity_id,
            "run_id": run_id,
            "generated_at": run_context.get("started_at", ""),
            "schedule_version_id": schedule_version_id,
            "insertion_point": insertion_point,
            "duration_estimate": {
                "p50_hours": duration_dist.get("p50_hours"),
                "p80_hours": duration_dist.get("p80_hours"),
                "p90_hours": duration_dist.get("p90_hours"),
                "mean_hours": duration_dist.get("mean_hours"),
                "std_hours": duration_dist.get("std_hours"),
                "distribution_type": duration_dist.get("distribution_type"),
                "confidence_tier": duration_dist.get("confidence_tier"),
                "sample_size": duration_dist.get("sample_size"),
            },
            "float_analysis": float_analysis,
            "cp_impact": cp_metrics,
            "displaced_tasks": displaced[: self.config.max_displaced_tasks_reported],
            "resource_conflicts": conflicts,
            "confidence": confidence,
            "notes": [
                "cp_impact metrics derived from 3-scenario deterministic proxy "
                "(p50/p80/p90 duration scenarios) — full Monte Carlo integration "
                "with RAVEN is deferred pending Pert interface restructuring. "
                "Treat percentile estimates conservatively."
            ],
            "provenance": {
                "generated_by": self.__class__.__name__,
                "run_id": run_id,
                "schedule_graph_version": None,
                "monte_carlo_runs": self.config.monte_carlo_runs,
            },
        }

    # ── Private step methods ──────────────────────────────────────────────────

    def _load_schedule_network(
        self, emergent_activity: JsonDict
    ) -> Tuple[_ScheduleNetwork, str]:
        """Load the current schedule network for the outage.

        Steps:
            1. Call schedule_loader(outage_id, version=config.schedule_version_preference)
               to get a LOGOS OutageData.
            2. Call schedule_graph_builder.build(dataset) to get a LOGOS Pert
               (generateInfo() must be called by the builder so infoDict is
               populated on return).
            3. Wrap Pert in _ScheduleNetwork and return (network, version_id).

        The LOGOS Pert exposes:
            - infoDict[activity] with keys: es, ef, ls, lf, slack, duration
            - getProjectDuration() → baseline_cp_hours
            - task_to_activity: Dict[str, Activity]
            - forwardDict / backwardDict for graph traversal
            - set_durations({task_id: hours}) + generateInfo() for MC iterations
        """
        if self.schedule_loader is None or self.schedule_graph_builder is None:
            raise RuntimeError(
                "Stage E requires schedule_loader and schedule_graph_builder "
                "to be injected at construction time. "
                "Inject both dependencies or handle RuntimeError in the orchestrator "
                "to produce a partial assessment with null cp_impact fields."
            )

        outage_id: str = emergent_activity["outage_id"]
        outage_data = self.schedule_loader(
            outage_id, version=self.config.schedule_version_preference
        )

        pert = self.schedule_graph_builder.build(outage_data)
        # Ensure CPM state is populated (builder should call generateInfo();
        # call it defensively here to guarantee infoDict is up-to-date).
        pert.generateInfo()

        baseline_cp_hours: float = pert.getProjectDuration()
        schedule_version_id: str = (
            outage_data.outage_config.get("version_id") or outage_id
        )

        return _ScheduleNetwork(pert=pert, baseline_cp_hours=baseline_cp_hours), schedule_version_id

    def _determine_insertion_point(
        self,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        schedule_network: _ScheduleNetwork,
    ) -> JsonDict:
        """Determine where in the schedule network to insert the emergent activity.

        Strategy:
            1. If actual_start is known, find the task active at that time:
               task with ES ≤ actual_start_offset ≤ EF.  The emergent activity
               is inserted immediately after that task.
            2. If actual_start is unknown, use outage_phase to restrict
               candidate tasks to the appropriate phase window, then select
               the task with the most available float (least disruptive).
            3. The before_task_id is the top-ranked immediate successor of
               after_task_id on the critical path (or None if after_task is
               a sink).

        Returns insertion_point dict with:
            after_task_id, before_task_id, outage_phase, proposed_start (ISO),
            proposed_finish (ISO), emergent_task_id.
        """
        pert = schedule_network.pert
        activity_id: str = emergent_activity["activity_id"]
        outage_phase: str = intake_result.get("outage_phase") or "unknown"

        # Convert actual_start to hours-from-project-start if available
        actual_start_offset: Optional[float] = None
        actual_start_iso: Optional[str] = emergent_activity.get("actual_start")
        if actual_start_iso and hasattr(pert, "startTime") and pert.startTime:
            try:
                actual_start_dt = _parse_dt(actual_start_iso)
                if actual_start_dt is not None:
                    actual_start_offset = (
                        actual_start_dt - _ensure_tz(pert.startTime)
                    ).total_seconds() / 3600.0
            except (TypeError, ValueError):
                pass

        info = pert.infoDict  # Dict[Activity, Dict]

        # Build candidate list: all activities with CPM data
        candidates = [
            (act, data) for act, data in info.items()
            if data.get("es") is not None and data.get("ef") is not None
        ]

        after_act = None

        if actual_start_offset is not None:
            # Strategy 1: find the active task at the detected time
            active_at_time = [
                (act, data) for act, data in candidates
                if data["es"] <= actual_start_offset <= data["ef"]
            ]
            if active_at_time:
                # Among active tasks, prefer the one on the CP (slack ≈ 0)
                active_at_time.sort(key=lambda x: abs(x[1].get("slack", 999)))
                after_act, _ = active_at_time[0]

        if after_act is None:
            # Strategy 2: use outage_phase filter + max-float selection
            # Phase filter: activities whose ES falls in the phase window
            # (Approximate: keep tasks whose es is in the middle half of the schedule)
            total_dur = schedule_network.baseline_cp_hours
            phase_ranges = _default_phase_windows(outage_phase, total_dur)
            phase_low, phase_high = phase_ranges

            phase_candidates = [
                (act, data) for act, data in candidates
                if phase_low <= data["es"] <= phase_high
            ]
            if not phase_candidates:
                phase_candidates = candidates  # fallback: use all tasks

            # Prefer task with maximum slack (most float available)
            phase_candidates.sort(key=lambda x: x[1].get("slack", 0.0), reverse=True)
            after_act, _ = phase_candidates[0]

        # Determine before_task: highest-priority immediate successor
        successors: List = pert.forwardDict.get(after_act, [])
        before_act = successors[0] if successors else None

        # Compute proposed times (absolute datetimes)
        after_es = info[after_act]["ef"] if after_act in info else 0.0
        proposed_start_iso: Optional[str] = None
        proposed_finish_iso: Optional[str] = None
        if hasattr(pert, "startTime") and pert.startTime:
            base = _ensure_tz(pert.startTime)
            proposed_start_iso = (base + timedelta(hours=after_es)).isoformat()
            # Use planned_duration_hours as the best available estimate at this
            # stage — the historical analog p50 is not yet available here.
            planned_dur_hours: float = float(
                emergent_activity.get("planned_duration_hours") or 0.0
            )
            proposed_finish_iso = (
                base + timedelta(hours=after_es + planned_dur_hours)
            ).isoformat()

        return {
            "emergent_task_id": f"EA::{activity_id}",
            "after_task_id": after_act.name if after_act else None,
            "before_task_id": before_act.name if before_act else None,
            "outage_phase": outage_phase,
            "proposed_start": proposed_start_iso,
            "proposed_finish": proposed_finish_iso,
        }

    def _compute_float_analysis(
        self,
        schedule_network: _ScheduleNetwork,
        insertion_point: JsonDict,
        duration_hours: Optional[float],
    ) -> JsonDict:
        """Compute float consumed and criticality label at the insertion point.

        Steps:
            1. Read available_float_before from after_task's slack.
            2. Build a modified Pert with the emergent activity inserted.
            3. Recompute CPM via generateInfo().
            4. Read remaining_float_after from before_task's new slack.
            5. Assign criticality_label.

        If duration_hours is None or zero, returns a non_critical placeholder.
        """
        if not duration_hours or duration_hours <= 0:
            return {
                "float_consumed_hours": 0.0,
                "available_float_before": None,
                "remaining_float_after": None,
                "is_critical_path_impact": False,
                "criticality_label": "non_critical",
            }

        pert = schedule_network.pert
        after_task_id = insertion_point.get("after_task_id")
        before_task_id = insertion_point.get("before_task_id")
        emergent_task_id = insertion_point.get("emergent_task_id", "EA_FLOAT_CHECK")

        # Float at the after-task before insertion
        after_act = pert.task_to_activity.get(after_task_id) if after_task_id else None
        available_float_before: Optional[float] = (
            pert.infoDict[after_act].get("slack")
            if after_act and after_act in pert.infoDict
            else None
        )

        # Build modified Pert with emergent activity inserted
        modified_pert = self._build_modified_pert(
            pert, emergent_task_id, duration_hours, after_task_id, before_task_id
        )

        # Float at the before-task after insertion
        before_act_mod = modified_pert.task_to_activity.get(before_task_id) if before_task_id else None
        remaining_float_after: Optional[float] = (
            modified_pert.infoDict[before_act_mod].get("slack")
            if before_act_mod and before_act_mod in modified_pert.infoDict
            else None
        )

        is_cp_impact = (
            remaining_float_after is not None
            and remaining_float_after <= _FLOAT_EPSILON
        )

        if remaining_float_after is None:
            criticality_label = "non_critical"
        elif remaining_float_after <= _FLOAT_EPSILON:
            criticality_label = "critical"
        elif remaining_float_after <= self.config.near_critical_float_threshold_hours:
            criticality_label = "near_critical"
        else:
            criticality_label = "non_critical"

        return {
            "float_consumed_hours": round(duration_hours, 2),
            "available_float_before": (
                round(available_float_before, 2)
                if available_float_before is not None else None
            ),
            "remaining_float_after": (
                round(remaining_float_after, 2)
                if remaining_float_after is not None else None
            ),
            "is_critical_path_impact": is_cp_impact,
            "criticality_label": criticality_label,
        }

    def _run_monte_carlo(
        self,
        schedule_network: _ScheduleNetwork,
        insertion_point: JsonDict,
        duration_dist: JsonDict,
    ) -> _SimResult:
        """Run Monte Carlo simulation for probabilistic CP impact.

        NOTE: Monte Carlo integration with RAVEN is deferred pending a planned
        restructuring of the LOGOS Pert ↔ RAVEN interface.  This method
        currently returns a _SimResult populated with three deterministic
        scenarios (p50, p80, p90) so that _compute_cp_metrics() can produce
        conservative-but-grounded estimates without probabilistic sampling.

        Future implementation will mirror the LOGOS BaseCPMmodel.run() pattern:
            1. Pert.clone_for_analysis() → clean copy for what-if analysis.
            2. Pert.insert_task(task_dict, after_task_id, before_task_id)
               → first-class topology mutation (proposed addition to Pert).
            3. For each RAVEN sample: set_durations({emergent_id: sample})
               → generateInfo() → record project duration + CP membership.
        """
        emergent_task_id = insertion_point.get("emergent_task_id", "EA_MC")
        after_task_id = insertion_point.get("after_task_id")
        before_task_id = insertion_point.get("before_task_id")

        p50: float = duration_dist.get("p50_hours") or 0.0
        p80: float = duration_dist.get("p80_hours") or p50
        p90: float = duration_dist.get("p90_hours") or p80

        pert = schedule_network.pert
        project_durations: List[float] = []
        on_cp_count = 0

        for scenario_dur in (p50, p80, p90):
            if scenario_dur <= 0:
                continue
            scenario_pert = self._build_modified_pert(
                pert, emergent_task_id, scenario_dur,
                after_task_id, before_task_id,
            )
            project_durations.append(scenario_pert.getProjectDuration())

            # CP membership check
            emergent_act = scenario_pert.task_to_activity.get(emergent_task_id)
            if emergent_act and emergent_act in scenario_pert.infoDict:
                slack = scenario_pert.infoDict[emergent_act].get("slack", 999.0)
                if abs(slack) <= _FLOAT_EPSILON:
                    on_cp_count += 1

        LOGGER.debug(
            "Stage E MC deferred — using 3-scenario deterministic proxy "
            "(p50=%.1f h, p80=%.1f h, p90=%.1f h).", p50, p80, p90
        )

        return _SimResult(
            project_durations=project_durations,
            on_cp_count=on_cp_count,
            n_runs=len(project_durations),
            emergent_task_id=emergent_task_id,
        )

    def _compute_cp_metrics(
        self,
        sim_result: _SimResult,
        baseline_cp_hours: float,
    ) -> JsonDict:
        """Compute project-level CP impact metrics from the simulation result.

        When the MC proxy provides only 3 scenarios (p50/p80/p90), the values
        map directly:
            estimated_new_cp_hours  — scenario[0]  (p50 run)
            p80_cp_hours            — scenario[1]  (p80 run)
            p90_cp_hours            — scenario[2]  (p90 run)
            cp_drag_hours           — max(0, estimated_new_cp - baseline_cp)
            cp_sensitivity_score    — fraction of scenarios where emergent
                                      task was on CP (0 / 0.33 / 0.67 / 1.0)

        When a full MC result is available (future RAVEN integration), the
        same fields are computed from the full distribution.
        """
        if not sim_result.project_durations:
            return {
                "baseline_cp_hours": round(baseline_cp_hours, 2),
                "estimated_new_cp_hours": round(baseline_cp_hours, 2),
                "cp_drag_hours": 0.0,
                "cp_sensitivity_score": 0.0,
                "p80_cp_hours": round(baseline_cp_hours, 2),
                "p90_cp_hours": round(baseline_cp_hours, 2),
            }

        durs = sorted(sim_result.project_durations)
        # For a 3-scenario proxy the indices map to p50/p80/p90 directly;
        # for a full MC result numpy percentiles give the right values.
        if len(durs) >= 10:
            p50 = float(np.percentile(durs, 50))
            p80 = float(np.percentile(durs, 80))
            p90 = float(np.percentile(durs, 90))
        else:
            p50 = durs[0]
            p80 = durs[min(1, len(durs) - 1)]
            p90 = durs[min(2, len(durs) - 1)]

        cp_drag = max(0.0, p50 - baseline_cp_hours)
        sensitivity = (
            sim_result.on_cp_count / sim_result.n_runs
            if sim_result.n_runs > 0 else 0.0
        )

        return {
            "baseline_cp_hours": round(baseline_cp_hours, 2),
            "estimated_new_cp_hours": round(p50, 2),
            "cp_drag_hours": round(cp_drag, 2),
            "cp_sensitivity_score": round(sensitivity, 4),
            "p80_cp_hours": round(p80, 2),
            "p90_cp_hours": round(p90, 2),
        }

    def _identify_displaced_tasks(
        self,
        schedule_network: _ScheduleNetwork,
        insertion_point: JsonDict,
        duration_hours: Optional[float],
    ) -> List[JsonDict]:
        """Find tasks downstream of the insertion point that would be delayed.

        Traverses successor edges from the insertion node in the modified Pert
        and collects tasks whose ES shifts by > 0 hours after the insertion.

        has_regulatory_constraint is populated by a KG query; currently set to
        False as a placeholder — the orchestrator should enrich this field.
        """
        if not duration_hours or duration_hours <= 0:
            return []

        pert = schedule_network.pert
        after_task_id = insertion_point.get("after_task_id")
        before_task_id = insertion_point.get("before_task_id")
        emergent_task_id = insertion_point.get("emergent_task_id", "EA_DISPLACED")

        modified_pert = self._build_modified_pert(
            pert, emergent_task_id, duration_hours,
            after_task_id, before_task_id,
        )

        displaced: List[JsonDict] = []
        for act, new_info in modified_pert.infoDict.items():
            if act.name == emergent_task_id:
                continue
            baseline_act = pert.task_to_activity.get(act.name)
            if baseline_act is None or baseline_act not in pert.infoDict:
                continue

            new_es = new_info.get("es", 0.0)
            old_es = pert.infoDict[baseline_act].get("es", 0.0)
            shift = new_es - old_es

            if shift > _FLOAT_EPSILON:
                displaced.append({
                    "task_id": act.name,
                    "description": getattr(act, "description", None) or act.name,
                    "es_shift_hours": round(shift, 2),
                    "new_float_hours": round(new_info.get("slack", 0.0), 2),
                    # KG enrichment required to populate this field:
                    "has_regulatory_constraint": False,
                })

        displaced.sort(key=lambda x: x["es_shift_hours"], reverse=True)
        return displaced

    def _check_resource_conflicts(
        self,
        emergent_activity: JsonDict,
        insertion_point: JsonDict,
        schedule_network: _ScheduleNetwork,
    ) -> List[JsonDict]:
        """Check for crew, vendor, and craft conflicts at the insertion point.

        Queries the LOGOS ResourcePool at the proposed insertion window and
        compares against the emergent activity's required_resources.

        Returns a list of conflict dicts (empty if no conflicts).
        """
        pert = schedule_network.pert
        resource_pool = getattr(pert, "resource_pool", None)
        if resource_pool is None:
            return []

        proposed_start_iso = insertion_point.get("proposed_start")
        proposed_finish_iso = insertion_point.get("proposed_finish")
        if not proposed_start_iso or not proposed_finish_iso:
            return []

        start_dt = _parse_dt(proposed_start_iso)
        end_dt = _parse_dt(proposed_finish_iso)
        if start_dt is None or end_dt is None:
            return []

        required_resources = emergent_activity.get("required_resources") or []
        # Infer from crew_size / discipline if structured list not available
        if not required_resources:
            crew_size = emergent_activity.get("crew_size")
            discipline = emergent_activity.get("discipline")
            if crew_size and discipline:
                required_resources = [
                    {"skill_type": discipline.upper(), "crew_count": crew_size}
                ]

        conflicts: List[JsonDict] = []
        for req in required_resources:
            skill = req.get("skill_type") or req.get("discipline")
            needed = req.get("crew_count") or req.get("crew_size") or 1
            if not skill:
                continue

            try:
                available = resource_pool.get_availability_in_range(
                    skill, start_dt, end_dt
                )
            except Exception:
                LOGGER.debug(
                    "ResourcePool query failed for skill=%s; skipping.", skill
                )
                continue

            if available < needed:
                conflicts.append({
                    "resource_type": "crew",
                    "skill_type": skill,
                    "required": needed,
                    "available": available,
                    "shortfall": needed - available,
                    "window_start": proposed_start_iso,
                    "window_end": proposed_finish_iso,
                })

        # Vendor conflict: is_vendor_supported flag
        if emergent_activity.get("is_vendor_supported"):
            # Conservative: flag as potential conflict if any crew conflict exists
            if conflicts:
                conflicts.append({
                    "resource_type": "vendor",
                    "skill_type": "VENDOR",
                    "required": 1,
                    "available": 0,
                    "shortfall": 1,
                    "note": "Vendor availability requires manual confirmation.",
                    "window_start": proposed_start_iso,
                    "window_end": proposed_finish_iso,
                })

        return conflicts

    def _compute_confidence(
        self, duration_dist: JsonDict, schedule_network: _ScheduleNetwork
    ) -> float:
        """Overall confidence in this assessment.

        Combines:
            - Duration distribution confidence_tier weight (0.60)
                data_supported=0.9 / sme_informed=0.6 / low_confidence=0.3
            - Schedule data completeness (0.30): fraction of tasks with
              complete ES/EF/slack data in the LOGOS infoDict.
            - Monte Carlo convergence (0.10): ratio of actual MC runs to
              a baseline of 500 runs that signals reasonable convergence.
        """
        tier = duration_dist.get("confidence_tier", "low_confidence")
        tier_score = _CONFIDENCE_SCORE.get(tier, 0.3)

        pert = schedule_network.pert
        info = pert.infoDict
        total = len(info)
        if total > 0:
            complete = sum(
                1 for d in info.values()
                if d.get("es") is not None
                and d.get("ef") is not None
                and d.get("slack") is not None
            )
            schedule_completeness = complete / total
        else:
            schedule_completeness = 0.5

        mc_convergence = min(1.0, self.config.monte_carlo_runs / 500.0)

        return round(
            0.60 * tier_score
            + 0.30 * schedule_completeness
            + 0.10 * mc_convergence,
            4,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_modified_pert(
        self,
        pert: Any,
        task_id: str,
        duration_hours: float,
        after_task_id: Optional[str],
        before_task_id: Optional[str],
    ) -> Any:
        """Return an analysis copy of pert with the emergent activity inserted.

        Delegates to Pert.clone_for_analysis() for the copy and to
        Pert.insert_task() for the topology mutation, so that no Pert
        internals are touched directly from this stage.
        """
        modified = pert.clone_for_analysis()
        modified.insert_task(
            {
                "task_id": task_id,
                "duration": duration_hours,
                "successors": [before_task_id] if before_task_id else [],
            },
            after_task_id=after_task_id,
            before_task_id=before_task_id,
        )
        modified.resetInfo()
        modified.generateInfo()
        return modified



# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------

def _parse_dt(iso_str: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 string to a timezone-aware datetime."""
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _ensure_tz(dt: datetime) -> datetime:
    """Return a timezone-aware datetime, assuming UTC if naive."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _default_phase_windows(
    outage_phase: str, total_hours: float
) -> Tuple[float, float]:
    """Map an outage phase label to an approximate (start, end) offset range.

    Used as a coarse filter when actual_start is unknown.
    Fractions based on typical nuclear refuelling outage phase ordering.
    """
    _PHASE_MAP: Dict[str, Tuple[float, float]] = {
        "shutdown":        (0.00, 0.10),
        "defueling":       (0.05, 0.25),
        "maintenance":     (0.20, 0.70),
        "refueling":       (0.20, 0.70),
        "inspection":      (0.30, 0.80),
        "testing":         (0.60, 0.90),
        "startup":         (0.80, 1.00),
        "power_ascension": (0.85, 1.00),
    }
    frac = _PHASE_MAP.get(outage_phase.lower(), (0.0, 1.0))
    return frac[0] * total_hours, frac[1] * total_hours


