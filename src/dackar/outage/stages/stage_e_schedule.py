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
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

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

# Regulatory constraint detection for displaced task enrichment.
# Mirrors the pattern set in stage_a_intake._REGULATORY_KEYWORDS_RE so that
# displaced tasks with TS/LCO/NRC/hold-point/surveillance language in their
# description are correctly flagged without creating a cross-stage import.
_REGULATORY_KEYWORDS_RE = re.compile(
    r"\b(TS\s*[\d.]+|technical\s+specification|LCO\s*[\d.]+|limiting\s+condition"
    r"|NRC|ALARA|CAP\b|corrective\s+action\s+program|surveillance|10\s*CFR"
    r"|operability\s+determination|hold\s+point|quality\s+hold"
    r"|mode\s+change|entry\s+condition)\b",
    re.IGNORECASE,
)

# Minimum credible duration for a Monte Carlo scenario (hours).
# Samples below this floor indicate bad data in the analog index and are
# clamped with a WARNING so the issue surfaces rather than silently producing
# nonsensical schedule arithmetic (e.g. zero-duration tasks on the CP).
_MIN_DURATION_HOURS = 0.1


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
    # Locked-baseline fields — populated when a separate baseline schedule
    # version is successfully loaded; None when baseline locking is disabled
    # or the baseline schedule is unavailable.
    locked_baseline_cp_hours: Optional[float] = None
    locked_baseline_start: Optional[datetime] = None  # for absolute finish computation
    working_start: Optional[datetime] = None          # for projected_finish computation


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

    baseline_schedule_version: str = "baseline"
    """Version tag for the locked kickoff baseline schedule.  When non-empty,
    Stage E loads a second copy of the schedule at this version to compute
    schedule variance (how far the outage has already slipped from the original
    plan) and total overrun (variance + cp_drag from this activity).
    Set to empty string to disable baseline locking."""

    high_crew_utilization_threshold: float = 0.80
    """Crew utilization fraction (0–1) above which a skill type is flagged as
    high-utilization in the crew_continuity section.  0.80 = 80% of available
    crew already committed at the insertion window."""

    fatigue_risk_off_shift_hours: float = 2.0
    """Hours of off-shift overlap in the insertion window that trigger a
    fatigue_risk flag.  When the proposed insertion window extends more than
    this many hours into off-shift time the crew continuity assessment flags
    a potential fatigue / overnight-work risk."""

    # ── Permit lead time ──────────────────────────────────────────────────────
    permit_lead_times_enabled: bool = True
    """Master switch for permit/approval lead time modeling.  When False the
    proposed_start is not adjusted and no permit overhead is added to Monte
    Carlo scenarios.  Set to False to restore the pre-permit-lead-time behavior
    for comparison or backward-compatible testing."""

    rp_hold_lead_time_hours: float = 4.0
    """Hours of RP (radiation protection) survey + ALARA briefing required
    before work can start when ``has_rp_hold`` is True.  Represents one RP
    review cycle; complex high-dose jobs may require more."""

    scaffold_lead_time_hours: float = 8.0
    """Hours for scaffold erection and inspection sign-off when
    ``requires_scaffold`` is True.  Sized as one full shift — scaffold must
    be erected, inspected, and certified before the activity starts."""

    clearance_lead_time_hours: float = 2.0
    """Hours for electrical or mechanical clearance / LOTO procedures when
    ``has_clearance`` is True.  LOTO establishment typically takes 1–3 hours
    depending on the number of isolation points."""

    permit_lead_time_mode: str = "max"
    """How to combine overlapping permits when multiple flags are active.

    ``"max"``  (default) — permits are requested in parallel; the longest
               single permit drives the delay.  Appropriate when the outage
               manager can start all approval processes simultaneously.

    ``"sum"``  — permits must be obtained sequentially; lead times are added.
               Use for plants whose procedures require serial sign-off (e.g.
               clearance must precede RP survey at that unit)."""


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
        monte_carlo: Unused — MC sampling is implemented directly in this
                     stage via ``_run_monte_carlo()`` using the LOGOS
                     ``set_durations()`` + ``generateInfo()`` pattern.
                     Reserved for future external injection.
        cp_analyzer: Unused — CP metrics are computed directly from the
                     ``_SimResult`` distributions.  Reserved for future
                     external injection.
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
        *,
        schedule_context: Any = None,
    ) -> JsonDict:
        """Execute Stage E for one emergent activity.

        Args:
            emergent_activity: EmergentActivity artifact.
            intake_result: Stage A output.
            historical_analogs: Stage D output (duration distribution consumed here).
            run_context: Run metadata block.
            schedule_context: Optional :class:`~stages.insertion_point_determiner.ScheduleContext`
                produced by the pre-pass ``InsertionPointDeterminer``.  When
                provided the insertion point determination step is skipped —
                ``schedule_context.to_insertion_point()`` is used directly.
                This avoids loading and traversing the schedule network twice
                (once in the pre-pass and once here).

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

        # Use pre-computed insertion point when the two-pass pre-pass ran; this
        # avoids traversing the schedule network a second time.
        if schedule_context is not None:
            insertion_point = schedule_context.to_insertion_point()
            LOGGER.debug(
                "Stage E: using pre-computed insertion point from ScheduleContext "
                "(after=%s, on_cp=%s)",
                getattr(schedule_context, "after_task_id", None),
                getattr(schedule_context, "insertion_on_cp", None),
            )
        else:
            insertion_point = self._determine_insertion_point(
                emergent_activity, intake_result, schedule_network
            )
        # ── Permit / approval lead time ───────────────────────────────────────
        # Computed from execution_mode_flags extracted by Stage A.  Lead time
        # is a fixed calendar overhead before the activity can start — it shifts
        # proposed_start forward and is added to every Monte Carlo scenario
        # duration so that CP drag correctly reflects the approval ceiling.
        execution_mode_flags: JsonDict = (
            intake_result.get("execution_mode_flags") or {}
        )
        if self.config.permit_lead_times_enabled:
            permit_lead_time = _compute_permit_lead_time(
                execution_mode_flags,
                rp_hold_hours=self.config.rp_hold_lead_time_hours,
                scaffold_hours=self.config.scaffold_lead_time_hours,
                clearance_hours=self.config.clearance_lead_time_hours,
                mode=self.config.permit_lead_time_mode,
            )
        else:
            permit_lead_time = {
                "total_lead_hours": 0.0,
                "rp_hold_hours": 0.0,
                "scaffold_hours": 0.0,
                "clearance_hours": 0.0,
                "start_adjusted": False,
                "combination_mode": self.config.permit_lead_time_mode,
                "notes": ["Permit lead time modeling disabled via config."],
            }

        lead_hours: float = permit_lead_time["total_lead_hours"]

        # Shift proposed_start / proposed_finish forward by lead_hours.
        if lead_hours > 0:
            insertion_point = _shift_insertion_times(insertion_point, lead_hours)

        duration_for_float = (
            duration_dist.get("p80_hours")
            if self.config.use_p80_for_float_analysis
            else duration_dist.get("p50_hours")
        )
        float_analysis = self._compute_float_analysis(
            schedule_network, insertion_point, duration_for_float
        )
        # Pass lead_hours to Monte Carlo so each scenario includes the permit
        # overhead in the effective activity duration fed to the CPM engine.
        sim_result = self._run_monte_carlo(
            schedule_network, insertion_point, duration_dist,
            permit_lead_hours=lead_hours,
        )
        cp_metrics = self._compute_cp_metrics(
            sim_result,
            schedule_network.baseline_cp_hours,
            locked_baseline_cp_hours=schedule_network.locked_baseline_cp_hours,
            locked_baseline_start=schedule_network.locked_baseline_start,
            working_start=schedule_network.working_start,
        )
        displaced = self._identify_displaced_tasks(
            schedule_network, insertion_point, duration_for_float
        )
        conflicts: List[JsonDict] = []
        if self.config.check_resource_conflicts:
            conflicts = self._check_resource_conflicts(
                emergent_activity, insertion_point, schedule_network
            )
        crew_continuity = self._assess_crew_continuity(
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
            "crew_continuity": crew_continuity,
            "permit_lead_time": permit_lead_time,
            "confidence": confidence,
            "notes": (
                [
                    f"cp_impact metrics derived from "
                    f"{self.config.monte_carlo_runs}-run lognormal Monte Carlo "
                    "simulation (LOGOS Pert CPM engine, BaseCPMmodel pattern)."
                ]
                if self.config.monte_carlo_runs >= 10
                else [
                    "cp_impact metrics derived from 3-scenario deterministic "
                    "proxy (p50/p80/p90 duration scenarios). "
                    "Set monte_carlo_runs ≥ 10 for probabilistic estimates."
                ]
            ),
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

        # Extract working schedule start for absolute datetime computations.
        working_start: Optional[datetime] = None
        if hasattr(pert, "startTime") and pert.startTime:
            working_start = _ensure_tz(pert.startTime)

        # Optionally load the locked kickoff baseline for schedule variance.
        locked_baseline_cp_hours: Optional[float] = None
        locked_baseline_start: Optional[datetime] = None
        if self.config.baseline_schedule_version:
            try:
                baseline_data = self.schedule_loader(
                    outage_id, version=self.config.baseline_schedule_version
                )
                baseline_pert = self.schedule_graph_builder.build(baseline_data)
                baseline_pert.generateInfo()
                locked_baseline_cp_hours = baseline_pert.getProjectDuration()
                if hasattr(baseline_pert, "startTime") and baseline_pert.startTime:
                    locked_baseline_start = _ensure_tz(baseline_pert.startTime)
            except Exception:  # noqa: BLE001
                LOGGER.debug(
                    "Stage E: locked baseline schedule unavailable for outage %s "
                    "(version=%s); schedule variance will not be reported.",
                    outage_id, self.config.baseline_schedule_version,
                )

        return _ScheduleNetwork(
            pert=pert,
            baseline_cp_hours=baseline_cp_hours,
            locked_baseline_cp_hours=locked_baseline_cp_hours,
            locked_baseline_start=locked_baseline_start,
            working_start=working_start,
        ), schedule_version_id

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
        *,
        permit_lead_hours: float = 0.0,
    ) -> _SimResult:
        """Run Monte Carlo simulation for probabilistic CP impact.

        Samples task duration from a lognormal distribution fitted to the
        Stage D percentile estimates (p50 as median; sigma fitted from p80
        when available, or from mean/std via method-of-moments).  Each
        sample is fed to the LOGOS Pert CPM engine using the
        ``BaseCPMmodel`` pattern: clone once, insert once, then loop
        ``set_durations()`` + ``resetInfo()`` + ``generateInfo()`` per
        iteration, recording the project duration and CP membership of the
        emergent activity.

        When ``set_durations`` is not available on the Pert object, falls
        back to cloning the Pert on every iteration (slower but correct for
        any duck-typed Pert-like object).

        When ``config.monte_carlo_runs < 10``, falls back to the lightweight
        3-scenario deterministic proxy (p50/p80/p90) for backwards
        compatibility with unit tests and low-latency scoring contexts.

        Args:
            permit_lead_hours: Fixed approval/permit overhead (hours) added to
                every sample before it is fed to the CPM engine.  This shifts
                the CP distribution uniformly upward, correctly increasing
                cp_drag for activities that require RP surveys, scaffold
                erection, or LOTO clearances before starting.
        """
        n_runs = self.config.monte_carlo_runs
        if n_runs < 10:
            return self._run_3scenario_proxy(
                schedule_network, insertion_point, duration_dist,
                permit_lead_hours=permit_lead_hours,
            )

        emergent_task_id = insertion_point.get("emergent_task_id", "EA_MC")
        after_task_id = insertion_point.get("after_task_id")
        before_task_id = insertion_point.get("before_task_id")

        # Seed the initial clone at p50 + lead (topology only; overwritten
        # per iteration via set_durations or a fresh clone).
        seed_dur = max(
            (duration_dist.get("p50_hours") or _MIN_DURATION_HOURS) + permit_lead_hours,
            _MIN_DURATION_HOURS,
        )
        pert = schedule_network.pert
        modified_pert = self._build_modified_pert(
            pert, emergent_task_id, seed_dur, after_task_id, before_task_id
        )

        sampler = self._build_duration_sampler(duration_dist, permit_lead_hours)

        project_durations: List[float] = []
        on_cp_count = 0

        if hasattr(modified_pert, "set_durations"):
            # Efficient LOGOS path: mutate durations in-place, regenerate CPM.
            for _ in range(n_runs):
                sampled_dur = sampler()
                modified_pert.set_durations({emergent_task_id: sampled_dur})
                modified_pert.resetInfo()
                modified_pert.generateInfo()
                project_durations.append(modified_pert.getProjectDuration())
                emergent_act = modified_pert.task_to_activity.get(emergent_task_id)
                if emergent_act and emergent_act in modified_pert.infoDict:
                    slack = modified_pert.infoDict[emergent_act].get("slack", 999.0)
                    if abs(slack) <= _FLOAT_EPSILON:
                        on_cp_count += 1
        else:
            # Safe fallback: clone Pert on every iteration.
            for _ in range(n_runs):
                sampled_dur = sampler()
                scenario_pert = self._build_modified_pert(
                    pert, emergent_task_id, sampled_dur,
                    after_task_id, before_task_id,
                )
                project_durations.append(scenario_pert.getProjectDuration())
                emergent_act = scenario_pert.task_to_activity.get(emergent_task_id)
                if emergent_act and emergent_act in scenario_pert.infoDict:
                    slack = scenario_pert.infoDict[emergent_act].get("slack", 999.0)
                    if abs(slack) <= _FLOAT_EPSILON:
                        on_cp_count += 1

        LOGGER.debug(
            "Stage E MC: %d samples for task '%s' "
            "(lognormal fit from p50/p80/std; permit_lead=%.1f h).",
            n_runs, emergent_task_id, permit_lead_hours,
        )
        return _SimResult(
            project_durations=project_durations,
            on_cp_count=on_cp_count,
            n_runs=n_runs,
            emergent_task_id=emergent_task_id,
        )

    def _run_3scenario_proxy(
        self,
        schedule_network: _ScheduleNetwork,
        insertion_point: JsonDict,
        duration_dist: JsonDict,
        *,
        permit_lead_hours: float = 0.0,
    ) -> _SimResult:
        """3-scenario deterministic proxy (p50 / p80 / p90).

        Used when ``config.monte_carlo_runs < 10`` — lightweight alternative
        for backwards-compatible tests and low-latency scoring contexts.
        ``_compute_cp_metrics()`` detects fewer than 10 samples and maps
        the three values directly to estimated_new_cp / p80_cp / p90_cp.
        """
        emergent_task_id = insertion_point.get("emergent_task_id", "EA_MC")
        after_task_id = insertion_point.get("after_task_id")
        before_task_id = insertion_point.get("before_task_id")

        p50: float = duration_dist.get("p50_hours") or 0.0
        p80: float = duration_dist.get("p80_hours") or p50
        p90: float = duration_dist.get("p90_hours") or p80

        def _clamp(val: float, label: str) -> float:
            if val < _MIN_DURATION_HOURS:
                LOGGER.warning(
                    "Stage E: duration scenario %s=%.4f h is below minimum "
                    "floor (%.1f h); clamping. Check analog index for "
                    "zero/negative actual_duration_hours entries.",
                    label, val, _MIN_DURATION_HOURS,
                )
                return _MIN_DURATION_HOURS
            return val

        p50 = _clamp(p50, "p50")
        p80 = _clamp(p80, "p80")
        p90 = _clamp(p90, "p90")

        effective_scenarios = (
            p50 + permit_lead_hours,
            p80 + permit_lead_hours,
            p90 + permit_lead_hours,
        )

        pert = schedule_network.pert
        project_durations: List[float] = []
        on_cp_count = 0

        for scenario_dur in effective_scenarios:
            if scenario_dur <= 0:
                continue
            scenario_pert = self._build_modified_pert(
                pert, emergent_task_id, scenario_dur,
                after_task_id, before_task_id,
            )
            project_durations.append(scenario_pert.getProjectDuration())
            emergent_act = scenario_pert.task_to_activity.get(emergent_task_id)
            if emergent_act and emergent_act in scenario_pert.infoDict:
                slack = scenario_pert.infoDict[emergent_act].get("slack", 999.0)
                if abs(slack) <= _FLOAT_EPSILON:
                    on_cp_count += 1

        LOGGER.debug(
            "Stage E MC: 3-scenario proxy "
            "(p50=%.1f h, p80=%.1f h, p90=%.1f h).", p50, p80, p90
        )
        return _SimResult(
            project_durations=project_durations,
            on_cp_count=on_cp_count,
            n_runs=len(project_durations),
            emergent_task_id=emergent_task_id,
        )

    def _build_duration_sampler(
        self,
        duration_dist: JsonDict,
        permit_lead_hours: float = 0.0,
    ) -> Callable[[], float]:
        """Return a callable that draws one lognormal sample (hours) per call.

        Fitting priority:
            1. ``mean_hours`` + ``std_hours`` → method-of-moments lognormal.
            2. ``p50_hours`` + ``p80_hours``  → sigma from 80th percentile
               (Φ⁻¹(0.80) ≈ 0.8416).
            3. ``p50_hours`` + ``p90_hours``  → sigma from 90th percentile
               (Φ⁻¹(0.90) ≈ 1.2816).
            4. ``p50_hours`` only             → minimal spread (σ = 0.10),
               near-deterministic behaviour.

        ``permit_lead_hours`` is added to every sample so that the approval
        overhead is always included in the effective task duration fed to the
        CPM engine.  Samples below ``_MIN_DURATION_HOURS`` are clamped.
        """
        p50 = max(
            duration_dist.get("p50_hours") or _MIN_DURATION_HOURS,
            _MIN_DURATION_HOURS,
        )
        p80 = duration_dist.get("p80_hours")
        p90 = duration_dist.get("p90_hours")
        mean = duration_dist.get("mean_hours")
        std = duration_dist.get("std_hours")

        if mean is not None and mean > 0 and std is not None and std > 0:
            # Method-of-moments: Var[X] = (exp(σ²) - 1) · E[X]²
            sigma = math.sqrt(math.log(1.0 + (std / mean) ** 2))
            mu = math.log(mean) - sigma ** 2 / 2.0
        elif p80 is not None and p80 > p50:
            mu = math.log(p50)
            sigma = (math.log(p80) - mu) / 0.8416
        elif p90 is not None and p90 > p50:
            mu = math.log(p50)
            sigma = (math.log(p90) - mu) / 1.2816
        else:
            mu = math.log(p50)
            sigma = 0.10

        sigma = max(sigma, 0.01)  # safety floor against degenerate inputs
        rng = np.random.default_rng()

        def _sample() -> float:
            raw = rng.lognormal(mean=mu, sigma=sigma)
            return max(raw + permit_lead_hours, _MIN_DURATION_HOURS)

        return _sample

    def _compute_cp_metrics(
        self,
        sim_result: _SimResult,
        baseline_cp_hours: float,
        locked_baseline_cp_hours: Optional[float] = None,
        locked_baseline_start: Optional[datetime] = None,
        working_start: Optional[datetime] = None,
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

        When locked_baseline_cp_hours is provided (baseline locking enabled),
        additional variance fields are computed:
            schedule_variance_hours — baseline_cp_hours - locked_baseline_cp_hours
                                      Positive means the outage has already slipped
                                      from the original kickoff plan before this
                                      activity is considered.
            total_overrun_hours     — max(0, estimated_new_cp - locked_baseline_cp_hours)
                                      Total slip vs. original plan after insertion.
            locked_baseline_finish  — ISO datetime of original planned outage finish.
            projected_finish_after_insertion
                                    — ISO datetime of projected finish after inserting
                                      the emergent activity (p50 scenario).

        When a full MC result is available (future RAVEN integration), the
        same fields are computed from the full distribution.
        """
        if not sim_result.project_durations:
            result: JsonDict = {
                "baseline_cp_hours": round(baseline_cp_hours, 2),
                "estimated_new_cp_hours": round(baseline_cp_hours, 2),
                "cp_drag_hours": 0.0,
                "cp_sensitivity_score": 0.0,
                "p80_cp_hours": round(baseline_cp_hours, 2),
                "p90_cp_hours": round(baseline_cp_hours, 2),
            }
            if locked_baseline_cp_hours is not None:
                result.update(
                    self._locked_baseline_fields(
                        baseline_cp_hours, baseline_cp_hours,
                        locked_baseline_cp_hours,
                        locked_baseline_start, working_start,
                    )
                )
            return result

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

        result = {
            "baseline_cp_hours": round(baseline_cp_hours, 2),
            "estimated_new_cp_hours": round(p50, 2),
            "cp_drag_hours": round(cp_drag, 2),
            "cp_sensitivity_score": round(sensitivity, 4),
            "p80_cp_hours": round(p80, 2),
            "p90_cp_hours": round(p90, 2),
        }
        if locked_baseline_cp_hours is not None:
            result.update(
                self._locked_baseline_fields(
                    p50, baseline_cp_hours,
                    locked_baseline_cp_hours,
                    locked_baseline_start, working_start,
                )
            )
        return result

    @staticmethod
    def _locked_baseline_fields(
        estimated_new_cp: float,
        working_cp: float,
        locked_baseline_cp_hours: float,
        locked_baseline_start: Optional[datetime],
        working_start: Optional[datetime],
    ) -> JsonDict:
        """Compute schedule-variance fields relative to the locked baseline.

        Args:
            estimated_new_cp: p50 project duration after inserting the activity.
            working_cp: Current working-plan duration (pre-insertion).
            locked_baseline_cp_hours: Original kickoff plan duration.
            locked_baseline_start: Original planned start datetime.
            working_start: Current working-plan start datetime.
        """
        schedule_variance = round(working_cp - locked_baseline_cp_hours, 2)
        total_overrun = round(max(0.0, estimated_new_cp - locked_baseline_cp_hours), 2)

        locked_finish_iso: Optional[str] = None
        if locked_baseline_start is not None:
            locked_finish_iso = (
                locked_baseline_start
                + timedelta(hours=locked_baseline_cp_hours)
            ).isoformat()

        projected_finish_iso: Optional[str] = None
        if working_start is not None:
            projected_finish_iso = (
                working_start + timedelta(hours=estimated_new_cp)
            ).isoformat()

        return {
            "locked_baseline_cp_hours": round(locked_baseline_cp_hours, 2),
            "schedule_variance_hours": schedule_variance,
            "total_overrun_hours": total_overrun,
            "locked_baseline_finish": locked_finish_iso,
            "projected_finish_after_insertion": projected_finish_iso,
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
                description = getattr(act, "description", None) or act.name
                displaced.append({
                    "task_id": act.name,
                    "description": description,
                    "es_shift_hours": round(shift, 2),
                    "new_float_hours": round(new_info.get("slack", 0.0), 2),
                    # Populated from the task description using the same
                    # regulatory keyword patterns as Stage A intake.  A KG
                    # lookup would be more authoritative but requires a
                    # connected driver; description matching covers the
                    # common cases (TS, LCO, NRC, surveillance, hold point).
                    "has_regulatory_constraint": _has_regulatory_constraint(description),
                })

        displaced.sort(key=lambda x: x["es_shift_hours"], reverse=True)
        return displaced

    def _check_resource_conflicts(
        self,
        emergent_activity: JsonDict,
        insertion_point: JsonDict,
        schedule_network: _ScheduleNetwork,
    ) -> List[JsonDict]:
        """Check for crew, equipment, location, and vendor conflicts at the insertion point.

        Queries the LOGOS ResourcePool, EquipmentPool, and LocationPool at the
        proposed insertion window and compares against the emergent activity's
        required_resources, required_equipment, and location_id.

        Returns a list of conflict dicts (empty if no conflicts).
        """
        pert = schedule_network.pert

        proposed_start_iso = insertion_point.get("proposed_start")
        proposed_finish_iso = insertion_point.get("proposed_finish")
        if not proposed_start_iso or not proposed_finish_iso:
            return []

        start_dt = _parse_dt(proposed_start_iso)
        end_dt = _parse_dt(proposed_finish_iso)
        if start_dt is None or end_dt is None:
            return []

        conflicts: List[JsonDict] = []

        # ── Crew conflicts ────────────────────────────────────────────────────
        resource_pool = getattr(pert, "resource_pool", None)
        if resource_pool is not None:
            required_resources = emergent_activity.get("required_resources") or []
            # Infer from crew_size / discipline if structured list not available
            if not required_resources:
                crew_size = emergent_activity.get("crew_size")
                discipline = emergent_activity.get("discipline")
                if crew_size and discipline:
                    required_resources = [
                        {"skill_type": discipline.upper(), "crew_count": crew_size}
                    ]

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

        # ── Equipment conflicts ───────────────────────────────────────────────
        equipment_pool = getattr(pert, "equipment_pool", None)
        if equipment_pool is not None:
            for eq_req in emergent_activity.get("required_equipment") or []:
                eq_id = eq_req.get("equipment_id")
                qty_needed = eq_req.get("quantity_needed") or 1
                if not eq_id:
                    continue

                try:
                    available = equipment_pool.get_availability_in_range(
                        eq_id, start_dt, end_dt
                    )
                except Exception:
                    LOGGER.debug(
                        "EquipmentPool query failed for equipment_id=%s; skipping.", eq_id
                    )
                    continue

                if available < qty_needed:
                    conflicts.append({
                        "resource_type": "equipment",
                        "equipment_id": eq_id,
                        "required": qty_needed,
                        "available": available,
                        "shortfall": qty_needed - available,
                        "window_start": proposed_start_iso,
                        "window_end": proposed_finish_iso,
                    })

        # ── Location conflicts ────────────────────────────────────────────────
        location_pool = getattr(pert, "location_pool", None)
        location_id = emergent_activity.get("location_id")
        if location_pool is not None and location_id:
            try:
                capacity = location_pool.get_capacity_in_range(
                    location_id, start_dt, end_dt
                )
                max_tasks = capacity.get("max_tasks", 0) if capacity else 0
                if max_tasks == 0:
                    conflicts.append({
                        "resource_type": "location",
                        "location_id": location_id,
                        "required": 1,
                        "available": 0,
                        "shortfall": 1,
                        "note": "Location inaccessible or at capacity during insertion window.",
                        "window_start": proposed_start_iso,
                        "window_end": proposed_finish_iso,
                    })
            except Exception:
                LOGGER.debug(
                    "LocationPool capacity query failed for location_id=%s; skipping.", location_id
                )

            try:
                if location_pool.is_confined_space(location_id):
                    conflicts.append({
                        "resource_type": "location",
                        "location_id": location_id,
                        "required": 1,
                        "available": 1,
                        "shortfall": 0,
                        "note": "Confined space — confined space entry permit required.",
                        "window_start": proposed_start_iso,
                        "window_end": proposed_finish_iso,
                        "confined_space": True,
                    })
            except Exception:
                LOGGER.debug(
                    "LocationPool confined-space query failed for location_id=%s; skipping.",
                    location_id,
                )

        # ── Vendor conflict: is_vendor_supported flag ─────────────────────────
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

    def _assess_crew_continuity(
        self,
        emergent_activity: JsonDict,
        insertion_point: JsonDict,
        schedule_network: _ScheduleNetwork,
    ) -> JsonDict:
        """Assess shift-boundary, fatigue, and background-utilization risks.

        Unlike ``_check_resource_conflicts()`` — which answers the binary
        question "do we have enough crew?" — this method quantifies *how tight*
        the resource situation is and whether the insertion window crosses shift
        boundaries or extends into off-shift hours.

        Three sub-assessments are produced:

        Shift calendar:
            ``off_shift_overlap_hours`` — hours of the insertion window that
            fall outside the active shift.  Zero on 24/7 schedules.
            ``shift_boundary_conflict`` — True when a shift-start event
            (crew handover) falls inside the window; mid-job handovers
            require explicit work-package transfer.
            ``fatigue_risk`` — True when ``off_shift_overlap_hours`` exceeds
            ``config.fatigue_risk_off_shift_hours``.

        Background utilization (per skill type in the crew pool):
            For each skill: available crew (minimum over window) vs. the
            number of workers already committed to other tasks in that window
            (derived from the infoDict ES/EF).  Flags skills above the
            ``config.high_crew_utilization_threshold`` fraction.

        Returns a dict suitable for inclusion in the Stage E artifact as the
        ``crew_continuity`` key.  When no crew pool is attached to the Pert
        (e.g. CPM-only stub schedules) the method returns a minimal dict with
        ``available: false`` rather than raising.
        """
        pert = schedule_network.pert

        proposed_start_iso = insertion_point.get("proposed_start")
        proposed_finish_iso = insertion_point.get("proposed_finish")
        if not proposed_start_iso or not proposed_finish_iso:
            return {"available": False, "reason": "insertion window datetimes not set"}

        start_dt = _parse_dt(proposed_start_iso)
        end_dt = _parse_dt(proposed_finish_iso)
        if start_dt is None or end_dt is None or end_dt <= start_dt:
            return {"available": False, "reason": "could not parse insertion window datetimes"}

        shift_start_hour: int = getattr(pert, "shift_start_hour", 0)
        working_hours_per_day: int = getattr(pert, "working_hours_per_day", 24)

        # ── Shift calendar analysis ───────────────────────────────────────────
        off_shift_hours = _off_shift_overlap_hours(
            start_dt, end_dt, shift_start_hour, working_hours_per_day
        )
        boundary_conflict = _has_shift_boundary(
            start_dt, end_dt, shift_start_hour, working_hours_per_day
        )
        fatigue_risk = off_shift_hours > self.config.fatigue_risk_off_shift_hours

        # ── Background utilization ────────────────────────────────────────────
        crew_pool = getattr(pert, "crew_pool", None)
        utilization_by_skill: JsonDict = {}
        peak_skill: Optional[str] = None
        peak_pct: float = 0.0

        if crew_pool is not None and hasattr(crew_pool, "resources"):
            # Hours from project start that correspond to the insertion window
            base_dt: Optional[datetime] = None
            if hasattr(pert, "startTime") and pert.startTime:
                base_dt = _ensure_tz(pert.startTime)

            win_start_offset: Optional[float] = None
            win_end_offset: Optional[float] = None
            if base_dt is not None:
                win_start_offset = (start_dt - base_dt).total_seconds() / 3600.0
                win_end_offset = (end_dt - base_dt).total_seconds() / 3600.0

            # Committed crew per skill: tasks active during the insertion window
            committed_by_skill: Dict[str, int] = {}
            if win_start_offset is not None and win_end_offset is not None:
                for act, info in pert.infoDict.items():
                    es = info.get("es", 0.0)
                    ef = info.get("ef", 0.0)
                    # Overlap: task's [ES, EF) intersects window [win_start, win_end)
                    if es < win_end_offset and ef > win_start_offset:
                        for req in getattr(act, "required_resources", []):
                            skill = req.get("skill_type")
                            count = int(req.get("crew_count") or 0)
                            if skill:
                                committed_by_skill[skill] = (
                                    committed_by_skill.get(skill, 0) + count
                                )

            for skill in sorted(crew_pool.resources):
                try:
                    available = crew_pool.get_availability_in_range(
                        skill, start_dt, end_dt
                    )
                except Exception:
                    LOGGER.debug(
                        "Stage E crew_continuity: pool query failed for skill=%s", skill
                    )
                    continue

                committed = committed_by_skill.get(skill, 0)
                free = max(0, available - committed)
                util_pct = (
                    round(min(100.0, committed / available * 100.0), 1)
                    if available > 0 else 100.0
                )
                high_util = (
                    util_pct / 100.0 >= self.config.high_crew_utilization_threshold
                )
                utilization_by_skill[skill] = {
                    "available": available,
                    "committed": committed,
                    "free": free,
                    "utilization_pct": util_pct,
                    "high_utilization": high_util,
                }
                if util_pct > peak_pct:
                    peak_pct = util_pct
                    peak_skill = skill

        # ── Notes ─────────────────────────────────────────────────────────────
        notes: List[str] = []
        if boundary_conflict:
            notes.append(
                "Insertion window spans a shift handover; "
                "work package transfer to incoming crew must be planned."
            )
        if fatigue_risk:
            notes.append(
                f"Insertion window extends {off_shift_hours:.1f} h into off-shift hours "
                f"(threshold {self.config.fatigue_risk_off_shift_hours:.1f} h); "
                "verify crew availability or plan for overtime authorisation."
            )
        high_util_skills = [s for s, d in utilization_by_skill.items() if d["high_utilization"]]
        if high_util_skills:
            notes.append(
                "High crew utilization at insertion window for: "
                + ", ".join(high_util_skills)
                + f" (threshold ≥{self.config.high_crew_utilization_threshold * 100:.0f}%)."
            )

        return {
            "available": True,
            "off_shift_overlap_hours": round(off_shift_hours, 2),
            "shift_boundary_conflict": boundary_conflict,
            "fatigue_risk": fatigue_risk,
            "utilization_at_window": utilization_by_skill,
            "peak_utilization_skill": peak_skill,
            "peak_utilization_pct": round(peak_pct, 1),
            "notes": notes,
        }

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

def _off_shift_overlap_hours(
    start: datetime,
    end: datetime,
    shift_start_hour: int,
    working_hours_per_day: int,
) -> float:
    """Return the number of hours in [start, end) that fall outside the active shift.

    On a 24/7 schedule (``working_hours_per_day >= 24``) this is always 0.
    For partial-day schedules the shift runs daily from ``shift_start_hour``
    for ``working_hours_per_day`` hours; hours outside that window are counted.

    Scans in 1-hour increments from the first whole hour >= start to end,
    weighting each segment by the fraction that overlaps [start, end).
    This is straightforward and correct up to the nearest minute; a more
    sophisticated closed-form approach is not needed for this application.
    """
    if working_hours_per_day >= 24 or end <= start:
        return 0.0

    shift_end_hour = (shift_start_hour + working_hours_per_day) % 24

    def _in_shift(h: int) -> bool:
        if shift_start_hour < shift_end_hour:
            return shift_start_hour <= h < shift_end_hour
        # Midnight-crossing shift
        return h >= shift_start_hour or h < shift_end_hour

    total = 0.0
    # Snap back to the start of the hour containing `start`
    cursor = start.replace(minute=0, second=0, microsecond=0)
    while cursor < end:
        seg_start = max(cursor, start)
        seg_end = min(cursor + timedelta(hours=1), end)
        if not _in_shift(cursor.hour):
            total += (seg_end - seg_start).total_seconds() / 3600.0
        cursor += timedelta(hours=1)
    return total


def _has_shift_boundary(
    start: datetime,
    end: datetime,
    shift_start_hour: int,
    working_hours_per_day: int,
) -> bool:
    """Return True if a shift-start event falls strictly inside (start, end).

    On 24/7 schedules there are no shift boundaries; returns False.
    A shift-start event occurs once per day at ``shift_start_hour``; crossing
    this boundary mid-job requires a formal work-package handover.
    """
    if working_hours_per_day >= 24 or end <= start:
        return False

    # Walk day-by-day from the day containing `start`
    candidate = start.replace(
        hour=shift_start_hour, minute=0, second=0, microsecond=0
    )
    # Start from the current day; advance to the next occurrence if it's <= start
    if candidate <= start:
        candidate += timedelta(days=1)
    return candidate < end


def _has_regulatory_constraint(text: Optional[str]) -> bool:
    """Return True if *text* contains regulatory-constraint keywords.

    Mirrors the pattern set in stage_a_intake so that displaced tasks and
    other description-bearing artifacts can be enriched without a KG lookup.
    """
    if not text:
        return False
    return bool(_REGULATORY_KEYWORDS_RE.search(text))


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


def _shift_insertion_times(
    insertion_point: JsonDict,
    lead_hours: float,
) -> JsonDict:
    """Return a new insertion_point dict with proposed_start and proposed_finish
    shifted forward by ``lead_hours``.

    If either ISO field is absent or unparseable the original value is kept
    unchanged (graceful degradation — the shift is best-effort).
    """
    if lead_hours <= 0:
        return insertion_point

    result = dict(insertion_point)
    delta = timedelta(hours=lead_hours)

    for key in ("proposed_start", "proposed_finish"):
        iso = insertion_point.get(key)
        if not iso:
            continue
        dt = _parse_dt(iso)
        if dt is not None:
            result[key] = (dt + delta).isoformat()

    return result


def _compute_permit_lead_time(
    execution_mode_flags: Dict[str, Any],
    *,
    rp_hold_hours: float,
    scaffold_hours: float,
    clearance_hours: float,
    mode: str = "max",
) -> JsonDict:
    """Compute the permit/approval lead time for an emergent activity.

    The lead time is the calendar overhead that must elapse *before* the
    activity can start — permits and approvals happen before the first wrench
    turn, not concurrently with the work.

    Args:
        execution_mode_flags: Dict with boolean keys ``has_rp_hold``,
            ``requires_scaffold``, ``has_clearance`` (all default False when
            absent — safe to pass an empty dict).
        rp_hold_hours: RP survey + ALARA briefing lead time (from config).
        scaffold_hours: Scaffold erection + inspection lead time (from config).
        clearance_hours: E/M LOTO procedure lead time (from config).
        mode: ``"max"`` (parallel permits, longest drives delay) or
              ``"sum"`` (sequential permits, all times added).

    Returns:
        Dict with keys:
            total_lead_hours        – effective delay before work can start
            rp_hold_hours           – contribution from RP hold (0 if inactive)
            scaffold_hours          – contribution from scaffold (0 if inactive)
            clearance_hours         – contribution from clearance (0 if inactive)
            start_adjusted          – True when total_lead_hours > 0
            combination_mode        – ``"max"`` or ``"sum"``
            notes                   – list of human-readable strings
    """
    flags = execution_mode_flags or {}
    rp_component    = rp_hold_hours  if flags.get("has_rp_hold")       else 0.0
    scaf_component  = scaffold_hours if flags.get("requires_scaffold")  else 0.0
    clear_component = clearance_hours if flags.get("has_clearance")     else 0.0

    components = [rp_component, scaf_component, clear_component]
    if mode == "sum":
        total = sum(components)
    else:  # "max" (default)
        total = max(components)

    notes: List[str] = []
    if rp_component > 0:
        notes.append(
            f"RP hold: {rp_component:.1f} h survey/briefing lead time required."
        )
    if scaf_component > 0:
        notes.append(
            f"Scaffold: {scaf_component:.1f} h erection/inspection lead time required."
        )
    if clear_component > 0:
        notes.append(
            f"Clearance/LOTO: {clear_component:.1f} h establishment lead time required."
        )
    if total > 0 and mode == "max":
        notes.append(
            f"Permits processed in parallel; critical path lead time = {total:.1f} h."
        )
    elif total > 0 and mode == "sum":
        notes.append(
            f"Permits processed sequentially; total lead time = {total:.1f} h."
        )

    return {
        "total_lead_hours": round(total, 2),
        "rp_hold_hours": round(rp_component, 2),
        "scaffold_hours": round(scaf_component, 2),
        "clearance_hours": round(clear_component, 2),
        "start_adjusted": total > 0,
        "combination_mode": mode,
        "notes": notes,
    }



