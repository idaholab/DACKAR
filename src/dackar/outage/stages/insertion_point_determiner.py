"""
Insertion point determination pre-pass for the two-pass analog retrieval design.

Background
----------
Stage D (analog retrieval) runs before Stage E (schedule impact) because Stage E
needs the duration distribution produced by Stage D to drive its Monte Carlo
simulation.  This ordering means Stage D cannot use schedule-position signals
during similarity scoring — it has no knowledge of *where* the emergent activity
will sit in the network.

The key insight is that **determining the insertion point does not require the
duration distribution**.  It only needs:

    1. The schedule network (LOGOS Pert, loadable independently).
    2. The emergent activity's ``actual_start`` / ``outage_phase`` metadata.

``InsertionPointDeterminer`` executes this lightweight pre-pass between Stage C
and Stage D, producing a ``ScheduleContext`` that carries:

    - Insertion topology: ``after_task_id``, ``before_task_id``, degree metrics.
    - Schedule risk indicators: ``available_float_hours``, ``insertion_on_cp``.

This context is then threaded through both Stage D and Stage E:

    Stage D:  ``_rerank_by_schedule_context()`` uses the signals to apply a
              structural affinity boost to the ranked analog pool (Layer 1).
              The query ``ActivityCase`` also receives ``predecessor_ids`` /
              ``successor_ids`` so ``DependencyPatternScorer`` activates
              automatically once historical topology data enters the index
              (Layer 2, deferred).

    Stage E:  ``ScheduleContext.to_insertion_point()`` provides the pre-computed
              insertion point dict, avoiding a redundant second determination.

Typical wiring in the orchestrator::

    from stages.insertion_point_determiner import InsertionPointDeterminer

    determiner = InsertionPointDeterminer(
        schedule_loader=my_loader,
        schedule_graph_builder=my_builder,
        schedule_version="working",
    )
    # Called between Stage C and Stage D:
    schedule_context = determiner.determine(emergent_activity, intake_result)

Graceful degradation
--------------------
All failures in ``determine()`` are caught and logged; the method returns
``None`` rather than raising.  Stage D and Stage E fall back to their existing
behavior when ``schedule_context`` is ``None``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)

JsonDict = Dict[str, Any]

# Minimum slack to be considered off the critical path (hours).
_FLOAT_EPSILON = 0.01


# ---------------------------------------------------------------------------
# ScheduleContext dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScheduleContext:
    """Immutable snapshot of the insertion site's schedule topology.

    Produced by :class:`InsertionPointDeterminer` and consumed by Stage D
    (structural affinity re-ranking) and Stage E (insertion point reuse).

    Attributes
    ----------
    activity_id
        The emergent activity this context was computed for.
    after_task_id
        The schedule task immediately before the insertion point (the activity
        the emergent task will follow).  ``None`` when the schedule is empty or
        no suitable predecessor was found.
    before_task_id
        The schedule task immediately after the insertion point (the first
        successor of ``after_task_id`` on the critical path).  ``None`` when
        ``after_task_id`` is a sink.
    outage_phase
        Outage phase label at the insertion site (from intake result or
        inferred from schedule position).
    proposed_start
        ISO-8601 datetime of the proposed insertion start, derived from the
        schedule network's project start + ``after_task`` EF.
    proposed_finish
        ISO-8601 datetime of the proposed insertion finish.  Uses
        ``planned_duration_hours`` from the emergent activity as the best
        available pre-analog estimate.  Stage D will overwrite this with the
        p50 distribution estimate when available.
    insertion_in_degree
        Number of predecessors of ``after_task_id`` in the schedule network.
        High in-degree signals a merge point (multiple work streams converging)
        where coordination overhead is elevated.
    insertion_out_degree
        Number of successors of ``after_task_id`` in the schedule network.
        High out-degree signals a burst point (work-package fan-out) where the
        insertion blocks multiple downstream tasks simultaneously.
    available_float_hours
        Total float (slack) at ``after_task_id`` before insertion.  Small
        values indicate the insertion site is near-critical; zero means
        any delay propagates directly to the outage completion date.
    insertion_on_cp
        True when ``available_float_hours`` ≤ ``_FLOAT_EPSILON`` — i.e. the
        insertion site is on or adjacent to the critical path.
    """

    activity_id: str
    after_task_id: Optional[str]
    before_task_id: Optional[str]
    outage_phase: str
    proposed_start: Optional[str]
    proposed_finish: Optional[str]
    insertion_in_degree: int
    insertion_out_degree: int
    available_float_hours: float
    insertion_on_cp: bool

    def to_insertion_point(self) -> JsonDict:
        """Return an insertion_point dict compatible with Stage E's artifact schema.

        Stage E accepts this in place of the output of
        ``_determine_insertion_point()`` so the schedule network does not need
        to be traversed a second time.
        """
        return {
            "emergent_task_id": f"EA::{self.activity_id}",
            "after_task_id": self.after_task_id,
            "before_task_id": self.before_task_id,
            "outage_phase": self.outage_phase,
            "proposed_start": self.proposed_start,
            "proposed_finish": self.proposed_finish,
        }


# ---------------------------------------------------------------------------
# InsertionPointDeterminer
# ---------------------------------------------------------------------------

class InsertionPointDeterminer:
    """Lightweight pre-pass that determines where the emergent activity inserts.

    Loads the schedule network (same loader / builder pair used by Stage E),
    finds the insertion point, and extracts topology metrics — all without
    needing the duration distribution from Stage D.

    Args:
        schedule_loader: ``callable(outage_id, version=...) → OutageData``.
            The same loader injected into ``ScheduleImpactAssessor``.
        schedule_graph_builder: Object with ``.build(outage_data) → Pert``.
            The builder must return a Pert with ``generateInfo()`` called so
            ``infoDict`` is populated.
        schedule_version: Schedule version tag to load.  Defaults to
            ``"working"`` (same default as Stage E).
        near_critical_float_threshold_hours: Float below which a site is
            considered near-critical.  Used only for logging; the binary
            ``insertion_on_cp`` uses ``_FLOAT_EPSILON``.
    """

    def __init__(
        self,
        schedule_loader=None,
        schedule_graph_builder=None,
        schedule_version: str = "working",
        near_critical_float_threshold_hours: float = 8.0,
    ) -> None:
        self.schedule_loader = schedule_loader
        self.schedule_graph_builder = schedule_graph_builder
        self.schedule_version = schedule_version
        self.near_critical_float_threshold_hours = near_critical_float_threshold_hours

    # ── Public entry point ────────────────────────────────────────────────────

    def determine(
        self,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
    ) -> Optional["ScheduleContext"]:
        """Determine the insertion point and return a ScheduleContext.

        Returns ``None`` on any failure (missing loader, schedule load error,
        empty schedule) so callers can safely treat ``None`` as "no context
        available — fall back to existing behavior."

        Args:
            emergent_activity: Raw EmergentActivity artifact (must include
                ``activity_id`` and ``outage_id``).
            intake_result: Stage A output (used for ``outage_phase``).
        """
        if self.schedule_loader is None or self.schedule_graph_builder is None:
            LOGGER.debug(
                "InsertionPointDeterminer: loader/builder not injected; skipping"
            )
            return None

        activity_id: str = emergent_activity.get("activity_id", "")
        outage_id: str = emergent_activity.get("outage_id", "")

        try:
            outage_data = self.schedule_loader(
                outage_id, version=self.schedule_version
            )
            pert = self.schedule_graph_builder.build(outage_data)
            pert.generateInfo()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "InsertionPointDeterminer: schedule load failed for outage %s: %s",
                outage_id, exc,
            )
            return None

        if not getattr(pert, "infoDict", None):
            LOGGER.debug(
                "InsertionPointDeterminer: empty schedule for outage %s", outage_id
            )
            return None

        try:
            return self._build_context(
                pert=pert,
                emergent_activity=emergent_activity,
                intake_result=intake_result,
                activity_id=activity_id,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "InsertionPointDeterminer: context extraction failed: %s", exc
            )
            return None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_context(
        self,
        pert: Any,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        activity_id: str,
    ) -> "ScheduleContext":
        outage_phase: str = (
            intake_result.get("outage_phase")
            or emergent_activity.get("outage_phase")
            or "unknown"
        )

        after_act = self._find_after_task(pert, emergent_activity, outage_phase)

        info = pert.infoDict
        fwd = getattr(pert, "forwardDict", {})
        bwd = getattr(pert, "backwardDict", {})

        # Successor / predecessor of after_task
        successors = fwd.get(after_act, []) if after_act is not None else []
        before_act = successors[0] if successors else None

        # Topology metrics at insertion site
        in_degree = len(bwd.get(after_act, [])) if after_act is not None else 0
        out_degree = len(successors)

        # Float at insertion site
        available_float: float = 0.0
        if after_act is not None and after_act in info:
            available_float = float(info[after_act].get("slack") or 0.0)

        on_cp = available_float <= _FLOAT_EPSILON

        # Absolute proposed times
        proposed_start_iso: Optional[str] = None
        proposed_finish_iso: Optional[str] = None
        base_dt: Optional[datetime] = None
        if hasattr(pert, "startTime") and pert.startTime:
            base_dt = _ensure_tz(pert.startTime)

        if after_act is not None and after_act in info and base_dt is not None:
            ef = info[after_act].get("ef", 0.0)
            proposed_start_iso = (base_dt + timedelta(hours=ef)).isoformat()
            planned_dur = float(
                emergent_activity.get("planned_duration_hours") or 0.0
            )
            proposed_finish_iso = (
                base_dt + timedelta(hours=ef + planned_dur)
            ).isoformat()

        if on_cp:
            LOGGER.debug(
                "InsertionPointDeterminer: activity %s inserts on CP "
                "(after=%s, float=%.2f h)",
                activity_id,
                after_act.name if after_act is not None else None,
                available_float,
            )
        elif available_float < self.near_critical_float_threshold_hours:
            LOGGER.debug(
                "InsertionPointDeterminer: activity %s near-critical insertion "
                "(after=%s, float=%.2f h)",
                activity_id,
                after_act.name if after_act is not None else None,
                available_float,
            )

        return ScheduleContext(
            activity_id=activity_id,
            after_task_id=after_act.name if after_act is not None else None,
            before_task_id=before_act.name if before_act is not None else None,
            outage_phase=outage_phase,
            proposed_start=proposed_start_iso,
            proposed_finish=proposed_finish_iso,
            insertion_in_degree=in_degree,
            insertion_out_degree=out_degree,
            available_float_hours=round(available_float, 2),
            insertion_on_cp=on_cp,
        )

    def _find_after_task(
        self,
        pert: Any,
        emergent_activity: JsonDict,
        outage_phase: str,
    ) -> Optional[Any]:
        """Find the schedule task the emergent activity should follow.

        Mirrors Stage E's ``_determine_insertion_point()`` strategy:

        1. If ``actual_start`` is known, find the task active at that offset.
        2. Otherwise filter by ``outage_phase`` window and pick max-float task.
        """
        info = pert.infoDict
        if not info:
            return None

        candidates = [
            (act, d) for act, d in info.items()
            if d.get("es") is not None and d.get("ef") is not None
        ]
        if not candidates:
            return None

        # Strategy 1: steer by actual_start
        actual_start_iso: Optional[str] = emergent_activity.get("actual_start")
        base_dt: Optional[datetime] = None
        if hasattr(pert, "startTime") and pert.startTime:
            base_dt = _ensure_tz(pert.startTime)

        if actual_start_iso and base_dt is not None:
            try:
                start_dt = _parse_dt(actual_start_iso)
                if start_dt is not None:
                    offset = (start_dt - base_dt).total_seconds() / 3600.0
                    active = [
                        (a, d) for a, d in candidates
                        if d["es"] <= offset <= d["ef"]
                    ]
                    if active:
                        active.sort(key=lambda x: abs(x[1].get("slack", 999)))
                        return active[0][0]
            except (TypeError, ValueError):
                pass

        # Strategy 2: phase-window filter + max-float
        total_dur = max((d["ef"] for _, d in candidates), default=1.0)
        phase_low, phase_high = _default_phase_windows(outage_phase, total_dur)
        phase_cands = [
            (a, d) for a, d in candidates
            if phase_low <= d["es"] <= phase_high
        ]
        if not phase_cands:
            phase_cands = candidates
        phase_cands.sort(key=lambda x: x[1].get("slack", 0.0), reverse=True)
        return phase_cands[0][0]


# ---------------------------------------------------------------------------
# Module-level helpers (mirrors stage_e_schedule utilities)
# ---------------------------------------------------------------------------

def _parse_dt(iso_str: Optional[str]) -> Optional[datetime]:
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
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _default_phase_windows(
    outage_phase: str, total_hours: float
) -> tuple:
    """Map outage phase label to approximate (start_offset, end_offset) range."""
    _PHASE_MAP = {
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
