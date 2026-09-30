"""
Schedule network builder — bridges ``OutageRecord`` + Stage D estimates
into a ``ScheduleNetwork`` ready for ``MonteCarloSimulator``.

This is the ``schedule_builder`` implementation expected by
:class:`~outage_uncertainty.workflows.outage_risk_workflow.OutageRiskWorkflow`.

Conversion logic
----------------
Each :class:`~outage_uncertainty.domain.activity.ActivityCase` in the outage
becomes a :class:`~outage_uncertainty.domain.schedule.ScheduleActivity`:

* ``activity_id``, ``name``, ``predecessors``, ``successors`` — direct copy
* ``baseline_duration_hours`` — from ``ActivityCase.planned_duration_hours``
  (``0.0`` when ``None``; a DEBUG warning is emitted)
* ``duration_distribution`` — from the matching ``ActivityEstimate``'s
  ``estimated_distribution``, if one was produced by Stage D.  When absent
  (activity not estimated, or low-confidence fallback), the field is ``None``
  so ``MonteCarloSimulator`` treats the activity as deterministic (uses the
  baseline duration every run).

Dangling-edge handling
----------------------
``ScheduleNetwork._validate()`` raises ``ValueError`` when a successor or
predecessor ID is not present in the network.  This is common in practice:

* Milestone tasks may have been skipped during P6 ingestion.
* External activities (cross-outage logic, summary tasks) may not be included.
* The outage slice loaded for UQ may be a sub-network of the full P6 schedule.

By default (``prune_dangling_edges=True``) the builder silently removes
references to activities that are not in the network and logs each pruned edge
at DEBUG level.  Set ``prune_dangling_edges=False`` to get the hard
``ValueError`` from ``ScheduleNetwork`` instead — useful in tests that
intentionally verify full graph integrity.

Baseline CP time
----------------
After building the network the builder runs CPM once with every activity set to
its ``baseline_duration_hours``.  The resulting ``cp_time`` is returned as the
second element of the tuple and used by ``ScenarioRunner`` / ``CriticalPathRiskAnalyzer``
as the planned-schedule reference point for robustness and expected-delay
calculations.
"""
from __future__ import annotations

import logging

from outage_uncertainty.domain.activity import ActivityCase
from outage_uncertainty.domain.outage import OutageRecord
from outage_uncertainty.domain.result_types import ActivityEstimate
from outage_uncertainty.domain.schedule import ScheduleActivity
from outage_uncertainty.schedule_risk.schedule_graph import ScheduleNetwork

logger = logging.getLogger(__name__)


class OutageRecordScheduleBuilder:
    """Convert an ``OutageRecord`` + Stage D estimates into a ``ScheduleNetwork``.

    Implements the ``schedule_builder`` protocol expected by
    :class:`~outage_uncertainty.workflows.outage_risk_workflow.OutageRiskWorkflow`.

    Parameters
    ----------
    prune_dangling_edges:
        When ``True`` (default), predecessor/successor IDs that do not resolve
        to a known activity in the outage are silently removed before the
        ``ScheduleNetwork`` is constructed.  This prevents ``ValueError`` from
        activities that reference milestones or external tasks not included in
        the activity list.

        When ``False``, ``ScheduleNetwork._validate()`` raises ``ValueError``
        on the first dangling reference — useful for strict validation in tests.
    """

    def __init__(self, *, prune_dangling_edges: bool = True) -> None:
        self.prune_dangling_edges = prune_dangling_edges

    # ------------------------------------------------------------------
    # Public interface — matches schedule_builder protocol
    # ------------------------------------------------------------------

    def build(
        self,
        planned_outage: OutageRecord,
        estimates: dict[str, ActivityEstimate],
        *,
        lags: dict[tuple[str, str], float] | None = None,
    ) -> tuple[ScheduleNetwork, float]:
        """Build a ``ScheduleNetwork`` and compute the baseline CP time.

        Parameters
        ----------
        planned_outage:
            The outage whose activities form the schedule network.
        estimates:
            ``{activity_id: ActivityEstimate}`` produced by Stage D.  Activities
            not present in this dict are treated as deterministic (their
            ``baseline_duration_hours`` is used as a constant every MC run).
        lags:
            Optional ``{(predecessor_id, successor_id): lag_hours}`` dict of
            finish-to-start lags, e.g. built from the P6 ``Dependency`` table.
            When ``None`` (default) all lags are treated as zero, which is
            correct for schedules where activities link with pure FS(0)
            dependencies.  Note: ``ActivityCase`` does not currently carry lag
            information, so callers that have lag data must supply it here
            directly (e.g. by extracting it from ``OutageDataset.dependencies``
            via ``P6DatasetAdapter``).

        Returns
        -------
        network:
            ``ScheduleNetwork`` ready for ``MonteCarloSimulator``.
        baseline_cp_time:
            CPM project duration computed with every activity at its planned
            duration — the reference point for robustness calculations.
        """
        activities = planned_outage.activities

        if not activities:
            logger.warning(
                "OutageRecordScheduleBuilder: outage '%s' has no activities; "
                "returning empty network with baseline_cp_time=0.0",
                planned_outage.outage_id,
            )
            return ScheduleNetwork([]), 0.0

        known_ids: set[str] = {a.activity_id for a in activities}
        schedule_activities = [
            self._to_schedule_activity(case, estimates, known_ids)
            for case in activities
        ]

        n_no_duration = sum(
            1 for sa in schedule_activities if sa.baseline_duration_hours == 0.0
        )
        if n_no_duration:
            logger.debug(
                "OutageRecordScheduleBuilder: %d / %d activities have "
                "baseline_duration_hours = 0.0 (planned_duration_hours was None)",
                n_no_duration,
                len(schedule_activities),
            )

        n_no_dist = sum(
            1 for sa in schedule_activities if sa.duration_distribution is None
        )
        if n_no_dist:
            logger.debug(
                "OutageRecordScheduleBuilder: %d / %d activities have no "
                "duration_distribution — MonteCarloSimulator will treat them "
                "as deterministic (baseline duration used every run)",
                n_no_dist,
                len(schedule_activities),
            )

        network = ScheduleNetwork(schedule_activities, lags=lags)

        # Baseline CP time: one CPM pass with every activity at its planned duration
        baseline_durations = {
            sa.activity_id: sa.baseline_duration_hours
            for sa in schedule_activities
        }
        baseline_result = network.compute_critical_path(baseline_durations)
        baseline_cp_time: float = baseline_result["cp_time"]

        logger.debug(
            "OutageRecordScheduleBuilder: built network with %d activities, "
            "baseline CP time = %.1f h, critical path length = %d activities",
            len(schedule_activities),
            baseline_cp_time,
            len(baseline_result["cp_path"]),
        )

        return network, baseline_cp_time

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_schedule_activity(
        self,
        case: ActivityCase,
        estimates: dict[str, ActivityEstimate],
        known_ids: set[str],
    ) -> ScheduleActivity:
        """Convert one ``ActivityCase`` to a ``ScheduleActivity``.

        The duration distribution comes from the Stage D estimate when present.
        Edge lists are pruned of dangling references when
        ``self.prune_dangling_edges`` is ``True``.
        """
        baseline = case.planned_duration_hours
        if baseline is None:
            logger.debug(
                "OutageRecordScheduleBuilder: activity '%s' has no "
                "planned_duration_hours; using 0.0 as baseline",
                case.activity_id,
            )
            baseline = 0.0

        estimate = estimates.get(case.activity_id)
        dist = estimate.estimated_distribution if estimate is not None else None

        predecessors = self._prune_edges(
            case.activity_id, "predecessor", case.predecessor_ids, known_ids
        )
        successors = self._prune_edges(
            case.activity_id, "successor", case.successor_ids, known_ids
        )

        return ScheduleActivity(
            activity_id=case.activity_id,
            name=case.cleaned_description or case.raw_description or case.activity_id,
            predecessors=predecessors,
            successors=successors,
            baseline_duration_hours=float(baseline),
            duration_distribution=dist,
        )

    def _prune_edges(
        self,
        activity_id: str,
        edge_type: str,
        edge_ids: list[str],
        known_ids: set[str],
    ) -> list[str]:
        """Return *edge_ids* filtered to only known activities.

        When ``prune_dangling_edges`` is ``False`` all IDs are returned as-is
        (``ScheduleNetwork._validate()`` will catch any unknowns).
        """
        if not self.prune_dangling_edges:
            return list(edge_ids)

        pruned = []
        for eid in edge_ids:
            if eid in known_ids:
                pruned.append(eid)
            else:
                logger.debug(
                    "OutageRecordScheduleBuilder: activity '%s' references "
                    "unknown %s '%s' — pruned (not in outage activity list)",
                    activity_id,
                    edge_type,
                    eid,
                )
        return pruned
