"""
Schedule network — lightweight CPM engine aligned with LOGOS ``Pert.generateInfo()``.

Supported relationship type: **FS with non-negative lag only.**
SS / FF / SF relationships and negative lags (start-lead overlap) are not
modelled; if the input data contains them they should be converted to FS
equivalents before construction.

Critical-path algorithm
-----------------------
The implementation is topological-order CPM, matching the LOGOS formulas
exactly so that ``ScheduleNetwork`` and LOGOS ``Pert`` agree on ES / EF / LS /
LF / float for the same input data.

Forward pass
~~~~~~~~~~~~
For each activity *v* in topological order::

    ES[v] = max(EF[p] + lag(p→v) for p in predecessors(v), default 0) + lead[v]
    EF[v] = ES[v] + sampled_duration[v]

where ``lag(p→v)`` is the FS lag on the edge (defaults to 0) and ``lead[v]``
is ``ScheduleActivity.mobilization_lead_hours`` (defaults to 0).

Source activities (no predecessors) start at ``ES = lead[v]`` — the default
of 0 reproduces the original zero-lag, zero-lead behaviour unchanged.

Backward pass
~~~~~~~~~~~~~
Project duration ``T = max(EF)`` across all sink activities::

    LF[v] = T  for every sink
    LF[u] = min(LF[v] - dur[v] - lag(u→v) - lead[v]  for all successors v of u)
    LS[u] = LF[u] - dur[u]

Critical path
~~~~~~~~~~~~~
Activities with zero total float: ``abs(LF[v] - EF[v]) < 1e-9``.

Lag validation
~~~~~~~~~~~~~~
``_validate_lags()`` checks that all lag values are finite and non-negative.
Lags for edges that do not appear in the network are silently ignored (they
contribute nothing to the CPM calculation and arise naturally when lag dicts
are built from a full P6 dataset that has been sub-sampled for UQ).
"""
from __future__ import annotations

import math
from collections import deque

from outage_uncertainty.domain.schedule import ScheduleActivity


class ScheduleNetwork:
    """Directed acyclic schedule network with FS-lag and mobilization-lead CPM.

    Parameters
    ----------
    activities:
        All activities in the network.  Every ID referenced in
        ``ScheduleActivity.predecessors`` or ``.successors`` must be present;
        ``_validate()`` raises ``ValueError`` otherwise.
    lags:
        Optional ``{(predecessor_id, successor_id): lag_hours}`` dict for
        finish-to-start lags.  Missing pairs default to 0.0.  Negative values
        raise ``ValueError`` (use ``mobilization_lead_hours`` on the
        ``ScheduleActivity`` for advance-notice delays instead).
    """

    def __init__(
        self,
        activities: list[ScheduleActivity],
        lags: dict[tuple[str, str], float] | None = None,
    ) -> None:
        self.activities: dict[str, ScheduleActivity] = {
            a.activity_id: a for a in activities
        }
        self._lags: dict[tuple[str, str], float] = dict(lags or {})
        self._validate()
        self._validate_lags()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Raise ValueError if any successor/predecessor reference is not in the network."""
        known = set(self.activities)
        for act in self.activities.values():
            for s in act.successors:
                if s not in known:
                    raise ValueError(
                        f"Activity '{act.activity_id}' references unknown successor '{s}'"
                    )
            for p in act.predecessors:
                if p not in known:
                    raise ValueError(
                        f"Activity '{act.activity_id}' references unknown predecessor '{p}'"
                    )

    def _validate_lags(self) -> None:
        """Raise ValueError if any lag value is negative or non-finite."""
        for (u, v), lag in self._lags.items():
            if not math.isfinite(lag):
                raise ValueError(
                    f"Lag for ({u!r} → {v!r}) is non-finite ({lag}); "
                    "all lags must be finite non-negative values."
                )
            if lag < 0.0:
                raise ValueError(
                    f"Lag for ({u!r} → {v!r}) is negative ({lag}); "
                    "negative lags are not supported. "
                    "Use ScheduleActivity.mobilization_lead_hours for advance-notice delays."
                )

    # ------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------

    def topological_sort(self) -> list[str]:
        indegree = {aid: 0 for aid in self.activities}
        for activity in self.activities.values():
            for successor in activity.successors:
                indegree[successor] = indegree.get(successor, 0) + 1

        queue = deque([aid for aid, deg in indegree.items() if deg == 0])
        order: list[str] = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for successor in self.activities[current].successors:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    queue.append(successor)

        if len(order) != len(self.activities):
            raise ValueError(
                f"Schedule network contains a cycle: only {len(order)} of "
                f"{len(self.activities)} activities could be ordered"
            )

        return order

    # ------------------------------------------------------------------
    # Critical-path calculation
    # ------------------------------------------------------------------

    def compute_critical_path(self, sampled_durations: dict[str, float]) -> dict:
        """Return CP time and critical-path activity list for one duration sample.

        Parameters
        ----------
        sampled_durations:
            ``{activity_id: duration_hours}`` for this simulation run.
            Activities absent from the dict are treated as zero-duration.

        Returns
        -------
        dict with keys:
            ``"cp_time"``  — project duration (max EF across all activities)
            ``"cp_path"``  — ordered list of activity IDs with zero total float
        """
        order = self.topological_sort()
        predecessors_map = {aid: set(act.predecessors) for aid, act in self.activities.items()}
        successors_map   = {aid: set(act.successors)   for aid, act in self.activities.items()}

        # ── Forward pass: ES / EF ────────────────────────────────────────
        ef: dict[str, float] = {}
        for activity_id in order:
            act  = self.activities[activity_id]
            preds = predecessors_map.get(activity_id, set())
            lead  = act.mobilization_lead_hours

            if preds:
                # ES = max(EF[p] + lag(p→v)) + lead[v]
                # lag(p→v) is FS lag; missing key defaults to 0.
                es = max(
                    ef[p] + self._lags.get((p, activity_id), 0.0)
                    for p in preds
                ) + lead
            else:
                # Source activity: starts at t = lead (0 when no mobilization)
                es = lead

            ef[activity_id] = es + sampled_durations.get(activity_id, 0.0)

        cp_time = max(ef.values(), default=0.0)

        # ── Backward pass: LF / LS ───────────────────────────────────────
        lf: dict[str, float] = {}
        for activity_id in reversed(order):
            succs = successors_map.get(activity_id, set())
            if not succs:
                lf[activity_id] = cp_time
            else:
                # LF[u] = min over successors v of:
                #   LF[v] - dur[v] - lag(u→v) - lead[v]
                # = LS[v] - lag(u→v) - lead[v]
                lf[activity_id] = min(
                    lf[s]
                    - sampled_durations.get(s, 0.0)
                    - self._lags.get((activity_id, s), 0.0)
                    - self.activities[s].mobilization_lead_hours
                    for s in succs
                )

        # ── Critical path: zero total float ──────────────────────────────
        # Total float = LF - EF  (equivalent to LS - ES)
        cp_path = [aid for aid in order if abs(lf[aid] - ef[aid]) < 1e-9]
        return {"cp_time": cp_time, "cp_path": cp_path}
