from __future__ import annotations

from collections import deque

from outage_uncertainty.domain.schedule import ScheduleActivity


class ScheduleNetwork:
    def __init__(self, activities: list[ScheduleActivity]):
        self.activities = {activity.activity_id: activity for activity in activities}
        self._validate()

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

    def compute_critical_path(self, sampled_durations: dict[str, float]) -> dict:
        order = self.topological_sort()
        predecessors_map = {aid: set(act.predecessors) for aid, act in self.activities.items()}
        successors_map = {aid: set(act.successors) for aid, act in self.activities.items()}

        # Forward pass — compute earliest finish for each activity
        ef: dict[str, float] = {}
        for activity_id in order:
            preds = predecessors_map.get(activity_id, set())
            es = max((ef[p] for p in preds), default=0.0)
            ef[activity_id] = es + sampled_durations.get(activity_id, 0.0)

        cp_time = max(ef.values(), default=0.0)

        # Backward pass — compute latest finish for each activity
        lf: dict[str, float] = {}
        for activity_id in reversed(order):
            succs = successors_map.get(activity_id, set())
            if not succs:
                lf[activity_id] = cp_time
            else:
                # LF[i] = min over successors j of (LF[j] - dur[j])
                lf[activity_id] = min(
                    lf[s] - sampled_durations.get(s, 0.0) for s in succs
                )

        # Critical path: zero total float (LF == EF)
        cp_path = [aid for aid in order if abs(lf[aid] - ef[aid]) < 1e-9]
        return {"cp_time": cp_time, "cp_path": cp_path}
