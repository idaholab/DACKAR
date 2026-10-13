from __future__ import annotations

from outage_uncertainty.domain.result_types import SimulationResult
from outage_uncertainty.schedule_risk.schedule_graph import ScheduleNetwork


class MonteCarloSimulator:
    def __init__(self, schedule_network: ScheduleNetwork, n_samples: int = 1000):
        self.schedule_network = schedule_network
        self.n_samples = n_samples

    def run(self) -> SimulationResult:
        cp_times: list[float] = []
        cp_paths: list[list[str]] = []
        activity_criticality: dict[str, int] = {}

        for _ in range(self.n_samples):
            sampled = {}
            for activity_id, activity in self.schedule_network.activities.items():
                if activity.duration_distribution is None:
                    sampled[activity_id] = activity.baseline_duration_hours
                else:
                    sampled[activity_id] = activity.duration_distribution.sample(1)[0]

            result = self.schedule_network.compute_critical_path(sampled)
            cp_times.append(result["cp_time"])
            cp_paths.append(result["cp_path"])

            for activity_id in result["cp_path"]:
                activity_criticality[activity_id] = activity_criticality.get(activity_id, 0) + 1

        return SimulationResult(
            cp_times=cp_times,
            cp_paths=cp_paths,
            activity_criticality=activity_criticality,
        )
