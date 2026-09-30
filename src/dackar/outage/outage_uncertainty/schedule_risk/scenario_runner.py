from __future__ import annotations

from outage_uncertainty.schedule_risk.cp_analyzer import CriticalPathRiskAnalyzer
from outage_uncertainty.schedule_risk.monte_carlo import MonteCarloSimulator
from outage_uncertainty.schedule_risk.schedule_graph import ScheduleNetwork


class ScenarioRunner:
    def __init__(self, simulator: MonteCarloSimulator | None = None, analyzer: CriticalPathRiskAnalyzer | None = None):
        self.simulator = simulator
        self.analyzer = analyzer or CriticalPathRiskAnalyzer()

    def run(self, network: ScheduleNetwork, baseline_cp_time: float, n_samples: int = 1000) -> dict:
        simulator = self.simulator or MonteCarloSimulator(network, n_samples=n_samples)
        sim_result = simulator.run()
        return self.analyzer.analyze(sim_result, baseline_cp_time=baseline_cp_time)
