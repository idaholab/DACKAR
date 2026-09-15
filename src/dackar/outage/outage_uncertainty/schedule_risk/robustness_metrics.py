"""
Schedule robustness and risk metrics computed from Monte Carlo simulation output.

This module provides a lightweight summary of project-level schedule risk,
suitable for reporting and dashboards.  For deeper per-activity diagnostics
(expected drag, CP sensitivity) use
:class:`~outage_uncertainty.schedule_risk.cp_analyzer.CriticalPathRiskAnalyzer`.

Metrics
-------
robustness
    Probability that the simulated project finish is at or within the
    baseline critical-path duration.  ``P(T_finish ≤ baseline_cp_time)``.

schedule_std_dev
    Standard deviation of the simulated project finish time.  Measures
    how dispersed the finish distribution is around its mean.

expected_delay
    ``max(0, E[T_finish] − baseline_cp_time)``.  Expected overrun in hours
    if the project is likely to finish late; zero when the schedule is
    expected to finish on time.

mean_finish
    Mean simulated project finish time (hours from schedule start).

p80_finish, p90_finish
    80th and 90th percentile of simulated finish time.  Use ``p90_finish``
    as a conservative schedule target when planning contingency.
"""
from __future__ import annotations

from outage_uncertainty.domain.result_types import SimulationResult


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std_dev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (equivalent to numpy's method 7)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    pos = (n - 1) * q
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


class RobustnessMetrics:
    """Compute project-level schedule robustness and risk metrics.

    All metrics are derived from a :class:`~outage_uncertainty.domain.result_types.SimulationResult`
    produced by :class:`~outage_uncertainty.schedule_risk.monte_carlo.MonteCarloSimulator`.
    """

    def compute(self, sim_result: SimulationResult, baseline_cp_time: float) -> dict:
        """Return a dict of schedule risk metrics.

        Args:
            sim_result: Output of ``MonteCarloSimulator.run()``.
            baseline_cp_time: Planned critical-path duration (hours from
                schedule start to scheduled finish).

        Returns:
            Dict with keys: ``robustness``, ``schedule_std_dev``,
            ``expected_delay``, ``mean_finish``, ``p80_finish``,
            ``p90_finish``.
        """
        cp_times = sim_result.cp_times
        if not cp_times:
            return {
                "robustness":      0.0,
                "schedule_std_dev": 0.0,
                "expected_delay":  0.0,
                "mean_finish":     0.0,
                "p80_finish":      0.0,
                "p90_finish":      0.0,
            }

        n = len(cp_times)
        robustness   = sum(t <= baseline_cp_time for t in cp_times) / n
        mean_finish  = _mean(cp_times)

        return {
            "robustness":       robustness,
            "schedule_std_dev": _std_dev(cp_times),
            "expected_delay":   max(0.0, mean_finish - baseline_cp_time),
            "mean_finish":      mean_finish,
            "p80_finish":       _percentile(cp_times, 0.80),
            "p90_finish":       _percentile(cp_times, 0.90),
        }
