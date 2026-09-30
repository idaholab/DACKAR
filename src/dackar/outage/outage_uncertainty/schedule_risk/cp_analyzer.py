"""
Critical-path risk analyzer — per-activity and project-level schedule metrics.

Richer than :class:`~outage_uncertainty.schedule_risk.robustness_metrics.RobustnessMetrics`
in that it produces per-activity diagnostics that help planners prioritise
mitigation.

Project-level metrics
---------------------
robustness
    ``P(T_finish ≤ baseline_cp_time)``.
p50_finish / p80_finish / p90_finish
    Finish time quantiles from the simulation distribution.
schedule_variance / schedule_std_dev
    Spread of the project completion distribution.
expected_delay
    ``max(0, E[T_finish] − baseline_cp_time)`` in hours.

Per-activity metrics
--------------------
criticality_index
    Fraction of simulations in which each activity appears on the critical
    path.  Non-CP activities score 0.

expected_finish_when_critical
    Mean project finish time across simulations where the activity is on the
    critical path.  Higher values identify activities whose criticality is
    associated with worse schedule outcomes.

expected_drag
    ``E[T_finish | i on CP] − E[T_finish | i not on CP]``.
    The average increase in project duration associated with the activity
    being on the critical path.  Positive values indicate the activity
    contributes to overruns when it is critical.

cp_sensitivity
    Point-biserial correlation between the activity's criticality indicator
    (0/1 per simulation run) and the project finish time.  Values near 1.0
    identify activities whose on-CP status most reliably predicts project
    delays — the primary candidates for pre-outage mitigation.
"""
from __future__ import annotations

from outage_uncertainty.domain.result_types import SimulationResult


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

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


def _point_biserial_corr(x_binary: list[int], y: list[float]) -> float:
    """Point-biserial correlation between binary indicator *x* and continuous *y*.

    Quantifies how strongly the activity's criticality status (0/1) predicts
    the project finish time.  Equivalent to Pearson r between binary x and
    continuous y; bounded in [-1, 1].  Positive values indicate that being
    on the critical path is associated with later finish times.
    """
    n = len(x_binary)
    if n < 2:
        return 0.0
    n1 = sum(x_binary)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return 0.0

    y_mean = sum(y) / n
    y_var  = sum((yi - y_mean) ** 2 for yi in y) / n
    if y_var <= 0.0:
        return 0.0
    y_std = y_var ** 0.5

    m1 = sum(yi for xi, yi in zip(x_binary, y) if xi == 1) / n1
    m0 = sum(yi for xi, yi in zip(x_binary, y) if xi == 0) / n0

    return (m1 - m0) / y_std * ((n1 * n0 / n ** 2) ** 0.5)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class CriticalPathRiskAnalyzer:
    """Produce project-level and per-activity schedule risk metrics.

    All inputs come from :class:`~outage_uncertainty.domain.result_types.SimulationResult`;
    no additional simulation is needed.
    """

    def analyze(self, sim_result: SimulationResult, baseline_cp_time: float) -> dict:
        """Return a comprehensive schedule risk dict.

        Args:
            sim_result: Output of ``MonteCarloSimulator.run()``.
            baseline_cp_time: Planned CP duration used for robustness
                and expected-delay calculations.

        Returns:
            Dict with both project-level and per-activity keys.  Per-activity
            dicts are keyed by ``activity_id`` and contain only activities
            that appeared on the critical path at least once.
        """
        cp_times = sim_result.cp_times
        _empty = {
            "robustness":                   0.0,
            "p50_finish":                   0.0,
            "p80_finish":                   0.0,
            "p90_finish":                   0.0,
            "schedule_variance":            0.0,
            "schedule_std_dev":             0.0,
            "expected_delay":               0.0,
            "criticality_index":            {},
            "expected_finish_when_critical":{},
            "expected_drag":                {},
            "cp_sensitivity":               {},
        }
        if not cp_times:
            return _empty

        n_runs = len(cp_times)

        # ── Project-level ─────────────────────────────────────────────────
        robustness    = sum(t <= baseline_cp_time for t in cp_times) / n_runs
        mean_t        = sum(cp_times) / n_runs
        variance      = sum((t - mean_t) ** 2 for t in cp_times) / n_runs
        std_dev       = variance ** 0.5
        expected_delay = max(0.0, mean_t - baseline_cp_time)

        criticality_index = {
            act_id: count / n_runs
            for act_id, count in sim_result.activity_criticality.items()
        }

        # ── Per-activity: group CP finish times by on-CP activities ────────
        # Precompute sets for O(1) membership tests inside the inner loop
        cp_path_sets = [set(path) for path in sim_result.cp_paths]

        activity_critical_times: dict[str, list[float]] = {}
        for t, path_set in zip(cp_times, cp_path_sets):
            for act_id in path_set:
                activity_critical_times.setdefault(act_id, []).append(t)

        expected_finish_when_critical = {
            act_id: sum(times) / len(times)
            for act_id, times in activity_critical_times.items()
        }

        # Expected drag: E[finish | i on CP] − E[finish | i not on CP]
        total_sum = sum(cp_times)
        expected_drag: dict[str, float] = {}
        for act_id, critical_times in activity_critical_times.items():
            n1 = len(critical_times)
            n0 = n_runs - n1
            if n0 == 0:
                expected_drag[act_id] = 0.0
                continue
            m1 = sum(critical_times) / n1
            m0 = (total_sum - sum(critical_times)) / n0
            expected_drag[act_id] = m1 - m0

        # CP sensitivity: point-biserial corr(criticality indicator, finish time)
        cp_sensitivity: dict[str, float] = {}
        for act_id in sim_result.activity_criticality:
            x = [1 if act_id in path_set else 0 for path_set in cp_path_sets]
            cp_sensitivity[act_id] = _point_biserial_corr(x, cp_times)

        return {
            "robustness":                    robustness,
            "p50_finish":                    _percentile(cp_times, 0.50),
            "p80_finish":                    _percentile(cp_times, 0.80),
            "p90_finish":                    _percentile(cp_times, 0.90),
            "schedule_variance":             variance,
            "schedule_std_dev":              std_dev,
            "expected_delay":                expected_delay,
            "criticality_index":             criticality_index,
            "expected_finish_when_critical": expected_finish_when_critical,
            "expected_drag":                 expected_drag,
            "cp_sensitivity":                cp_sensitivity,
        }
