"""
Empirical distribution fitter with weighted percentile support.

Weighted percentiles
--------------------
Each duration in the routine group carries a ``relevance_weight`` assigned by
:class:`~outage_uncertainty.retrieval.neighbor_selector.NeighborSelector`.
High-similarity neighbours receive proportionally more weight, so the P50/P80
estimate is dominated by the closest historical analogs rather than treating
all matches as equally informative.

Algorithm (weighted Type-7 percentile)
    1. Sort ``(duration, weight)`` pairs by duration.
    2. Build a weighted CDF: cumulative normalised weight after each point.
    3. Interpolate linearly between adjacent points to find the value at
       quantile *q*.

When weights are uniform or ``None``, this reduces to the standard
interpolated percentile.
"""
from __future__ import annotations

from outage_uncertainty.domain.duration import DurationDistribution
from outage_uncertainty.uncertainty.outlier_handler import OutlierSeparation


class DistributionFitter:
    """Fit an empirical :class:`DurationDistribution` from a sample.

    Args:
        percentiles: Quantile levels to compute.  Default ``(0.10, 0.50,
            0.80, 0.90)``.
    """

    def __init__(
        self,
        percentiles: tuple[float, ...] = (0.10, 0.50, 0.80, 0.90),
    ) -> None:
        self.percentiles = percentiles

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        durations: list[float],
        weights: list[float] | None = None,
    ) -> DurationDistribution:
        """Fit a distribution from *durations* with optional *weights*.

        Args:
            durations: Duration values (hours).
            weights: Relevance weights for each duration.  If ``None``,
                uniform weights are assumed.

        Returns:
            A :class:`DurationDistribution` with ``distribution_type``
            ``"empirical"`` and weighted p10/p50/p80/p90 percentiles.
        """
        if not durations:
            return DurationDistribution(distribution_type="empirical")

        w = weights if weights is not None else [1.0 / len(durations)] * len(durations)

        p10, p50, p80, p90 = (
            _weighted_percentile(durations, w, q) for q in self.percentiles
        )
        return DurationDistribution(
            distribution_type="empirical",
            samples=sorted(durations),
            p10=p10,
            p50=p50,
            p80=p80,
            p90=p90,
        )

    def fit_from_separation(self, separation: OutlierSeparation) -> DurationDistribution:
        """Fit a two-component mixture distribution from an :class:`OutlierSeparation`.

        The routine group drives the primary percentile estimates (``p10`` …
        ``p90``).  When an extended group is also present:

        * ``dist.extended_samples`` and ``dist.mixture_weight`` are populated so
          that :meth:`~DurationDistribution.sample`, :meth:`~DurationDistribution.mean`,
          and :meth:`~DurationDistribution.variance` reflect the full mixture.
        * Mixture-aware ``p80`` / ``p90`` (combining both pools with their
          respective mixture weights) are stored under
          ``parameters["mixture_p80"]`` / ``"mixture_p90"]`` for planners who
          want a single tail quantile that accounts for disrupted executions.
        """
        dist = self.fit(separation.routine, weights=separation.routine_weights)
        dist.parameters["extended_fraction"] = separation.extended_fraction
        dist.parameters["n_routine"] = float(separation.n_routine)
        dist.parameters["n_total"] = float(separation.n_total)
        dist.parameters["outlier_method"] = separation.method
        if separation.threshold is not None:
            dist.parameters["outlier_threshold"] = separation.threshold

        # Gap 1: populate mixture fields when disruption-driven samples exist
        if separation.extended:
            ef = separation.extended_fraction
            dist.extended_samples = list(separation.extended)
            dist.mixture_weight = ef

            # Mixture-aware percentiles: combine both pools with mass-weighted weights.
            # Routine samples carry total mass (1 - ef); extended carry ef.
            if separation.extended_weights and ef > 0.0:
                combined_d = separation.routine + separation.extended
                combined_w = (
                    [w * (1.0 - ef) for w in separation.routine_weights]
                    + [w * ef for w in separation.extended_weights]
                )
                dist.parameters["mixture_p80"] = _weighted_percentile(
                    combined_d, combined_w, 0.80
                )
                dist.parameters["mixture_p90"] = _weighted_percentile(
                    combined_d, combined_w, 0.90
                )

        return dist


# ---------------------------------------------------------------------------
# Weighted percentile helper
# ---------------------------------------------------------------------------

def _weighted_percentile(
    values: list[float],
    weights: list[float],
    q: float,
) -> float:
    """Compute the *q*-th weighted percentile of *values*.

    Uses linear interpolation between adjacent CDF points (equivalent to
    NumPy's ``percentile`` method 7 generalised to arbitrary weights).

    Args:
        values: Data points.
        weights: Non-negative weights (need not sum to 1).
        q: Quantile in [0, 1].

    Returns:
        Interpolated percentile value.
    """
    if not values:
        return 0.0

    # Sort by value, keeping weights aligned
    pairs = sorted(zip(values, weights), key=lambda p: p[0])
    sorted_vals = [p[0] for p in pairs]
    sorted_w = [p[1] for p in pairs]

    n = len(sorted_vals)
    total = sum(sorted_w)
    if total <= 0.0:
        # Degenerate case: all zero weights → uniform
        sorted_w = [1.0] * n
        total = float(n)

    # Build cumulative CDF as midpoint positions:
    # place point i at (cumulative_weight_before_i + 0.5 * w_i) / total
    # This is the weighted analogue of (i + 0.5) / n positioning.
    cdf: list[float] = []
    cumulative = 0.0
    for w in sorted_w:
        cdf.append((cumulative + 0.5 * w) / total)
        cumulative += w

    # Clamp q to the data range
    if q <= cdf[0]:
        return sorted_vals[0]
    if q >= cdf[-1]:
        return sorted_vals[-1]

    # Linear interpolation between adjacent CDF points
    for i in range(1, n):
        if cdf[i] >= q:
            # Interpolate between i-1 and i
            span = cdf[i] - cdf[i - 1]
            if span <= 0.0:
                return sorted_vals[i]
            frac = (q - cdf[i - 1]) / span
            return sorted_vals[i - 1] + frac * (sorted_vals[i] - sorted_vals[i - 1])

    return sorted_vals[-1]
