from __future__ import annotations


from dataclasses import dataclass, field
from statistics import mean
import random


@dataclass
class DurationDistribution:
    """Empirical duration distribution, optionally representing a two-component
    mixture of routine and disruption-driven execution modes.

    Attributes
    ----------
    distribution_type
        ``"empirical"`` for similarity-fitted distributions;
        ``"fallback_constant"`` when only planned duration is available.
    parameters
        Metadata dict: ``extended_fraction``, ``n_routine``, ``mixture_p80``,
        ``mixture_p90``, etc.  Populated by :class:`DistributionFitter`.
    samples
        Routine-pool durations (sorted ascending).  Used for clean-execution
        percentile interpretation.
    extended_samples
        Disruption-driven durations separated by :class:`OutlierHandler`.
        ``None`` when no outliers were detected.
    mixture_weight
        Probability of disrupted execution (= ``extended_fraction`` from
        separation).  When 0, the distribution is purely routine.
    p10 / p50 / p80 / p90
        Weighted percentiles of the *routine* pool.  Mixture-aware percentiles
        are stored under ``parameters["mixture_p80"]`` / ``"mixture_p90"``.
    """

    distribution_type: str = "empirical"
    parameters: dict[str, float] = field(default_factory=dict)
    samples: list[float] | None = None
    extended_samples: list[float] | None = None   # Gap 1: disruption-driven pool
    mixture_weight: float = 0.0                   # Gap 1: P(disrupted execution)
    p10: float | None = None
    p50: float | None = None
    p80: float | None = None
    p90: float | None = None

    def sample(self, n: int = 1) -> list[float]:
        """Draw *n* samples from the distribution.

        When ``extended_samples`` is populated and ``mixture_weight > 0``,
        each draw is taken from the disruption-driven pool with probability
        ``mixture_weight`` and from the routine pool otherwise.  This produces
        realistic heavy-tailed samples for downstream Monte Carlo simulation.
        """
        if self.samples:
            if self.extended_samples and self.mixture_weight > 0.0:
                results: list[float] = []
                for _ in range(n):
                    if random.random() < self.mixture_weight:
                        results.append(random.choice(self.extended_samples))
                    else:
                        results.append(random.choice(self.samples))
                return results
            return [random.choice(self.samples) for _ in range(n)]
        location = self.parameters.get("location", 1.0)
        return [location for _ in range(n)]

    def mean(self) -> float:
        """Mixture mean: ``(1 - w) * E[routine] + w * E[extended]``."""
        if self.samples:
            routine_mean = mean(self.samples)
            if self.extended_samples and self.mixture_weight > 0.0:
                extended_mean = mean(self.extended_samples)
                return (
                    (1.0 - self.mixture_weight) * routine_mean
                    + self.mixture_weight * extended_mean
                )
            return routine_mean
        return self.parameters.get("location", 0.0)

    def variance(self) -> float:
        """Mixture variance via the law of total variance.

        ``Var(X) = E[Var(X|Z)] + Var(E[X|Z])``

        where *Z* is the mixture component indicator (0 = routine, 1 = extended).
        When no extended pool is present this reduces to the routine sample
        variance.
        """
        if not self.samples or len(self.samples) < 2:
            return 0.0

        mu_r = mean(self.samples)
        var_r = sum((x - mu_r) ** 2 for x in self.samples) / (len(self.samples) - 1)

        if not self.extended_samples or self.mixture_weight <= 0.0:
            return var_r

        mu_e = mean(self.extended_samples)
        var_e = (
            sum((x - mu_e) ** 2 for x in self.extended_samples)
            / (len(self.extended_samples) - 1)
            if len(self.extended_samples) >= 2
            else 0.0
        )

        p = self.mixture_weight
        q = 1.0 - p
        mu_total = q * mu_r + p * mu_e

        within_variance = q * var_r + p * var_e
        between_variance = q * (mu_r - mu_total) ** 2 + p * (mu_e - mu_total) ** 2
        return within_variance + between_variance
