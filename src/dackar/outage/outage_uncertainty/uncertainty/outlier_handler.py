"""
Outlier separation for historical activity durations.

The core insight from the PDF: outage task duration distributions are often
bimodal — a "routine execution" cluster plus a lower-probability
"disruption-driven" tail caused by unexpected findings, rework, or emergent
coordination issues.  Pooling both modes into a single distribution inflates
the P80/P90 for tasks that almost always run cleanly, and can underestimate
when the disruption mode is frequent.

This module separates the two populations so downstream components can:
  - Fit the main distribution on the *routine* group
  - Report the *extended_fraction* as a risk signal
  - Optionally model the disruption mode separately (future work)

Separation strategies
---------------------
keep_all
    No separation; all durations are treated as routine.  Safe fallback for
    very small samples (n < ``MIN_SAMPLES_FOR_SEPARATION``).

trim_symmetric
    Symmetric percentage trim.  Removes the top and bottom ``trim_pct``
    fraction.  Useful when both tails are noise; rarely appropriate for
    duration data which is naturally right-skewed.

iqr
    Upper-fence IQR separation: fence = Q3 + 1.5 × IQR.  Durations above
    the fence go to the *extended* group.  No lower fence — very short
    executions are genuine and should not be removed.

mad
    Median + k × MAD upper threshold (default k = 3.0).  More robust than
    IQR for small samples (n ≥ 4) and strongly skewed distributions because
    the MAD is insensitive to extreme values.

Weight alignment
----------------
Durations and their relevance_weights (from NeighborSelector) are processed
as aligned pairs so that the separation preserves the correct weight for each
retained duration.  Weights are re-normalised within each group.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Separation is unreliable below this sample count — fall back to keep_all
MIN_SAMPLES_FOR_SEPARATION = 4


@dataclass
class OutlierSeparation:
    """Result of separating routine from disruption-driven durations.

    Attributes
    ----------
    routine, routine_weights
        Durations (and their re-normalised relevance weights) that fall
        within the normal execution range.
    extended, extended_weights
        Durations classified as disruption-driven (above the fence).
    extended_fraction
        Fraction of the total sample that was classified as extended.
    method
        Which strategy produced this separation.
    threshold
        The computed upper fence value (``None`` for keep_all / trim).
    """

    routine: list[float] = field(default_factory=list)
    routine_weights: list[float] = field(default_factory=list)
    extended: list[float] = field(default_factory=list)
    extended_weights: list[float] = field(default_factory=list)
    method: str = "keep_all"
    threshold: float | None = None

    @property
    def extended_fraction(self) -> float:
        n = len(self.routine) + len(self.extended)
        return len(self.extended) / n if n > 0 else 0.0

    @property
    def n_routine(self) -> int:
        return len(self.routine)

    @property
    def n_total(self) -> int:
        return len(self.routine) + len(self.extended)


class OutlierHandler:
    """Separate routine durations from disruption-driven outliers.

    Args:
        strategy: One of ``'keep_all'``, ``'trim_symmetric'``, ``'iqr'``,
            ``'mad'``.  Default ``'iqr'``.
        trim_pct: Fraction to remove from each tail when
            ``strategy='trim_symmetric'``.  Default 0.10 (10 %).
        mad_scale: Multiplier k for the MAD upper threshold
            (threshold = median + k × MAD).  Default 3.0.
    """

    VALID_STRATEGIES = frozenset({"keep_all", "trim_symmetric", "iqr", "mad"})

    def __init__(
        self,
        strategy: str = "iqr",
        trim_pct: float = 0.10,
        mad_scale: float = 3.0,
    ) -> None:
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(self.VALID_STRATEGIES)}; got '{strategy}'"
            )
        self.strategy = strategy
        self.trim_pct = trim_pct
        self.mad_scale = mad_scale

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def separate(
        self,
        durations: list[float],
        weights: list[float] | None = None,
    ) -> OutlierSeparation:
        """Separate *durations* into routine and extended groups.

        Args:
            durations: List of historical actual durations (hours).
            weights: Corresponding relevance weights from
                :class:`~outage_uncertainty.retrieval.neighbor_selector.NeighborSelector`.
                If ``None``, uniform weights are used.

        Returns:
            :class:`OutlierSeparation` with both groups and their weights.
        """
        n = len(durations)
        if n == 0:
            return OutlierSeparation(method=self.strategy)

        # Build / validate weights
        w = list(weights) if weights is not None else [1.0 / n] * n
        if len(w) != n:
            logger.warning(
                "OutlierHandler: weights length %d ≠ durations length %d; "
                "using uniform weights",
                len(w),
                n,
            )
            w = [1.0 / n] * n

        # Fall through to keep_all for tiny samples regardless of strategy
        if n < MIN_SAMPLES_FOR_SEPARATION and self.strategy != "keep_all":
            logger.debug(
                "OutlierHandler: n=%d < %d; using keep_all",
                n,
                MIN_SAMPLES_FOR_SEPARATION,
            )
            return self._keep_all(durations, w)

        if self.strategy == "keep_all":
            return self._keep_all(durations, w)
        if self.strategy == "trim_symmetric":
            return self._trim_symmetric(durations, w)
        if self.strategy == "iqr":
            return self._iqr(durations, w)
        if self.strategy == "mad":
            return self._mad(durations, w)

        return self._keep_all(durations, w)  # unreachable, but safe

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _keep_all(durations: list[float], weights: list[float]) -> OutlierSeparation:
        nw = _normalise(weights)
        return OutlierSeparation(
            routine=list(durations),
            routine_weights=nw,
            method="keep_all",
        )

    def _trim_symmetric(
        self, durations: list[float], weights: list[float]
    ) -> OutlierSeparation:
        pairs = sorted(zip(durations, weights), key=lambda p: p[0])
        cut = max(1, int(len(pairs) * self.trim_pct))
        kept = pairs[cut:-cut] if cut < len(pairs) // 2 else pairs
        if not kept:
            kept = pairs
        d = [p[0] for p in kept]
        w = _normalise([p[1] for p in kept])
        return OutlierSeparation(routine=d, routine_weights=w, method="trim_symmetric")

    @staticmethod
    def _interpolated_percentile(sorted_vals: list[float], q: float) -> float:
        """Linear-interpolation percentile on a pre-sorted list."""
        n = len(sorted_vals)
        if n == 1:
            return sorted_vals[0]
        pos = (n - 1) * q
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

    def _iqr(self, durations: list[float], weights: list[float]) -> OutlierSeparation:
        sorted_d = sorted(durations)
        q1 = self._interpolated_percentile(sorted_d, 0.25)
        q3 = self._interpolated_percentile(sorted_d, 0.75)
        iqr = q3 - q1
        fence = q3 + 1.5 * iqr

        return self._split_at(durations, weights, fence, method="iqr")

    def _mad(self, durations: list[float], weights: list[float]) -> OutlierSeparation:
        sorted_d = sorted(durations)
        median = self._interpolated_percentile(sorted_d, 0.50)
        abs_devs = sorted(abs(d - median) for d in sorted_d)
        mad = self._interpolated_percentile(abs_devs, 0.50)
        fence = median + self.mad_scale * mad

        return self._split_at(durations, weights, fence, method="mad")

    @staticmethod
    def _split_at(
        durations: list[float],
        weights: list[float],
        fence: float,
        method: str,
    ) -> OutlierSeparation:
        routine_d, routine_w = [], []
        extended_d, extended_w = [], []

        for d, w in zip(durations, weights):
            if d <= fence:
                routine_d.append(d)
                routine_w.append(w)
            else:
                extended_d.append(d)
                extended_w.append(w)

        # Guarantee at least one routine sample (avoid empty routine group)
        if not routine_d:
            routine_d, routine_w = list(durations), list(weights)
            extended_d, extended_w = [], []

        return OutlierSeparation(
            routine=routine_d,
            routine_weights=_normalise(routine_w),
            extended=extended_d,
            extended_weights=_normalise(extended_w) if extended_w else [],
            method=method,
            threshold=fence,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _normalise(weights: list[float]) -> list[float]:
    """Return weights normalised to sum to 1.0; uniform if all zero."""
    total = sum(weights)
    if total <= 0.0:
        n = len(weights)
        return [1.0 / n] * n if n > 0 else []
    return [w / total for w in weights]
