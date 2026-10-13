"""
Neighbor selector: soft top-k with power-normalised relevance weights.

The core problem with a hard similarity threshold is fragility on rare tasks:
if an activity has never appeared in the historical database in exactly the
same form, a 0.65 cutoff may return zero matches and trigger the fallback
constant distribution — which is less informative than using weak-but-present
analogues with an appropriate low-confidence warning.

This implementation always returns up to *top_k* matches (soft selection)
and attaches a normalised ``relevance_weight`` to each so that downstream
weighted statistics (Phase 3) can discount weak matches automatically.

Relevance weighting
-------------------
Each selected match receives a weight proportional to ``score^α``:

    raw_weight_i  = max(score_i, 0)^α
    relevance_weight_i = raw_weight_i / Σ raw_weight_j

With α = 2 (default), a match at score 0.8 gets four times the weight of
a match at score 0.4.  α = 1 gives linear weighting; α → ∞ gives winner-
takes-all.

Low-coverage warning
--------------------
``warn_below`` (default 0.4): if the *best* match in the selected set has
total_score < warn_below the selector flags low evidence.  The
``SimilarityAssessmentWorkflow`` propagates this flag as a warning string on
the ``ActivityEstimate`` (see ``DurationEstimator``).
"""
from __future__ import annotations

import dataclasses
import logging

from outage_uncertainty.domain.result_types import SimilarityMatch

logger = logging.getLogger(__name__)


class NeighborSelector:
    """Select the best neighbors and assign normalised relevance weights.

    Args:
        top_k: Maximum number of neighbors to return.
        min_score: Optional hard floor.  Matches strictly below this score
            are excluded before the top-k cut.  Default ``0.0`` (disabled) —
            prefer ``warn_below`` for soft signalling instead.
        warn_below: If the highest-scoring selected match has
            ``total_score < warn_below``, the selector considers coverage
            "low".  Callers can check :meth:`has_low_coverage` or read the
            ``low_coverage`` attribute after calling :meth:`select`.
        weight_exponent: α for power weighting.  Higher values give
            stronger relative preference to top matches.  Default ``2.0``.

    Thread-safety
    -------------
    ``NeighborSelector`` instances are **not thread-safe**.  :meth:`select`
    writes back to ``self.low_coverage`` as a side-effect, so concurrent
    calls on the same instance from multiple threads will race.  In a
    multithreaded pipeline, either:

    * create one ``NeighborSelector`` per thread / task, or
    * guard shared instances with an external lock and read
      ``low_coverage`` within the same critical section as the
      :meth:`select` call that set it.

    The current single-threaded outage pipeline is unaffected.
    """

    def __init__(
        self,
        top_k: int = 30,
        min_score: float = 0.0,
        warn_below: float = 0.4,
        weight_exponent: float = 2.0,
    ) -> None:
        self.top_k = top_k
        self.min_score = min_score
        self.warn_below = warn_below
        self.weight_exponent = weight_exponent
        self.low_coverage: bool = False   # updated by each call to select()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select(self, matches: list[SimilarityMatch]) -> list[SimilarityMatch]:
        """Sort, optionally hard-floor, take top-k, assign relevance weights.

        Returns a **new** list of :class:`SimilarityMatch` copies whose
        ``relevance_weight`` fields carry the power-normalised weights.  The
        input *matches* are never modified, so callers that hold references to
        the original objects see no side-effects.

        Returns an empty list when no matches are available.
        """
        if not matches:
            self.low_coverage = True
            return []

        # Sort descending by total score
        sorted_matches = sorted(matches, key=lambda m: m.total_score, reverse=True)

        # Optional hard floor (disabled by default; prefer warn_below)
        if self.min_score > 0.0:
            sorted_matches = [m for m in sorted_matches if m.total_score >= self.min_score]

        selected = sorted_matches[: self.top_k]

        if not selected:
            self.low_coverage = True
            return []

        # ---- Relevance weights via power normalisation -------------------
        raw_weights = [max(m.total_score, 0.0) ** self.weight_exponent for m in selected]
        total = sum(raw_weights)

        if total > 0.0:
            final = [
                dataclasses.replace(m, relevance_weight=rw / total)
                for m, rw in zip(selected, raw_weights)
            ]
        else:
            # All scores are exactly 0 — assign uniform weights
            uniform = 1.0 / len(selected)
            final = [dataclasses.replace(m, relevance_weight=uniform) for m in selected]

        # ---- Low-coverage flag ------------------------------------------
        self.low_coverage = final[0].total_score < self.warn_below
        if self.low_coverage:
            logger.debug(
                "NeighborSelector: low coverage — best score %.3f < warn_below %.3f",
                final[0].total_score,
                self.warn_below,
            )

        return final

    def has_low_coverage(
        self,
        matches: list[SimilarityMatch] | None = None,
    ) -> bool:
        """Return ``True`` if the best match is below the warning threshold.

        Two calling modes:

        **Pre-select** — pass the raw match list before calling
        :meth:`select`.  The method computes the best score directly from
        *matches* and compares it against ``warn_below``.

        **Post-select** — call with no argument (or ``matches=None``) after
        :meth:`select` has run.  Returns the cached ``self.low_coverage`` flag
        that was set during the most recent :meth:`select` call.  This
        correctly reflects the filtered, top-k–selected set rather than the
        original pre-filter candidates.

        .. code-block:: python

            # pre-select check
            if selector.has_low_coverage(raw_matches):
                ...

            selected = selector.select(raw_matches)

            # post-select check (reads cached flag)
            if selector.has_low_coverage():
                ...
        """
        if matches is None:
            # Post-select: use the flag set by the most recent select() call.
            return self.low_coverage
        # Pre-select: compute from the provided raw match list.
        if not matches:
            return True
        return max(m.total_score for m in matches) < self.warn_below
