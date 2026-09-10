"""
Duration estimator — orchestrates the full uncertainty pipeline.

Data flow
---------
1. Extract ``(duration, relevance_weight)`` pairs from the neighbour list,
   keeping only matches that have an actual duration.  Weights are
   re-normalised after filtering so they continue to sum to 1.0.

2. ``OutlierHandler.separate(durations, weights)`` → ``OutlierSeparation``
   Separates the routine execution cluster from disruption-driven outliers.
   The separation preserves weight alignment so the correct weights flow to
   the fitter.

3. ``DistributionFitter.fit_from_separation(separation)``
   → ``DurationDistribution``
   Fits weighted percentiles on the *routine* group.  Stores
   ``extended_fraction`` in ``distribution.parameters`` as a disruption
   risk signal.

4. ``ConfidenceEstimator.classify(query, matches, separation)``
   → ``ConfidenceResult(score, tier, rationale)``
   Uses power-normalised relevance weights for the similarity component.

5. Build warnings from tier, extended_fraction, and neighbour coverage.

6. Return ``ActivityEstimate`` with all fields populated.

Fallback path
-------------
If *matches* is empty or all matched neighbours have ``None`` actual
durations, the ``HierarchicalFallbackPolicy`` is invoked with access to
the full ``historical_activities`` list so it can apply the 4-level
relaxed-matching strategy.
"""
from __future__ import annotations

import logging

from outage_uncertainty.domain.activity import ActivityCase
from outage_uncertainty.domain.result_types import ActivityEstimate, SimilarityMatch

logger = logging.getLogger(__name__)

# Threshold: log a warning when extended_fraction exceeds this
_HIGH_DISRUPTION_FRACTION = 0.30


class DurationEstimator:
    """Orchestrate outlier separation, distribution fitting, and confidence.

    Args:
        outlier_handler: :class:`~outage_uncertainty.uncertainty.outlier_handler.OutlierHandler`
        fitter: :class:`~outage_uncertainty.uncertainty.distribution_fitter.DistributionFitter`
        confidence_estimator: :class:`~outage_uncertainty.uncertainty.confidence.ConfidenceEstimator`
        fallback_policy: :class:`~outage_uncertainty.uncertainty.fallback_policy.HierarchicalFallbackPolicy`
    """

    def __init__(
        self,
        outlier_handler,
        fitter,
        confidence_estimator,
        fallback_policy,
        *,
        low_coverage_threshold: float = 0.4,
    ) -> None:
        self.outlier_handler = outlier_handler
        self.fitter = fitter
        self.confidence_estimator = confidence_estimator
        self.fallback_policy = fallback_policy
        # Mirrors NeighborSelector.warn_below — keep in sync when constructing
        # the pipeline so the warning fires at the same threshold as the
        # low-coverage flag set by the selector.
        self.low_coverage_threshold = low_coverage_threshold

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def estimate(
        self,
        query: ActivityCase,
        matches: list[SimilarityMatch],
        historical_activities: list[ActivityCase] | None = None,
    ) -> ActivityEstimate:
        """Produce a :class:`ActivityEstimate` for *query* given *matches*.

        Args:
            query: The planned activity being estimated.
            matches: Top-k neighbours from
                :class:`~outage_uncertainty.retrieval.neighbor_selector.NeighborSelector`.
                May be empty.
            historical_activities: Full historical corpus.  Passed to the
                fallback policy so it can apply relaxed matching when the
                primary search yields no usable durations.

        Returns:
            :class:`ActivityEstimate` with distribution, confidence, and
            warnings.
        """
        # ---- Step 1: extract (duration, weight) pairs -------------------
        pairs = [
            (m.candidate_duration_hours, m.relevance_weight)
            for m in matches
            if m.candidate_duration_hours is not None
        ]

        if not pairs:
            logger.debug(
                "DurationEstimator: no durations in matches for '%s'; "
                "invoking fallback",
                query.activity_id,
            )
            return self.fallback_policy.estimate(query, historical_activities)

        durations = [p[0] for p in pairs]
        raw_weights = [p[1] for p in pairs]

        # Re-normalise weights after filtering (some matches may have been dropped)
        total_w = sum(raw_weights)
        weights = (
            [w / total_w for w in raw_weights]
            if total_w > 0.0
            else [1.0 / len(raw_weights)] * len(raw_weights)
        )

        # ---- Step 2: outlier separation ---------------------------------
        separation = self.outlier_handler.separate(durations, weights)

        if separation.n_routine == 0:
            # All durations were classified as extended outliers — treat as no data
            estimate = self.fallback_policy.estimate(query, historical_activities)
            estimate.warnings.append(
                f"All {len(durations)} matched durations were classified as outliers "
                f"(extended pool). Fallback estimate used; consider reviewing the "
                f"outlier threshold or expanding the historical dataset."
            )
            return estimate

        # ---- Step 3: fit distribution on routine group ------------------
        distribution = self.fitter.fit_from_separation(separation)

        # ---- Step 4: confidence tier ------------------------------------
        result = self.confidence_estimator.classify(query, matches, separation)
        distribution.parameters["confidence_tier"] = result.tier

        # ---- Step 5: build warnings -------------------------------------
        warnings = self._build_warnings(
            result, separation, matches,
            low_coverage_threshold=self.low_coverage_threshold,
        )

        # ---- Step 6: assemble estimate ----------------------------------
        return ActivityEstimate(
            activity_id=query.activity_id,
            estimated_distribution=distribution,
            confidence_score=result.score,
            confidence_tier=result.tier,
            support_count=separation.n_routine,
            matched_cases=matches,
            warnings=warnings,
            uncertainty_type=result.uncertainty_type,
            recommended_action=result.recommended_action,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_warnings(
        result,
        separation,
        matches,
        *,
        low_coverage_threshold: float = 0.4,
    ) -> list[str]:
        warnings: list[str] = []

        # Confidence tier
        if result.tier == "low":
            warnings.append(f"Low-confidence estimate. {result.rationale}")
        elif result.tier == "medium":
            warnings.append(f"Medium-confidence estimate. {result.rationale}")

        # Disruption-driven tail
        if separation.extended_fraction >= _HIGH_DISRUPTION_FRACTION:
            pct = round(separation.extended_fraction * 100)
            warnings.append(
                f"{pct}% of historical analogues showed disrupted execution "
                f"(above the {separation.method.upper()} fence). "
                "Consider scheduling contingency."
            )

        # Low routine support after separation
        if separation.n_routine < 5:
            warnings.append(
                f"Only {separation.n_routine} routine analogue(s) available "
                "after outlier separation; distribution may be unreliable."
            )

        # Low-coverage flag — threshold must match NeighborSelector.warn_below
        # used when building the match list so both checks fire at the same point.
        if matches and max(m.total_score for m in matches) < low_coverage_threshold:
            warnings.append(
                f"Best analogue similarity is below {low_coverage_threshold:.2g}. "
                "The retrieved cases may not be representative."
            )

        return warnings
