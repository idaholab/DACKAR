"""
Hierarchical fallback policy for duration estimation.

The fallback fires when the similarity search produces no usable durations —
either because the historical corpus is empty or because all neighbours have
None actual_duration_hours.  With the soft top-k selector from Phase 2 this
is rare for non-empty corpora, but the hierarchy ensures the system always
returns *some* estimate rather than crashing.

Fallback hierarchy (most specific → most generic)
--------------------------------------------------
Level 1  plant + task_family + component_family
    Filter historical to same plant, same task family, same component family.
    Best analogy quality — same plant procedures, same equipment class.

Level 2  task_family + component_family  (cross-plant)
    Drop the plant constraint.  Useful when the activity type is common but
    the specific plant has little history (e.g. new unit, rare maintenance).

Level 3  task_family only  (fleet-level task class)
    Drop component family.  Coarser but captures broad execution patterns
    for a craft / activity type across the fleet.

Level 4  planned_duration_hours constant  (generic prior)
    Use the planned duration as a point estimate.  Always fires if all
    previous levels fail.  Confidence score 0.1 — essentially "no data."

All fallback levels assign confidence_tier ``"low"`` because the fallback
path by definition means the primary similarity search failed to find usable
analogues.  Users should treat fallback estimates as upper-bound guides only.

Minimum support
---------------
Levels 1–3 require at least ``min_support`` (default 3) matching historical
activities to produce an estimate.  If fewer are found the level is skipped.
Three is deliberately conservative: with 1–2 samples the distribution
percentiles would be highly unreliable and could mislead planners.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from dackar.outage.outage_uncertainty.domain.activity import ActivityCase
from dackar.outage.outage_uncertainty.domain.duration import DurationDistribution
from dackar.outage.outage_uncertainty.domain.result_types import ActivityEstimate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FallbackLevel:
    """Internal descriptor for one level of the hierarchy."""
    name: str
    fields: tuple[str, ...]   # ActivityCase attributes to match


_LEVELS: list[_FallbackLevel] = [
    _FallbackLevel(
        name="plant_task_component",
        fields=("plant_id", "task_family", "component_family"),
    ),
    _FallbackLevel(
        name="fleet_task_component",
        fields=("task_family", "component_family"),
    ),
    _FallbackLevel(
        name="fleet_task_family",
        fields=("task_family",),
    ),
]


class HierarchicalFallbackPolicy:
    """Produce a fallback estimate by progressively relaxing matching criteria.

    Args:
        min_support: Minimum number of historical activities required at a
            given level to use that level's distribution.  Default 3.
        fitter_percentiles: Percentile levels for the fitted distribution.
    """

    def __init__(
        self,
        min_support: int = 3,
        fitter_percentiles: tuple[float, ...] = (0.10, 0.50, 0.80, 0.90),
    ) -> None:
        self.min_support = min_support
        self._fitter_percentiles = fitter_percentiles

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def estimate(
        self,
        query: ActivityCase,
        historical_activities: list[ActivityCase] | None = None,
    ) -> ActivityEstimate:
        """Return a fallback :class:`ActivityEstimate` for *query*.

        Tries each level of the hierarchy in order.  Falls through to the
        planned-duration constant if no level produces enough analogues.

        Args:
            query: The planned activity being estimated.
            historical_activities: Full historical corpus (may be empty or
                ``None``).

        Returns:
            :class:`ActivityEstimate` with ``confidence_tier="low"`` and
            an appropriate warning message.
        """
        historical = historical_activities or []

        for level in _LEVELS:
            analogues = self._filter(historical, query, level.fields)
            durations = [
                a.actual_duration_hours
                for a in analogues
                if a.actual_duration_hours is not None
            ]
            if len(durations) >= self.min_support:
                logger.debug(
                    "HierarchicalFallbackPolicy: level '%s' found %d analogues",
                    level.name,
                    len(durations),
                )
                return self._estimate_from_durations(
                    query,
                    durations,
                    level_name=level.name,
                    n_matched=len(analogues),
                )

        # Level 4: generic prior — planned duration
        return self._planned_duration_prior(query)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter(
        historical: list[ActivityCase],
        query: ActivityCase,
        fields: tuple[str, ...],
    ) -> list[ActivityCase]:
        """Return activities matching *query* on all *fields* (non-None only).

        A level requires at least one of its fields to be non-``None`` on both
        the query and the candidate.  Without this guard, a query whose labels
        are all ``None`` (the P6 adapter emits ``None`` labels before
        ``LabelMapper`` runs) would skip every field check and "match" the
        entire corpus, returning an unfiltered sample under a specific level
        name (e.g. unrelated welding activities reported as
        ``fleet_task_component``).
        """
        # The level is only meaningful if the query itself constrains at least
        # one of the level's fields; otherwise this level cannot discriminate.
        if all(getattr(query, f, None) is None for f in fields):
            return []

        result = []
        for act in historical:
            if not act.is_historical():
                continue
            matched_field = False
            match = True
            for field in fields:
                qval = getattr(query, field, None)
                hval = getattr(act, field, None)
                # If either side is unknown, skip the field check (don't penalise)
                if qval is None or hval is None:
                    continue
                if qval != hval:
                    match = False
                    break
                matched_field = True
            # Require at least one field actually compared equal on both sides,
            # so a candidate with all-None labels is not treated as a match.
            if match and matched_field:
                result.append(act)
        return result

    def _estimate_from_durations(
        self,
        query: ActivityCase,
        durations: list[float],
        level_name: str,
        n_matched: int,
    ) -> ActivityEstimate:
        from dackar.outage.outage_uncertainty.uncertainty.distribution_fitter import (
            DistributionFitter,
            _weighted_percentile,
        )
        # Use keep_all for fallback — sample sizes are small
        fitter = DistributionFitter(percentiles=self._fitter_percentiles)
        distribution = fitter.fit(durations)   # uniform weights at fallback level
        distribution.parameters["fallback_level"] = level_name
        distribution.parameters["n_fallback_analogues"] = float(n_matched)

        return ActivityEstimate(
            activity_id=query.activity_id,
            estimated_distribution=distribution,
            confidence_score=0.25,   # fallback = low but not zero (we have some data)
            confidence_tier="low",
            support_count=len(durations),
            matched_cases=[],
            warnings=[
                f"Fallback estimate (level: {level_name}, n={len(durations)}). "
                "No sufficiently similar activities found via primary search; "
                "expert review recommended."
            ],
            uncertainty_type="epistemic",
            recommended_action=(
                f"Fallback level '{level_name}' used — SME review recommended "
                "before committing to schedule assumptions based on this estimate."
            ),
        )

    @staticmethod
    def _planned_duration_prior(query: ActivityCase) -> ActivityEstimate:
        # Only substitute 1.0 for a genuinely missing planned duration — a
        # 0-hour planned duration (P6 milestones) is a real value, not missing,
        # and `or 1.0` would wrongly rewrite it to a 1-hour estimate.
        baseline = (
            query.planned_duration_hours
            if query.planned_duration_hours is not None
            else 1.0
        )
        distribution = DurationDistribution(
            distribution_type="fallback_constant",
            parameters={"location": baseline, "fallback_level": "planned_duration_prior"},
            p10=baseline,
            p50=baseline,
            p80=baseline,
            p90=baseline,
        )
        return ActivityEstimate(
            activity_id=query.activity_id,
            estimated_distribution=distribution,
            confidence_score=0.10,
            confidence_tier="low",
            support_count=0,
            matched_cases=[],
            warnings=[
                "Fallback estimate (level: planned_duration_prior). "
                "No historical analogues found at any level; "
                "estimate equals planned duration. Expert review required."
            ],
            uncertainty_type="epistemic",
            recommended_action=(
                "No historical analogues found — expert review or field walkdown "
                "required; do not rely on this estimate for critical path planning."
            ),
        )


# ---------------------------------------------------------------------------
# Backwards-compatible alias
# ---------------------------------------------------------------------------

# The original stub was named FallbackPolicy.  Keep the alias so existing
# code that imports FallbackPolicy by name still works.
FallbackPolicy = HierarchicalFallbackPolicy
