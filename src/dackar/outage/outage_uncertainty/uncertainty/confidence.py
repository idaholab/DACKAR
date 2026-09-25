"""
Three-tier confidence estimation for duration estimates.

The PDF distinguishes two uncertainty types:

  Aleatory  — genuine execution-time variability (can be modelled)
  Epistemic — lack of knowledge from too few / too dissimilar analogues

The confidence tier captures the *epistemic* dimension: it signals how much
trust the planner should place in the estimate regardless of how wide the
distribution is.

Tiers
-----
high    Strong historical analogue coverage; estimate is reliable.
        Reserve this tier for situations where a planner could act on the
        P80 interval without further review.

medium  Moderate coverage; usable but flag for planner awareness.
        Appropriate when the estimate informs scheduling decisions but
        a contingency buffer is advisable.

low     Weak or no analogues; expert review required before committing
        to schedule assumptions based on this estimate.

Score formula
-------------
The scalar confidence score uses the relevance_weights from Phase 2
(power-normalised) rather than a raw average so that the closest analogues
dominate:

    weighted_sim   = Σ(relevance_weight_i × total_score_i)
    support_factor = min(n_routine / 20, 1.0)
    score          = 0.6 × weighted_sim + 0.4 × support_factor

The 0.6 / 0.4 split reflects that similarity quality matters more than raw
count but both are necessary for a high-confidence estimate.

Tier thresholds (all conditions must hold simultaneously)
---------------------------------------------------------
high   : score ≥ 0.70  AND  n_routine ≥ 10  AND  best_match ≥ 0.70
         AND  outages_represented ≥ high_outage_threshold  (when provided)
medium : score ≥ 0.45  AND  n_routine ≥  5  AND  best_match ≥ 0.50
         AND  outages_represented ≥ medium_outage_threshold  (when provided)
low    : otherwise

Outage diversity gate
---------------------
Patterns learned from a single outage reflect within-outage variance, not
the genuine between-outage variability that defines execution uncertainty.
When ``outages_represented`` is passed to :meth:`classify` (value > 0), the
tier is capped based on how many distinct outages the analogue pool spans:

    outages_represented == 0  →  gate not applied (backward-compatible default)
    outages_represented  < medium_outage_threshold  →  capped at ``"low"``
    outages_represented  < high_outage_threshold    →  capped at ``"medium"``
"""
from __future__ import annotations

from dataclasses import dataclass

from outage_uncertainty.domain.activity import ActivityCase
from outage_uncertainty.domain.result_types import SimilarityMatch
from outage_uncertainty.uncertainty.outlier_handler import OutlierSeparation


@dataclass
class ConfidenceResult:
    """Result of confidence estimation.

    Attributes
    ----------
    score
        Scalar in [0, 1].
    tier
        Qualitative tier: ``"high"``, ``"medium"``, or ``"low"``.
    rationale
        Human-readable explanation of the tier assignment.
    uncertainty_type
        Gap 2: dominant uncertainty source.

        * ``"epistemic"`` – weak analogue coverage; the estimate is uncertain
          because we lack data, not because the task is inherently variable.
          Action: SME review / better work-package definition.
        * ``"aleatory"`` – well-characterised natural execution variability.
          Action: add schedule float proportional to P90–P50 spread.
        * ``"mixed"`` – both signals present (moderate coverage *and* high
          disruption rate or spread).  Action: contingency buffer + awareness.
    recommended_action
        Plain-language scheduling guidance derived from ``uncertainty_type``
        and the tier.
    """

    score: float
    tier: str        # "high" | "medium" | "low"
    rationale: str
    uncertainty_type: str = "unknown"   # "epistemic" | "aleatory" | "mixed"
    recommended_action: str = ""


# Thresholds as named constants so they can be overridden via subclassing
# or adjusted in tests without monkey-patching.
_HIGH_SCORE   = 0.70
_HIGH_SUPPORT = 10
_HIGH_BEST    = 0.70

_MED_SCORE    = 0.45
_MED_SUPPORT  = 5
_MED_BEST     = 0.50

# Outage diversity gate: minimum distinct outages required per tier.
# Set to 0 to disable the gate entirely.
_HIGH_OUTAGES = 3
_MED_OUTAGES  = 2


class ConfidenceEstimator:
    """Compute a scalar confidence score and tier for a duration estimate.

    Args:
        similarity_weight: Weight given to the similarity component of the
            score (the rest goes to support count).  Default 0.6.
        support_saturation: Number of routine samples at which the support
            factor saturates at 1.0.  Default 20.
        high_score_threshold: Minimum score for the ``"high"`` tier.
        high_support_threshold: Minimum routine-sample count for ``"high"``.
        high_best_match_threshold: Minimum best-match score for ``"high"``.
        medium_score_threshold: Minimum score for the ``"medium"`` tier.
        medium_support_threshold: Minimum routine-sample count for
            ``"medium"``.
        medium_best_match_threshold: Minimum best-match score for
            ``"medium"``.
    """

    def __init__(
        self,
        similarity_weight: float = 0.6,
        support_saturation: int = 20,
        high_score_threshold: float = _HIGH_SCORE,
        high_support_threshold: int = _HIGH_SUPPORT,
        high_best_match_threshold: float = _HIGH_BEST,
        medium_score_threshold: float = _MED_SCORE,
        medium_support_threshold: int = _MED_SUPPORT,
        medium_best_match_threshold: float = _MED_BEST,
        high_outage_threshold: int = _HIGH_OUTAGES,
        medium_outage_threshold: int = _MED_OUTAGES,
    ) -> None:
        self._sim_w = similarity_weight
        self._sup_sat = support_saturation
        self._high = (high_score_threshold, high_support_threshold, high_best_match_threshold)
        self._med = (medium_score_threshold, medium_support_threshold, medium_best_match_threshold)
        self._high_outages = high_outage_threshold
        self._med_outages = medium_outage_threshold

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify(
        self,
        query: ActivityCase,  # noqa: ARG002  (reserved for future query-level features)
        matches: list[SimilarityMatch],
        separation: OutlierSeparation,
        outages_represented: int = 0,
    ) -> ConfidenceResult:
        """Return a :class:`ConfidenceResult` for the given evidence.

        Args:
            query: The planned activity being estimated (reserved for future
                query-level adjustments).
            matches: Neighbours selected by
                :class:`~outage_uncertainty.retrieval.neighbor_selector.NeighborSelector`.
            separation: Outlier separation result from
                :class:`~outage_uncertainty.uncertainty.outlier_handler.OutlierHandler`.
            outages_represented: Number of distinct outages the analogue pool
                spans.  When ``0`` (default) the outage diversity gate is not
                applied, preserving backward compatibility.  Pass the value
                from ``retrieval_summary["outages_represented"]`` in Stage D to
                enable the gate.

        Returns:
            :class:`ConfidenceResult` with scalar score, tier, and rationale.
        """
        if not matches or separation.n_routine == 0:
            return ConfidenceResult(
                score=0.0,
                tier="low",
                rationale="No usable historical analogues found.",
                uncertainty_type="epistemic",
                recommended_action=(
                    "No historical data found — SME review or field walkdown "
                    "required before committing to schedule."
                ),
            )

        score = self._compute_score(matches, separation.n_routine)
        best = max(m.total_score for m in matches)
        n = separation.n_routine

        tier, rationale = self._classify_tier(score, n, best, outages_represented)
        cv = self._compute_cv(separation.routine)
        uncertainty_type, recommended_action = self._classify_uncertainty(
            tier, cv, separation.extended_fraction
        )
        return ConfidenceResult(
            score=score,
            tier=tier,
            rationale=rationale,
            uncertainty_type=uncertainty_type,
            recommended_action=recommended_action,
        )

    # Backwards-compatible scalar interface (used by tests expecting a float)
    def score(
        self,
        query: ActivityCase,
        matches: list[SimilarityMatch],
        durations: list[float],
    ) -> float:
        """Return confidence score as a plain float (legacy interface).

        Prefer :meth:`classify` in new code.
        """
        from outage_uncertainty.uncertainty.outlier_handler import OutlierSeparation
        n = len(durations)
        sep = OutlierSeparation(
            routine=durations,
            routine_weights=[1.0 / n] * n if n > 0 else [],
        )
        return self.classify(query, matches, sep).score

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_score(
        self, matches: list[SimilarityMatch], n_routine: int
    ) -> float:
        # Weighted average similarity using relevance_weights (sum to 1.0)
        weighted_sim = sum(m.relevance_weight * m.total_score for m in matches)
        support_factor = min(n_routine / self._sup_sat, 1.0)
        raw = self._sim_w * weighted_sim + (1.0 - self._sim_w) * support_factor
        return max(0.0, min(1.0, raw))

    def _classify_tier(
        self, score: float, n_routine: int, best_match: float,
        outages_represented: int = 0,
    ) -> tuple[str, str]:
        high_sc, high_n, high_best = self._high
        med_sc, med_n, med_best = self._med

        # Outage diversity gate: applied only when outages_represented > 0
        # (value of 0 means "not provided" — gate disabled for backward compat).
        outage_gate_active = outages_represented > 0
        outage_suffix = (
            f", outages={outages_represented}" if outage_gate_active else ""
        )

        if score >= high_sc and n_routine >= high_n and best_match >= high_best:
            if outage_gate_active and outages_represented < self._high_outages:
                # Sufficient analogue count but data spans too few outages;
                # cap at medium to avoid over-claiming cross-cycle validity.
                if outages_represented >= self._med_outages:
                    return (
                        "medium",
                        f"Moderate evidence (outage diversity cap): score={score:.2f}, "
                        f"n={n_routine}, best_match={best_match:.2f}"
                        f"{outage_suffix}. "
                        f"Needs ≥{self._high_outages} outages for 'high' tier.",
                    )
                return (
                    "low",
                    f"Weak evidence (outage diversity cap): score={score:.2f}, "
                    f"n={n_routine}, best_match={best_match:.2f}"
                    f"{outage_suffix}. "
                    f"Needs ≥{self._med_outages} outages for 'medium' tier.",
                )
            return (
                "high",
                f"Strong evidence: score={score:.2f}, n={n_routine}, "
                f"best_match={best_match:.2f}{outage_suffix}.",
            )

        if score >= med_sc and n_routine >= med_n and best_match >= med_best:
            # No outage gate at medium tier — it is already a "use with caution"
            # tier; further penalizing for outage diversity would be too strict.
            return (
                "medium",
                f"Moderate evidence: score={score:.2f}, n={n_routine}, "
                f"best_match={best_match:.2f}{outage_suffix}.",
            )

        return (
            "low",
            f"Weak evidence: score={score:.2f}, n={n_routine}, "
            f"best_match={best_match:.2f}{outage_suffix}. Expert review recommended.",
        )

    @staticmethod
    def _compute_cv(routine: list[float]) -> float:
        """Coefficient of variation of the routine duration pool."""
        if len(routine) < 2:
            return 0.0
        mu = sum(routine) / len(routine)
        if mu <= 0.0:
            return 0.0
        var = sum((x - mu) ** 2 for x in routine) / (len(routine) - 1)
        return (var ** 0.5) / mu

    @staticmethod
    def _classify_uncertainty(
        tier: str, cv: float, extended_fraction: float
    ) -> tuple[str, str]:
        """Map confidence signals to an uncertainty type and recommended action.

        Decision logic (in priority order):
        1. Low tier → epistemic regardless of spread (not enough data to
           characterise variability reliably).
        2. High disruption fraction (≥ 25 %) → mixed (even if confidence is
           high, the process is not under control).
        3. High CV (≥ 0.5) → aleatory (genuine execution variability).
        4. Medium tier → mixed (moderate coverage, moderate spread).
        5. High tier + low CV + low disruption → aleatory (well-characterised).
        """
        if tier == "low":
            return (
                "epistemic",
                "Weak analogue coverage — SME review or field walkdown required "
                "before committing to schedule.",
            )
        if extended_fraction >= 0.25:
            return (
                "mixed",
                "Significant disruption rate in historical analogues — add schedule "
                "contingency and consider pre-job walkdown.",
            )
        if cv >= 0.5:
            return (
                "aleatory",
                "High natural execution variability — add schedule float proportional "
                "to the P90–P50 spread.",
            )
        if tier == "medium":
            return (
                "mixed",
                "Moderate analogue coverage — use P80 estimate with awareness; "
                "contingency buffer advisable.",
            )
        return (
            "aleatory",
            "Estimate is well-characterised; variability is natural — minor schedule "
            "contingency may suffice.",
        )
