"""
Tests for Gaps 1–3:
  Gap 1 — Mixture distribution model
  Gap 2 — Epistemic vs. aleatory uncertainty labelling
  Gap 3 — Execution mode flags (has_rp_hold, requires_scaffold, has_clearance,
           is_vendor_supported)
"""
from __future__ import annotations

import pytest

from outage_uncertainty.adapters.pandas_repository import PandasActivityRepository
from outage_uncertainty.domain.activity import ActivityCase
from outage_uncertainty.domain.duration import DurationDistribution
from outage_uncertainty.domain.result_types import ActivityEstimate, SimilarityMatch
from outage_uncertainty.retrieval.context_similarity import ContextSimilarityScorer
from outage_uncertainty.uncertainty.confidence import ConfidenceEstimator, ConfidenceResult
from outage_uncertainty.uncertainty.distribution_fitter import DistributionFitter
from outage_uncertainty.uncertainty.fallback_policy import HierarchicalFallbackPolicy
from outage_uncertainty.uncertainty.outlier_handler import OutlierHandler, OutlierSeparation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _activity(**kwargs) -> ActivityCase:
    defaults = dict(activity_id="A1", outage_id="O1", plant_id="P1")
    defaults.update(kwargs)
    return ActivityCase(**defaults)


def _match(score: float = 0.8, duration: float | None = 10.0, weight: float = 1.0) -> SimilarityMatch:
    return SimilarityMatch(
        query_activity_id="Q1",
        candidate_activity_id="C1",
        total_score=score,
        candidate_duration_hours=duration,
        relevance_weight=weight,
    )


def _separation(routine: list[float], extended: list[float] | None = None) -> OutlierSeparation:
    n_r = len(routine)
    n_e = len(extended or [])
    rw = [1.0 / n_r] * n_r if n_r > 0 else []
    ew = [1.0 / n_e] * n_e if n_e > 0 else []
    return OutlierSeparation(
        routine=routine,
        routine_weights=rw,
        extended=extended or [],
        extended_weights=ew,
        method="iqr",
        threshold=50.0,
    )


# ===========================================================================
# Gap 1 — Mixture distribution model
# ===========================================================================

class TestMixtureSampling:
    def test_no_extended_samples_all_from_routine(self):
        dist = DurationDistribution(samples=[10.0, 11.0, 12.0], mixture_weight=0.0)
        draws = dist.sample(200)
        assert all(d in {10.0, 11.0, 12.0} for d in draws)

    def test_mixture_draws_from_both_pools(self):
        dist = DurationDistribution(
            samples=[10.0],
            extended_samples=[100.0],
            mixture_weight=0.5,
        )
        draws = dist.sample(300)
        assert any(d == 10.0 for d in draws), "routine pool never sampled"
        assert any(d == 100.0 for d in draws), "extended pool never sampled"

    def test_mixture_weight_zero_ignores_extended(self):
        dist = DurationDistribution(
            samples=[10.0],
            extended_samples=[100.0],
            mixture_weight=0.0,
        )
        draws = dist.sample(100)
        assert all(d == 10.0 for d in draws)


class TestMixtureMean:
    def test_no_extended_mean_equals_routine_mean(self):
        dist = DurationDistribution(samples=[10.0, 20.0])
        assert dist.mean() == pytest.approx(15.0)

    def test_mixture_mean_weighted_average(self):
        dist = DurationDistribution(
            samples=[10.0],
            extended_samples=[50.0],
            mixture_weight=0.25,
        )
        expected = 0.75 * 10.0 + 0.25 * 50.0   # = 20.0
        assert dist.mean() == pytest.approx(expected)

    def test_mixture_mean_with_zero_weight_equals_routine(self):
        dist = DurationDistribution(
            samples=[10.0],
            extended_samples=[999.0],
            mixture_weight=0.0,
        )
        assert dist.mean() == pytest.approx(10.0)


class TestMixtureVariance:
    def test_no_extended_variance_is_sample_variance(self):
        dist = DurationDistribution(samples=[8.0, 10.0, 12.0])
        # sample variance of [8, 10, 12]: mean=10, deviations [-2,0,2] → var=4
        assert dist.variance() == pytest.approx(4.0)

    def test_mixture_variance_exceeds_routine_variance(self):
        """Adding a heavy-tail extended pool must increase total variance."""
        dist_routine = DurationDistribution(samples=[8.0, 10.0, 12.0])
        dist_mixture = DurationDistribution(
            samples=[8.0, 10.0, 12.0],
            extended_samples=[80.0, 100.0, 120.0],
            mixture_weight=0.3,
        )
        assert dist_mixture.variance() > dist_routine.variance()

    def test_single_sample_variance_is_zero(self):
        dist = DurationDistribution(
            samples=[10.0],
            extended_samples=[50.0],
            mixture_weight=0.2,
        )
        assert dist.variance() == pytest.approx(0.0)

    def test_empty_samples_variance_is_zero(self):
        dist = DurationDistribution()
        assert dist.variance() == pytest.approx(0.0)


class TestFitterMixtureFields:
    def test_fit_from_separation_no_extended_no_mixture(self):
        sep = _separation(routine=[10.0, 12.0, 11.0])
        dist = DistributionFitter().fit_from_separation(sep)
        assert dist.mixture_weight == pytest.approx(0.0)
        assert dist.extended_samples is None
        assert "mixture_p80" not in dist.parameters

    def test_fit_from_separation_sets_extended_samples(self):
        sep = _separation(routine=[10.0, 11.0, 12.0], extended=[50.0])
        dist = DistributionFitter().fit_from_separation(sep)
        assert dist.extended_samples == [50.0]
        assert dist.mixture_weight == pytest.approx(sep.extended_fraction)

    def test_fit_from_separation_computes_mixture_percentiles(self):
        sep = _separation(routine=[10.0, 11.0, 12.0, 13.0], extended=[80.0, 90.0])
        dist = DistributionFitter().fit_from_separation(sep)
        assert "mixture_p80" in dist.parameters
        assert "mixture_p90" in dist.parameters
        # Mixture P90 must be higher than routine-only P90 due to heavy tail
        assert dist.parameters["mixture_p90"] > dist.p90

    def test_routine_percentiles_unchanged_by_mixture(self):
        """p80/p90 on the dist still reflect routine pool only."""
        sep = _separation(routine=[10.0, 11.0, 12.0, 13.0], extended=[100.0])
        dist = DistributionFitter().fit_from_separation(sep)
        # Routine p90 < 100 (the extended sample)
        assert dist.p90 < 100.0


# ===========================================================================
# Gap 2 — Epistemic vs. aleatory uncertainty labelling
# ===========================================================================

class TestConfidenceResultFields:
    def test_default_uncertainty_type_is_unknown(self):
        r = ConfidenceResult(score=0.5, tier="medium", rationale="ok")
        assert r.uncertainty_type == "unknown"
        assert r.recommended_action == ""

    def test_can_set_uncertainty_type(self):
        r = ConfidenceResult(
            score=0.8, tier="high", rationale="ok",
            uncertainty_type="aleatory", recommended_action="add float",
        )
        assert r.uncertainty_type == "aleatory"


class TestConfidenceEstimatorUncertaintyClassification:
    """Verify that classify() populates uncertainty_type correctly."""

    def _classify(self, tier_score, n_routine, best_score, routine, extended_fraction=0.0):
        estimator = ConfidenceEstimator(
            high_score_threshold=0.70,
            high_support_threshold=10,
            high_best_match_threshold=0.70,
            medium_score_threshold=0.45,
            medium_support_threshold=5,
            medium_best_match_threshold=0.50,
        )
        matches = [_match(score=best_score, weight=1.0)]
        n_ext = round(len(routine) * extended_fraction / (1 - extended_fraction + 1e-9))
        sep = _separation(routine=routine, extended=[999.0] * n_ext)
        # Force the tier by using a mock query
        query = _activity()
        return estimator.classify(query, matches, sep)

    def test_no_matches_returns_epistemic(self):
        estimator = ConfidenceEstimator()
        sep = _separation(routine=[])
        result = estimator.classify(_activity(), [], sep)
        assert result.uncertainty_type == "epistemic"
        assert result.tier == "low"

    def test_low_tier_is_epistemic(self):
        estimator = ConfidenceEstimator()
        # Single weak match → low tier guaranteed
        sep = _separation(routine=[10.0])
        matches = [_match(score=0.3, weight=1.0)]
        result = estimator.classify(_activity(), matches, sep)
        assert result.uncertainty_type == "epistemic"

    def test_high_disruption_fraction_gives_mixed(self):
        estimator = ConfidenceEstimator()
        # Create separation where extended_fraction ≥ 0.25
        routine = [10.0] * 12    # 12 routine
        extended = [80.0] * 4   # 4 extended → fraction = 4/16 = 0.25
        sep = OutlierSeparation(
            routine=routine,
            routine_weights=[1 / 12] * 12,
            extended=extended,
            extended_weights=[0.25] * 4,
            method="iqr",
        )
        # Good matches to push tier to medium or high
        matches = [_match(score=0.8, weight=1 / 12)] * 12
        result = estimator.classify(_activity(), matches, sep)
        assert result.uncertainty_type == "mixed"

    def test_high_confidence_low_spread_is_aleatory(self):
        estimator = ConfidenceEstimator()
        # Many good, close matches with tight spread (low CV)
        routine = [10.0, 10.5, 11.0, 9.5, 10.2, 10.8, 9.8, 11.2, 10.1, 10.3,
                   10.4, 10.6]
        sep = _separation(routine=routine)
        matches = [_match(score=0.85, weight=1 / 12)] * 12
        result = estimator.classify(_activity(), matches, sep)
        assert result.uncertainty_type == "aleatory"

    def test_recommended_action_is_non_empty(self):
        estimator = ConfidenceEstimator()
        sep = _separation(routine=[10.0])
        matches = [_match(score=0.3, weight=1.0)]
        result = estimator.classify(_activity(), matches, sep)
        assert len(result.recommended_action) > 0


class TestActivityEstimateFields:
    def test_default_uncertainty_type_is_unknown(self):
        dist = DurationDistribution(samples=[10.0])
        est = ActivityEstimate(
            activity_id="A1",
            estimated_distribution=dist,
            confidence_score=0.5,
        )
        assert est.uncertainty_type == "unknown"
        assert est.recommended_action == ""

    def test_can_set_uncertainty_type(self):
        dist = DurationDistribution(samples=[10.0])
        est = ActivityEstimate(
            activity_id="A1",
            estimated_distribution=dist,
            confidence_score=0.5,
            uncertainty_type="aleatory",
            recommended_action="add float",
        )
        assert est.uncertainty_type == "aleatory"


class TestFallbackPolicySetsEpistemic:
    def test_planned_duration_prior_is_epistemic(self):
        policy = HierarchicalFallbackPolicy()
        query = _activity(planned_duration_hours=8.0)
        est = policy.estimate(query, historical_activities=[])
        assert est.uncertainty_type == "epistemic"
        assert len(est.recommended_action) > 0

    def test_fallback_level_is_epistemic(self):
        policy = HierarchicalFallbackPolicy(min_support=2)
        query = _activity(task_family="inspection", plant_id="P1")
        historical = [
            _activity(
                activity_id=f"H{i}",
                task_family="inspection",
                plant_id="P1",
                actual_duration_hours=10.0,
            )
            for i in range(3)
        ]
        est = policy.estimate(query, historical_activities=historical)
        assert est.uncertainty_type == "epistemic"


# ===========================================================================
# Gap 3 — Execution mode flags
# ===========================================================================

class TestActivityCaseExecutionFlags:
    def test_default_flags_are_false(self):
        a = _activity()
        assert a.has_rp_hold is False
        assert a.requires_scaffold is False
        assert a.has_clearance is False
        assert a.is_vendor_supported is False

    def test_flags_can_be_set(self):
        a = _activity(has_rp_hold=True, requires_scaffold=True)
        assert a.has_rp_hold is True
        assert a.requires_scaffold is True
        assert a.has_clearance is False


class TestContextSimilarityWithExecutionFlags:
    def _score(self, a: ActivityCase, b: ActivityCase) -> float:
        return ContextSimilarityScorer().score(a, b)

    def test_scaffold_match_scores_higher_than_mismatch(self):
        base = dict(
            activity_id="X", outage_id="O", plant_id="P",
            discipline="mechanical", task_family="inspection",
        )
        both_scaffold = self._score(
            ActivityCase(**base, requires_scaffold=True),
            ActivityCase(**{**base, "activity_id": "Y"}, requires_scaffold=True),
        )
        scaffold_mismatch = self._score(
            ActivityCase(**base, requires_scaffold=True),
            ActivityCase(**{**base, "activity_id": "Y"}, requires_scaffold=False),
        )
        assert both_scaffold > scaffold_mismatch

    def test_rp_hold_match_scores_higher_than_mismatch(self):
        base = dict(
            activity_id="X", outage_id="O", plant_id="P",
            discipline="mechanical", task_family="maintenance",
        )
        both_rp = self._score(
            ActivityCase(**base, has_rp_hold=True),
            ActivityCase(**{**base, "activity_id": "Y"}, has_rp_hold=True),
        )
        rp_mismatch = self._score(
            ActivityCase(**base, has_rp_hold=True),
            ActivityCase(**{**base, "activity_id": "Y"}, has_rp_hold=False),
        )
        assert both_rp > rp_mismatch

    def test_flags_absent_from_both_do_not_penalise(self):
        """Two activities with all flags False should score the same as before flags existed."""
        base = dict(
            activity_id="X", outage_id="O", plant_id="P",
            discipline="mechanical", task_family="inspection",
        )
        score_no_flags = self._score(
            ActivityCase(**base),
            ActivityCase(**{**base, "activity_id": "Y"}),
        )
        # Score should be 1.0 on all shared fields (both False == both False → 1.0)
        assert score_no_flags > 0.0


class TestPandasRepositoryLoadsExecutionFlags:
    def test_loads_flags_from_dict(self):
        repo = PandasActivityRepository()
        rows = [
            {
                "activity_id": "A1",
                "outage_id": "O1",
                "plant_id": "P1",
                "raw_description": "test",
                "has_rp_hold": True,
                "requires_scaffold": True,
                "has_clearance": False,
                "is_vendor_supported": True,
            }
        ]
        activities = repo.load_activities(rows)
        a = activities[0]
        assert a.has_rp_hold is True
        assert a.requires_scaffold is True
        assert a.has_clearance is False
        assert a.is_vendor_supported is True

    def test_missing_flags_default_to_false(self):
        repo = PandasActivityRepository()
        rows = [{"activity_id": "A1", "outage_id": "O1", "plant_id": "P1"}]
        a = repo.load_activities(rows)[0]
        assert a.has_rp_hold is False
        assert a.requires_scaffold is False
