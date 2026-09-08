"""
Tests for Gaps 4 and 7:
  Gap 4 — Dependency pattern similarity scorer
  Gap 7 — Richer schedule-level metrics
"""
from __future__ import annotations

import math

import pytest

from outage_uncertainty.domain.activity import ActivityCase
from outage_uncertainty.domain.result_types import SimulationResult, SimilarityMatch
from outage_uncertainty.retrieval.context_similarity import ContextSimilarityScorer
from outage_uncertainty.retrieval.dependency_similarity import (
    DependencyPatternScorer,
    topological_role,
)
from outage_uncertainty.retrieval.lexical_similarity import LexicalSimilarityScorer
from outage_uncertainty.retrieval.semantic_similarity import SemanticSimilarityScorer
from outage_uncertainty.retrieval.similarity_engine import SimilarityAggregator, SimilarityEngine
from outage_uncertainty.schedule_risk.cp_analyzer import CriticalPathRiskAnalyzer, _point_biserial_corr
from outage_uncertainty.schedule_risk.robustness_metrics import RobustnessMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _activity(
    activity_id: str = "A1",
    predecessors: list[str] | None = None,
    successors: list[str] | None = None,
) -> ActivityCase:
    return ActivityCase(
        activity_id=activity_id,
        outage_id="O1",
        plant_id="P1",
        predecessor_ids=predecessors or [],
        successor_ids=successors or [],
    )


def _sim(cp_times: list[float], cp_paths: list[list[str]]) -> SimulationResult:
    activity_criticality: dict[str, int] = {}
    for path in cp_paths:
        for act_id in path:
            activity_criticality[act_id] = activity_criticality.get(act_id, 0) + 1
    return SimulationResult(
        cp_times=cp_times,
        cp_paths=cp_paths,
        activity_criticality=activity_criticality,
    )


# ===========================================================================
# Gap 4 — Dependency pattern similarity
# ===========================================================================

class TestTopologicalRole:
    def test_isolated(self):
        assert topological_role(_activity(predecessors=[], successors=[])) == "isolated"

    def test_source(self):
        assert topological_role(_activity(predecessors=[], successors=["B"])) == "source"

    def test_sink(self):
        assert topological_role(_activity(predecessors=["A"], successors=[])) == "sink"

    def test_chain(self):
        assert topological_role(_activity(predecessors=["A"], successors=["C"])) == "chain"

    def test_merge(self):
        assert topological_role(_activity(predecessors=["A", "B"], successors=["C"])) == "merge"

    def test_merge_no_successors_classified_as_sink(self):
        # sink check fires before merge: multiple predecessors + no successors → sink
        assert topological_role(_activity(predecessors=["A", "B"], successors=[])) == "sink"

    def test_burst(self):
        assert topological_role(_activity(predecessors=["A"], successors=["C", "D"])) == "burst"

    def test_burst_no_predecessors_classified_as_source(self):
        # source check fires before burst: no predecessors + multiple successors → source
        assert topological_role(_activity(predecessors=[], successors=["C", "D"])) == "source"

    def test_internal(self):
        assert topological_role(
            _activity(predecessors=["A", "B"], successors=["C", "D"])
        ) == "internal"


class TestDependencyPatternScorer:
    def setup_method(self):
        self.scorer = DependencyPatternScorer()

    def test_identical_roles_score_one(self):
        a = _activity(predecessors=["X"], successors=["Y"])   # chain
        b = _activity(predecessors=["P"], successors=["Q"])   # chain
        assert self.scorer.score(a, b) == pytest.approx(1.0)

    def test_both_isolated_score_one(self):
        a = _activity()
        b = _activity(activity_id="B")
        assert self.scorer.score(a, b) == pytest.approx(1.0)

    def test_source_vs_chain_partial_credit(self):
        source = _activity(predecessors=[], successors=["B"])
        chain  = _activity(predecessors=["A"], successors=["C"])
        score  = self.scorer.score(source, chain)
        assert 0.0 < score < 1.0

    def test_source_vs_sink_low_score(self):
        source = _activity(predecessors=[], successors=["B"])
        sink   = _activity(predecessors=["A"], successors=[])
        score  = self.scorer.score(source, sink)
        # No partial credit defined for source↔sink
        # role_sim = 0; degree_sim can still contribute via degree match
        assert score < 0.6

    def test_same_degree_scores_higher_than_different_degree(self):
        a1 = _activity(predecessors=["X", "Y"], successors=["Z"])     # merge, in=2
        a2 = _activity(predecessors=["P", "Q"], successors=["R"])     # merge, in=2
        b  = _activity(predecessors=["X", "Y", "W"], successors=["Z"]) # merge, in=3
        score_same = self.scorer.score(a1, a2)
        score_diff = self.scorer.score(a1, b)
        assert score_same > score_diff

    def test_degree_similarity_decays_exponentially(self):
        scorer = DependencyPatternScorer(degree_scale=2.0, role_weight=0.0)
        a = _activity(predecessors=["X"], successors=[])
        b = _activity(predecessors=["X", "Y"], successors=[])  # in-degree differs by 1
        expected_in_sim  = math.exp(-1 / 2.0)
        expected_out_sim = math.exp(0)   # both out-degree 0
        expected = 0.5 * (expected_in_sim + expected_out_sim)
        assert scorer.score(a, b) == pytest.approx(expected, abs=1e-9)

    def test_score_in_zero_one(self):
        a = _activity(predecessors=["A", "B"], successors=["C"])
        b = _activity(predecessors=[], successors=["D", "E", "F"])
        score = self.scorer.score(a, b)
        assert 0.0 <= score <= 1.0


class TestSimilarityEngineWithDependencyScorer:
    def test_dependency_score_populated(self):
        scorer = DependencyPatternScorer()
        engine = SimilarityEngine(
            lexical_scorer=LexicalSimilarityScorer(),
            semantic_scorer=SemanticSimilarityScorer(),
            context_scorer=ContextSimilarityScorer(),
            aggregator=SimilarityAggregator(weights={
                "lexical": 0.15, "semantic": 0.30,
                "context": 0.35, "dependency": 0.20,
            }),
            dependency_scorer=scorer,
        )
        a = ActivityCase(
            activity_id="A", outage_id="O", plant_id="P",
            raw_description="replace pump",
            predecessor_ids=["X"], successor_ids=["Y"],
        )
        b = ActivityCase(
            activity_id="B", outage_id="O", plant_id="P",
            raw_description="replace pump",
            predecessor_ids=["P"], successor_ids=["Q"],
        )
        match = engine.compare(a, b)
        # Both are chain nodes → dependency_score should be 1.0
        assert match.dependency_score == pytest.approx(1.0)
        assert 0.0 <= match.total_score <= 1.0

    def test_no_dependency_scorer_gives_zero_dependency_score(self):
        engine = SimilarityEngine(
            lexical_scorer=LexicalSimilarityScorer(),
            semantic_scorer=SemanticSimilarityScorer(),
            context_scorer=ContextSimilarityScorer(),
            aggregator=SimilarityAggregator(),
        )
        a = ActivityCase(activity_id="A", outage_id="O", plant_id="P")
        b = ActivityCase(activity_id="B", outage_id="O", plant_id="P")
        match = engine.compare(a, b)
        assert match.dependency_score == pytest.approx(0.0)


class TestSimilarityAggregator:
    def test_dependency_zero_weight_no_effect(self):
        agg = SimilarityAggregator()  # default: dependency weight = 0
        s1 = agg.combine(lexical=0.5, semantic=0.5, context=0.5)
        s2 = agg.combine(lexical=0.5, semantic=0.5, context=0.5, dependency=1.0)
        assert s1 == pytest.approx(s2)

    def test_dependency_contributes_when_weighted(self):
        agg = SimilarityAggregator(weights={
            "lexical": 0.15, "semantic": 0.30, "context": 0.35, "dependency": 0.20,
        })
        s_no_dep  = agg.combine(lexical=0.5, semantic=0.5, context=0.5, dependency=0.0)
        s_full_dep = agg.combine(lexical=0.5, semantic=0.5, context=0.5, dependency=1.0)
        assert s_full_dep > s_no_dep

    def test_backward_compat_no_dependency_kwarg(self):
        agg = SimilarityAggregator()
        # Old call signature — must not raise
        result = agg.combine(lexical=0.3, semantic=0.4, context=0.3)
        assert isinstance(result, float)


class TestSimilarityMatchHasDependencyScore:
    def test_default_dependency_score_is_zero(self):
        m = SimilarityMatch(
            query_activity_id="Q", candidate_activity_id="C", total_score=0.5
        )
        assert m.dependency_score == pytest.approx(0.0)


# ===========================================================================
# Gap 7 — Richer schedule-level metrics
# ===========================================================================

class TestRobustnessMetrics:
    def setup_method(self):
        self.rm = RobustnessMetrics()

    def _make_sim(self, cp_times: list[float]) -> SimulationResult:
        return SimulationResult(cp_times=cp_times, cp_paths=[], activity_criticality={})

    def test_empty_sim_returns_zeros(self):
        result = self.rm.compute(self._make_sim([]), baseline_cp_time=100.0)
        assert result["robustness"] == pytest.approx(0.0)
        assert result["schedule_std_dev"] == pytest.approx(0.0)

    def test_robustness_all_on_time(self):
        sim = self._make_sim([90.0, 95.0, 100.0])
        r = self.rm.compute(sim, baseline_cp_time=100.0)
        assert r["robustness"] == pytest.approx(1.0)

    def test_robustness_none_on_time(self):
        sim = self._make_sim([110.0, 120.0])
        r = self.rm.compute(sim, baseline_cp_time=100.0)
        assert r["robustness"] == pytest.approx(0.0)

    def test_expected_delay_positive_when_late(self):
        sim = self._make_sim([110.0, 120.0, 130.0])
        r = self.rm.compute(sim, baseline_cp_time=100.0)
        assert r["expected_delay"] > 0.0
        assert r["expected_delay"] == pytest.approx(20.0)  # mean=120, baseline=100

    def test_expected_delay_zero_when_early(self):
        sim = self._make_sim([80.0, 90.0, 95.0])
        r = self.rm.compute(sim, baseline_cp_time=100.0)
        assert r["expected_delay"] == pytest.approx(0.0)

    def test_schedule_std_dev_is_positive(self):
        sim = self._make_sim([90.0, 100.0, 110.0])
        r = self.rm.compute(sim, baseline_cp_time=100.0)
        assert r["schedule_std_dev"] > 0.0

    def test_p80_finish_geq_mean(self):
        sim = self._make_sim([80.0, 90.0, 100.0, 110.0, 120.0])
        r = self.rm.compute(sim, baseline_cp_time=100.0)
        assert r["p80_finish"] >= r["mean_finish"]

    def test_all_keys_present(self):
        sim = self._make_sim([100.0])
        r = self.rm.compute(sim, baseline_cp_time=100.0)
        for key in ("robustness", "schedule_std_dev", "expected_delay",
                    "mean_finish", "p80_finish", "p90_finish"):
            assert key in r


class TestCriticalPathRiskAnalyzer:
    def setup_method(self):
        self.analyzer = CriticalPathRiskAnalyzer()

    def test_empty_sim_returns_zeros(self):
        result = self.analyzer.analyze(
            SimulationResult(cp_times=[], cp_paths=[], activity_criticality={}),
            baseline_cp_time=100.0,
        )
        assert result["robustness"] == pytest.approx(0.0)
        assert result["criticality_index"] == {}

    def test_schedule_variance_positive_for_variable_sim(self):
        sim = _sim(
            cp_times=[90.0, 100.0, 110.0],
            cp_paths=[["A"], ["A"], ["A"]],
        )
        result = self.analyzer.analyze(sim, baseline_cp_time=100.0)
        assert result["schedule_variance"] > 0.0
        assert result["schedule_std_dev"] == pytest.approx(result["schedule_variance"] ** 0.5)

    def test_criticality_index_sums_correctly(self):
        sim = _sim(
            cp_times=[100.0, 100.0, 100.0, 100.0],
            cp_paths=[["A"], ["A"], ["B"], ["B"]],
        )
        result = self.analyzer.analyze(sim, baseline_cp_time=100.0)
        assert result["criticality_index"]["A"] == pytest.approx(0.5)
        assert result["criticality_index"]["B"] == pytest.approx(0.5)

    def test_expected_finish_when_critical_higher_for_late_activity(self):
        # Activity A is on CP only in the late runs; B is on CP in the early ones.
        sim = _sim(
            cp_times=[80.0, 80.0, 120.0, 120.0],
            cp_paths=[["B"], ["B"], ["A"], ["A"]],
        )
        result = self.analyzer.analyze(sim, baseline_cp_time=100.0)
        assert result["expected_finish_when_critical"]["A"] > \
               result["expected_finish_when_critical"]["B"]

    def test_expected_drag_positive_for_late_activity(self):
        # A is on CP only in the two late runs (120h); off CP in the early runs (80h).
        sim = _sim(
            cp_times=[80.0, 80.0, 120.0, 120.0],
            cp_paths=[["B"], ["B"], ["A"], ["A"]],
        )
        result = self.analyzer.analyze(sim, baseline_cp_time=100.0)
        # Drag for A = E[finish|A on CP] - E[finish|A off CP] = 120 - 80 = 40
        assert result["expected_drag"]["A"] == pytest.approx(40.0)

    def test_cp_sensitivity_positive_for_late_activity(self):
        sim = _sim(
            cp_times=[80.0, 80.0, 120.0, 120.0],
            cp_paths=[["B"], ["B"], ["A"], ["A"]],
        )
        result = self.analyzer.analyze(sim, baseline_cp_time=100.0)
        assert result["cp_sensitivity"]["A"] > 0.0

    def test_cp_sensitivity_in_minus_one_to_one(self):
        sim = _sim(
            cp_times=[70.0, 90.0, 110.0, 130.0],
            cp_paths=[["X"], ["X"], ["Y"], ["Y"]],
        )
        result = self.analyzer.analyze(sim, baseline_cp_time=100.0)
        for act_id, sens in result["cp_sensitivity"].items():
            assert -1.0 <= sens <= 1.0, f"sensitivity for {act_id} out of range: {sens}"

    def test_all_new_keys_present(self):
        sim = _sim(cp_times=[100.0], cp_paths=[["A"]])
        result = self.analyzer.analyze(sim, baseline_cp_time=100.0)
        for key in (
            "schedule_variance", "schedule_std_dev", "expected_delay",
            "expected_finish_when_critical", "expected_drag", "cp_sensitivity",
        ):
            assert key in result, f"missing key: {key}"


class TestPointBiserialCorr:
    def test_perfect_positive_correlation(self):
        # x=1 always paired with higher y
        x = [0, 0, 1, 1]
        y = [10.0, 10.0, 20.0, 20.0]
        r = _point_biserial_corr(x, y)
        assert r == pytest.approx(1.0)

    def test_zero_correlation_equal_means(self):
        x = [0, 1, 0, 1]
        y = [10.0, 10.0, 10.0, 10.0]
        r = _point_biserial_corr(x, y)
        assert r == pytest.approx(0.0)

    def test_all_same_class_returns_zero(self):
        x = [1, 1, 1]
        y = [10.0, 20.0, 30.0]
        assert _point_biserial_corr(x, y) == pytest.approx(0.0)

    def test_result_in_range(self):
        x = [0, 1, 1, 0, 1]
        y = [5.0, 15.0, 10.0, 8.0, 20.0]
        r = _point_biserial_corr(x, y)
        assert -1.0 <= r <= 1.0
