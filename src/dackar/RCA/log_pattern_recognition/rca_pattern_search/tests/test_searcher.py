"""Unit tests for rca_pattern_search.searcher"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from ..config import SearchConfig
from ..indexer import IncidentIndex
from ..models import IncidentFingerprint, SearchResult
from ..searcher import PatternSearcher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T0 = datetime(2024, 1, 1, 0, 0, 0)

CFG = SearchConfig(
    beta=0.1,
    delta=0.5,
    min_jaccard=0.3,
    top_k=3,
    weight_profile="equal",
    freq_threshold=10,
)

CFG_NO_FILTER = SearchConfig(
    min_jaccard=0.0,
    top_k=10,
    weight_profile="equal",
    freq_threshold=10,
)


def _fp(
    episode_id: str,
    event_types: list[str],
    freq: int = 2,
    known_rca: str | None = None,
) -> IncidentFingerprint:
    types = sorted(event_types)
    return IncidentFingerprint(
        episode_id=episode_id,
        asset_id="A1",
        window_start=T0,
        window_end=T0 + timedelta(hours=1),
        density=0.01,
        event_set=frozenset(types),
        event_seq=types,
        freq_vec={t: freq for t in types},
        known_rca=known_rca,
    )


def _build_index(*fingerprints: IncidentFingerprint, cfg: SearchConfig = CFG) -> IncidentIndex:
    idx = IncidentIndex(cfg)
    idx.add_batch(list(fingerprints))
    return idx


# ---------------------------------------------------------------------------
# _resolve_weights
# ---------------------------------------------------------------------------

class TestResolveWeights:
    def test_equal(self):
        s = PatternSearcher(_build_index(), CFG)
        a, b, g = s._resolve_weights("equal")
        assert pytest.approx(a) == 1 / 3
        assert pytest.approx(b) == 1 / 3
        assert pytest.approx(g) == 1 / 3

    def test_flooding(self):
        s = PatternSearcher(_build_index(), CFG)
        a, b, g = s._resolve_weights("flooding")
        assert g > a and g > b

    def test_cascade(self):
        s = PatternSearcher(_build_index(), CFG)
        a, b, g = s._resolve_weights("cascade")
        assert b > a and b > g

    def test_custom(self):
        cfg = SearchConfig(weight_profile="custom", alpha=0.5, beta_w=0.3, gamma=0.2)
        s = PatternSearcher(_build_index(cfg=cfg), cfg)
        a, b, g = s._resolve_weights("custom")
        assert pytest.approx(a) == 0.5
        assert pytest.approx(b) == 0.3
        assert pytest.approx(g) == 0.2

    def test_unknown_profile_raises(self):
        s = PatternSearcher(_build_index(), CFG)
        with pytest.raises(ValueError):
            s._resolve_weights("unknown_profile")


# ---------------------------------------------------------------------------
# search() — empty / no-candidate paths
# ---------------------------------------------------------------------------

class TestSearchEdgeCases:
    def test_empty_index_returns_empty(self):
        idx = IncidentIndex(CFG)
        s = PatternSearcher(idx, CFG)
        q = _fp("Q", ["A", "B"])
        assert s.search(q) == []

    def test_no_shared_event_types_returns_empty(self):
        idx = _build_index(_fp("EP_1", ["X", "Y"]))
        s = PatternSearcher(idx, CFG)
        q = _fp("Q", ["A", "B"])
        assert s.search(q) == []

    def test_below_jaccard_threshold_filtered(self):
        # EP_1 shares 1/4 events with query → Jaccard = 1/4 = 0.25 < min_jaccard=0.3
        idx = _build_index(_fp("EP_1", ["A", "B", "C", "D"]))
        s = PatternSearcher(idx, CFG)
        q = _fp("Q", ["A"])    # Jaccard({A}, {A,B,C,D}) = 1/4 = 0.25
        assert s.search(q) == []

    def test_empty_query_event_set_returns_empty(self):
        idx = _build_index(_fp("EP_1", ["A"]))
        s = PatternSearcher(idx, CFG)
        q = _fp("Q", [])
        assert s.search(q) == []


# ---------------------------------------------------------------------------
# search() — result correctness
# ---------------------------------------------------------------------------

class TestSearchResults:
    def test_returns_search_result_objects(self):
        idx = _build_index(_fp("EP_1", ["A", "B"]))
        s = PatternSearcher(idx, CFG_NO_FILTER)
        q = _fp("Q", ["A", "B"])
        results = s.search(q)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)

    def test_identical_query_scores_one(self):
        idx = _build_index(_fp("EP_1", ["A", "B"], freq=3))
        s = PatternSearcher(idx, CFG_NO_FILTER)
        q = _fp("Q", ["A", "B"], freq=3)
        results = s.search(q)
        r = results[0]
        assert r.jaccard_score == pytest.approx(1.0)
        assert r.nlcs_score == pytest.approx(1.0)
        assert r.emd_score == pytest.approx(1.0)
        assert r.combined_score == pytest.approx(1.0)

    def test_matched_and_exclusive_events(self):
        idx = _build_index(_fp("EP_1", ["A", "B", "C"]))
        s = PatternSearcher(idx, CFG_NO_FILTER)
        # Query has A, B, D — episode has A, B, C
        q = _fp("Q", ["A", "B", "D"])
        results = s.search(q)
        r = results[0]
        assert r.matched_events == {"A", "B"}
        assert r.query_only_events == {"D"}
        assert r.episode_only_events == {"C"}

    def test_known_rca_propagated(self):
        idx = _build_index(_fp("EP_1", ["A", "B"], known_rca="pump_cavitation"))
        s = PatternSearcher(idx, CFG_NO_FILTER)
        q = _fp("Q", ["A", "B"])
        results = s.search(q)
        assert results[0].known_rca == "pump_cavitation"

    def test_known_rca_none_when_not_set(self):
        idx = _build_index(_fp("EP_1", ["A", "B"]))
        s = PatternSearcher(idx, CFG_NO_FILTER)
        q = _fp("Q", ["A", "B"])
        results = s.search(q)
        assert results[0].known_rca is None

    def test_episode_window_correct(self):
        idx = _build_index(_fp("EP_1", ["A"]))
        s = PatternSearcher(idx, CFG_NO_FILTER)
        q = _fp("Q", ["A"])
        results = s.search(q)
        assert results[0].episode_window == (T0, T0 + timedelta(hours=1))

    def test_weight_profile_stored_in_result(self):
        idx = _build_index(_fp("EP_1", ["A", "B"]))
        s = PatternSearcher(idx, CFG_NO_FILTER)
        q = _fp("Q", ["A", "B"])
        results = s.search(q, weight_profile="flooding")
        assert results[0].weight_profile == "flooding"


# ---------------------------------------------------------------------------
# search() — ranking and top-k
# ---------------------------------------------------------------------------

class TestRankingAndTopK:
    def test_sorted_by_combined_score_descending(self):
        # EP_perfect shares all types; EP_partial shares one
        idx = _build_index(
            _fp("EP_perfect", ["A", "B", "C"]),
            _fp("EP_partial", ["A", "X", "Y"]),
        )
        s = PatternSearcher(idx, CFG_NO_FILTER)
        q = _fp("Q", ["A", "B", "C"])
        results = s.search(q)
        scores = [r.combined_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limit_respected(self):
        fps = [_fp(f"EP_{i}", ["A", "B"]) for i in range(10)]
        idx = _build_index(*fps)
        cfg = SearchConfig(min_jaccard=0.0, top_k=3, weight_profile="equal", freq_threshold=10)
        s = PatternSearcher(idx, cfg)
        q = _fp("Q", ["A", "B"])
        results = s.search(q)
        assert len(results) <= 3

    def test_perfect_match_ranked_first(self):
        idx = _build_index(
            _fp("EP_perfect", ["A", "B", "C"], freq=5),
            _fp("EP_poor",    ["A", "Z", "W"], freq=5),
        )
        s = PatternSearcher(idx, CFG_NO_FILTER)
        q = _fp("Q", ["A", "B", "C"], freq=5)
        results = s.search(q)
        assert results[0].episode_id == "EP_perfect"


# ---------------------------------------------------------------------------
# search() — weight profile switching
# ---------------------------------------------------------------------------

class TestWeightProfiles:
    def _make_scenario(self):
        """
        EP_flood: A dominates heavily (alarm-flooding pattern).
        EP_order: A and B balanced (clean cascade pattern).

        The EMD metric (TV distance on normalized distributions) distinguishes
        these because they have *different per-type ratios*, not just different
        total counts.
        """
        fp_flood = IncidentFingerprint(
            episode_id="EP_flood",
            asset_id="A1",
            window_start=T0, window_end=T0 + timedelta(hours=1),
            density=0.1,
            event_set=frozenset({"A", "B"}),
            event_seq=["A", "B"],
            freq_vec={"A": 20, "B": 2},   # A dominates — flooding signature
            known_rca=None,
        )
        fp_order = IncidentFingerprint(
            episode_id="EP_order",
            asset_id="A1",
            window_start=T0, window_end=T0 + timedelta(hours=1),
            density=0.01,
            event_set=frozenset({"A", "B"}),
            event_seq=["A", "B"],
            freq_vec={"A": 2, "B": 2},    # balanced — cascade signature
            known_rca=None,
        )
        return fp_flood, fp_order

    def _query(self, freq_a: int = 2, freq_b: int = 2) -> IncidentFingerprint:
        return IncidentFingerprint(
            episode_id="Q",
            asset_id="A1",
            window_start=T0, window_end=T0 + timedelta(hours=1),
            density=0.01,
            event_set=frozenset({"A", "B"}),
            event_seq=["A", "B"],
            freq_vec={"A": freq_a, "B": freq_b},
            known_rca=None,
        )

    def test_flooding_profile_favours_high_repetition(self):
        """Flooding query (A dominates) should rank EP_flood first under flooding profile.

        EMD(TV) distinguishes because EP_flood has ratio ~0.91/0.09 vs query
        ~0.90/0.10 (very close), while EP_order has ratio 0.5/0.5 (far from query).
        """
        fp_flood, fp_order = self._make_scenario()
        idx = _build_index(fp_flood, fp_order, cfg=CFG_NO_FILTER)
        s = PatternSearcher(idx, CFG_NO_FILTER)

        # Query with A dominating heavily — close to EP_flood distribution
        q = self._query(freq_a=18, freq_b=2)
        results_flooding = s.search(q, weight_profile="flooding")
        assert results_flooding[0].episode_id == "EP_flood"

    def test_equal_profile_uses_config_default(self):
        """Passing weight_profile=None should use config.weight_profile."""
        fp_flood, fp_order = self._make_scenario()
        idx = _build_index(fp_flood, fp_order, cfg=CFG_NO_FILTER)
        cfg = SearchConfig(min_jaccard=0.0, top_k=10, weight_profile="equal",
                           freq_threshold=10)
        s = PatternSearcher(idx, cfg)
        q = self._query(freq_a=2, freq_b=2)
        r_explicit = s.search(q, weight_profile="equal")
        r_default  = s.search(q, weight_profile=None)
        assert [r.episode_id for r in r_explicit] == [r.episode_id for r in r_default]

    def test_all_scores_present_regardless_of_profile(self):
        """All three metric scores are always computed and non-None."""
        fp_flood, fp_order = self._make_scenario()
        idx = _build_index(fp_flood, fp_order, cfg=CFG_NO_FILTER)
        s = PatternSearcher(idx, CFG_NO_FILTER)
        q = self._query(freq_a=2, freq_b=2)
        for profile in ("equal", "flooding", "cascade"):
            results = s.search(q, weight_profile=profile)
            for r in results:
                assert r.jaccard_score is not None
                assert r.nlcs_score is not None
                assert r.emd_score is not None
                assert r.combined_score is not None


# ---------------------------------------------------------------------------
# end-to-end: index + searcher together
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_pipeline(self):
        """Add fingerprints, search, verify ranking and event set decomposition."""
        idx = IncidentIndex(CFG_NO_FILTER)
        idx.add(_fp("EP_A", ["ALM_001", "ALM_002", "SOE_TRIP"], known_rca="pump_fault"))
        idx.add(_fp("EP_B", ["ALM_001", "ALM_003"]))
        idx.add(_fp("EP_C", ["SOE_CLOSE", "SOE_OPEN"]))

        s = PatternSearcher(idx, CFG_NO_FILTER)
        # Query close to EP_A
        q = _fp("Q", ["ALM_001", "ALM_002", "SOE_TRIP"])
        results = s.search(q)

        assert results[0].episode_id == "EP_A"
        assert results[0].known_rca == "pump_fault"
        assert results[0].jaccard_score == pytest.approx(1.0)
        # EP_C should not appear (no overlap with query)
        ep_ids = {r.episode_id for r in results}
        assert "EP_C" not in ep_ids


# ---------------------------------------------------------------------------
# EMD normalization modes
# ---------------------------------------------------------------------------

class TestEmdNormalizationModes:
    def test_tv_mode_default_backward_compatible(self):
        cfg = SearchConfig(emd_normalization_mode="tv", weight_profile="equal")
        idx = _build_index(
            _fp("ep1", ["A"], freq=10),
            _fp("ep2", ["A"], freq=2),
            cfg=cfg,
        )
        searcher = PatternSearcher(idx, cfg)
        query = _fp("q", ["A"], freq=5)
        results = searcher.search(query)
        assert len(results) == 2
        assert results[0].combined_score > 0.0

    def test_empirical_max_mode_requires_factor(self):
        cfg = SearchConfig(emd_normalization_mode="empirical_max", weight_profile="equal")
        idx = _build_index(
            _fp("ep1", ["A"], freq=10),
            _fp("ep2", ["A"], freq=2),
            cfg=cfg,
        )
        searcher = PatternSearcher(idx, cfg)
        query = _fp("q", ["A"], freq=5)
        with pytest.raises(RuntimeError, match="requires.*compute_emd_normalization_factor"):
            searcher.search(query)

    def test_empirical_max_mode_with_factor_set(self):
        cfg = SearchConfig(emd_normalization_mode="empirical_max", weight_profile="equal")
        idx = _build_index(
            _fp("ep1", ["A"], freq=10),
            _fp("ep2", ["A"], freq=2),
            cfg=cfg,
        )
        idx.compute_emd_normalization_factor()
        searcher = PatternSearcher(idx, cfg)
        query = _fp("q", ["A"], freq=5)
        results = searcher.search(query)
        assert len(results) == 2
        assert results[0].combined_score > 0.0

    def test_empirical_max_scores_differ_from_tv(self):
        cfg_tv = SearchConfig(emd_normalization_mode="tv", weight_profile="equal")
        cfg_em = SearchConfig(emd_normalization_mode="empirical_max", weight_profile="equal")
        fp1 = _fp("ep1", ["A"], freq=10)
        fp2 = _fp("ep2", ["A"], freq=2)
        query = _fp("q", ["A"], freq=5)

        idx_tv = _build_index(fp1, fp2, cfg=cfg_tv)
        results_tv = PatternSearcher(idx_tv, cfg_tv).search(query)

        idx_em = _build_index(fp1, fp2, cfg=cfg_em)
        idx_em.compute_emd_normalization_factor()
        results_em = PatternSearcher(idx_em, cfg_em).search(query)

        assert len(results_tv) == len(results_em) == 2
        assert results_tv[0].emd_score != results_em[0].emd_score
