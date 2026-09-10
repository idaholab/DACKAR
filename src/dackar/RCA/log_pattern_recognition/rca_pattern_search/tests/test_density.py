"""Unit tests for rca_pattern_search.density"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta

import numpy as np

from ..config import SearchConfig
from ..density import (
    EpisodeDetector,
    _extract_contiguous_regions,
    _kde_evaluate,
    _merge_overlapping,
)
from ..models import UnifiedEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T0 = datetime(2024, 1, 1, 0, 0, 0)
CFG = SearchConfig(beta=0.2, delta=0.5, kde_bandwidth="auto", freq_threshold=3)


def _ev(ts: datetime, asset: str = "A1") -> UnifiedEvent:
    return UnifiedEvent("id", asset, "alarm", "TYPE_A", ts, None)


def _cluster(center: datetime, n: int, spread_s: float = 30.0) -> list[UnifiedEvent]:
    """n events clustered around center with Gaussian spread."""
    rng = np.random.default_rng(42)
    offsets = rng.normal(0, spread_s, n)
    return [_ev(center + timedelta(seconds=float(o))) for o in offsets]


# ---------------------------------------------------------------------------
# _kde_evaluate
# ---------------------------------------------------------------------------

class TestKdeEvaluate:
    def test_single_event_peak_at_location(self):
        t_seconds = np.array([0.0])
        bw = 10.0
        grid = np.arange(-50.0, 51.0, 1.0)
        kde = _kde_evaluate(t_seconds, grid, bw, grid_res=1.0)
        # Peak should be at t=0
        peak_idx = np.argmax(kde)
        assert grid[peak_idx] == pytest.approx(0.0, abs=1.0)

    def test_symmetric_around_event(self):
        t_seconds = np.array([100.0])
        bw = 20.0
        grid = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        kde = _kde_evaluate(t_seconds, grid, bw, grid_res=10.0)
        # Symmetric: values at ±10 and ±20 should match
        assert kde[1] == pytest.approx(kde[3], rel=1e-6)
        assert kde[0] == pytest.approx(kde[4], rel=1e-6)

    def test_two_clusters_produce_two_peaks(self):
        # Events at t=0 and t=1000
        t_seconds = np.concatenate([
            np.linspace(-20, 20, 10),
            np.linspace(980, 1020, 10),
        ])
        bw = 30.0
        grid = np.arange(0.0, 1001.0, 5.0)
        kde = _kde_evaluate(t_seconds, grid, bw, grid_res=5.0)
        # Minimum should be somewhere in the middle
        mid = len(grid) // 2
        assert kde[0] > kde[mid]
        assert kde[-1] > kde[mid]

    def test_no_events_returns_zeros(self):
        kde = _kde_evaluate(np.array([]), np.arange(0.0, 100.0, 1.0), bw=10.0, grid_res=1.0)
        assert np.all(kde == 0.0)


# ---------------------------------------------------------------------------
# _extract_contiguous_regions
# ---------------------------------------------------------------------------

class TestExtractContiguousRegions:
    def _grid_from(self, n, res=1.0):
        return np.arange(0.0, float(n), res)

    def test_single_run(self):
        mask = np.array([False, False, True, True, True, False, False])
        grid = self._grid_from(7)
        regions = _extract_contiguous_regions(mask, grid, T0, grid_res=1.0)
        assert len(regions) == 1
        t_s, t_e = regions[0]
        assert t_s == T0 + timedelta(seconds=2)
        assert t_e == T0 + timedelta(seconds=4)

    def test_two_runs(self):
        mask = np.array([True, True, False, False, True, True, True])
        grid = self._grid_from(7)
        regions = _extract_contiguous_regions(mask, grid, T0, grid_res=1.0)
        assert len(regions) == 2

    def test_all_false(self):
        mask = np.zeros(10, dtype=bool)
        regions = _extract_contiguous_regions(mask, self._grid_from(10), T0, 1.0)
        assert regions == []

    def test_all_true(self):
        mask = np.ones(5, dtype=bool)
        regions = _extract_contiguous_regions(mask, self._grid_from(5), T0, 1.0)
        assert len(regions) == 1

    def test_single_point_run_gets_nonzero_duration(self):
        mask = np.array([False, True, False])
        grid = self._grid_from(3)
        regions = _extract_contiguous_regions(mask, grid, T0, grid_res=1.0)
        t_s, t_e = regions[0]
        assert t_e > t_s


# ---------------------------------------------------------------------------
# _merge_overlapping
# ---------------------------------------------------------------------------

class TestMergeOverlapping:
    def _dt(self, h):
        return T0 + timedelta(hours=h)

    def test_non_overlapping(self):
        b = [(self._dt(0), self._dt(1)), (self._dt(2), self._dt(3))]
        assert _merge_overlapping(b) == b

    def test_overlapping_merged(self):
        b = [(self._dt(0), self._dt(2)), (self._dt(1), self._dt(3))]
        result = _merge_overlapping(b)
        assert len(result) == 1
        assert result[0] == (self._dt(0), self._dt(3))

    def test_touching_merged(self):
        b = [(self._dt(0), self._dt(1)), (self._dt(1), self._dt(2))]
        result = _merge_overlapping(b)
        assert len(result) == 1

    def test_empty(self):
        assert _merge_overlapping([]) == []

    def test_unsorted_input(self):
        b = [(self._dt(2), self._dt(3)), (self._dt(0), self._dt(1))]
        result = _merge_overlapping(b)
        assert result[0][0] == self._dt(0)

    def test_contained_interval(self):
        # Small interval fully inside large one
        b = [(self._dt(0), self._dt(5)), (self._dt(1), self._dt(3))]
        result = _merge_overlapping(b)
        assert len(result) == 1
        assert result[0] == (self._dt(0), self._dt(5))


# ---------------------------------------------------------------------------
# EpisodeDetector.compute_reference_density
# ---------------------------------------------------------------------------

class TestComputeReferenceDensity:
    def test_basic(self):
        det = EpisodeDetector(CFG)
        events = [_ev(T0 + timedelta(seconds=i * 60)) for i in range(6)]
        rho = det.compute_reference_density(events, T0, T0 + timedelta(minutes=10))
        assert rho == pytest.approx(6 / 600)

    def test_zero_duration_returns_zero(self):
        det = EpisodeDetector(CFG)
        rho = det.compute_reference_density([_ev(T0)], T0, T0)
        assert rho == pytest.approx(0.0)

    def test_events_outside_window_not_counted(self):
        det = EpisodeDetector(CFG)
        events = [
            _ev(T0 - timedelta(seconds=1)),
            _ev(T0 + timedelta(seconds=300)),
            _ev(T0 + timedelta(seconds=601)),
        ]
        rho = det.compute_reference_density(events, T0, T0 + timedelta(seconds=600))
        assert rho == pytest.approx(1 / 600)


# ---------------------------------------------------------------------------
# EpisodeDetector.detect — integration
# ---------------------------------------------------------------------------

class TestDetect:
    def _make_detector(self, **kwargs):
        return EpisodeDetector(SearchConfig(**kwargs))

    def test_empty_events_returns_empty(self):
        det = EpisodeDetector(CFG)
        assert det.detect([], rho_query=0.01, query_duration=600) == []

    def test_zero_rho_query_returns_empty(self):
        det = EpisodeDetector(CFG)
        assert det.detect([_ev(T0)], rho_query=0.0, query_duration=600) == []

    def test_single_dense_cluster_detected(self):
        """A tight cluster of events should produce exactly one episode."""
        det = self._make_detector(beta=0.1, delta=0.3, kde_bandwidth="auto")
        # 20 events in 2 minutes — very dense
        cluster = _cluster(T0 + timedelta(hours=1), n=20, spread_s=30)
        # rho_query from a 10-minute window with 10 events
        rho_query = 10 / 600
        query_duration = 600.0
        boundaries = det.detect(cluster, rho_query, query_duration)
        assert len(boundaries) == 1
        # The cluster center should fall inside the detected boundary
        cluster_center = T0 + timedelta(hours=1)
        assert any(s <= cluster_center <= e for s, e in boundaries)

    def test_two_separated_clusters_detected(self):
        """Two well-separated dense clusters should produce two episodes."""
        det = self._make_detector(beta=0.1, delta=0.3, kde_bandwidth="auto")
        c1 = _cluster(T0 + timedelta(hours=2), n=15, spread_s=20)
        c2 = _cluster(T0 + timedelta(hours=8), n=15, spread_s=20)
        all_events = c1 + c2
        rho_query = 10 / 600
        boundaries = det.detect(all_events, rho_query, query_duration=600.0)
        assert len(boundaries) == 2

    def test_boundaries_sorted_ascending(self):
        det = EpisodeDetector(CFG)
        c1 = _cluster(T0 + timedelta(hours=1), n=15, spread_s=20)
        c2 = _cluster(T0 + timedelta(hours=5), n=15, spread_s=20)
        boundaries = det.detect(c1 + c2, rho_query=10 / 600, query_duration=600.0)
        starts = [s for s, _ in boundaries]
        assert starts == sorted(starts)

    def test_boundaries_are_expanded(self):
        """Returned boundaries should extend beyond the raw event span (beta > 0)."""
        det = self._make_detector(beta=0.3, delta=0.3, kde_bandwidth="auto")
        # spread_s=120 → raw event span ~720s (6σ), well above min_dur=60s
        cluster = _cluster(T0 + timedelta(hours=1), n=20, spread_s=120)
        boundaries = det.detect(cluster, rho_query=5 / 600, query_duration=600.0)
        assert len(boundaries) >= 1
        earliest = min(e.timestamp_start for e in cluster)
        latest   = max(e.timestamp_start for e in cluster)
        s, e = boundaries[0]
        assert s <= earliest
        assert e >= latest

    def test_sparse_noise_does_not_produce_episodes(self):
        """Very sparse events should not exceed the threshold."""
        det = self._make_detector(beta=0.1, delta=0.8, kde_bandwidth="auto")
        # 1 event per day over 30 days — extremely sparse
        events = [_ev(T0 + timedelta(days=i)) for i in range(30)]
        # rho_query = 20 events / 600 s (very dense query)
        rho_query = 20 / 600
        boundaries = det.detect(events, rho_query, query_duration=600.0)
        assert boundaries == []


# ---------------------------------------------------------------------------
# EpisodeDetector.assign_episode_ids
# ---------------------------------------------------------------------------

class TestAssignEpisodeIds:
    def test_no_boundaries_all_none(self):
        det = EpisodeDetector(CFG)
        evs = [_ev(T0 + timedelta(minutes=i)) for i in range(5)]
        result = det.assign_episode_ids(evs, [])
        assert all(e.episode_id is None for e in result)

    def test_events_inside_boundary_get_id(self):
        det = EpisodeDetector(CFG)
        t_s = T0
        t_e = T0 + timedelta(hours=1)
        evs = [_ev(T0 + timedelta(minutes=i)) for i in range(4)]
        result = det.assign_episode_ids(evs, [(t_s, t_e)])
        assert all(e.episode_id is not None for e in result)
        assert all(e.episode_id == result[0].episode_id for e in result)

    def test_events_outside_boundary_get_none(self):
        det = EpisodeDetector(CFG)
        t_s = T0 + timedelta(hours=1)
        t_e = T0 + timedelta(hours=2)
        evs = [
            _ev(T0),                          # before episode
            _ev(T0 + timedelta(hours=1, minutes=30)),  # inside
            _ev(T0 + timedelta(hours=3)),      # after episode
        ]
        result = det.assign_episode_ids(evs, [(t_s, t_e)])
        assert result[0].episode_id is None
        assert result[1].episode_id is not None
        assert result[2].episode_id is None

    def test_episode_id_format(self):
        det = EpisodeDetector(CFG)
        evs = [_ev(T0 + timedelta(minutes=10), asset="PUMP_01")]
        result = det.assign_episode_ids(evs, [(T0, T0 + timedelta(hours=1))])
        assert result[0].episode_id == "EP_PUMP_01_00000"

    def test_two_episodes_different_ids(self):
        det = EpisodeDetector(CFG)
        ep1 = (T0, T0 + timedelta(hours=1))
        ep2 = (T0 + timedelta(hours=2), T0 + timedelta(hours=3))
        evs = [
            _ev(T0 + timedelta(minutes=30)),
            _ev(T0 + timedelta(hours=2, minutes=30)),
        ]
        result = det.assign_episode_ids(evs, [ep1, ep2])
        assert result[0].episode_id != result[1].episode_id

    def test_original_events_not_mutated(self):
        det = EpisodeDetector(CFG)
        evs = [_ev(T0 + timedelta(minutes=10))]
        original_ep_id = evs[0].episode_id
        det.assign_episode_ids(evs, [(T0, T0 + timedelta(hours=1))])
        assert evs[0].episode_id == original_ep_id   # dataclasses.replace, not in-place


# ---------------------------------------------------------------------------
# bandwidth_scan
# ---------------------------------------------------------------------------

class TestBandwidthScan:
    def test_empty_history_returns_empty(self):
        det = EpisodeDetector(CFG)
        result = det.bandwidth_scan([], rho_query=1.0, query_duration=3600.0)
        assert result == {}

    def test_explicit_bandwidth_list(self):
        det = EpisodeDetector(CFG)
        events = _cluster(T0 + timedelta(hours=1), n=50, spread_s=60.0)
        rho_q = 50.0 / 3600.0
        result = det.bandwidth_scan(events, rho_query=rho_q, query_duration=3600.0,
                                   bandwidths=[60.0, 300.0, 1800.0])
        assert isinstance(result, dict)
        assert len(result) == 3
        assert set(result.keys()) == {60.0, 300.0, 1800.0}
        for count in result.values():
            assert isinstance(count, int) and count >= 0

    def test_narrower_bandwidth_detects_more_episodes(self):
        det = EpisodeDetector(CFG)
        c1 = _cluster(T0 + timedelta(hours=1), n=30, spread_s=30.0)
        c2 = _cluster(T0 + timedelta(hours=6), n=30, spread_s=30.0)
        events = c1 + c2
        rho_q = 60.0 / 3600.0
        result = det.bandwidth_scan(events, rho_query=rho_q, query_duration=3600.0,
                                   bandwidths=[300.0, 3600.0])
        narrow = result[300.0]
        wide = result[3600.0]
        assert narrow >= wide, "Narrower bandwidth should detect at least as many episodes"

    def test_default_bandwidths_eight_scales(self):
        det = EpisodeDetector(CFG)
        events = _cluster(T0 + timedelta(hours=1), n=50, spread_s=60.0)
        rho_q = 50.0 / 3600.0
        result = det.bandwidth_scan(events, rho_query=rho_q, query_duration=3600.0,
                                   bandwidths=None)
        expected_scales = [3600/32, 3600/16, 3600/8, 3600/4, 3600/2, 3600, 2*3600, 4*3600]
        assert len(result) == len(expected_scales)
        for scale in expected_scales:
            assert scale in result

    def test_ignores_negative_bandwidths(self):
        det = EpisodeDetector(CFG)
        events = _cluster(T0, n=20, spread_s=30.0)
        result = det.bandwidth_scan(events, rho_query=1.0, query_duration=3600.0,
                                   bandwidths=[60.0, -60.0, 300.0])
        assert len(result) == 2 and -60.0 not in result
        assert 60.0 in result and 300.0 in result
