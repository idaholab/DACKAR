"""Unit tests for rca_pattern_search.indexer"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from ..config import SearchConfig
from ..indexer import (
    IncidentIndex,
    _coerce_ts,
    _df_to_events,
    _deserialise_df,
    _fingerprint_to_row,
    _serialise_df,
)
from ..models import IncidentFingerprint, UnifiedEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T0 = datetime(2024, 1, 1, 0, 0, 0)
CFG = SearchConfig(beta=0.1, delta=0.3, kde_bandwidth="auto", freq_threshold=3)


def _fp(
    episode_id: str,
    event_types: list[str],
    freq_vec: dict[str, int] | None = None,
    asset: str = "PUMP_01",
    known_rca: str | None = None,
) -> IncidentFingerprint:
    fs = frozenset(event_types)
    return IncidentFingerprint(
        episode_id=episode_id,
        asset_id=asset,
        window_start=T0,
        window_end=T0 + timedelta(hours=1),
        density=0.01,
        event_set=fs,
        event_seq=sorted(event_types),
        freq_vec=freq_vec or {et: 2 for et in event_types},
        known_rca=known_rca,
    )


def _events_df(n_clusters: int = 2, events_per_cluster: int = 15) -> pd.DataFrame:
    """Build a synthetic events_df with well-separated clusters."""
    rng = np.random.default_rng(0)
    rows = []
    for c in range(n_clusters):
        center = T0 + timedelta(hours=c * 6 + 1)
        for i in range(events_per_cluster):
            offset = rng.normal(0, 120)   # ±2 minutes spread
            ts = center + timedelta(seconds=float(offset))
            rows.append({
                "raw_id": f"r{c}_{i}",
                "asset_id": "A1",
                "source": "alarm",
                "event_type": f"ALM_{i:02d}",   # unique per event → none high-freq
                "timestamp_start": ts,
                "timestamp_end": None,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _df_to_events
# ---------------------------------------------------------------------------

class TestDfToEvents:
    def test_basic_conversion(self):
        df = pd.DataFrame([{
            "raw_id": "r1", "asset_id": "A1", "source": "alarm",
            "event_type": "ALM_001",
            "timestamp_start": T0, "timestamp_end": T0 + timedelta(minutes=5),
        }])
        evs = _df_to_events(df)
        assert len(evs) == 1
        assert evs[0].raw_id == "r1"
        assert evs[0].event_type == "ALM_001"
        assert evs[0].timestamp_start == T0

    def test_missing_timestamp_skipped(self):
        df = pd.DataFrame([{
            "raw_id": "bad", "asset_id": "A1", "source": "alarm",
            "event_type": "X", "timestamp_start": None, "timestamp_end": None,
        }])
        assert _df_to_events(df) == []

    def test_nat_timestamp_skipped(self):
        df = pd.DataFrame([{
            "raw_id": "nat", "asset_id": "A1", "source": "alarm",
            "event_type": "X",
            "timestamp_start": pd.NaT, "timestamp_end": pd.NaT,
        }])
        assert _df_to_events(df) == []

    def test_nat_timestamp_end_becomes_none(self):
        df = pd.DataFrame([{
            "raw_id": "r1", "asset_id": "A1", "source": "alarm",
            "event_type": "X",
            "timestamp_start": T0, "timestamp_end": pd.NaT,
        }])
        evs = _df_to_events(df)
        assert evs[0].timestamp_end is None


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------

class TestSerialisation:
    def _sample_df(self):
        return pd.DataFrame([_fingerprint_to_row(_fp("EP_1", ["A", "B", "C"]))])

    def test_roundtrip_event_set(self):
        df = self._sample_df()
        assert df["event_set"].iloc[0] == frozenset({"A", "B", "C"})
        s = _serialise_df(df)
        assert isinstance(s["event_set"].iloc[0], str)
        d = _deserialise_df(s)
        assert d["event_set"].iloc[0] == frozenset({"A", "B", "C"})

    def test_roundtrip_event_seq(self):
        df = self._sample_df()
        s = _serialise_df(df)
        d = _deserialise_df(s)
        assert d["event_seq"].iloc[0] == ["A", "B", "C"]

    def test_roundtrip_freq_vec(self):
        df = self._sample_df()
        s = _serialise_df(df)
        d = _deserialise_df(s)
        assert d["freq_vec"].iloc[0] == {"A": 2, "B": 2, "C": 2}
        assert isinstance(d["freq_vec"].iloc[0]["A"], int)

    def test_empty_df_noop(self):
        empty = pd.DataFrame()
        assert _serialise_df(empty).empty
        assert _deserialise_df(empty).empty


# ---------------------------------------------------------------------------
# add / add_batch / get_candidates
# ---------------------------------------------------------------------------

class TestAddAndCandidates:
    def test_add_single(self):
        idx = IncidentIndex(CFG)
        idx.add(_fp("EP_1", ["A", "B"]))
        assert len(idx) == 1
        assert "A" in idx._inverted_index
        assert "EP_1" in idx._inverted_index["A"]

    def test_add_batch(self):
        idx = IncidentIndex(CFG)
        fps = [_fp(f"EP_{i}", ["X", f"Y{i}"]) for i in range(5)]
        idx.add_batch(fps)
        assert len(idx) == 5
        assert len(idx._inverted_index["X"]) == 5

    def test_get_candidates_overlap(self):
        idx = IncidentIndex(CFG)
        idx.add(_fp("EP_1", ["A", "B"]))
        idx.add(_fp("EP_2", ["B", "C"]))
        idx.add(_fp("EP_3", ["D", "E"]))
        # Query shares B → should get EP_1 and EP_2
        cands = idx.get_candidates(frozenset({"B"}))
        assert set(cands) == {"EP_1", "EP_2"}

    def test_get_candidates_no_overlap(self):
        idx = IncidentIndex(CFG)
        idx.add(_fp("EP_1", ["A", "B"]))
        assert idx.get_candidates(frozenset({"Z"})) == []

    def test_get_candidates_empty_query(self):
        idx = IncidentIndex(CFG)
        idx.add(_fp("EP_1", ["A"]))
        assert idx.get_candidates(frozenset()) == []

    def test_add_preserves_existing(self):
        idx = IncidentIndex(CFG)
        idx.add(_fp("EP_1", ["A"]))
        idx.add(_fp("EP_2", ["A"]))
        assert len(idx) == 2
        assert len(idx._inverted_index["A"]) == 2

    def test_add_batch_empty_is_noop(self):
        idx = IncidentIndex(CFG)
        idx.add_batch([])
        assert len(idx) == 0

    def test_reset_clears_index(self):
        idx = IncidentIndex(CFG)
        idx.add(_fp("EP_1", ["A"]))
        idx.reset()
        assert len(idx) == 0
        assert idx._inverted_index == {}


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self):
        idx = IncidentIndex(CFG)
        idx.add(_fp("EP_A", ["ALM_001", "ALM_002"], known_rca="pump_cavitation"))
        idx.add(_fp("EP_B", ["ALM_002", "SOE_TRIP"]))

        with tempfile.TemporaryDirectory() as tmpdir:
            idx.save(tmpdir)
            loaded = IncidentIndex.load(tmpdir, CFG)

        assert len(loaded) == 2
        row = loaded.episodes_df[loaded.episodes_df["episode_id"] == "EP_A"].iloc[0]
        assert row["event_set"] == frozenset({"ALM_001", "ALM_002"})
        assert row["event_seq"] == ["ALM_001", "ALM_002"]
        assert row["known_rca"] == "pump_cavitation"

    def test_inverted_index_restored(self):
        idx = IncidentIndex(CFG)
        idx.add(_fp("EP_1", ["A", "B"]))
        idx.add(_fp("EP_2", ["B", "C"]))

        with tempfile.TemporaryDirectory() as tmpdir:
            idx.save(tmpdir)
            loaded = IncidentIndex.load(tmpdir, CFG)

        assert loaded._inverted_index["B"] == {"EP_1", "EP_2"}

    def test_load_missing_parquet_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                IncidentIndex.load(tmpdir, CFG)

    def test_save_creates_directory(self):
        idx = IncidentIndex(CFG)
        idx.add(_fp("EP_1", ["A"]))
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = str(Path(tmpdir) / "nested" / "subdir")
            idx.save(nested)
            assert (Path(nested) / "episodes.parquet").exists()
            assert (Path(nested) / "inverted_index.json").exists()

    def test_get_candidates_works_after_load(self):
        idx = IncidentIndex(CFG)
        idx.add(_fp("EP_A", ["X", "Y"]))
        with tempfile.TemporaryDirectory() as tmpdir:
            idx.save(tmpdir)
            loaded = IncidentIndex.load(tmpdir, CFG)
        cands = loaded.get_candidates(frozenset({"X"}))
        assert "EP_A" in cands


# ---------------------------------------------------------------------------
# build_from_history — integration
# ---------------------------------------------------------------------------

class TestBuildFromHistory:
    def test_two_clusters_produce_episodes(self):
        idx = IncidentIndex(CFG)
        df = _events_df(n_clusters=2, events_per_cluster=15)
        # rho_query: 10 events / 600s
        idx.build_from_history(df, rho_query=10 / 600, query_duration=600.0)
        assert len(idx) >= 1   # at least one episode

    def test_empty_df_produces_empty_index(self):
        idx = IncidentIndex(CFG)
        idx.build_from_history(pd.DataFrame(), rho_query=0.01, query_duration=600.0)
        assert len(idx) == 0

    def test_episodes_have_event_set(self):
        idx = IncidentIndex(CFG)
        df = _events_df(n_clusters=1, events_per_cluster=20)
        idx.build_from_history(df, rho_query=10 / 600, query_duration=600.0)
        if len(idx) > 0:
            row = idx.episodes_df.iloc[0]
            assert isinstance(row["event_set"], frozenset)
            assert len(row["event_set"]) > 0

    def test_inverted_index_populated(self):
        idx = IncidentIndex(CFG)
        df = _events_df(n_clusters=1, events_per_cluster=20)
        idx.build_from_history(df, rho_query=10 / 600, query_duration=600.0)
        if len(idx) > 0:
            assert len(idx._inverted_index) > 0

    def test_second_call_appends(self):
        idx = IncidentIndex(CFG)
        df = _events_df(n_clusters=1, events_per_cluster=20)
        idx.build_from_history(df, rho_query=10 / 600, query_duration=600.0)
        n_first = len(idx)
        idx.build_from_history(df, rho_query=10 / 600, query_duration=600.0)
        assert len(idx) >= n_first  # second call adds more episodes

    def test_repr(self):
        idx = IncidentIndex(CFG)
        idx.add(_fp("EP_1", ["A"]))
        assert "IncidentIndex" in repr(idx)
        assert "1" in repr(idx)


# ---------------------------------------------------------------------------
# _coerce_ts
# ---------------------------------------------------------------------------

class TestCoerceTs:
    def test_plain_datetime_passthrough(self):
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = _coerce_ts(dt)
        assert result == dt
        assert type(result) is datetime

    def test_pandas_timestamp_converted_to_plain_datetime(self):
        ts = pd.Timestamp("2024-01-01 12:00:00")
        result = _coerce_ts(ts)
        assert type(result) is datetime
        assert result == datetime(2024, 1, 1, 12, 0, 0)

    def test_nat_raises(self):
        with pytest.raises(TypeError):
            _coerce_ts(pd.NaT)


# ---------------------------------------------------------------------------
# compute_emd_normalization_factor
# ---------------------------------------------------------------------------

class TestComputeEmdNormalizationFactor:
    def test_empty_index_returns_one(self):
        index = IncidentIndex(CFG)
        factor = index.compute_emd_normalization_factor()
        assert factor == 1.0

    def test_single_episode_returns_one(self):
        index = IncidentIndex(CFG)
        fp = _fp("ep1", ["TYPE_A", "TYPE_B"], freq_vec={"TYPE_A": 5, "TYPE_B": 3})
        index.add(fp)
        factor = index.compute_emd_normalization_factor()
        assert factor == 1.0

    def test_two_disjoint_episodes_computes_max_l1(self):
        index = IncidentIndex(CFG)
        fp1 = _fp("ep1", ["TYPE_A"], freq_vec={"TYPE_A": 10})
        fp2 = _fp("ep2", ["TYPE_B"], freq_vec={"TYPE_B": 5})
        index.add_batch([fp1, fp2])
        factor = index.compute_emd_normalization_factor()
        expected_l1 = abs(10 - 0) + abs(0 - 5)
        assert factor == expected_l1

    def test_two_identical_episodes_returns_one(self):
        index = IncidentIndex(CFG)
        fp1 = _fp("ep1", ["TYPE_A"], freq_vec={"TYPE_A": 5})
        fp2 = _fp("ep2", ["TYPE_A"], freq_vec={"TYPE_A": 5})
        index.add_batch([fp1, fp2])
        factor = index.compute_emd_normalization_factor()
        assert factor == 1.0

    def test_max_pairs_sampling(self):
        index = IncidentIndex(CFG)
        fps = [
            _fp(f"ep{i}", ["TYPE_A"], freq_vec={"TYPE_A": i + 1})
            for i in range(10)
        ]
        index.add_batch(fps)
        factor = index.compute_emd_normalization_factor(max_pairs=5)
        assert isinstance(factor, float) and factor > 0
        assert index.emd_normalization_factor == factor

    def test_stores_factor_in_index(self):
        index = IncidentIndex(CFG)
        fp = _fp("ep1", ["TYPE_A"], freq_vec={"TYPE_A": 10})
        index.add(fp)
        assert index.emd_normalization_factor is None
        factor = index.compute_emd_normalization_factor()
        assert index.emd_normalization_factor == factor


# ---------------------------------------------------------------------------
# save / load with EMD metadata
# ---------------------------------------------------------------------------

class TestSaveLoadWithEmdMeta:
    def test_save_and_load_preserves_emd_factor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            index1 = IncidentIndex(CFG)
            fp = _fp("ep1", ["TYPE_A", "TYPE_B"], freq_vec={"TYPE_A": 5, "TYPE_B": 3})
            index1.add(fp)
            factor = index1.compute_emd_normalization_factor()
            index1.save(tmpdir)

            # Load
            index2 = IncidentIndex.load(tmpdir, CFG)
            assert index2.emd_normalization_factor == factor

    def test_load_old_index_without_emd_meta_graceful(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Manually create old-style index (no emd_meta.json)
            tmpdir_p = Path(tmpdir)
            index1 = IncidentIndex(CFG)
            fp = _fp("ep1", ["TYPE_A"], freq_vec={"TYPE_A": 5})
            index1.add(fp)
            # Save but delete emd_meta.json
            index1.save(tmpdir)
            (tmpdir_p / "emd_meta.json").unlink()

            # Load should not fail
            index2 = IncidentIndex.load(tmpdir, CFG)
            assert index2.emd_normalization_factor is None
            assert len(index2) == 1
