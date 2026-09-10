"""Unit tests for rca_pattern_search.extractor"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from ..config import SearchConfig
from ..extractor import (
    IncidentExtractor,
    _expand_window,
    _compute_density,
    _derive_soe_end_timestamps,
    _parse_ts,
)
from ..models import UnifiedEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T0 = datetime(2024, 1, 1, 12, 0, 0)
T1 = datetime(2024, 1, 1, 12, 10, 0)   # 10 minutes later

CFG = SearchConfig(beta=0.2, freq_threshold=3)


def _alarm_log(alarms: list[dict]) -> dict:
    return {"alarms": alarms}


def _soe_log(records: list[dict]) -> dict:
    return {"records": records}


def _telemetry(asset_id: str, anomalies: list[dict]) -> dict:
    return {"asset_id": asset_id, "anomalies": anomalies}


def _alarm(alarm_id, asset_id, ts, state="active", ack=None):
    d = {"alarm_id": alarm_id, "asset_id": asset_id, "timestamp": ts, "state": state}
    if ack is not None:
        d["acknowledged_at"] = ack
    return d


def _soe_rec(record_id, asset_id, signal_id, transition, ts):
    return {
        "record_id": record_id,
        "asset_id": asset_id,
        "signal_id": signal_id,
        "transition": transition,
        "timestamp": ts,
    }


def _anomaly(anomaly_id, sensor_id, pattern, ts_start, ts_end=None, promoted=True, score=None):
    d = {
        "anomaly_id": anomaly_id,
        "sensor_id": sensor_id,
        "pattern": pattern,
        "timestamp_start": ts_start,
        "timestamp_end": ts_end,
        "promoted_to_kg_event": promoted,
    }
    if score is not None:
        d["severity_score"] = score
    return d


# ---------------------------------------------------------------------------
# _parse_ts
# ---------------------------------------------------------------------------

class TestParseTs:
    def test_datetime_passthrough(self):
        assert _parse_ts(T0) == T0

    def test_iso_string(self):
        assert _parse_ts("2024-01-01T12:00:00") == T0

    def test_none_returns_none(self):
        assert _parse_ts(None) is None

    def test_bad_string_returns_none(self):
        assert _parse_ts("not-a-date") is None

    def test_nat_returns_none(self):
        try:
            import pandas as pd
            assert _parse_ts(pd.NaT) is None
        except ImportError:
            pytest.skip("pandas not installed")


# ---------------------------------------------------------------------------
# _expand_window
# ---------------------------------------------------------------------------

class TestExpandWindow:
    def test_symmetric_expansion(self):
        duration = (T1 - T0).total_seconds()   # 600 s
        exp_start, exp_end = _expand_window(T0, T1, beta=0.2)
        assert exp_start == T0 - timedelta(seconds=0.2 * duration)
        assert exp_end   == T1 + timedelta(seconds=0.2 * duration)

    def test_zero_beta_no_change(self):
        assert _expand_window(T0, T1, beta=0.0) == (T0, T1)


# ---------------------------------------------------------------------------
# _compute_density
# ---------------------------------------------------------------------------

class TestComputeDensity:
    def _make_events(self, timestamps):
        return [
            UnifiedEvent("r", "A", "alarm", "X", ts, None)
            for ts in timestamps
        ]

    def test_all_in_window(self):
        # 6 events in 600 s → 0.01 /s
        timestamps = [T0 + timedelta(seconds=i * 60) for i in range(6)]
        evs = self._make_events(timestamps)
        assert _compute_density(evs, T0, T1) == pytest.approx(6 / 600)

    def test_some_outside_window(self):
        # Only 3 of 5 are in [T0, T1]
        timestamps = [
            T0 - timedelta(seconds=1),
            T0,
            T0 + timedelta(seconds=300),
            T1,
            T1 + timedelta(seconds=1),
        ]
        evs = self._make_events(timestamps)
        assert _compute_density(evs, T0, T1) == pytest.approx(3 / 600)

    def test_zero_duration_returns_zero(self):
        evs = self._make_events([T0])
        assert _compute_density(evs, T0, T0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _derive_soe_end_timestamps
# ---------------------------------------------------------------------------

class TestDeriveSoeEndTimestamps:
    def test_opposing_transition_found(self):
        t_trip = T0
        t_reset = T0 + timedelta(minutes=5)
        records = [
            _soe_rec("r1", "A1", "SIG_001", "trip",  t_trip),
            _soe_rec("r2", "A1", "SIG_001", "reset", t_reset),
        ]
        result = _derive_soe_end_timestamps(records)
        assert result["r1"] == t_reset   # trip ends at reset
        assert result["r2"] is None      # reset has no following opposing

    def test_same_transition_skipped(self):
        t1 = T0
        t2 = T0 + timedelta(minutes=2)
        t3 = T0 + timedelta(minutes=5)
        records = [
            _soe_rec("r1", "A1", "SIG_001", "trip", t1),
            _soe_rec("r2", "A1", "SIG_001", "trip", t2),   # same → not opposing
            _soe_rec("r3", "A1", "SIG_001", "reset", t3),
        ]
        result = _derive_soe_end_timestamps(records)
        assert result["r1"] == t3   # first opposing after r1 is r3

    def test_different_signals_independent(self):
        records = [
            _soe_rec("r1", "A1", "SIG_A", "trip",  T0),
            _soe_rec("r2", "A1", "SIG_B", "reset", T0 + timedelta(minutes=1)),
            _soe_rec("r3", "A1", "SIG_A", "reset", T0 + timedelta(minutes=3)),
        ]
        result = _derive_soe_end_timestamps(records)
        assert result["r1"] == T0 + timedelta(minutes=3)   # SIG_A trip→reset
        assert result["r2"] is None                         # SIG_B no opposing
        assert result["r3"] is None


# ---------------------------------------------------------------------------
# to_unified_events filtering
# ---------------------------------------------------------------------------

class TestToUnifiedEvents:
    def setup_method(self):
        self.ex = IncidentExtractor(CFG)

    def test_suppressed_alarm_excluded(self):
        log = _alarm_log([
            _alarm("ALM_OK",  "A1", T0, state="active"),
            _alarm("ALM_SUP", "A1", T0, state="suppressed"),
        ])
        evs = self.ex.to_unified_events(log, _soe_log([]), [])
        ids = {e.raw_id for e in evs}
        assert "ALM_OK" in ids
        assert "ALM_SUP" not in ids

    def test_alarm_missing_timestamp_skipped(self):
        log = _alarm_log([{"alarm_id": "BAD", "asset_id": "A1", "state": "active"}])
        evs = self.ex.to_unified_events(log, _soe_log([]), [])
        assert evs == []

    def test_soe_all_included(self):
        soe = _soe_log([
            _soe_rec("r1", "A1", "SIG_001", "trip", T0),
            _soe_rec("r2", "A1", "SIG_001", "reset", T1),
        ])
        evs = self.ex.to_unified_events(_alarm_log([]), soe, [])
        assert len(evs) == 2
        assert all(e.source == "soe" for e in evs)

    def test_soe_event_type_format(self):
        soe = _soe_log([_soe_rec("r1", "A1", "SIG_001", "trip", T0)])
        evs = self.ex.to_unified_events(_alarm_log([]), soe, [])
        assert evs[0].event_type == "SIG_001::trip"

    def test_anomaly_promoted_included(self):
        tel = _telemetry("A1", [_anomaly("AN1", "TEMP_01", "spike", T0, promoted=True)])
        evs = self.ex.to_unified_events(_alarm_log([]), _soe_log([]), [tel])
        assert len(evs) == 1
        assert evs[0].event_type == "TEMP_01::spike"

    def test_anomaly_not_promoted_excluded(self):
        tel = _telemetry("A1", [_anomaly("AN1", "TEMP_01", "spike", T0, promoted=False)])
        evs = self.ex.to_unified_events(_alarm_log([]), _soe_log([]), [tel])
        assert evs == []

    def test_anomaly_severity_fallback_passes(self):
        rec = _anomaly("AN1", "TEMP_01", "spike", T0, score=0.8)
        del rec["promoted_to_kg_event"]
        tel = _telemetry("A1", [rec])
        evs = self.ex.to_unified_events(_alarm_log([]), _soe_log([]), [tel])
        assert len(evs) == 1

    def test_anomaly_severity_fallback_blocks(self):
        rec = _anomaly("AN1", "TEMP_01", "spike", T0, score=0.2)
        del rec["promoted_to_kg_event"]
        tel = _telemetry("A1", [rec])
        evs = self.ex.to_unified_events(_alarm_log([]), _soe_log([]), [tel])
        assert evs == []

    def test_anomaly_asset_id_fallback_to_top_level(self):
        rec = {
            "anomaly_id": "AN1", "sensor_id": "S1", "pattern": "drift",
            "timestamp_start": T0, "promoted_to_kg_event": True,
        }
        tel = _telemetry("ASSET_TOP", [rec])
        evs = self.ex.to_unified_events(_alarm_log([]), _soe_log([]), [tel])
        assert evs[0].asset_id == "ASSET_TOP"


# ---------------------------------------------------------------------------
# _derive_fingerprint
# ---------------------------------------------------------------------------

class TestDeriveFingerprint:
    def _events(self, types_and_times):
        return [
            UnifiedEvent("id", "A1", "alarm", et, ts, None)
            for et, ts in types_and_times
        ]

    def test_empty_input(self):
        s, seq, fv = IncidentExtractor._derive_fingerprint([], freq_threshold=3)
        assert s == frozenset()
        assert seq == []
        assert fv == {}

    def test_basic_fingerprint(self):
        evs = self._events([
            ("A", T0),
            ("B", T0 + timedelta(minutes=1)),
            ("A", T0 + timedelta(minutes=2)),
        ])
        s, seq, fv = IncidentExtractor._derive_fingerprint(evs, freq_threshold=5)
        assert s == frozenset({"A", "B"})
        assert seq == ["A", "B"]  # A appears first
        assert fv == {"A": 2, "B": 1}

    def test_high_freq_excluded_from_set_and_seq(self):
        # A appears 4 times (> threshold=3), B once
        evs = self._events(
            [("A", T0 + timedelta(seconds=i)) for i in range(4)]
            + [("B", T0 + timedelta(minutes=5))]
        )
        s, seq, fv = IncidentExtractor._derive_fingerprint(evs, freq_threshold=3)
        assert "A" not in s
        assert "A" not in seq
        assert "B" in s
        assert fv["A"] == 4   # still in freq_vec

    def test_event_seq_ordered_by_first_occurrence(self):
        evs = self._events([
            ("C", T0 + timedelta(minutes=2)),
            ("A", T0),
            ("B", T0 + timedelta(minutes=1)),
            ("A", T0 + timedelta(minutes=3)),  # second occurrence of A
        ])
        _, seq, _ = IncidentExtractor._derive_fingerprint(evs, freq_threshold=5)
        assert seq == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# extract() integration
# ---------------------------------------------------------------------------

class TestExtract:
    def setup_method(self):
        self.ex = IncidentExtractor(CFG)

    def test_returns_fingerprint(self):
        log = _alarm_log([_alarm("ALM_1", "A1", T0 + timedelta(seconds=30))])
        fp = self.ex.extract(log, _soe_log([]), [], "INC_001", T0, T1)
        assert fp.episode_id == "INC_001"
        assert "ALM_1" in fp.event_set

    def test_expanded_window_stored(self):
        duration = (T1 - T0).total_seconds()
        exp_start = T0 - timedelta(seconds=0.2 * duration)
        exp_end   = T1 + timedelta(seconds=0.2 * duration)
        fp = self.ex.extract(_alarm_log([]), _soe_log([]), [], "INC_001", T0, T1)
        assert fp.window_start == exp_start
        assert fp.window_end   == exp_end

    def test_event_outside_expanded_window_excluded(self):
        duration = (T1 - T0).total_seconds()
        far_future = T1 + timedelta(seconds=duration)  # well beyond expanded end
        log = _alarm_log([
            _alarm("IN",  "A1", T0 + timedelta(seconds=30)),
            _alarm("OUT", "A1", far_future),
        ])
        fp = self.ex.extract(log, _soe_log([]), [], "INC_001", T0, T1)
        assert "IN"  in fp.event_set
        assert "OUT" not in fp.event_set

    def test_density_over_expanded_window(self):
        # CFG beta=0.2, window 600 s → expanded duration = 600 * 1.4 = 840 s
        # 6 events inside original window, all within expanded window → density = 6/840
        alarms = [_alarm(f"A{i}", "A1", T0 + timedelta(seconds=i * 60)) for i in range(6)]
        fp = self.ex.extract(_alarm_log(alarms), _soe_log([]), [], "INC_001", T0, T1)
        assert fp.density == pytest.approx(6 / 840)

    def test_known_rca_from_metadata(self):
        fp = self.ex.extract(
            _alarm_log([]), _soe_log([]), [], "INC_001", T0, T1,
            metadata={"known_rca": "pump_cavitation"}
        )
        assert fp.known_rca == "pump_cavitation"
