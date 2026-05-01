"""
test_tskr_temporal_scorer.py — unit tests for temporal_relations and TSKRTemporalScorerV1.

Coverage:
  - temporal_relations:  allen_relation (all 5 relations, epsilon boundary, scores),
                         onset_lag_hours (positive/negative/zero)
  - parse_dt / clamp01   (utility functions)
  - Extraction helpers:  _extract_anomaly_windows, _summarize_anomaly_windows,
                         _extract_signal_ids
  - Telemetry scoring:   _telemetry_support_score, _severity_weight
  - Count / consistency: _effective_anomaly_count, _anomaly_count_score,
                         _lag_consistency_score
  - Operator family:     _infer_operator_family
  - Allen scoring:       _score_against_anomalies (each relation, mixed windows,
                         severity-weighted lag stats)
  - Recurrence:          _recurrence_trend, _build_recurrence_profile,
                         _score_from_recurrence_profile
  - Latency:             _score_expected_latency, _latency_alignment_details
  - Pattern scorer:      _score_failure_mode_pattern (fields, contradiction flag)
  - Integration:         score() entry point (patterns, summary, provenance)

Run directly:   python test_tskr_temporal_scorer.py
Or via pytest:  pytest test_tskr_temporal_scorer.py
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from orchestrators.temporal_relations import (
    Interval,
    allen_relation,
    onset_lag_hours,
    CAUSAL_PRIORITY,
    RELATION_SCORE,
    PRECEDES, OVERLAPS, CONTAINS, DURING, FOLLOWS,
)
from orchestrators.tskr_temporal_scorer import (
    TSKRTemporalScorerV1,
    TSKRTemporalScorerConfig,
    RecurrenceProfile,
    parse_dt,
    clamp01,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

BASE = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def h(n: float) -> datetime:
    """BASE + n hours."""
    return BASE + timedelta(hours=n)


def ivl(start_h: float, end_h: float) -> Interval:
    return Interval(start=h(start_h), end=h(end_h))


def sc(cfg: TSKRTemporalScorerConfig = None) -> TSKRTemporalScorerV1:
    return TSKRTemporalScorerV1(config=cfg)


def win(start_h: float, end_h: float, severity: str = "medium",
        severity_score=None, sensor_id: str = "S1", pattern: str = None) -> dict:
    """Pre-parsed anomaly window (datetime objects)."""
    return {
        "sensor_id": sensor_id,
        "start": h(start_h),
        "end": h(end_h),
        "severity": severity,
        "severity_score": severity_score,
        "pattern": pattern,
    }


def sig_block(sensor_id: str, anomalies: list) -> dict:
    """Build a telemetry_summary signal block with string timestamps."""
    return {
        "sensor_id": sensor_id,
        "anomalies": anomalies,
    }


def anom(start_h: float, end_h: float, severity: str = "medium",
         score: float = None, interval_type: str = None) -> dict:
    """Anomaly block with ISO string timestamps."""
    a = {
        "timestamp_start": h(start_h).isoformat(),
        "timestamp_end": h(end_h).isoformat(),
        "severity": severity,
    }
    if score is not None:
        a["score"] = score
    if interval_type is not None:
        a["interval_type"] = interval_type
    return a


def approx(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol


# ═══════════════════════════════════════════════════════════════════════════════
# temporal_relations — allen_relation
# ═══════════════════════════════════════════════════════════════════════════════

# Event spans hours 0..10; epsilon=0.5h throughout.
EVENT = ivl(0, 10)
EPS   = 0.5


def test_allen_precedes():
    """Anomaly ends well before event starts → PRECEDES."""
    rel, score = allen_relation(ivl(-5, -1), EVENT, EPS)
    assert rel == PRECEDES
    assert approx(score, RELATION_SCORE[PRECEDES])


def test_allen_overlaps():
    """Anomaly starts before event, ends inside event → OVERLAPS."""
    rel, score = allen_relation(ivl(-3, 5), EVENT, EPS)
    assert rel == OVERLAPS
    assert approx(score, RELATION_SCORE[OVERLAPS])


def test_allen_contains():
    """Anomaly encompasses the entire event → CONTAINS."""
    rel, score = allen_relation(ivl(-2, 12), EVENT, EPS)
    assert rel == CONTAINS
    assert approx(score, RELATION_SCORE[CONTAINS])


def test_allen_during():
    """Anomaly starts inside event interval → DURING."""
    rel, score = allen_relation(ivl(3, 7), EVENT, EPS)
    assert rel == DURING
    assert approx(score, RELATION_SCORE[DURING])


def test_allen_follows():
    """Anomaly starts after event ends → FOLLOWS."""
    rel, score = allen_relation(ivl(11, 13), EVENT, EPS)
    assert rel == FOLLOWS
    assert approx(score, RELATION_SCORE[FOLLOWS])


def test_allen_relation_scores_match_table():
    """Returned score must equal RELATION_SCORE[relation] for every relation."""
    cases = [
        (ivl(-5, -1), PRECEDES),
        (ivl(-3,  5), OVERLAPS),
        (ivl(-2, 12), CONTAINS),
        (ivl( 3,  7), DURING),
        (ivl(11, 13), FOLLOWS),
    ]
    for a_ivl, expected_rel in cases:
        rel, score = allen_relation(a_ivl, EVENT, EPS)
        assert rel == expected_rel
        assert approx(score, RELATION_SCORE[expected_rel]), f"score mismatch for {expected_rel}"


def test_allen_epsilon_boundary_precedes_not_triggered():
    """Anomaly end exactly at b_start − epsilon is NOT classified as PRECEDES."""
    # b_s = h(0); eps = 0.5h; boundary = h(-0.5)
    # Condition: a_e < b_s - eps  →  a_e < h(-0.5)  (strict <)
    # So a ending at exactly h(-0.5) should NOT be PRECEDES.
    rel, _ = allen_relation(ivl(-2, -0.5), EVENT, EPS)
    assert rel != PRECEDES, "exact epsilon boundary should not classify as PRECEDES"


def test_allen_epsilon_boundary_follows_not_triggered():
    """Anomaly start exactly at b_end + epsilon is NOT classified as FOLLOWS."""
    # b_e = h(10); eps = 0.5h; boundary = h(10.5)
    # Condition: a_s > b_e + eps  →  a_s > h(10.5)  (strict >)
    rel, _ = allen_relation(ivl(10.5, 12), EVENT, EPS)
    assert rel != FOLLOWS, "exact epsilon boundary should not classify as FOLLOWS"


def test_allen_point_event_during():
    """A point anomaly (start == end) inside the event is DURING."""
    point = Interval(start=h(5), end=h(5))
    rel, _ = allen_relation(point, EVENT, EPS)
    assert rel == DURING


def test_allen_open_interval_touching_start_is_precedes():
    """Open-end anomaly touching event start is treated as PRECEDES."""
    rel, _ = allen_relation(ivl(-2, 0), EVENT, epsilon_hours=0.0, interval_type="open")
    assert rel == PRECEDES


def test_allen_closed_interval_touching_start_not_precedes():
    """Closed anomaly touching event start should not be PRECEDES."""
    rel, _ = allen_relation(ivl(-2, 0), EVENT, epsilon_hours=0.0, interval_type="closed")
    assert rel != PRECEDES


def test_allen_causal_priority_order():
    """CAUSAL_PRIORITY matches expected order: OVERLAPS > CONTAINS > PRECEDES > DURING > FOLLOWS."""
    assert CAUSAL_PRIORITY == (OVERLAPS, CONTAINS, PRECEDES, DURING, FOLLOWS)


# ═══════════════════════════════════════════════════════════════════════════════
# temporal_relations — onset_lag_hours
# ═══════════════════════════════════════════════════════════════════════════════

def test_onset_lag_positive_causal():
    """Anomaly starts before event → positive lag (causal candidate)."""
    a = ivl(-3, 5)     # anomaly onset 3h before event onset
    lag = onset_lag_hours(a, EVENT)
    assert approx(lag, 3.0)


def test_onset_lag_negative_symptom():
    """Anomaly starts after event → negative lag (likely a symptom)."""
    a = ivl(4, 7)      # anomaly onset 4h after event onset
    lag = onset_lag_hours(a, EVENT)
    assert approx(lag, -4.0)


def test_onset_lag_zero():
    """Anomaly and event share the same start → lag = 0."""
    a = ivl(0, 5)
    lag = onset_lag_hours(a, EVENT)
    assert approx(lag, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# parse_dt / clamp01
# ═══════════════════════════════════════════════════════════════════════════════

def test_parse_dt_valid_iso():
    dt = parse_dt("2024-06-01T12:00:00+00:00")
    assert isinstance(dt, datetime)
    assert dt.year == 2024 and dt.month == 6 and dt.hour == 12


def test_parse_dt_z_suffix():
    dt = parse_dt("2024-06-01T12:00:00Z")
    assert isinstance(dt, datetime)


def test_parse_dt_none_returns_none():
    assert parse_dt(None) is None


def test_parse_dt_empty_string_returns_none():
    assert parse_dt("") is None


def test_clamp01_below_zero():
    assert clamp01(-0.5) == 0.0


def test_clamp01_above_one():
    assert clamp01(1.5) == 1.0


def test_clamp01_within_range():
    assert approx(clamp01(0.7), 0.7)


# ═══════════════════════════════════════════════════════════════════════════════
# Extraction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def test_extract_anomaly_windows_normal():
    ts = {"signals": [sig_block("S1", [anom(0, 2, "high"), anom(5, 7, "low")])]}
    windows = sc()._extract_anomaly_windows(ts)
    assert len(windows) == 2
    assert windows[0]["sensor_id"] == "S1"
    assert windows[0]["severity"] == "high"
    assert windows[0]["interval_type"] == "closed"


def test_extract_anomaly_windows_interval_type_propagation():
    ts = {"signals": [sig_block("S1", [anom(0, 2, "high", interval_type="half_open_end")])]}
    windows = sc()._extract_anomaly_windows(ts)
    assert windows[0]["interval_type"] == "half_open_end"


def test_extract_anomaly_windows_instrument_validity_propagation():
    ts = {
        "signals": [
            {
                "sensor_id": "S1",
                "instrument_validity_flag": "out_of_calibration",
                "anomalies": [anom(0, 2, "high")],
            }
        ]
    }
    windows = sc()._extract_anomaly_windows(ts)
    assert windows[0]["instrument_validity_flag"] == "out_of_calibration"


def test_extract_anomaly_windows_skips_missing_timestamp():
    ts = {"signals": [sig_block("S1", [{"severity": "medium"}])]}
    windows = sc()._extract_anomaly_windows(ts)
    assert windows == []


def test_extract_anomaly_windows_sorted_chronologically():
    ts = {"signals": [sig_block("S1", [anom(5, 7), anom(1, 3)])]}
    windows = sc()._extract_anomaly_windows(ts)
    assert windows[0]["start"] < windows[1]["start"]


def test_extract_anomaly_windows_empty_signals():
    assert sc()._extract_anomaly_windows({"signals": []}) == []


def test_extract_anomaly_windows_no_key():
    assert sc()._extract_anomaly_windows({}) == []


def test_summarize_anomaly_windows_empty():
    summary = sc()._summarize_anomaly_windows([])
    assert summary["window_start"] is None
    assert summary["duration_hours"] is None


def test_summarize_anomaly_windows_single():
    summary = sc()._summarize_anomaly_windows([win(0, 2)])
    assert summary["duration_hours"] == 2.0
    assert summary["window_start"] == h(0)
    assert summary["window_end"] == h(2)


def test_summarize_anomaly_windows_multiple():
    summary = sc()._summarize_anomaly_windows([win(0, 3), win(5, 8)])
    assert summary["window_start"] == h(0)
    assert summary["window_end"] == h(8)
    assert approx(summary["duration_hours"], 8.0)


def test_extract_signal_ids_with_anomalies():
    ts = {"signals": [
        sig_block("S1", [anom(0, 1)]),
        sig_block("S2", [anom(2, 3)]),
    ]}
    ids = sc()._extract_signal_ids(ts)
    assert "S1" in ids and "S2" in ids


def test_extract_signal_ids_skips_sensor_without_anomalies():
    ts = {"signals": [
        sig_block("S1", [anom(0, 1)]),
        {"sensor_id": "S2", "anomalies": []},
    ]}
    ids = sc()._extract_signal_ids(ts)
    assert "S1" in ids
    assert "S2" not in ids


# ═══════════════════════════════════════════════════════════════════════════════
# Telemetry scoring
# ═══════════════════════════════════════════════════════════════════════════════

def test_telemetry_support_empty_returns_zero():
    assert sc()._telemetry_support_score({"signals": []}) == 0.0


def test_telemetry_support_high_severity():
    ts = {"signals": [sig_block("S1", [anom(0, 1, "high")])]}
    assert approx(sc()._telemetry_support_score(ts), 0.9)


def test_telemetry_support_medium_severity():
    ts = {"signals": [sig_block("S1", [anom(0, 1, "medium")])]}
    assert approx(sc()._telemetry_support_score(ts), 0.7)


def test_telemetry_support_low_severity():
    ts = {"signals": [sig_block("S1", [anom(0, 1, "low")])]}
    assert approx(sc()._telemetry_support_score(ts), 0.5)


def test_telemetry_support_unknown_severity_uses_floor():
    ts = {"signals": [sig_block("S1", [anom(0, 1, "unknown")])]}
    cfg = TSKRTemporalScorerConfig(telemetry_support_floor=0.35)
    assert approx(sc(cfg)._telemetry_support_score(ts), 0.35)


def test_telemetry_support_numeric_score_overrides_severity():
    """Numeric score=0.95 should win over medium (0.70) via max()."""
    a = {**anom(0, 1, "medium"), "score": 0.95}
    ts = {"signals": [{"sensor_id": "S1", "anomalies": [a]}]}
    result = sc()._telemetry_support_score(ts)
    assert approx(result, 0.95)


def test_telemetry_support_mean_over_multiple_anomalies():
    ts = {"signals": [sig_block("S1", [anom(0, 1, "high"), anom(2, 3, "low")])]}
    # (0.9 + 0.5) / 2 = 0.7
    assert approx(sc()._telemetry_support_score(ts), 0.7)


def test_severity_weight_numeric_score():
    assert approx(sc()._severity_weight({"severity_score": 0.8}), 0.8)


def test_severity_weight_high_string():
    assert approx(sc()._severity_weight({"severity": "high"}), 0.9)


def test_severity_weight_medium_string():
    assert approx(sc()._severity_weight({"severity": "medium"}), 0.7)


def test_severity_weight_low_string():
    assert approx(sc()._severity_weight({"severity": "low"}), 0.5)


def test_severity_weight_unknown_defaults_to_medium():
    assert approx(sc()._severity_weight({"severity": "critical"}), 0.5)


def test_severity_weight_penalized_for_invalid_instrument():
    scorer = sc()
    base = scorer._severity_weight({"severity": "high"})
    penalized = scorer._severity_weight(
        {"severity": "high", "instrument_validity_flag": "out_of_calibration"}
    )
    assert penalized < base
    assert approx(penalized, 0.495)  # 0.9 * 0.55


# ═══════════════════════════════════════════════════════════════════════════════
# Count / consistency
# ═══════════════════════════════════════════════════════════════════════════════

def test_effective_anomaly_count_empty():
    assert approx(sc()._effective_anomaly_count([]), 0.0)


def test_effective_anomaly_count_single_high():
    assert approx(sc()._effective_anomaly_count([win(0, 1, "high")]), 0.9)


def test_effective_anomaly_count_multiple_mixed():
    ws = [win(0, 1, "high"), win(2, 3, "low")]   # 0.9 + 0.5
    assert approx(sc()._effective_anomaly_count(ws), 1.4)


def test_anomaly_count_score_zero():
    assert sc()._anomaly_count_score(0) == 0.0


def test_anomaly_count_score_partial_below_one():
    assert approx(sc()._anomaly_count_score(0.5), 0.25)  # 0.5 * 0.5


def test_anomaly_count_score_exactly_one():
    assert approx(sc()._anomaly_count_score(1.0), 0.5)


def test_anomaly_count_score_between_one_and_two():
    assert approx(sc()._anomaly_count_score(1.5), 0.5)


def test_anomaly_count_score_exactly_two():
    assert approx(sc()._anomaly_count_score(2.0), 0.7)


def test_anomaly_count_score_between_two_and_three():
    assert approx(sc()._anomaly_count_score(2.5), 0.7)


def test_anomaly_count_score_exactly_three():
    assert approx(sc()._anomaly_count_score(3.0), 0.85)


def test_anomaly_count_score_exactly_four_or_more():
    assert approx(sc()._anomaly_count_score(4.0), 1.0)
    assert approx(sc()._anomaly_count_score(10.0), 1.0)


def test_lag_consistency_none_returns_half():
    assert approx(sc()._lag_consistency_score(None), 0.5)


def test_lag_consistency_very_tight():
    assert approx(sc()._lag_consistency_score(0.0), 1.0)
    assert approx(sc()._lag_consistency_score(0.25), 1.0)


def test_lag_consistency_moderate():
    assert approx(sc()._lag_consistency_score(0.5), 0.8)
    assert approx(sc()._lag_consistency_score(1.0), 0.8)


def test_lag_consistency_loose():
    assert approx(sc()._lag_consistency_score(2.0), 0.55)
    assert approx(sc()._lag_consistency_score(4.0), 0.55)


def test_lag_consistency_very_loose():
    assert approx(sc()._lag_consistency_score(5.0), 0.3)


def test_normalized_weighted_sum_rescales_non_convex_weights():
    scorer = sc()
    val = scorer._normalized_weighted_sum([
        (1.0, 0.45),
        (0.0, 0.30),
        (0.0, 0.10),
        (0.0, 0.10),
        (0.0, 0.15),
        (0.0, 0.10),
    ])
    assert approx(val, 0.375, tol=1e-3)  # 0.45 / 1.20


# ═══════════════════════════════════════════════════════════════════════════════
# Operator family
# ═══════════════════════════════════════════════════════════════════════════════

def test_infer_operator_family_full():
    result = sc()._infer_operator_family(h(0), h(10), [win(0, 1)])
    assert result == "interval_interval"


def test_infer_operator_family_no_anomalies():
    result = sc()._infer_operator_family(h(0), h(10), [])
    assert result == "interval_only"


def test_infer_operator_family_no_event_interval():
    result = sc()._infer_operator_family(None, None, [win(0, 1)])
    assert result == "anomaly_only"


def test_infer_operator_family_nothing():
    result = sc()._infer_operator_family(None, None, [])
    assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# _score_against_anomalies
# ═══════════════════════════════════════════════════════════════════════════════
# event = hours 0..10; epsilon = 0.5h (default)

def test_score_against_anomalies_no_windows_returns_fallback():
    s = sc()
    rel, mean_lag, std_lag, anomaly_score = s._score_against_anomalies(
        event_start=h(0), event_end=h(10), anomaly_windows=[]
    )
    assert rel == "unknown"
    assert mean_lag is None
    assert std_lag is None
    assert approx(anomaly_score, s.config.fallback_confidence)


def test_score_against_anomalies_no_event_start():
    s = sc()
    rel, _, _, score = s._score_against_anomalies(
        event_start=None, event_end=None, anomaly_windows=[win(-3, -1)]
    )
    assert rel == "unknown"
    assert approx(score, s.config.fallback_confidence)


def test_score_against_anomalies_precedes():
    """Single anomaly that precedes the event."""
    # anomaly ends at h(-1), event starts at h(0) → PRECEDES
    rel, mean_lag, std_lag, anomaly_score = sc()._score_against_anomalies(
        event_start=h(0), event_end=h(10), anomaly_windows=[win(-5, -1, "medium")]
    )
    assert rel == PRECEDES
    assert mean_lag is not None and mean_lag > 0     # causal: A onset before B onset
    assert std_lag == 0.0                            # single window → std = 0
    assert approx(anomaly_score, RELATION_SCORE[PRECEDES])


def test_score_against_anomalies_overlaps():
    """Anomaly starts before event and is still active at event onset → OVERLAPS."""
    rel, mean_lag, std_lag, anomaly_score = sc()._score_against_anomalies(
        event_start=h(0), event_end=h(10), anomaly_windows=[win(-3, 5, "high")]
    )
    assert rel == OVERLAPS
    assert mean_lag is not None and mean_lag > 0
    assert approx(anomaly_score, RELATION_SCORE[OVERLAPS])


def test_score_against_anomalies_contains():
    """Anomaly encompasses event → CONTAINS."""
    rel, _, _, anomaly_score = sc()._score_against_anomalies(
        event_start=h(0), event_end=h(10), anomaly_windows=[win(-2, 12, "medium")]
    )
    assert rel == CONTAINS
    assert approx(anomaly_score, RELATION_SCORE[CONTAINS])


def test_score_against_anomalies_during():
    """Anomaly starts inside event → DURING; no causal lag pairs."""
    rel, mean_lag, _, anomaly_score = sc()._score_against_anomalies(
        event_start=h(0), event_end=h(10), anomaly_windows=[win(3, 7, "medium")]
    )
    assert rel == DURING
    assert mean_lag is None    # DURING not in causal set
    assert approx(anomaly_score, RELATION_SCORE[DURING])


def test_score_against_anomalies_follows():
    """Anomaly starts after event ends → FOLLOWS."""
    rel, mean_lag, _, anomaly_score = sc()._score_against_anomalies(
        event_start=h(0), event_end=h(10), anomaly_windows=[win(11, 13, "medium")]
    )
    assert rel == FOLLOWS
    assert mean_lag is None    # FOLLOWS not in causal set
    assert approx(anomaly_score, RELATION_SCORE[FOLLOWS])


def test_score_against_anomalies_overlaps_beats_precedes():
    """When both OVERLAPS and PRECEDES are present, dominant = OVERLAPS."""
    windows = [
        win(-5, -1, "medium"),   # PRECEDES
        win(-3, 5, "medium"),    # OVERLAPS
    ]
    rel, _, _, _ = sc()._score_against_anomalies(
        event_start=h(0), event_end=h(10), anomaly_windows=windows
    )
    assert rel == OVERLAPS


def test_score_against_anomalies_severity_weighted_lag():
    """Mean lag is severity-weighted; high-severity window should dominate."""
    # high window: onset at h(-6), lag = 6h, weight = 0.9
    # low window:  onset at h(-1), lag = 1h, weight = 0.5
    # both PRECEDES (end before h(-0.5))
    windows = [
        win(-6, -2, "high"),
        win(-1.5, -1, "low"),
    ]
    _, mean_lag, _, _ = sc()._score_against_anomalies(
        event_start=h(0), event_end=h(10), anomaly_windows=windows
    )
    assert mean_lag is not None
    # Pure arithmetic mean would be (6+1)/2=3.5; weighted mean pulls toward 6h (heavy weight)
    expected = (6 * 0.9 + 1.5 * 0.5) / (0.9 + 0.5)
    assert approx(mean_lag, expected, tol=0.01)


def test_score_against_anomalies_single_causal_window_std_zero():
    """Single causal window → std_lag = 0."""
    _, _, std_lag, _ = sc()._score_against_anomalies(
        event_start=h(0), event_end=h(10), anomaly_windows=[win(-3, -1, "medium")]
    )
    assert std_lag == 0.0


def test_score_against_anomalies_weighted_score():
    """Anomaly score is severity-weighted mean of RELATION_SCORE values."""
    # Two windows: PRECEDES (0.75) weight 0.9, DURING (0.30) weight 0.5
    windows = [
        win(-5, -1, "high"),   # PRECEDES, weight=0.9
        win(3, 7, "low"),      # DURING, weight=0.5
    ]
    _, _, _, anomaly_score = sc()._score_against_anomalies(
        event_start=h(0), event_end=h(10), anomaly_windows=windows
    )
    expected = (RELATION_SCORE[PRECEDES] * 0.9 + RELATION_SCORE[DURING] * 0.5) / (0.9 + 0.5)
    assert approx(anomaly_score, expected, tol=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# Recurrence
# ═══════════════════════════════════════════════════════════════════════════════

def test_recurrence_trend_insufficient_data():
    assert TSKRTemporalScorerV1._recurrence_trend([]) == "insufficient_data"
    assert TSKRTemporalScorerV1._recurrence_trend([5.0]) == "insufficient_data"
    assert TSKRTemporalScorerV1._recurrence_trend([5.0, 3.0]) == "insufficient_data"


def test_recurrence_trend_increasing():
    # First half mean (1 interval) = 8.0; second half mean (2 intervals) = (3+1)/2=2
    # ratio = 2/8 = 0.25 < 0.75 → increasing
    assert TSKRTemporalScorerV1._recurrence_trend([8.0, 3.0, 1.0]) == "increasing"


def test_recurrence_trend_decreasing():
    # First half mean = 1.0; second half mean = (5+9)/2=7 → ratio=7 > 1.33
    assert TSKRTemporalScorerV1._recurrence_trend([1.0, 5.0, 9.0]) == "decreasing"


def test_recurrence_trend_stable():
    # ratio = 1.0 → stable
    assert TSKRTemporalScorerV1._recurrence_trend([4.0, 4.0, 4.0]) == "stable"


def test_build_recurrence_profile_no_past_events():
    profile = sc()._build_recurrence_profile(
        fm_id="FM-01", component_id="C-01", past_events=[]
    )
    assert profile.count == 0
    assert profile.mean_inter_event_days is None
    assert profile.trend == "insufficient_data"


def test_build_recurrence_profile_fm_match():
    pe = {
        "matched_failure_mode_ids": ["FM-01"],
        "component_id": None,
        "timestamp_start": h(-48).isoformat(),
        "resolved": True,
        "time_distance_days": 2,
    }
    profile = sc()._build_recurrence_profile(
        fm_id="FM-01", component_id="C-01", past_events=[pe]
    )
    assert profile.count == 1
    assert profile.most_recent_days_ago == 2


def test_build_recurrence_profile_component_match():
    pe = {
        "matched_failure_mode_ids": [],
        "component_id": "C-01",
        "timestamp_start": h(-72).isoformat(),
        "resolved": False,
        "time_distance_days": 3,
    }
    profile = sc()._build_recurrence_profile(
        fm_id="FM-99", component_id="C-01", past_events=[pe]
    )
    assert profile.count == 1
    assert profile.unresolved_count == 1


def test_build_recurrence_profile_mean_inter_event():
    events = [
        {
            "matched_failure_mode_ids": ["FM-01"],
            "component_id": None,
            "timestamp_start": h(-720).isoformat(),   # 30 days ago
            "resolved": True,
        },
        {
            "matched_failure_mode_ids": ["FM-01"],
            "component_id": None,
            "timestamp_start": h(-360).isoformat(),   # 15 days ago
            "resolved": True,
        },
    ]
    profile = sc()._build_recurrence_profile(
        fm_id="FM-01", component_id=None, past_events=events
    )
    assert profile.count == 2
    assert profile.mean_inter_event_days is not None
    assert approx(profile.mean_inter_event_days, 15.0, tol=0.1)


def test_score_from_recurrence_count_zero():
    p = RecurrenceProfile(fm_id=None, component_id=None, count=0,
                          mean_inter_event_days=None, trend="insufficient_data",
                          unresolved_count=0, most_recent_days_ago=None)
    assert sc()._score_from_recurrence_profile(p) == 0.0


def test_score_from_recurrence_count_one_base():
    p = RecurrenceProfile(fm_id=None, component_id=None, count=1,
                          mean_inter_event_days=None, trend="stable",
                          unresolved_count=0, most_recent_days_ago=None)
    assert approx(sc()._score_from_recurrence_profile(p), 0.35)


def test_score_from_recurrence_count_two_to_three():
    p = RecurrenceProfile(fm_id=None, component_id=None, count=3,
                          mean_inter_event_days=None, trend="stable",
                          unresolved_count=0, most_recent_days_ago=None)
    assert approx(sc()._score_from_recurrence_profile(p), 0.55)


def test_score_from_recurrence_count_four_to_six():
    p = RecurrenceProfile(fm_id=None, component_id=None, count=5,
                          mean_inter_event_days=None, trend="stable",
                          unresolved_count=0, most_recent_days_ago=None)
    assert approx(sc()._score_from_recurrence_profile(p), 0.70)


def test_score_from_recurrence_count_above_six():
    p = RecurrenceProfile(fm_id=None, component_id=None, count=8,
                          mean_inter_event_days=None, trend="stable",
                          unresolved_count=0, most_recent_days_ago=None)
    assert approx(sc()._score_from_recurrence_profile(p), 0.80)


def test_score_from_recurrence_increasing_trend_bonus():
    p = RecurrenceProfile(fm_id=None, component_id=None, count=1,
                          mean_inter_event_days=None, trend="increasing",
                          unresolved_count=0, most_recent_days_ago=None)
    assert approx(sc()._score_from_recurrence_profile(p), 0.35 + 0.15)


def test_score_from_recurrence_unresolved_bonus():
    p = RecurrenceProfile(fm_id=None, component_id=None, count=1,
                          mean_inter_event_days=None, trend="stable",
                          unresolved_count=1, most_recent_days_ago=None)
    assert approx(sc()._score_from_recurrence_profile(p), 0.35 + 0.10)


def test_score_from_recurrence_recent_bonus():
    p = RecurrenceProfile(fm_id=None, component_id=None, count=1,
                          mean_inter_event_days=None, trend="stable",
                          unresolved_count=0, most_recent_days_ago=30)
    assert approx(sc()._score_from_recurrence_profile(p), 0.35 + 0.05)


def test_score_from_recurrence_all_bonuses_clamped():
    """All bonuses on a high count should clamp at 1.0."""
    p = RecurrenceProfile(fm_id=None, component_id=None, count=8,
                          mean_inter_event_days=None, trend="increasing",
                          unresolved_count=2, most_recent_days_ago=10)
    result = sc()._score_from_recurrence_profile(p)
    assert result == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Latency alignment
# ═══════════════════════════════════════════════════════════════════════════════

def test_score_expected_latency_no_lag_returns_fallback():
    s = sc()
    result = s._score_expected_latency(mean_lag_hours=None, expected_min=3, expected_max=8)
    assert approx(result, s.config.fallback_confidence)


def test_score_expected_latency_no_bounds_returns_half():
    result = sc()._score_expected_latency(mean_lag_hours=5.0, expected_min=None, expected_max=None)
    assert approx(result, 0.5)


def test_score_expected_latency_within_range():
    result = sc()._score_expected_latency(mean_lag_hours=5.0, expected_min=3.0, expected_max=8.0)
    assert approx(result, 1.0)


def test_score_expected_latency_too_fast():
    # lag=1, mn=3 → 1 - (3-1)/3 = 1 - 0.6667 = 0.3333
    result = sc()._score_expected_latency(mean_lag_hours=1.0, expected_min=3.0, expected_max=8.0)
    assert approx(result, 1.0 - 2.0 / 3.0, tol=1e-3)


def test_score_expected_latency_too_slow():
    # lag=10, mx=8 → 1 - (10-8)/8 = 1 - 0.25 = 0.75
    result = sc()._score_expected_latency(mean_lag_hours=10.0, expected_min=3.0, expected_max=8.0)
    assert approx(result, 0.75, tol=1e-3)


def test_score_expected_latency_only_min_satisfied():
    result = sc()._score_expected_latency(mean_lag_hours=5.0, expected_min=3.0, expected_max=None)
    assert approx(result, 1.0)


def test_score_expected_latency_only_min_not_satisfied():
    # lag=1, mn=3 → 1/3 = 0.3333
    result = sc()._score_expected_latency(mean_lag_hours=1.0, expected_min=3.0, expected_max=None)
    assert approx(result, 1.0 / 3.0, tol=1e-3)


def test_score_expected_latency_only_max_satisfied():
    result = sc()._score_expected_latency(mean_lag_hours=5.0, expected_min=None, expected_max=8.0)
    assert approx(result, 1.0)


def test_score_expected_latency_only_max_exceeded():
    # lag=10, mx=8 → 1 - (10-8)/8 = 0.75
    result = sc()._score_expected_latency(mean_lag_hours=10.0, expected_min=None, expected_max=8.0)
    assert approx(result, 0.75, tol=1e-3)


def test_latency_alignment_details_no_lag():
    details = sc()._latency_alignment_details(mean_lag_hours=None, expected_min=3, expected_max=8)
    assert details["latency_violation_type"] == "not_available"
    assert details["observed_lag_hours"] is None
    assert details["latency_alignment_score"] == sc().config.fallback_confidence


def test_latency_alignment_details_too_fast():
    details = sc()._latency_alignment_details(mean_lag_hours=1.0, expected_min=3.0, expected_max=8.0)
    assert details["latency_violation_type"] == "too_fast"
    assert approx(details["observed_lag_hours"], 1.0)


def test_latency_alignment_details_too_slow():
    details = sc()._latency_alignment_details(mean_lag_hours=10.0, expected_min=3.0, expected_max=8.0)
    assert details["latency_violation_type"] == "too_slow"


def test_latency_alignment_details_no_violation():
    details = sc()._latency_alignment_details(mean_lag_hours=5.0, expected_min=3.0, expected_max=8.0)
    assert details["latency_violation_type"] == "none"
    assert approx(details["latency_alignment_score"], 1.0)


def test_latency_alignment_details_no_bounds():
    details = sc()._latency_alignment_details(mean_lag_hours=5.0, expected_min=None, expected_max=None)
    assert details["latency_violation_type"] == "none"
    assert approx(details["latency_alignment_score"], 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# _score_failure_mode_pattern
# ═══════════════════════════════════════════════════════════════════════════════

def _make_fm(fm_id="FM-01", component_id="C-01",
             latency_min=None, latency_max=None) -> dict:
    return {
        "fm_id": fm_id,
        "component_id": component_id,
        "expected_latency_min_hours": latency_min,
        "expected_latency_max_hours": latency_max,
    }


def test_pattern_required_fields_present():
    """All required output fields must be present."""
    pattern = sc()._score_failure_mode_pattern(
        event_id="EVT-001", asset_id="ASSET-01",
        event_start=h(0), event_end=h(10),
        anomaly_windows=[win(-3, -1, "high")],
        anomaly_window_summary=sc()._summarize_anomaly_windows([win(-3, -1, "high")]),
        signal_ids=["S1"],
        telemetry_support=0.7,
        operator_family="interval_interval",
        fm=_make_fm(),
        past_events=[],
    )
    required = {
        "pattern_id", "event_id", "asset_id", "target_type", "target_id",
        "component_id", "relation", "confidence", "support",
        "temporal_contradiction", "recurrence_count", "recurrence_trend",
    }
    assert required.issubset(pattern.keys())


def test_pattern_no_temporal_contradiction_for_precedes():
    """PRECEDES relation with no latency violation → temporal_contradiction=False."""
    pattern = sc()._score_failure_mode_pattern(
        event_id="E1", asset_id="A1",
        event_start=h(0), event_end=h(10),
        anomaly_windows=[win(-5, -1, "medium")],
        anomaly_window_summary=sc()._summarize_anomaly_windows([win(-5, -1)]),
        signal_ids=["S1"],
        telemetry_support=0.7,
        operator_family="interval_interval",
        fm=_make_fm(latency_min=1.0, latency_max=10.0),
        past_events=[],
    )
    assert pattern["temporal_contradiction"] is False


def test_pattern_temporal_contradiction_for_follows():
    """FOLLOWS relation → temporal_contradiction=True."""
    pattern = sc()._score_failure_mode_pattern(
        event_id="E1", asset_id="A1",
        event_start=h(0), event_end=h(10),
        anomaly_windows=[win(11, 13, "medium")],   # FOLLOWS
        anomaly_window_summary=sc()._summarize_anomaly_windows([win(11, 13)]),
        signal_ids=["S1"],
        telemetry_support=0.5,
        operator_family="interval_interval",
        fm=_make_fm(),
        past_events=[],
    )
    assert pattern["temporal_contradiction"] is True


def test_pattern_temporal_contradiction_from_latency_violation():
    """too_fast latency violation → temporal_contradiction=True even without FOLLOWS."""
    pattern = sc()._score_failure_mode_pattern(
        event_id="E1", asset_id="A1",
        event_start=h(0), event_end=h(10),
        anomaly_windows=[win(-5, -1, "medium")],   # PRECEDES — causal, but...
        anomaly_window_summary=sc()._summarize_anomaly_windows([win(-5, -1)]),
        signal_ids=["S1"],
        telemetry_support=0.5,
        operator_family="interval_interval",
        fm=_make_fm(latency_min=10.0, latency_max=50.0),  # observed lag ≪ expected min
        past_events=[],
    )
    assert pattern["temporal_contradiction"] is True


def test_pattern_temporal_contradiction_from_stage_b_relation():
    """Stage-B follow relation should force temporal contradiction."""
    pattern = sc()._score_failure_mode_pattern(
        event_id="E1", asset_id="A1",
        event_start=h(0), event_end=h(10),
        anomaly_windows=[win(-5, -1, "medium")],   # PRECEDES on direct scoring
        anomaly_window_summary=sc()._summarize_anomaly_windows([win(-5, -1)]),
        signal_ids=["S1"],
        telemetry_support=0.7,
        operator_family="interval_interval",
        fm=_make_fm(latency_min=1.0, latency_max=10.0),
        past_events=[],
        stage_b_allen_relation="follows",
    )
    assert pattern["stage_b_temporal_contradiction"] is True
    assert pattern["temporal_contradiction"] is True


def test_pattern_confidence_clamped_at_one():
    """Confidence must never exceed 1.0."""
    pattern = sc()._score_failure_mode_pattern(
        event_id="E1", asset_id="A1",
        event_start=h(0), event_end=h(10),
        anomaly_windows=[win(-3, 5, "high")] * 5,
        anomaly_window_summary=sc()._summarize_anomaly_windows([win(-3, 5, "high")]),
        signal_ids=["S1"],
        telemetry_support=1.0,
        operator_family="interval_interval",
        fm=_make_fm(latency_min=2.0, latency_max=6.0),
        past_events=[],
    )
    assert pattern["confidence"] <= 1.0


def test_pattern_target_id_matches_fm_id():
    fm = _make_fm(fm_id="FM-SEAL-001")
    pattern = sc()._score_failure_mode_pattern(
        event_id="E1", asset_id="A1",
        event_start=h(0), event_end=h(10),
        anomaly_windows=[],
        anomaly_window_summary=sc()._summarize_anomaly_windows([]),
        signal_ids=[],
        telemetry_support=0.0,
        operator_family=None,
        fm=fm,
        past_events=[],
    )
    assert pattern["target_id"] == "FM-SEAL-001"
    assert pattern["pattern_id"] == "TSKR::FM-SEAL-001"


# ═══════════════════════════════════════════════════════════════════════════════
# Integration — score()
# ═══════════════════════════════════════════════════════════════════════════════

def _make_telemetry(start_h: float, end_h: float, severity: str = "high") -> dict:
    return {
        "asset_id": "ASSET-01",
        "signals": [sig_block("S1", [anom(start_h, end_h, severity)])],
    }


def _make_event(start_h: float = 0.0, end_h: float = 10.0) -> dict:
    return {
        "event_id": "EVT-001",
        "asset_id": "ASSET-01",
        "timestamp_start": h(start_h).isoformat(),
        "timestamp_end": h(end_h).isoformat(),
    }


def _make_kg_context(*fm_ids: str, out_of_boundary_anomalies=None) -> dict:
    return {
        "failure_modes": [_make_fm(fm_id=fid) for fid in fm_ids],
        "past_events": [],
        "out_of_boundary_anomalies": out_of_boundary_anomalies or [],
    }


def test_score_empty_failure_modes():
    result = sc().score(
        event=_make_event(),
        telemetry_summary=_make_telemetry(-3, -1),
        kg_context={"failure_modes": [], "past_events": []},
        operational_context=None,
        run_context={"run_id": "R-001"},
    )
    assert result["patterns"] == []
    assert result["summary"]["n_patterns"] == 0
    assert result["summary"]["has_temporal_support"] is False


def test_score_single_fm_produces_one_pattern():
    result = sc().score(
        event=_make_event(),
        telemetry_summary=_make_telemetry(-3, -1, "high"),
        kg_context=_make_kg_context("FM-01"),
        operational_context=None,
        run_context={"run_id": "R-001"},
    )
    assert len(result["patterns"]) == 1
    assert result["patterns"][0]["target_id"] == "FM-01"
    assert result["summary"]["n_patterns"] == 1


def test_score_patterns_sorted_by_confidence_descending():
    """Higher-confidence FM should appear first."""
    # FM-HIGH gets a causal OVERLAPS anomaly; FM-LOW gets a post-event FOLLOWS anomaly.
    ts = {
        "asset_id": "ASSET-01",
        "signals": [
            sig_block("S1", [anom(-3, 5, "high")]),   # OVERLAPS → strong confidence
        ],
    }
    kg = {
        "failure_modes": [
            {"fm_id": "FM-LOW", "component_id": "C1",
             "expected_latency_min_hours": None, "expected_latency_max_hours": None},
            {"fm_id": "FM-HIGH", "component_id": "C1",
             "expected_latency_min_hours": 1.0, "expected_latency_max_hours": 6.0},
        ],
        "past_events": [],
    }
    result = sc().score(
        event=_make_event(),
        telemetry_summary=ts,
        kg_context=kg,
        operational_context=None,
        run_context={"run_id": "R-001"},
    )
    confidences = [p["confidence"] for p in result["patterns"]]
    assert confidences == sorted(confidences, reverse=True)


def test_score_summary_fields():
    result = sc().score(
        event=_make_event(),
        telemetry_summary=_make_telemetry(-3, -1, "high"),
        kg_context=_make_kg_context("FM-01", "FM-02"),
        operational_context=None,
        run_context={"run_id": "R-42"},
    )
    s = result["summary"]
    assert s["n_patterns"] == 2
    assert s["mode"] == "deterministic_v1"
    assert "anomaly_point_count" in s
    assert "avg_confidence" in s
    assert "top_supported_targets" in s
    assert "tone_vocabulary_version" in s
    assert "dominant_tone" in s


def test_score_summary_reports_tone_uncertainty_for_transient_only():
    ts = {"asset_id": "ASSET-01", "signals": [sig_block("S1", [anom(-0.05, 0.0, "low")])]}
    result = sc().score(
        event=_make_event(),
        telemetry_summary=ts,
        kg_context=_make_kg_context("FM-01"),
        operational_context=None,
        run_context={"run_id": "R-42"},
    )
    summary = result["summary"]
    assert summary["dominant_tone"] == "transient_excursion"
    assert summary["tone_calibration_uncertainty"] is True


def test_score_provenance_fields():
    result = sc().score(
        event=_make_event(),
        telemetry_summary=_make_telemetry(-3, -1),
        kg_context=_make_kg_context("FM-01"),
        operational_context=None,
        run_context={"run_id": "R-42"},
    )
    prov = result["provenance"]
    assert prov["generated_by"] == "TSKRTemporalScorerV1"
    assert prov["run_id"] == "R-42"
    assert "generated_at" in prov


def test_score_event_and_asset_id_propagated():
    result = sc().score(
        event=_make_event(),
        telemetry_summary=_make_telemetry(-3, -1),
        kg_context=_make_kg_context("FM-01"),
        operational_context=None,
        run_context={"run_id": "R-001"},
    )
    assert result["event_id"] == "EVT-001"
    assert result["asset_id"] == "ASSET-01"


def test_score_temporal_contradiction_reflected_in_pattern():
    """FOLLOWS anomaly → temporal_contradiction=True in the resulting pattern."""
    ts = {"asset_id": "ASSET-01",
          "signals": [sig_block("S1", [anom(12, 14, "medium")])]}   # FOLLOWS
    result = sc().score(
        event=_make_event(),
        telemetry_summary=ts,
        kg_context=_make_kg_context("FM-01"),
        operational_context=None,
        run_context={"run_id": "R-001"},
    )
    assert result["patterns"][0]["temporal_contradiction"] is True


def test_score_applies_stage_b_allen_handshake():
    """Stage-C pattern should carry Stage-B follow contradiction for same component."""
    ts = {"asset_id": "ASSET-01", "signals": [sig_block("S1", [anom(-3, -1, "medium")])]}
    kg = _make_kg_context(
        "FM-01",
        out_of_boundary_anomalies=[
            {"component_id": "C-01", "allen_relation": "follows"},
        ],
    )
    result = sc().score(
        event=_make_event(),
        telemetry_summary=ts,
        kg_context=kg,
        operational_context=None,
        run_context={"run_id": "R-001"},
    )
    pattern = result["patterns"][0]
    assert pattern["stage_b_allen_relation"] == "follows"
    assert pattern["stage_b_temporal_contradiction"] is True
    assert pattern["temporal_contradiction"] is True


def test_score_summary_reports_unmatched_cr_stats():
    kg = _make_kg_context("FM-01")
    kg["past_events"] = [
        {"event_id": "CMMS::CR::1", "event_type": "cmms_cr", "matched_failure_mode_ids": ["FM-01"]},
        {"event_id": "CMMS::CR::2", "event_type": "cmms_cr", "matched_failure_mode_ids": []},
    ]
    result = sc().score(
        event=_make_event(),
        telemetry_summary=_make_telemetry(-3, -1),
        kg_context=kg,
        operational_context=None,
        run_context={"run_id": "R-001"},
    )
    summary = result["summary"]
    assert summary["total_cr_count"] == 2
    assert summary["unmatched_cr_count"] == 1
    assert 0.49 <= float(summary["unmatched_cr_rate"]) <= 0.51
    assert summary["high_cr_match_failure_rate"] is True


def test_score_summary_zero_unmatched_when_no_cr_history():
    result = sc().score(
        event=_make_event(),
        telemetry_summary=_make_telemetry(-3, -1),
        kg_context=_make_kg_context("FM-01"),
        operational_context=None,
        run_context={"run_id": "R-001"},
    )
    summary = result["summary"]
    assert summary["total_cr_count"] == 0
    assert summary["unmatched_cr_count"] == 0
    assert float(summary["unmatched_cr_rate"]) == 0.0
    assert summary["high_cr_match_failure_rate"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# Phase B — signal_support_score / recurrence_support_score / alarm-SOE restriction
# ═══════════════════════════════════════════════════════════════════════════════

def _alarm_log(start_h: float, severity: str = "HIGH", clock_ok: bool = True) -> dict:
    """Minimal alarm_log dict with one alarm activation."""
    return {
        "quality": {"clock_sync_ok": clock_ok},
        "alarms": [{"activated_at": h(start_h).isoformat(), "severity": severity}],
    }


def _soe_log(start_h: float, is_protection: bool = False, clock_ok: bool = True) -> dict:
    """Minimal soe_log dict with one SOE record."""
    return {
        "quality": {"clock_sync_ok": clock_ok},
        "records": [{"timestamp": h(start_h).isoformat(), "is_protection_signal": is_protection}],
    }


def test_phase_b_intermediates_present_in_output():
    """signal_support_score and recurrence_support_score must appear in pattern dict."""
    pattern = sc()._score_failure_mode_pattern(
        event_id="E1", asset_id="A1",
        event_start=h(0), event_end=h(10),
        anomaly_windows=[win(-3, -1, "high")],
        anomaly_window_summary=sc()._summarize_anomaly_windows([win(-3, -1, "high")]),
        signal_ids=["S1"],
        telemetry_support=0.7,
        operator_family="interval_interval",
        fm=_make_fm(),
        past_events=[],
    )
    assert "signal_support_score" in pattern
    assert "recurrence_support_score" in pattern
    assert 0.0 <= pattern["signal_support_score"] <= 1.0
    assert 0.0 <= pattern["recurrence_support_score"] <= 1.0


def test_phase_b_alarm_only_zero_anomaly_count():
    """Alarm-only input must not contribute to anomaly_count (telemetry restriction)."""
    alarm_wins = sc()._extract_alarm_windows(_alarm_log(-3.0))
    pattern = sc()._score_failure_mode_pattern(
        event_id="E1", asset_id="A1",
        event_start=h(0), event_end=h(10),
        anomaly_windows=[],            # no telemetry
        alarm_soe_windows=alarm_wins,
        anomaly_window_summary=sc()._summarize_anomaly_windows(alarm_wins),
        signal_ids=[],
        telemetry_support=0.0,
        operator_family="interval_interval",
        fm=_make_fm(),
        past_events=[],
    )
    assert pattern["anomaly_count"] == 0


def test_phase_b_soe_only_zero_anomaly_count():
    """SOE-only input must not contribute to anomaly_count (telemetry restriction)."""
    soe_wins = sc()._extract_soe_windows(_soe_log(-2.0, is_protection=True))
    pattern = sc()._score_failure_mode_pattern(
        event_id="E1", asset_id="A1",
        event_start=h(0), event_end=h(10),
        anomaly_windows=[],
        alarm_soe_windows=soe_wins,
        anomaly_window_summary=sc()._summarize_anomaly_windows(soe_wins),
        signal_ids=[],
        telemetry_support=0.0,
        operator_family="interval_interval",
        fm=_make_fm(),
        past_events=[],
    )
    assert pattern["anomaly_count"] == 0


def test_phase_b_telemetry_count_excludes_alarm_windows():
    """With telemetry + alarm, anomaly_count reflects only telemetry windows."""
    alarm_wins = sc()._extract_alarm_windows(_alarm_log(-4.0))
    telemetry_wins = [win(-3, -1, "high"), win(-5, -4, "medium")]
    pattern = sc()._score_failure_mode_pattern(
        event_id="E1", asset_id="A1",
        event_start=h(0), event_end=h(10),
        anomaly_windows=telemetry_wins,
        alarm_soe_windows=alarm_wins,
        anomaly_window_summary=sc()._summarize_anomaly_windows(telemetry_wins + alarm_wins),
        signal_ids=["S1"],
        telemetry_support=0.7,
        operator_family="interval_interval",
        fm=_make_fm(),
        past_events=[],
    )
    assert pattern["anomaly_count"] == len(telemetry_wins)


def test_phase_b_alarm_contributes_to_lag_consistency():
    """Alarm onset timing tightens lag spread → lag_consistency != no-window default."""
    # No telemetry, alarm at -2h (causal PRECEDES)
    alarm_wins = sc()._extract_alarm_windows(_alarm_log(-2.0))
    pattern = sc()._score_failure_mode_pattern(
        event_id="E1", asset_id="A1",
        event_start=h(0), event_end=h(10),
        anomaly_windows=[],
        alarm_soe_windows=alarm_wins,
        anomaly_window_summary=sc()._summarize_anomaly_windows(alarm_wins),
        signal_ids=[],
        telemetry_support=0.0,
        operator_family="interval_interval",
        fm=_make_fm(),
        past_events=[],
    )
    # With a single causal alarm window std_lag = 0 → lag_consistency = 1.0
    assert pattern["lag_consistency"] == 1.0


def test_phase_b_no_history_recurrence_score_zero():
    """No past events → recurrence_support_score == 0."""
    pattern = sc()._score_failure_mode_pattern(
        event_id="E1", asset_id="A1",
        event_start=h(0), event_end=h(10),
        anomaly_windows=[win(-3, -1, "high")],
        anomaly_window_summary=sc()._summarize_anomaly_windows([win(-3, -1, "high")]),
        signal_ids=["S1"],
        telemetry_support=0.7,
        operator_family="interval_interval",
        fm=_make_fm(),
        past_events=[],
    )
    assert pattern["recurrence_support_score"] == 0.0


def test_phase_b_with_history_recurrence_score_positive():
    """Past events matching the FM → recurrence_support_score > 0."""
    past = [{"matched_failure_mode_ids": ["FM-01"], "component_id": None,
             "timestamp_start": h(-720).isoformat(), "resolved": True, "time_distance_days": 30}]
    pattern = sc()._score_failure_mode_pattern(
        event_id="E1", asset_id="A1",
        event_start=h(0), event_end=h(10),
        anomaly_windows=[win(-3, -1, "high")],
        anomaly_window_summary=sc()._summarize_anomaly_windows([win(-3, -1, "high")]),
        signal_ids=["S1"],
        telemetry_support=0.7,
        operator_family="interval_interval",
        fm=_make_fm(),
        past_events=past,
    )
    assert pattern["recurrence_support_score"] > 0.0


def test_phase_b_signal_support_increases_with_telemetry():
    """Adding strong telemetry should increase signal_support_score."""
    pattern_no_tel = sc()._score_failure_mode_pattern(
        event_id="E1", asset_id="A1",
        event_start=h(0), event_end=h(10),
        anomaly_windows=[],
        anomaly_window_summary=sc()._summarize_anomaly_windows([]),
        signal_ids=[],
        telemetry_support=0.0,
        operator_family=None,
        fm=_make_fm(),
        past_events=[],
    )
    pattern_tel = sc()._score_failure_mode_pattern(
        event_id="E1", asset_id="A1",
        event_start=h(0), event_end=h(10),
        anomaly_windows=[win(-3, -1, "high")] * 4,
        anomaly_window_summary=sc()._summarize_anomaly_windows([win(-3, -1, "high")]),
        signal_ids=["S1"],
        telemetry_support=0.9,
        operator_family="interval_interval",
        fm=_make_fm(latency_min=1.0, latency_max=5.0),
        past_events=[],
    )
    assert pattern_tel["signal_support_score"] > pattern_no_tel["signal_support_score"]


def test_phase_b_score_entry_point_alarm_only_anomaly_count_zero():
    """score() with alarm_log only (no telemetry anomalies) → pattern anomaly_count == 0."""
    ts = {"asset_id": "ASSET-01", "signals": []}  # no telemetry anomalies
    result = sc().score(
        event=_make_event(),
        telemetry_summary=ts,
        kg_context=_make_kg_context("FM-01"),
        operational_context=None,
        run_context={"run_id": "R-alarm"},
        alarm_log=_alarm_log(-3.0),
    )
    pattern = result["patterns"][0]
    assert pattern["anomaly_count"] == 0
    # alarm should still be visible in total window count
    assert result["summary"]["anomaly_point_count"] == 1


def test_phase_b_score_entry_point_soe_only_anomaly_count_zero():
    """score() with soe_log only (no telemetry anomalies) → pattern anomaly_count == 0."""
    ts = {"asset_id": "ASSET-01", "signals": []}
    result = sc().score(
        event=_make_event(),
        telemetry_summary=ts,
        kg_context=_make_kg_context("FM-01"),
        operational_context=None,
        run_context={"run_id": "R-soe"},
        soe_log=_soe_log(-2.0),
    )
    pattern = result["patterns"][0]
    assert pattern["anomaly_count"] == 0
    assert result["summary"]["anomaly_point_count"] == 1


def test_phase_b_intermediates_both_present_in_score_output():
    """score() output patterns include both Phase B intermediates."""
    result = sc().score(
        event=_make_event(),
        telemetry_summary=_make_telemetry(-3, -1, "high"),
        kg_context=_make_kg_context("FM-01"),
        operational_context=None,
        run_context={"run_id": "R-001"},
    )
    pattern = result["patterns"][0]
    assert "signal_support_score" in pattern
    assert "recurrence_support_score" in pattern


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests.")
