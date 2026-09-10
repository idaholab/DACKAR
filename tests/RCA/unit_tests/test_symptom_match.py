"""
test_symptom_match.py — standalone unit tests for
RuleBasedCausalityEngineV32._symptom_match_score and _dominant_telemetry_pattern

Run directly:   python test_symptom_match.py
Or via pytest:  pytest test_symptom_match.py

Key invariants:
  1. No symptom data on either side → neutral 0.5
  2. Pattern match only → 1.0  (pattern_weight=0.6 dominates)
  3. Pattern mismatch only → 0.0
  4. Perfect symptom type overlap only → 1.0  (type_weight=0.4)
  5. Partial symptom type overlap → F1 < 1.0
  6. Both signals, both match → 1.0
  7. Pattern match + zero type overlap → 0.6
  8. Event anomaly_pattern used as fallback when no telemetry anomalies
  9. expected_symptoms string (semicolon-delimited) parsed as fallback for expected_symptom_types
 10. _dominant_telemetry_pattern: no signals → None
 11. _dominant_telemetry_pattern: "unknown" patterns excluded
 12. _dominant_telemetry_pattern: returns most frequent pattern
 13. Pattern aliases/format variants normalize to the same class
 14. Symptom type matching tolerates case and delimiter differences
"""
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from orchestrators.causality_engine_v32 import RuleBasedCausalityEngineV32


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_engine():
    return RuleBasedCausalityEngineV32()


def make_fm(fm_id="FM-001", expected_anomaly_pattern=None, expected_symptom_types=None, expected_symptoms=None):
    fm = {"fm_id": fm_id}
    if expected_anomaly_pattern is not None:
        fm["expected_anomaly_pattern"] = expected_anomaly_pattern
    if expected_symptom_types is not None:
        fm["expected_symptom_types"] = expected_symptom_types
    if expected_symptoms is not None:
        fm["expected_symptoms"] = expected_symptoms
    return fm


def make_event(anomaly_pattern=None, symptom_types=None):
    event = {"id": "EVT-001", "asset_id": "ASSET-001"}
    if anomaly_pattern or symptom_types:
        sig = {}
        if anomaly_pattern:
            sig["anomaly_pattern"] = anomaly_pattern
        if symptom_types:
            sig["symptom_types"] = symptom_types
        event["symptom_signature"] = sig
    return event


def make_telemetry(signals=None):
    return {"signals": signals or []}


def make_signal_with_anomaly(sensor_id, pattern):
    return {
        "sensor_id": sensor_id,
        "anomalies": [{"pattern": pattern, "severity": "medium"}],
    }


def assert_approx(actual, expected, tol=0.01, label=""):
    assert abs(actual - expected) <= tol, (
        f"{label}: expected ~{expected}, got {actual}"
    )


# ── _symptom_match_score tests ─────────────────────────────────────────────

def test_no_symptom_data_returns_neutral():
    """No fm_pattern, no fm_types, no event types → 0.5."""
    e = make_engine()
    score = e._symptom_match_score(
        make_event(),
        make_fm(),
        make_telemetry(),
    )
    assert_approx(score, 0.5, label="neutral")
    print("  PASS test_no_symptom_data_returns_neutral")


def test_pattern_match_only_returns_1():
    """
    FM expected 'drift', telemetry dominant pattern 'drift', no symptom types.
    pattern_score=1.0, pattern_weight=0.6, type_weight=0.0
    result = 1.0*0.6 / 0.6 = 1.0
    """
    e = make_engine()
    score = e._symptom_match_score(
        make_event(),
        make_fm(expected_anomaly_pattern="drift"),
        make_telemetry(signals=[make_signal_with_anomaly("S1", "drift")]),
    )
    assert_approx(score, 1.0, label="pattern match")
    print("  PASS test_pattern_match_only_returns_1")


def test_pattern_mismatch_only_returns_0():
    """
    FM expected 'drift', telemetry dominant pattern 'spike'.
    pattern_score=0.0, pattern_weight=0.6
    result = 0.0*0.6 / 0.6 = 0.0
    """
    e = make_engine()
    score = e._symptom_match_score(
        make_event(),
        make_fm(expected_anomaly_pattern="drift"),
        make_telemetry(signals=[make_signal_with_anomaly("S1", "spike")]),
    )
    assert_approx(score, 0.0, label="pattern mismatch")
    print("  PASS test_pattern_mismatch_only_returns_0")


def test_symptom_type_perfect_match_returns_1():
    """
    No pattern data. FM types = event types → F1 = 1.0.
    type_weight=0.4 → result = 1.0
    """
    e = make_engine()
    score = e._symptom_match_score(
        make_event(symptom_types=["DO_elevation", "backpressure_increase"]),
        make_fm(expected_symptom_types=["DO_elevation", "backpressure_increase"]),
        make_telemetry(),
    )
    assert_approx(score, 1.0, label="perfect type match")
    print("  PASS test_symptom_type_perfect_match_returns_1")


def test_symptom_type_partial_match():
    """
    FM types = {A, B, C}, event types = {A, B}.
    precision=1.0, recall=2/3, F1 = 2*(2/3)/(5/3) = 0.8
    result = 0.8
    """
    e = make_engine()
    score = e._symptom_match_score(
        make_event(symptom_types=["A", "B"]),
        make_fm(expected_symptom_types=["A", "B", "C"]),
        make_telemetry(),
    )
    assert_approx(score, 0.8, label="partial type match")
    print("  PASS test_symptom_type_partial_match")


def test_both_signals_match_returns_1():
    """
    Pattern match (1.0, w=0.6) + full type overlap (1.0, w=0.4) → 1.0
    """
    e = make_engine()
    score = e._symptom_match_score(
        make_event(symptom_types=["DO_elevation"]),
        make_fm(expected_anomaly_pattern="drift", expected_symptom_types=["DO_elevation"]),
        make_telemetry(signals=[make_signal_with_anomaly("S1", "drift")]),
    )
    assert_approx(score, 1.0, label="both match")
    print("  PASS test_both_signals_match_returns_1")


def test_pattern_match_zero_type_overlap():
    """
    Pattern match (1.0, w=0.6) + zero type overlap (0.0, w=0.4) → 0.6
    """
    e = make_engine()
    score = e._symptom_match_score(
        make_event(symptom_types=["vibration"]),
        make_fm(expected_anomaly_pattern="drift", expected_symptom_types=["DO_elevation"]),
        make_telemetry(signals=[make_signal_with_anomaly("S1", "drift")]),
    )
    assert_approx(score, 0.6, label="pattern match, zero type overlap")
    print("  PASS test_pattern_match_zero_type_overlap")


def test_event_anomaly_pattern_fallback():
    """
    No telemetry anomalies → _dominant_telemetry_pattern returns None.
    Fallback to event.symptom_signature.anomaly_pattern = 'drift'.
    FM expected 'drift' → match → 1.0
    """
    e = make_engine()
    score = e._symptom_match_score(
        make_event(anomaly_pattern="drift"),   # no telemetry, use event fallback
        make_fm(expected_anomaly_pattern="drift"),
        make_telemetry(signals=[]),             # no anomalies
    )
    assert_approx(score, 1.0, label="event fallback pattern match")
    print("  PASS test_event_anomaly_pattern_fallback")


def test_expected_symptoms_string_fallback():
    """
    FM has expected_symptoms='DO_elevation;backpressure_increase' (string),
    no expected_symptom_types list. Should parse semicolon-delimited string.
    event types = {DO_elevation} → recall=0.5, precision=1.0, F1=2/3 ≈ 0.667
    """
    e = make_engine()
    score = e._symptom_match_score(
        make_event(symptom_types=["DO_elevation"]),
        make_fm(expected_symptoms="DO_elevation;backpressure_increase"),
        make_telemetry(),
    )
    assert_approx(score, 0.667, tol=0.02, label="string symptoms fallback")
    print("  PASS test_expected_symptoms_string_fallback")


def test_no_event_types_but_fm_has_types():
    """
    FM has expected_symptom_types but event has no symptom_types → type sub-signal absent.
    Falls back to neutral only if no pattern data either.
    """
    e = make_engine()
    score = e._symptom_match_score(
        make_event(),  # no symptom_signature
        make_fm(expected_symptom_types=["DO_elevation", "backpressure"]),
        make_telemetry(),
    )
    # type_weight=0.0 (no event types), pattern_weight=0.0 (no fm_pattern) → neutral
    assert_approx(score, 0.5, label="no event types")
    print("  PASS test_no_event_types_but_fm_has_types")


def test_pattern_alias_normalization_matches():
    """
    FM pattern 'gradual_drift' and telemetry pattern 'drift' should match after
    normalization/canonicalization.
    """
    e = make_engine()
    score = e._symptom_match_score(
        make_event(),
        make_fm(expected_anomaly_pattern="gradual_drift"),
        make_telemetry(signals=[make_signal_with_anomaly("S1", "drift")]),
    )
    assert_approx(score, 1.0, label="pattern alias normalization")
    print("  PASS test_pattern_alias_normalization_matches")


def test_symptom_type_case_and_delimiter_normalization():
    """
    expected_symptoms string with mixed delimiters/case should align with
    event symptom_types after normalization.
    """
    e = make_engine()
    score = e._symptom_match_score(
        make_event(symptom_types=["DO Elevation", "BackPressure-Increase"]),
        make_fm(expected_symptoms="do_elevation; backpressure increase"),
        make_telemetry(),
    )
    assert score >= 0.95, f"normalized symptom type match should be high, got {score}"
    print("  PASS test_symptom_type_case_and_delimiter_normalization")


# ── _dominant_telemetry_pattern tests ─────────────────────────────────────

def test_dominant_pattern_no_signals():
    """No signals → None."""
    e = make_engine()
    assert e._dominant_telemetry_pattern(make_telemetry()) is None
    print("  PASS test_dominant_pattern_no_signals")


def test_dominant_pattern_no_anomalies():
    """Signals with no anomalies → None."""
    e = make_engine()
    assert e._dominant_telemetry_pattern(make_telemetry(signals=[{"sensor_id": "S1", "anomalies": []}])) is None
    print("  PASS test_dominant_pattern_no_anomalies")


def test_dominant_pattern_single():
    """Single anomaly → returns its pattern."""
    e = make_engine()
    ts = make_telemetry(signals=[make_signal_with_anomaly("S1", "drift")])
    assert e._dominant_telemetry_pattern(ts) == "drift"
    print("  PASS test_dominant_pattern_single")


def test_dominant_pattern_excludes_unknown():
    """'unknown' patterns are excluded; next-most-frequent wins."""
    e = make_engine()
    ts = make_telemetry(signals=[
        {"sensor_id": "S1", "anomalies": [{"pattern": "unknown"}, {"pattern": "drift"}]},
        {"sensor_id": "S2", "anomalies": [{"pattern": "unknown"}, {"pattern": "unknown"}]},
    ])
    # "unknown" count=3 but excluded; "drift" count=1 → wins
    assert e._dominant_telemetry_pattern(ts) == "drift"
    print("  PASS test_dominant_pattern_excludes_unknown")


def test_dominant_pattern_returns_most_frequent():
    """Multiple patterns → returns most frequent."""
    e = make_engine()
    ts = make_telemetry(signals=[
        {"sensor_id": "S1", "anomalies": [{"pattern": "drift"}, {"pattern": "spike"}]},
        {"sensor_id": "S2", "anomalies": [{"pattern": "drift"}, {"pattern": "drift"}]},
    ])
    # drift=3, spike=1 → drift
    assert e._dominant_telemetry_pattern(ts) == "drift"
    print("  PASS test_dominant_pattern_returns_most_frequent")


# ── Main runner ───────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_no_symptom_data_returns_neutral,
    test_pattern_match_only_returns_1,
    test_pattern_mismatch_only_returns_0,
    test_symptom_type_perfect_match_returns_1,
    test_symptom_type_partial_match,
    test_both_signals_match_returns_1,
    test_pattern_match_zero_type_overlap,
    test_event_anomaly_pattern_fallback,
    test_expected_symptoms_string_fallback,
    test_no_event_types_but_fm_has_types,
    test_pattern_alias_normalization_matches,
    test_symptom_type_case_and_delimiter_normalization,
    test_dominant_pattern_no_signals,
    test_dominant_pattern_no_anomalies,
    test_dominant_pattern_single,
    test_dominant_pattern_excludes_unknown,
    test_dominant_pattern_returns_most_frequent,
]


def run_all():
    print(f"\n=== test_symptom_match ({len(ALL_TESTS)} tests) ===")
    passed, failed = 0, 0
    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
            import traceback
            print(f"  FAIL {fn.__name__}: {exc}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
