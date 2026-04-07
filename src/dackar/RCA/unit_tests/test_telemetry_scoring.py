"""
test_telemetry_scoring.py — standalone unit tests for
RuleBasedCausalityEngineV32._telemetry_score_for_fm,
                           _temporal_score_for_fm, and
                           _temporal_posture

Run directly:   python test_telemetry_scoring.py
Or via pytest:  pytest test_telemetry_scoring.py

_telemetry_score_for_fm formula:
  no anomalies         → 0.20 baseline
  base                 = min(1.0, 0.35 + 0.12 * anomaly_count + 0.08 * severity_points)
  telemetry seed type  → +0.10
  pattern match        → +0.12
  pattern mismatch     → -0.08

_temporal_score_for_fm formula (no TSKR):
  no anomalies → 0.075  (temporal_precedence=0.30 × 0.25 weight)
  anomalies    → 0.3925 (tskr fallback=0.55, precedence=0.50, latency fallback=0.30)
  with TSKR "precedes" (conf=0.80, latency=0.70, support=0.50) → 0.78
  temporal_contradiction → subtract 0.25

_temporal_posture:
  temporal_contradiction=True               → "contradicted"
  temporal_score≥0.65 + precedence≥0.70 + latency≥0.60 → "supported"
  temporal_score≥0.40                       → "partial"
  otherwise                                 → "weak"
"""
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from orchestrators.causality_engine_v32 import RuleBasedCausalityEngineV32


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_engine():
    return RuleBasedCausalityEngineV32()


def make_fm(fm_id="FM-001", expected_anomaly_pattern=None):
    fm = {"fm_id": fm_id}
    if expected_anomaly_pattern:
        fm["expected_anomaly_pattern"] = expected_anomaly_pattern
    return fm


def make_telemetry(signals=None):
    return {"signals": signals or []}


def make_signal(sensor_id, anomalies=None):
    return {"sensor_id": sensor_id, "anomalies": anomalies or []}


def make_anomaly(pattern="drift", severity="medium"):
    return {"pattern": pattern, "severity": severity}


def make_components(component_id, seed_match_type=None):
    c = {"component_id": component_id}
    if seed_match_type:
        c["seed_match_type"] = seed_match_type
    return {component_id: c}


def make_tskr_index(fm_id, confidence, relation, latency_alignment_score=0.0, support=0.0,
                    temporal_contradiction=False):
    pattern = {
        "target_id": fm_id,
        "confidence": confidence,
        "relation": relation,
        "latency_alignment_score": latency_alignment_score,
        "support": support,
        "temporal_contradiction": temporal_contradiction,
    }
    return {fm_id: [pattern]}


def assert_approx(actual, expected, tol=0.005, label=""):
    assert abs(actual - expected) <= tol, (
        f"{label}: expected ~{expected}, got {actual}"
    )


# ── _telemetry_score_for_fm tests ─────────────────────────────────────────

def test_telemetry_no_signals_returns_baseline():
    """No signals → 0.20."""
    e = make_engine()
    assert e._telemetry_score_for_fm(make_telemetry(), make_fm(), None, {}) == 0.20
    print("  PASS test_telemetry_no_signals_returns_baseline")


def test_telemetry_signals_but_no_anomalies():
    """Signals exist but no anomalies → 0.20."""
    e = make_engine()
    ts = make_telemetry(signals=[make_signal("S1", anomalies=[])])
    assert e._telemetry_score_for_fm(ts, make_fm(), None, {}) == 0.20
    print("  PASS test_telemetry_signals_but_no_anomalies")


def test_telemetry_one_high_anomaly():
    """
    1 high anomaly: severity_points=1.0
    base = 0.35 + 0.12*1 + 0.08*1.0 = 0.55
    """
    e = make_engine()
    ts = make_telemetry(signals=[make_signal("S1", [make_anomaly(severity="high")])])
    assert_approx(e._telemetry_score_for_fm(ts, make_fm(), None, {}), 0.55, label="1 high anomaly")
    print("  PASS test_telemetry_one_high_anomaly")


def test_telemetry_one_medium_anomaly():
    """
    1 medium anomaly: severity_points=0.7
    base = 0.35 + 0.12*1 + 0.08*0.7 = 0.35 + 0.12 + 0.056 = 0.526
    """
    e = make_engine()
    ts = make_telemetry(signals=[make_signal("S1", [make_anomaly(severity="medium")])])
    assert_approx(e._telemetry_score_for_fm(ts, make_fm(), None, {}), 0.526, label="1 medium anomaly")
    print("  PASS test_telemetry_one_medium_anomaly")


def test_telemetry_seed_component_adds_bonus():
    """
    1 medium anomaly (base=0.526) + telemetry seed → +0.10 = 0.626
    """
    e = make_engine()
    ts = make_telemetry(signals=[make_signal("S1", [make_anomaly(severity="medium")])])
    comps = make_components("COMP-01", seed_match_type="telemetry")
    assert_approx(
        e._telemetry_score_for_fm(ts, make_fm(), "COMP-01", comps),
        0.626, label="seed bonus"
    )
    print("  PASS test_telemetry_seed_component_adds_bonus")


def test_telemetry_pattern_match_adds_bonus():
    """
    1 medium anomaly (base=0.526) + pattern match → +0.12 = 0.646
    """
    e = make_engine()
    ts = make_telemetry(signals=[make_signal("S1", [make_anomaly(pattern="drift", severity="medium")])])
    assert_approx(
        e._telemetry_score_for_fm(ts, make_fm(expected_anomaly_pattern="drift"), None, {}),
        0.646, label="pattern match bonus"
    )
    print("  PASS test_telemetry_pattern_match_adds_bonus")


def test_telemetry_pattern_mismatch_applies_penalty():
    """
    1 medium anomaly (base=0.526) + pattern mismatch → -0.08 = 0.446
    """
    e = make_engine()
    ts = make_telemetry(signals=[make_signal("S1", [make_anomaly(pattern="spike", severity="medium")])])
    assert_approx(
        e._telemetry_score_for_fm(ts, make_fm(expected_anomaly_pattern="drift"), None, {}),
        0.446, label="pattern mismatch penalty"
    )
    print("  PASS test_telemetry_pattern_mismatch_applies_penalty")


def test_telemetry_capped_at_1():
    """Many high anomalies cannot exceed 1.0."""
    e = make_engine()
    ts = make_telemetry(signals=[
        make_signal("S1", [make_anomaly(severity="high")] * 5),
        make_signal("S2", [make_anomaly(severity="high")] * 5),
    ])
    score = e._telemetry_score_for_fm(ts, make_fm(), None, {})
    assert score <= 1.0
    assert score > 0.35
    print("  PASS test_telemetry_capped_at_1")


# ── _temporal_score_for_fm tests ──────────────────────────────────────────

def test_temporal_no_anomalies_no_tskr():
    """
    No anomaly signals, no TSKR pattern.
    temporal_precedence = 0.30 (unknown relation, no anomalies)
    temporal = 0.25*0.30 = 0.075
    """
    e = make_engine()
    result = e._temporal_score_for_fm(make_fm(), make_telemetry(), None, {})
    assert_approx(result["temporal"], 0.075, label="no anomalies, no tskr")
    print("  PASS test_temporal_no_anomalies_no_tskr")


def test_temporal_anomalies_no_tskr():
    """
    Anomaly signals present, no TSKR.
    tskr fallback=0.55, precedence=0.50 (unknown+anomalies), latency fallback=0.30
    temporal = 0.35*0.55 + 0.25*0.50 + 0.25*0.30 + 0.15*0.0
             = 0.1925 + 0.125 + 0.075 = 0.3925
    """
    e = make_engine()
    ts = make_telemetry(signals=[make_signal("S1", [make_anomaly()])])
    result = e._temporal_score_for_fm(make_fm(), ts, None, {})
    assert_approx(result["temporal"], 0.3925, tol=0.01, label="anomalies, no tskr")
    assert result["tskr_pattern_match"] == 0.55
    print("  PASS test_temporal_anomalies_no_tskr")


def test_temporal_with_tskr_precedes():
    """
    TSKR pattern: precedes, confidence=0.80, latency_alignment=0.70, support=0.50
    temporal = 0.35*0.80 + 0.25*1.0 + 0.25*0.70 + 0.15*0.50
             = 0.28 + 0.25 + 0.175 + 0.075 = 0.78
    """
    e = make_engine()
    tskr_idx = make_tskr_index("FM-001", confidence=0.80, relation="precedes",
                                latency_alignment_score=0.70, support=0.50)
    result = e._temporal_score_for_fm(make_fm("FM-001"), make_telemetry(), None, tskr_idx)
    assert_approx(result["temporal"], 0.78, tol=0.01, label="tskr precedes")
    assert result["relation"] == "precedes"
    print("  PASS test_temporal_with_tskr_precedes")


def test_temporal_contradiction_applies_penalty():
    """
    Same TSKR as above (0.78) but temporal_contradiction=True → 0.78 - 0.25 = 0.53
    """
    e = make_engine()
    tskr_idx = make_tskr_index("FM-001", confidence=0.80, relation="precedes",
                                latency_alignment_score=0.70, support=0.50,
                                temporal_contradiction=True)
    result = e._temporal_score_for_fm(make_fm("FM-001"), make_telemetry(), None, tskr_idx)
    assert_approx(result["temporal"], 0.53, tol=0.01, label="contradiction penalty")
    assert result["temporal_contradiction"] is True
    print("  PASS test_temporal_contradiction_applies_penalty")


def test_temporal_follows_relation_low_precedence():
    """
    'follows' relation → temporal_precedence=0.20 (anomaly appeared after event).
    With no other data, temporal = 0.25*0.20 = 0.05.
    """
    e = make_engine()
    tskr_idx = make_tskr_index("FM-001", confidence=0.0, relation="follows")
    result = e._temporal_score_for_fm(make_fm("FM-001"), make_telemetry(), None, tskr_idx)
    assert result["temporal_precedence"] == 0.20
    print("  PASS test_temporal_follows_relation_low_precedence")


# ── _temporal_posture tests ────────────────────────────────────────────────

def test_posture_temporal_contradiction():
    e = make_engine()
    assert e._temporal_posture(0.90, 0.90, 0.90, temporal_contradiction=True) == "contradicted"
    print("  PASS test_posture_temporal_contradiction")


def test_posture_supported():
    """temporal_score≥0.65, precedence≥0.70, latency≥0.60 → supported."""
    e = make_engine()
    assert e._temporal_posture(0.70, 0.75, 0.65, temporal_contradiction=False) == "supported"
    print("  PASS test_posture_supported")


def test_posture_partial():
    """temporal_score≥0.40 but not all 3 thresholds → partial."""
    e = make_engine()
    assert e._temporal_posture(0.45, 0.50, 0.30, temporal_contradiction=False) == "partial"
    print("  PASS test_posture_partial")


def test_posture_weak():
    """temporal_score < 0.40 → weak."""
    e = make_engine()
    assert e._temporal_posture(0.20, 0.30, 0.10, temporal_contradiction=False) == "weak"
    print("  PASS test_posture_weak")


def test_posture_supported_requires_all_three_thresholds():
    """High temporal_score but low precedence → falls to 'partial' not 'supported'."""
    e = make_engine()
    result = e._temporal_posture(0.70, 0.50, 0.65, temporal_contradiction=False)
    assert result != "supported", f"Should not be supported with low precedence; got {result}"
    print("  PASS test_posture_supported_requires_all_three_thresholds")


# ── Main runner ───────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_telemetry_no_signals_returns_baseline,
    test_telemetry_signals_but_no_anomalies,
    test_telemetry_one_high_anomaly,
    test_telemetry_one_medium_anomaly,
    test_telemetry_seed_component_adds_bonus,
    test_telemetry_pattern_match_adds_bonus,
    test_telemetry_pattern_mismatch_applies_penalty,
    test_telemetry_capped_at_1,
    test_temporal_no_anomalies_no_tskr,
    test_temporal_anomalies_no_tskr,
    test_temporal_with_tskr_precedes,
    test_temporal_contradiction_applies_penalty,
    test_temporal_follows_relation_low_precedence,
    test_posture_temporal_contradiction,
    test_posture_supported,
    test_posture_partial,
    test_posture_weak,
    test_posture_supported_requires_all_three_thresholds,
]


def run_all():
    print(f"\n=== test_telemetry_scoring ({len(ALL_TESTS)} tests) ===")
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
