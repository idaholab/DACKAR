"""
test_past_event_scoring.py — standalone unit tests for
RuleBasedCausalityEngineV32._telemetry_score_for_past_event and
                           _evidence_score_for_past_event

Run directly:   python test_past_event_scoring.py
Or via pytest:  pytest test_past_event_scoring.py

_telemetry_score_for_past_event formula (after severity fix):
  no anomalies              → 0.20 baseline
  base = min(1.0, 0.35 + 0.12 * anomaly_count + 0.08 * severity_points)
    severity_points: high=1.0, medium=0.7, low=0.4, other=0.5
  + 0.10 if pe has matched_failure_mode_ids
  + 0.05 if pe has matched_component_ids

_evidence_score_for_past_event formula (after recency fix):
  recency = _recency_factor(pe.time_distance_days)
    None→0.75, ≤90d→1.0, ≤365d→0.85, ≤730d→0.70, >730d→0.55
  score = 0.25
  + 0.15 * recency  if matched_asset_ids
  + 0.15 * recency  if matched_component_ids
  + 0.20 * recency  if matched_failure_mode_ids
  doc_type recency: best rf per type from time_distance_days per document
    (timeless types: SOP/FMEA/MANUAL/SPEC/OE → rf=1.0)
  + 0.10 * best_rf  if CR or WO present
  + 0.10 * best_rf  if ECA or RCA present
  + 0.04 (flat)     if FMEA present
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


def make_telemetry(signals=None):
    return {"signals": signals or []}


def make_signal(sensor_id, anomalies=None):
    return {"sensor_id": sensor_id, "anomalies": anomalies or []}


def make_anomaly(severity="medium"):
    return {"pattern": "drift", "severity": severity}


def make_pe(matched_failure_mode_ids=None, matched_component_ids=None,
            matched_asset_ids=None, time_distance_days=None):
    pe = {}
    if matched_failure_mode_ids is not None:
        pe["matched_failure_mode_ids"] = matched_failure_mode_ids
    if matched_component_ids is not None:
        pe["matched_component_ids"] = matched_component_ids
    if matched_asset_ids is not None:
        pe["matched_asset_ids"] = matched_asset_ids
    if time_distance_days is not None:
        pe["time_distance_days"] = time_distance_days
    return pe


def make_doc(doc_type, time_distance_days=None):
    d = {"doc_type": doc_type}
    if time_distance_days is not None:
        d["time_distance_days"] = time_distance_days
    return d


def assert_approx(actual, expected, tol=0.005, label=""):
    assert abs(actual - expected) <= tol, (
        f"{label}: expected ~{expected}, got {actual}"
    )


# ── _telemetry_score_for_past_event tests ─────────────────────────────────

def test_telemetry_pe_no_anomalies_returns_baseline():
    """No anomalies → 0.20."""
    e = make_engine()
    assert e._telemetry_score_for_past_event(make_telemetry(), make_pe()) == 0.20
    print("  PASS test_telemetry_pe_no_anomalies_returns_baseline")


def test_telemetry_pe_one_high_anomaly():
    """
    1 high anomaly: severity_points=1.0
    base = 0.35 + 0.12*1 + 0.08*1.0 = 0.55
    """
    e = make_engine()
    ts = make_telemetry([make_signal("S1", [make_anomaly("high")])])
    assert_approx(e._telemetry_score_for_past_event(ts, make_pe()), 0.55,
                  label="1 high anomaly")
    print("  PASS test_telemetry_pe_one_high_anomaly")


def test_telemetry_pe_one_medium_anomaly():
    """
    1 medium anomaly: severity_points=0.7
    base = 0.35 + 0.12*1 + 0.08*0.7 = 0.526
    """
    e = make_engine()
    ts = make_telemetry([make_signal("S1", [make_anomaly("medium")])])
    assert_approx(e._telemetry_score_for_past_event(ts, make_pe()), 0.526,
                  label="1 medium anomaly")
    print("  PASS test_telemetry_pe_one_medium_anomaly")


def test_telemetry_pe_one_low_anomaly():
    """
    1 low anomaly: severity_points=0.4
    base = 0.35 + 0.12*1 + 0.08*0.4 = 0.502
    """
    e = make_engine()
    ts = make_telemetry([make_signal("S1", [make_anomaly("low")])])
    assert_approx(e._telemetry_score_for_past_event(ts, make_pe()), 0.502,
                  label="1 low anomaly")
    print("  PASS test_telemetry_pe_one_low_anomaly")


def test_telemetry_pe_high_severity_exceeds_medium():
    """High-severity anomaly must score higher than medium-severity anomaly."""
    e = make_engine()
    ts_high = make_telemetry([make_signal("S1", [make_anomaly("high")])])
    ts_med = make_telemetry([make_signal("S1", [make_anomaly("medium")])])
    score_high = e._telemetry_score_for_past_event(ts_high, make_pe())
    score_med = e._telemetry_score_for_past_event(ts_med, make_pe())
    assert score_high > score_med, (
        f"High ({score_high}) should exceed medium ({score_med})"
    )
    print("  PASS test_telemetry_pe_high_severity_exceeds_medium")


def test_telemetry_pe_fm_match_adds_bonus():
    """
    1 high anomaly (base=0.55) + matched_failure_mode_ids → +0.10 = 0.65
    """
    e = make_engine()
    ts = make_telemetry([make_signal("S1", [make_anomaly("high")])])
    pe = make_pe(matched_failure_mode_ids=["FM-001"])
    assert_approx(e._telemetry_score_for_past_event(ts, pe), 0.65,
                  label="FM match bonus")
    print("  PASS test_telemetry_pe_fm_match_adds_bonus")


def test_telemetry_pe_component_match_adds_bonus():
    """
    1 high anomaly (base=0.55) + matched_component_ids → +0.05 = 0.60
    """
    e = make_engine()
    ts = make_telemetry([make_signal("S1", [make_anomaly("high")])])
    pe = make_pe(matched_component_ids=["COMP-01"])
    assert_approx(e._telemetry_score_for_past_event(ts, pe), 0.60,
                  label="component match bonus")
    print("  PASS test_telemetry_pe_component_match_adds_bonus")


def test_telemetry_pe_capped_at_1():
    """Many high anomalies with FM and component match cannot exceed 1.0."""
    e = make_engine()
    ts = make_telemetry([
        make_signal("S1", [make_anomaly("high")] * 5),
        make_signal("S2", [make_anomaly("high")] * 5),
    ])
    pe = make_pe(matched_failure_mode_ids=["FM-001"], matched_component_ids=["C1"])
    score = e._telemetry_score_for_past_event(ts, pe)
    assert score <= 1.0, f"Score {score} exceeds cap of 1.0"
    assert score > 0.35
    print("  PASS test_telemetry_pe_capped_at_1")


# ── _evidence_score_for_past_event tests ─────────────────────────────────

def test_evidence_pe_baseline_no_docs_no_matches():
    """No docs, no matches → 0.25 baseline."""
    e = make_engine()
    assert_approx(e._evidence_score_for_past_event([], make_pe()), 0.25,
                  label="baseline")
    print("  PASS test_evidence_pe_baseline_no_docs_no_matches")


def test_evidence_pe_fm_match_recency_scaled():
    """
    matched_failure_mode_ids, pe.time_distance_days=None (recency=0.75)
    score = 0.25 + 0.20 * 0.75 = 0.40
    """
    e = make_engine()
    pe = make_pe(matched_failure_mode_ids=["FM-001"])
    assert_approx(e._evidence_score_for_past_event([], pe), 0.40,
                  label="FM match, unknown recency")
    print("  PASS test_evidence_pe_fm_match_recency_scaled")


def test_evidence_pe_cr_doc_recent_full_bonus():
    """
    CR doc with time_distance_days=30 (recency=1.0) → +0.10*1.0 = 0.10
    score = 0.25 + 0.10 = 0.35
    """
    e = make_engine()
    docs = [make_doc("CR", time_distance_days=30)]
    assert_approx(e._evidence_score_for_past_event(docs, make_pe()), 0.35,
                  label="recent CR")
    print("  PASS test_evidence_pe_cr_doc_recent_full_bonus")


def test_evidence_pe_cr_doc_old_reduced_bonus():
    """
    CR doc with time_distance_days=400 (recency=0.70, since 365 < 400 ≤ 730)
    → +0.10*0.70 = 0.07; score = 0.25 + 0.07 = 0.32
    """
    e = make_engine()
    docs = [make_doc("CR", time_distance_days=400)]
    assert_approx(e._evidence_score_for_past_event(docs, make_pe()), 0.32,
                  label="old CR, recency=0.70")
    print("  PASS test_evidence_pe_cr_doc_old_reduced_bonus")


def test_evidence_pe_recency_reduces_cr_bonus():
    """Old CR must score less than recent CR — recency is applied to doc bonus."""
    e = make_engine()
    recent = e._evidence_score_for_past_event([make_doc("CR", 30)], make_pe())
    old = e._evidence_score_for_past_event([make_doc("CR", 400)], make_pe())
    assert recent > old, f"Recent CR ({recent}) should exceed old CR ({old})"
    print("  PASS test_evidence_pe_recency_reduces_cr_bonus")


def test_evidence_pe_eca_doc_full_bonus():
    """
    ECA doc with time_distance_days=60 (recency=1.0) → +0.10*1.0
    score = 0.25 + 0.10 = 0.35
    """
    e = make_engine()
    docs = [make_doc("ECA", time_distance_days=60)]
    assert_approx(e._evidence_score_for_past_event(docs, make_pe()), 0.35,
                  label="recent ECA")
    print("  PASS test_evidence_pe_eca_doc_full_bonus")


def test_evidence_pe_fmea_doc_flat():
    """FMEA is timeless (rf=1.0) → flat +0.04 regardless of age."""
    e = make_engine()
    docs_young = [make_doc("FMEA", time_distance_days=10)]
    docs_old = [make_doc("FMEA", time_distance_days=2000)]
    score_young = e._evidence_score_for_past_event(docs_young, make_pe())
    score_old = e._evidence_score_for_past_event(docs_old, make_pe())
    assert_approx(score_young, 0.29, label="young FMEA")
    assert_approx(score_old, 0.29, label="old FMEA (same — timeless)")
    assert score_young == score_old, "FMEA score should not depend on document age"
    print("  PASS test_evidence_pe_fmea_doc_flat")


def test_evidence_pe_cr_unknown_age():
    """
    CR with time_distance_days=None (recency=0.75) → +0.10*0.75 = 0.075
    score = 0.25 + 0.075 = 0.325
    """
    e = make_engine()
    docs = [make_doc("CR")]  # no time_distance_days
    assert_approx(e._evidence_score_for_past_event(docs, make_pe()), 0.325,
                  label="CR unknown age, recency=0.75")
    print("  PASS test_evidence_pe_cr_unknown_age")


def test_evidence_pe_combined_all_components():
    """
    matched_asset + FM + component (pe time_distance_days=30 → recency=1.0)
    + ECA doc (30 days → rf=1.0)
    score = 0.25 + 0.15*1.0 + 0.15*1.0 + 0.20*1.0 + 0.10*1.0 = 0.85
    """
    e = make_engine()
    pe = make_pe(
        matched_asset_ids=["ASSET-001"],
        matched_component_ids=["COMP-01"],
        matched_failure_mode_ids=["FM-001"],
        time_distance_days=30,
    )
    docs = [make_doc("ECA", time_distance_days=30)]
    assert_approx(e._evidence_score_for_past_event(docs, pe), 0.85,
                  label="combined all components")
    print("  PASS test_evidence_pe_combined_all_components")


# ── _temporal_score_for_past_event tests ─────────────────────────────────
#
# Formula (when timestamps available):
#   recency_precedence: past_time >= current → 0.05
#                       delta_days ≤ 30 → 0.95; ≤ 180 → 0.80; ≤ 365 → 0.70
#                       ≤ 3650 → 0.55; else → 0.35
#   base_precedence: delta_h ≤ 72 → 0.85; ≤ 720 → 0.60; else → 0.35
#   temporal_precedence = max(recency_precedence, base_precedence, relation_score)
#   tskr_pattern_match: pattern.confidence if pattern; 0.70 if anomaly signals; else 0.0
#   latency_consistency: pattern latency if pattern; 0.60 if anomaly signals; else 0.30
#   temporal = 0.35*tskr_match + 0.30*precedence + 0.20*latency + 0.15*support
#   temporal_contradiction → subtract 0.20
#
# When current_event_time or past_time is None:
#   temporal = 0.40 (flat fallback), temporal_precedence = 0.40

from datetime import datetime, timedelta, timezone


def make_event_time(days_ahead=0):
    """Current event time (UTC now + days_ahead)."""
    return datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(days=days_ahead)


def make_past_event(days_before, event_id="PE-001"):
    """Past event with timestamp days_before days before make_event_time()."""
    ts = make_event_time() - timedelta(days=days_before)
    return {"event_id": event_id, "timestamp_start": ts.isoformat()}


def make_future_past_event(event_id="PE-001"):
    """Past event that happened AFTER current event (erroneous ordering)."""
    ts = make_event_time() + timedelta(days=5)
    return {"event_id": event_id, "timestamp_start": ts.isoformat()}


def test_temporal_pe_no_timestamps_returns_fallback():
    """
    Neither current_event_time nor past event timestamp → flat fallback.
    temporal=0.40, temporal_precedence=0.40
    """
    e = make_engine()
    pe = {"event_id": "PE-001"}   # no timestamp
    result = e._temporal_score_for_past_event(None, pe, make_telemetry(), {})
    assert_approx(result["temporal"], 0.40, label="no timestamps → temporal")
    assert_approx(result["temporal_precedence"], 0.40, label="no timestamps → precedence")
    print("  PASS test_temporal_pe_no_timestamps_returns_fallback")


def test_temporal_pe_recent_event_no_anomalies():
    """
    Past event 10 days ago, no TSKR, no anomalies.
    recency_precedence=0.95 (≤30d), base_precedence=0.60 (240h ≤ 720h)
    relation_score=0.30 (unknown, no anomalies)
    temporal_precedence = max(0.95, 0.60, 0.30) = 0.95
    tskr_match=0.0, latency=0.30, support=0.0
    temporal = 0.35*0.0 + 0.30*0.95 + 0.20*0.30 + 0.15*0.0 = 0.345
    """
    e = make_engine()
    result = e._temporal_score_for_past_event(
        make_event_time(), make_past_event(10), make_telemetry(), {}
    )
    assert_approx(result["temporal"], 0.345, tol=0.01, label="recent, no anomalies")
    assert_approx(result["temporal_precedence"], 0.95, label="recency_precedence wins")
    print("  PASS test_temporal_pe_recent_event_no_anomalies")


def test_temporal_pe_recent_event_with_anomalies():
    """
    Past event 10 days ago, anomaly signals present, no TSKR.
    tskr_match=0.70 (anomaly fallback), latency=0.60 (anomaly fallback)
    temporal_precedence=0.95
    temporal = 0.35*0.70 + 0.30*0.95 + 0.20*0.60 + 0.15*0.0 = 0.245+0.285+0.12 = 0.65
    """
    e = make_engine()
    ts = make_telemetry([make_signal("S1", [make_anomaly("high")])])
    result = e._temporal_score_for_past_event(
        make_event_time(), make_past_event(10), ts, {}
    )
    assert_approx(result["temporal"], 0.65, tol=0.01, label="recent + anomalies")
    assert result["tskr_pattern_match"] == pytest_approx(0.70, abs=0.01) if False else (
        abs(result["tskr_pattern_match"] - 0.70) <= 0.01
    )
    print("  PASS test_temporal_pe_recent_event_with_anomalies")


def test_temporal_pe_old_event_lower_precedence():
    """
    Past event 600 days ago (> 365, ≤ 3650) → recency_precedence=0.55.
    base_precedence: 600d * 24h = 14400h > 720h → 0.35
    temporal_precedence = max(0.55, 0.35, relation_score=0.30) = 0.55
    temporal = 0.30*0.55 + 0.20*0.30 = 0.165 + 0.060 = 0.225
    """
    e = make_engine()
    result = e._temporal_score_for_past_event(
        make_event_time(), make_past_event(600), make_telemetry(), {}
    )
    assert_approx(result["temporal_precedence"], 0.55, label="old event precedence")
    assert result["temporal"] < 0.40, (
        f"Old event should have low temporal score, got {result['temporal']}"
    )
    print("  PASS test_temporal_pe_old_event_lower_precedence")


def test_temporal_pe_recent_higher_than_old():
    """Recent past events must score higher than old ones (monotonicity check)."""
    e = make_engine()
    recent = e._temporal_score_for_past_event(
        make_event_time(), make_past_event(10), make_telemetry(), {}
    )["temporal"]
    old = e._temporal_score_for_past_event(
        make_event_time(), make_past_event(600), make_telemetry(), {}
    )["temporal"]
    assert recent > old, f"Recent ({recent}) should exceed old ({old})"
    print("  PASS test_temporal_pe_recent_higher_than_old")


def test_temporal_pe_future_past_event_lower_than_genuine_past():
    """
    Past event timestamp is AFTER current event time → recency_precedence=0.05.
    base_precedence may still win, but the future event must score lower than
    a genuine past event at the same time-distance (5 days).
    """
    e = make_engine()
    result_future = e._temporal_score_for_past_event(
        make_event_time(), make_future_past_event(), make_telemetry(), {}
    )
    result_past = e._temporal_score_for_past_event(
        make_event_time(), make_past_event(5), make_telemetry(), {}
    )
    assert result_future["temporal"] < result_past["temporal"], (
        f"Future past event ({result_future['temporal']}) should score below "
        f"genuine past event ({result_past['temporal']})"
    )
    print("  PASS test_temporal_pe_future_past_event_lower_than_genuine_past")


def test_temporal_pe_contradiction_applies_penalty():
    """
    temporal_contradiction=True → subtract 0.20 from temporal score.
    Use recent event with anomalies (base=0.65), then add contradiction via TSKR index.
    """
    e = make_engine()
    ts = make_telemetry([make_signal("S1", [make_anomaly("high")])])
    # Inject a contradicting TSKR pattern for the past event id
    tskr_idx = {
        "PE-001": [{
            "target_id": "PE-001",
            "confidence": 0.70,
            "relation": "precedes",
            "latency_alignment_score": 0.60,
            "support": 0.50,
            "temporal_contradiction": True,
        }]
    }
    result_no_contr = e._temporal_score_for_past_event(
        make_event_time(), make_past_event(10), ts, {}
    )
    result_contr = e._temporal_score_for_past_event(
        make_event_time(), make_past_event(10), ts, tskr_idx
    )
    assert result_contr["temporal"] < result_no_contr["temporal"], (
        "Contradiction should reduce temporal score"
    )
    assert result_contr["temporal_contradiction"] is True
    print("  PASS test_temporal_pe_contradiction_applies_penalty")


def test_temporal_pe_recency_precedence_boundary_30_days():
    """
    Exactly 30 days ago → recency_precedence=0.95 (≤30 bracket, inclusive).
    31 days ago → recency_precedence=0.80 (≤180 bracket).
    """
    e = make_engine()
    r30 = e._temporal_score_for_past_event(
        make_event_time(), make_past_event(30), make_telemetry(), {}
    )
    r31 = e._temporal_score_for_past_event(
        make_event_time(), make_past_event(31), make_telemetry(), {}
    )
    # Both recency_precedence values are reflected in temporal_precedence
    assert r30["temporal"] >= r31["temporal"], (
        "30-day event should not score below 31-day event"
    )
    print("  PASS test_temporal_pe_recency_precedence_boundary_30_days")


# ── Main runner ───────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_telemetry_pe_no_anomalies_returns_baseline,
    test_telemetry_pe_one_high_anomaly,
    test_telemetry_pe_one_medium_anomaly,
    test_telemetry_pe_one_low_anomaly,
    test_telemetry_pe_high_severity_exceeds_medium,
    test_telemetry_pe_fm_match_adds_bonus,
    test_telemetry_pe_component_match_adds_bonus,
    test_telemetry_pe_capped_at_1,
    test_evidence_pe_baseline_no_docs_no_matches,
    test_evidence_pe_fm_match_recency_scaled,
    test_evidence_pe_cr_doc_recent_full_bonus,
    test_evidence_pe_cr_doc_old_reduced_bonus,
    test_evidence_pe_recency_reduces_cr_bonus,
    test_evidence_pe_eca_doc_full_bonus,
    test_evidence_pe_fmea_doc_flat,
    test_evidence_pe_cr_unknown_age,
    test_evidence_pe_combined_all_components,
    test_temporal_pe_no_timestamps_returns_fallback,
    test_temporal_pe_recent_event_no_anomalies,
    test_temporal_pe_recent_event_with_anomalies,
    test_temporal_pe_old_event_lower_precedence,
    test_temporal_pe_recent_higher_than_old,
    test_temporal_pe_future_past_event_lower_than_genuine_past,
    test_temporal_pe_contradiction_applies_penalty,
    test_temporal_pe_recency_precedence_boundary_30_days,
]


def run_all():
    print(f"\n=== test_past_event_scoring ({len(ALL_TESTS)} tests) ===")
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
