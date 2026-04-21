"""
test_causality_scoring.py — standalone unit tests for scoring methods in
RuleBasedCausalityEngineV32:
  - _combine_scores           (weighted composite)
  - _structural_score_for_fm  (component topology lookup)
  - _evidence_score_for_fm    (doc-type contribution table)
  - _governance_details       (PM compliance signal)

Run directly:   python test_causality_scoring.py
Or via pytest:  pytest test_causality_scoring.py
"""
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from orchestrators.causality_engine_v32 import (
    RuleBasedCausalityEngineV32,
    CausalityEngineConfigV32,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_engine(weights=None):
    cfg = CausalityEngineConfigV32()
    if weights:
        cfg.weights = weights
    return RuleBasedCausalityEngineV32(config=cfg)


def make_doc(doc_type, time_distance_days=None):
    return {"doc_id": f"DOC-{doc_type}", "doc_type": doc_type, "time_distance_days": time_distance_days}


def assert_approx(actual, expected, tol=0.001, label=""):
    assert abs(actual - expected) <= tol, (
        f"{label}: expected ~{expected}, got {actual}"
    )


# ── _combine_scores ───────────────────────────────────────────────────────────

def test_combine_scores_all_ones():
    """All sub-scores=1.0 → composite=1.0 (weights sum to 1.0)."""
    e = make_engine()
    scores = {"structural": 1.0, "temporal": 1.0, "telemetry": 1.0, "evidence": 1.0, "governance": 1.0}
    assert_approx(e._combine_scores(scores), 1.0, label="composite")
    print("  PASS test_combine_scores_all_ones")


def test_combine_scores_all_zeros():
    e = make_engine()
    scores = {"structural": 0.0, "temporal": 0.0, "telemetry": 0.0, "evidence": 0.0, "governance": 0.0}
    assert_approx(e._combine_scores(scores), 0.0, label="composite")
    print("  PASS test_combine_scores_all_zeros")


def test_combine_scores_default_weights():
    """Verify default weights: structural=0.30, temporal=0.20, telemetry=0.20, evidence=0.20, governance=0.10."""
    e = make_engine()
    scores = {"structural": 1.0, "temporal": 0.0, "telemetry": 0.0, "evidence": 0.0, "governance": 0.0}
    # Only structural fires: 0.30 * 1.0 = 0.30
    assert_approx(e._combine_scores(scores), 0.30, label="structural only")
    print("  PASS test_combine_scores_default_weights")


def test_combine_scores_result_bounded_zero_to_one():
    """_combine_scores always returns a value in [0.0, 1.0]."""
    e = make_engine()
    scores = {"structural": 1.0, "temporal": 1.0, "telemetry": 1.0, "evidence": 1.0, "governance": 1.0}
    result = e._combine_scores(scores)
    assert 0.0 <= result <= 1.0
    print("  PASS test_combine_scores_result_bounded_zero_to_one")


# ── _structural_score_for_fm ─────────────────────────────────────────────────

def test_structural_score_seed_component():
    """component with seed_match_type='seed' → 0.85."""
    e = make_engine()
    components = {"COMP-A": {"component_id": "COMP-A", "seed_match_type": "seed"}}
    assert_approx(e._structural_score_for_fm("COMP-A", components), 0.85, label="seed")
    print("  PASS test_structural_score_seed_component")


def test_structural_score_telemetry_component():
    """component with seed_match_type='telemetry' → 0.90."""
    e = make_engine()
    components = {"COMP-B": {"component_id": "COMP-B", "seed_match_type": "telemetry"}}
    assert_approx(e._structural_score_for_fm("COMP-B", components), 0.90, label="telemetry")
    print("  PASS test_structural_score_telemetry_component")


def test_structural_score_known_component_no_seed_type():
    """Component in dict but no seed_match_type → 0.75."""
    e = make_engine()
    components = {"COMP-C": {"component_id": "COMP-C"}}
    assert_approx(e._structural_score_for_fm("COMP-C", components), 0.75, label="known, no seed type")
    print("  PASS test_structural_score_known_component_no_seed_type")


def test_structural_score_unknown_component():
    """Component NOT in dict → 0.40."""
    e = make_engine()
    assert_approx(e._structural_score_for_fm("COMP-UNKNOWN", {}), 0.40, label="unknown component")
    print("  PASS test_structural_score_unknown_component")


# ── _evidence_score_for_fm ────────────────────────────────────────────────────

def test_evidence_score_baseline_no_docs():
    """No documents → baseline = 0.30."""
    e = make_engine()
    assert_approx(e._evidence_score_for_fm([]), 0.30, label="baseline")
    print("  PASS test_evidence_score_baseline_no_docs")


def test_evidence_score_fmea_adds_0_12():
    """FMEA doc → 0.30 + 0.12 = 0.42."""
    e = make_engine()
    docs = [make_doc("FMEA")]
    assert_approx(e._evidence_score_for_fm(docs), 0.42, label="FMEA")
    print("  PASS test_evidence_score_fmea_adds_0_12")


def test_evidence_score_cr_adds_0_15_at_full_recency():
    """Recent CR (time_distance_days=0) → recency=1.0 → +0.15 → 0.45."""
    e = make_engine()
    docs = [make_doc("CR", time_distance_days=0)]
    assert_approx(e._evidence_score_for_fm(docs), 0.45, label="CR recent")
    print("  PASS test_evidence_score_cr_adds_0_15_at_full_recency")


def test_evidence_score_eca_adds_0_22_at_full_recency():
    """Recent ECA (time_distance_days=0) → +0.22 → 0.52."""
    e = make_engine()
    docs = [make_doc("ECA", time_distance_days=0)]
    assert_approx(e._evidence_score_for_fm(docs), 0.52, label="ECA recent")
    print("  PASS test_evidence_score_eca_adds_0_22_at_full_recency")


def test_evidence_score_sop_adds_0_08():
    """SOP (timeless) → +0.08 → 0.38."""
    e = make_engine()
    docs = [make_doc("SOP")]
    assert_approx(e._evidence_score_for_fm(docs), 0.38, label="SOP")
    print("  PASS test_evidence_score_sop_adds_0_08")


def test_evidence_score_capped_at_1():
    """Multiple high-value docs cannot exceed 1.0."""
    e = make_engine()
    docs = [make_doc("FMEA"), make_doc("ECA", 0), make_doc("CR", 0), make_doc("SOP"), make_doc("OE")]
    score = e._evidence_score_for_fm(docs)
    assert score <= 1.0
    assert score > 0.70  # should be well above baseline
    print("  PASS test_evidence_score_capped_at_1")


# ── _governance_details ───────────────────────────────────────────────────────

def test_governance_no_pm_data_returns_neutral():
    e = make_engine()
    result = e._governance_details(pm_compliance=None)
    assert_approx(result["score"], 0.5, label="no pm data")
    assert result["pm_data_available"] is False
    print("  PASS test_governance_no_pm_data_returns_neutral")


def test_governance_all_checks_pass_returns_neutral():
    e = make_engine()
    pm = {"checks": [{"status": "pass", "check_type": "inspection"}]}
    result = e._governance_details(pm_compliance=pm, fm_name="corrosion")
    assert_approx(result["score"], 0.5, label="all pass")
    print("  PASS test_governance_all_checks_pass_returns_neutral")


def test_governance_relevant_failure_raises_score():
    """Failed inspection check relevant to 'corrosion' → score > 0.5."""
    e = make_engine()
    pm = {"checks": [
        {"status": "fail", "check_type": "inspection", "overdue_by_days": 10},
    ]}
    result = e._governance_details(pm_compliance=pm, fm_name="corrosion")
    assert result["score"] > 0.5, f"Expected score>0.5, got {result['score']}"
    assert len(result["relevant_failed_checks"]) == 1
    print("  PASS test_governance_relevant_failure_raises_score")


def test_governance_overdue_above_30_adds_extra_boost():
    """Overdue > 30 days → overdue_boost = 0.05."""
    e = make_engine()
    pm = {"checks": [
        {"status": "fail", "check_type": "inspection", "overdue_by_days": 60},
    ]}
    # "leakage" is in the inspection keyword set and is a concrete, testable trigger.
    # ("fouling" was removed from inspection keywords — it caused false positives
    # when expansion joint inspection was matched to tube-fouling candidates.)
    result = e._governance_details(pm_compliance=pm, fm_name="air in-leakage boundary")
    assert_approx(result["overdue_boost"], 0.05, label="overdue_boost")
    print("  PASS test_governance_overdue_above_30_adds_extra_boost")


def test_governance_irrelevant_failure_returns_neutral():
    """Failed check for a keyword that doesn't match candidate → score stays 0.5."""
    e = make_engine()
    pm = {"checks": [
        {"status": "fail", "check_type": "calibration", "overdue_by_days": 30},
    ]}
    # 'calibration' keyword set: {calibrat, instrument, sensor, drift, measurement, transmitter, indication}
    # 'air_inleakage' has none of these
    result = e._governance_details(pm_compliance=pm, fm_name="air inleakage")
    assert_approx(result["score"], 0.5, label="irrelevant failure")
    print("  PASS test_governance_irrelevant_failure_returns_neutral")


def test_telemetry_anomaly_precedes_structural_score():
    """Sprint 2 fix: seed_match_type 'telemetry_anomaly_precedes' → structural score 0.80."""
    e = make_engine()
    components = {
        "COMP-001": {"seed_match_type": "telemetry_anomaly_precedes"},
        "COMP-002": {"seed_match_type": "seed"},
        "COMP-003": {"seed_match_type": "telemetry"},
        "COMP-004": {},           # no seed_match_type → default neighbor
        "COMP-005": None,         # guard against None
    }
    assert_approx(e._structural_score_for_fm("COMP-001", components), 0.80, label="telemetry_anomaly_precedes")
    assert_approx(e._structural_score_for_fm("COMP-002", components), 0.85, label="seed")
    assert_approx(e._structural_score_for_fm("COMP-003", components), 0.90, label="telemetry")
    assert_approx(e._structural_score_for_fm("COMP-004", components), 0.75, label="default neighbor")
    assert_approx(e._structural_score_for_fm("UNKNOWN", components),  0.40, label="unknown component")
    print("  PASS test_telemetry_anomaly_precedes_structural_score")


def test_weight_sum_constraint_raises_on_misconfiguration():
    """Sprint 2 fix: CausalityEngineConfigV32 raises ValueError when weights don't sum to 1.0."""
    import pytest
    bad_weights = {"structural": 0.40, "temporal": 0.20, "telemetry": 0.20,
                   "evidence": 0.20, "governance": 0.10}  # sum = 1.10
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        CausalityEngineConfigV32(weights=bad_weights)
    print("  PASS test_weight_sum_constraint_raises_on_misconfiguration")


def test_authority_weights_constant_present():
    """Sprint 2 fix: _AUTHORITY_WEIGHTS class constant exists with all 6 tiers defined."""
    e = make_engine()
    expected_tiers = {"plant_instance", "plant_procedure", "plant_fmea",
                      "plant_family", "oe_iris", "oe_adams"}
    assert expected_tiers == set(e._AUTHORITY_WEIGHTS.keys()), (
        f"Missing tiers: {expected_tiers - set(e._AUTHORITY_WEIGHTS.keys())}"
    )
    assert e._AUTHORITY_WEIGHTS["plant_instance"] == 1.00
    assert e._AUTHORITY_WEIGHTS["oe_adams"] == 0.30
    assert e._AUTHORITY_WEIGHTS["plant_instance"] > e._AUTHORITY_WEIGHTS["oe_iris"]
    print("  PASS test_authority_weights_constant_present")


def test_pre_evidence_threshold_defaults():
    """Sprint 1 fix: minimum_pre_evidence_threshold=0.10 < minimum_evidence_threshold=0.35."""
    cfg = CausalityEngineConfigV32()
    assert cfg.minimum_pre_evidence_threshold == 0.10
    assert cfg.minimum_evidence_threshold == 0.35
    print("  PASS test_pre_evidence_threshold_defaults")


def test_pre_evidence_threshold_allows_sparse_proxy():
    """Sprint 1 fix: candidate with proxy evidence 0.15 (>0.10, <0.35) passes Stage D filter.

    Before the fix, meets_evidence_threshold used minimum_evidence_threshold (0.35), so
    a candidate with proxy score 0.15 would be filtered at Stage D, never reaching evidence retrieval.
    After the fix, Stage D uses minimum_pre_evidence_threshold (0.10).
    """
    e = make_engine()
    # Simulate a candidate whose meets_evidence_threshold was set with the new 0.10 threshold.
    # proxy evidence 0.15: 0.15 >= 0.10 → True (passes); 0.15 >= 0.35 → False (old behavior would block)
    proxy_evidence = 0.15
    meets_new = proxy_evidence >= e.config.minimum_pre_evidence_threshold
    meets_old = proxy_evidence >= e.config.minimum_evidence_threshold
    assert meets_new is True, "proxy 0.15 should pass minimum_pre_evidence_threshold (0.10)"
    assert meets_old is False, "proxy 0.15 should NOT pass minimum_evidence_threshold (0.35)"

    candidate = {"composite_score": 0.45, "meets_evidence_threshold": meets_new}
    assert e._candidate_meets_threshold(candidate) is True, (
        "Stage D filter should pass candidate with sparse proxy evidence"
    )
    print("  PASS test_pre_evidence_threshold_allows_sparse_proxy")


# ── Main runner ───────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_combine_scores_all_ones,
    test_combine_scores_all_zeros,
    test_combine_scores_default_weights,
    test_combine_scores_result_bounded_zero_to_one,
    test_structural_score_seed_component,
    test_structural_score_telemetry_component,
    test_structural_score_known_component_no_seed_type,
    test_structural_score_unknown_component,
    test_evidence_score_baseline_no_docs,
    test_evidence_score_fmea_adds_0_12,
    test_evidence_score_cr_adds_0_15_at_full_recency,
    test_evidence_score_eca_adds_0_22_at_full_recency,
    test_evidence_score_sop_adds_0_08,
    test_evidence_score_capped_at_1,
    test_governance_no_pm_data_returns_neutral,
    test_governance_all_checks_pass_returns_neutral,
    test_governance_relevant_failure_raises_score,
    test_governance_overdue_above_30_adds_extra_boost,
    test_governance_irrelevant_failure_returns_neutral,
    test_pre_evidence_threshold_defaults,
    test_pre_evidence_threshold_allows_sparse_proxy,
    test_telemetry_anomaly_precedes_structural_score,
    test_weight_sum_constraint_raises_on_misconfiguration,
    test_authority_weights_constant_present,
]


def run_all():
    print(f"\n=== test_causality_scoring ({len(ALL_TESTS)} tests) ===")
    passed, failed = 0, 0
    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"  FAIL {fn.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
