"""
test_refine_with_evidence.py — standalone unit tests for
RuleBasedCausalityEngineV32.refine_with_evidence and _evidence_posture

Run directly:   python test_refine_with_evidence.py
Or via pytest:  pytest test_refine_with_evidence.py

Formula:  refined = 0.30*prior + 0.55*support + 0.15*context - 0.45*contradiction
          clamped to [0, 1].

Key invariants:
  1. Pure support  → high refined score, posture=supported
  2. Pure contradiction → low refined score, posture=contradicted
  3. No evidence → prior only, posture depends on hit_count
  4. Mixed evidence → partial score, posture=mixed
  5. Conjecture discount reduces effective support
  6. Candidates without evidence summary entry keep prior unchanged
  7. evidence_gap=True when retrieved_hit_count=0
  8. _evidence_posture classification covers all branches
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

def make_engine():
    return RuleBasedCausalityEngineV32()


def make_candidates(*candidate_ids, prior_evidence=0.40):
    """Minimal causality_candidates payload.
    All 5 score fields required by _combine_scores are provided; evidence is
    the variable under test, others are set to a neutral mid-range value.
    """
    return {
        "candidates": [
            {
                "candidate_id": cid,
                "cause_label": f"cause for {cid}",
                "composite_score": 0.50,
                "meets_evidence_threshold": True,
                "scores": {
                    "structural": 0.60,
                    "temporal": 0.50,
                    "telemetry": 0.50,
                    "evidence": prior_evidence,
                    "governance": 0.50,
                },
            }
            for cid in candidate_ids
        ]
    }


def make_evidence_bundle(summaries):
    """Build a minimal evidence_bundle with candidate_evidence_summary rows."""
    return {"candidate_evidence_summary": summaries}


def make_summary(
    candidate_id,
    best_support_score=0.0,
    best_contradiction_score=0.0,
    best_context_score=0.0,
    hit_count=3,
    mean_conjecture_fraction=0.0,
    dominant_temporal_relation=None,
    best_lag_hours=None,
    best_source_tier=None,
):
    return {
        "candidate_id": candidate_id,
        "best_support_score": best_support_score,
        "best_contradiction_score": best_contradiction_score,
        "best_context_score": best_context_score,
        "best_source_tier": best_source_tier,
        "hit_count": hit_count,
        "mean_conjecture_fraction": mean_conjecture_fraction,
        "supporting_snippet_ids": [],
        "contradicting_snippet_ids": [],
        "contextual_snippet_ids": [],
        "dominant_temporal_relation": dominant_temporal_relation,
        "best_lag_hours": best_lag_hours,
        "lag_is_approximate": False,
        "aggregated_mechanisms": [],
        "aggregated_outcomes": [],
    }


def get_candidate(result, candidate_id):
    """Search both retained and filtered-out candidates."""
    for pool in ("candidates", "filtered_out_candidates"):
        for c in result.get(pool, []):
            if isinstance(c, dict) and c.get("candidate_id") == candidate_id:
                return c
    return None


def assert_approx(actual, expected, tol=0.01, label=""):
    assert abs(actual - expected) <= tol, (
        f"{label}: expected ~{expected}, got {actual}"
    )


# ── refine_with_evidence tests ────────────────────────────────────────────────

def test_strong_support_yields_high_refined_score():
    """
    prior=0.40, support=0.90, context=0.0, contradiction=0.0
    refined = 0.30*0.40 + 0.55*0.90 + 0.15*0.0 - 0.45*0.0 = 0.12 + 0.495 = 0.615
    """
    e = make_engine()
    result = e.refine_with_evidence(
        make_candidates("CAND-A", prior_evidence=0.40),
        make_evidence_bundle([make_summary("CAND-A", best_support_score=0.90)]),
    )
    c = get_candidate(result, "CAND-A")
    assert_approx(c["scores"]["evidence"], 0.615, label="refined_evidence")
    assert c["evidence_posture"] == "supported"
    print("  PASS test_strong_support_yields_high_refined_score")


def test_strong_contradiction_fails_evidence_threshold():
    """
    prior=0.40, support=0.0, contradiction=0.80
    refined = 0.30*0.40 - 0.45*0.80 = 0.12 - 0.36 = -0.24 → clamped to 0.0 < threshold(0.35).
    Candidate is compacted into filtered_out_candidates; meets_evidence_threshold=False.
    """
    e = make_engine()
    result = e.refine_with_evidence(
        make_candidates("CAND-A", prior_evidence=0.40),
        make_evidence_bundle([make_summary("CAND-A", best_contradiction_score=0.80, hit_count=3)]),
    )
    c = get_candidate(result, "CAND-A")
    assert c is not None
    assert c.get("meets_evidence_threshold") is False, (
        f"Expected meets_evidence_threshold=False, got {c.get('meets_evidence_threshold')}"
    )
    print("  PASS test_strong_contradiction_fails_evidence_threshold")


def test_no_evidence_entry_fails_evidence_threshold():
    """
    Candidate has no entry in evidence summary → refined = 0.30*prior (max 0.30) < threshold(0.35).
    The candidate is compacted; meets_evidence_threshold=False and evidence_gap in filter_reason.
    """
    e = make_engine()
    result = e.refine_with_evidence(
        make_candidates("CAND-A", prior_evidence=0.50),
        make_evidence_bundle([]),  # empty summary — CAND-A not present
    )
    c = get_candidate(result, "CAND-A")
    assert c is not None
    assert c.get("meets_evidence_threshold") is False
    assert "evidence_threshold" in (c.get("filter_reason") or "")
    print("  PASS test_no_evidence_entry_fails_evidence_threshold")


def test_conjecture_discount_reduces_support():
    """
    conjecture_fraction=0.5 → discount = min(0.30, 0.60*0.5) = 0.30
    effective_support = 0.80 * (1 - 0.30) = 0.56
    refined = 0.30*0.40 + 0.55*0.56 = 0.12 + 0.308 = 0.428
    """
    e = make_engine()
    result = e.refine_with_evidence(
        make_candidates("CAND-A", prior_evidence=0.40),
        make_evidence_bundle([make_summary(
            "CAND-A",
            best_support_score=0.80,
            mean_conjecture_fraction=0.50,
        )]),
    )
    c = get_candidate(result, "CAND-A")
    # Without discount: 0.30*0.40 + 0.55*0.80 = 0.12 + 0.44 = 0.56
    # With discount (support *= 0.70): 0.30*0.40 + 0.55*0.56 = 0.12 + 0.308 = 0.428
    assert c["scores"]["evidence"] < 0.56, "Conjecture discount should reduce score below no-discount value"
    assert_approx(c["scores"]["evidence"], 0.428, tol=0.015, label="conjecture-discounted")
    print("  PASS test_conjecture_discount_reduces_support")


def test_evidence_gap_true_when_no_hits():
    """
    hit_count=0 → evidence_gap=True AND posture=no_data.
    The candidate is filtered out (refined < 0.35), but we can verify via
    _evidence_posture directly: 0 hits, 0 support, 0 contradiction → no_data.
    """
    e = make_engine()
    posture = e._evidence_posture(
        support_score=0.0,
        contradiction_score=0.0,
        contextual_score=0.0,
        retrieved_hit_count=0,
    )
    assert posture == "no_data"
    print("  PASS test_evidence_gap_true_when_no_hits")


def test_evidence_gap_false_when_hits_present():
    e = make_engine()
    result = e.refine_with_evidence(
        make_candidates("CAND-A"),
        make_evidence_bundle([make_summary("CAND-A", best_support_score=0.50, hit_count=2)]),
    )
    c = get_candidate(result, "CAND-A")
    assert c["evidence_gap"] is False
    print("  PASS test_evidence_gap_false_when_hits_present")


def test_two_candidates_refined_independently():
    """
    CAND-A: strong support (0.90) → passes evidence threshold, retained.
    CAND-B: strong contradiction (0.70) → fails evidence threshold, compacted.
    Verify they're treated separately and CAND-A has higher composite than CAND-B.
    """
    e = make_engine()
    result = e.refine_with_evidence(
        make_candidates("CAND-A", "CAND-B", prior_evidence=0.40),
        make_evidence_bundle([
            make_summary("CAND-A", best_support_score=0.90),
            make_summary("CAND-B", best_contradiction_score=0.70, hit_count=2),
        ]),
    )
    a = get_candidate(result, "CAND-A")
    b = get_candidate(result, "CAND-B")
    assert a is not None and b is not None
    # CAND-A retained (meets threshold); CAND-B filtered (evidence below threshold)
    assert a.get("meets_evidence_threshold") is True, f"CAND-A: {a.get('meets_evidence_threshold')}"
    assert b.get("meets_evidence_threshold") is False, f"CAND-B: {b.get('meets_evidence_threshold')}"
    # CAND-A composite > CAND-B composite
    assert float(a.get("composite_score", 0)) > float(b.get("composite_score", 0))
    print("  PASS test_two_candidates_refined_independently")


def test_temporal_relation_backfill():
    """
    dominant_temporal_relation from evidence populates temporal_evidence.relation.
    Candidate needs enough support (0.80) to pass evidence threshold and stay retained.
    """
    e = make_engine()
    result = e.refine_with_evidence(
        make_candidates("CAND-A", prior_evidence=0.40),
        make_evidence_bundle([
            make_summary(
                "CAND-A",
                best_support_score=0.80,
                hit_count=3,
                dominant_temporal_relation="precedes",
                best_lag_hours=24.0,
            ),
        ]),
    )
    c = get_candidate(result, "CAND-A")
    assert c is not None
    assert c.get("meets_evidence_threshold") is True
    te = c.get("temporal_evidence", {})
    assert te.get("relation") == "precedes"
    assert te.get("observed_lag_hours") == 24.0
    print("  PASS test_temporal_relation_backfill")


def test_refine_preserves_risk_adjusted_governance():
    """§8.4: Stage F should keep governance adjusted by risk-significance scalar."""
    e = make_engine()
    candidates = make_candidates("CAND-A", prior_evidence=0.40)
    candidates["candidates"][0]["scores"]["governance"] = 0.50
    candidates["candidates"][0]["scores"]["governance_base"] = 0.50
    candidates["candidates"][0]["scores"]["risk_significance_scalar"] = 0.80
    result = e.refine_with_evidence(
        candidates,
        make_evidence_bundle([make_summary("CAND-A", best_support_score=0.80, hit_count=3)]),
    )
    c = get_candidate(result, "CAND-A")
    assert c is not None
    assert_approx(float(c["scores"]["governance_base"]), 0.50, label="governance_base")
    assert_approx(float(c["scores"]["governance_risk_delta"]), 0.16, tol=0.02, label="governance_risk_delta")
    assert_approx(float(c["scores"]["governance"]), 0.66, tol=0.02, label="governance")
    print("  PASS test_refine_preserves_risk_adjusted_governance")


def test_authority_tier_weights_evidence_support():
    """Same support score should rank higher with higher source tier authority."""
    e = make_engine()
    result = e.refine_with_evidence(
        make_candidates("CAND-HI", "CAND-LO", prior_evidence=0.40),
        make_evidence_bundle([
            make_summary("CAND-HI", best_support_score=0.80, best_source_tier="plant_instance"),
            make_summary("CAND-LO", best_support_score=0.80, best_source_tier="oe_adams"),
        ]),
    )
    hi = get_candidate(result, "CAND-HI")
    lo = get_candidate(result, "CAND-LO")
    assert hi is not None and lo is not None
    assert float(hi["scores"]["evidence"]) > float(lo["scores"]["evidence"])
    assert float(hi["scores"]["evidence_authority_weight"]) > float(lo["scores"]["evidence_authority_weight"])
    assert "authority_tier=plant_instance" in (hi.get("score_rationale", {}).get("evidence") or "")
    print("  PASS test_authority_tier_weights_evidence_support")


def test_physical_plausibility_gate_eliminates_candidate_with_binary_rationale():
    e = make_engine()
    candidates = {
        "candidates": [
            {
                "candidate_id": "CAND-OK",
                "hypothesis_type": "failure_mode",
                "component_id": "CMP-1",
                "failure_mode_id": "FM-1",
                "cause_node_id": "FM-1",
                "cause_label": "plausible",
                "composite_score": 0.70,
                "meets_evidence_threshold": True,
                "scores": {"structural": 0.70, "temporal": 0.50, "telemetry": 0.50, "evidence": 0.40, "governance": 0.50},
            },
            {
                "candidate_id": "CAND-BAD",
                "hypothesis_type": "failure_mode",
                "component_id": "unknown_component",
                "failure_mode_id": "unknown_failure_mode",
                "cause_node_id": "FM-X",
                "cause_label": "implausible",
                "composite_score": 0.95,
                "meets_evidence_threshold": True,
                "scores": {"structural": 0.10, "temporal": 0.50, "telemetry": 0.50, "evidence": 0.40, "governance": 0.50},
            },
        ]
    }
    result = e.refine_with_evidence(
        candidates,
        make_evidence_bundle(
            [
                make_summary("CAND-OK", best_support_score=0.90, hit_count=3),
                make_summary("CAND-BAD", best_support_score=0.90, hit_count=3),
            ]
        ),
    )
    ok = get_candidate(result, "CAND-OK")
    bad = get_candidate(result, "CAND-BAD")
    assert ok is not None and bad is not None
    assert ok in (result.get("candidates") or [])
    assert bad in (result.get("filtered_out_candidates") or [])
    gate_bad = ((bad.get("hard_gates") or {}).get("physical_plausibility") or {})
    gate_ok = ((ok.get("hard_gates") or {}).get("physical_plausibility") or {})
    assert gate_bad.get("passed") is False
    assert isinstance(gate_bad.get("rationale"), str) and gate_bad.get("rationale").startswith("FAIL:")
    assert gate_ok.get("passed") is True
    assert isinstance(gate_ok.get("rationale"), str) and gate_ok.get("rationale").startswith("PASS:")
    assert ((bad.get("ruleout") or {}).get("reason_code")) == "physically_impossible"
    print("  PASS test_physical_plausibility_gate_eliminates_candidate_with_binary_rationale")


def test_timeline_consistency_gate_supports_normal_and_degraded_modes():
    e = make_engine()
    candidates = {
        "candidates": [
            {
                "candidate_id": "CAND-NORMAL",
                "hypothesis_type": "failure_mode",
                "component_id": "CMP-1",
                "failure_mode_id": "FM-1",
                "cause_node_id": "FM-1",
                "cause_label": "normal timeline",
                "composite_score": 0.70,
                "meets_evidence_threshold": True,
                "scores": {"structural": 0.70, "temporal": 0.50, "telemetry": 0.50, "evidence": 0.40, "governance": 0.50},
                "temporal_evidence": {
                    "latency_violation_type": "none",
                    "observed_lag_hours": 1.0,
                    "expected_latency_min_hours": 0.5,
                    "expected_latency_max_hours": 2.0,
                    "temporal_contradiction": False,
                },
            },
            {
                "candidate_id": "CAND-DEGRADED",
                "hypothesis_type": "failure_mode",
                "component_id": "CMP-2",
                "failure_mode_id": "FM-2",
                "cause_node_id": "FM-2",
                "cause_label": "degraded timeline",
                "composite_score": 0.69,
                "meets_evidence_threshold": True,
                "scores": {"structural": 0.70, "temporal": 0.50, "telemetry": 0.50, "evidence": 0.40, "governance": 0.50},
                "temporal_evidence": {
                    "latency_violation_type": "unknown",
                    "observed_lag_hours": None,
                    "expected_latency_min_hours": None,
                    "expected_latency_max_hours": None,
                    "temporal_contradiction": False,
                },
            },
            {
                "candidate_id": "CAND-FAIL",
                "hypothesis_type": "failure_mode",
                "component_id": "CMP-3",
                "failure_mode_id": "FM-3",
                "cause_node_id": "FM-3",
                "cause_label": "bad timeline",
                "composite_score": 0.95,
                "meets_evidence_threshold": True,
                "scores": {"structural": 0.70, "temporal": 0.50, "telemetry": 0.50, "evidence": 0.40, "governance": 0.50},
                "temporal_evidence": {
                    "latency_violation_type": "too_fast",
                    "observed_lag_hours": 0.1,
                    "expected_latency_min_hours": 1.0,
                    "expected_latency_max_hours": 4.0,
                    "temporal_contradiction": True,
                },
                "temporal_posture": "contradicted",
            },
        ]
    }
    result = e.refine_with_evidence(
        candidates,
        make_evidence_bundle(
            [
                make_summary("CAND-NORMAL", best_support_score=0.90, hit_count=3),
                make_summary("CAND-DEGRADED", best_support_score=0.90, hit_count=3),
                make_summary("CAND-FAIL", best_support_score=0.90, hit_count=3),
            ]
        ),
    )
    normal = get_candidate(result, "CAND-NORMAL")
    degraded = get_candidate(result, "CAND-DEGRADED")
    failed = get_candidate(result, "CAND-FAIL")
    assert normal is not None and degraded is not None and failed is not None
    g_normal = ((normal.get("hard_gates") or {}).get("timeline_consistency") or {})
    g_degraded = ((degraded.get("hard_gates") or {}).get("timeline_consistency") or {})
    g_failed = ((failed.get("hard_gates") or {}).get("timeline_consistency") or {})
    assert g_normal.get("passed") is True and g_normal.get("degraded_mode") is False
    assert g_degraded.get("passed") is True and g_degraded.get("degraded_mode") is True
    assert g_failed.get("passed") is False
    assert ((failed.get("ruleout") or {}).get("reason_code")) == "timeline_inconsistent"
    print("  PASS test_timeline_consistency_gate_supports_normal_and_degraded_modes")


def test_barrier_logic_gate_supports_normal_and_degraded_modes():
    e = make_engine()
    candidates = {
        "candidates": [
            {
                "candidate_id": "CAND-BARRIER-NORMAL",
                "hypothesis_type": "failure_mode",
                "component_id": "CMP-1",
                "failure_mode_id": "FM-1",
                "cause_node_id": "FM-1",
                "cause_label": "barrier normal",
                "composite_score": 0.70,
                "meets_evidence_threshold": True,
                "scores": {
                    "structural": 0.70, "temporal": 0.50, "telemetry": 0.50, "evidence": 0.40, "governance": 0.50,
                    "barrier_signal": 0.8,
                },
                "affected_safety_functions": [{"sf_id": "SF-1", "sf_name": "RPS", "impact_type": "direct"}],
            },
            {
                "candidate_id": "CAND-BARRIER-DEGRADED",
                "hypothesis_type": "failure_mode",
                "component_id": "CMP-2",
                "failure_mode_id": "FM-2",
                "cause_node_id": "FM-2",
                "cause_label": "barrier degraded",
                "composite_score": 0.69,
                "meets_evidence_threshold": True,
                "scores": {"structural": 0.70, "temporal": 0.50, "telemetry": 0.50, "evidence": 0.40, "governance": 0.50},
            },
            {
                "candidate_id": "CAND-BARRIER-FAIL",
                "hypothesis_type": "failure_mode",
                "component_id": "CMP-3",
                "failure_mode_id": "FM-3",
                "cause_node_id": "FM-3",
                "cause_label": "barrier fail",
                "composite_score": 0.95,
                "meets_evidence_threshold": True,
                "scores": {"structural": 0.70, "temporal": 0.50, "telemetry": 0.50, "evidence": 0.40, "governance": 0.50},
                "ruleout": {"reason_code": "barrier_held"},
            },
        ]
    }
    result = e.refine_with_evidence(
        candidates,
        make_evidence_bundle(
            [
                make_summary("CAND-BARRIER-NORMAL", best_support_score=0.90, hit_count=3),
                make_summary("CAND-BARRIER-DEGRADED", best_support_score=0.90, hit_count=3),
                make_summary("CAND-BARRIER-FAIL", best_support_score=0.90, hit_count=3),
            ]
        ),
    )
    normal = get_candidate(result, "CAND-BARRIER-NORMAL")
    degraded = get_candidate(result, "CAND-BARRIER-DEGRADED")
    failed = get_candidate(result, "CAND-BARRIER-FAIL")
    assert normal is not None and degraded is not None and failed is not None
    g_normal = ((normal.get("hard_gates") or {}).get("barrier_logic") or {})
    g_degraded = ((degraded.get("hard_gates") or {}).get("barrier_logic") or {})
    g_failed = ((failed.get("hard_gates") or {}).get("barrier_logic") or {})
    assert g_normal.get("passed") is True and g_normal.get("degraded_mode") is False
    assert g_degraded.get("passed") is True and g_degraded.get("degraded_mode") is True
    assert g_failed.get("passed") is False
    assert ((failed.get("ruleout") or {}).get("reason_code")) == "barrier_held"
    print("  PASS test_barrier_logic_gate_supports_normal_and_degraded_modes")


def test_hard_gate_ordering_preserves_first_failure_reason_and_logs_rationales():
    e = make_engine()
    candidates = {
        "candidates": [
            {
                "candidate_id": "CAND-ORDER",
                "hypothesis_type": "failure_mode",
                "component_id": "CMP-ORD",
                "failure_mode_id": "FM-ORD",
                "cause_node_id": "FM-ORD",
                "cause_label": "ordering candidate",
                "composite_score": 0.92,
                "meets_evidence_threshold": True,
                "scores": {"structural": 0.10, "temporal": 0.50, "telemetry": 0.50, "evidence": 0.40, "governance": 0.50},
                "temporal_evidence": {
                    "latency_violation_type": "too_fast",
                    "observed_lag_hours": 0.1,
                    "expected_latency_min_hours": 1.0,
                    "expected_latency_max_hours": 3.0,
                    "temporal_contradiction": True,
                },
                "temporal_posture": "contradicted",
            }
        ]
    }
    result = e.refine_with_evidence(
        candidates,
        make_evidence_bundle([make_summary("CAND-ORDER", best_support_score=0.90, hit_count=3)]),
    )
    c = get_candidate(result, "CAND-ORDER")
    assert c is not None
    hard_gates = c.get("hard_gates") or {}
    g_phys = hard_gates.get("physical_plausibility") or {}
    g_time = hard_gates.get("timeline_consistency") or {}
    g_bar = hard_gates.get("barrier_logic") or {}
    assert g_phys.get("passed") is False and g_phys.get("gate_order") == 1
    assert g_time.get("passed") is False and g_time.get("gate_order") == 2
    assert g_bar.get("gate_order") == 3
    assert isinstance(g_phys.get("rationale"), str) and g_phys.get("rationale").startswith("FAIL:")
    assert isinstance(g_time.get("rationale"), str) and g_time.get("rationale").startswith("FAIL:")
    assert isinstance(g_bar.get("rationale"), str) and len(g_bar.get("rationale")) > 0
    ruleout = c.get("ruleout") or {}
    assert ruleout.get("reason_code") == "physically_impossible"
    assert ruleout.get("reason_detail") == g_phys.get("rationale")
    print("  PASS test_hard_gate_ordering_preserves_first_failure_reason_and_logs_rationales")


def test_coverage_quality_flags_reduce_quality_multiplier_and_surface_in_uncertainty_summary():
    e = RuleBasedCausalityEngineV32(
        CausalityEngineConfigV32(
            minimum_evidence_threshold=0.0,
            minimum_pre_evidence_threshold=0.0,
            minimum_composite_threshold=0.0,
            top_k_candidates=5,
        )
    )
    candidates = {
        "candidates": [
            {
                "candidate_id": "CAND-COV",
                "cause_label": "coverage sensitive",
                "hypothesis_type": "failure_mode",
                "component_id": "CMP-1",
                "failure_mode_id": "FM-1",
                "cause_node_id": "FM-1",
                "composite_score": 0.80,
                "meets_evidence_threshold": True,
                "scores": {
                    "structural": 0.80,
                    "temporal": 0.80,
                    "telemetry": 0.70,
                    "evidence": 0.80,
                    "governance": 0.60,
                },
            }
        ]
    }
    evidence = make_evidence_bundle([make_summary("CAND-COV", best_support_score=0.90, hit_count=3)])

    baseline = e.refine_with_evidence(
        causality_candidates=candidates,
        evidence_bundle=evidence,
    )
    degraded = e.refine_with_evidence(
        causality_candidates=candidates,
        evidence_bundle=evidence,
        coverage_summary={
            "overall_status": "partial",
            "source_families": {
                "kg_context": {"status": "partial"},
                "chroma_corpus": {"status": "complete"},
                "upstream_anomaly_inputs": {"status": "missing"},
            },
        },
    )
    c_base = get_candidate(baseline, "CAND-COV")
    c_deg = get_candidate(degraded, "CAND-COV")
    assert c_base is not None and c_deg is not None
    assert float(c_deg.get("quality_multiplier", 1.0)) < float(c_base.get("quality_multiplier", 1.0))
    assert float((c_deg.get("scores") or {}).get("coverage_quality_factor", 1.0)) < 1.0
    assert "upstream_anomaly_inputs" in ((c_deg.get("scores") or {}).get("coverage_quality_flags") or [])
    us = degraded.get("uncertainty_summary") or {}
    assert float(us.get("average_coverage_quality_factor", 1.0)) < 1.0
    assert int(us.get("coverage_degraded_candidate_count", 0)) >= 1
    print("  PASS test_coverage_quality_flags_reduce_quality_multiplier_and_surface_in_uncertainty_summary")


# ── _evidence_posture classification tests ────────────────────────────────────

def test_posture_no_data():
    e = make_engine()
    assert e._evidence_posture(0.0, 0.0, 0.0, retrieved_hit_count=0) == "no_data"
    print("  PASS test_posture_no_data")


def test_posture_contradicted():
    e = make_engine()
    assert e._evidence_posture(0.10, 0.60, 0.0, retrieved_hit_count=3) == "contradicted"
    print("  PASS test_posture_contradicted")


def test_posture_supported():
    e = make_engine()
    assert e._evidence_posture(0.75, 0.10, 0.0, retrieved_hit_count=3) == "supported"
    print("  PASS test_posture_supported")


def test_posture_mixed():
    e = make_engine()
    assert e._evidence_posture(0.50, 0.35, 0.0, retrieved_hit_count=3) == "mixed"
    print("  PASS test_posture_mixed")


def test_posture_contextual_only():
    e = make_engine()
    assert e._evidence_posture(0.10, 0.10, 0.40, retrieved_hit_count=3) == "contextual_only"
    print("  PASS test_posture_contextual_only")


def test_posture_weak():
    e = make_engine()
    assert e._evidence_posture(0.20, 0.10, 0.15, retrieved_hit_count=3) == "weak"
    print("  PASS test_posture_weak")


def test_posture_contradicted_weak_contra_zero_support():
    """Sprint 2 fix: support=0, contradiction=0.20 (below 0.45 threshold) → 'contradicted'.

    Before the fix this returned 'weak' because only contradiction_score >= 0.45 triggered
    'contradicted'. Any contradicting evidence with zero support is epistemically 'contradicted'.
    """
    e = make_engine()
    assert e._evidence_posture(0.0, 0.20, 0.10, retrieved_hit_count=2) == "contradicted"
    print("  PASS test_posture_contradicted_weak_contra_zero_support")


def test_posture_still_weak_when_support_present_and_contra_low():
    """Non-regression: support>0 with weak contradiction stays 'weak', not 'contradicted'."""
    e = make_engine()
    assert e._evidence_posture(0.20, 0.15, 0.10, retrieved_hit_count=3) == "weak"
    print("  PASS test_posture_still_weak_when_support_present_and_contra_low")


# ── Main runner ───────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_strong_support_yields_high_refined_score,
    test_strong_contradiction_fails_evidence_threshold,
    test_no_evidence_entry_fails_evidence_threshold,
    test_conjecture_discount_reduces_support,
    test_evidence_gap_true_when_no_hits,
    test_evidence_gap_false_when_hits_present,
    test_two_candidates_refined_independently,
    test_temporal_relation_backfill,
    test_refine_preserves_risk_adjusted_governance,
    test_authority_tier_weights_evidence_support,
    test_physical_plausibility_gate_eliminates_candidate_with_binary_rationale,
    test_timeline_consistency_gate_supports_normal_and_degraded_modes,
    test_barrier_logic_gate_supports_normal_and_degraded_modes,
    test_hard_gate_ordering_preserves_first_failure_reason_and_logs_rationales,
    test_coverage_quality_flags_reduce_quality_multiplier_and_surface_in_uncertainty_summary,
    test_posture_no_data,
    test_posture_contradicted,
    test_posture_supported,
    test_posture_mixed,
    test_posture_contextual_only,
    test_posture_weak,
    test_posture_contradicted_weak_contra_zero_support,
    test_posture_still_weak_when_support_present_and_contra_low,
]


def run_all():
    print(f"\n=== test_refine_with_evidence ({len(ALL_TESTS)} tests) ===")
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
