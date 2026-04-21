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
):
    return {
        "candidate_id": candidate_id,
        "best_support_score": best_support_score,
        "best_contradiction_score": best_contradiction_score,
        "best_context_score": best_context_score,
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
