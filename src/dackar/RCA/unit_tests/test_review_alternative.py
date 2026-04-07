"""
test_review_alternative.py — standalone unit tests for
RuleBasedCausalityEngineV32._eligible_review_alternative and the
review-alternative reinstatement logic in refine_with_evidence

Run directly:   python test_review_alternative.py
Or via pytest:  pytest test_review_alternative.py

_eligible_review_alternative rules:
  1. score_gap ≤ review_alternative_gap (0.10) → eligible
  2. score_gap > 0.10 → not eligible
  3. other_candidate.temporal_posture == "contradicted" → not eligible
  4. other_candidate.temporal_evidence.temporal_contradiction=True → not eligible
  5. Either candidate is None/empty → False

Reinstatement logic (refine_with_evidence):
  6. Exactly 1 passes threshold + 1 fails + gap ≤ 0.10 → reinstated
  7. Exactly 1 passes + 1 fails + gap > 0.10 → NOT reinstated
  8. Zero pass → no reinstatement
  9. Two or more pass → no reinstatement attempt
 10. Reinstated candidate has retained_as_review_alternative=True
"""
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from orchestrators.causality_engine_v32 import RuleBasedCausalityEngineV32, CausalityEngineConfigV32


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_engine():
    return RuleBasedCausalityEngineV32()


def make_candidate(candidate_id, composite_score, meets_evidence_threshold=True,
                   temporal_posture=None, temporal_contradiction=False):
    c = {
        "candidate_id": candidate_id,
        "cause_label": f"cause for {candidate_id}",
        "composite_score": composite_score,
        "meets_evidence_threshold": meets_evidence_threshold,
        "scores": {
            "structural": 0.70,
            "temporal": 0.50,
            "telemetry": 0.50,
            "evidence": 0.50 if meets_evidence_threshold else 0.20,
            "governance": 0.50,
        },
    }
    if temporal_posture:
        c["temporal_posture"] = temporal_posture
    if temporal_contradiction:
        c["temporal_evidence"] = {"temporal_contradiction": True}
    return c


def make_causality_candidates(*candidates):
    return {
        "candidates": list(candidates),
        "summary": {},
        "filtered_out_candidates": [],
    }


def make_evidence_bundle(summaries=None):
    return {"candidate_evidence_summary": summaries or []}


def make_summary(candidate_id, best_support_score=0.0, best_contradiction_score=0.0,
                 best_context_score=0.0, hit_count=1):
    return {
        "candidate_id": candidate_id,
        "best_support_score": best_support_score,
        "best_contradiction_score": best_contradiction_score,
        "best_context_score": best_context_score,
        "hit_count": hit_count,
        "mean_conjecture_fraction": 0.0,
        "supporting_snippet_ids": [],
        "contradicting_snippet_ids": [],
        "contextual_snippet_ids": [],
        "dominant_temporal_relation": None,
        "best_lag_hours": None,
        "lag_is_approximate": False,
        "aggregated_mechanisms": [],
        "aggregated_outcomes": [],
    }


def get_retained(result):
    return result.get("candidates", [])


def get_filtered(result):
    return result.get("filtered_out_candidates", [])


def find_candidate(pool, candidate_id):
    for c in pool:
        if isinstance(c, dict) and c.get("candidate_id") == candidate_id:
            return c
    return None


# ── _eligible_review_alternative unit tests ───────────────────────────────

def test_eligible_when_gap_within_threshold():
    """Gap ≤ 0.10 and no temporal contradiction → eligible."""
    e = make_engine()
    primary = make_candidate("FM::A", composite_score=0.55)
    other = make_candidate("FM::B", composite_score=0.48)   # gap=0.07 ≤ 0.10
    assert e._eligible_review_alternative(primary, other) is True
    print("  PASS test_eligible_when_gap_within_threshold")


def test_not_eligible_when_gap_exceeds_threshold():
    """Gap > 0.10 → not eligible."""
    e = make_engine()
    primary = make_candidate("FM::A", composite_score=0.65)
    other = make_candidate("FM::B", composite_score=0.50)   # gap=0.15 > 0.10
    assert e._eligible_review_alternative(primary, other) is False
    print("  PASS test_not_eligible_when_gap_exceeds_threshold")


def test_not_eligible_when_temporal_posture_contradicted():
    """other.temporal_posture == 'contradicted' → not eligible, regardless of gap."""
    e = make_engine()
    primary = make_candidate("FM::A", composite_score=0.55)
    other = make_candidate("FM::B", composite_score=0.50, temporal_posture="contradicted")
    assert e._eligible_review_alternative(primary, other) is False
    print("  PASS test_not_eligible_when_temporal_posture_contradicted")


def test_not_eligible_when_temporal_contradiction_flag():
    """other.temporal_evidence.temporal_contradiction=True → not eligible."""
    e = make_engine()
    primary = make_candidate("FM::A", composite_score=0.55)
    other = make_candidate("FM::B", composite_score=0.50, temporal_contradiction=True)
    assert e._eligible_review_alternative(primary, other) is False
    print("  PASS test_not_eligible_when_temporal_contradiction_flag")


def test_not_eligible_when_other_is_none():
    """None candidate → False."""
    e = make_engine()
    primary = make_candidate("FM::A", composite_score=0.55)
    assert e._eligible_review_alternative(primary, None) is False
    print("  PASS test_not_eligible_when_other_is_none")


def test_eligible_exact_gap_boundary():
    """Gap exactly = 0.10 → eligible (boundary is inclusive on ≤)."""
    e = make_engine()
    primary = make_candidate("FM::A", composite_score=0.60)
    other = make_candidate("FM::B", composite_score=0.50)   # gap=0.10 exactly
    assert e._eligible_review_alternative(primary, other) is True
    print("  PASS test_eligible_exact_gap_boundary")


# ── Reinstatement via refine_with_evidence ────────────────────────────────
#
# To trigger reinstatement:
#   - Primary must PASS threshold after refinement (evidence score ≥ 0.35)
#   - Alternative must FAIL threshold (evidence score < 0.35)
#   - Gap between composite scores ≤ 0.10
#
# Scoring:
#   refined_evidence = 0.30*prior + 0.55*support + 0.15*context - 0.45*contradiction
#   composite = 0.30*structural + 0.20*temporal + 0.20*telemetry + 0.20*evidence + 0.10*governance

def make_refine_candidates_for_reinstatement(gap_within_threshold=True):
    """
    Primary (FM::PRIMARY):   prior=0.50, support=0.80 → refined_evidence=0.59
      scores pre-refinement: structural=0.70, temporal=0.50, telemetry=0.50,
                             evidence=0.50, governance=0.50
      post-refinement evidence=0.59
      composite = 0.30*0.70 + 0.20*0.50 + 0.20*0.50 + 0.20*0.59 + 0.10*0.50 = 0.578

    Alternative (FM::ALT):   prior=0.50, no support → refined_evidence=0.15
      same structural/temporal/telemetry/governance
      post-refinement evidence=0.15
      composite = 0.30*0.70 + 0.20*0.50 + 0.20*0.50 + 0.20*0.15 + 0.10*0.50 = 0.49

    gap within threshold: 0.578 - 0.49 = 0.088 ≤ 0.10 → reinstated
    gap above threshold: raise primary composite → structural=0.90
      primary composite = 0.30*0.90 + 0.20*0.50 + 0.20*0.50 + 0.20*0.59 + 0.10*0.50 = 0.638
      gap = 0.638 - 0.49 = 0.148 > 0.10 → NOT reinstated
    """
    structural_primary = 0.70 if gap_within_threshold else 0.90
    primary = {
        "candidate_id": "FM::PRIMARY",
        "cause_label": "primary cause",
        "composite_score": 0.578 if gap_within_threshold else 0.638,
        "meets_evidence_threshold": True,
        "scores": {
            "structural": structural_primary,
            "temporal": 0.50,
            "telemetry": 0.50,
            "evidence": 0.50,
            "governance": 0.50,
        },
    }
    alternative = {
        "candidate_id": "FM::ALT",
        "cause_label": "alternative cause",
        "composite_score": 0.49,
        "meets_evidence_threshold": False,   # fails evidence threshold pre-refinement
        "scores": {
            "structural": 0.70,
            "temporal": 0.50,
            "telemetry": 0.50,
            "evidence": 0.50,
            "governance": 0.50,
        },
    }
    return primary, alternative


def test_reinstatement_within_gap():
    """
    1 passes threshold, 1 fails, gap ≤ 0.10 → alternative reinstated with
    retained_as_review_alternative=True.
    """
    e = make_engine()
    primary, alt = make_refine_candidates_for_reinstatement(gap_within_threshold=True)
    candidates = make_causality_candidates(primary, alt)
    evidence = make_evidence_bundle([
        make_summary("FM::PRIMARY", best_support_score=0.80, hit_count=3),
        # FM::ALT has no evidence summary → refined_evidence = 0.30*0.50 = 0.15 < 0.35
    ])
    result = e.refine_with_evidence(candidates, evidence)

    retained = get_retained(result)
    assert len(retained) == 2, f"Expected 2 retained (primary + reinstated alt), got {len(retained)}"

    reinstated = find_candidate(retained, "FM::ALT")
    assert reinstated is not None, "FM::ALT not found in retained"
    assert reinstated.get("retained_as_review_alternative") is True, (
        "Reinstated candidate should have retained_as_review_alternative=True"
    )
    print("  PASS test_reinstatement_within_gap")


def test_no_reinstatement_when_gap_exceeds_threshold():
    """
    1 passes threshold, 1 fails, gap > 0.10 → alternative stays filtered.
    """
    e = make_engine()
    primary, alt = make_refine_candidates_for_reinstatement(gap_within_threshold=False)
    candidates = make_causality_candidates(primary, alt)
    evidence = make_evidence_bundle([
        make_summary("FM::PRIMARY", best_support_score=0.80, hit_count=3),
    ])
    result = e.refine_with_evidence(candidates, evidence)

    retained = get_retained(result)
    assert len(retained) == 1, f"Expected 1 retained (primary only), got {len(retained)}"
    assert find_candidate(retained, "FM::ALT") is None, "FM::ALT should not be in retained"
    print("  PASS test_no_reinstatement_when_gap_exceeds_threshold")


def test_no_reinstatement_when_zero_pass():
    """No candidates pass threshold → no reinstatement attempt."""
    e = make_engine()
    c1 = {
        "candidate_id": "FM::A",
        "cause_label": "cause A",
        "composite_score": 0.25,
        "meets_evidence_threshold": False,
        "scores": {"structural": 0.50, "temporal": 0.40, "telemetry": 0.40, "evidence": 0.20, "governance": 0.40},
    }
    c2 = {
        "candidate_id": "FM::B",
        "cause_label": "cause B",
        "composite_score": 0.22,
        "meets_evidence_threshold": False,
        "scores": {"structural": 0.50, "temporal": 0.35, "telemetry": 0.35, "evidence": 0.15, "governance": 0.35},
    }
    result = e.refine_with_evidence(make_causality_candidates(c1, c2), make_evidence_bundle([]))
    assert len(get_retained(result)) == 0, "No candidates should be retained"
    print("  PASS test_no_reinstatement_when_zero_pass")


def test_no_reinstatement_when_two_pass():
    """
    Two candidates both pass threshold → reinstatement logic is not triggered.
    Both should appear in retained.
    """
    e = make_engine()
    c1 = {
        "candidate_id": "FM::A",
        "cause_label": "cause A",
        "composite_score": 0.60,
        "meets_evidence_threshold": True,
        "scores": {"structural": 0.70, "temporal": 0.50, "telemetry": 0.50, "evidence": 0.50, "governance": 0.50},
    }
    c2 = {
        "candidate_id": "FM::B",
        "cause_label": "cause B",
        "composite_score": 0.55,
        "meets_evidence_threshold": True,
        "scores": {"structural": 0.65, "temporal": 0.50, "telemetry": 0.50, "evidence": 0.50, "governance": 0.50},
    }
    evidence = make_evidence_bundle([
        make_summary("FM::A", best_support_score=0.80, hit_count=2),
        make_summary("FM::B", best_support_score=0.80, hit_count=2),
    ])
    result = e.refine_with_evidence(make_causality_candidates(c1, c2), evidence)
    retained = get_retained(result)
    # Both should be retained; neither should be marked as review_alternative
    assert len(retained) == 2, f"Expected 2 retained, got {len(retained)}"
    for c in retained:
        assert c.get("retained_as_review_alternative") is not True, (
            f"{c['candidate_id']} should not be marked as review_alternative"
        )
    print("  PASS test_no_reinstatement_when_two_pass")


# ── Main runner ───────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_eligible_when_gap_within_threshold,
    test_not_eligible_when_gap_exceeds_threshold,
    test_not_eligible_when_temporal_posture_contradicted,
    test_not_eligible_when_temporal_contradiction_flag,
    test_not_eligible_when_other_is_none,
    test_eligible_exact_gap_boundary,
    test_reinstatement_within_gap,
    test_no_reinstatement_when_gap_exceeds_threshold,
    test_no_reinstatement_when_zero_pass,
    test_no_reinstatement_when_two_pass,
]


def run_all():
    print(f"\n=== test_review_alternative ({len(ALL_TESTS)} tests) ===")
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
