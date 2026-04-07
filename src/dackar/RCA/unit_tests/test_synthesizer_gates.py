"""
test_synthesizer_gates.py — standalone unit tests for
RuleValidatedRCASynthesizerV31._all_claims_cited and
                              _passes_minimum_evidence_gate

Run directly:   python test_synthesizer_gates.py
Or via pytest:  pytest test_synthesizer_gates.py

_all_claims_cited:
  1. NONE primary → always True (no claims to cite)
  2. No citations on primary → False
  3. Valid citations on primary → True
  4. Alternative with substantive claims but no citations → False
  5. Alternative with no substantive claims → citations not required → True

_passes_minimum_evidence_gate:
  1. NONE primary → False
  2. No candidate_id → False
  3. composite_score < minimum_primary_score (0.35) → False
  4. No citations → False
  5. Evidence list empty → False
  6. No supporting evidence linked to primary → False
  7. Supporting evidence present → True
"""
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from synthesis.rca_synthesizer_v31 import RuleValidatedRCASynthesizerV31


# ── Stub LLM ──────────────────────────────────────────────────────────────────

class _StubLLM:
    def generate_json(self, model, prompt, temperature=0.1):
        return {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_synthesizer():
    return RuleValidatedRCASynthesizerV31(llm_client=_StubLLM())


def minimal_citation():
    return {"claim_summary": "A claim.", "source_type": "evidence_snippet", "source_id": "SNIP-001", "excerpt": "x"}


def minimal_supporting_evidence(candidate_id):
    return {
        "evidence_id": "EV-001",
        "source_type": "WO",
        "source_id": "WO-001",
        "support_role": "supporting",
        "linked_candidate_id": candidate_id,
        "summary": "Supporting.",
        "excerpt": "Found degraded.",
    }


def minimal_card(candidate_id="FM::CAND-A", composite_score=0.82, citations=None,
                 evidence=None, alternatives=None):
    return {
        "primary_hypothesis": {
            "candidate_id": candidate_id,
            "cause_label": "air in-leakage",
            "hypothesis_type": "failure_mode",
            "narrative": "Narrative.",
            "why_primary": ["Reason."],
            "uncertainties": [],
            "composite_score": composite_score,
            "citations": citations if citations is not None else [minimal_citation()],
        },
        "alternatives": alternatives or [],
        "evidence": evidence if evidence is not None else [minimal_supporting_evidence(candidate_id)],
        "recommended_actions": [
            {"action_id": "A001", "action_type": "corrective", "description": "Fix it.", "priority": "high"}
        ],
        "executive_summary": {
            "decision_status": "review_required",
            "primary_conclusion": "Conclusion.",
            "analyst_attention_flags": [],
        },
        "analyst_review": {
            "decision_required": False,
            "questions_to_resolve": [],
            "writeback_recommendation": "hold_until_review",
        },
    }


# ── _all_claims_cited tests ────────────────────────────────────────────────

def test_claims_cited_none_primary():
    """NONE card → True (no claims to cite)."""
    s = make_synthesizer()
    card = minimal_card(candidate_id="NONE", citations=[])
    assert s._all_claims_cited(card) is True
    print("  PASS test_claims_cited_none_primary")


def test_claims_cited_no_primary_citations():
    """Primary has no citations → False."""
    s = make_synthesizer()
    card = minimal_card(citations=[])
    assert s._all_claims_cited(card) is False
    print("  PASS test_claims_cited_no_primary_citations")


def test_claims_cited_primary_with_citations():
    """Primary with citations, no alternatives → True."""
    s = make_synthesizer()
    card = minimal_card(citations=[minimal_citation()])
    assert s._all_claims_cited(card) is True
    print("  PASS test_claims_cited_primary_with_citations")


def test_claims_cited_alternative_substantive_no_citations():
    """Alternative has reason_not_primary but no citations → False."""
    s = make_synthesizer()
    card = minimal_card(alternatives=[
        {
            "candidate_id": "FM::CAND-B",
            "reason_not_primary": "Lower score.",
            "supports": ["Some support"],
            "weaknesses": ["Some weakness"],
            # NO citations field
        }
    ])
    assert s._all_claims_cited(card) is False
    print("  PASS test_claims_cited_alternative_substantive_no_citations")


def test_claims_cited_alternative_no_substantive_claims():
    """Alternative with no reason_not_primary, supports, weaknesses → citations not required."""
    s = make_synthesizer()
    card = minimal_card(alternatives=[
        {
            "candidate_id": "FM::CAND-B",
            # no reason_not_primary, no supports, no weaknesses
        }
    ])
    assert s._all_claims_cited(card) is True
    print("  PASS test_claims_cited_alternative_no_substantive_claims")


def test_claims_cited_alternative_with_citations():
    """Alternative with substantive claims AND citations → True."""
    s = make_synthesizer()
    card = minimal_card(alternatives=[
        {
            "candidate_id": "FM::CAND-B",
            "reason_not_primary": "Lower score.",
            "supports": ["Some support"],
            "weaknesses": ["Some weakness"],
            "citations": [minimal_citation()],
        }
    ])
    assert s._all_claims_cited(card) is True
    print("  PASS test_claims_cited_alternative_with_citations")


# ── _passes_minimum_evidence_gate tests ─────────────────────────────────────

def test_gate_none_primary_fails():
    """NONE primary → gate=False."""
    s = make_synthesizer()
    card = minimal_card(candidate_id="NONE", citations=[])
    assert s._passes_minimum_evidence_gate(card) is False
    print("  PASS test_gate_none_primary_fails")


def test_gate_low_composite_score_fails():
    """composite_score < 0.35 → gate=False."""
    s = make_synthesizer()
    card = minimal_card(composite_score=0.30)
    assert s._passes_minimum_evidence_gate(card) is False
    print("  PASS test_gate_low_composite_score_fails")


def test_gate_no_citations_fails():
    """No citations → gate=False."""
    s = make_synthesizer()
    card = minimal_card(composite_score=0.82, citations=[])
    assert s._passes_minimum_evidence_gate(card) is False
    print("  PASS test_gate_no_citations_fails")


def test_gate_no_evidence_rows_fails():
    """Empty evidence list → no supporting evidence → gate=False."""
    s = make_synthesizer()
    card = minimal_card(evidence=[])
    assert s._passes_minimum_evidence_gate(card) is False
    print("  PASS test_gate_no_evidence_rows_fails")


def test_gate_only_contextual_evidence_fails():
    """Evidence present but no 'supporting' role linked to primary → gate=False."""
    s = make_synthesizer()
    card = minimal_card(evidence=[
        {
            "evidence_id": "EV-001",
            "source_type": "WO",
            "source_id": "WO-001",
            "support_role": "contextual",   # not "supporting"
            "linked_candidate_id": "FM::CAND-A",
            "summary": "Contextual.",
            "excerpt": "Normal readings.",
        }
    ])
    assert s._passes_minimum_evidence_gate(card) is False
    print("  PASS test_gate_only_contextual_evidence_fails")


def test_gate_supporting_evidence_wrong_candidate_fails():
    """Supporting evidence linked to a different candidate → gate=False."""
    s = make_synthesizer()
    card = minimal_card(evidence=[
        {
            "evidence_id": "EV-001",
            "source_type": "WO",
            "source_id": "WO-001",
            "support_role": "supporting",
            "linked_candidate_id": "FM::OTHER-CAND",  # wrong candidate
            "summary": "Supporting other.",
            "excerpt": "Degraded.",
        }
    ])
    assert s._passes_minimum_evidence_gate(card) is False
    print("  PASS test_gate_supporting_evidence_wrong_candidate_fails")


def test_gate_passes_all_requirements():
    """composite≥0.35, citations present, supporting evidence linked to primary → True."""
    s = make_synthesizer()
    card = minimal_card(
        candidate_id="FM::CAND-A",
        composite_score=0.82,
        citations=[minimal_citation()],
        evidence=[minimal_supporting_evidence("FM::CAND-A")],
    )
    assert s._passes_minimum_evidence_gate(card) is True
    print("  PASS test_gate_passes_all_requirements")


def test_gate_composite_exactly_at_threshold():
    """composite_score == 0.35 (minimum_primary_score) → passes."""
    s = make_synthesizer()
    card = minimal_card(
        candidate_id="FM::CAND-A",
        composite_score=0.35,
        citations=[minimal_citation()],
        evidence=[minimal_supporting_evidence("FM::CAND-A")],
    )
    assert s._passes_minimum_evidence_gate(card) is True
    print("  PASS test_gate_composite_exactly_at_threshold")


# ── Main runner ───────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_claims_cited_none_primary,
    test_claims_cited_no_primary_citations,
    test_claims_cited_primary_with_citations,
    test_claims_cited_alternative_substantive_no_citations,
    test_claims_cited_alternative_no_substantive_claims,
    test_claims_cited_alternative_with_citations,
    test_gate_none_primary_fails,
    test_gate_low_composite_score_fails,
    test_gate_no_citations_fails,
    test_gate_no_evidence_rows_fails,
    test_gate_only_contextual_evidence_fails,
    test_gate_supporting_evidence_wrong_candidate_fails,
    test_gate_passes_all_requirements,
    test_gate_composite_exactly_at_threshold,
]


def run_all():
    print(f"\n=== test_synthesizer_gates ({len(ALL_TESTS)} tests) ===")
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
