"""
test_synthesizer_fallback.py — standalone unit tests for
RuleValidatedRCASynthesizerV31._fallback_card

Run directly:   python test_synthesizer_fallback.py
Or via pytest:  pytest test_synthesizer_fallback.py

Key invariants:
  1. No candidates → NONE card (candidate_id="NONE", decision_status="insufficient_evidence")
  2. NONE card passes _validate_card_semantics
  3. Single candidate → primary card uses that candidate
  4. Alternatives drawn from selected_candidates[1:3]
  5. Evidence items linked to out-of-card candidate IDs get linked_candidate_id stripped (A10 fix)
  6. Card with top candidate passes _validate_card_semantics
  7. Evidence evidence_id values are sequential EV-001, EV-002, ...
"""
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from synthesis.rca_synthesizer_v31 import RuleValidatedRCASynthesizerV31


# ── Stub LLM client ───────────────────────────────────────────────────────────

class _StubLLM:
    def generate_json(self, model, prompt, temperature=0.1):
        return {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_synthesizer():
    return RuleValidatedRCASynthesizerV31(llm_client=_StubLLM())


def make_candidate(candidate_id, cause_label, composite_score=0.70):
    return {
        "candidate_id": candidate_id,
        "cause_label": cause_label,
        "hypothesis_type": "failure_mode",
        "cause_node_id": candidate_id.replace("FM::", ""),
        "composite_score": composite_score,
        "confidence_label": "high" if composite_score >= 0.75 else "medium",
        "evidence_posture": "supported",
        "temporal_posture": "partial",
        "supporting_evidence_refs": [],
        "scores": {"structural": 0.85, "temporal": 0.60, "telemetry": 0.60,
                   "evidence": 0.50, "governance": 0.50},
    }


def make_evidence_item(snippet_id, doc_id, linked_candidate_id=None, support_role="contextual"):
    return {
        "snippet_id": snippet_id,
        "doc_id": doc_id,
        "section": "test",
        "score": 0.7,
        "snippet": f"Excerpt from {doc_id}",
        "metadata": {
            "doc_type": "WO",
            "authority_level": "mandatory",
            "support_role": support_role,
            "linked_candidate_id": linked_candidate_id,
            "query_type": "candidate",
        },
    }


def make_run_context():
    return {"run_id": "RUN-001"}


def make_event():
    return {"id": "EVT-001", "asset_id": "U2-CONDENSER-MAIN"}


def make_causality_candidates(*candidates):
    return {"candidates": list(candidates), "summary": {}}


def make_evidence_bundle():
    return {"candidate_evidence_summary": []}


# ── Test functions ─────────────────────────────────────────────────────────────

def test_no_candidates_produces_none_card():
    """With empty selected_candidates, primary_hypothesis.candidate_id = 'NONE'."""
    s = make_synthesizer()
    card = s._fallback_card(
        rca_id="RCA-001",
        event=make_event(),
        selected_candidates=[],
        selected_evidence=[],
        causality_candidates=make_causality_candidates(),
        evidence_bundle=make_evidence_bundle(),
        run_context=make_run_context(),
        prior_errors=[],
    )
    assert card["primary_hypothesis"]["candidate_id"] == "NONE"
    assert card["executive_summary"]["decision_status"] == "insufficient_evidence"
    print("  PASS test_no_candidates_produces_none_card")


def test_none_card_passes_validation():
    """NONE card should produce only the expected 'evidence empty' structural note, no semantic errors."""
    s = make_synthesizer()
    card = s._fallback_card(
        rca_id="RCA-001",
        event=make_event(),
        selected_candidates=[],
        selected_evidence=[],
        causality_candidates=make_causality_candidates(),
        evidence_bundle=make_evidence_bundle(),
        run_context=make_run_context(),
        prior_errors=[],
    )
    errors = s._validate_card_semantics(card)
    # NONE cards have no evidence by design; only the structural "evidence empty" note is expected
    unexpected = [e for e in errors if e != "evidence empty"]
    assert unexpected == [], f"Unexpected errors in NONE card: {unexpected}"
    print("  PASS test_none_card_passes_validation")


def test_single_candidate_becomes_primary():
    """Top candidate is selected as primary_hypothesis.candidate_id."""
    s = make_synthesizer()
    cand = make_candidate("FM::AIR-INLEAK", "air in-leakage", composite_score=0.82)
    card = s._fallback_card(
        rca_id="RCA-001",
        event=make_event(),
        selected_candidates=[cand],
        selected_evidence=[],
        causality_candidates=make_causality_candidates(cand),
        evidence_bundle=make_evidence_bundle(),
        run_context=make_run_context(),
        prior_errors=[],
    )
    assert card["primary_hypothesis"]["candidate_id"] == "FM::AIR-INLEAK"
    assert card["primary_hypothesis"]["cause_label"] == "air in-leakage"
    print("  PASS test_single_candidate_becomes_primary")


def test_alternatives_drawn_from_candidates_1_and_2():
    """selected_candidates[1] and [2] become alternatives; [3] is not in the card."""
    s = make_synthesizer()
    c0 = make_candidate("FM::AIR-INLEAK",  "air in-leakage",  0.82)
    c1 = make_candidate("FM::TUBE-FOUL",   "tube fouling",    0.60)
    c2 = make_candidate("FM::CW-TEMP",     "CW temp rise",    0.55)
    c3 = make_candidate("FM::HVAC-DEGRAD", "HVAC degradation",0.40)

    card = s._fallback_card(
        rca_id="RCA-001",
        event=make_event(),
        selected_candidates=[c0, c1, c2, c3],
        selected_evidence=[],
        causality_candidates=make_causality_candidates(c0, c1, c2, c3),
        evidence_bundle=make_evidence_bundle(),
        run_context=make_run_context(),
        prior_errors=[],
    )
    alt_ids = [a["candidate_id"] for a in card.get("alternatives", [])]
    assert "FM::TUBE-FOUL" in alt_ids
    assert "FM::CW-TEMP"   in alt_ids
    assert "FM::HVAC-DEGRAD" not in alt_ids
    print("  PASS test_alternatives_drawn_from_candidates_1_and_2")


def test_evidence_linked_to_out_of_card_candidate_stripped():
    """
    Evidence linked to a candidate NOT in {primary} ∪ {top-2 alternatives}
    must have linked_candidate_id stripped to None (A10 fix).
    """
    s = make_synthesizer()
    c0 = make_candidate("FM::PRIMARY", "primary cause", 0.82)
    c1 = make_candidate("FM::ALT-1",   "alt 1",         0.60)
    # c2 is NOT in selected_candidates → not in card
    # Evidence linked to c2 must be stripped
    ev_linked_to_c2 = make_evidence_item(
        "SNIP-X", "DOC-X", linked_candidate_id="FM::NOT-IN-CARD"
    )
    ev_linked_to_c0 = make_evidence_item(
        "SNIP-Y", "DOC-Y", linked_candidate_id="FM::PRIMARY", support_role="supporting"
    )

    card = s._fallback_card(
        rca_id="RCA-001",
        event=make_event(),
        selected_candidates=[c0, c1],
        selected_evidence=[ev_linked_to_c2, ev_linked_to_c0],
        causality_candidates=make_causality_candidates(c0, c1),
        evidence_bundle=make_evidence_bundle(),
        run_context=make_run_context(),
        prior_errors=[],
    )

    # Verify validation passes (no unknown linked_candidate_id errors)
    errors = s._validate_card_semantics(card)
    assert not any("linked_candidate_id unknown" in e for e in errors), (
        f"linked_candidate_id stripping failed. Errors: {errors}"
    )
    print("  PASS test_evidence_linked_to_out_of_card_candidate_stripped")


def test_card_with_candidate_passes_validation():
    """A card produced from a well-formed candidate should pass _validate_card_semantics."""
    s = make_synthesizer()
    cand = make_candidate("FM::AIR-INLEAK", "air in-leakage", 0.82)
    cand["supporting_evidence_refs"] = ["SNIP-001"]  # ensures citations are populated
    ev = make_evidence_item("SNIP-001", "WO-2024-12001", linked_candidate_id="FM::AIR-INLEAK",
                            support_role="supporting")

    card = s._fallback_card(
        rca_id="RCA-001",
        event=make_event(),
        selected_candidates=[cand],
        selected_evidence=[ev],
        causality_candidates=make_causality_candidates(cand),
        evidence_bundle=make_evidence_bundle(),
        run_context=make_run_context(),
        prior_errors=[],
    )
    errors = s._validate_card_semantics(card)
    assert errors == [], f"Unexpected validation errors: {errors}"
    print("  PASS test_card_with_candidate_passes_validation")


def test_evidence_ids_are_sequential():
    """Evidence rows get EV-001, EV-002, ... IDs."""
    s = make_synthesizer()
    cand = make_candidate("FM::AIR-INLEAK", "air in-leakage", 0.82)
    evidence_items = [
        make_evidence_item(f"SNIP-{i}", f"DOC-{i}") for i in range(1, 4)
    ]
    card = s._fallback_card(
        rca_id="RCA-001",
        event=make_event(),
        selected_candidates=[cand],
        selected_evidence=evidence_items,
        causality_candidates=make_causality_candidates(cand),
        evidence_bundle=make_evidence_bundle(),
        run_context=make_run_context(),
        prior_errors=[],
    )
    ev_ids = [e["evidence_id"] for e in card.get("evidence", [])]
    assert ev_ids == ["EV-001", "EV-002", "EV-003"], f"Got: {ev_ids}"
    print("  PASS test_evidence_ids_are_sequential")


def test_fallback_evidence_balances_primary_and_alternative():
    """Fallback evidence block should include alternatives when candidate-linked evidence exists."""
    s = make_synthesizer()
    c0 = make_candidate("FM::PRIMARY", "primary cause", 0.82)
    c1 = make_candidate("FM::ALT-1", "alt cause", 0.74)
    ev_primary = make_evidence_item("SNIP-P", "DOC-P", linked_candidate_id="FM::PRIMARY", support_role="supporting")
    ev_alt = make_evidence_item("SNIP-A", "DOC-A", linked_candidate_id="FM::ALT-1", support_role="supporting")
    ev_context = make_evidence_item("SNIP-C", "DOC-C", linked_candidate_id=None, support_role="contextual")
    card = s._fallback_card(
        rca_id="RCA-001",
        event=make_event(),
        selected_candidates=[c0, c1],
        selected_evidence=[ev_primary, ev_context, ev_alt],
        causality_candidates=make_causality_candidates(c0, c1),
        evidence_bundle=make_evidence_bundle(),
        run_context=make_run_context(),
        prior_errors=[],
    )
    linked_ids = {row.get("linked_candidate_id") for row in (card.get("evidence") or []) if row.get("linked_candidate_id")}
    assert "FM::PRIMARY" in linked_ids
    assert "FM::ALT-1" in linked_ids
    print("  PASS test_fallback_evidence_balances_primary_and_alternative")


def test_safety_postprocessing_adds_flag_and_escalates_priority():
    """C4/C5: safety-significant primary adds analyst flag and escalates action priority."""
    s = make_synthesizer()
    card = {
        "primary_hypothesis": {"candidate_id": "FM::PRIMARY"},
        "executive_summary": {"analyst_attention_flags": []},
        "recommended_actions": [{"action_id": "A1", "action_type": "monitoring", "description": "x", "priority": "medium"}],
    }
    causality = {
        "candidates": [
            {
                "candidate_id": "FM::PRIMARY",
                "affected_safety_functions": [{"sf_name": "Reactor Protection", "sf_category": "reactor_protection"}],
            }
        ]
    }
    s._apply_safety_significance_postprocessing(card, causality)
    assert card["recommended_actions"][0]["priority"] == "critical"
    flags = card["executive_summary"]["analyst_attention_flags"]
    assert any("safety functions" in f.lower() for f in flags)
    print("  PASS test_safety_postprocessing_adds_flag_and_escalates_priority")


def test_risk_postprocessing_adds_visibility_to_summary_and_primary():
    """Follow-up: risk scalar should surface in analyst flags and primary text."""
    s = make_synthesizer()
    card = {
        "primary_hypothesis": {"candidate_id": "FM::PRIMARY", "why_primary": [], "uncertainties": []},
        "executive_summary": {"analyst_attention_flags": []},
        "recommended_actions": [{"action_id": "A1", "action_type": "monitoring", "description": "x", "priority": "low"}],
    }
    causality = {
        "candidates": [
            {
                "candidate_id": "FM::PRIMARY",
                "scores": {"risk_significance_scalar": 0.8, "risk_significance_tier": "high"},
            }
        ]
    }
    s._apply_safety_significance_postprocessing(card, causality)
    flags = card["executive_summary"]["analyst_attention_flags"]
    assert any("risk significance scalar" in f.lower() for f in flags)
    assert any("risk significance assessment" in x.lower() for x in card["primary_hypothesis"]["why_primary"])
    assert any("heuristic" in x.lower() for x in card["primary_hypothesis"]["uncertainties"])
    assert card["recommended_actions"][0]["priority"] == "high"
    print("  PASS test_risk_postprocessing_adds_visibility_to_summary_and_primary")


def test_enforce_balanced_card_evidence_adds_missing_alternative_row():
    s = make_synthesizer()
    card = {
        "primary_hypothesis": {"candidate_id": "FM::PRIMARY"},
        "alternatives": [{"candidate_id": "FM::ALT-1"}],
        "evidence": [
            {
                "evidence_id": "EV-001",
                "source_type": "evidence_snippet",
                "source_id": "SNIP-P",
                "doc_id": "DOC-P",
                "support_role": "supporting",
                "linked_candidate_id": "FM::PRIMARY",
                "summary": "Primary support",
                "excerpt": "x",
            }
        ],
    }
    selected_candidates = [
        make_candidate("FM::PRIMARY", "primary cause", 0.82),
        make_candidate("FM::ALT-1", "alt cause", 0.70),
    ]
    evidence_pool = [
        make_evidence_item("SNIP-A", "DOC-A", linked_candidate_id="FM::ALT-1", support_role="supporting"),
    ]
    s._enforce_balanced_card_evidence(
        card=card,
        selected_candidates=selected_candidates,
        evidence_pool=evidence_pool,
        max_rows=10,
    )
    linked_ids = {row.get("linked_candidate_id") for row in card.get("evidence", [])}
    assert "FM::ALT-1" in linked_ids
    print("  PASS test_enforce_balanced_card_evidence_adds_missing_alternative_row")


# ── Main runner ───────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_no_candidates_produces_none_card,
    test_none_card_passes_validation,
    test_single_candidate_becomes_primary,
    test_alternatives_drawn_from_candidates_1_and_2,
    test_evidence_linked_to_out_of_card_candidate_stripped,
    test_card_with_candidate_passes_validation,
    test_evidence_ids_are_sequential,
    test_fallback_evidence_balances_primary_and_alternative,
    test_safety_postprocessing_adds_flag_and_escalates_priority,
    test_risk_postprocessing_adds_visibility_to_summary_and_primary,
    test_enforce_balanced_card_evidence_adds_missing_alternative_row,
]


def run_all():
    print(f"\n=== test_synthesizer_fallback ({len(ALL_TESTS)} tests) ===")
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
