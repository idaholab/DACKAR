"""
test_synthesizer_validation.py — standalone unit tests for
RuleValidatedRCASynthesizerV31._validate_card_semantics

Run directly:   python test_synthesizer_validation.py
Or via pytest:  pytest test_synthesizer_validation.py

_validate_card_semantics returns a list of error strings.
Empty list = valid card.

Key invariants tested:
  1. Minimal valid card → no errors
  2. Missing required top-level fields → specific error strings
  3. Evidence item with unknown linked_candidate_id → error
  4. Evidence item with valid linked_candidate_id (primary or alternative) → no error
  5. Invalid support_role value → error
  6. Missing evidence_id / source_type / source_id → errors
  7. Missing action fields → errors
  8. Action linked_candidate_id not in card → error
  9. Alternatives validation: missing reason_not_primary, invalid supports/weaknesses
"""
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from synthesis.rca_synthesizer_v31 import RuleValidatedRCASynthesizerV31


# ── Stub LLM client (never called in validation) ──────────────────────────────

class _StubLLM:
    def generate_json(self, model, prompt, temperature=0.1):
        return {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_synthesizer():
    return RuleValidatedRCASynthesizerV31(llm_client=_StubLLM())


def minimal_valid_card():
    """Returns a card that passes all validations."""
    return {
        "executive_summary": {
            "decision_status": "primary_identified",
            "primary_conclusion": "Air in-leakage confirmed as root cause.",
            "analyst_attention_flags": [],
        },
        "primary_hypothesis": {
            "candidate_id": "FM::CAND-A",
            "cause_label": "Air in-leakage",
            "hypothesis_type": "failure_mode",
            "narrative": "Expansion joint weld degraded due to thermal cycling.",
            "why_primary": "Highest composite score; DO elevation is diagnostic.",
            "uncertainties": [],
            "composite_score": 0.82,
            "citations": ["WO-2024-12001"],
        },
        "alternatives": [
            {
                "candidate_id": "FM::CAND-B",
                "reason_not_primary": "DO within normal limits contradicts this hypothesis.",
                "supports": ["Lower backpressure baseline"],
                "weaknesses": ["DO elevation inconsistent with tube fouling"],
            }
        ],
        "evidence": [
            {
                "evidence_id": "E001",
                "source_type": "WO",
                "source_id": "WO-2024-12001",
                "support_role": "supporting",
                "summary": "Helium detected at expansion joint.",
                "excerpt": "Active air in-leakage pathway confirmed.",
                "linked_candidate_id": "FM::CAND-A",
            }
        ],
        "recommended_actions": [
            {
                "action_id": "A001",
                "action_type": "corrective",
                "description": "Replace expansion joint.",
                "priority": "high",
            }
        ],
        "analyst_review": {
            "decision_required": False,
            "writeback_recommendation": "Update CR with confirmed root cause.",
            "questions_to_resolve": [],
        },
    }


# ── Test functions ─────────────────────────────────────────────────────────────

def test_minimal_valid_card_passes():
    s = make_synthesizer()
    errors = s._validate_card_semantics(minimal_valid_card())
    assert errors == [], f"Expected no errors, got: {errors}"
    print("  PASS test_minimal_valid_card_passes")


def test_missing_decision_status():
    s = make_synthesizer()
    card = minimal_valid_card()
    del card["executive_summary"]["decision_status"]
    errors = s._validate_card_semantics(card)
    assert any("decision_status" in e for e in errors), f"Missing error for decision_status. Got: {errors}"
    print("  PASS test_missing_decision_status")


def test_missing_primary_conclusion():
    s = make_synthesizer()
    card = minimal_valid_card()
    del card["executive_summary"]["primary_conclusion"]
    errors = s._validate_card_semantics(card)
    assert any("primary_conclusion" in e for e in errors)
    print("  PASS test_missing_primary_conclusion")


def test_missing_primary_candidate_id():
    s = make_synthesizer()
    card = minimal_valid_card()
    del card["primary_hypothesis"]["candidate_id"]
    errors = s._validate_card_semantics(card)
    assert any("candidate_id" in e and "primary" in e for e in errors)
    print("  PASS test_missing_primary_candidate_id")


def test_missing_primary_narrative():
    s = make_synthesizer()
    card = minimal_valid_card()
    del card["primary_hypothesis"]["narrative"]
    errors = s._validate_card_semantics(card)
    assert any("narrative" in e for e in errors)
    print("  PASS test_missing_primary_narrative")


def test_evidence_linked_to_unknown_candidate():
    """Evidence linked to a candidate_id not in primary or alternatives → error."""
    s = make_synthesizer()
    card = minimal_valid_card()
    card["evidence"][0]["linked_candidate_id"] = "FM::UNKNOWN-CAND"
    errors = s._validate_card_semantics(card)
    assert any("linked_candidate_id unknown" in e for e in errors), (
        f"Expected 'linked_candidate_id unknown' error. Got: {errors}"
    )
    print("  PASS test_evidence_linked_to_unknown_candidate")


def test_evidence_linked_to_alternative_is_valid():
    """Evidence linked to an alternative candidate_id → no error."""
    s = make_synthesizer()
    card = minimal_valid_card()
    card["evidence"].append({
        "evidence_id": "E002",
        "source_type": "WO",
        "source_id": "WO-2024-11847",
        "support_role": "contradicting",
        "summary": "Tube inspection normal.",
        "excerpt": "All tubes within normal limits.",
        "linked_candidate_id": "FM::CAND-B",  # alternative — should be valid
    })
    errors = s._validate_card_semantics(card)
    assert not any("E002" in e or "linked_candidate_id" in e for e in errors), (
        f"Unexpected errors for valid alternative link: {errors}"
    )
    print("  PASS test_evidence_linked_to_alternative_is_valid")


def test_evidence_invalid_support_role():
    s = make_synthesizer()
    card = minimal_valid_card()
    card["evidence"][0]["support_role"] = "invalid_role"
    errors = s._validate_card_semantics(card)
    assert any("support_role invalid" in e for e in errors)
    print("  PASS test_evidence_invalid_support_role")


def test_evidence_missing_source_id():
    s = make_synthesizer()
    card = minimal_valid_card()
    del card["evidence"][0]["source_id"]
    errors = s._validate_card_semantics(card)
    assert any("source_id" in e for e in errors)
    print("  PASS test_evidence_missing_source_id")


def test_action_linked_to_unknown_candidate():
    s = make_synthesizer()
    card = minimal_valid_card()
    card["recommended_actions"][0]["linked_candidate_id"] = "FM::GHOST"
    errors = s._validate_card_semantics(card)
    assert any("linked_candidate_id unknown" in e for e in errors)
    print("  PASS test_action_linked_to_unknown_candidate")


def test_action_missing_description():
    s = make_synthesizer()
    card = minimal_valid_card()
    del card["recommended_actions"][0]["description"]
    errors = s._validate_card_semantics(card)
    assert any("description" in e for e in errors)
    print("  PASS test_action_missing_description")


def test_alternative_missing_reason_not_primary():
    s = make_synthesizer()
    card = minimal_valid_card()
    del card["alternatives"][0]["reason_not_primary"]
    errors = s._validate_card_semantics(card)
    assert any("reason_not_primary" in e for e in errors)
    print("  PASS test_alternative_missing_reason_not_primary")


def test_empty_evidence_list_is_error():
    s = make_synthesizer()
    card = minimal_valid_card()
    card["evidence"] = []
    errors = s._validate_card_semantics(card)
    assert any("evidence empty" in e for e in errors)
    print("  PASS test_empty_evidence_list_is_error")


def test_empty_recommended_actions_is_error():
    s = make_synthesizer()
    card = minimal_valid_card()
    card["recommended_actions"] = []
    errors = s._validate_card_semantics(card)
    assert any("recommended_actions empty" in e for e in errors)
    print("  PASS test_empty_recommended_actions_is_error")


# ── Main runner ───────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_minimal_valid_card_passes,
    test_missing_decision_status,
    test_missing_primary_conclusion,
    test_missing_primary_candidate_id,
    test_missing_primary_narrative,
    test_evidence_linked_to_unknown_candidate,
    test_evidence_linked_to_alternative_is_valid,
    test_evidence_invalid_support_role,
    test_evidence_missing_source_id,
    test_action_linked_to_unknown_candidate,
    test_action_missing_description,
    test_alternative_missing_reason_not_primary,
    test_empty_evidence_list_is_error,
    test_empty_recommended_actions_is_error,
]


def run_all():
    print(f"\n=== test_synthesizer_validation ({len(ALL_TESTS)} tests) ===")
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
