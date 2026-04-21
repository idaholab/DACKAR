"""
test_fallback_decision.py — standalone unit tests for
RuleValidatedRCASynthesizerV31._fallback_decision_status_from_posture and
                              _fallback_confidence_and_decision

Run directly:   python test_fallback_decision.py
Or via pytest:  pytest test_fallback_decision.py

_fallback_decision_status_from_posture:
  1. passed_minimum_evidence_gate=False              → "insufficient_evidence"
  2. contradicting > 0                               → "review_required"
  3. evidence_posture == "contradicted"              → "review_required"
  4. temporal_contradiction=True                     → "review_required"
  5. temporal_posture == "contradicted"              → "review_required"
  6. All clean (gate=True, no contradictions)        → "candidate_ready"

_fallback_confidence_and_decision:
  1. passed_minimum_evidence_gate=False              → low / insufficient_evidence
  2. supporting >= 2                                 → medium / review_required
  3. supporting < 2 (gate passed)                    → low / review_required
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


def make_evidence_summary(posture="supported", supporting=1, contradicting=0,
                          contextual=0, missing=0):
    return {
        "posture": posture,
        "supporting": supporting,
        "contradicting": contradicting,
        "contextual": contextual,
        "missing": missing,
    }


def make_pattern_posture(temporal_posture="supported", temporal_contradiction=False):
    return {
        "temporal_posture": temporal_posture,
        "temporal_contradiction": temporal_contradiction,
    }


# ── _fallback_decision_status_from_posture tests ─────────────────────────────

def test_decision_status_insufficient_when_gate_fails():
    """passed_minimum_evidence_gate=False → 'insufficient_evidence'."""
    s = make_synthesizer()
    result = s._fallback_decision_status_from_posture(
        evidence_summary=make_evidence_summary(),
        pattern_posture=make_pattern_posture(),
        passed_minimum_evidence_gate=False,
    )
    assert result == "insufficient_evidence", f"Expected insufficient_evidence, got {result}"
    print("  PASS test_decision_status_insufficient_when_gate_fails")


def test_decision_status_review_when_contradicting_evidence():
    """contradicting > 0 → 'review_required'."""
    s = make_synthesizer()
    result = s._fallback_decision_status_from_posture(
        evidence_summary=make_evidence_summary(supporting=1, contradicting=1),
        pattern_posture=make_pattern_posture(),
        passed_minimum_evidence_gate=True,
    )
    assert result == "review_required", f"Expected review_required, got {result}"
    print("  PASS test_decision_status_review_when_contradicting_evidence")


def test_decision_status_review_when_posture_contradicted():
    """evidence_posture == 'contradicted' → 'review_required'."""
    s = make_synthesizer()
    result = s._fallback_decision_status_from_posture(
        evidence_summary=make_evidence_summary(posture="contradicted", supporting=0, contradicting=0),
        pattern_posture=make_pattern_posture(),
        passed_minimum_evidence_gate=True,
    )
    assert result == "review_required", f"Expected review_required, got {result}"
    print("  PASS test_decision_status_review_when_posture_contradicted")


def test_decision_status_review_when_temporal_contradiction():
    """temporal_contradiction=True → 'review_required'."""
    s = make_synthesizer()
    result = s._fallback_decision_status_from_posture(
        evidence_summary=make_evidence_summary(),
        pattern_posture=make_pattern_posture(temporal_contradiction=True),
        passed_minimum_evidence_gate=True,
    )
    assert result == "review_required", f"Expected review_required, got {result}"
    print("  PASS test_decision_status_review_when_temporal_contradiction")


def test_decision_status_review_when_temporal_posture_contradicted():
    """temporal_posture == 'contradicted' → 'review_required'."""
    s = make_synthesizer()
    result = s._fallback_decision_status_from_posture(
        evidence_summary=make_evidence_summary(),
        pattern_posture=make_pattern_posture(temporal_posture="contradicted"),
        passed_minimum_evidence_gate=True,
    )
    assert result == "review_required", f"Expected review_required, got {result}"
    print("  PASS test_decision_status_review_when_temporal_posture_contradicted")


def test_decision_status_candidate_ready_when_all_clean():
    """No contradictions, gate passed → 'candidate_ready'."""
    s = make_synthesizer()
    result = s._fallback_decision_status_from_posture(
        evidence_summary=make_evidence_summary(posture="supported", contradicting=0),
        pattern_posture=make_pattern_posture(temporal_posture="supported", temporal_contradiction=False),
        passed_minimum_evidence_gate=True,
    )
    assert result == "candidate_ready", f"Expected candidate_ready, got {result}"
    print("  PASS test_decision_status_candidate_ready_when_all_clean")


def test_decision_status_gate_check_takes_priority():
    """Gate failure overrides any clean evidence — always insufficient_evidence."""
    s = make_synthesizer()
    result = s._fallback_decision_status_from_posture(
        evidence_summary=make_evidence_summary(posture="supported", supporting=3, contradicting=0),
        pattern_posture=make_pattern_posture(temporal_posture="supported"),
        passed_minimum_evidence_gate=False,   # gate fails
    )
    assert result == "insufficient_evidence"
    print("  PASS test_decision_status_gate_check_takes_priority")


def test_decision_status_contextual_only_is_candidate_ready():
    """contextual_only posture with no contradictions and gate passed → 'candidate_ready'."""
    s = make_synthesizer()
    result = s._fallback_decision_status_from_posture(
        evidence_summary=make_evidence_summary(posture="contextual_only", supporting=0, contradicting=0),
        pattern_posture=make_pattern_posture(),
        passed_minimum_evidence_gate=True,
    )
    assert result == "candidate_ready", (
        f"contextual_only with no contradictions should be candidate_ready, got {result}"
    )
    print("  PASS test_decision_status_contextual_only_is_candidate_ready")


# ── _fallback_confidence_and_decision tests ──────────────────────────────────

def test_confidence_decision_gate_fails():
    """passed_minimum_evidence_gate=False → low confidence + insufficient_evidence."""
    s = make_synthesizer()
    result = s._fallback_confidence_and_decision(
        evidence_summary=make_evidence_summary(supporting=5),
        passed_minimum_evidence_gate=False,
    )
    assert result["confidence_label"] == "low"
    assert result["decision_status"] == "insufficient_evidence"
    assert result["analyst_attention_flags"]
    print("  PASS test_confidence_decision_gate_fails")


def test_confidence_decision_strong_support():
    """supporting >= 2 and gate passed → medium confidence + review_required."""
    s = make_synthesizer()
    result = s._fallback_confidence_and_decision(
        evidence_summary=make_evidence_summary(supporting=2),
        passed_minimum_evidence_gate=True,
    )
    assert result["confidence_label"] == "medium"
    assert result["decision_status"] == "review_required"
    print("  PASS test_confidence_decision_strong_support")


def test_confidence_decision_weak_support():
    """supporting < 2 and gate passed → low confidence + review_required."""
    s = make_synthesizer()
    result = s._fallback_confidence_and_decision(
        evidence_summary=make_evidence_summary(supporting=1),
        passed_minimum_evidence_gate=True,
    )
    assert result["confidence_label"] == "low"
    assert result["decision_status"] == "review_required"
    print("  PASS test_confidence_decision_weak_support")


def test_confidence_decision_no_support():
    """supporting=0 (gate passed) → low confidence."""
    s = make_synthesizer()
    result = s._fallback_confidence_and_decision(
        evidence_summary=make_evidence_summary(supporting=0, posture="weak"),
        passed_minimum_evidence_gate=True,
    )
    assert result["confidence_label"] == "low"
    print("  PASS test_confidence_decision_no_support")


def test_confidence_decision_result_has_attention_flags():
    """All paths return non-empty analyst_attention_flags."""
    s = make_synthesizer()
    for gate, supporting in [(False, 0), (True, 0), (True, 2)]:
        result = s._fallback_confidence_and_decision(
            evidence_summary=make_evidence_summary(supporting=supporting),
            passed_minimum_evidence_gate=gate,
        )
        assert result.get("analyst_attention_flags"), (
            f"analyst_attention_flags should be non-empty "
            f"(gate={gate}, supporting={supporting})"
        )
    print("  PASS test_confidence_decision_result_has_attention_flags")


def test_high_confidence_achievable_with_fallback():
    """Sprint 1 fix: _calibrate_primary_confidence can reach 'high' with fallback_used=True.

    Before the fix, 'and not fallback_used' in the high-confidence condition and
    the subsequent cap block made high confidence structurally impossible.
    """
    s = make_synthesizer()
    posture = {
        "passed_minimum_evidence_gate": True,
        "supporting_evidence_count": 3,
        "contradicting_evidence_count": 0,
        "contextual_evidence_count": 1,
        "evidence_posture": "supported",
        "primary_score": 0.72,
        "runner_up_gap": 0.15,
        "recurrence_score": 0.50,
        "recurrence_confidence": "medium",
        "common_cause_score": 0.0,
        "common_cause_confidence": "none",
        "suspected_common_cause": False,
        "candidate_in_common_cause_cluster": False,
        "temporal_posture": "supported",
        "temporal_contradiction": False,
        "latency_violation_type": "none",
        "fallback_used": True,
    }
    result = s._calibrate_primary_confidence(posture)
    assert result == "high", f"expected 'high', got '{result}'"
    print("  PASS test_high_confidence_achievable_with_fallback")


# ── Main runner ───────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_decision_status_insufficient_when_gate_fails,
    test_decision_status_review_when_contradicting_evidence,
    test_decision_status_review_when_posture_contradicted,
    test_decision_status_review_when_temporal_contradiction,
    test_decision_status_review_when_temporal_posture_contradicted,
    test_decision_status_candidate_ready_when_all_clean,
    test_decision_status_gate_check_takes_priority,
    test_decision_status_contextual_only_is_candidate_ready,
    test_confidence_decision_gate_fails,
    test_confidence_decision_strong_support,
    test_confidence_decision_weak_support,
    test_confidence_decision_no_support,
    test_confidence_decision_result_has_attention_flags,
    test_high_confidence_achievable_with_fallback,
]


def run_all():
    print(f"\n=== test_fallback_decision ({len(ALL_TESTS)} tests) ===")
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
