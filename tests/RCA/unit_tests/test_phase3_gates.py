"""
Phase 3 gating and analyst-override lifecycle tests.
"""
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from orchestrators.causality_engine_v32 import RuleBasedCausalityEngineV32
from synthesis.analyst_override_processor import AnalystOverrideProcessor


def test_decision_posture_requires_review_on_near_tie_or_blocked_candidates():
    posture = RuleBasedCausalityEngineV32._summarize_decision_posture(
        [
            {"candidate_id": "C1", "primary_eligibility": "blocked", "near_tie_with": []},
            {"candidate_id": "C2", "primary_eligibility": "eligible", "near_tie_with": ["C3"]},
        ]
    )
    assert posture["recommended_decision_status"] == "review_required"
    assert posture["near_tie"] is True
    assert posture["contradiction_blocked_count"] == 1


def test_decision_posture_candidate_ready_when_no_gate_violations():
    posture = RuleBasedCausalityEngineV32._summarize_decision_posture(
        [
            {"candidate_id": "C1", "primary_eligibility": "eligible", "near_tie_with": []},
            {"candidate_id": "C2", "primary_eligibility": "eligible", "near_tie_with": []},
        ]
    )
    assert posture["recommended_decision_status"] == "candidate_ready"
    assert posture["near_tie"] is False
    assert posture["contradiction_blocked_count"] == 0


def test_analyst_gate_override_records_reinstatement_lifecycle_event():
    processor = AnalystOverrideProcessor()
    base_card = {
        "rca_id": "RCA-1",
        "event_id": "EVT-1",
        "primary_hypothesis": {
            "candidate_id": "C1",
            "cause_label": "Cause 1",
            "composite_score": 0.7,
            "confidence_label": "medium",
        },
        "alternatives": [{"candidate_id": "C2", "cause_label": "Cause 2"}],
        "executive_summary": {
            "decision_status": "review_required",
            "analyst_attention_flags": [],
        },
        "analyst_review": {
            "decision_required": True,
            "writeback_recommendation": "hold_until_review",
            "questions_to_resolve": [],
        },
        "provenance": {"generated_by": "test"},
    }
    card, record = processor.apply(
        rca_card=base_card,
        override_input={
            "override_type": "gate_override_timeline",
            "rationale": "Timeline contradiction resolved with new SOE evidence.",
            "gate_override_candidate_id": "C2",
            "gate_override_evidence_refs": ["SOE::123"],
            "writeback_decision": "defer",
        },
        run_context={"run_id": "RUN-1", "asset_id": "A1"},
    )
    assert record["gate_override"]["lifecycle_event"] == "reinstated_by_analyst"
    assert record["gate_override"]["gate_type"] == "timeline"
    assert card["executive_summary"]["decision_status"] == "review_required"
    assert card["analyst_review"]["decision_required"] is True
