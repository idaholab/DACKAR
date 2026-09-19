"""
test_phase3_hallucination_guard.py — unit tests for Issue 10 (Phase 3):
extended hallucination guard in RuleValidatedRCASynthesizerV31.

Tests:
  TestValidateAndRepairLlmSections  (10 tests) — _validate_and_repair_llm_sections
  TestSynthesisQualityField         (6 tests)  — synthesis_quality in validation_status
"""
import copy
import sys
from pathlib import Path
from typing import Optional

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from synthesis.rca_synthesizer_v31 import RuleValidatedRCASynthesizerV31


# ── helpers ───────────────────────────────────────────────────────────────────

class _StubLLM:
    def __init__(self, output=None):
        self._output = output

    def generate_json(self, model, prompt, temperature=0.1):
        return self._output


def _synth(llm_output=None):
    return RuleValidatedRCASynthesizerV31(llm_client=_StubLLM(llm_output))


_VALID_IDS = {"FM::A", "FM::B", "NONE"}


def _base_card():
    """Card with two valid IDs used as primary / alternative / contributing."""
    return {
        "executive_summary": {
            "decision_status": "primary_identified",
            "primary_conclusion": "Root cause confirmed.",
            "analyst_attention_flags": [],
        },
        "primary_hypothesis": {
            "candidate_id": "FM::A",
            "cause_label": "Weld degradation",
            "hypothesis_type": "failure_mode",
            "narrative": "Thermal cycling degraded weld.",
            "why_primary": "Highest composite score.",
            "uncertainties": [],
            "composite_score": 0.85,
            "citations": ["WO-001"],
        },
        "alternatives": [
            {
                "candidate_id": "FM::B",
                "reason_not_primary": "DO contradicts.",
                "supports": ["Lower pressure"],
                "weaknesses": ["Inconsistent DO signal"],
            }
        ],
        "contributing_causes": [
            {
                "candidate_id": "FM::B",
                "cause_label": "Fouling",
                "contribution_type": "contributing",
                "rationale": "Increased thermal load.",
                "citations": [],
            }
        ],
        "evidence": [
            {
                "evidence_id": "E001",
                "source_type": "WO",
                "source_id": "WO-001",
                "support_role": "supporting",
                "summary": "Helium detected.",
                "excerpt": "Active leakage confirmed.",
                "linked_candidate_id": "FM::A",
            }
        ],
        "recommended_actions": [
            {
                "action_id": "A001",
                "action_type": "corrective",
                "description": "Replace joint.",
                "priority": "high",
                "linked_candidate_id": "FM::B",
            }
        ],
        "analyst_review": {
            "decision_required": False,
            "writeback_recommendation": "Update CR.",
            "questions_to_resolve": [],
        },
        "validation_status": {
            "schema_valid": False,
            "all_claims_cited": False,
            "passed_minimum_evidence_gate": False,
            "validation_errors": [],
            "retry_count": 0,
            "fallback_used": False,
            "synthesis_quality": "full_llm",
        },
    }


# ── TestValidateAndRepairLlmSections ─────────────────────────────────────────

class TestValidateAndRepairLlmSections:

    def test_no_hallucination_returns_zero(self):
        card = _base_card()
        count = RuleValidatedRCASynthesizerV31._validate_and_repair_llm_sections(
            card, _VALID_IDS
        )
        assert count == 0

    def test_contributing_cause_with_invented_id_removed(self):
        card = _base_card()
        card["contributing_causes"].append({
            "candidate_id": "FM::INVENTED",
            "cause_label": "LLM hallucination",
            "rationale": "...",
        })
        count = RuleValidatedRCASynthesizerV31._validate_and_repair_llm_sections(
            card, _VALID_IDS
        )
        assert count == 1
        assert all(c["candidate_id"] in _VALID_IDS for c in card["contributing_causes"])

    def test_alternative_with_invented_id_removed(self):
        card = _base_card()
        card["alternatives"].append({
            "candidate_id": "FM::GHOST",
            "reason_not_primary": "...",
        })
        count = RuleValidatedRCASynthesizerV31._validate_and_repair_llm_sections(
            card, _VALID_IDS
        )
        assert count == 1
        assert all(a["candidate_id"] in _VALID_IDS for a in card["alternatives"])

    def test_action_linked_candidate_id_nullified(self):
        card = _base_card()
        card["recommended_actions"][0]["linked_candidate_id"] = "FM::PHANTOM"
        count = RuleValidatedRCASynthesizerV31._validate_and_repair_llm_sections(
            card, _VALID_IDS
        )
        assert count == 1
        assert card["recommended_actions"][0]["linked_candidate_id"] is None

    def test_evidence_linked_candidate_id_nullified(self):
        card = _base_card()
        card["evidence"][0]["linked_candidate_id"] = "FM::MADE_UP"
        count = RuleValidatedRCASynthesizerV31._validate_and_repair_llm_sections(
            card, _VALID_IDS
        )
        assert count == 1
        assert card["evidence"][0]["linked_candidate_id"] is None

    def test_none_linked_candidate_id_untouched(self):
        card = _base_card()
        card["evidence"][0]["linked_candidate_id"] = None
        card["recommended_actions"][0]["linked_candidate_id"] = None
        count = RuleValidatedRCASynthesizerV31._validate_and_repair_llm_sections(
            card, _VALID_IDS
        )
        assert count == 0

    def test_multiple_sections_hallucinated_cumulative_count(self):
        card = _base_card()
        card["contributing_causes"].append(
            {"candidate_id": "FM::FAKE1", "rationale": "..."}
        )
        card["alternatives"].append(
            {"candidate_id": "FM::FAKE2", "reason_not_primary": "..."}
        )
        card["evidence"][0]["linked_candidate_id"] = "FM::FAKE3"
        card["recommended_actions"][0]["linked_candidate_id"] = "FM::FAKE4"
        count = RuleValidatedRCASynthesizerV31._validate_and_repair_llm_sections(
            card, _VALID_IDS
        )
        assert count == 4

    def test_non_dict_contributing_cause_removed_counted(self):
        card = _base_card()
        card["contributing_causes"].append("not_a_dict")
        count = RuleValidatedRCASynthesizerV31._validate_and_repair_llm_sections(
            card, _VALID_IDS
        )
        assert count == 1
        for c in card["contributing_causes"]:
            assert isinstance(c, dict)

    def test_none_sentinel_accepted_in_contributing_causes(self):
        card = _base_card()
        card["contributing_causes"] = [
            {"candidate_id": "NONE", "rationale": "No contributing cause identified."}
        ]
        count = RuleValidatedRCASynthesizerV31._validate_and_repair_llm_sections(
            card, _VALID_IDS
        )
        assert count == 0
        assert len(card["contributing_causes"]) == 1

    def test_empty_sections_no_error(self):
        card = _base_card()
        card["contributing_causes"] = []
        card["alternatives"] = []
        card["evidence"] = []
        card["recommended_actions"] = []
        count = RuleValidatedRCASynthesizerV31._validate_and_repair_llm_sections(
            card, _VALID_IDS
        )
        assert count == 0


# ── TestSynthesisQualityField ─────────────────────────────────────────────────

def _minimal_causality_candidates(candidate_ids=("FM::A",)):
    return {
        "candidates": [
            {
                "candidate_id": cid,
                "composite_score": 0.80,
                "cause_label": "Test cause",
                "failure_mode_id": cid,
                "component_id": "C::X",
                "hypothesis_type": "failure_mode",
                "review_required": False,
            }
            for cid in candidate_ids
        ]
    }


def _minimal_evidence_bundle():
    return {"bundle_id": "BND-001", "results": []}


def _minimal_event():
    return {"event_id": "EVT-001", "id": "EVT-001", "description": "Test event"}


def _minimal_run_context():
    return {"run_id": "RUN-001"}


def _good_llm_card(candidate_ids=("FM::A",)):
    """LLM output that references only real candidate IDs."""
    primary_id = candidate_ids[0]
    return {
        "rca_id": "RCA::EVT-001::test",
        "event_id": "EVT-001",
        "pipeline_config": {},
        "executive_summary": {
            "decision_status": "primary_identified",
            "primary_conclusion": "Root cause confirmed.",
            "confidence_label": "medium",
            "analyst_attention_flags": [],
        },
        "primary_hypothesis": {
            "candidate_id": primary_id,
            "cause_label": "Weld degradation",
            "hypothesis_type": "failure_mode",
            "narrative": "Thermal cycling degraded weld.",
            "why_primary": "Highest composite score.",
            "uncertainties": [],
            "composite_score": 0.82,
            "citations": ["WO-001"],
        },
        "alternatives": [],
        "contributing_causes": [],
        "evidence": [
            {
                "evidence_id": "E001",
                "source_type": "WO",
                "source_id": "WO-001",
                "support_role": "supporting",
                "summary": "Helium detected.",
                "excerpt": "Leakage confirmed.",
                "linked_candidate_id": primary_id,
            }
        ],
        "recommended_actions": [
            {
                "action_id": "A001",
                "action_type": "corrective",
                "description": "Replace joint.",
                "priority": "high",
            }
        ],
        "analyst_review": {
            "decision_required": False,
            "writeback_recommendation": "Update CR.",
            "questions_to_resolve": [],
        },
        "validation_status": {
            "schema_valid": True,
            "all_claims_cited": True,
            "passed_minimum_evidence_gate": True,
            "validation_errors": [],
            "retry_count": 0,
            "fallback_used": False,
            "synthesis_quality": "full_llm",
        },
    }


def _llm_card_with_invented_contributing_cause(candidate_ids=("FM::A",)):
    card = _good_llm_card(candidate_ids)
    card["contributing_causes"] = [
        {
            "candidate_id": "FM::INVENTED",
            "cause_label": "Hallucinated cause",
            "rationale": "LLM invented this.",
        }
    ]
    return card


class TestSynthesisQualityField:

    def test_synthesis_quality_present_in_validation_status(self):
        s = _synth(_good_llm_card(("FM::A",)))
        card = s.synthesize(
            event=_minimal_event(),
            telemetry_summary={},
            kg_context={},
            tskr_patterns=None,
            causality_candidates=_minimal_causality_candidates(("FM::A",)),
            evidence_bundle=_minimal_evidence_bundle(),
            operational_context=None,
            pm_compliance=None,
            ishikawa_matrix=None,
            run_context=_minimal_run_context(),
        )
        assert "synthesis_quality" in card["validation_status"]

    def test_full_llm_when_no_repairs_needed(self):
        s = _synth(_good_llm_card(("FM::A",)))
        card = s.synthesize(
            event=_minimal_event(),
            telemetry_summary={},
            kg_context={},
            tskr_patterns=None,
            causality_candidates=_minimal_causality_candidates(("FM::A",)),
            evidence_bundle=_minimal_evidence_bundle(),
            operational_context=None,
            pm_compliance=None,
            ishikawa_matrix=None,
            run_context=_minimal_run_context(),
        )
        assert card["validation_status"]["synthesis_quality"] == "full_llm"

    def test_partial_llm_when_contributing_cause_repaired(self):
        s = _synth(_llm_card_with_invented_contributing_cause(("FM::A",)))
        card = s.synthesize(
            event=_minimal_event(),
            telemetry_summary={},
            kg_context={},
            tskr_patterns=None,
            causality_candidates=_minimal_causality_candidates(("FM::A",)),
            evidence_bundle=_minimal_evidence_bundle(),
            operational_context=None,
            pm_compliance=None,
            ishikawa_matrix=None,
            run_context=_minimal_run_context(),
        )
        assert card["validation_status"]["synthesis_quality"] == "partial_llm"
        assert card["contributing_causes"] == []

    def test_deterministic_when_fallback_used(self):
        # LLM returns None → triggers fallback
        s = _synth(None)
        card = s.synthesize(
            event=_minimal_event(),
            telemetry_summary={},
            kg_context={},
            tskr_patterns=None,
            causality_candidates=_minimal_causality_candidates(("FM::A",)),
            evidence_bundle=_minimal_evidence_bundle(),
            operational_context=None,
            pm_compliance=None,
            ishikawa_matrix=None,
            run_context=_minimal_run_context(),
        )
        assert card["validation_status"]["synthesis_quality"] == "deterministic"
        assert card["validation_status"]["fallback_used"] is True

    def test_primary_hallucination_triggers_fallback_deterministic(self):
        # LLM invents a primary candidate_id → hard-reject → fallback
        bad_card = _good_llm_card(("FM::A",))
        bad_card["primary_hypothesis"]["candidate_id"] = "FM::INVENTED_PRIMARY"
        s = _synth(bad_card)
        card = s.synthesize(
            event=_minimal_event(),
            telemetry_summary={},
            kg_context={},
            tskr_patterns=None,
            causality_candidates=_minimal_causality_candidates(("FM::A",)),
            evidence_bundle=_minimal_evidence_bundle(),
            operational_context=None,
            pm_compliance=None,
            ishikawa_matrix=None,
            run_context=_minimal_run_context(),
        )
        assert card["validation_status"]["synthesis_quality"] == "deterministic"
        assert card["validation_status"]["fallback_used"] is True

    def test_repair_does_not_affect_primary_hypothesis(self):
        # Even with invented contributing_causes the primary is preserved
        s = _synth(_llm_card_with_invented_contributing_cause(("FM::A",)))
        card = s.synthesize(
            event=_minimal_event(),
            telemetry_summary={},
            kg_context={},
            tskr_patterns=None,
            causality_candidates=_minimal_causality_candidates(("FM::A",)),
            evidence_bundle=_minimal_evidence_bundle(),
            operational_context=None,
            pm_compliance=None,
            ishikawa_matrix=None,
            run_context=_minimal_run_context(),
        )
        assert card["primary_hypothesis"]["candidate_id"] == "FM::A"
        assert card["validation_status"]["synthesis_quality"] == "partial_llm"
