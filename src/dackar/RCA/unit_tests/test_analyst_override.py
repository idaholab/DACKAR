"""
Unit tests for AnalystOverrideProcessor.
"""
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

import pytest
from synthesis.analyst_override_processor import AnalystOverrideProcessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rca_card(
    rca_id="RCA-001",
    event_id="EVT-001",
    asset_id="ASSET-A",
    primary_id="CAND-A",
    primary_label="Bearing wear",
    primary_score=0.72,
    alternatives=None,
    decision_status="review_required",
    decision_required=True,
    writeback_recommendation="hold_until_review",
):
    if alternatives is None:
        alternatives = [
            {
                "candidate_id": "CAND-B",
                "cause_label": "Lubrication failure",
                "hypothesis_type": "failure_mode",
                "composite_score": 0.55,
                "confidence_label": "medium",
                "reason_not_primary": "Lower telemetry support",
                "supports": ["Elevated temperature"],
                "weaknesses": ["No lubricant analysis"],
            }
        ]
    return {
        "rca_id": rca_id,
        "event_id": event_id,
        "asset_id": asset_id,
        "primary_hypothesis": {
            "candidate_id": primary_id,
            "cause_label": primary_label,
            "hypothesis_type": "failure_mode",
            "fm_id": "FM-BEAR-WEAR",
            "narrative": "Bearing shows progressive wear pattern.",
            "why_primary": ["Vibration signature matches", "Elevated temperature"],
            "uncertainties": ["No oil sample"],
            "composite_score": primary_score,
            "confidence_label": "medium",
            "citations": [],
        },
        "alternatives": alternatives,
        "executive_summary": {
            "decision_status": decision_status,
            "primary_conclusion": primary_label,
            "confidence_label": "medium",
            "analyst_attention_flags": [],
        },
        "analyst_review": {
            "decision_required": decision_required,
            "questions_to_resolve": ["What falsifies bearing wear?"],
            "writeback_recommendation": writeback_recommendation,
        },
        "evidence": [],
        "recommended_actions": [],
        "validation_status": {
            "schema_valid": True,
            "all_claims_cited": True,
            "passed_minimum_evidence_gate": True,
            "validation_errors": [],
            "retry_count": 0,
            "fallback_used": False,
        },
        "provenance": {
            "pipeline_version": "v31",
            "generated_by": "RuleValidatedRCASynthesizerV31",
        },
    }


def _make_run_context(run_id="run-001"):
    return {"run_id": run_id}


def _processor():
    return AnalystOverrideProcessor()


# ---------------------------------------------------------------------------
# accept — basic acceptance
# ---------------------------------------------------------------------------

class TestAccept:
    def test_accept_sets_decision_required_false(self):
        card = _make_rca_card()
        modified, record = _processor().apply(
            card,
            {"override_type": "accept", "rationale": "Evidence is clear.", "writeback_decision": "accept"},
            _make_run_context(),
        )
        assert modified["analyst_review"]["decision_required"] is False

    def test_accept_sets_writeback_ready(self):
        card = _make_rca_card()
        modified, _ = _processor().apply(
            card,
            {"override_type": "accept", "rationale": "ok", "writeback_decision": "accept"},
            _make_run_context(),
        )
        assert modified["analyst_review"]["writeback_recommendation"] == "ready_if_accepted"

    def test_accept_sets_decision_status_candidate_ready(self):
        card = _make_rca_card()
        modified, _ = _processor().apply(
            card,
            {"override_type": "accept", "rationale": "ok", "writeback_decision": "accept"},
            _make_run_context(),
        )
        assert modified["executive_summary"]["decision_status"] == "candidate_ready"

    def test_accept_does_not_change_primary(self):
        card = _make_rca_card()
        modified, _ = _processor().apply(
            card,
            {"override_type": "accept", "rationale": "ok", "writeback_decision": "accept"},
            _make_run_context(),
        )
        assert modified["primary_hypothesis"]["candidate_id"] == "CAND-A"

    def test_accept_preserves_original_card_immutability(self):
        card = _make_rca_card()
        original_decision = card["analyst_review"]["decision_required"]
        _processor().apply(
            card,
            {"override_type": "accept", "rationale": "ok", "writeback_decision": "accept"},
            _make_run_context(),
        )
        # Original card must not be mutated (processor deep-copies)
        assert card["analyst_review"]["decision_required"] == original_decision


# ---------------------------------------------------------------------------
# accept_with_caveats
# ---------------------------------------------------------------------------

class TestAcceptWithCaveats:
    def test_caveats_stored_in_analyst_review(self):
        card = _make_rca_card()
        modified, record = _processor().apply(
            card,
            {
                "override_type": "accept_with_caveats",
                "rationale": "Accept but note oil sample pending.",
                "writeback_decision": "accept",
                "caveats": ["Oil sample result awaited", "Re-inspect in 30 days"],
            },
            _make_run_context(),
        )
        assert modified["analyst_review"]["caveats"] == ["Oil sample result awaited", "Re-inspect in 30 days"]

    def test_caveats_in_override_record(self):
        card = _make_rca_card()
        _, record = _processor().apply(
            card,
            {
                "override_type": "accept_with_caveats",
                "rationale": "caveated",
                "writeback_decision": "defer",
                "caveats": ["Check follow-up"],
            },
            _make_run_context(),
        )
        assert record["caveats"] == ["Check follow-up"]


# ---------------------------------------------------------------------------
# primary_candidate_change
# ---------------------------------------------------------------------------

class TestPrimaryCandidateChange:
    def test_new_primary_candidate_id_correct(self):
        card = _make_rca_card()
        modified, _ = _processor().apply(
            card,
            {
                "override_type": "primary_candidate_change",
                "rationale": "Lubrication failure better supported by chemistry data.",
                "override_primary_candidate_id": "CAND-B",
                "writeback_decision": "accept",
            },
            _make_run_context(),
        )
        assert modified["primary_hypothesis"]["candidate_id"] == "CAND-B"

    def test_new_primary_cause_label_correct(self):
        card = _make_rca_card()
        modified, _ = _processor().apply(
            card,
            {
                "override_type": "primary_candidate_change",
                "rationale": "reason",
                "override_primary_candidate_id": "CAND-B",
                "writeback_decision": "accept",
            },
            _make_run_context(),
        )
        assert modified["primary_hypothesis"]["cause_label"] == "Lubrication failure"

    def test_old_primary_demoted_to_alternatives(self):
        card = _make_rca_card()
        modified, _ = _processor().apply(
            card,
            {
                "override_type": "primary_candidate_change",
                "rationale": "reason",
                "override_primary_candidate_id": "CAND-B",
                "writeback_decision": "accept",
            },
            _make_run_context(),
        )
        alt_ids = [a["candidate_id"] for a in modified["alternatives"]]
        assert "CAND-A" in alt_ids

    def test_demoted_primary_reason_contains_rationale(self):
        card = _make_rca_card()
        modified, _ = _processor().apply(
            card,
            {
                "override_type": "primary_candidate_change",
                "rationale": "chemistry data decisive",
                "override_primary_candidate_id": "CAND-B",
                "writeback_decision": "accept",
            },
            _make_run_context(),
        )
        demoted = next(
            a for a in modified["alternatives"] if a["candidate_id"] == "CAND-A"
        )
        assert "chemistry data decisive" in demoted["reason_not_primary"]

    def test_new_primary_flagged_as_analyst_override(self):
        card = _make_rca_card()
        modified, _ = _processor().apply(
            card,
            {
                "override_type": "primary_candidate_change",
                "rationale": "reason",
                "override_primary_candidate_id": "CAND-B",
                "writeback_decision": "accept",
            },
            _make_run_context(),
        )
        assert modified["primary_hypothesis"].get("analyst_override") is True

    def test_executive_summary_updated(self):
        card = _make_rca_card()
        modified, _ = _processor().apply(
            card,
            {
                "override_type": "primary_candidate_change",
                "rationale": "reason",
                "override_primary_candidate_id": "CAND-B",
                "writeback_decision": "accept",
            },
            _make_run_context(),
        )
        assert modified["executive_summary"]["primary_conclusion"] == "Lubrication failure"

    def test_override_record_contains_original_and_new(self):
        card = _make_rca_card()
        _, record = _processor().apply(
            card,
            {
                "override_type": "primary_candidate_change",
                "rationale": "reason",
                "override_primary_candidate_id": "CAND-B",
                "writeback_decision": "accept",
            },
            _make_run_context(),
        )
        assert record["original_primary"]["candidate_id"] == "CAND-A"
        assert record["override_primary"]["candidate_id"] == "CAND-B"


# ---------------------------------------------------------------------------
# reject_all
# ---------------------------------------------------------------------------

class TestRejectAll:
    def test_reject_sets_decision_required_true(self):
        card = _make_rca_card(decision_required=False, writeback_recommendation="ready_if_accepted")
        modified, _ = _processor().apply(
            card,
            {"override_type": "reject_all", "rationale": "RCA is premature.", "writeback_decision": "reject"},
            _make_run_context(),
        )
        assert modified["analyst_review"]["decision_required"] is True

    def test_reject_sets_writeback_hold(self):
        card = _make_rca_card(decision_required=False, writeback_recommendation="ready_if_accepted")
        modified, _ = _processor().apply(
            card,
            {"override_type": "reject_all", "rationale": "premature", "writeback_decision": "reject"},
            _make_run_context(),
        )
        assert modified["analyst_review"]["writeback_recommendation"] == "hold_until_review"

    def test_reject_sets_decision_status_review_required(self):
        card = _make_rca_card(decision_status="candidate_ready")
        modified, _ = _processor().apply(
            card,
            {"override_type": "reject_all", "rationale": "premature", "writeback_decision": "reject"},
            _make_run_context(),
        )
        assert modified["executive_summary"]["decision_status"] == "review_required"


# ---------------------------------------------------------------------------
# Override record structure
# ---------------------------------------------------------------------------

class TestOverrideRecord:
    def test_record_has_required_fields(self):
        card = _make_rca_card()
        _, record = _processor().apply(
            card,
            {"override_type": "accept", "rationale": "ok", "writeback_decision": "accept"},
            _make_run_context("run-xyz"),
        )
        for field in [
            "override_id", "run_id", "event_id", "asset_id", "created_at",
            "override_type", "rationale", "writeback_decision",
            "original_primary", "provenance",
        ]:
            assert field in record, f"Missing field: {field}"

    def test_override_id_format(self):
        card = _make_rca_card(event_id="EVT-999")
        _, record = _processor().apply(
            card,
            {"override_type": "accept", "rationale": "ok", "writeback_decision": "accept"},
            _make_run_context(),
        )
        assert record["override_id"].startswith("OVRD::EVT-999::")

    def test_run_id_propagated(self):
        card = _make_rca_card()
        _, record = _processor().apply(
            card,
            {"override_type": "accept", "rationale": "ok", "writeback_decision": "accept"},
            _make_run_context("run-abc"),
        )
        assert record["run_id"] == "run-abc"

    def test_original_primary_snapshot_correct(self):
        card = _make_rca_card(primary_id="CAND-A", primary_score=0.72)
        _, record = _processor().apply(
            card,
            {"override_type": "accept", "rationale": "ok", "writeback_decision": "accept"},
            _make_run_context(),
        )
        assert record["original_primary"]["candidate_id"] == "CAND-A"
        assert record["original_primary"]["composite_score"] == 0.72

    def test_provenance_fields(self):
        card = _make_rca_card(rca_id="RCA-XYZ")
        _, record = _processor().apply(
            card,
            {"override_type": "accept", "rationale": "ok", "writeback_decision": "accept"},
            _make_run_context(),
        )
        assert record["provenance"]["base_rca_card_id"] == "RCA-XYZ"
        assert record["provenance"]["generated_by"] == "AnalystOverrideProcessor"

    def test_provenance_captures_original_decision_status(self):
        card = _make_rca_card(decision_status="review_required")
        _, record = _processor().apply(
            card,
            {"override_type": "accept", "rationale": "ok", "writeback_decision": "accept"},
            _make_run_context(),
        )
        assert record["provenance"]["original_decision_status"] == "review_required"

    def test_card_provenance_stamped_with_override_id(self):
        card = _make_rca_card()
        modified, record = _processor().apply(
            card,
            {"override_type": "accept", "rationale": "ok", "writeback_decision": "accept"},
            _make_run_context(),
        )
        assert modified["provenance"]["analyst_override_id"] == record["override_id"]

    def test_analyst_id_stored(self):
        card = _make_rca_card()
        _, record = _processor().apply(
            card,
            {
                "override_type": "accept",
                "rationale": "ok",
                "writeback_decision": "accept",
                "analyst_id": "jsmith",
            },
            _make_run_context(),
        )
        assert record["analyst_id"] == "jsmith"

    def test_analyst_id_null_when_absent(self):
        card = _make_rca_card()
        _, record = _processor().apply(
            card,
            {"override_type": "accept", "rationale": "ok", "writeback_decision": "accept"},
            _make_run_context(),
        )
        assert record["analyst_id"] is None


# ---------------------------------------------------------------------------
# Questions resolved
# ---------------------------------------------------------------------------

class TestQuestionsResolved:
    def test_questions_stored_in_record(self):
        card = _make_rca_card()
        _, record = _processor().apply(
            card,
            {
                "override_type": "accept",
                "rationale": "ok",
                "writeback_decision": "accept",
                "questions_resolved": [
                    {
                        "question": "What falsifies bearing wear?",
                        "answer": "Normal vibration after overhaul.",
                        "resolution_type": "resolved",
                    }
                ],
            },
            _make_run_context(),
        )
        assert len(record["questions_resolved"]) == 1
        assert record["questions_resolved"][0]["answer"] == "Normal vibration after overhaul."

    def test_empty_questions_resolved_is_empty_list(self):
        card = _make_rca_card()
        _, record = _processor().apply(
            card,
            {"override_type": "accept", "rationale": "ok", "writeback_decision": "accept"},
            _make_run_context(),
        )
        assert record["questions_resolved"] == []


# ---------------------------------------------------------------------------
# Evidence additions
# ---------------------------------------------------------------------------

class TestEvidenceAdditions:
    def test_evidence_addition_stored(self):
        card = _make_rca_card()
        _, record = _processor().apply(
            card,
            {
                "override_type": "accept",
                "rationale": "ok",
                "writeback_decision": "accept",
                "evidence_additions": [
                    {
                        "evidence_ref": "WO-2024-99999",
                        "support_role": "supporting",
                        "added_by_analyst": True,
                        "notes": "Inspection result confirms wear pattern.",
                    }
                ],
            },
            _make_run_context(),
        )
        assert len(record["evidence_additions"]) == 1
        assert record["evidence_additions"][0]["evidence_ref"] == "WO-2024-99999"
        assert record["evidence_additions"][0]["added_by_analyst"] is True

    def test_empty_evidence_additions_is_empty_list(self):
        card = _make_rca_card()
        _, record = _processor().apply(
            card,
            {"override_type": "accept", "rationale": "ok", "writeback_decision": "accept"},
            _make_run_context(),
        )
        assert record["evidence_additions"] == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_invalid_override_type_raises(self):
        card = _make_rca_card()
        with pytest.raises(ValueError, match="override_type"):
            _processor().apply(
                card,
                {"override_type": "BOGUS", "rationale": "ok", "writeback_decision": "accept"},
                _make_run_context(),
            )

    def test_missing_rationale_raises(self):
        card = _make_rca_card()
        with pytest.raises(ValueError, match="rationale"):
            _processor().apply(
                card,
                {"override_type": "accept", "rationale": "", "writeback_decision": "accept"},
                _make_run_context(),
            )

    def test_missing_rationale_key_raises(self):
        card = _make_rca_card()
        with pytest.raises(ValueError, match="rationale"):
            _processor().apply(
                card,
                {"override_type": "accept", "writeback_decision": "accept"},
                _make_run_context(),
            )

    def test_invalid_writeback_decision_raises(self):
        card = _make_rca_card()
        with pytest.raises(ValueError, match="writeback_decision"):
            _processor().apply(
                card,
                {"override_type": "accept", "rationale": "ok", "writeback_decision": "approve"},
                _make_run_context(),
            )

    def test_primary_change_missing_target_id_raises(self):
        card = _make_rca_card()
        with pytest.raises(ValueError, match="override_primary_candidate_id"):
            _processor().apply(
                card,
                {
                    "override_type": "primary_candidate_change",
                    "rationale": "reason",
                    "writeback_decision": "accept",
                },
                _make_run_context(),
            )

    def test_primary_change_unknown_target_raises(self):
        card = _make_rca_card()
        with pytest.raises(ValueError, match="not present"):
            _processor().apply(
                card,
                {
                    "override_type": "primary_candidate_change",
                    "rationale": "reason",
                    "override_primary_candidate_id": "CAND-DOES-NOT-EXIST",
                    "writeback_decision": "accept",
                },
                _make_run_context(),
            )

    def test_primary_change_same_as_current_raises(self):
        card = _make_rca_card(primary_id="CAND-A")
        with pytest.raises(ValueError, match="already the current primary"):
            _processor().apply(
                card,
                {
                    "override_type": "primary_candidate_change",
                    "rationale": "reason",
                    "override_primary_candidate_id": "CAND-A",
                    "writeback_decision": "accept",
                },
                _make_run_context(),
            )

    def test_invalid_resolution_type_raises(self):
        card = _make_rca_card()
        with pytest.raises(ValueError, match="resolution_type"):
            _processor().apply(
                card,
                {
                    "override_type": "accept",
                    "rationale": "ok",
                    "writeback_decision": "accept",
                    "questions_resolved": [
                        {"question": "q", "answer": "a", "resolution_type": "BOGUS"}
                    ],
                },
                _make_run_context(),
            )

    def test_invalid_support_role_raises(self):
        card = _make_rca_card()
        with pytest.raises(ValueError, match="support_role"):
            _processor().apply(
                card,
                {
                    "override_type": "accept",
                    "rationale": "ok",
                    "writeback_decision": "accept",
                    "evidence_additions": [
                        {"evidence_ref": "DOC-1", "support_role": "BOGUS", "added_by_analyst": True}
                    ],
                },
                _make_run_context(),
            )
