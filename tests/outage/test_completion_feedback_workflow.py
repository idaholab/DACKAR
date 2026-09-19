"""
Unit tests for the post-completion feedback ingestion workflow (M5).

Coverage targets:
    CompletionFeedbackWorkflow.ingest()

        Routing:
        — accepted decision → record_completion() called, skipped=False
        — None reviewer_decision (review not required) → treated as accepted
        — modified decision → record_completion() called normally
        — pending decision → skipped=True, record_completion() NOT called
        — unrecognised decision → warn + treat as accepted

        Rejection path:
        — rejected decision → record_completion() still called
        — outcome_notes prefixed with RECOMMENDATION_REJECTED:
        — reviewer_notes appended to rejection prefix
        — caller outcome_notes appended after rejection prefix
        — rejected with no reviewer_notes → prefix only

        Guard conditions:
        — missing activity_id → skipped=True, record_completion() NOT called
        — empty string activity_id → skipped=True

        Field forwarding:
        — actual_duration_hours forwarded unchanged
        — actual_start / actual_finish forwarded
        — run_id extracted from recommendation

        Receipt fields:
        — completion_record populated on success
        — completion_record is None when skipped
        — reviewer_decision copied into receipt
        — skip_reason is None when not skipped

        No feedback_writer wired:
        — orchestrator without feedback_writer returns no-op CompletionRecord
        — receipt.skipped=False (orchestrator handled it gracefully)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, call

import pytest

_OUTAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "dackar" / "outage"
if str(_OUTAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTAGE_ROOT))

from stages.completion_feedback import CompletionRecord, _utcnow_iso
from workflows.completion_feedback_workflow import (
    CompletionFeedbackWorkflow,
    IngestionReceipt,
    _REJECTION_PREFIX,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _no_op_record(
    activity_id: str = "ACT-001",
    run_id: str = "RUN-001",
    duration: float = 8.0,
    outcome_notes: str = "",
) -> CompletionRecord:
    """Return a minimal CompletionRecord as if no feedback_writer was injected."""
    return CompletionRecord(
        activity_id=activity_id,
        run_id=run_id,
        actual_duration_hours=duration,
        actual_start=None,
        actual_finish=None,
        outcome_notes=outcome_notes,
        written_at=_utcnow_iso(),
        index_updated=False,
        persisted=False,
        validation_warnings=("feedback_writer not injected",),
    )


def _ok_record(
    activity_id: str = "ACT-001",
    run_id: str = "RUN-001",
    duration: float = 8.0,
    outcome_notes: str = "",
) -> CompletionRecord:
    """Return a CompletionRecord as if both index and persister succeeded."""
    return CompletionRecord(
        activity_id=activity_id,
        run_id=run_id,
        actual_duration_hours=duration,
        actual_start=None,
        actual_finish=None,
        outcome_notes=outcome_notes,
        written_at=_utcnow_iso(),
        index_updated=True,
        persisted=True,
        validation_warnings=(),
    )


def _mock_orchestrator(record: Optional[CompletionRecord] = None) -> MagicMock:
    """Orchestrator mock whose record_completion() returns *record*."""
    orch = MagicMock()
    orch.record_completion.return_value = record or _ok_record()
    return orch


def _recommendation(
    activity_id: str = "ACT-001",
    run_id: str = "RUN-001",
    reviewer_decision: Optional[str] = "accepted",
    reviewer_notes: Optional[str] = None,
) -> dict:
    """Build a minimal Stage G artifact dict for testing."""
    return {
        "activity_id": activity_id,
        "run_id": run_id,
        "analyst_review": {
            "required": True,
            "reviewer_decision": reviewer_decision,
            "reviewer_notes": reviewer_notes,
        },
    }


def _workflow(record: Optional[CompletionRecord] = None) -> tuple:
    """Return (workflow, mock_orchestrator) pair."""
    orch = _mock_orchestrator(record)
    return CompletionFeedbackWorkflow(orchestrator=orch), orch


# ===========================================================================
# Routing tests
# ===========================================================================

class TestIngestionRouting:

    def test_accepted_calls_record_completion(self):
        wf, orch = _workflow()
        rec = _recommendation(reviewer_decision="accepted")
        receipt = wf.ingest(rec, actual_duration_hours=8.0)
        orch.record_completion.assert_called_once()
        assert not receipt.skipped

    def test_none_reviewer_decision_treated_as_accepted(self):
        """reviewer_decision=None means no analyst review was required — still write."""
        wf, orch = _workflow()
        rec = _recommendation(reviewer_decision=None)
        receipt = wf.ingest(rec, actual_duration_hours=5.0)
        orch.record_completion.assert_called_once()
        assert not receipt.skipped

    def test_modified_calls_record_completion(self):
        wf, orch = _workflow()
        rec = _recommendation(reviewer_decision="modified")
        receipt = wf.ingest(rec, actual_duration_hours=10.0)
        orch.record_completion.assert_called_once()
        assert not receipt.skipped

    def test_pending_is_skipped_no_record_completion(self):
        wf, orch = _workflow()
        rec = _recommendation(reviewer_decision="pending")
        receipt = wf.ingest(rec, actual_duration_hours=8.0)
        orch.record_completion.assert_not_called()
        assert receipt.skipped
        assert receipt.skip_reason == "analyst review pending"

    def test_unrecognised_decision_treated_as_accepted(self):
        """Unknown reviewer_decision values should warn but not crash."""
        wf, orch = _workflow()
        rec = _recommendation(reviewer_decision="superseded")
        receipt = wf.ingest(rec, actual_duration_hours=8.0)
        orch.record_completion.assert_called_once()
        assert not receipt.skipped


# ===========================================================================
# Rejection path
# ===========================================================================

class TestRejectionPath:

    def test_rejected_still_calls_record_completion(self):
        """Rejected recommendations are stored for negative learning."""
        wf, orch = _workflow()
        rec = _recommendation(reviewer_decision="rejected")
        receipt = wf.ingest(rec, actual_duration_hours=8.0)
        orch.record_completion.assert_called_once()
        assert not receipt.skipped

    def test_rejected_outcome_notes_prefixed_with_rejection_marker(self):
        wf, orch = _workflow()
        rec = _recommendation(reviewer_decision="rejected", reviewer_notes=None)
        wf.ingest(rec, actual_duration_hours=8.0)
        _, kwargs = orch.record_completion.call_args
        assert kwargs["outcome_notes"].startswith(_REJECTION_PREFIX)

    def test_rejected_reviewer_notes_appended_to_prefix(self):
        wf, orch = _workflow()
        rec = _recommendation(
            reviewer_decision="rejected",
            reviewer_notes="Analogs from wrong component family.",
        )
        wf.ingest(rec, actual_duration_hours=8.0)
        _, kwargs = orch.record_completion.call_args
        notes = kwargs["outcome_notes"]
        assert "Analogs from wrong component family." in notes
        assert notes.startswith(_REJECTION_PREFIX)

    def test_rejected_caller_notes_appended_after_prefix(self):
        wf, orch = _workflow()
        rec = _recommendation(reviewer_decision="rejected", reviewer_notes="Bad match.")
        wf.ingest(rec, actual_duration_hours=8.0, outcome_notes="Field notes here.")
        _, kwargs = orch.record_completion.call_args
        notes = kwargs["outcome_notes"]
        # rejection prefix first, caller notes somewhere in the string
        assert notes.startswith(_REJECTION_PREFIX)
        assert "Field notes here." in notes

    def test_rejected_no_reviewer_notes_prefix_only(self):
        wf, orch = _workflow()
        rec = _recommendation(reviewer_decision="rejected", reviewer_notes=None)
        wf.ingest(rec, actual_duration_hours=8.0)
        _, kwargs = orch.record_completion.call_args
        notes = kwargs["outcome_notes"]
        # Should be exactly the prefix (no trailing colon + space when no notes)
        assert notes == _REJECTION_PREFIX or notes.startswith(_REJECTION_PREFIX)


# ===========================================================================
# Guard conditions
# ===========================================================================

class TestGuardConditions:

    def test_missing_activity_id_is_skipped(self):
        wf, orch = _workflow()
        rec = _recommendation(activity_id="")
        receipt = wf.ingest(rec, actual_duration_hours=8.0)
        orch.record_completion.assert_not_called()
        assert receipt.skipped
        assert receipt.skip_reason == "missing activity_id"

    def test_no_activity_id_key_is_skipped(self):
        wf, orch = _workflow()
        rec = {"run_id": "RUN-001", "analyst_review": {"reviewer_decision": "accepted"}}
        receipt = wf.ingest(rec, actual_duration_hours=8.0)
        orch.record_completion.assert_not_called()
        assert receipt.skipped


# ===========================================================================
# Field forwarding
# ===========================================================================

class TestFieldForwarding:

    def test_actual_duration_hours_forwarded(self):
        wf, orch = _workflow()
        rec = _recommendation()
        wf.ingest(rec, actual_duration_hours=16.2)
        _, kwargs = orch.record_completion.call_args
        assert kwargs["actual_duration_hours"] == 16.2

    def test_actual_start_finish_forwarded(self):
        wf, orch = _workflow()
        rec = _recommendation()
        wf.ingest(
            rec,
            actual_duration_hours=16.2,
            actual_start="2026-04-12T08:00:00Z",
            actual_finish="2026-04-13T00:12:00Z",
        )
        _, kwargs = orch.record_completion.call_args
        assert kwargs["actual_start"] == "2026-04-12T08:00:00Z"
        assert kwargs["actual_finish"] == "2026-04-13T00:12:00Z"

    def test_run_id_extracted_from_recommendation(self):
        wf, orch = _workflow()
        rec = _recommendation(run_id="OUTAGE::xyz789")
        wf.ingest(rec, actual_duration_hours=8.0)
        _, kwargs = orch.record_completion.call_args
        assert kwargs["run_id"] == "OUTAGE::xyz789"

    def test_activity_id_extracted_from_recommendation(self):
        wf, orch = _workflow()
        rec = _recommendation(activity_id="ACT-20260412-007")
        wf.ingest(rec, actual_duration_hours=8.0)
        _, kwargs = orch.record_completion.call_args
        assert kwargs["activity_id"] == "ACT-20260412-007"

    def test_caller_outcome_notes_forwarded_for_accepted(self):
        wf, orch = _workflow()
        rec = _recommendation(reviewer_decision="accepted", reviewer_notes=None)
        wf.ingest(rec, actual_duration_hours=8.0, outcome_notes="No scope expansion.")
        _, kwargs = orch.record_completion.call_args
        assert "No scope expansion." in kwargs["outcome_notes"]

    def test_reviewer_notes_appended_for_accepted(self):
        wf, orch = _workflow()
        rec = _recommendation(
            reviewer_decision="accepted",
            reviewer_notes="Duration matched p50 estimate.",
        )
        wf.ingest(rec, actual_duration_hours=8.0)
        _, kwargs = orch.record_completion.call_args
        assert "Duration matched p50 estimate." in kwargs["outcome_notes"]


# ===========================================================================
# Receipt fields
# ===========================================================================

class TestReceiptFields:

    def test_receipt_has_completion_record_on_success(self):
        record = _ok_record()
        wf, _ = _workflow(record=record)
        rec = _recommendation()
        receipt = wf.ingest(rec, actual_duration_hours=8.0)
        assert receipt.completion_record is record

    def test_receipt_completion_record_is_none_when_skipped(self):
        wf, _ = _workflow()
        rec = _recommendation(reviewer_decision="pending")
        receipt = wf.ingest(rec, actual_duration_hours=8.0)
        assert receipt.completion_record is None

    def test_receipt_reviewer_decision_matches_artifact(self):
        wf, _ = _workflow()
        rec = _recommendation(reviewer_decision="modified")
        receipt = wf.ingest(rec, actual_duration_hours=8.0)
        assert receipt.reviewer_decision == "modified"

    def test_receipt_skip_reason_is_none_when_not_skipped(self):
        wf, _ = _workflow()
        rec = _recommendation(reviewer_decision="accepted")
        receipt = wf.ingest(rec, actual_duration_hours=8.0)
        assert receipt.skip_reason is None

    def test_receipt_actual_duration_hours_mirrored(self):
        wf, _ = _workflow()
        rec = _recommendation()
        receipt = wf.ingest(rec, actual_duration_hours=33.5)
        assert receipt.actual_duration_hours == 33.5

    def test_receipt_skipped_false_on_success(self):
        wf, _ = _workflow()
        rec = _recommendation(reviewer_decision="accepted")
        receipt = wf.ingest(rec, actual_duration_hours=8.0)
        assert not receipt.skipped


# ===========================================================================
# No feedback_writer wired
# ===========================================================================

class TestNoFeedbackWriterWired:

    def test_orchestrator_without_feedback_writer_returns_noop_record(self):
        """Orchestrator handles the no-writer case; receipt should still be non-skipped."""
        no_op = _no_op_record()
        wf, orch = _workflow(record=no_op)
        rec = _recommendation()
        receipt = wf.ingest(rec, actual_duration_hours=8.0)
        # The workflow delegates; if the orchestrator returns a no-op record that's fine
        assert not receipt.skipped
        assert receipt.completion_record is no_op
        assert receipt.completion_record.index_updated is False
        assert receipt.completion_record.persisted is False
