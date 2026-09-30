"""
Post-Completion Feedback Ingestion Workflow — M5.

Closes the learning loop after an emergent activity completes in the field.
Accepts a Stage G recommendation artifact plus field-measured completion data
and writes the result back into the historical analog index via the
orchestrator's ``record_completion()`` method.

Why a workflow layer?
---------------------
``OutageActivityOrchestrator.record_completion()`` accepts loose keyword
arguments.  The Stage G artifact already carries ``activity_id``, ``run_id``,
and the analyst's verdict (``analyst_review.reviewer_decision``), so callers
should not have to re-extract those fields manually.  This workflow:

1. Extracts the relevant fields from the recommendation artifact.
2. Routes based on ``reviewer_decision`` — accepted, rejected, modified,
   pending, or absent.
3. Prefixes ``outcome_notes`` with a rejection marker when the analyst
   rejected the recommendation, so the data is still stored for negative
   learning without silently polluting the index.
4. Delegates to ``orchestrator.record_completion()`` and wraps the result
   in an :class:`IngestionReceipt`.

Rejection storage rationale
----------------------------
When an analyst rejects a recommendation, the actual field duration is still
valid historical data.  The rejection reflects recommendation quality, not
activity outcome.  We therefore write the record with an
``RECOMMENDATION_REJECTED: <reviewer_notes>`` prefix in ``outcome_notes`` so
future Stage D retrievals can identify and optionally down-weight it.  Skipping
write-back on rejection would starve the index of real-world data from the very
activities that were hard to predict.

Usage::

    from workflows.completion_feedback_workflow import CompletionFeedbackWorkflow

    workflow = CompletionFeedbackWorkflow(orchestrator=orchestrator)
    receipt = workflow.ingest(
        recommendation=stage_g_artifact,
        actual_duration_hours=16.2,
        actual_start="2026-04-12T08:00:00Z",
        actual_finish="2026-04-13T00:12:00Z",
        outcome_notes="Packing replaced; no scope expansion.",
    )
    # receipt.skipped          → False
    # receipt.completion_record.index_updated → True
    # receipt.completion_record.persisted     → True
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)

JsonDict = Dict[str, Any]

# Sentinel prefix written to outcome_notes for rejected recommendations.
# Downstream consumers can filter on this prefix when training or querying.
_REJECTION_PREFIX = "RECOMMENDATION_REJECTED"

# Reviewer decisions that trigger a write-back (everything except "pending").
_WRITE_DECISIONS = {"accepted", "rejected", "modified", None}


# ---------------------------------------------------------------------------
# IngestionReceipt
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IngestionReceipt:
    """Immutable receipt for one post-completion ingestion operation.

    Attributes
    ----------
    activity_id
        The emergent activity that completed.
    run_id
        The pipeline run that produced the original recommendation.
    reviewer_decision
        The analyst's verdict extracted from the recommendation artifact.
        ``None`` when no analyst review was performed (review not required).
    actual_duration_hours
        Field-measured duration passed to the workflow.
    completion_record
        The :class:`~stages.completion_feedback.CompletionRecord` returned
        by the orchestrator.  ``None`` when the ingestion was skipped.
    skipped
        ``True`` when the workflow decided not to call ``record_completion()``
        (e.g., review still pending, or ``activity_id`` is missing).
    skip_reason
        Human-readable explanation for why ingestion was skipped.
        ``None`` when ``skipped=False``.
    """

    activity_id: str
    run_id: str
    reviewer_decision: Optional[str]
    actual_duration_hours: float
    completion_record: Optional[Any]   # CompletionRecord | None
    skipped: bool
    skip_reason: Optional[str]


# ---------------------------------------------------------------------------
# CompletionFeedbackWorkflow
# ---------------------------------------------------------------------------

class CompletionFeedbackWorkflow:
    """Bridge between a Stage G recommendation artifact and the feedback loop.

    Args:
        orchestrator: An ``OutageActivityOrchestrator`` instance (or any object
                      with a ``record_completion(**kwargs)`` method).  The
                      orchestrator must have a ``feedback_writer`` injected to
                      actually update the index and persist to CSV; without it
                      the orchestrator logs a WARNING and returns a no-op
                      ``CompletionRecord``.
    """

    def __init__(self, orchestrator: Any) -> None:
        self._orchestrator = orchestrator

    # ── Public entry point ────────────────────────────────────────────────────

    def ingest(
        self,
        recommendation: JsonDict,
        actual_duration_hours: float,
        *,
        actual_start: Optional[str] = None,
        actual_finish: Optional[str] = None,
        outcome_notes: str = "",
    ) -> IngestionReceipt:
        """Ingest a completed Stage G recommendation into the analog index.

        Args:
            recommendation: The full Stage G artifact dict
                            (``outage_activity_recommendation.json`` shape).
            actual_duration_hours: Field-measured execution duration.  Must be
                                   > 0; otherwise ``CompletionFeedbackWriter``
                                   will skip the write and record a validation
                                   warning.
            actual_start: ISO-8601 field start timestamp (optional).
            actual_finish: ISO-8601 field finish timestamp (optional).
            outcome_notes: Caller-supplied free-text notes (optional).
                           When ``reviewer_decision == "rejected"`` these are
                           appended after the rejection prefix.

        Returns:
            :class:`IngestionReceipt` with the outcome of the ingestion.
        """
        activity_id: str = recommendation.get("activity_id", "") or ""
        run_id: str = recommendation.get("run_id", "") or ""

        # ── Guard: missing activity_id ────────────────────────────────────────
        if not activity_id:
            LOGGER.warning(
                "CompletionFeedbackWorkflow.ingest: recommendation has no "
                "activity_id — ingestion skipped (run_id=%s)", run_id,
            )
            return IngestionReceipt(
                activity_id=activity_id,
                run_id=run_id,
                reviewer_decision=None,
                actual_duration_hours=actual_duration_hours,
                completion_record=None,
                skipped=True,
                skip_reason="missing activity_id",
            )

        # ── Extract analyst review fields ─────────────────────────────────────
        analyst_review: JsonDict = recommendation.get("analyst_review") or {}
        reviewer_decision: Optional[str] = analyst_review.get("reviewer_decision")
        reviewer_notes: str = analyst_review.get("reviewer_notes") or ""

        # ── Guard: pending review ─────────────────────────────────────────────
        if reviewer_decision == "pending":
            LOGGER.info(
                "CompletionFeedbackWorkflow.ingest: analyst review is pending "
                "for activity %s (run=%s) — ingestion deferred", activity_id, run_id,
            )
            return IngestionReceipt(
                activity_id=activity_id,
                run_id=run_id,
                reviewer_decision=reviewer_decision,
                actual_duration_hours=actual_duration_hours,
                completion_record=None,
                skipped=True,
                skip_reason="analyst review pending",
            )

        # ── Warn on unrecognised decision (treat as accepted) ─────────────────
        if reviewer_decision not in _WRITE_DECISIONS:
            LOGGER.warning(
                "CompletionFeedbackWorkflow.ingest: unrecognised "
                "reviewer_decision=%r for activity %s — treating as accepted",
                reviewer_decision, activity_id,
            )

        # ── Build final outcome_notes ─────────────────────────────────────────
        final_notes = self._build_outcome_notes(
            reviewer_decision=reviewer_decision,
            reviewer_notes=reviewer_notes,
            caller_notes=outcome_notes,
        )

        # ── Delegate to orchestrator ──────────────────────────────────────────
        LOGGER.debug(
            "CompletionFeedbackWorkflow.ingest: calling record_completion for "
            "activity=%s run=%s reviewer_decision=%r duration=%.2fh",
            activity_id, run_id, reviewer_decision, actual_duration_hours,
        )
        completion_record = self._orchestrator.record_completion(
            activity_id=activity_id,
            run_id=run_id,
            actual_duration_hours=actual_duration_hours,
            actual_start=actual_start,
            actual_finish=actual_finish,
            outcome_notes=final_notes,
        )

        return IngestionReceipt(
            activity_id=activity_id,
            run_id=run_id,
            reviewer_decision=reviewer_decision,
            actual_duration_hours=actual_duration_hours,
            completion_record=completion_record,
            skipped=False,
            skip_reason=None,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_outcome_notes(
        reviewer_decision: Optional[str],
        reviewer_notes: str,
        caller_notes: str,
    ) -> str:
        """Compose the final outcome_notes string.

        - ``rejected`` → ``"RECOMMENDATION_REJECTED: {reviewer_notes}\\n{caller_notes}"``
        - all other decisions → ``"{caller_notes}\\n{reviewer_notes}"`` (blank lines stripped)
        """
        if reviewer_decision == "rejected":
            prefix = _REJECTION_PREFIX
            if reviewer_notes:
                prefix = f"{_REJECTION_PREFIX}: {reviewer_notes}"
            parts = [prefix]
            if caller_notes:
                parts.append(caller_notes)
            return "\n".join(parts)

        # accepted / modified / None / unrecognised
        parts = [p for p in (caller_notes, reviewer_notes) if p]
        return "\n".join(parts)
