"""
Completion feedback loop for the outage analytics pipeline.

After an emergent activity completes in the field, its actual duration must
be written back into the historical analog index so future Stage D retrievals
use it.  Without this loop the analog index grows only by manual data entry
and each outage cycle repeats the same estimation errors.

This module provides:

    CompletionRecord          — immutable receipt for one write-back operation
    CompletionFeedbackWriter  — coordinates index update + persistence
    CsvAnalogPersister        — CSV write-through backend

Typical wiring::

    from stages.completion_feedback import CompletionFeedbackWriter, CsvAnalogPersister

    writer = CompletionFeedbackWriter(
        index=analog_index,          # same HistoricalActivityIndex used by Stage D
        persister=CsvAnalogPersister("/data/analogs/activities.csv"),
    )

    result = writer.record_completion(
        activity_id="ACT-20260412-001",
        run_id="OUTAGE::abc123",
        actual_duration_hours=16.2,
        actual_start="2026-04-12T08:00:00Z",
        actual_finish="2026-04-13T00:12:00Z",
        outcome_notes="Packing replaced; no scope expansion.",
    )
    # result.index_updated → True   (in-memory index hot-updated)
    # result.persisted     → True   (row written to CSV)

Both operations are best-effort: failures are logged and reflected in
``CompletionRecord`` fields rather than raised.  This prevents completion
recording from blocking the primary pipeline.
"""
from __future__ import annotations

import copy
import csv
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(iso_str: str) -> Optional[datetime]:
    """Parse ISO-8601 string to a timezone-aware datetime; None on failure."""
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# CompletionRecord
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompletionRecord:
    """Immutable receipt for one analog feedback write-back operation.

    Attributes
    ----------
    activity_id
        The emergent activity that completed.
    run_id
        The pipeline run that produced the original recommendation.
    actual_duration_hours
        Observed field duration.
    actual_start, actual_finish
        Field-reported ISO-8601 timestamps.
    outcome_notes
        Free-text notes from the operator or outage manager.
    written_at
        UTC timestamp when ``record_completion()`` was called.
    index_updated
        True if the in-memory ``HistoricalActivityIndex`` was successfully
        updated.  False when no index was injected, or upsert raised.
    persisted
        True if the ``AnalogPersister`` successfully stored the record.
        False when no persister was injected, or append raised.
    validation_warnings
        Non-fatal issues detected during write-back (e.g. duration <= 0).
        When non-empty, neither ``index_updated`` nor ``persisted`` will be
        True for that specific call.
    """

    activity_id: str
    run_id: str
    actual_duration_hours: float
    actual_start: Optional[str]
    actual_finish: Optional[str]
    outcome_notes: str
    written_at: str
    index_updated: bool
    persisted: bool
    validation_warnings: tuple  # tuple[str, ...] — frozen dataclass requires tuple


# ---------------------------------------------------------------------------
# CompletionFeedbackWriter
# ---------------------------------------------------------------------------

class CompletionFeedbackWriter:
    """Write actual completion data back to the historical analog index.

    Coordinates two operations on every successful write-back:

    1. ``index.upsert(updated_activity)`` — hot-updates the in-memory
       ``HistoricalActivityIndex`` so the current session's Stage D retrievals
       immediately benefit from the real execution time.

    2. ``persister.append(updated_activity)`` — writes the record to the
       backing store so the *next* outage cycle's ``build()`` call includes
       this activity with its actual duration.

    Both operations are best-effort: failures are logged as warnings and
    reflected in ``CompletionRecord.index_updated`` / ``.persisted`` rather
    than raised.

    Args:
        index: The ``HistoricalActivityIndex`` shared with Stage D.  When
               ``None`` the in-memory update is skipped (``index_updated=False``).
        persister: Persistence backend.  ``NoOpAnalogPersister`` (from
                   ``orchestrators.protocols``) is the safe default for tests.
                   When ``None`` the persistence step is skipped.
    """

    def __init__(self, index: Any = None, persister: Any = None) -> None:
        self._index = index
        self._persister = persister

    # ── Public entry point ────────────────────────────────────────────────────

    def record_completion(
        self,
        *,
        activity_id: str,
        run_id: str,
        actual_duration_hours: float,
        actual_start: Optional[str] = None,
        actual_finish: Optional[str] = None,
        outcome_notes: str = "",
        outage_id: Optional[str] = None,
        plant_id: Optional[str] = None,
    ) -> CompletionRecord:
        """Record that an emergent activity has completed in the field.

        Args:
            activity_id: The activity that finished.
            run_id: The pipeline run that analysed this activity.
            actual_duration_hours: Observed field duration (must be > 0).
            actual_start: ISO-8601 field start timestamp (optional).
            actual_finish: ISO-8601 field finish timestamp (optional).
            outcome_notes: Free-text operator notes (optional).
            outage_id: Used when reconstructing an unknown activity.
            plant_id: Used when reconstructing an unknown activity.

        Returns:
            :class:`CompletionRecord` with write-back status and any
            validation warnings.
        """
        warnings: list[str] = []
        written_at = _utcnow_iso()

        # ── Validation ────────────────────────────────────────────────────────
        if not activity_id:
            warnings.append("activity_id is empty — write-back skipped")
            LOGGER.warning("CompletionFeedbackWriter: activity_id is empty; skipping")
            return self._make_record(
                activity_id="", run_id=run_id,
                actual_duration_hours=actual_duration_hours,
                actual_start=actual_start, actual_finish=actual_finish,
                outcome_notes=outcome_notes, written_at=written_at,
                index_updated=False, persisted=False, warnings=warnings,
            )

        if actual_duration_hours <= 0:
            msg = (
                f"actual_duration_hours={actual_duration_hours:.4f} is not positive; "
                "write-back skipped to prevent corrupting the analog index"
            )
            warnings.append(msg)
            LOGGER.warning("CompletionFeedbackWriter: %s", msg)
            return self._make_record(
                activity_id=activity_id, run_id=run_id,
                actual_duration_hours=actual_duration_hours,
                actual_start=actual_start, actual_finish=actual_finish,
                outcome_notes=outcome_notes, written_at=written_at,
                index_updated=False, persisted=False, warnings=warnings,
            )

        # ── Build updated ActivityCase ────────────────────────────────────────
        updated = self._build_updated_activity(
            activity_id=activity_id,
            actual_duration_hours=actual_duration_hours,
            actual_start=actual_start,
            actual_finish=actual_finish,
            outcome_notes=outcome_notes,
            outage_id=outage_id,
            plant_id=plant_id,
        )

        # ── Hot-update the in-memory index ────────────────────────────────────
        index_updated = False
        if self._index is not None:
            try:
                self._index.upsert(updated)
                index_updated = True
                LOGGER.debug(
                    "CompletionFeedbackWriter: index updated for %s "
                    "(actual_duration_hours=%.2f h)",
                    activity_id, actual_duration_hours,
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"index.upsert() failed: {exc}"
                warnings.append(msg)
                LOGGER.warning("CompletionFeedbackWriter: %s", msg)

        # ── Persist to backing store ──────────────────────────────────────────
        persisted = False
        if self._persister is not None:
            try:
                self._persister.append(updated)
                persisted = True
                LOGGER.debug(
                    "CompletionFeedbackWriter: persisted activity %s", activity_id
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"persister.append() failed: {exc}"
                warnings.append(msg)
                LOGGER.warning("CompletionFeedbackWriter: %s", msg)

        return self._make_record(
            activity_id=activity_id, run_id=run_id,
            actual_duration_hours=actual_duration_hours,
            actual_start=actual_start, actual_finish=actual_finish,
            outcome_notes=outcome_notes, written_at=written_at,
            index_updated=index_updated, persisted=persisted,
            warnings=warnings,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_updated_activity(
        self,
        *,
        activity_id: str,
        actual_duration_hours: float,
        actual_start: Optional[str],
        actual_finish: Optional[str],
        outcome_notes: str,
        outage_id: Optional[str],
        plant_id: Optional[str],
    ) -> Any:
        """Return an ActivityCase with completion fields stamped.

        If the activity is already in the index, all existing fields are
        preserved and only the completion fields are overwritten.  If it is
        not in the index (or no index is injected), a minimal ActivityCase
        is constructed from the provided kwargs so the persister can still
        store it for the next ``build()`` cycle.
        """
        try:
            from dackar.outage.outage_uncertainty.domain.activity import ActivityCase
        except ImportError:
            from outage_uncertainty.domain.activity import ActivityCase  # type: ignore[no-redef]

        # Try to retrieve the existing record from the index
        existing = None
        if self._index is not None:
            try:
                existing = self._index.get(activity_id)
            except Exception:  # noqa: BLE001
                pass  # not found or index error — reconstruct below

        if existing is not None:
            # Shallow-copy the existing ActivityCase and overwrite completion fields
            updated = copy.copy(existing)
            updated.actual_duration_hours = actual_duration_hours
            if actual_start:
                updated.actual_start = _parse_iso(actual_start)
            if actual_finish:
                updated.actual_finish = _parse_iso(actual_finish)
            if outcome_notes:
                meta = dict(updated.metadata)
                meta["outcome_notes"] = outcome_notes
                updated.metadata = meta
            return updated

        # Minimal reconstruction for unknown activities
        LOGGER.debug(
            "CompletionFeedbackWriter: %s not found in index; "
            "constructing minimal ActivityCase for persistence",
            activity_id,
        )
        meta: dict = {}
        if outcome_notes:
            meta["outcome_notes"] = outcome_notes
        return ActivityCase(
            activity_id=activity_id,
            outage_id=outage_id or "",
            plant_id=plant_id or "",
            actual_duration_hours=actual_duration_hours,
            actual_start=_parse_iso(actual_start) if actual_start else None,
            actual_finish=_parse_iso(actual_finish) if actual_finish else None,
            is_emergent=True,
            metadata=meta,
        )

    @staticmethod
    def _make_record(
        *,
        activity_id: str,
        run_id: str,
        actual_duration_hours: float,
        actual_start: Optional[str],
        actual_finish: Optional[str],
        outcome_notes: str,
        written_at: str,
        index_updated: bool,
        persisted: bool,
        warnings: list,
    ) -> CompletionRecord:
        return CompletionRecord(
            activity_id=activity_id,
            run_id=run_id,
            actual_duration_hours=actual_duration_hours,
            actual_start=actual_start,
            actual_finish=actual_finish,
            outcome_notes=outcome_notes,
            written_at=written_at,
            index_updated=index_updated,
            persisted=persisted,
            validation_warnings=tuple(warnings),
        )


# ---------------------------------------------------------------------------
# CsvAnalogPersister
# ---------------------------------------------------------------------------

# Canonical column order for the CSV backing store.  Must match the fields
# that PandasActivityRepository.load_activities() reads so that persisted rows
# are immediately usable on the next index build() call.
_CSV_COLUMNS: tuple = (
    "activity_id", "outage_id", "plant_id", "unit_id",
    "raw_description", "cleaned_description",
    "planned_duration_hours", "actual_duration_hours",
    "actual_start", "actual_finish",
    "discipline", "task_family", "component_family",
    "system_name", "work_order_type",
    "is_emergent", "is_rework",
    "has_rp_hold", "requires_scaffold", "has_clearance", "is_vendor_supported",
    "crew_size", "outage_phase",
)


class CsvAnalogPersister:
    """Append completed ActivityCase records to a CSV file.

    The file is created (with a header row) if it does not yet exist.  Each
    ``append()`` call adds one row.  Column order follows ``_CSV_COLUMNS``
    so that rows written across sessions are compatible and can be loaded
    directly by :class:`~outage_uncertainty.adapters.pandas_repository.PandasActivityRepository`.

    Args:
        filepath: Path to the CSV file.  The parent directory must exist.
    """

    def __init__(self, filepath: Any) -> None:
        self._path = str(filepath)

    def append(self, activity: Any) -> None:
        """Append one row to the CSV file, creating it with a header if needed."""
        row = {col: getattr(activity, col, None) for col in _CSV_COLUMNS}

        # Serialise datetime fields to ISO strings
        for dt_field in ("actual_start", "actual_finish"):
            val = row.get(dt_field)
            if val is not None and hasattr(val, "isoformat"):
                row[dt_field] = val.isoformat()

        needs_header = (
            not os.path.exists(self._path)
            or os.path.getsize(self._path) == 0
        )
        with open(self._path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(_CSV_COLUMNS))
            if needs_header:
                writer.writeheader()
            writer.writerow(row)

        LOGGER.debug(
            "CsvAnalogPersister: appended activity %s to %s",
            getattr(activity, "activity_id", "?"), self._path,
        )
