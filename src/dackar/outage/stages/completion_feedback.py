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

    from dackar.outage.stages.completion_feedback import CompletionFeedbackWriter, CsvAnalogPersister

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
import math
import numbers
import os
import time
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


def _valid_duration(v: Any) -> bool:
    """True when *v* is a positive, finite real number.

    Uses numbers.Real (not int/float) so numpy scalars from a pandas column
    (np.int64 / np.float32) are accepted — ``isinstance(np.int64(2), int)`` is
    False.  bool is rejected (it is a Real but not a duration).  math.isfinite
    on a huge Python int raises OverflowError, so the float conversion is
    guarded to preserve the never-raises contract.
    """
    if not isinstance(v, numbers.Real) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(float(v)) and float(v) > 0
    except (OverflowError, ValueError, TypeError):
        return False


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

        # B6: reject non-numeric, non-finite (NaN/inf), and non-positive values.
        # `nan <= 0` is False (so NaN slipped past the old check and was written
        # to the index and CSV), and `None` crashed the `:.4f` format, breaking
        # the "never raises" contract.  _valid_duration also accepts numpy
        # scalars (np.int64/np.float32 from a pandas column) and survives a huge
        # Python int (math.isfinite OverflowError) — PR#79 [3].
        if not _valid_duration(actual_duration_hours):
            msg = (
                f"actual_duration_hours={actual_duration_hours!r} is not a positive "
                "finite number; write-back skipped to prevent corrupting the analog index"
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
        from dackar.outage.outage_uncertainty.domain.activity import ActivityCase

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
            # I16: only overwrite when the incoming timestamp parses — an
            # unparseable string must not wipe a valid existing actual_start/
            # actual_finish with None.
            if actual_start:
                parsed_start = _parse_iso(actual_start)
                if parsed_start is not None:
                    updated.actual_start = parsed_start
                else:
                    LOGGER.warning(
                        "CompletionFeedbackWriter: unparseable actual_start %r for %s; "
                        "keeping existing value", actual_start, activity_id,
                    )
            if actual_finish:
                parsed_finish = _parse_iso(actual_finish)
                if parsed_finish is not None:
                    updated.actual_finish = parsed_finish
                else:
                    LOGGER.warning(
                        "CompletionFeedbackWriter: unparseable actual_finish %r for %s; "
                        "keeping existing value", actual_finish, activity_id,
                    )
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


class _FileLock:
    """Minimal cross-process advisory lock via an exclusive lock file.

    Standard-library only (no fcntl dependency, so it also works on Windows).
    Acquisition retries until ``timeout`` seconds elapse.  On timeout, a lock
    older than ``stale_after`` seconds is assumed to belong to a crashed writer:
    it is unlinked and the exclusive open is retried once, so a leftover lock
    cannot permanently defeat mutual exclusion (PR#79 [2]).  Only if the lock is
    fresh (another writer is genuinely active) do we proceed without it, logging
    a warning, rather than blocking the "never raises" write-back path.
    """

    def __init__(self, lock_path: str, timeout: float = 10.0,
                 stale_after: float = 30.0) -> None:
        self._lock_path = lock_path
        self._timeout = timeout
        self._stale_after = stale_after
        self._fd: Optional[int] = None

    def _try_open(self) -> bool:
        try:
            self._fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            return True
        except FileExistsError:
            return False

    def __enter__(self) -> "_FileLock":
        deadline = time.monotonic() + self._timeout
        while True:
            if self._try_open():
                return self
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)

        # Timed out.  If the existing lock is stale (crashed holder), take it
        # over: unlink and retry the exclusive open once.
        try:
            age = time.time() - os.path.getmtime(self._lock_path)
        except OSError:
            age = None
        if age is not None and age >= self._stale_after:
            LOGGER.warning(
                "CsvAnalogPersister: lock %s is stale (%.0fs old); taking it over.",
                self._lock_path, age,
            )
            try:
                os.unlink(self._lock_path)
            except OSError:
                pass
            if self._try_open():
                return self

        # Lock is fresh (another writer is active) or could not be reclaimed:
        # proceed without it rather than blocking the write-back.
        LOGGER.warning(
            "CsvAnalogPersister: lock %s held past %.0fs; proceeding without it.",
            self._lock_path, self._timeout,
        )
        self._fd = None
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                try:
                    os.unlink(self._lock_path)
                except OSError:
                    pass


class CsvAnalogPersister:
    """Upsert completed ActivityCase records into a CSV analog store.

    ``append()`` is an idempotent upsert keyed by ``activity_id``: re-ingesting
    the same id replaces that row rather than adding a duplicate (so a double
    ingest does not grow the file, and the last write wins on reload).  The file
    is rewritten whole via a temp file + ``os.replace`` (atomic), so a crash
    mid-write cannot leave a partial or header-less file.

    Column handling: the rewritten header is the UNION of the existing file's
    columns and ``_CSV_COLUMNS`` (existing order first), so extra columns in a
    seeded historical file — e.g. ``contractor_flag``, which
    ``PandasActivityRepository.load_activities()`` reads — are preserved, not
    dropped (PR#79 [0]).

    Side-effect files (same directory as the CSV): a ``<path>.lock`` exclusive
    lock serialises concurrent writers (see :class:`_FileLock`), and a transient
    ``<path>.tmp.<pid>`` holds the new content before the atomic rename.

    If an existing file cannot be read (malformed CSV, oversized field,
    encoding or OS error), ``append()`` refuses to overwrite it and raises, so
    the caller reports ``persisted=False`` rather than silently replacing the
    whole history with one row (PR#79 [1]).

    Args:
        filepath: Path to the CSV file.  The parent directory must exist.
    """

    def __init__(self, filepath: Any) -> None:
        self._path = str(filepath)

    def append(self, activity: Any) -> None:
        """Upsert one activity row by activity_id (see class docstring)."""
        row = {col: getattr(activity, col, None) for col in _CSV_COLUMNS}
        for dt_field in ("actual_start", "actual_finish"):
            val = row.get(dt_field)
            if val is not None and hasattr(val, "isoformat"):
                row[dt_field] = val.isoformat()

        activity_id = row.get("activity_id")

        with _FileLock(self._path + ".lock"):
            # Load existing rows and their header.  A file that exists but cannot
            # be read must NOT be silently replaced with a single-row file
            # (PR#79 [1]): re-raise so record_completion reports persisted=False.
            existing: list = []
            existing_fields: list = []
            if os.path.exists(self._path) and os.path.getsize(self._path) > 0:
                try:
                    with open(self._path, newline="", encoding="utf-8") as fh:
                        reader = csv.DictReader(fh)
                        existing = list(reader)
                        existing_fields = list(reader.fieldnames or [])
                except (OSError, csv.Error, UnicodeDecodeError):
                    LOGGER.warning(
                        "CsvAnalogPersister: could not read %s; refusing to "
                        "overwrite existing analog history",
                        self._path, exc_info=True,
                    )
                    raise

            # Upsert by activity_id (dedup); append when the id is new or absent.
            replaced = False
            if activity_id:
                for i, r in enumerate(existing):
                    if r.get("activity_id") == activity_id:
                        existing[i] = {**r, **row}  # preserve extra cols on the row
                        replaced = True
                        break
            if not replaced:
                existing.append(row)

            # Header = existing columns first, then any _CSV_COLUMNS not already
            # present — preserves extra columns in a seeded file (PR#79 [0]).
            fieldnames = existing_fields + [
                c for c in _CSV_COLUMNS if c not in existing_fields
            ]

            # Atomic rewrite: temp file in the same dir, then os.replace; fsync
            # the file and the directory so the rename is durable (PR#79 [5]).
            # Unlink the temp file if the write fails, leaving no .tmp litter
            # (PR#79 [6]).
            directory = os.path.dirname(self._path) or "."
            tmp_path = f"{self._path}.tmp.{os.getpid()}"
            try:
                with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    for r in existing:
                        writer.writerow({col: r.get(col) for col in fieldnames})
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, self._path)
                self._fsync_dir(directory)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        LOGGER.debug(
            "CsvAnalogPersister: %s activity %s in %s",
            "replaced" if replaced else "appended",
            activity_id or "?", self._path,
        )

    @staticmethod
    def _fsync_dir(directory: str) -> None:
        """fsync a directory so a rename into it is durable on POSIX (best effort)."""
        try:
            dfd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dfd)
        except OSError:
            pass  # not supported on this platform (e.g. Windows)
        finally:
            os.close(dfd)
