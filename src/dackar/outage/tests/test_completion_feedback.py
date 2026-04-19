"""
Unit tests for the completion feedback loop.

Coverage targets:
    HistoricalActivityIndex.upsert()
        — update existing activity in place
        — add new activity
        — ValueError on empty activity_id
        — upserted activity visible to search() immediately

    CompletionRecord
        — frozen dataclass (cannot mutate fields)
        — validation_warnings is a tuple

    CompletionFeedbackWriter.record_completion()
        Validation:
        — zero duration → warnings, no write-back
        — negative duration → warnings, no write-back
        — empty activity_id → warnings, no write-back
        Index update:
        — existing activity updated, index_updated=True
        — actual_duration_hours written to index entry
        — no index injected → index_updated=False
        Persistence:
        — NoOpAnalogPersister.appended receives the activity, persisted=True
        — no persister injected → persisted=False
        Graceful degradation:
        — index.upsert() raises → persisted attempt still made
        — persister.append() raises → persisted=False, no exception propagated
        Existing activity copy:
        — existing fields preserved, completion fields overwritten
        — metadata outcome_notes merged

    CompletionFeedbackWriter unknown activity reconstruction:
        — no index → minimal ActivityCase passed to persister
        — index has no matching id → minimal ActivityCase constructed

    CsvAnalogPersister.append()
        — creates file with header on first call
        — correct columns in header
        — appends a data row on second call (no duplicate header)

    NoOpAnalogPersister
        — accumulates activities in .appended
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

_OUTAGE_ROOT = Path(__file__).parent.parent
if str(_OUTAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTAGE_ROOT))

from outage_uncertainty.retrieval.retrieval_index import HistoricalActivityIndex
from outage_uncertainty.domain.activity import ActivityCase
from stages.completion_feedback import (
    CompletionRecord,
    CompletionFeedbackWriter,
    CsvAnalogPersister,
    _CSV_COLUMNS,
)
from orchestrators.protocols import NoOpAnalogPersister


# ===========================================================================
# Helpers
# ===========================================================================

def _activity(
    activity_id: str = "ACT-001",
    outage_id: str = "R2026",
    plant_id: str = "PLANT-A",
    actual_duration_hours: float | None = 8.0,
    **kwargs,
) -> ActivityCase:
    return ActivityCase(
        activity_id=activity_id,
        outage_id=outage_id,
        plant_id=plant_id,
        actual_duration_hours=actual_duration_hours,
        **kwargs,
    )


def _writer(index=None, persister=None) -> CompletionFeedbackWriter:
    return CompletionFeedbackWriter(index=index, persister=persister)


def _index_with(*activities: ActivityCase) -> HistoricalActivityIndex:
    idx = HistoricalActivityIndex()
    idx.build(list(activities))
    return idx


# ===========================================================================
# HistoricalActivityIndex.upsert()
# ===========================================================================

class TestHistoricalActivityIndexUpsert:

    def test_upsert_updates_existing_in_place(self):
        act = _activity("ACT-001", actual_duration_hours=8.0)
        idx = _index_with(act)
        updated = _activity("ACT-001", actual_duration_hours=16.5)
        idx.upsert(updated)
        assert idx.get("ACT-001").actual_duration_hours == 16.5

    def test_upsert_adds_new_activity(self):
        idx = _index_with(_activity("ACT-001"))
        assert len(idx) == 1
        idx.upsert(_activity("ACT-NEW"))
        assert len(idx) == 2
        assert idx.get("ACT-NEW") is not None

    def test_upsert_empty_activity_id_raises_value_error(self):
        idx = HistoricalActivityIndex()
        with pytest.raises(ValueError, match="activity_id"):
            idx.upsert(_activity(activity_id=""))

    def test_upserted_activity_visible_to_search_immediately(self):
        act = _activity("ACT-001", raw_description="valve packing replacement")
        idx = _index_with(act)
        updated = _activity(
            "ACT-001",
            raw_description="valve packing replacement",
            actual_duration_hours=20.0,
        )
        idx.upsert(updated)
        # search still returns ACT-001 (prescorer is stateless)
        query = _activity("Q-000", raw_description="valve packing")
        candidates = idx.search(query, top_k=10)
        assert "ACT-001" in candidates

    def test_upsert_preserves_index_length_on_update(self):
        idx = _index_with(_activity("A1"), _activity("A2"), _activity("A3"))
        idx.upsert(_activity("A2", actual_duration_hours=99.0))
        assert len(idx) == 3


# ===========================================================================
# CompletionRecord
# ===========================================================================

class TestCompletionRecord:

    def _make(self, **kwargs) -> CompletionRecord:
        defaults = dict(
            activity_id="ACT-001",
            run_id="OUTAGE::abc",
            actual_duration_hours=8.0,
            actual_start=None,
            actual_finish=None,
            outcome_notes="",
            written_at="2026-04-16T00:00:00+00:00",
            index_updated=True,
            persisted=True,
            validation_warnings=(),
        )
        defaults.update(kwargs)
        return CompletionRecord(**defaults)

    def test_fields_accessible(self):
        rec = self._make()
        assert rec.activity_id == "ACT-001"
        assert rec.index_updated is True
        assert rec.persisted is True

    def test_frozen_cannot_mutate(self):
        rec = self._make()
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            rec.index_updated = False  # type: ignore[misc]

    def test_validation_warnings_is_tuple(self):
        rec = self._make(validation_warnings=("w1", "w2"))
        assert isinstance(rec.validation_warnings, tuple)
        assert rec.validation_warnings == ("w1", "w2")


# ===========================================================================
# CompletionFeedbackWriter — validation
# ===========================================================================

class TestCompletionFeedbackWriterValidation:

    def test_zero_duration_returns_warning_no_write(self):
        persister = NoOpAnalogPersister()
        w = _writer(persister=persister)
        rec = w.record_completion(
            activity_id="ACT-001",
            run_id="R1",
            actual_duration_hours=0.0,
        )
        assert rec.index_updated is False
        assert rec.persisted is False
        assert any("not positive" in msg for msg in rec.validation_warnings)
        assert len(persister.appended) == 0

    def test_negative_duration_returns_warning_no_write(self):
        persister = NoOpAnalogPersister()
        w = _writer(persister=persister)
        rec = w.record_completion(
            activity_id="ACT-001",
            run_id="R1",
            actual_duration_hours=-5.0,
        )
        assert rec.index_updated is False
        assert rec.persisted is False
        assert len(rec.validation_warnings) > 0

    def test_empty_activity_id_returns_warning_no_write(self):
        persister = NoOpAnalogPersister()
        w = _writer(persister=persister)
        rec = w.record_completion(
            activity_id="",
            run_id="R1",
            actual_duration_hours=8.0,
        )
        assert rec.index_updated is False
        assert rec.persisted is False
        assert any("empty" in msg.lower() for msg in rec.validation_warnings)
        assert len(persister.appended) == 0

    def test_valid_call_produces_no_validation_warnings(self):
        persister = NoOpAnalogPersister()
        w = _writer(persister=persister)
        rec = w.record_completion(
            activity_id="ACT-001",
            run_id="R1",
            actual_duration_hours=8.0,
        )
        assert rec.validation_warnings == ()


# ===========================================================================
# CompletionFeedbackWriter — index update
# ===========================================================================

class TestCompletionFeedbackWriterIndexUpdate:

    def test_index_updated_true_for_existing_activity(self):
        act = _activity("ACT-001", actual_duration_hours=8.0)
        idx = _index_with(act)
        w = _writer(index=idx)
        rec = w.record_completion(
            activity_id="ACT-001",
            run_id="R1",
            actual_duration_hours=16.0,
        )
        assert rec.index_updated is True

    def test_actual_duration_written_to_index(self):
        act = _activity("ACT-001", actual_duration_hours=8.0)
        idx = _index_with(act)
        w = _writer(index=idx)
        w.record_completion(
            activity_id="ACT-001",
            run_id="R1",
            actual_duration_hours=16.0,
        )
        assert idx.get("ACT-001").actual_duration_hours == 16.0

    def test_no_index_injected_returns_index_updated_false(self):
        w = _writer()
        rec = w.record_completion(
            activity_id="ACT-001",
            run_id="R1",
            actual_duration_hours=8.0,
        )
        assert rec.index_updated is False

    def test_unknown_activity_added_to_index_via_upsert(self):
        idx = _index_with(_activity("ACT-OTHER"))
        w = _writer(index=idx)
        rec = w.record_completion(
            activity_id="ACT-NEW",
            run_id="R1",
            actual_duration_hours=5.0,
        )
        assert rec.index_updated is True
        assert idx.get("ACT-NEW").actual_duration_hours == 5.0


# ===========================================================================
# CompletionFeedbackWriter — persistence
# ===========================================================================

class TestCompletionFeedbackWriterPersistence:

    def test_persister_receives_activity_persisted_true(self):
        persister = NoOpAnalogPersister()
        w = _writer(persister=persister)
        rec = w.record_completion(
            activity_id="ACT-001",
            run_id="R1",
            actual_duration_hours=8.0,
        )
        assert rec.persisted is True
        assert len(persister.appended) == 1
        assert persister.appended[0].activity_id == "ACT-001"

    def test_no_persister_returns_persisted_false(self):
        w = _writer()
        rec = w.record_completion(
            activity_id="ACT-001",
            run_id="R1",
            actual_duration_hours=8.0,
        )
        assert rec.persisted is False

    def test_persisted_activity_has_correct_duration(self):
        persister = NoOpAnalogPersister()
        w = _writer(persister=persister)
        w.record_completion(
            activity_id="ACT-001",
            run_id="R1",
            actual_duration_hours=12.5,
        )
        assert persister.appended[0].actual_duration_hours == 12.5


# ===========================================================================
# CompletionFeedbackWriter — graceful degradation
# ===========================================================================

class _FailingIndex:
    """Stub that raises on every method call."""

    def get(self, _):
        raise RuntimeError("index unavailable")

    def upsert(self, _):
        raise RuntimeError("index upsert failed")


class _FailingPersister:
    def append(self, _):
        raise IOError("disk full")


class TestCompletionFeedbackWriterGracefulDegradation:

    def test_failing_index_still_attempts_persister(self):
        persister = NoOpAnalogPersister()
        w = _writer(index=_FailingIndex(), persister=persister)
        rec = w.record_completion(
            activity_id="ACT-001",
            run_id="R1",
            actual_duration_hours=8.0,
        )
        assert rec.index_updated is False
        assert rec.persisted is True          # persister still ran
        assert len(persister.appended) == 1

    def test_failing_index_adds_warning(self):
        w = _writer(index=_FailingIndex())
        rec = w.record_completion(
            activity_id="ACT-001",
            run_id="R1",
            actual_duration_hours=8.0,
        )
        assert any("upsert" in msg.lower() for msg in rec.validation_warnings)

    def test_failing_persister_returns_persisted_false_no_exception(self):
        w = _writer(persister=_FailingPersister())
        rec = w.record_completion(
            activity_id="ACT-001",
            run_id="R1",
            actual_duration_hours=8.0,
        )
        assert rec.persisted is False
        assert any("append" in msg.lower() for msg in rec.validation_warnings)

    def test_both_failing_still_returns_record(self):
        w = _writer(index=_FailingIndex(), persister=_FailingPersister())
        rec = w.record_completion(
            activity_id="ACT-001",
            run_id="R1",
            actual_duration_hours=8.0,
        )
        assert isinstance(rec, CompletionRecord)
        assert rec.index_updated is False
        assert rec.persisted is False
        assert len(rec.validation_warnings) >= 2


# ===========================================================================
# CompletionFeedbackWriter — existing activity copy behaviour
# ===========================================================================

class TestCompletionFeedbackWriterExistingActivityCopy:

    def test_existing_fields_preserved_on_copy(self):
        act = _activity(
            "ACT-001",
            outage_id="R2026",
            plant_id="PLANT-A",
            discipline="Mechanical",
            actual_duration_hours=8.0,
        )
        idx = _index_with(act)
        persister = NoOpAnalogPersister()
        w = _writer(index=idx, persister=persister)
        w.record_completion(
            activity_id="ACT-001",
            run_id="R1",
            actual_duration_hours=16.0,
        )
        stored = persister.appended[0]
        assert stored.discipline == "Mechanical"
        assert stored.outage_id == "R2026"

    def test_actual_duration_overwritten(self):
        act = _activity("ACT-001", actual_duration_hours=8.0)
        idx = _index_with(act)
        persister = NoOpAnalogPersister()
        w = _writer(index=idx, persister=persister)
        w.record_completion(
            activity_id="ACT-001",
            run_id="R1",
            actual_duration_hours=24.0,
        )
        assert persister.appended[0].actual_duration_hours == 24.0

    def test_outcome_notes_merged_into_metadata(self):
        act = _activity("ACT-001", actual_duration_hours=8.0)
        idx = _index_with(act)
        persister = NoOpAnalogPersister()
        w = _writer(index=idx, persister=persister)
        w.record_completion(
            activity_id="ACT-001",
            run_id="R1",
            actual_duration_hours=16.0,
            outcome_notes="Packing replaced; no scope expansion.",
        )
        stored = persister.appended[0]
        assert stored.metadata.get("outcome_notes") == "Packing replaced; no scope expansion."

    def test_actual_start_finish_parsed_and_set(self):
        act = _activity("ACT-001", actual_duration_hours=8.0)
        idx = _index_with(act)
        persister = NoOpAnalogPersister()
        w = _writer(index=idx, persister=persister)
        w.record_completion(
            activity_id="ACT-001",
            run_id="R1",
            actual_duration_hours=16.0,
            actual_start="2026-04-12T08:00:00Z",
            actual_finish="2026-04-13T00:12:00Z",
        )
        stored = persister.appended[0]
        assert stored.actual_start is not None
        assert stored.actual_finish is not None
        assert isinstance(stored.actual_start, datetime)


# ===========================================================================
# CompletionFeedbackWriter — unknown activity reconstruction
# ===========================================================================

class TestCompletionFeedbackWriterReconstruction:

    def test_no_index_builds_minimal_activity_for_persister(self):
        persister = NoOpAnalogPersister()
        w = _writer(persister=persister)
        w.record_completion(
            activity_id="ACT-UNKNOWN",
            run_id="R1",
            actual_duration_hours=4.0,
            outage_id="R2026",
            plant_id="PLANT-B",
        )
        stored = persister.appended[0]
        assert stored.activity_id == "ACT-UNKNOWN"
        assert stored.actual_duration_hours == 4.0
        assert stored.outage_id == "R2026"
        assert stored.plant_id == "PLANT-B"
        assert stored.is_emergent is True

    def test_unknown_id_in_non_empty_index_builds_minimal_activity(self):
        idx = _index_with(_activity("ACT-001"))
        persister = NoOpAnalogPersister()
        w = _writer(index=idx, persister=persister)
        w.record_completion(
            activity_id="ACT-GHOST",
            run_id="R1",
            actual_duration_hours=3.0,
        )
        stored = persister.appended[0]
        assert stored.activity_id == "ACT-GHOST"
        assert stored.actual_duration_hours == 3.0

    def test_minimal_reconstruction_outcome_notes_in_metadata(self):
        persister = NoOpAnalogPersister()
        w = _writer(persister=persister)
        w.record_completion(
            activity_id="ACT-UNKNOWN",
            run_id="R1",
            actual_duration_hours=4.0,
            outcome_notes="Replacement valve on backorder; workaround applied.",
        )
        stored = persister.appended[0]
        assert stored.metadata.get("outcome_notes") == "Replacement valve on backorder; workaround applied."


# ===========================================================================
# CsvAnalogPersister
# ===========================================================================

class TestCsvAnalogPersister:

    def test_creates_file_with_header_on_first_write(self, tmp_path):
        filepath = tmp_path / "analogs.csv"
        persister = CsvAnalogPersister(filepath)
        act = _activity("ACT-001")
        persister.append(act)
        assert filepath.exists()
        with open(filepath, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
        assert header == list(_CSV_COLUMNS)

    def test_data_row_written_correctly(self, tmp_path):
        filepath = tmp_path / "analogs.csv"
        persister = CsvAnalogPersister(filepath)
        act = _activity("ACT-001", outage_id="R2026", plant_id="PLANT-A")
        persister.append(act)
        with open(filepath, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["activity_id"] == "ACT-001"
        assert rows[0]["outage_id"] == "R2026"

    def test_second_append_does_not_add_duplicate_header(self, tmp_path):
        filepath = tmp_path / "analogs.csv"
        persister = CsvAnalogPersister(filepath)
        persister.append(_activity("ACT-001"))
        persister.append(_activity("ACT-002"))
        with open(filepath, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["activity_id"] == "ACT-001"
        assert rows[1]["activity_id"] == "ACT-002"

    def test_datetime_fields_serialised_to_iso_string(self, tmp_path):
        filepath = tmp_path / "analogs.csv"
        persister = CsvAnalogPersister(filepath)
        act = _activity(
            "ACT-001",
            actual_start=datetime(2026, 4, 12, 8, 0, 0, tzinfo=timezone.utc),
            actual_finish=datetime(2026, 4, 13, 0, 12, 0, tzinfo=timezone.utc),
        )
        persister.append(act)
        with open(filepath, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            row = next(reader)
        assert "T" in row["actual_start"]   # ISO-8601 format

    def test_all_csv_columns_present_in_header(self, tmp_path):
        filepath = tmp_path / "analogs.csv"
        persister = CsvAnalogPersister(filepath)
        persister.append(_activity("ACT-001"))
        with open(filepath, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
        for col in _CSV_COLUMNS:
            assert col in header

    def test_appends_to_existing_non_empty_file(self, tmp_path):
        filepath = tmp_path / "analogs.csv"
        persister = CsvAnalogPersister(filepath)
        persister.append(_activity("ACT-001"))
        # Second persister instance simulates next session
        persister2 = CsvAnalogPersister(filepath)
        persister2.append(_activity("ACT-002"))
        with open(filepath, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 2


# ===========================================================================
# NoOpAnalogPersister
# ===========================================================================

class TestNoOpAnalogPersister:

    def test_accumulates_activities(self):
        p = NoOpAnalogPersister()
        a1 = _activity("ACT-001")
        a2 = _activity("ACT-002")
        p.append(a1)
        p.append(a2)
        assert len(p.appended) == 2
        assert p.appended[0] is a1
        assert p.appended[1] is a2

    def test_starts_empty(self):
        p = NoOpAnalogPersister()
        assert p.appended == []

    def test_accepts_any_object(self):
        p = NoOpAnalogPersister()
        p.append("not an activity")
        p.append(42)
        assert len(p.appended) == 2
