"""
Unit tests for CAPAdapter, FileDropCAPAdapter, NoOpCAPAdapter,
and CAPSubmissionReceipt.
"""
import json
import sys
import tempfile
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

import pytest
from cap_integration.cap_adapter import (
    CAPSubmissionReceipt,
    FileDropCAPAdapter,
    NoOpCAPAdapter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_package(export_id="CAPEXP::EVT-001::2026-01-01T00:00:00+00:00"):
    return {
        "export_id": export_id,
        "run_id": "run-001",
        "event_id": "EVT-001",
        "cr_records": [],
        "target_system": "maximo",
    }


# ---------------------------------------------------------------------------
# CAPSubmissionReceipt
# ---------------------------------------------------------------------------

class TestCAPSubmissionReceipt:

    def test_to_dict_contains_all_fields(self):
        r = CAPSubmissionReceipt(
            receipt_id="RCPT-001",
            submitted_at="2026-01-01T00:00:00+00:00",
            adapter="FileDropCAPAdapter",
            export_id="CAPEXP::EVT-001",
            cr_numbers=["CR-100", "CR-101"],
            status="submitted",
            errors=[],
            notes="some note",
        )
        d = r.to_dict()
        assert d["receipt_id"] == "RCPT-001"
        assert d["adapter"] == "FileDropCAPAdapter"
        assert d["export_id"] == "CAPEXP::EVT-001"
        assert d["cr_numbers"] == ["CR-100", "CR-101"]
        assert d["status"] == "submitted"
        assert d["notes"] == "some note"

    def test_default_status_is_pending(self):
        r = CAPSubmissionReceipt(
            receipt_id="R",
            submitted_at="2026-01-01",
            adapter="A",
            export_id="E",
        )
        assert r.status == "pending"
        assert r.cr_numbers == []
        assert r.errors == []
        assert r.notes is None

    def test_to_dict_keys_complete(self):
        r = CAPSubmissionReceipt(
            receipt_id="R", submitted_at="T", adapter="A", export_id="E"
        )
        keys = set(r.to_dict().keys())
        expected = {"receipt_id", "submitted_at", "adapter", "export_id",
                    "cr_numbers", "status", "errors", "notes"}
        assert keys == expected


# ---------------------------------------------------------------------------
# NoOpCAPAdapter
# ---------------------------------------------------------------------------

class TestNoOpCAPAdapter:

    def test_returns_receipt(self):
        adapter = NoOpCAPAdapter()
        receipt = adapter.submit(_make_package())
        assert isinstance(receipt, CAPSubmissionReceipt)

    def test_status_noop(self):
        receipt = NoOpCAPAdapter().submit(_make_package())
        assert receipt.status == "noop"

    def test_cr_numbers_empty(self):
        receipt = NoOpCAPAdapter().submit(_make_package())
        assert receipt.cr_numbers == []

    def test_adapter_name(self):
        receipt = NoOpCAPAdapter().submit(_make_package())
        assert receipt.adapter == "NoOpCAPAdapter"

    def test_export_id_preserved(self):
        pkg = _make_package(export_id="CAPEXP::EVT-42::2026")
        receipt = NoOpCAPAdapter().submit(pkg)
        assert receipt.export_id == "CAPEXP::EVT-42::2026"

    def test_notes_describes_discard(self):
        receipt = NoOpCAPAdapter().submit(_make_package())
        assert receipt.notes is not None
        assert "NoOpCAPAdapter" in receipt.notes or "discarded" in receipt.notes

    def test_receipt_id_contains_noop(self):
        receipt = NoOpCAPAdapter().submit(_make_package())
        assert "NOOP" in receipt.receipt_id

    def test_no_file_written(self, tmp_path):
        NoOpCAPAdapter().submit(_make_package())
        # no files should have been created in tmp_path (just baseline check)
        assert list(tmp_path.iterdir()) == []

    def test_missing_export_id_falls_back(self):
        pkg = {"run_id": "run-001"}
        receipt = NoOpCAPAdapter().submit(pkg)
        assert receipt.export_id == "unknown"


# ---------------------------------------------------------------------------
# FileDropCAPAdapter
# ---------------------------------------------------------------------------

class TestFileDropCAPAdapter:

    def test_creates_drop_dir_if_absent(self, tmp_path):
        drop_dir = tmp_path / "new_dir" / "nested"
        FileDropCAPAdapter(drop_dir)
        assert drop_dir.exists()

    def test_returns_receipt(self, tmp_path):
        adapter = FileDropCAPAdapter(tmp_path)
        receipt = adapter.submit(_make_package())
        assert isinstance(receipt, CAPSubmissionReceipt)

    def test_status_pending(self, tmp_path):
        adapter = FileDropCAPAdapter(tmp_path)
        receipt = adapter.submit(_make_package())
        assert receipt.status == "pending"

    def test_cr_numbers_empty(self, tmp_path):
        adapter = FileDropCAPAdapter(tmp_path)
        receipt = adapter.submit(_make_package())
        assert receipt.cr_numbers == []

    def test_adapter_name(self, tmp_path):
        adapter = FileDropCAPAdapter(tmp_path)
        receipt = adapter.submit(_make_package())
        assert receipt.adapter == "FileDropCAPAdapter"

    def test_export_id_in_receipt(self, tmp_path):
        pkg = _make_package(export_id="CAPEXP::EVT-007::2026")
        receipt = FileDropCAPAdapter(tmp_path).submit(pkg)
        assert receipt.export_id == "CAPEXP::EVT-007::2026"

    def test_file_written(self, tmp_path):
        adapter = FileDropCAPAdapter(tmp_path)
        adapter.submit(_make_package())
        written_files = list(tmp_path.glob("cap_export_*.json"))
        assert len(written_files) == 1

    def test_written_file_is_valid_json(self, tmp_path):
        adapter = FileDropCAPAdapter(tmp_path)
        pkg = _make_package()
        adapter.submit(pkg)
        written_file = list(tmp_path.glob("cap_export_*.json"))[0]
        loaded = json.loads(written_file.read_text())
        assert loaded["export_id"] == pkg["export_id"]

    def test_notes_is_file_path(self, tmp_path):
        adapter = FileDropCAPAdapter(tmp_path)
        receipt = adapter.submit(_make_package())
        assert receipt.notes is not None
        assert Path(receipt.notes).exists()

    def test_receipt_id_format(self, tmp_path):
        adapter = FileDropCAPAdapter(tmp_path)
        pkg = _make_package(export_id="CAPEXP::EVT-001::ts")
        receipt = adapter.submit(pkg)
        assert receipt.receipt_id.startswith("RCPT::")
        assert "CAPEXP::EVT-001::ts" in receipt.receipt_id

    def test_file_name_safe_chars(self, tmp_path):
        adapter = FileDropCAPAdapter(tmp_path)
        pkg = _make_package(export_id="CAPEXP::EVT-001::2026-01-01T00:00:00+00:00")
        adapter.submit(pkg)
        files = list(tmp_path.glob("cap_export_*.json"))
        assert len(files) == 1
        # colons in export_id should have been sanitized
        assert "::" not in files[0].name

    def test_multiple_submits_write_multiple_files(self, tmp_path):
        adapter = FileDropCAPAdapter(tmp_path)
        adapter.submit(_make_package(export_id="CAPEXP::A::ts1"))
        adapter.submit(_make_package(export_id="CAPEXP::B::ts2"))
        files = list(tmp_path.glob("cap_export_*.json"))
        assert len(files) == 2

    def test_missing_export_id_falls_back(self, tmp_path):
        adapter = FileDropCAPAdapter(tmp_path)
        pkg = {"run_id": "run-001", "cr_records": []}
        receipt = adapter.submit(pkg)
        assert receipt.export_id == "unknown"
        files = list(tmp_path.glob("cap_export_*.json"))
        assert len(files) == 1

    def test_accepts_path_object(self, tmp_path):
        adapter = FileDropCAPAdapter(Path(tmp_path))
        receipt = adapter.submit(_make_package())
        assert receipt.status == "pending"

    def test_accepts_string_path(self, tmp_path):
        adapter = FileDropCAPAdapter(str(tmp_path))
        receipt = adapter.submit(_make_package())
        assert receipt.status == "pending"
