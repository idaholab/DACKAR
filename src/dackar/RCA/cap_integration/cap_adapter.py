"""
cap_adapter — CAPAdapter Protocol, FileDropCAPAdapter, NoOpCAPAdapter.

Concrete live adapters (MaximoCAPAdapter, SAPPMCAPAdapter) live in separate
files and implement the same Protocol.  See CAP_INTEGRATION_GUIDE.md for the
implementation skeleton.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

JsonDict = Dict[str, Any]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_export_filename(export_id: str) -> str:
    """Return a filesystem-safe stem for CAP export files."""
    normalized = str(export_id or "unknown").replace("::", "_").replace(" ", "_")
    # Windows-invalid chars: <>:"/\|?* ; keep output deterministic across OSes.
    return re.sub(r'[<>:"/\\|?*]', "_", normalized)


# ---------------------------------------------------------------------------
# Submission receipt
# ---------------------------------------------------------------------------

@dataclass
class CAPSubmissionReceipt:
    """
    Returned by every CAPAdapter.submit() call.

    Attributes
    ----------
    receipt_id:
        Unique ID for this submission attempt.
    submitted_at:
        ISO-8601 UTC timestamp.
    adapter:
        Class name of the adapter that produced this receipt.
    export_id:
        ``export_id`` from the CAPExportPackage that was submitted.
    cr_numbers:
        CMMS-assigned corrective action / work order numbers.
        Empty list for adapters that do not receive synchronous confirmation
        (e.g. ``FileDropCAPAdapter``).
    status:
        ``"submitted"`` — accepted by CMMS synchronously.
        ``"pending"``   — written to drop zone; CMMS import job pending.
        ``"partial"``   — some records submitted, some failed.
        ``"failed"``    — no records submitted.
        ``"noop"``      — NoOpCAPAdapter; nothing was done.
    errors:
        List of per-record error dicts when status is partial or failed.
    notes:
        Free-text adapter notes (e.g. path of the written file).
    """

    receipt_id: str
    submitted_at: str
    adapter: str
    export_id: str
    cr_numbers: List[str] = field(default_factory=list)
    status: str = "pending"
    errors: List[JsonDict] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> JsonDict:
        return {
            "receipt_id": self.receipt_id,
            "submitted_at": self.submitted_at,
            "adapter": self.adapter,
            "export_id": self.export_id,
            "cr_numbers": self.cr_numbers,
            "status": self.status,
            "errors": self.errors,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class CAPAdapter(Protocol):
    """
    Protocol for CAP submission adapters.

    All implementations must be idempotent with respect to
    ``package["export_id"]``: submitting the same package twice must not
    create duplicate CMMS records.
    """

    def submit(self, package: JsonDict) -> CAPSubmissionReceipt:
        """
        Submit a CAPExportPackage to the target CMMS (or file drop zone).

        Parameters
        ----------
        package:
            Dict conforming to ``schemas/cap_export_package.json``.

        Returns
        -------
        CAPSubmissionReceipt
        """
        ...


# ---------------------------------------------------------------------------
# FileDropCAPAdapter
# ---------------------------------------------------------------------------

class FileDropCAPAdapter:
    """
    Writes the CAPExportPackage as a JSON file to a watched directory.

    This is the default adapter for dev and pre-live deployments.  The plant's
    CMMS import job polls the directory and ingests the file asynchronously.
    Returns a receipt with ``status="pending"`` and an empty ``cr_numbers``
    list — CMMS record numbers are not available at submission time.

    Parameters
    ----------
    drop_dir:
        Directory where export files are written.  Created if absent.
    """

    def __init__(self, drop_dir: str | Path) -> None:
        self.drop_dir = Path(drop_dir)
        self.drop_dir.mkdir(parents=True, exist_ok=True)

    def submit(self, package: JsonDict) -> CAPSubmissionReceipt:
        export_id = package.get("export_id") or "unknown"
        submitted_at = _utcnow_iso()
        receipt_id = f"RCPT::{export_id}::{submitted_at}"

        safe_name = _safe_export_filename(export_id)
        file_path = self.drop_dir / f"cap_export_{safe_name}.json"
        file_path.write_text(json.dumps(package, indent=2, default=str))

        return CAPSubmissionReceipt(
            receipt_id=receipt_id,
            submitted_at=submitted_at,
            adapter=self.__class__.__name__,
            export_id=export_id,
            cr_numbers=[],
            status="pending",
            notes=str(file_path),
        )


# ---------------------------------------------------------------------------
# NoOpCAPAdapter
# ---------------------------------------------------------------------------

class NoOpCAPAdapter:
    """
    Silently discards the package.  Used in unit tests and CI.

    Returns a receipt with ``status="noop"`` and an empty ``cr_numbers`` list.
    Makes no file I/O and has no external dependencies.
    """

    def submit(self, package: JsonDict) -> CAPSubmissionReceipt:
        export_id = package.get("export_id") or "unknown"
        submitted_at = _utcnow_iso()
        return CAPSubmissionReceipt(
            receipt_id=f"RCPT::NOOP::{submitted_at}",
            submitted_at=submitted_at,
            adapter=self.__class__.__name__,
            export_id=export_id,
            cr_numbers=[],
            status="noop",
            notes="NoOpCAPAdapter — package discarded.",
        )
