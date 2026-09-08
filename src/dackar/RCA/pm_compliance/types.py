"""Shared typing helpers for PM compliance (no heavy runtime deps)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


JsonDict = Dict[str, Any]


class PMExportTaskRow(TypedDict, total=False):
    """One row from a CMMS export or adapter (before verification).

    *check_type* must match ``schemas/pm_compliance.json`` enum for ``checks[].check_type``.
    """

    task_code: str
    check_id: str
    check_type: str
    component_id: Optional[str]
    applicable_fm_ids: List[str]
    frequency_days: Optional[int]
    last_pm_date: Optional[str]
    next_due_date: Optional[str]
    scheduled_date: Optional[str]
    completed_date: Optional[str]
    as_found_last: Optional[str]
    wo_id: Optional[str]
    source_ref: Optional[str]
    details: Optional[str]
    evidence_refs: List[str]
    # Explicit overrides when export already computed status
    compliance_status: str
    missed_cycles: int

