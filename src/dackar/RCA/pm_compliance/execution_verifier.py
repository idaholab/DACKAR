"""PMExecutionVerifier — map scheduled PM to pass/fail/unknown *checks* rows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from orchestrators.causality_engine_v32 import parse_dt
from .types import JsonDict

_VALID_CHECK_TYPES = frozenset({
    "scheduled_pm",
    "surveillance_test",
    "calibration",
    "inspection",
    "lubrication",
    "functional_test",
    "other",
})


@dataclass
class PMExecutionVerifier:
    """Verifies each PM task against *event* time: overdue, missed cycles, check status."""

    event_timestamp_iso: str

    def _event_dt(self) -> Optional[datetime]:
        return parse_dt(self.event_timestamp_iso)

    def verify_rows(self, rows: List[JsonDict]) -> Tuple[List[JsonDict], List[str]]:
        """Return (checks_for_schema, data_quality_notes)."""
        notes: List[str] = []
        checks: List[JsonDict] = []
        event_dt = self._event_dt()
        if not event_dt:
            notes.append("event.timestamp_start not parseable — all checks marked unknown")
            for i, r in enumerate(rows):
                checks.append(self._make_unknown_check(r, f"unparseable_event_{i}"))
            return checks, notes

        for i, r in enumerate(rows):
            cid = (r.get("check_id") or r.get("task_code") or f"row_{i}").strip()
            ctype = (r.get("check_type") or "other").strip()
            if ctype not in _VALID_CHECK_TYPES:
                notes.append(f"check_id {cid}: coerced check_type {ctype!r} to other")
                ctype = "other"

            if r.get("compliance_status") in (
                "not_applicable",
                "n_a",
            ):
                st = "pass"
                overdue = 0.0
                notes.append(
                    f"check_id {cid}: not_applicable (e.g. CBM/operating-hour PM without runtime "
                    f"input — per architecture treat as non-schedule signal)"
                )
            elif r.get("compliance_status") in ("compliant", "pass"):
                st = "pass"
                overdue = 0.0
            elif r.get("compliance_status") in ("overdue", "missed", "fail", "non_compliant"):
                st = "fail"
                overdue = float(r.get("overdue_by_days") or r.get("overdue_days") or 0.0)
            else:
                st, overdue = self._derive_status(r, event_dt, notes, cid)

            check: JsonDict = {
                "check_id": cid,
                "check_type": ctype,
                "status": st,
                "overdue_by_days": overdue,
            }
            if r.get("source_ref"):
                check["source_ref"] = r["source_ref"]
            for key in (
                "scheduled_date",
                "completed_date",
                "component_id",
                "applicable_fm_ids",
                "evidence_refs",
            ):
                if r.get(key) is not None:
                    check[key] = r[key]
            dtl = r.get("details")
            if dtl is not None:
                check["details"] = dtl
            elif st == "pass" and r.get("compliance_status") in ("not_applicable", "n_a"):
                check["details"] = "compliance_status=not_applicable (architecture §6 / §7)"
            if r.get("wo_id"):
                check["wo_id"] = r["wo_id"]
            checks.append(check)

        return checks, notes

    @staticmethod
    def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
        """Ensure *dt* is tz-aware (UTC) so comparisons never raise TypeError."""
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def _derive_status(
        self,
        r: JsonDict,
        event_dt: datetime,
        notes: List[str],
        cid: str,
    ) -> Tuple[str, float]:
        next_due = self._as_utc(parse_dt(r.get("next_due_date") or r.get("next_due")) if (
            r.get("next_due_date") or r.get("next_due")
        ) else None)
        last = self._as_utc(parse_dt(r.get("last_pm_date") or r.get("completed_date") or r.get("last_completed")))
        event_dt = self._as_utc(event_dt)  # type: ignore[assignment]

        if r.get("missed_cycles", 0) and int(r["missed_cycles"]) > 0:
            overdue = 0.0
            if next_due and event_dt > next_due:
                overdue = (event_dt - next_due).total_seconds() / 86400.0
            return "fail", max(0.0, overdue)

        if not next_due and not last:
            notes.append(f"check_id {cid}: no schedule dates — status unknown")
            return "unknown", 0.0

        if next_due and event_dt > next_due:
            overdue_days = (event_dt - next_due).total_seconds() / 86400.0
            return "fail", max(0.0, overdue_days)

        return "pass", 0.0

    def _make_unknown_check(self, r: JsonDict, fallback_id: str) -> JsonDict:
        return {
            "check_id": (r.get("check_id") or r.get("task_code") or fallback_id).strip(),
            "check_type": (r.get("check_type") or "other").strip(),
            "status": "unknown",
            "overdue_by_days": 0.0,
        }
