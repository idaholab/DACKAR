"""PMScheduleLoader — load PM schedule / task rows for an asset scope."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from dackar.RCA._timeutils import parse_dt
from .types import JsonDict


class PMScheduleLoader:
    """Loads PM task definitions for *asset_id* and optional *component_ids*.

    Phase 1: accepts pre-parsed export rows (CSV / JSON / adapter). Real-time CMMS
    API integration is Phase 2 — see ``PM_Compliance_Module_Architecture.md`` §6.
    """

    def __init__(self, asset_id: str, component_ids: Optional[Sequence[str]] = None) -> None:
        self._asset_id = asset_id
        self._component_ids = set(component_ids) if component_ids else set()

    def load_from_export_rows(
        self,
        rows: Optional[Sequence[JsonDict]],
        window_start: Optional[str] = None,
    ) -> List[JsonDict]:
        """Filter and return rows for this asset scope."""
        loaded, _ = self.load_from_export_rows_with_notes(rows, window_start=window_start)
        return loaded

    def load_from_export_rows_with_notes(
        self,
        rows: Optional[Sequence[JsonDict]],
        window_start: Optional[str] = None,
    ) -> tuple[List[JsonDict], List[str]]:
        """Filter rows for scope and return ``(rows, data_quality_notes)``.

        Rows missing ``check_id`` (or fallback ``task_code``) or ``check_type``
        are dropped to avoid ambiguous downstream governance interpretation.

        When *window_start* (ISO-8601) is given, rows whose most-recent activity
        date (``completed_date`` / ``last_pm_date`` / ``scheduled_date`` /
        ``next_due_date``) falls strictly before it are dropped as out-of-window
        (MR#56 review I2). Rows with no parseable date are kept — an undated task
        is typically a never-run / overdue candidate that still belongs in scope.
        """
        if not rows:
            return [], []
        ws = parse_dt(window_start) if window_start else None
        out: List[JsonDict] = []
        notes: List[str] = []
        for r in rows:
            aid = r.get("asset_id")
            if aid is not None and aid != self._asset_id:
                continue
            cid = r.get("component_id")
            if self._component_ids and cid and cid not in self._component_ids:
                continue
            check_id = str(r.get("check_id") or r.get("task_code") or "").strip()
            check_type = str(r.get("check_type") or "").strip()
            if not check_id or not check_type:
                notes.append(
                    "Dropped PM export row missing required check_id/task_code or check_type."
                )
                continue
            if ws is not None:
                activity = parse_dt(
                    r.get("completed_date")
                    or r.get("last_pm_date")
                    or r.get("scheduled_date")
                    or r.get("next_due_date")
                )
                if activity is not None and activity < ws:
                    notes.append(
                        f"Dropped PM row {check_id!r}: last activity {activity.date().isoformat()} "
                        f"predates lookback window start {ws.date().isoformat()}."
                    )
                    continue
            out.append(dict(r))
        return out, notes
