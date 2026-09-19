"""PMScheduleLoader — load PM schedule / task rows for an asset scope."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .types import JsonDict


class PMScheduleLoader:
    """Loads PM task definitions for *asset_id* and optional *component_ids*.

    Phase 1: accepts pre-parsed export rows (CSV / JSON / adapter). Real-time CMMS
    API integration is Phase 2 — see ``PM_Compliance_Module_Architecture.md`` §6.
    """

    def __init__(self, asset_id: str, component_ids: Optional[Sequence[str]] = None) -> None:
        self._asset_id = asset_id
        self._component_ids = set(component_ids) if component_ids else set()

    def load_from_export_rows(self, rows: Optional[Sequence[JsonDict]]) -> List[JsonDict]:
        """Filter and return rows for this asset scope."""
        loaded, _ = self.load_from_export_rows_with_notes(rows)
        return loaded

    def load_from_export_rows_with_notes(
        self,
        rows: Optional[Sequence[JsonDict]],
    ) -> tuple[List[JsonDict], List[str]]:
        """Filter rows for scope and return ``(rows, data_quality_notes)``.

        Rows missing ``check_id`` (or fallback ``task_code``) or ``check_type``
        are dropped to avoid ambiguous downstream governance interpretation.
        """
        if not rows:
            return [], []
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
            out.append(dict(r))
        return out, notes
