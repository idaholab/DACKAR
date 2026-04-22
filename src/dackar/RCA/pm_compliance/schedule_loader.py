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
        """Filter and return rows for this asset scope.

        Rows missing ``check_id`` or ``check_type`` are dropped with a note that
        the caller can surface via ``data_quality_notes`` on the artifact.
        """
        if not rows:
            return []
        out: List[JsonDict] = []
        for r in rows:
            aid = r.get("asset_id")
            if aid is not None and aid != self._asset_id:
                continue
            cid = r.get("component_id")
            if self._component_ids and cid and cid not in self._component_ids:
                continue
            out.append(dict(r))
        return out
