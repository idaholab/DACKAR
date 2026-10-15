from __future__ import annotations

from collections.abc import Iterable

from dackar.outage.outage_uncertainty.adapters.p6_dataset_adapter import (
    _to_bool,
    _to_float,
    _to_str,
)
from dackar.outage.outage_uncertainty.domain.activity import ActivityCase


class PandasActivityRepository:
    """Build :class:`ActivityCase` objects from plain dict rows.

    The rows typically come from ``DataFrame.to_dict("records")``, so missing
    cells arrive as ``float('nan')`` rather than ``None``.  Values are coerced
    through the shared ``_to_float`` / ``_to_str`` / ``_to_bool`` helpers so a
    missing numeric field becomes ``None`` (not ``nan``) and a missing string
    becomes ``None`` (not the literal ``"None"``); otherwise ``nan`` would pass
    the ``is not None`` filters downstream and surface as ``p50=nan``.
    """

    def load_activities(self, rows: Iterable[dict]) -> list[ActivityCase]:
        """Return one :class:`ActivityCase` per row in *rows*.

        Args:
            rows: Iterable of dict records (e.g. ``DataFrame.to_dict``
                output).  Missing values may be ``None`` or ``float('nan')``.

        Returns:
            List of :class:`ActivityCase` with NaN-coerced optional fields.
        """
        activities: list[ActivityCase] = []
        for row in rows:
            activity = ActivityCase(
                activity_id=_to_str(row.get("activity_id")) or "",
                outage_id=_to_str(row.get("outage_id")) or "",
                plant_id=_to_str(row.get("plant_id")) or "",
                unit_id=_to_str(row.get("unit_id")),
                raw_description=_to_str(row.get("raw_description")) or "",
                planned_duration_hours=_to_float(row.get("planned_duration_hours")),
                actual_duration_hours=_to_float(row.get("actual_duration_hours")),
                discipline=_to_str(row.get("discipline")),
                task_family=_to_str(row.get("task_family")),
                component_family=_to_str(row.get("component_family")),
                system_name=_to_str(row.get("system_name")),
                work_order_type=_to_str(row.get("work_order_type")),
                is_emergent=_to_bool(row.get("is_emergent")),
                is_rework=_to_bool(row.get("is_rework")),
                has_rp_hold=_to_bool(row.get("has_rp_hold")),
                requires_scaffold=_to_bool(row.get("requires_scaffold")),
                has_clearance=_to_bool(row.get("has_clearance")),
                is_vendor_supported=_to_bool(row.get("is_vendor_supported")),
                crew_size=row.get("crew_size"),
                contractor_flag=row.get("contractor_flag"),
                outage_phase=_to_str(row.get("outage_phase")),
                predecessor_ids=list(row.get("predecessor_ids", [])),
                successor_ids=list(row.get("successor_ids", [])),
                metadata=dict(row.get("metadata", {})),
            )
            activities.append(activity)
        return activities
