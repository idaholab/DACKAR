from __future__ import annotations

from collections.abc import Iterable

from outage_uncertainty.domain.activity import ActivityCase


class PandasActivityRepository:
    def load_activities(self, rows: Iterable[dict]) -> list[ActivityCase]:
        activities: list[ActivityCase] = []
        for row in rows:
            activity = ActivityCase(
                activity_id=str(row.get("activity_id", "")),
                outage_id=str(row.get("outage_id", "")),
                plant_id=str(row.get("plant_id", "")),
                unit_id=row.get("unit_id"),
                raw_description=str(row.get("raw_description", "")),
                planned_duration_hours=row.get("planned_duration_hours"),
                actual_duration_hours=row.get("actual_duration_hours"),
                discipline=row.get("discipline"),
                task_family=row.get("task_family"),
                component_family=row.get("component_family"),
                system_name=row.get("system_name"),
                work_order_type=row.get("work_order_type"),
                is_emergent=bool(row.get("is_emergent", False)),
                is_rework=bool(row.get("is_rework", False)),
                has_rp_hold=bool(row.get("has_rp_hold", False)),
                requires_scaffold=bool(row.get("requires_scaffold", False)),
                has_clearance=bool(row.get("has_clearance", False)),
                is_vendor_supported=bool(row.get("is_vendor_supported", False)),
                crew_size=row.get("crew_size"),
                contractor_flag=row.get("contractor_flag"),
                outage_phase=row.get("outage_phase"),
                predecessor_ids=list(row.get("predecessor_ids", [])),
                successor_ids=list(row.get("successor_ids", [])),
                metadata=dict(row.get("metadata", {})),
            )
            activities.append(activity)
        return activities
