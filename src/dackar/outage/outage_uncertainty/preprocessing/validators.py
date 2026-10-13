from __future__ import annotations

from outage_uncertainty.domain.activity import ActivityCase


class ActivityValidator:
    def validate(self, activity: ActivityCase) -> list[str]:
        errors: list[str] = []
        if not activity.activity_id:
            errors.append("activity_id is required")
        if not activity.outage_id:
            errors.append("outage_id is required")
        if not activity.plant_id:
            errors.append("plant_id is required")
        return errors
