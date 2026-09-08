from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .activity import ActivityCase


@dataclass
class OutageRecord:
    outage_id: str
    plant_id: str
    unit_id: str | None
    start_date: datetime
    end_date: datetime | None = None
    activities: list[ActivityCase] = field(default_factory=list)

    def get_activity(self, activity_id: str) -> ActivityCase:
        for activity in self.activities:
            if activity.activity_id == activity_id:
                return activity
        raise KeyError(f"Activity not found: {activity_id}")
