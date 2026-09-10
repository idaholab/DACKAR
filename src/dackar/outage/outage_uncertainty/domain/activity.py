from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ActivityCase:
    activity_id: str
    outage_id: str
    plant_id: str
    unit_id: str | None = None

    raw_description: str = ""
    cleaned_description: str | None = None

    planned_start: datetime | None = None
    planned_finish: datetime | None = None
    actual_start: datetime | None = None
    actual_finish: datetime | None = None

    planned_duration_hours: float | None = None
    actual_duration_hours: float | None = None

    discipline: str | None = None
    task_family: str | None = None
    component_family: str | None = None
    system_name: str | None = None
    work_order_type: str | None = None

    is_emergent: bool = False
    is_rework: bool = False
    # Gap 3: execution mode flags — strong predictors of duration variance
    has_rp_hold: bool = False          # radiation protection hold point required
    requires_scaffold: bool = False    # scaffolding erection / removal in scope
    has_clearance: bool = False        # electrical / mechanical clearance needed
    is_vendor_supported: bool = False  # OEM / specialty vendor performing work

    crew_size: int | None = None
    contractor_flag: bool | None = None
    outage_phase: str | None = None

    predecessor_ids: list[str] = field(default_factory=list)
    successor_ids: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_actual_duration(self) -> float | None:
        if self.actual_start and self.actual_finish:
            return (self.actual_finish - self.actual_start).total_seconds() / 3600.0
        return None

    def is_historical(self) -> bool:
        return self.actual_duration_hours is not None

    def is_query_candidate(self) -> bool:
        return self.planned_duration_hours is not None
