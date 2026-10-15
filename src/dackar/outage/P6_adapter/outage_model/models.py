from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


TaskStatus = Literal["not_started", "in_progress", "complete", "cancelled", "unknown"]
VersionType = Literal["baseline", "working", "approved_update", "as_run"]
RelationshipType = Literal["FS", "SS", "FF", "SF"]


class Outage(BaseModel):
    """A planned or in-progress plant outage."""

    outage_id: str
    plant_id: Optional[str] = None
    unit_id: Optional[str] = None
    outage_name: str
    outage_type: Optional[str] = None
    start_planned: Optional[datetime] = None
    finish_planned: Optional[datetime] = None
    start_actual: Optional[datetime] = None
    finish_actual: Optional[datetime] = None
    status: Optional[str] = None
    description: Optional[str] = None


class OutagePhase(BaseModel):
    """A named phase (sub-interval) within an outage."""

    phase_id: str
    outage_id: str
    phase_name: str
    sequence: Optional[int] = None
    start_planned: Optional[datetime] = None
    finish_planned: Optional[datetime] = None
    start_actual: Optional[datetime] = None
    finish_actual: Optional[datetime] = None


class ScheduleVersion(BaseModel):
    """A specific version of an outage schedule (baseline, working, as-run, ...)."""

    schedule_version_id: str
    outage_id: str
    version_name: str
    version_type: VersionType
    status_date: Optional[datetime] = None
    created_at: Optional[datetime] = None
    source_system: Optional[str] = None
    source_file: Optional[str] = None
    is_official: bool = False


class WBS(BaseModel):
    """A work-breakdown-structure node within a schedule version."""

    wbs_id: str
    outage_id: str
    schedule_version_id: str
    parent_wbs_id: Optional[str] = None
    wbs_code: Optional[str] = None
    wbs_name: str
    wbs_path: Optional[str] = None
    level: Optional[int] = None


class ScheduleTask(BaseModel):
    """A single scheduled activity within a schedule version."""

    task_id: str
    outage_id: str
    schedule_version_id: str
    wbs_id: Optional[str] = None
    task_code: Optional[str] = None
    task_name: str
    task_type: Optional[str] = None
    milestone_flag: bool = False
    planned_start: Optional[datetime] = None
    planned_finish: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_finish: Optional[datetime] = None
    planned_duration_hours: Optional[float] = None
    remaining_duration_hours: Optional[float] = None
    total_float_hours: Optional[float] = None
    free_float_hours: Optional[float] = None
    critical_flag: Optional[bool] = None
    status: TaskStatus = "unknown"
    percent_complete: Optional[float] = None
    physical_percent_complete: Optional[float] = None
    calendar_id: Optional[str] = None
    primary_constraint_type: Optional[str] = None
    primary_constraint_date: Optional[datetime] = None
    scope_origin: Optional[str] = None
    source_system: Optional[str] = None
    source_record_id: Optional[str] = None


class Dependency(BaseModel):
    """A precedence relationship between two scheduled tasks."""

    dependency_id: str
    schedule_version_id: str
    predecessor_task_id: str
    successor_task_id: str
    relationship_type: RelationshipType
    lag_hours: float = 0.0
    lag_calendar_id: Optional[str] = None
    driving_flag: Optional[bool] = None
    external_link_flag: Optional[bool] = None


class Resource(BaseModel):
    """A labor, crew, or equipment resource that can be assigned to tasks."""

    resource_id: str
    resource_code: Optional[str] = None
    resource_name: str
    resource_type: Optional[str] = None
    craft: Optional[str] = None
    vendor: Optional[str] = None
    org_unit: Optional[str] = None
    calendar_id: Optional[str] = None


class ResourceAssignment(BaseModel):
    """An assignment of a resource to a task, with planned and actual units."""

    assignment_id: str
    schedule_version_id: str
    task_id: str
    resource_id: str
    role_name: Optional[str] = None
    planned_units: Optional[float] = None
    actual_units: Optional[float] = None
    remaining_units: Optional[float] = None
    planned_cost: Optional[float] = None
    actual_cost: Optional[float] = None


class Calendar(BaseModel):
    """A working-time calendar referenced by tasks and resources."""

    calendar_id: str
    calendar_name: str
    calendar_type: Optional[str] = None
    timezone: Optional[str] = None
    work_pattern_json: dict[str, Any] = Field(default_factory=dict)


class ActivityCode(BaseModel):
    """A categorization code value that can be applied to tasks."""

    activity_code_id: str
    code_type: str
    code_value: str
    code_description: Optional[str] = None


class TaskActivityCode(BaseModel):
    """A link between a task and an activity code (many-to-many join)."""

    task_id: str
    activity_code_id: str


class TaskConstraint(BaseModel):
    """A date or scheduling constraint imposed on a task."""

    task_constraint_id: str
    task_id: str
    constraint_type: str
    constraint_date: Optional[datetime] = None
    reason: Optional[str] = None
    hard_flag: Optional[bool] = None


class WorkPackage(BaseModel):
    """A bundle of work scoped to an outage, with discipline and approval state."""

    work_package_id: str
    outage_id: str
    package_code: Optional[str] = None
    package_name: str
    package_type: Optional[str] = None
    discipline: Optional[str] = None
    system_id: Optional[str] = None
    approval_status: Optional[str] = None


class ScopeChangeEvent(BaseModel):
    """A recorded change to outage scope (task added, removed, or modified)."""

    scope_change_id: str
    outage_id: str
    task_id: Optional[str] = None
    change_type: str
    change_timestamp: Optional[datetime] = None
    reason: Optional[str] = None
    approver: Optional[str] = None
    impact_hours: Optional[float] = None


class DelayEvent(BaseModel):
    """A recorded delay affecting one or more tasks, with cause and impact."""

    delay_event_id: str
    outage_id: str
    task_id: Optional[str] = None
    delay_category: Optional[str] = None
    delay_reason: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    delay_hours: Optional[float] = None
    impact_hours: Optional[float] = None


class WorkWindow(BaseModel):
    """A time window during which particular work is permitted."""

    work_window_id: str
    outage_id: str
    window_type: Optional[str] = None
    window_name: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class Clearance(BaseModel):
    """A safety clearance (tag-out) on a system during the outage."""

    clearance_id: str
    outage_id: str
    clearance_code: Optional[str] = None
    system_id: Optional[str] = None
    status: Optional[str] = None
    issued_time: Optional[datetime] = None
    released_time: Optional[datetime] = None
