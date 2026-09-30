from __future__ import annotations

from typing import Any


PANDAS_PHYSICAL_SCHEMA: dict[str, dict[str, str]] = {
    "outages": {
        "outage_id": "string",
        "plant_id": "string",
        "unit_id": "string",
        "outage_name": "string",
        "outage_type": "string",
        "start_planned": "datetime64[ns]",
        "finish_planned": "datetime64[ns]",
        "start_actual": "datetime64[ns]",
        "finish_actual": "datetime64[ns]",
        "status": "string",
        "description": "string",
    },
    "outage_phases": {
        "phase_id": "string",
        "outage_id": "string",
        "phase_name": "string",
        "sequence": "Int64",
        "start_planned": "datetime64[ns]",
        "finish_planned": "datetime64[ns]",
        "start_actual": "datetime64[ns]",
        "finish_actual": "datetime64[ns]",
    },
    "schedule_versions": {
        "schedule_version_id": "string",
        "outage_id": "string",
        "version_name": "string",
        "version_type": "string",
        "status_date": "datetime64[ns]",
        "created_at": "datetime64[ns]",
        "source_system": "string",
        "source_file": "string",
        "is_official": "boolean",
    },
    "wbs": {
        "wbs_id": "string",
        "outage_id": "string",
        "schedule_version_id": "string",
        "parent_wbs_id": "string",
        "wbs_code": "string",
        "wbs_name": "string",
        "wbs_path": "string",
        "level": "Int64",
    },
    "schedule_tasks": {
        "task_id": "string",
        "outage_id": "string",
        "schedule_version_id": "string",
        "wbs_id": "string",
        "task_code": "string",
        "task_name": "string",
        "task_type": "string",
        "milestone_flag": "boolean",
        "planned_start": "datetime64[ns]",
        "planned_finish": "datetime64[ns]",
        "actual_start": "datetime64[ns]",
        "actual_finish": "datetime64[ns]",
        "planned_duration_hours": "Float64",
        "remaining_duration_hours": "Float64",
        "total_float_hours": "Float64",
        "free_float_hours": "Float64",
        "critical_flag": "boolean",
        "status": "string",
        "percent_complete": "Float64",
        "physical_percent_complete": "Float64",
        "calendar_id": "string",
        "primary_constraint_type": "string",
        "primary_constraint_date": "datetime64[ns]",
        "scope_origin": "string",
        "source_system": "string",
        "source_record_id": "string",
    },
    "dependencies": {
        "dependency_id": "string",
        "schedule_version_id": "string",
        "predecessor_task_id": "string",
        "successor_task_id": "string",
        "relationship_type": "string",
        "lag_hours": "Float64",
        "lag_calendar_id": "string",
        "driving_flag": "boolean",
        "external_link_flag": "boolean",
    },
    "resources": {
        "resource_id": "string",
        "resource_code": "string",
        "resource_name": "string",
        "resource_type": "string",
        "craft": "string",
        "vendor": "string",
        "org_unit": "string",
        "calendar_id": "string",
    },
    "resource_assignments": {
        "assignment_id": "string",
        "schedule_version_id": "string",
        "task_id": "string",
        "resource_id": "string",
        "role_name": "string",
        "planned_units": "Float64",
        "actual_units": "Float64",
        "remaining_units": "Float64",
        "planned_cost": "Float64",
        "actual_cost": "Float64",
    },
    "calendars": {
        "calendar_id": "string",
        "calendar_name": "string",
        "calendar_type": "string",
        "timezone": "string",
        "work_pattern_json": "object",
    },
    "activity_codes": {
        "activity_code_id": "string",
        "code_type": "string",
        "code_value": "string",
        "code_description": "string",
    },
    "task_activity_codes": {
        "task_id": "string",
        "activity_code_id": "string",
    },
    "task_constraints": {
        "task_constraint_id": "string",
        "task_id": "string",
        "constraint_type": "string",
        "constraint_date": "datetime64[ns]",
        "reason": "string",
        "hard_flag": "boolean",
    },
    "work_packages": {
        "work_package_id": "string",
        "outage_id": "string",
        "package_code": "string",
        "package_name": "string",
        "package_type": "string",
        "discipline": "string",
        "system_id": "string",
        "approval_status": "string",
    },
    "scope_change_events": {
        "scope_change_id": "string",
        "outage_id": "string",
        "task_id": "string",
        "change_type": "string",
        "change_timestamp": "datetime64[ns]",
        "reason": "string",
        "approver": "string",
        "impact_hours": "Float64",
    },
    "delay_events": {
        "delay_event_id": "string",
        "outage_id": "string",
        "task_id": "string",
        "delay_category": "string",
        "delay_reason": "string",
        "start_time": "datetime64[ns]",
        "end_time": "datetime64[ns]",
        "delay_hours": "Float64",
        "impact_hours": "Float64",
    },
    "work_windows": {
        "work_window_id": "string",
        "outage_id": "string",
        "window_type": "string",
        "window_name": "string",
        "start_time": "datetime64[ns]",
        "end_time": "datetime64[ns]",
    },
    "clearances": {
        "clearance_id": "string",
        "outage_id": "string",
        "clearance_code": "string",
        "system_id": "string",
        "status": "string",
        "issued_time": "datetime64[ns]",
        "released_time": "datetime64[ns]",
    },
}


PRIMARY_KEYS: dict[str, list[str]] = {
    "outages": ["outage_id"],
    "outage_phases": ["phase_id"],
    "schedule_versions": ["schedule_version_id"],
    "wbs": ["wbs_id"],
    "schedule_tasks": ["task_id"],
    "dependencies": ["dependency_id"],
    "resources": ["resource_id"],
    "resource_assignments": ["assignment_id"],
    "calendars": ["calendar_id"],
    "activity_codes": ["activity_code_id"],
    "task_activity_codes": ["task_id", "activity_code_id"],
    "task_constraints": ["task_constraint_id"],
    "work_packages": ["work_package_id"],
    "scope_change_events": ["scope_change_id"],
    "delay_events": ["delay_event_id"],
    "work_windows": ["work_window_id"],
    "clearances": ["clearance_id"],
}


FOREIGN_KEYS: dict[str, list[tuple[str, str, str]]] = {
    "outage_phases": [("outage_id", "outages", "outage_id")],
    "schedule_versions": [("outage_id", "outages", "outage_id")],
    "wbs": [
        ("outage_id", "outages", "outage_id"),
        ("schedule_version_id", "schedule_versions", "schedule_version_id"),
    ],
    "schedule_tasks": [
        ("outage_id", "outages", "outage_id"),
        ("schedule_version_id", "schedule_versions", "schedule_version_id"),
        ("wbs_id", "wbs", "wbs_id"),
        ("calendar_id", "calendars", "calendar_id"),
    ],
    "dependencies": [
        ("schedule_version_id", "schedule_versions", "schedule_version_id"),
        ("predecessor_task_id", "schedule_tasks", "task_id"),
        ("successor_task_id", "schedule_tasks", "task_id"),
    ],
    "resources": [("calendar_id", "calendars", "calendar_id")],
    "resource_assignments": [
        ("schedule_version_id", "schedule_versions", "schedule_version_id"),
        ("task_id", "schedule_tasks", "task_id"),
        ("resource_id", "resources", "resource_id"),
    ],
    "task_activity_codes": [
        ("task_id", "schedule_tasks", "task_id"),
        ("activity_code_id", "activity_codes", "activity_code_id"),
    ],
    "task_constraints": [("task_id", "schedule_tasks", "task_id")],
    "work_packages": [("outage_id", "outages", "outage_id")],
    "scope_change_events": [
        ("outage_id", "outages", "outage_id"),
        ("task_id", "schedule_tasks", "task_id"),
    ],
    "delay_events": [
        ("outage_id", "outages", "outage_id"),
        ("task_id", "schedule_tasks", "task_id"),
    ],
    "work_windows": [("outage_id", "outages", "outage_id")],
    "clearances": [("outage_id", "outages", "outage_id")],
}


DATETIME_COLUMNS = {
    table_name: [
        column_name
        for column_name, dtype in table_schema.items()
        if dtype == "datetime64[ns]"
    ]
    for table_name, table_schema in PANDAS_PHYSICAL_SCHEMA.items()
}


NUMERIC_COLUMNS = {
    table_name: [
        column_name
        for column_name, dtype in table_schema.items()
        if dtype in {"Float64", "Int64"}
    ]
    for table_name, table_schema in PANDAS_PHYSICAL_SCHEMA.items()
}


BOOLEAN_COLUMNS = {
    table_name: [
        column_name
        for column_name, dtype in table_schema.items()
        if dtype == "boolean"
    ]
    for table_name, table_schema in PANDAS_PHYSICAL_SCHEMA.items()
}


STRING_COLUMNS = {
    table_name: [
        column_name
        for column_name, dtype in table_schema.items()
        if dtype == "string"
    ]
    for table_name, table_schema in PANDAS_PHYSICAL_SCHEMA.items()
}


def empty_table_schema(table_name: str) -> dict[str, Any]:
    return PANDAS_PHYSICAL_SCHEMA[table_name].copy()
