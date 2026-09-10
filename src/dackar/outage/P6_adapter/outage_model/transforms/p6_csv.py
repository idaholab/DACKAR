from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from outage_model.dataset import OutageDataset
from outage_model.schema import PANDAS_PHYSICAL_SCHEMA
from outage_model.transforms.common import (
    build_canonical_id,
    ensure_columns,
    read_csv_if_exists,
    standardize_columns,
)


class P6CsvTransformer:
    """Transform a simple P6-like CSV export into the canonical outage dataset."""

    def __init__(
        self,
        outage_id: str,
        outage_name: str,
        schedule_version_id: str,
        version_name: str,
        version_type: str = "baseline",
        source_system: str = "primavera_p6",
    ) -> None:
        self.outage_id = outage_id
        self.outage_name = outage_name
        self.schedule_version_id = schedule_version_id
        self.version_name = version_name
        self.version_type = version_type
        self.source_system = source_system

    def transform_directory(self, base_dir: Path) -> OutageDataset:
        activities = standardize_columns(read_csv_if_exists(base_dir / "activities.csv"))
        relationships = standardize_columns(read_csv_if_exists(base_dir / "relationships.csv"))
        wbs = standardize_columns(read_csv_if_exists(base_dir / "wbs.csv"))
        resources = standardize_columns(read_csv_if_exists(base_dir / "resources.csv"))
        assignments = standardize_columns(read_csv_if_exists(base_dir / "assignments.csv"))
        calendars = standardize_columns(read_csv_if_exists(base_dir / "calendars.csv"))
        activity_codes = standardize_columns(read_csv_if_exists(base_dir / "activity_codes.csv"))
        task_activity_codes = standardize_columns(read_csv_if_exists(base_dir / "task_activity_codes.csv"))

        dataset = OutageDataset(
            outages=self._build_outages(activities),
            schedule_versions=self._build_schedule_versions(),
            wbs=self._build_wbs(wbs),
            schedule_tasks=self._build_schedule_tasks(activities),
            dependencies=self._build_dependencies(relationships),
            resources=self._build_resources(resources),
            resource_assignments=self._build_resource_assignments(assignments),
            calendars=self._build_calendars(calendars),
            activity_codes=self._build_activity_codes(activity_codes),
            task_activity_codes=self._build_task_activity_codes(task_activity_codes),
        )
        return dataset.apply_schema()

    def _build_outages(self, activities: pd.DataFrame) -> pd.DataFrame:
        activities = ensure_columns(
            activities,
            ["planned_start", "planned_finish", "actual_start", "actual_finish"],
        )
        planned_start = pd.to_datetime(activities["planned_start"], errors="coerce").min()
        planned_finish = pd.to_datetime(activities["planned_finish"], errors="coerce").max()
        actual_start = pd.to_datetime(activities["actual_start"], errors="coerce").min()
        actual_finish = pd.to_datetime(activities["actual_finish"], errors="coerce").max()

        return pd.DataFrame(
            [
                {
                    "outage_id": self.outage_id,
                    "outage_name": self.outage_name,
                    "outage_type": "refueling",
                    "start_planned": planned_start,
                    "finish_planned": planned_finish,
                    "start_actual": actual_start,
                    "finish_actual": actual_finish,
                    "status": "loaded",
                    "description": "Derived from P6-like activity export",
                }
            ]
        )

    def _build_schedule_versions(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "schedule_version_id": self.schedule_version_id,
                    "outage_id": self.outage_id,
                    "version_name": self.version_name,
                    "version_type": self.version_type,
                    "source_system": self.source_system,
                    "source_file": pd.NA,
                    "is_official": True,
                }
            ]
        )

    def _build_wbs(self, wbs: pd.DataFrame) -> pd.DataFrame:
        if wbs.empty:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["wbs"].keys())

        wbs = ensure_columns(wbs, ["wbs_id", "parent_wbs_id", "wbs_code", "wbs_name", "wbs_path", "level"])

        return pd.DataFrame(
            {
                "wbs_id": [build_canonical_id(self.outage_id, self.schedule_version_id, value) for value in wbs["wbs_id"]],
                "outage_id": self.outage_id,
                "schedule_version_id": self.schedule_version_id,
                "parent_wbs_id": [
                    build_canonical_id(self.outage_id, self.schedule_version_id, value) if pd.notna(value) and str(value).strip() else pd.NA
                    for value in wbs["parent_wbs_id"]
                ],
                "wbs_code": wbs["wbs_code"],
                "wbs_name": wbs["wbs_name"],
                "wbs_path": wbs["wbs_path"],
                "level": wbs["level"],
            }
        )

    def _build_schedule_tasks(self, activities: pd.DataFrame) -> pd.DataFrame:
        if activities.empty:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["schedule_tasks"].keys())

        activities = ensure_columns(
            activities,
            [
                "activity_id",
                "activity_name",
                "task_type",
                "milestone_flag",
                "planned_start",
                "planned_finish",
                "actual_start",
                "actual_finish",
                "planned_duration_hours",
                "remaining_duration_hours",
                "total_float_hours",
                "free_float_hours",
                "critical_flag",
                "status",
                "percent_complete",
                "physical_percent_complete",
                "calendar_id",
                "primary_constraint_type",
                "primary_constraint_date",
                "scope_origin",
                "wbs_id",
            ],
        )

        return pd.DataFrame(
            {
                "task_id": [build_canonical_id(self.outage_id, self.schedule_version_id, value) for value in activities["activity_id"]],
                "outage_id": self.outage_id,
                "schedule_version_id": self.schedule_version_id,
                "wbs_id": [
                    build_canonical_id(self.outage_id, self.schedule_version_id, value) if pd.notna(value) else pd.NA
                    for value in activities["wbs_id"]
                ],
                "task_code": activities["activity_id"].astype("string"),
                "task_name": activities["activity_name"],
                "task_type": activities["task_type"],
                "milestone_flag": activities["milestone_flag"],
                "planned_start": pd.to_datetime(activities["planned_start"], errors="coerce"),
                "planned_finish": pd.to_datetime(activities["planned_finish"], errors="coerce"),
                "actual_start": pd.to_datetime(activities["actual_start"], errors="coerce"),
                "actual_finish": pd.to_datetime(activities["actual_finish"], errors="coerce"),
                "planned_duration_hours": activities["planned_duration_hours"],
                "remaining_duration_hours": activities["remaining_duration_hours"],
                "total_float_hours": activities["total_float_hours"],
                "free_float_hours": activities["free_float_hours"],
                "critical_flag": activities["critical_flag"],
                "status": activities["status"],
                "percent_complete": activities["percent_complete"],
                "physical_percent_complete": activities["physical_percent_complete"],
                "calendar_id": activities["calendar_id"],
                "primary_constraint_type": activities["primary_constraint_type"],
                "primary_constraint_date": pd.to_datetime(activities["primary_constraint_date"], errors="coerce"),
                "scope_origin": activities["scope_origin"],
                "source_system": self.source_system,
                "source_record_id": activities["activity_id"].astype("string"),
            }
        )

    def _build_dependencies(self, relationships: pd.DataFrame) -> pd.DataFrame:
        if relationships.empty:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["dependencies"].keys())

        relationships = ensure_columns(
            relationships,
            ["relationship_id", "predecessor_activity_id", "successor_activity_id", "relationship_type", "lag_hours", "lag_calendar_id", "driving_flag", "external_link_flag"],
        )

        return pd.DataFrame(
            {
                "dependency_id": [build_canonical_id(self.schedule_version_id, value) for value in relationships["relationship_id"]],
                "schedule_version_id": self.schedule_version_id,
                "predecessor_task_id": [build_canonical_id(self.outage_id, self.schedule_version_id, value) for value in relationships["predecessor_activity_id"]],
                "successor_task_id": [build_canonical_id(self.outage_id, self.schedule_version_id, value) for value in relationships["successor_activity_id"]],
                "relationship_type": relationships["relationship_type"],
                "lag_hours": relationships["lag_hours"],
                "lag_calendar_id": relationships["lag_calendar_id"],
                "driving_flag": relationships["driving_flag"],
                "external_link_flag": relationships["external_link_flag"],
            }
        )

    def _build_resources(self, resources: pd.DataFrame) -> pd.DataFrame:
        if resources.empty:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["resources"].keys())

        resources = ensure_columns(resources, ["resource_id", "resource_code", "resource_name", "resource_type", "craft", "vendor", "org_unit", "calendar_id"])

        return pd.DataFrame(
            {
                "resource_id": resources["resource_id"].astype("string"),
                "resource_code": resources["resource_code"],
                "resource_name": resources["resource_name"],
                "resource_type": resources["resource_type"],
                "craft": resources["craft"],
                "vendor": resources["vendor"],
                "org_unit": resources["org_unit"],
                "calendar_id": resources["calendar_id"],
            }
        )

    def _build_resource_assignments(self, assignments: pd.DataFrame) -> pd.DataFrame:
        if assignments.empty:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["resource_assignments"].keys())

        assignments = ensure_columns(assignments, ["assignment_id", "activity_id", "resource_id", "role_name", "planned_units", "actual_units", "remaining_units", "planned_cost", "actual_cost"])

        return pd.DataFrame(
            {
                "assignment_id": [build_canonical_id(self.schedule_version_id, value) for value in assignments["assignment_id"]],
                "schedule_version_id": self.schedule_version_id,
                "task_id": [build_canonical_id(self.outage_id, self.schedule_version_id, value) for value in assignments["activity_id"]],
                "resource_id": assignments["resource_id"].astype("string"),
                "role_name": assignments["role_name"],
                "planned_units": assignments["planned_units"],
                "actual_units": assignments["actual_units"],
                "remaining_units": assignments["remaining_units"],
                "planned_cost": assignments["planned_cost"],
                "actual_cost": assignments["actual_cost"],
            }
        )

    def _build_calendars(self, calendars: pd.DataFrame) -> pd.DataFrame:
        if calendars.empty:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["calendars"].keys())

        calendars = ensure_columns(calendars, ["calendar_id", "calendar_name", "calendar_type", "timezone", "work_pattern_json"])
        calendars = calendars[list(PANDAS_PHYSICAL_SCHEMA["calendars"].keys())].copy()

        def _parse_json(value):
            if isinstance(value, dict):
                return value
            if pd.isna(value) or str(value).strip() == "":
                return {}
            try:
                return json.loads(str(value))
            except (json.JSONDecodeError, TypeError):
                return {"raw": str(value)}

        calendars["work_pattern_json"] = calendars["work_pattern_json"].map(_parse_json)
        return calendars

    def _build_activity_codes(self, activity_codes: pd.DataFrame) -> pd.DataFrame:
        if activity_codes.empty:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["activity_codes"].keys())

        activity_codes = ensure_columns(activity_codes, ["activity_code_id", "code_type", "code_value", "code_description"])
        return activity_codes[list(PANDAS_PHYSICAL_SCHEMA["activity_codes"].keys())].copy()

    def _build_task_activity_codes(self, task_activity_codes: pd.DataFrame) -> pd.DataFrame:
        if task_activity_codes.empty:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["task_activity_codes"].keys())

        task_activity_codes = ensure_columns(task_activity_codes, ["activity_id", "activity_code_id"])
        return pd.DataFrame(
            {
                "task_id": [build_canonical_id(self.outage_id, self.schedule_version_id, value) for value in task_activity_codes["activity_id"]],
                "activity_code_id": task_activity_codes["activity_code_id"].astype("string"),
            }
        )
