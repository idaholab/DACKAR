from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from outage_model.dataset import OutageDataset
from outage_model.schema import PANDAS_PHYSICAL_SCHEMA
from outage_model.transforms.common import build_canonical_id, ensure_columns, normalize_name


SECTION_TABLE = "%T"
SECTION_FIELDS = "%F"
SECTION_ROW = "%R"
SECTION_END = "%E"


@dataclass
class XERTable:
    name: str
    fields: list[str]
    rows: list[list[str]]

    def to_dataframe(self) -> pd.DataFrame:
        if not self.fields:
            return pd.DataFrame()
        padded_rows = [row + [None] * (len(self.fields) - len(row)) for row in self.rows]
        trimmed_rows = [row[: len(self.fields)] for row in padded_rows]
        return pd.DataFrame(trimmed_rows, columns=[normalize_name(field) for field in self.fields])


class XERParser:
    """Parse Primavera P6 XER files into table-name -> DataFrame."""

    def __init__(self, path: str | Path, encoding: str = "utf-8", errors: str = "replace") -> None:
        self.path = Path(path)
        self.encoding = encoding
        self.errors = errors

    def parse(self) -> dict[str, pd.DataFrame]:
        tables: dict[str, XERTable] = {}
        current_table_name: str | None = None
        current_fields: list[str] = []
        current_rows: list[list[str]] = []

        with self.path.open("r", encoding=self.encoding, errors=self.errors, newline="") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                marker = parts[0]

                if marker == SECTION_TABLE:
                    if current_table_name is not None:
                        tables[current_table_name] = XERTable(
                            name=current_table_name,
                            fields=current_fields,
                            rows=current_rows,
                        )
                    current_table_name = normalize_name(parts[1]) if len(parts) > 1 else "unknown"
                    current_fields = []
                    current_rows = []
                elif marker == SECTION_FIELDS:
                    current_fields = parts[1:]
                elif marker == SECTION_ROW:
                    current_rows.append(parts[1:])
                elif marker == SECTION_END:
                    if current_table_name is not None:
                        tables[current_table_name] = XERTable(
                            name=current_table_name,
                            fields=current_fields,
                            rows=current_rows,
                        )
                    current_table_name = None
                    current_fields = []
                    current_rows = []

        if current_table_name is not None:
            tables[current_table_name] = XERTable(
                name=current_table_name,
                fields=current_fields,
                rows=current_rows,
            )

        return {table_name: table.to_dataframe() for table_name, table in tables.items()}


class P6XERTransformer:
    """Transform a real Primavera P6 XER export into the canonical outage dataset.

    The transformer reads common P6 XER tables directly, including:
    PROJECT, PROJWBS, TASK, TASKPRED, RSRC, TASKRSRC, CALENDAR,
    ACTVCODE, TASKACTV, and ACTVTYPE when present.
    """

    def __init__(
        self,
        outage_id: str | None = None,
        outage_name: str | None = None,
        schedule_version_id: str | None = None,
        version_name: str | None = None,
        version_type: str = "working",
        source_system: str = "primavera_p6_xer",
        project_id: str | None = None,
    ) -> None:
        # Store user-supplied values separately so transform_file can be called
        # multiple times without the first call's resolved values polluting subsequent ones.
        self._user_outage_id = outage_id
        self._user_outage_name = outage_name
        self._user_schedule_version_id = schedule_version_id
        self._user_version_name = version_name
        self.version_type = version_type
        self.source_system = source_system
        self.project_id = str(project_id) if project_id is not None else None
        # Public attributes reflect the last resolved values (set after each transform_file call).
        self.outage_id = outage_id
        self.outage_name = outage_name
        self.schedule_version_id = schedule_version_id
        self.version_name = version_name

    def transform_file(self, xer_path: str | Path) -> OutageDataset:
        xer_path = Path(xer_path)
        tables = XERParser(xer_path).parse()

        projects = self._select_project_frame(tables.get("project", pd.DataFrame()))
        project_row = projects.iloc[0] if not projects.empty else pd.Series(dtype="object")

        # Resolve IDs from XER content only when not supplied by the caller.
        # Always start from the original user-supplied values (_user_*) so this
        # method can be called multiple times on different files safely.
        outage_id = self._user_outage_id or self._pick_first_value(project_row, ["proj_short_name", "proj_id", "proj_name"]) or xer_path.stem
        outage_name = self._user_outage_name or self._pick_first_value(project_row, ["proj_name", "proj_short_name"]) or outage_id
        schedule_version_id = self._user_schedule_version_id or build_canonical_id(outage_id, "xer")
        version_name = self._user_version_name or xer_path.stem

        # Update public attributes so callers can inspect the resolved values after the call.
        self.outage_id = str(outage_id)
        self.outage_name = str(outage_name)
        self.schedule_version_id = str(schedule_version_id)
        self.version_name = str(version_name)

        task_df = self._select_project_rows(tables.get("task", pd.DataFrame()), "proj_id")
        wbs_df = self._select_project_rows(tables.get("projwbs", pd.DataFrame()), "proj_id")
        pred_df = self._select_project_rows(tables.get("taskpred", pd.DataFrame()), None, task_df, "task_id")
        rsrc_df = tables.get("rsrc", pd.DataFrame()).copy()
        taskrsrc_df = self._select_project_rows(tables.get("taskrsrc", pd.DataFrame()), None, task_df, "task_id")
        calendar_df = tables.get("calendar", pd.DataFrame()).copy()
        actvcode_df = tables.get("actvcode", pd.DataFrame()).copy()
        taskactv_df = self._select_project_rows(tables.get("taskactv", pd.DataFrame()), None, task_df, "task_id")
        actvtype_df = tables.get("actvtype", pd.DataFrame()).copy()

        dataset = OutageDataset(
            outages=self._build_outages(project_row, task_df),
            schedule_versions=self._build_schedule_versions(project_row, xer_path),
            wbs=self._build_wbs(wbs_df),
            schedule_tasks=self._build_schedule_tasks(task_df),
            dependencies=self._build_dependencies(pred_df),
            resources=self._build_resources(rsrc_df),
            resource_assignments=self._build_resource_assignments(taskrsrc_df),
            calendars=self._build_calendars(calendar_df),
            activity_codes=self._build_activity_codes(actvcode_df, actvtype_df),
            task_activity_codes=self._build_task_activity_codes(taskactv_df),
            task_constraints=self._build_task_constraints(task_df),
        )
        return dataset.apply_schema()

    def _pick_first_value(self, row: pd.Series, keys: Iterable[str]) -> str | None:
        for key in keys:
            if key in row and pd.notna(row[key]) and str(row[key]).strip():
                return str(row[key]).strip()
        return None

    def _select_project_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        if self.project_id is None or "proj_id" not in df.columns:
            return df.iloc[[0]].copy()
        filtered = df[df["proj_id"].astype(str) == self.project_id].copy()
        return filtered if not filtered.empty else df.iloc[[0]].copy()

    def _select_project_rows(
        self,
        df: pd.DataFrame,
        direct_project_field: str | None,
        parent_df: pd.DataFrame | None = None,
        parent_key: str | None = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        if self.project_id is not None and direct_project_field and direct_project_field in df.columns:
            filtered = df[df[direct_project_field].astype(str) == self.project_id].copy()
            if not filtered.empty:
                return filtered
        if parent_df is not None and parent_key and parent_key in df.columns and not parent_df.empty and parent_key in parent_df.columns:
            allowed = set(parent_df[parent_key].astype(str).dropna())
            filtered = df[df[parent_key].astype(str).isin(allowed)].copy()
            if not filtered.empty:
                return filtered
        return df.copy()

    def _first_existing(self, row: pd.Series, names: Iterable[str]):
        for name in names:
            if name in row.index:
                return row[name]
        return pd.NA

    def _series_first_existing(self, df: pd.DataFrame, names: Iterable[str]) -> pd.Series:
        for name in names:
            if name in df.columns:
                return df[name]
        return pd.Series([pd.NA] * len(df), index=df.index)

    def _coerce_datetime_series(self, series: pd.Series) -> pd.Series:
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().any():
            return parsed
        # Fallback for common XER date styles: 15-Jan-26 08:00, 15-Jan-26, 15-Jan-2026
        for fmt in ("%d-%b-%y %H:%M", "%d-%b-%y", "%d-%b-%Y %H:%M", "%d-%b-%Y"):
            candidate = pd.to_datetime(series, errors="coerce", format=fmt)
            if candidate.notna().any():
                return candidate
        return parsed

    def _build_outages(self, project_row: pd.Series, task_df: pd.DataFrame) -> pd.DataFrame:
        planned_start = self._coerce_datetime_series(self._series_first_existing(task_df, ["target_start_date", "early_start_date", "start_date"])) if not task_df.empty else pd.Series(dtype="datetime64[ns]")
        planned_finish = self._coerce_datetime_series(self._series_first_existing(task_df, ["target_end_date", "early_end_date", "end_date"])) if not task_df.empty else pd.Series(dtype="datetime64[ns]")
        actual_start = self._coerce_datetime_series(self._series_first_existing(task_df, ["act_start_date"])) if not task_df.empty else pd.Series(dtype="datetime64[ns]")
        actual_finish = self._coerce_datetime_series(self._series_first_existing(task_df, ["act_end_date"])) if not task_df.empty else pd.Series(dtype="datetime64[ns]")

        return pd.DataFrame([
            {
                "outage_id": self.outage_id,
                "plant_id": self._pick_first_value(project_row, ["proj_short_name", "obs_id"]),
                "unit_id": self._pick_first_value(project_row, ["proj_short_name"]),
                "outage_name": self.outage_name,
                "outage_type": "refueling",
                "start_planned": planned_start.min() if not planned_start.empty else pd.NaT,
                "finish_planned": planned_finish.max() if not planned_finish.empty else pd.NaT,
                "start_actual": actual_start.min() if not actual_start.empty else pd.NaT,
                "finish_actual": actual_finish.max() if not actual_finish.empty else pd.NaT,
                "status": self._pick_first_value(project_row, ["status_code"]) or "loaded",
                "description": f"Derived from XER project {self._pick_first_value(project_row, ['proj_name', 'proj_short_name']) or self.outage_id}",
            }
        ])

    def _build_schedule_versions(self, project_row: pd.Series, xer_path: Path) -> pd.DataFrame:
        status_date = self._coerce_datetime_series(pd.Series([self._first_existing(project_row, ["last_recalc_date", "plan_end_date", "scd_end_date"])]))
        created_at = self._coerce_datetime_series(pd.Series([self._first_existing(project_row, ["create_date", "update_date"])]))
        return pd.DataFrame([
            {
                "schedule_version_id": self.schedule_version_id,
                "outage_id": self.outage_id,
                "version_name": self.version_name,
                "version_type": self.version_type,
                "status_date": status_date.iloc[0] if not status_date.empty else pd.NaT,
                "created_at": created_at.iloc[0] if not created_at.empty else pd.NaT,
                "source_system": self.source_system,
                "source_file": str(xer_path.name),
                "is_official": True,
            }
        ])

    def _build_wbs(self, wbs: pd.DataFrame) -> pd.DataFrame:
        if wbs.empty:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["wbs"].keys())
        wbs = ensure_columns(wbs, ["wbs_id", "parent_wbs_id", "wbs_short_name", "wbs_name", "seq_num"])
        levels = wbs["wbs_name"].fillna("").astype(str).str.count("\\.") + 1
        return pd.DataFrame({
            "wbs_id": [build_canonical_id(self.outage_id, self.schedule_version_id, value) for value in wbs["wbs_id"].astype(str)],
            "outage_id": self.outage_id,
            "schedule_version_id": self.schedule_version_id,
            "parent_wbs_id": [build_canonical_id(self.outage_id, self.schedule_version_id, value) if pd.notna(value) and str(value).strip() else pd.NA for value in wbs["parent_wbs_id"]],
            "wbs_code": self._series_first_existing(wbs, ["wbs_short_name", "wbs_name"]).astype("string"),
            "wbs_name": self._series_first_existing(wbs, ["wbs_name", "wbs_short_name"]).astype("string"),
            "wbs_path": self._series_first_existing(wbs, ["wbs_name"]).astype("string"),
            "level": pd.to_numeric(self._series_first_existing(wbs, ["seq_num"]), errors="coerce").fillna(levels),
        })

    def _build_schedule_tasks(self, task_df: pd.DataFrame) -> pd.DataFrame:
        if task_df.empty:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["schedule_tasks"].keys())
        task_df = ensure_columns(task_df, [
            "task_id", "wbs_id", "task_code", "task_name", "task_type", "status_code",
            "target_start_date", "target_end_date", "act_start_date", "act_end_date",
            "target_drtn_hr_cnt", "remain_drtn_hr_cnt", "total_float_hr_cnt", "free_float_hr_cnt",
            "phys_complete_pct", "complete_pct_type", "clndr_id", "cstr_type", "cstr_date",
        ])
        milestone_types = {"tt_mile", "tt_finmile", "tt_startmile", "tt_wt_mile"}
        task_type = task_df["task_type"].astype("string").str.lower()
        status_map = {
            "tk_notstart": "not_started",
            "tk_active": "in_progress",
            "tk_complete": "complete",
        }
        # P6 stores duration-based and physical percent complete separately.
        duration_pct = pd.to_numeric(self._series_first_existing(task_df, ["complete_pct", "target_complete_pct"]), errors="coerce")
        physical_pct = pd.to_numeric(self._series_first_existing(task_df, ["phys_complete_pct"]), errors="coerce")

        return pd.DataFrame({
            "task_id": [build_canonical_id(self.outage_id, self.schedule_version_id, value) for value in task_df["task_id"].astype(str)],
            "outage_id": self.outage_id,
            "schedule_version_id": self.schedule_version_id,
            "wbs_id": [build_canonical_id(self.outage_id, self.schedule_version_id, value) if pd.notna(value) and str(value).strip() else pd.NA for value in task_df["wbs_id"]],
            "task_code": self._series_first_existing(task_df, ["task_code", "task_id"]).astype("string"),
            "task_name": self._series_first_existing(task_df, ["task_name"]).astype("string"),
            "task_type": task_df["task_type"].astype("string"),
            "milestone_flag": task_type.isin(milestone_types),
            "planned_start": self._coerce_datetime_series(self._series_first_existing(task_df, ["target_start_date", "early_start_date"])),
            "planned_finish": self._coerce_datetime_series(self._series_first_existing(task_df, ["target_end_date", "early_end_date"])),
            "actual_start": self._coerce_datetime_series(self._series_first_existing(task_df, ["act_start_date"])),
            "actual_finish": self._coerce_datetime_series(self._series_first_existing(task_df, ["act_end_date"])),
            "planned_duration_hours": pd.to_numeric(self._series_first_existing(task_df, ["target_drtn_hr_cnt", "orig_drtn_hr_cnt", "remain_drtn_hr_cnt"]), errors="coerce"),
            "remaining_duration_hours": pd.to_numeric(self._series_first_existing(task_df, ["remain_drtn_hr_cnt"]), errors="coerce"),
            "total_float_hours": pd.to_numeric(self._series_first_existing(task_df, ["total_float_hr_cnt", "total_float_path"]), errors="coerce"),
            "free_float_hours": pd.to_numeric(self._series_first_existing(task_df, ["free_float_hr_cnt"]), errors="coerce"),
            "critical_flag": pd.to_numeric(self._series_first_existing(task_df, ["total_float_hr_cnt"]), errors="coerce").apply(lambda v: bool(v <= 0) if pd.notna(v) else pd.NA),
            "status": task_df["status_code"].astype("string").str.lower().map(status_map).fillna("unknown"),
            "percent_complete": duration_pct,
            "physical_percent_complete": physical_pct,
            "calendar_id": self._series_first_existing(task_df, ["clndr_id"]).astype("string"),
            "primary_constraint_type": self._series_first_existing(task_df, ["cstr_type"]).astype("string"),
            "primary_constraint_date": self._coerce_datetime_series(self._series_first_existing(task_df, ["cstr_date"])),
            "scope_origin": pd.Series(["baseline"] * len(task_df), index=task_df.index),
            "source_system": pd.Series([self.source_system] * len(task_df), index=task_df.index),
            "source_record_id": task_df["task_id"].astype("string"),
        })

    def _build_dependencies(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        if pred_df.empty:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["dependencies"].keys())
        pred_df = ensure_columns(pred_df, ["task_pred_id", "task_id", "pred_task_id", "pred_type", "lag_hr_cnt"])
        return pd.DataFrame({
            "dependency_id": [build_canonical_id(self.schedule_version_id, value) for value in pred_df["task_pred_id"].astype(str)],
            "schedule_version_id": self.schedule_version_id,
            "predecessor_task_id": [build_canonical_id(self.outage_id, self.schedule_version_id, value) for value in pred_df["pred_task_id"].astype(str)],
            "successor_task_id": [build_canonical_id(self.outage_id, self.schedule_version_id, value) for value in pred_df["task_id"].astype(str)],
            "relationship_type": self._series_first_existing(pred_df, ["pred_type"]).astype("string").str.upper(),
            "lag_hours": pd.to_numeric(self._series_first_existing(pred_df, ["lag_hr_cnt"]), errors="coerce"),
            "lag_calendar_id": self._series_first_existing(pred_df, ["clndr_id"]).astype("string"),
            "driving_flag": pd.NA,
            "external_link_flag": pd.NA,
        })

    def _build_resources(self, rsrc_df: pd.DataFrame) -> pd.DataFrame:
        if rsrc_df.empty:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["resources"].keys())
        rsrc_df = ensure_columns(rsrc_df, ["rsrc_id", "rsrc_short_name", "rsrc_name", "rsrc_type", "clndr_id"])
        return pd.DataFrame({
            "resource_id": [str(value) for value in rsrc_df["rsrc_id"].astype(str)],
            "resource_code": self._series_first_existing(rsrc_df, ["rsrc_short_name"]).astype("string"),
            "resource_name": self._series_first_existing(rsrc_df, ["rsrc_name", "rsrc_short_name"]).astype("string"),
            "resource_type": self._series_first_existing(rsrc_df, ["rsrc_type"]).astype("string"),
            "craft": pd.NA,
            "vendor": pd.NA,
            "org_unit": pd.NA,
            "calendar_id": self._series_first_existing(rsrc_df, ["clndr_id"]).astype("string"),
        })

    def _build_resource_assignments(self, taskrsrc_df: pd.DataFrame) -> pd.DataFrame:
        if taskrsrc_df.empty:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["resource_assignments"].keys())
        taskrsrc_df = ensure_columns(taskrsrc_df, ["taskrsrc_id", "task_id", "rsrc_id", "target_qty", "act_reg_qty", "remain_qty", "target_cost", "act_reg_cost"])
        return pd.DataFrame({
            "assignment_id": [build_canonical_id(self.schedule_version_id, value) for value in taskrsrc_df["taskrsrc_id"].astype(str)],
            "schedule_version_id": self.schedule_version_id,
            "task_id": [build_canonical_id(self.outage_id, self.schedule_version_id, value) for value in taskrsrc_df["task_id"].astype(str)],
            "resource_id": taskrsrc_df["rsrc_id"].astype("string"),
            "role_name": pd.NA,
            "planned_units": pd.to_numeric(self._series_first_existing(taskrsrc_df, ["target_qty"]), errors="coerce"),
            "actual_units": pd.to_numeric(self._series_first_existing(taskrsrc_df, ["act_reg_qty", "act_ot_qty"]), errors="coerce"),
            "remaining_units": pd.to_numeric(self._series_first_existing(taskrsrc_df, ["remain_qty"]), errors="coerce"),
            "planned_cost": pd.to_numeric(self._series_first_existing(taskrsrc_df, ["target_cost"]), errors="coerce"),
            "actual_cost": pd.to_numeric(self._series_first_existing(taskrsrc_df, ["act_reg_cost", "act_ot_cost"]), errors="coerce"),
        })

    def _build_calendars(self, calendar_df: pd.DataFrame) -> pd.DataFrame:
        if calendar_df.empty:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["calendars"].keys())
        calendar_df = ensure_columns(calendar_df, ["clndr_id", "clndr_name", "clndr_type", "clndr_data"])
        return pd.DataFrame({
            "calendar_id": calendar_df["clndr_id"].astype("string"),
            "calendar_name": self._series_first_existing(calendar_df, ["clndr_name"]).astype("string"),
            "calendar_type": self._series_first_existing(calendar_df, ["clndr_type"]).astype("string"),
            "timezone": pd.NA,
            "work_pattern_json": [
                {
                    "raw_calendar_data": value if pd.notna(value) else None,
                }
                for value in self._series_first_existing(calendar_df, ["clndr_data"]).tolist()
            ],
        })

    def _build_activity_codes(self, actvcode_df: pd.DataFrame, actvtype_df: pd.DataFrame) -> pd.DataFrame:
        if actvcode_df.empty:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["activity_codes"].keys())
        actvcode_df = ensure_columns(actvcode_df, ["actv_code_id", "actv_code_name", "short_name", "actv_code_type_id"])
        code_type_lookup: dict[str, str] = {}
        if not actvtype_df.empty:
            actvtype_df = ensure_columns(actvtype_df, ["actv_code_type_id", "actv_code_type", "actv_short_len"])
            code_type_lookup = {
                str(row["actv_code_type_id"]): str(row["actv_code_type"])
                for _, row in actvtype_df.iterrows()
                if pd.notna(row["actv_code_type_id"]) and pd.notna(row.get("actv_code_type"))
            }
        return pd.DataFrame({
            "activity_code_id": actvcode_df["actv_code_id"].astype("string"),
            "code_type": [code_type_lookup.get(str(value), str(value)) if pd.notna(value) else pd.NA for value in actvcode_df["actv_code_type_id"]],
            "code_value": self._series_first_existing(actvcode_df, ["short_name", "actv_code_name"]).astype("string"),
            "code_description": self._series_first_existing(actvcode_df, ["actv_code_name"]).astype("string"),
        })

    def _build_task_activity_codes(self, taskactv_df: pd.DataFrame) -> pd.DataFrame:
        if taskactv_df.empty:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["task_activity_codes"].keys())
        taskactv_df = ensure_columns(taskactv_df, ["task_id", "actv_code_id"])
        return pd.DataFrame({
            "task_id": [build_canonical_id(self.outage_id, self.schedule_version_id, value) for value in taskactv_df["task_id"].astype(str)],
            "activity_code_id": taskactv_df["actv_code_id"].astype("string"),
        })

    def _build_task_constraints(self, task_df: pd.DataFrame) -> pd.DataFrame:
        if task_df.empty or "cstr_type" not in task_df.columns:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["task_constraints"].keys())
        constrained = task_df[task_df["cstr_type"].notna() & (task_df["cstr_type"].astype(str).str.strip() != "")].copy()
        if constrained.empty:
            return pd.DataFrame(columns=PANDAS_PHYSICAL_SCHEMA["task_constraints"].keys())
        constrained = ensure_columns(constrained, ["task_id", "cstr_type", "cstr_date"])
        constraint_type = constrained["cstr_type"].astype("string")
        hard_codes = {"cso", "ceo", "mso", "mfo", "mson", "mfon"}
        return pd.DataFrame({
            "task_constraint_id": [build_canonical_id(self.schedule_version_id, "constraint", value) for value in constrained["task_id"].astype(str)],
            "task_id": [build_canonical_id(self.outage_id, self.schedule_version_id, value) for value in constrained["task_id"].astype(str)],
            "constraint_type": constraint_type,
            "constraint_date": self._coerce_datetime_series(self._series_first_existing(constrained, ["cstr_date"])),
            "reason": pd.NA,
            "hard_flag": constraint_type.str.lower().isin(hard_codes),
        })
