from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .schema import (
    BOOLEAN_COLUMNS,
    DATETIME_COLUMNS,
    NUMERIC_COLUMNS,
    PANDAS_PHYSICAL_SCHEMA,
    STRING_COLUMNS,
)


@dataclass
class OutageDataset:
    outages: pd.DataFrame = field(default_factory=pd.DataFrame)
    outage_phases: pd.DataFrame = field(default_factory=pd.DataFrame)
    schedule_versions: pd.DataFrame = field(default_factory=pd.DataFrame)
    wbs: pd.DataFrame = field(default_factory=pd.DataFrame)
    schedule_tasks: pd.DataFrame = field(default_factory=pd.DataFrame)
    dependencies: pd.DataFrame = field(default_factory=pd.DataFrame)
    resources: pd.DataFrame = field(default_factory=pd.DataFrame)
    resource_assignments: pd.DataFrame = field(default_factory=pd.DataFrame)
    calendars: pd.DataFrame = field(default_factory=pd.DataFrame)
    activity_codes: pd.DataFrame = field(default_factory=pd.DataFrame)
    task_activity_codes: pd.DataFrame = field(default_factory=pd.DataFrame)
    task_constraints: pd.DataFrame = field(default_factory=pd.DataFrame)
    work_packages: pd.DataFrame = field(default_factory=pd.DataFrame)
    scope_change_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    delay_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    work_windows: pd.DataFrame = field(default_factory=pd.DataFrame)
    clearances: pd.DataFrame = field(default_factory=pd.DataFrame)

    def apply_schema(self) -> "OutageDataset":
        for table_name, schema in PANDAS_PHYSICAL_SCHEMA.items():
            df = getattr(self, table_name)
            setattr(self, table_name, coerce_dataframe_to_schema(df, schema, table_name))
        return self

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {table_name: getattr(self, table_name) for table_name in PANDAS_PHYSICAL_SCHEMA}



def coerce_dataframe_to_schema(
    df: pd.DataFrame,
    schema: dict[str, str],
    table_name: str | None = None,
) -> pd.DataFrame:
    df = df.copy()

    for column_name in schema:
        if column_name not in df.columns:
            df[column_name] = pd.NA

    df = df[list(schema.keys())]

    current_name = table_name or "table"

    for column_name in DATETIME_COLUMNS.get(current_name, []):
        df[column_name] = pd.to_datetime(df[column_name], errors="coerce")

    for column_name in NUMERIC_COLUMNS.get(current_name, []):
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
        target_dtype = schema[column_name]
        df[column_name] = df[column_name].astype(target_dtype)

    for column_name in BOOLEAN_COLUMNS.get(current_name, []):
        df[column_name] = _coerce_boolean_series(df[column_name])

    for column_name in STRING_COLUMNS.get(current_name, []):
        df[column_name] = df[column_name].astype("string")

    for column_name, dtype in schema.items():
        if dtype == "object":
            continue
        if dtype == "datetime64[ns]":
            continue
        if dtype in {"Float64", "Int64", "boolean", "string"}:
            continue
        df[column_name] = df[column_name].astype(dtype)

    return df



def _coerce_boolean_series(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }

    def convert(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        return mapping.get(normalized, pd.NA)

    return series.map(convert).astype("boolean")
