"""
Tests for outage_model.transforms.common and outage_model.dataset utilities.
"""
from __future__ import annotations

import pandas as pd
import pytest

from outage_model.transforms.common import (
    build_canonical_id,
    ensure_columns,
    normalize_name,
    normalize_text_value,
    standardize_columns,
)
from outage_model.dataset import OutageDataset, _coerce_boolean_series, coerce_dataframe_to_schema
from outage_model.schema import PANDAS_PHYSICAL_SCHEMA


# ---------------------------------------------------------------------------
# normalize_name
# ---------------------------------------------------------------------------

class TestNormalizeName:
    def test_lowercase(self):
        assert normalize_name("TaskID") == "taskid"

    def test_spaces_to_underscores(self):
        assert normalize_name("task name") == "task_name"

    def test_hyphens_to_underscores(self):
        assert normalize_name("task-name") == "task_name"

    def test_slashes_to_underscores(self):
        assert normalize_name("task/name") == "task_name"

    def test_strips_whitespace(self):
        assert normalize_name("  task_name  ") == "task_name"

    def test_combined(self):
        assert normalize_name(" Task/Name-Here ") == "task_name_here"


# ---------------------------------------------------------------------------
# normalize_text_value
# ---------------------------------------------------------------------------

class TestNormalizeTextValue:
    @pytest.mark.parametrize("value", ["null", "NULL", "none", "NONE", "nan", "NaN", "na", "NA", ""])
    def test_null_like_returns_na(self, value):
        assert pd.isna(normalize_text_value(value))

    def test_none_input_returns_na(self):
        assert pd.isna(normalize_text_value(None))

    def test_valid_text_returned(self):
        assert normalize_text_value("  hello  ") == "hello"

    def test_zero_string_not_null(self):
        assert normalize_text_value("0") == "0"


# ---------------------------------------------------------------------------
# build_canonical_id
# ---------------------------------------------------------------------------

class TestBuildCanonicalId:
    def test_simple_two_parts(self):
        assert build_canonical_id("outage1", "xer") == "outage1:xer"

    def test_three_parts(self):
        assert build_canonical_id("outage1", "v1", "T001") == "outage1:v1:T001"

    def test_none_parts_stripped(self):
        assert build_canonical_id("outage1", None, "T001") == "outage1:T001"

    def test_empty_string_parts_stripped(self):
        assert build_canonical_id("outage1", "", "T001") == "outage1:T001"

    def test_whitespace_only_parts_stripped(self):
        assert build_canonical_id("outage1", "  ", "T001") == "outage1:T001"

    def test_single_part(self):
        assert build_canonical_id("outage1") == "outage1"

    def test_all_empty_returns_empty(self):
        assert build_canonical_id("", None, "  ") == ""


# ---------------------------------------------------------------------------
# ensure_columns
# ---------------------------------------------------------------------------

class TestEnsureColumns:
    def test_adds_missing_columns(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = ensure_columns(df, ["a", "b", "c"])
        assert "b" in result.columns
        assert "c" in result.columns

    def test_missing_columns_are_na(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = ensure_columns(df, ["a", "b"])
        assert result["b"].isna().all()

    def test_existing_columns_unchanged(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = ensure_columns(df, ["a", "b", "c"])
        pd.testing.assert_series_equal(result["a"], df["a"])
        pd.testing.assert_series_equal(result["b"], df["b"])

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"a": [1]})
        ensure_columns(df, ["b"])
        assert "b" not in df.columns


# ---------------------------------------------------------------------------
# standardize_columns
# ---------------------------------------------------------------------------

class TestStandardizeColumns:
    def test_renames_to_snake_case(self):
        df = pd.DataFrame(columns=["Task ID", "Task-Name", "WBS/Code"])
        result = standardize_columns(df)
        assert list(result.columns) == ["task_id", "task_name", "wbs_code"]


# ---------------------------------------------------------------------------
# _coerce_boolean_series
# ---------------------------------------------------------------------------

class TestCoerceBooleanSeries:
    @pytest.mark.parametrize("value,expected", [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("y", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("n", False),
    ])
    def test_string_mappings(self, value, expected):
        result = _coerce_boolean_series(pd.Series([value]))
        assert result.iloc[0] == expected

    def test_bool_passthrough(self):
        result = _coerce_boolean_series(pd.Series([True, False]))
        assert result.iloc[0] == True
        assert result.iloc[1] == False

    def test_na_input_stays_na(self):
        result = _coerce_boolean_series(pd.Series([pd.NA]))
        assert pd.isna(result.iloc[0])

    def test_unrecognized_value_becomes_na(self):
        result = _coerce_boolean_series(pd.Series(["maybe"]))
        assert pd.isna(result.iloc[0])

    def test_mixed_series(self):
        s = pd.Series(["true", "false", pd.NA, "yes", "0"])
        result = _coerce_boolean_series(s)
        assert result.iloc[0] == True
        assert result.iloc[1] == False
        assert pd.isna(result.iloc[2])
        assert result.iloc[3] == True
        assert result.iloc[4] == False

    def test_output_dtype_is_boolean(self):
        result = _coerce_boolean_series(pd.Series(["true", "false"]))
        assert str(result.dtype) == "boolean"


# ---------------------------------------------------------------------------
# coerce_dataframe_to_schema
# ---------------------------------------------------------------------------

class TestCoerceDataframeToSchema:
    """
    coerce_dataframe_to_schema dispatches by table_name to pick up the correct
    column type lists.  Tests use real canonical table names so the lookup works.
    """

    def test_adds_missing_columns(self):
        # Use a known table; supply only outage_id — all other columns should appear.
        schema = PANDAS_PHYSICAL_SCHEMA["outages"]
        df = pd.DataFrame({"outage_id": ["O1"]})
        result = coerce_dataframe_to_schema(df, schema, "outages")
        assert set(schema.keys()).issubset(set(result.columns))

    def test_drops_extra_columns(self):
        schema = PANDAS_PHYSICAL_SCHEMA["outages"]
        df = pd.DataFrame({"outage_id": ["O1"], "__extra__": [99]})
        result = coerce_dataframe_to_schema(df, schema, "outages")
        assert "__extra__" not in result.columns

    def test_datetime_coercion(self):
        # planned_start is datetime64[ns] in schedule_tasks
        schema = PANDAS_PHYSICAL_SCHEMA["schedule_tasks"]
        df = pd.DataFrame({
            "task_id": ["t1"],
            "task_name": ["T"],
            "planned_start": ["2026-01-10 06:00"],
            "planned_finish": ["not-a-date"],
        })
        result = coerce_dataframe_to_schema(df, schema, "schedule_tasks")
        assert pd.notna(result["planned_start"].iloc[0])
        assert pd.isna(result["planned_finish"].iloc[0])

    def test_float64_nullable(self):
        # planned_duration_hours is Float64 in schedule_tasks
        schema = PANDAS_PHYSICAL_SCHEMA["schedule_tasks"]
        df = pd.DataFrame({
            "task_id": ["t1", "t2", "t3"],
            "task_name": ["A", "B", "C"],
            "planned_duration_hours": ["36.5", None, "bad"],
        })
        result = coerce_dataframe_to_schema(df, schema, "schedule_tasks")
        assert result["planned_duration_hours"].iloc[0] == pytest.approx(36.5)
        assert pd.isna(result["planned_duration_hours"].iloc[1])
        assert pd.isna(result["planned_duration_hours"].iloc[2])

    def test_int64_nullable(self):
        # level is Int64 in wbs
        schema = PANDAS_PHYSICAL_SCHEMA["wbs"]
        df = pd.DataFrame({
            "wbs_id": ["w1", "w2"],
            "wbs_name": ["Root", "Child"],
            "level": ["2", None],
        })
        result = coerce_dataframe_to_schema(df, schema, "wbs")
        assert result["level"].iloc[0] == 2
        assert pd.isna(result["level"].iloc[1])

    def test_boolean_coercion(self):
        # critical_flag is boolean in schedule_tasks
        schema = PANDAS_PHYSICAL_SCHEMA["schedule_tasks"]
        df = pd.DataFrame({
            "task_id": ["t1", "t2", "t3"],
            "task_name": ["A", "B", "C"],
            "critical_flag": ["true", "false", pd.NA],
        })
        result = coerce_dataframe_to_schema(df, schema, "schedule_tasks")
        assert result["critical_flag"].iloc[0] == True
        assert result["critical_flag"].iloc[1] == False
        assert pd.isna(result["critical_flag"].iloc[2])

    def test_string_coercion(self):
        # task_name is string in schedule_tasks
        schema = PANDAS_PHYSICAL_SCHEMA["schedule_tasks"]
        df = pd.DataFrame({"task_id": ["t1"], "task_name": ["hello"]})
        result = coerce_dataframe_to_schema(df, schema, "schedule_tasks")
        assert str(result["task_name"].dtype) == "string"


# ---------------------------------------------------------------------------
# OutageDataset
# ---------------------------------------------------------------------------

class TestOutageDataset:
    def test_default_construction_has_empty_dataframes(self):
        ds = OutageDataset()
        assert isinstance(ds.schedule_tasks, pd.DataFrame)
        assert ds.schedule_tasks.empty

    def test_as_dict_contains_all_schema_tables(self):
        ds = OutageDataset()
        result = ds.as_dict()
        assert set(result.keys()) == set(PANDAS_PHYSICAL_SCHEMA.keys())

    def test_apply_schema_adds_canonical_columns(self):
        ds = OutageDataset(
            schedule_tasks=pd.DataFrame([{"task_id": "t1", "task_name": "A task"}])
        )
        ds.apply_schema()
        expected_columns = set(PANDAS_PHYSICAL_SCHEMA["schedule_tasks"].keys())
        assert expected_columns.issubset(set(ds.schedule_tasks.columns))

    def test_apply_schema_returns_self(self):
        ds = OutageDataset()
        result = ds.apply_schema()
        assert result is ds
