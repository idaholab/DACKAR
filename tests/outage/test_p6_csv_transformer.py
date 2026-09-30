"""
Tests for outage_model.transforms.p6_csv.P6CsvTransformer
and the load_mock_dataset convenience loader.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from outage_model.dataset import OutageDataset
from outage_model.loaders.example_loader import load_mock_dataset
from outage_model.transforms.p6_csv import P6CsvTransformer


OUTAGE_ID = "RFO-2026-U1"
VERSION_ID = "RFO-2026-U1:BL1"


def task_canonical_id(raw_id: str) -> str:
    return f"{OUTAGE_ID}:{VERSION_ID}:{raw_id}"


def wbs_canonical_id(raw_id: str) -> str:
    return f"{OUTAGE_ID}:{VERSION_ID}:{raw_id}"


# ---------------------------------------------------------------------------
# Happy-path: dataset shape from mock CSV export
# ---------------------------------------------------------------------------

class TestCsvTransformerShape:
    @pytest.fixture(scope="class")
    def dataset(self, mock_csv_dir) -> OutageDataset:
        return load_mock_dataset(mock_csv_dir)

    def test_returns_outage_dataset(self, dataset):
        assert isinstance(dataset, OutageDataset)

    def test_outages_has_one_row(self, dataset):
        assert len(dataset.outages) == 1

    def test_schedule_versions_has_one_row(self, dataset):
        assert len(dataset.schedule_versions) == 1

    def test_schedule_tasks_row_count(self, dataset):
        assert len(dataset.schedule_tasks) == 4

    def test_dependencies_row_count(self, dataset):
        assert len(dataset.dependencies) == 3

    def test_wbs_row_count(self, dataset):
        assert len(dataset.wbs) == 4

    def test_resources_row_count(self, dataset):
        assert len(dataset.resources) == 2

    def test_resource_assignments_row_count(self, dataset):
        assert len(dataset.resource_assignments) == 2

    def test_calendars_row_count(self, dataset):
        assert len(dataset.calendars) == 1


# ---------------------------------------------------------------------------
# Outage identity
# ---------------------------------------------------------------------------

class TestCsvTransformerIdentity:
    @pytest.fixture(scope="class")
    def dataset(self, mock_csv_dir) -> OutageDataset:
        return load_mock_dataset(mock_csv_dir)

    def test_outage_id(self, dataset):
        assert dataset.outages["outage_id"].iloc[0] == OUTAGE_ID

    def test_outage_name(self, dataset):
        assert dataset.outages["outage_name"].iloc[0] == "Unit 1 Refueling Outage 2026"

    def test_schedule_version_id(self, dataset):
        assert dataset.schedule_versions["schedule_version_id"].iloc[0] == VERSION_ID

    def test_version_type_is_baseline(self, dataset):
        assert dataset.schedule_versions["version_type"].iloc[0] == "baseline"

    def test_source_system(self, dataset):
        assert dataset.schedule_versions["source_system"].iloc[0] == "primavera_p6"


# ---------------------------------------------------------------------------
# schedule_tasks field mapping
# ---------------------------------------------------------------------------

class TestCsvTransformerTaskFields:
    @pytest.fixture(scope="class")
    def tasks(self, mock_csv_dir) -> pd.DataFrame:
        return load_mock_dataset(mock_csv_dir).schedule_tasks

    def test_task_ids_use_canonical_format(self, tasks):
        expected = {task_canonical_id(aid) for aid in ["A1000", "A1100", "A1200", "A9000"]}
        assert set(tasks["task_id"]) == expected

    def test_task_code_is_original_activity_id(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("A1000")].iloc[0]
        assert row["task_code"] == "A1000"

    def test_source_record_id_is_original_activity_id(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("A1000")].iloc[0]
        assert row["source_record_id"] == "A1000"

    def test_milestone_flag_for_milestone_row(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("A9000")].iloc[0]
        assert row["milestone_flag"] == True

    def test_milestone_flag_for_task_row(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("A1000")].iloc[0]
        assert row["milestone_flag"] == False

    def test_status_complete(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("A1000")].iloc[0]
        assert row["status"] == "complete"

    def test_status_in_progress(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("A1200")].iloc[0]
        assert row["status"] == "in_progress"

    def test_status_not_started(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("A9000")].iloc[0]
        assert row["status"] == "not_started"

    def test_planned_duration_hours(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("A1000")].iloc[0]
        assert row["planned_duration_hours"] == pytest.approx(36.0)

    def test_critical_flag_coercion_from_string(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("A1000")].iloc[0]
        assert row["critical_flag"] == True

    def test_planned_start_is_datetime(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("A1000")].iloc[0]
        assert pd.notna(row["planned_start"])

    def test_wbs_id_uses_canonical_format(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("A1000")].iloc[0]
        assert row["wbs_id"] == wbs_canonical_id("WBS-100")


# ---------------------------------------------------------------------------
# dependencies field mapping
# ---------------------------------------------------------------------------

class TestCsvTransformerDependencies:
    @pytest.fixture(scope="class")
    def deps(self, mock_csv_dir) -> pd.DataFrame:
        return load_mock_dataset(mock_csv_dir).dependencies

    def test_relationship_types(self, deps):
        assert set(deps["relationship_type"]) == {"FS"}

    def test_lag_hours(self, deps):
        lags = sorted(deps["lag_hours"].dropna().tolist())
        assert lags == pytest.approx([0.0, 0.0, 8.0])

    def test_predecessor_ids_canonical(self, deps):
        assert all(str(p).startswith(OUTAGE_ID) for p in deps["predecessor_task_id"])

    def test_specific_lagged_link(self, deps):
        link = deps[
            (deps["predecessor_task_id"] == task_canonical_id("A1100")) &
            (deps["successor_task_id"] == task_canonical_id("A1200"))
        ]
        assert len(link) == 1
        assert link["lag_hours"].iloc[0] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# WBS parent_wbs_id — empty-string regression
# ---------------------------------------------------------------------------

class TestCsvTransformerWBS:
    @pytest.fixture(scope="class")
    def wbs(self, mock_csv_dir) -> pd.DataFrame:
        return load_mock_dataset(mock_csv_dir).wbs

    def test_root_nodes_have_na_parent(self, wbs):
        roots = wbs[wbs["wbs_id"].isin([wbs_canonical_id("WBS-100"), wbs_canonical_id("WBS-900")])]
        assert roots["parent_wbs_id"].isna().all()

    def test_child_node_parent_is_canonical_id(self, wbs):
        child = wbs[wbs["wbs_id"] == wbs_canonical_id("WBS-110")].iloc[0]
        assert child["parent_wbs_id"] == wbs_canonical_id("WBS-100")

    def test_empty_string_parent_becomes_na(self, tmp_path):
        """Regression: empty-string parent_wbs_id must become pd.NA, not a malformed canonical ID."""
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        (csv_dir / "wbs.csv").write_text(
            "wbs_id,parent_wbs_id,wbs_code,wbs_name,wbs_path,level\n"
            "W1,,Root,Root node,/Root,1\n"
            "W2,,Second root,Second root,,1\n",
            encoding="utf-8",
        )
        for fname in ["activities.csv", "relationships.csv", "resources.csv",
                      "assignments.csv", "calendars.csv", "activity_codes.csv",
                      "task_activity_codes.csv"]:
            (csv_dir / fname).write_text("", encoding="utf-8")

        transformer = P6CsvTransformer("O1", "Outage 1", "O1:V1", "Version 1")
        ds = transformer.transform_directory(csv_dir)
        assert ds.wbs["parent_wbs_id"].isna().all()


# ---------------------------------------------------------------------------
# Calendars — work_pattern_json parsed from JSON string
# ---------------------------------------------------------------------------

class TestCsvTransformerCalendars:
    @pytest.fixture(scope="class")
    def calendars(self, mock_csv_dir) -> pd.DataFrame:
        return load_mock_dataset(mock_csv_dir).calendars

    def test_work_pattern_json_is_dict(self, calendars):
        value = calendars["work_pattern_json"].iloc[0]
        assert isinstance(value, dict), f"Expected dict, got {type(value)}"

    def test_work_pattern_json_parsed_from_well_formed_json_string(self, tmp_path):
        """A properly-escaped JSON string in the CSV produces a parsed dict with the correct keys."""
        csv_dir = tmp_path / "csv_wfj"
        csv_dir.mkdir()
        # RFC 4180: internal double-quotes in a quoted field must be doubled.
        (csv_dir / "calendars.csv").write_text(
            'calendar_id,calendar_name,calendar_type,timezone,work_pattern_json\n'
            'CAL1,Test,resource,UTC,"{""shift_hours"": 12, ""work_days"": [""Mon""]}"\n',
            encoding="utf-8",
        )
        for fname in ["activities.csv", "relationships.csv", "resources.csv",
                      "assignments.csv", "activity_codes.csv", "task_activity_codes.csv",
                      "wbs.csv"]:
            (csv_dir / fname).write_text("", encoding="utf-8")
        transformer = P6CsvTransformer("O1", "Outage 1", "O1:V1", "Version 1")
        ds = transformer.transform_directory(csv_dir)
        value = ds.calendars["work_pattern_json"].iloc[0]
        assert isinstance(value, dict)
        assert "shift_hours" in value

    def test_work_pattern_json_malformed_string(self, tmp_path):
        """Malformed JSON string must produce {"raw": ...} rather than raising."""
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        (csv_dir / "calendars.csv").write_text(
            "calendar_id,calendar_name,calendar_type,timezone,work_pattern_json\n"
            'CAL1,Test,resource,UTC,"{not valid json}"\n',
            encoding="utf-8",
        )
        for fname in ["activities.csv", "relationships.csv", "resources.csv",
                      "assignments.csv", "activity_codes.csv", "task_activity_codes.csv",
                      "wbs.csv"]:
            (csv_dir / fname).write_text("", encoding="utf-8")

        transformer = P6CsvTransformer("O1", "Outage 1", "O1:V1", "Version 1")
        ds = transformer.transform_directory(csv_dir)
        value = ds.calendars["work_pattern_json"].iloc[0]
        assert isinstance(value, dict)
        assert "raw" in value

    def test_work_pattern_json_dict_passthrough(self, tmp_path):
        """A column that already contains a dict must pass through unchanged."""
        import pandas as _pd
        from outage_model.transforms.p6_csv import P6CsvTransformer as _T
        from outage_model.transforms.common import ensure_columns as _ec

        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        (csv_dir / "calendars.csv").write_text(
            'calendar_id,calendar_name,calendar_type,timezone,work_pattern_json\n'
            'CAL1,Test,resource,UTC,"{}"\n',
            encoding="utf-8",
        )
        for fname in ["activities.csv", "relationships.csv", "resources.csv",
                      "assignments.csv", "activity_codes.csv", "task_activity_codes.csv",
                      "wbs.csv"]:
            (csv_dir / fname).write_text("", encoding="utf-8")

        transformer = P6CsvTransformer("O1", "Outage 1", "O1:V1", "Version 1")
        ds = transformer.transform_directory(csv_dir)
        value = ds.calendars["work_pattern_json"].iloc[0]
        assert isinstance(value, dict)

    def test_missing_work_pattern_json_becomes_empty_dict(self, tmp_path):
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        (csv_dir / "calendars.csv").write_text(
            "calendar_id,calendar_name,calendar_type,timezone,work_pattern_json\n"
            "CAL1,Test,resource,UTC,\n",
            encoding="utf-8",
        )
        for fname in ["activities.csv", "relationships.csv", "resources.csv",
                      "assignments.csv", "activity_codes.csv", "task_activity_codes.csv",
                      "wbs.csv"]:
            (csv_dir / fname).write_text("", encoding="utf-8")

        transformer = P6CsvTransformer("O1", "Outage 1", "O1:V1", "Version 1")
        ds = transformer.transform_directory(csv_dir)
        value = ds.calendars["work_pattern_json"].iloc[0]
        assert value == {}


# ---------------------------------------------------------------------------
# Missing CSV files handled gracefully
# ---------------------------------------------------------------------------

class TestCsvTransformerMissingFiles:
    def test_missing_all_optional_files_returns_empty_tables(self, tmp_path):
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        # Only write activities so outage dates can be derived
        (csv_dir / "activities.csv").write_text(
            "activity_id,activity_name,planned_start,planned_finish\n"
            "A1,Task One,2026-04-10,2026-04-20\n",
            encoding="utf-8",
        )
        transformer = P6CsvTransformer("O1", "Outage 1", "O1:V1", "Version 1")
        ds = transformer.transform_directory(csv_dir)

        assert ds.wbs.empty
        assert ds.dependencies.empty
        assert ds.resources.empty
        assert ds.resource_assignments.empty
        assert ds.calendars.empty

    def test_missing_activities_returns_empty_schedule_tasks(self, tmp_path):
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        transformer = P6CsvTransformer("O1", "Outage 1", "O1:V1", "Version 1")
        ds = transformer.transform_directory(csv_dir)
        assert ds.schedule_tasks.empty
