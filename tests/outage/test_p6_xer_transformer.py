"""
Tests for outage_model.transforms.p6_xer.P6XERTransformer
and the load_xer_dataset convenience loader.
"""
from __future__ import annotations

import pandas as pd
import pytest

from outage_model.dataset import OutageDataset
from outage_model.loaders.xer_loader import load_xer_dataset
from outage_model.transforms.p6_xer import P6XERTransformer
from .conftest import make_xer, MINIMAL_XER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OUTAGE_ID = "RFO-29"
VERSION_ID = "RFO-29:xer"


def task_canonical_id(raw_id: str) -> str:
    return f"{OUTAGE_ID}:{VERSION_ID}:{raw_id}"


# ---------------------------------------------------------------------------
# Happy-path: dataset shape
# ---------------------------------------------------------------------------

class TestXERTransformerShape:
    @pytest.fixture(scope="class")
    def dataset(self, sample_xer_path) -> OutageDataset:
        return load_xer_dataset(sample_xer_path)

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

    def test_activity_codes_row_count(self, dataset):
        assert len(dataset.activity_codes) == 2

    def test_task_activity_codes_row_count(self, dataset):
        assert len(dataset.task_activity_codes) == 3

    def test_task_constraints_row_count(self, dataset):
        assert len(dataset.task_constraints) == 1


# ---------------------------------------------------------------------------
# Outage & schedule version identity
# ---------------------------------------------------------------------------

class TestXERTransformerIdentity:
    @pytest.fixture(scope="class")
    def dataset(self, sample_xer_path) -> OutageDataset:
        return load_xer_dataset(sample_xer_path)

    def test_outage_id_derived_from_proj_short_name(self, dataset):
        assert dataset.outages["outage_id"].iloc[0] == OUTAGE_ID

    def test_outage_name_derived_from_proj_name(self, dataset):
        assert "Sample Refueling Outage 29" in str(dataset.outages["outage_name"].iloc[0])

    def test_schedule_version_id_format(self, dataset):
        assert dataset.schedule_versions["schedule_version_id"].iloc[0] == VERSION_ID

    def test_schedule_version_source_system(self, dataset):
        assert dataset.schedule_versions["source_system"].iloc[0] == "primavera_p6_xer"

    def test_user_supplied_outage_id_overrides_xer(self, sample_xer_path):
        ds = load_xer_dataset(sample_xer_path, outage_id="MY-OUTAGE")
        assert dataset_outage_id(ds) == "MY-OUTAGE"

    def test_user_supplied_version_id_overrides_xer(self, sample_xer_path):
        ds = load_xer_dataset(sample_xer_path, schedule_version_id="BL1")
        assert ds.schedule_versions["schedule_version_id"].iloc[0] == "BL1"


def dataset_outage_id(ds: OutageDataset) -> str:
    return ds.outages["outage_id"].iloc[0]


# ---------------------------------------------------------------------------
# schedule_tasks field mapping
# ---------------------------------------------------------------------------

class TestXERTransformerTaskFields:
    @pytest.fixture(scope="class")
    def tasks(self, sample_xer_path) -> pd.DataFrame:
        return load_xer_dataset(sample_xer_path).schedule_tasks

    def test_task_ids_use_canonical_format(self, tasks):
        expected = {task_canonical_id("3001"), task_canonical_id("3002"),
                    task_canonical_id("3003"), task_canonical_id("3004")}
        assert set(tasks["task_id"]) == expected

    def test_milestone_flag_tt_startmile(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("3001")].iloc[0]
        assert row["milestone_flag"] == True

    def test_milestone_flag_tt_finmile(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("3004")].iloc[0]
        assert row["milestone_flag"] == True

    def test_milestone_flag_tt_task(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("3002")].iloc[0]
        assert row["milestone_flag"] == False

    def test_status_complete(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("3001")].iloc[0]
        assert row["status"] == "complete"

    def test_status_in_progress(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("3003")].iloc[0]
        assert row["status"] == "in_progress"

    def test_status_not_started(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("3004")].iloc[0]
        assert row["status"] == "not_started"

    def test_planned_duration_hours(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("3002")].iloc[0]
        assert row["planned_duration_hours"] == pytest.approx(36.0)

    def test_wbs_id_uses_canonical_format(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("3001")].iloc[0]
        assert row["wbs_id"].startswith(OUTAGE_ID)

    def test_source_record_id_is_original_p6_id(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("3002")].iloc[0]
        assert row["source_record_id"] == "3002"

    def test_percent_and_physical_pct_are_separate_fields(self, tasks):
        # After the fix: physical_percent_complete comes from phys_complete_pct;
        # percent_complete comes from complete_pct (absent in this XER → all NA).
        row = tasks[tasks["task_id"] == task_canonical_id("3001")].iloc[0]
        assert row["physical_percent_complete"] == pytest.approx(100.0)

    def test_physical_pct_in_progress_task(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("3003")].iloc[0]
        assert row["physical_percent_complete"] == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# critical_flag — NaN-safe behaviour (regression for the bug fix)
# ---------------------------------------------------------------------------

class TestXERTransformerCriticalFlag:
    @pytest.fixture(scope="class")
    def tasks(self, sample_xer_path) -> pd.DataFrame:
        return load_xer_dataset(sample_xer_path).schedule_tasks

    def test_zero_float_is_critical(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("3001")].iloc[0]
        assert row["critical_flag"] == True

    def test_negative_float_is_critical(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("3002")].iloc[0]
        assert row["critical_flag"] == True

    def test_positive_float_is_not_critical(self, tasks):
        row = tasks[tasks["task_id"] == task_canonical_id("3004")].iloc[0]
        assert row["critical_flag"] == False

    def test_missing_float_produces_na_not_false(self, tmp_path):
        """Regression: NaN float_hr_cnt must yield pd.NA, not silently False."""
        xer = make_xer(tmp_path, """\
            ERMHDR\t8.4\t2026-01-01\ttest\tuser
            %T\tPROJECT
            %F\tproj_id\tproj_short_name\tproj_name
            %R\t1\tP1\tProject 1
            %E
            %T\tTASK
            %F\ttask_id\tproj_id\ttask_code\ttask_name\ttask_type\tstatus_code
            %R\t101\t1\tT01\tNo float task\tTT_Task\tTK_NotStart
            %E
        """)
        ds = load_xer_dataset(xer)
        flag = ds.schedule_tasks["critical_flag"].iloc[0]
        assert pd.isna(flag), f"Expected pd.NA for missing float, got {flag!r}"


# ---------------------------------------------------------------------------
# dependencies field mapping
# ---------------------------------------------------------------------------

class TestXERTransformerDependencies:
    @pytest.fixture(scope="class")
    def deps(self, sample_xer_path) -> pd.DataFrame:
        return load_xer_dataset(sample_xer_path).dependencies

    def test_relationship_types_are_uppercase(self, deps):
        assert set(deps["relationship_type"]) == {"FS"}

    def test_lag_hours_values(self, deps):
        lags = sorted(deps["lag_hours"].dropna().tolist())
        assert lags == pytest.approx([0.0, 0.0, 8.0])

    def test_predecessor_ids_use_canonical_format(self, deps):
        assert all(str(pid).startswith(OUTAGE_ID) for pid in deps["predecessor_task_id"])

    def test_successor_ids_use_canonical_format(self, deps):
        assert all(str(sid).startswith(OUTAGE_ID) for sid in deps["successor_task_id"])

    def test_specific_fs_link(self, deps):
        link = deps[
            (deps["predecessor_task_id"] == task_canonical_id("3001")) &
            (deps["successor_task_id"] == task_canonical_id("3002"))
        ]
        assert len(link) == 1
        assert link["lag_hours"].iloc[0] == pytest.approx(0.0)

    def test_lagged_link(self, deps):
        link = deps[
            (deps["predecessor_task_id"] == task_canonical_id("3002")) &
            (deps["successor_task_id"] == task_canonical_id("3003"))
        ]
        assert len(link) == 1
        assert link["lag_hours"].iloc[0] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# WBS field mapping
# ---------------------------------------------------------------------------

class TestXERTransformerWBS:
    @pytest.fixture(scope="class")
    def wbs(self, sample_xer_path) -> pd.DataFrame:
        return load_xer_dataset(sample_xer_path).wbs

    def test_root_node_has_na_parent(self, wbs):
        root = wbs[wbs["wbs_id"] == f"{OUTAGE_ID}:{VERSION_ID}:2001"].iloc[0]
        assert pd.isna(root["parent_wbs_id"])

    def test_child_node_parent_is_canonical_id(self, wbs):
        child = wbs[wbs["wbs_id"] == f"{OUTAGE_ID}:{VERSION_ID}:2002"].iloc[0]
        assert child["parent_wbs_id"] == f"{OUTAGE_ID}:{VERSION_ID}:2001"

    def test_wbs_code_populated(self, wbs):
        assert wbs["wbs_code"].notna().all()


# ---------------------------------------------------------------------------
# Activity codes
# ---------------------------------------------------------------------------

class TestXERTransformerActivityCodes:
    @pytest.fixture(scope="class")
    def codes(self, sample_xer_path) -> pd.DataFrame:
        return load_xer_dataset(sample_xer_path).activity_codes

    def test_code_types_resolved_from_actvtype(self, codes):
        assert set(codes["code_type"]) == {"Discipline", "System"}

    def test_code_values(self, codes):
        assert set(codes["code_value"]) == {"MECH", "RCS"}


# ---------------------------------------------------------------------------
# Task constraints
# ---------------------------------------------------------------------------

class TestXERTransformerTaskConstraints:
    @pytest.fixture(scope="class")
    def constraints(self, sample_xer_path) -> pd.DataFrame:
        return load_xer_dataset(sample_xer_path).task_constraints

    def test_one_constrained_task(self, constraints):
        assert len(constraints) == 1

    def test_cso_is_hard_constraint(self, constraints):
        assert constraints["hard_flag"].iloc[0] == True

    def test_constraint_task_id_canonical(self, constraints):
        assert constraints["task_id"].iloc[0] == task_canonical_id("3003")

    def test_constraint_type_value(self, constraints):
        assert constraints["constraint_type"].iloc[0] == "CSO"


# ---------------------------------------------------------------------------
# Calendars
# ---------------------------------------------------------------------------

class TestXERTransformerCalendars:
    @pytest.fixture(scope="class")
    def calendars(self, sample_xer_path) -> pd.DataFrame:
        return load_xer_dataset(sample_xer_path).calendars

    def test_calendar_name(self, calendars):
        assert calendars["calendar_name"].iloc[0] == "24x7 Outage Calendar"

    def test_work_pattern_json_is_dict(self, calendars):
        value = calendars["work_pattern_json"].iloc[0]
        assert isinstance(value, dict)

    def test_work_pattern_json_has_raw_key(self, calendars):
        value = calendars["work_pattern_json"].iloc[0]
        assert "raw_calendar_data" in value


# ---------------------------------------------------------------------------
# Transformer reuse across multiple files
# ---------------------------------------------------------------------------

class TestXERTransformerReuse:
    def test_second_call_uses_second_xer_identity(self, tmp_path, sample_xer_path):
        """
        Regression: calling transform_file twice must not bleed the first
        file's resolved outage_id into the second call.
        """
        second_xer = make_xer(tmp_path, """\
            ERMHDR\t8.4\t2026-01-01\ttest\tuser
            %T\tPROJECT
            %F\tproj_id\tproj_short_name\tproj_name
            %R\t999\tRFO-99\tOutage Ninety Nine
            %E
        """)
        transformer = P6XERTransformer()  # no user-supplied IDs
        ds1 = transformer.transform_file(sample_xer_path)
        ds2 = transformer.transform_file(second_xer)

        assert ds1.outages["outage_id"].iloc[0] == "RFO-29"
        assert ds2.outages["outage_id"].iloc[0] == "RFO-99"

    def test_user_supplied_id_is_stable_across_calls(self, tmp_path, sample_xer_path):
        second_xer = make_xer(tmp_path, """\
            ERMHDR\t8.4\t2026-01-01\ttest\tuser
            %T\tPROJECT
            %F\tproj_id\tproj_short_name\tproj_name
            %R\t999\tRFO-99\tOutage Ninety Nine
            %E
        """)
        transformer = P6XERTransformer(outage_id="FIXED-ID")
        ds1 = transformer.transform_file(sample_xer_path)
        ds2 = transformer.transform_file(second_xer)

        assert ds1.outages["outage_id"].iloc[0] == "FIXED-ID"
        assert ds2.outages["outage_id"].iloc[0] == "FIXED-ID"


# ---------------------------------------------------------------------------
# Multi-project XER: project_id filtering
# ---------------------------------------------------------------------------

class TestXERTransformerMultiProject:
    @pytest.fixture
    def multi_project_xer(self, tmp_path) -> tuple:
        xer = make_xer(tmp_path, """\
            ERMHDR\t8.4\t2026-01-01\ttest\tuser
            %T\tPROJECT
            %F\tproj_id\tproj_short_name\tproj_name
            %R\t1\tP-ONE\tProject One
            %R\t2\tP-TWO\tProject Two
            %E
            %T\tTASK
            %F\ttask_id\tproj_id\ttask_code\ttask_name\ttask_type\tstatus_code\ttarget_start_date\ttarget_end_date\ttarget_drtn_hr_cnt\tremain_drtn_hr_cnt\ttotal_float_hr_cnt\tfree_float_hr_cnt\tphys_complete_pct\tclndr_id\tcstr_type\tcstr_date
            %R\t101\t1\tT01\tProject One Task\tTT_Task\tTK_NotStart\t2026-01-10 06:00\t2026-01-11 18:00\t36\t36\t0\t0\t0\t501\t\t
            %R\t201\t2\tT02\tProject Two Task\tTT_Task\tTK_NotStart\t2026-02-10 06:00\t2026-02-11 18:00\t36\t36\t0\t0\t0\t501\t\t
            %E
        """)
        return xer

    def test_project_id_filter_selects_correct_project(self, multi_project_xer):
        ds = load_xer_dataset(multi_project_xer, project_id="2")
        assert ds.outages["outage_id"].iloc[0] == "P-TWO"

    def test_project_id_filter_scopes_tasks(self, multi_project_xer):
        ds = load_xer_dataset(multi_project_xer, project_id="2")
        assert len(ds.schedule_tasks) == 1
        assert "T02" in str(ds.schedule_tasks["task_code"].iloc[0])


# ---------------------------------------------------------------------------
# Datetime fallback formats
# ---------------------------------------------------------------------------

class TestXERTransformerDatetimeFallback:
    def test_parses_xer_date_without_time(self, tmp_path):
        """%d-%b-%y format (no time component)."""
        xer = make_xer(tmp_path, """\
            ERMHDR\t8.4\t2026-01-01\ttest\tuser
            %T\tPROJECT
            %F\tproj_id\tproj_short_name\tproj_name
            %R\t1\tP1\tProject 1
            %E
            %T\tTASK
            %F\ttask_id\tproj_id\ttask_code\ttask_name\ttask_type\tstatus_code\ttarget_start_date\ttarget_end_date\ttarget_drtn_hr_cnt\tremain_drtn_hr_cnt\ttotal_float_hr_cnt\tfree_float_hr_cnt\tphys_complete_pct\tclndr_id\tcstr_type\tcstr_date
            %R\t101\t1\tT01\tDate test task\tTT_Task\tTK_NotStart\t10-Jan-26\t15-Jan-26\t120\t120\t0\t0\t0\t501\t\t
            %E
        """)
        ds = load_xer_dataset(xer)
        start = ds.schedule_tasks["planned_start"].iloc[0]
        assert pd.notna(start), "Planned start should be parsed from %d-%b-%y format"
        assert start.month == 1
        assert start.day == 10

    def test_parses_xer_date_with_time(self, tmp_path):
        """%d-%b-%y %H:%M format."""
        xer = make_xer(tmp_path, """\
            ERMHDR\t8.4\t2026-01-01\ttest\tuser
            %T\tPROJECT
            %F\tproj_id\tproj_short_name\tproj_name
            %R\t1\tP1\tProject 1
            %E
            %T\tTASK
            %F\ttask_id\tproj_id\ttask_code\ttask_name\ttask_type\tstatus_code\ttarget_start_date\ttarget_end_date\ttarget_drtn_hr_cnt\tremain_drtn_hr_cnt\ttotal_float_hr_cnt\tfree_float_hr_cnt\tphys_complete_pct\tclndr_id\tcstr_type\tcstr_date
            %R\t101\t1\tT01\tDate test task\tTT_Task\tTK_NotStart\t10-Jan-26 06:00\t15-Jan-26 18:00\t120\t120\t0\t0\t0\t501\t\t
            %E
        """)
        ds = load_xer_dataset(xer)
        start = ds.schedule_tasks["planned_start"].iloc[0]
        assert pd.notna(start)
        assert start.hour == 6

    def test_missing_dates_become_nat(self, tmp_path):
        xer = make_xer(tmp_path, """\
            ERMHDR\t8.4\t2026-01-01\ttest\tuser
            %T\tPROJECT
            %F\tproj_id\tproj_short_name\tproj_name
            %R\t1\tP1\tProject 1
            %E
            %T\tTASK
            %F\ttask_id\tproj_id\ttask_code\ttask_name\ttask_type\tstatus_code\ttarget_start_date\ttarget_end_date\ttarget_drtn_hr_cnt\tremain_drtn_hr_cnt\ttotal_float_hr_cnt\tfree_float_hr_cnt\tphys_complete_pct\tclndr_id\tcstr_type\tcstr_date
            %R\t101\t1\tT01\tNo dates\tTT_Task\tTK_NotStart\t\t\t0\t0\t0\t0\t0\t501\t\t
            %E
        """)
        ds = load_xer_dataset(xer)
        assert pd.isna(ds.schedule_tasks["planned_start"].iloc[0])
        assert pd.isna(ds.schedule_tasks["planned_finish"].iloc[0])


# ---------------------------------------------------------------------------
# Missing optional tables handled gracefully
# ---------------------------------------------------------------------------

class TestXERTransformerMissingTables:
    def test_xer_without_resources_produces_empty_resources(self, tmp_path):
        xer = make_xer(tmp_path, MINIMAL_XER)
        ds = load_xer_dataset(xer)
        assert ds.resources.empty

    def test_xer_without_calendar_produces_empty_calendars(self, tmp_path):
        xer = make_xer(tmp_path, MINIMAL_XER)
        ds = load_xer_dataset(xer)
        assert ds.calendars.empty

    def test_xer_without_activity_codes_produces_empty_table(self, tmp_path):
        xer = make_xer(tmp_path, MINIMAL_XER)
        ds = load_xer_dataset(xer)
        assert ds.activity_codes.empty

    def test_xer_without_constraints_produces_empty_table(self, tmp_path):
        xer = make_xer(tmp_path, MINIMAL_XER)
        ds = load_xer_dataset(xer)
        assert ds.task_constraints.empty

    def test_minimal_xer_tasks_and_deps_populated(self, tmp_path):
        xer = make_xer(tmp_path, MINIMAL_XER)
        ds = load_xer_dataset(xer)
        assert len(ds.schedule_tasks) == 2
        assert len(ds.dependencies) == 1
