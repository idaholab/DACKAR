"""
Tests for outage_model.transforms.p6_xer.XERParser.
"""
from __future__ import annotations

import textwrap

import pandas as pd
import pytest

from outage_model.transforms.p6_xer import XERParser
from .conftest import make_xer


class TestXERParserTableExtraction:
    def test_parses_known_tables_from_sample(self, sample_xer_path):
        tables = XERParser(sample_xer_path).parse()
        expected = {"project", "projwbs", "task", "taskpred", "rsrc", "taskrsrc",
                    "calendar", "actvtype", "actvcode", "taskactv"}
        assert expected.issubset(tables.keys())

    def test_all_values_are_dataframes(self, sample_xer_path):
        tables = XERParser(sample_xer_path).parse()
        for name, df in tables.items():
            assert isinstance(df, pd.DataFrame), f"{name} is not a DataFrame"

    def test_project_table_row_count(self, sample_xer_path):
        tables = XERParser(sample_xer_path).parse()
        assert len(tables["project"]) == 1

    def test_task_table_row_count(self, sample_xer_path):
        tables = XERParser(sample_xer_path).parse()
        assert len(tables["task"]) == 4

    def test_taskpred_table_row_count(self, sample_xer_path):
        tables = XERParser(sample_xer_path).parse()
        assert len(tables["taskpred"]) == 3

    def test_column_names_are_normalised(self, sample_xer_path):
        tables = XERParser(sample_xer_path).parse()
        for col in tables["task"].columns:
            assert col == col.lower(), f"Column '{col}' is not lowercase"
            assert " " not in col, f"Column '{col}' contains spaces"


class TestXERParserFieldMapping:
    def test_project_proj_short_name(self, sample_xer_path):
        tables = XERParser(sample_xer_path).parse()
        assert tables["project"]["proj_short_name"].iloc[0] == "RFO-29"

    def test_project_proj_name(self, sample_xer_path):
        tables = XERParser(sample_xer_path).parse()
        assert tables["project"]["proj_name"].iloc[0] == "Sample Refueling Outage 29"

    def test_task_type_values_present(self, sample_xer_path):
        tables = XERParser(sample_xer_path).parse()
        task_types = set(tables["task"]["task_type"].str.lower())
        assert "tt_startmile" in task_types
        assert "tt_task" in task_types
        assert "tt_finmile" in task_types

    def test_taskpred_pred_types(self, sample_xer_path):
        tables = XERParser(sample_xer_path).parse()
        assert set(tables["taskpred"]["pred_type"]) == {"FS"}


class TestXERParserEdgeCases:
    def test_missing_end_marker_still_captures_last_table(self, tmp_path):
        xer = make_xer(tmp_path, """\
            ERMHDR\t8.4\t2026-01-01\ttest\tuser
            %T\tPROJECT
            %F\tproj_id\tproj_name
            %R\t1\tTest Project
        """)
        tables = XERParser(xer).parse()
        assert "project" in tables
        assert len(tables["project"]) == 1

    def test_empty_table_produces_empty_dataframe(self, tmp_path):
        xer = make_xer(tmp_path, """\
            ERMHDR\t8.4\t2026-01-01\ttest\tuser
            %T\tPROJECT
            %F\tproj_id\tproj_name
            %E
        """)
        tables = XERParser(xer).parse()
        assert "project" in tables
        assert tables["project"].empty

    def test_table_with_no_field_line_returns_empty_dataframe(self, tmp_path):
        xer = make_xer(tmp_path, """\
            ERMHDR\t8.4\t2026-01-01\ttest\tuser
            %T\tPROJECT
            %R\t1\tTest
            %E
        """)
        tables = XERParser(xer).parse()
        assert tables["project"].empty

    def test_row_with_fewer_fields_is_padded_with_none(self, tmp_path):
        xer = make_xer(tmp_path, """\
            ERMHDR\t8.4\t2026-01-01\ttest\tuser
            %T\tPROJECT
            %F\tproj_id\tproj_name\tproj_status
            %R\t1\tShort row
            %E
        """)
        tables = XERParser(xer).parse()
        assert len(tables["project"].columns) == 3
        assert tables["project"]["proj_status"].iloc[0] is None

    def test_multiple_tables_in_sequence(self, tmp_path):
        xer = make_xer(tmp_path, """\
            ERMHDR\t8.4\t2026-01-01\ttest\tuser
            %T\tPROJECT
            %F\tproj_id\tproj_name
            %R\t1\tProj A
            %E
            %T\tTASK
            %F\tproj_id\ttask_id\ttask_name
            %R\t1\t101\tTask One
            %R\t1\t102\tTask Two
            %E
        """)
        tables = XERParser(xer).parse()
        assert len(tables["project"]) == 1
        assert len(tables["task"]) == 2

    def test_table_names_normalised_to_lowercase(self, tmp_path):
        xer = make_xer(tmp_path, """\
            ERMHDR\t8.4\t2026-01-01\ttest\tuser
            %T\tPROJWBS
            %F\twbs_id\twbs_name
            %R\t1\tRoot
            %E
        """)
        tables = XERParser(xer).parse()
        assert "projwbs" in tables
