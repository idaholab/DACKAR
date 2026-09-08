"""
Tests for P6DatasetAdapter.

Covers:
- Direct field mapping (identity, timestamps, duration, metadata stash)
- Computed actual_duration_hours from timestamp delta
- Dependency index: predecessor_ids and successor_ids
- Resource index: is_vendor_supported, crew_size, contractor_flag
- Scope change index: is_emergent, is_rework
- Phase window join: outage_phase
- Milestone skipping (skip_milestones=True / False)
- LabelMapper delegation: discipline/task_family/component_family are None from adapter
- Empty DataFrames: adapter yields rows, missing indices degrade gracefully
- get_activity_row: happy path and KeyError on unknown ID
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from outage_model.dataset import OutageDataset
from outage_model.models import Outage
from outage_uncertainty.adapters.p6_dataset_adapter import P6DatasetAdapter


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def outage():
    return Outage(
        outage_id="RF-22",
        plant_id="PLANT-A",
        unit_id="UNIT-1",
        outage_name="RF-22 Refueling",
    )


def _make_task(
    task_id="T-001",
    task_name="Replace SG feedwater nozzle",
    milestone_flag=False,
    planned_start=None,
    planned_finish=None,
    actual_start=None,
    actual_finish=None,
    planned_duration_hours=12.0,
    scope_origin=None,
    critical_flag=True,
    total_float_hours=0.0,
    task_code=None,
    task_type="task_dependent",
    wbs_id="WBS-1",
    schedule_version_id="V-1",
    outage_id="RF-22",
):
    return {
        "task_id": task_id,
        "outage_id": outage_id,
        "schedule_version_id": schedule_version_id,
        "wbs_id": wbs_id,
        "task_code": task_code or task_id,
        "task_name": task_name,
        "task_type": task_type,
        "milestone_flag": milestone_flag,
        "planned_start": planned_start or pd.Timestamp("2024-03-01 08:00"),
        "planned_finish": planned_finish or pd.Timestamp("2024-03-01 20:00"),
        "actual_start": actual_start,
        "actual_finish": actual_finish,
        "planned_duration_hours": planned_duration_hours,
        "scope_origin": scope_origin,
        "critical_flag": critical_flag,
        "total_float_hours": total_float_hours,
    }


@pytest.fixture
def single_task_dataset():
    """Dataset with one non-milestone task, no joins populated."""
    return OutageDataset(
        schedule_tasks=pd.DataFrame([_make_task()])
    )


# ---------------------------------------------------------------------------
# 1. Direct field mapping
# ---------------------------------------------------------------------------

class TestDirectFieldMapping:
    def test_activity_id(self, single_task_dataset, outage):
        row = _first_row(single_task_dataset, outage)
        assert row["activity_id"] == "T-001"

    def test_outage_id(self, single_task_dataset, outage):
        row = _first_row(single_task_dataset, outage)
        assert row["outage_id"] == "RF-22"

    def test_plant_id_from_outage(self, single_task_dataset, outage):
        row = _first_row(single_task_dataset, outage)
        assert row["plant_id"] == "PLANT-A"

    def test_unit_id_from_outage(self, single_task_dataset, outage):
        row = _first_row(single_task_dataset, outage)
        assert row["unit_id"] == "UNIT-1"

    def test_raw_description_from_task_name(self, single_task_dataset, outage):
        row = _first_row(single_task_dataset, outage)
        assert row["raw_description"] == "Replace SG feedwater nozzle"

    def test_planned_duration_hours(self, single_task_dataset, outage):
        row = _first_row(single_task_dataset, outage)
        assert row["planned_duration_hours"] == 12.0

    def test_timestamps_converted_to_datetime(self, single_task_dataset, outage):
        row = _first_row(single_task_dataset, outage)
        assert isinstance(row["planned_start"], datetime)
        assert isinstance(row["planned_finish"], datetime)

    def test_p6_metadata_stash(self, single_task_dataset, outage):
        row = _first_row(single_task_dataset, outage)
        p6 = row["metadata"]["p6"]
        assert p6["task_code"] == "T-001"
        assert p6["task_type"] == "task_dependent"
        assert p6["critical_flag"] is True
        assert p6["total_float_hours"] == 0.0
        assert p6["wbs_id"] == "WBS-1"
        assert p6["schedule_version_id"] == "V-1"


# ---------------------------------------------------------------------------
# 2. Taxonomy fields are None (delegated to LabelMapper)
# ---------------------------------------------------------------------------

class TestTaxonomyDelegation:
    def test_discipline_is_none(self, single_task_dataset, outage):
        row = _first_row(single_task_dataset, outage)
        assert row["discipline"] is None

    def test_task_family_is_none(self, single_task_dataset, outage):
        row = _first_row(single_task_dataset, outage)
        assert row["task_family"] is None

    def test_component_family_is_none(self, single_task_dataset, outage):
        row = _first_row(single_task_dataset, outage)
        assert row["component_family"] is None

    def test_execution_mode_flags_default_false(self, single_task_dataset, outage):
        row = _first_row(single_task_dataset, outage)
        assert row["has_rp_hold"] is False
        assert row["requires_scaffold"] is False
        assert row["has_clearance"] is False


# ---------------------------------------------------------------------------
# 3. Computed actual_duration_hours
# ---------------------------------------------------------------------------

class TestActualDurationComputed:
    def test_computed_from_timestamps(self, outage):
        ds = OutageDataset(schedule_tasks=pd.DataFrame([_make_task(
            actual_start=pd.Timestamp("2024-03-01 08:00"),
            actual_finish=pd.Timestamp("2024-03-01 21:30"),
        )]))
        row = _first_row(ds, outage)
        assert abs(row["actual_duration_hours"] - 13.5) < 1e-6

    def test_none_when_actual_start_missing(self, outage):
        ds = OutageDataset(schedule_tasks=pd.DataFrame([_make_task(
            actual_start=None,
            actual_finish=pd.Timestamp("2024-03-01 21:30"),
        )]))
        row = _first_row(ds, outage)
        assert row["actual_duration_hours"] is None

    def test_none_when_both_missing(self, single_task_dataset, outage):
        row = _first_row(single_task_dataset, outage)
        assert row["actual_duration_hours"] is None

    def test_negative_timestamps_returns_none(self, outage):
        # actual_finish before actual_start → nonsense duration → None
        ds = OutageDataset(schedule_tasks=pd.DataFrame([_make_task(
            actual_start=pd.Timestamp("2024-03-01 21:00"),
            actual_finish=pd.Timestamp("2024-03-01 08:00"),
        )]))
        row = _first_row(ds, outage)
        assert row["actual_duration_hours"] is None


# ---------------------------------------------------------------------------
# 4. Dependency index
# ---------------------------------------------------------------------------

class TestDependencyIndex:
    @pytest.fixture
    def dataset_with_deps(self):
        tasks = pd.DataFrame([
            _make_task("T-001"),
            _make_task("T-002"),
            _make_task("T-003"),
        ])
        deps = pd.DataFrame([
            {"dependency_id": "D-1", "schedule_version_id": "V-1",
             "predecessor_task_id": "T-001", "successor_task_id": "T-002",
             "relationship_type": "FS", "lag_hours": 0.0},
            {"dependency_id": "D-2", "schedule_version_id": "V-1",
             "predecessor_task_id": "T-001", "successor_task_id": "T-003",
             "relationship_type": "FS", "lag_hours": 0.0},
        ])
        return OutageDataset(schedule_tasks=tasks, dependencies=deps)

    def test_successor_ids(self, dataset_with_deps, outage):
        rows = {r["activity_id"]: r for r in _all_rows(dataset_with_deps, outage)}
        assert set(rows["T-001"]["successor_ids"]) == {"T-002", "T-003"}

    def test_predecessor_ids(self, dataset_with_deps, outage):
        rows = {r["activity_id"]: r for r in _all_rows(dataset_with_deps, outage)}
        assert rows["T-002"]["predecessor_ids"] == ["T-001"]
        assert rows["T-003"]["predecessor_ids"] == ["T-001"]

    def test_no_deps_gives_empty_lists(self, single_task_dataset, outage):
        row = _first_row(single_task_dataset, outage)
        assert row["predecessor_ids"] == []
        assert row["successor_ids"] == []


# ---------------------------------------------------------------------------
# 5. Resource index
# ---------------------------------------------------------------------------

class TestResourceIndex:
    @pytest.fixture
    def dataset_with_resources(self):
        tasks = pd.DataFrame([
            _make_task("T-001"),
            _make_task("T-002"),
            _make_task("T-003"),
        ])
        resources = pd.DataFrame([
            {"resource_id": "R-internal", "resource_name": "Mechanic A",
             "vendor": None, "org_unit": "internal"},
            {"resource_id": "R-vendor",   "resource_name": "OEM Tech",
             "vendor": "Westinghouse", "org_unit": "vendor"},
            {"resource_id": "R-contract", "resource_name": "Contractor B",
             "vendor": None, "org_unit": "ABC_contractors"},
        ])
        assignments = pd.DataFrame([
            # T-001: one internal resource
            {"assignment_id": "A-1", "schedule_version_id": "V-1",
             "task_id": "T-001", "resource_id": "R-internal"},
            # T-002: two resources (internal + vendor)
            {"assignment_id": "A-2", "schedule_version_id": "V-1",
             "task_id": "T-002", "resource_id": "R-internal"},
            {"assignment_id": "A-3", "schedule_version_id": "V-1",
             "task_id": "T-002", "resource_id": "R-vendor"},
            # T-003: contractor resource
            {"assignment_id": "A-4", "schedule_version_id": "V-1",
             "task_id": "T-003", "resource_id": "R-contract"},
        ])
        return OutageDataset(
            schedule_tasks=tasks,
            resources=resources,
            resource_assignments=assignments,
        )

    def test_vendor_task_flagged(self, dataset_with_resources, outage):
        rows = {r["activity_id"]: r for r in _all_rows(dataset_with_resources, outage)}
        assert rows["T-002"]["is_vendor_supported"] is True

    def test_internal_only_task_not_vendor(self, dataset_with_resources, outage):
        rows = {r["activity_id"]: r for r in _all_rows(dataset_with_resources, outage)}
        assert rows["T-001"]["is_vendor_supported"] is False

    def test_crew_size_counts_distinct_resources(self, dataset_with_resources, outage):
        rows = {r["activity_id"]: r for r in _all_rows(dataset_with_resources, outage)}
        assert rows["T-001"]["crew_size"] == 1
        assert rows["T-002"]["crew_size"] == 2

    def test_contractor_flag_with_configured_orgs(self, dataset_with_resources, outage):
        adapter = P6DatasetAdapter(contractor_org_units={"ABC_contractors"})
        rows = {r["activity_id"]: r for r in adapter.iter_activity_rows(dataset_with_resources, outage)}
        assert rows["T-003"]["contractor_flag"] is True
        assert rows["T-001"]["contractor_flag"] is False

    def test_contractor_flag_none_when_orgs_not_configured(self, dataset_with_resources, outage):
        rows = {r["activity_id"]: r for r in _all_rows(dataset_with_resources, outage)}
        # No contractor_org_units configured → unknown for all tasks
        assert rows["T-001"]["contractor_flag"] is None
        assert rows["T-003"]["contractor_flag"] is None

    def test_no_resources_crew_size_none(self, single_task_dataset, outage):
        row = _first_row(single_task_dataset, outage)
        assert row["crew_size"] is None


# ---------------------------------------------------------------------------
# 6. Scope change index
# ---------------------------------------------------------------------------

class TestScopeChangeIndex:
    @pytest.fixture
    def dataset_with_scope_changes(self):
        tasks = pd.DataFrame([
            _make_task("T-001"),
            _make_task("T-002"),
            _make_task("T-003"),
        ])
        scope = pd.DataFrame([
            {"scope_change_id": "SC-1", "outage_id": "RF-22",
             "task_id": "T-001", "change_type": "emergent"},
            {"scope_change_id": "SC-2", "outage_id": "RF-22",
             "task_id": "T-002", "change_type": "rework"},
        ])
        return OutageDataset(schedule_tasks=tasks, scope_change_events=scope)

    def test_emergent_task_flagged(self, dataset_with_scope_changes, outage):
        rows = {r["activity_id"]: r for r in _all_rows(dataset_with_scope_changes, outage)}
        assert rows["T-001"]["is_emergent"] is True
        assert rows["T-002"]["is_emergent"] is False

    def test_rework_task_flagged(self, dataset_with_scope_changes, outage):
        rows = {r["activity_id"]: r for r in _all_rows(dataset_with_scope_changes, outage)}
        assert rows["T-002"]["is_rework"] is True
        assert rows["T-001"]["is_rework"] is False

    def test_no_scope_changes_all_false(self, single_task_dataset, outage):
        row = _first_row(single_task_dataset, outage)
        assert row["is_emergent"] is False
        assert row["is_rework"] is False

    def test_custom_emergent_type(self, outage):
        tasks = pd.DataFrame([_make_task("T-001")])
        scope = pd.DataFrame([
            {"scope_change_id": "SC-1", "outage_id": "RF-22",
             "task_id": "T-001", "change_type": "forced_addition"},
        ])
        ds = OutageDataset(schedule_tasks=tasks, scope_change_events=scope)
        adapter = P6DatasetAdapter(emergent_change_types={"forced_addition"})
        row = next(adapter.iter_activity_rows(ds, outage))
        assert row["is_emergent"] is True


# ---------------------------------------------------------------------------
# 7. Phase window join
# ---------------------------------------------------------------------------

class TestPhaseWindowJoin:
    @pytest.fixture
    def dataset_with_phases(self):
        tasks = pd.DataFrame([
            _make_task("T-outage",
                       planned_start=pd.Timestamp("2024-03-05 10:00"),
                       planned_finish=pd.Timestamp("2024-03-05 22:00")),
            _make_task("T-prep",
                       planned_start=pd.Timestamp("2024-03-01 08:00"),
                       planned_finish=pd.Timestamp("2024-03-01 20:00")),
            _make_task("T-none",
                       planned_start=pd.Timestamp("2024-04-01 08:00"),
                       planned_finish=pd.Timestamp("2024-04-01 20:00")),
        ])
        phases = pd.DataFrame([
            {"phase_id": "P-1", "outage_id": "RF-22", "phase_name": "pre-outage",
             "sequence": 1,
             "start_planned": pd.Timestamp("2024-03-01"),
             "finish_planned": pd.Timestamp("2024-03-03 23:59")},
            {"phase_id": "P-2", "outage_id": "RF-22", "phase_name": "forced-outage",
             "sequence": 2,
             "start_planned": pd.Timestamp("2024-03-04"),
             "finish_planned": pd.Timestamp("2024-03-10 23:59")},
        ])
        return OutageDataset(schedule_tasks=tasks, outage_phases=phases)

    def test_phase_matched(self, dataset_with_phases, outage):
        rows = {r["activity_id"]: r for r in _all_rows(dataset_with_phases, outage)}
        assert rows["T-outage"]["outage_phase"] == "forced-outage"
        assert rows["T-prep"]["outage_phase"] == "pre-outage"

    def test_no_matching_phase_gives_none(self, dataset_with_phases, outage):
        rows = {r["activity_id"]: r for r in _all_rows(dataset_with_phases, outage)}
        assert rows["T-none"]["outage_phase"] is None

    def test_no_phases_all_none(self, single_task_dataset, outage):
        row = _first_row(single_task_dataset, outage)
        assert row["outage_phase"] is None


# ---------------------------------------------------------------------------
# 8. Milestone skipping
# ---------------------------------------------------------------------------

class TestMilestoneSkipping:
    @pytest.fixture
    def dataset_with_milestone(self):
        return OutageDataset(schedule_tasks=pd.DataFrame([
            _make_task("T-001"),
            _make_task("M-001", task_name="Outage Start", milestone_flag=True,
                       planned_duration_hours=0.0),
        ]))

    def test_milestone_skipped_by_default(self, dataset_with_milestone, outage):
        ids = [r["activity_id"] for r in _all_rows(dataset_with_milestone, outage)]
        assert "M-001" not in ids
        assert "T-001" in ids

    def test_milestone_included_when_flag_off(self, dataset_with_milestone, outage):
        adapter = P6DatasetAdapter()
        rows = list(adapter.iter_activity_rows(dataset_with_milestone, outage, skip_milestones=False))
        ids = [r["activity_id"] for r in rows]
        assert "M-001" in ids
        assert "T-001" in ids


# ---------------------------------------------------------------------------
# 9. Empty DataFrames — graceful degradation
# ---------------------------------------------------------------------------

class TestEmptyDatasets:
    def test_empty_schedule_tasks_yields_nothing(self, outage):
        ds = OutageDataset()
        adapter = P6DatasetAdapter()
        rows = list(adapter.iter_activity_rows(ds, outage))
        assert rows == []

    def test_missing_join_tables_degrade_gracefully(self, outage):
        # Only schedule_tasks populated; all join tables empty
        ds = OutageDataset(schedule_tasks=pd.DataFrame([_make_task()]))
        row = _first_row(ds, outage)
        assert row["predecessor_ids"] == []
        assert row["successor_ids"] == []
        assert row["is_vendor_supported"] is False
        assert row["crew_size"] is None
        assert row["is_emergent"] is False
        assert row["outage_phase"] is None


# ---------------------------------------------------------------------------
# 10. get_activity_row
# ---------------------------------------------------------------------------

class TestGetActivityRow:
    def test_returns_correct_row(self, outage):
        ds = OutageDataset(schedule_tasks=pd.DataFrame([
            _make_task("T-001"),
            _make_task("T-002"),
        ]))
        adapter = P6DatasetAdapter()
        row = adapter.get_activity_row("T-002", ds, outage)
        assert row["activity_id"] == "T-002"

    def test_raises_key_error_on_unknown_id(self, single_task_dataset, outage):
        adapter = P6DatasetAdapter()
        with pytest.raises(KeyError, match="T-UNKNOWN"):
            adapter.get_activity_row("T-UNKNOWN", single_task_dataset, outage)

    def test_raises_key_error_on_empty_dataset(self, outage):
        adapter = P6DatasetAdapter()
        with pytest.raises(KeyError):
            adapter.get_activity_row("T-001", OutageDataset(), outage)


# ---------------------------------------------------------------------------
# 11. ActivityService.ingest_from_p6 integration smoke-test
# ---------------------------------------------------------------------------

class TestIngestFromP6Integration:
    def test_ingest_from_p6_returns_activity_cases(self, outage):
        """End-to-end: OutageDataset → ActivityCase list via ActivityService."""
        from outage_uncertainty.services.activity_service import ActivityService
        from outage_uncertainty.workflows.activity_ingestion_workflow import ActivityIngestionWorkflow
        from outage_uncertainty.adapters.pandas_repository import PandasActivityRepository
        from outage_uncertainty.preprocessing.label_mapper import TaskLabelMapper
        from outage_uncertainty.preprocessing.feature_builder import ActivityFeatureBuilder

        # Minimal text cleaner stub
        class _IdentityCleaner:
            def clean(self, activity):
                activity.cleaned_description = activity.raw_description
                return activity

        workflow = ActivityIngestionWorkflow(
            repository=PandasActivityRepository(),
            cleaner=_IdentityCleaner(),
            label_mapper=TaskLabelMapper(),
            feature_builder=ActivityFeatureBuilder(embedder=None),
        )
        service = ActivityService(workflow)

        ds = OutageDataset(schedule_tasks=pd.DataFrame([
            _make_task("T-001", task_name="Inspect RCP pump seal"),
            _make_task("T-002", task_name="Replace motor bearing"),
            _make_task("M-001", milestone_flag=True,
                       task_name="Start", planned_duration_hours=0.0),
        ]))

        activities = service.ingest_from_p6(ds, outage)

        # Milestones skipped by default
        assert len(activities) == 2
        ids = {a.activity_id for a in activities}
        assert ids == {"T-001", "T-002"}

        # LabelMapper populated taxonomy from task names
        by_id = {a.activity_id: a for a in activities}
        assert by_id["T-001"].component_family == "pump"   # "pump" keyword in description
        assert by_id["T-002"].component_family == "motor"  # "motor" keyword

    def test_ingest_from_p6_includes_milestones_when_flag_off(self, outage):
        from outage_uncertainty.services.activity_service import ActivityService
        from outage_uncertainty.workflows.activity_ingestion_workflow import ActivityIngestionWorkflow
        from outage_uncertainty.adapters.pandas_repository import PandasActivityRepository
        from outage_uncertainty.preprocessing.label_mapper import TaskLabelMapper
        from outage_uncertainty.preprocessing.feature_builder import ActivityFeatureBuilder

        class _IdentityCleaner:
            def clean(self, activity):
                activity.cleaned_description = activity.raw_description
                return activity

        workflow = ActivityIngestionWorkflow(
            repository=PandasActivityRepository(),
            cleaner=_IdentityCleaner(),
            label_mapper=TaskLabelMapper(),
            feature_builder=ActivityFeatureBuilder(embedder=None),
        )
        service = ActivityService(workflow)

        ds = OutageDataset(schedule_tasks=pd.DataFrame([
            _make_task("T-001"),
            _make_task("M-001", milestone_flag=True,
                       task_name="Start", planned_duration_hours=0.0),
        ]))

        activities = service.ingest_from_p6(ds, outage, skip_milestones=False)
        assert len(activities) == 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_row(dataset, outage) -> dict:
    adapter = P6DatasetAdapter()
    return next(adapter.iter_activity_rows(dataset, outage))


def _all_rows(dataset, outage) -> list[dict]:
    adapter = P6DatasetAdapter()
    return list(adapter.iter_activity_rows(dataset, outage))
