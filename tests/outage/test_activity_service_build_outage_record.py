"""
Tests for ActivityService.build_outage_record().

Covers:
- Returns OutageRecord with correct header fields from Outage
- start_date prefers start_actual over start_planned
- end_date prefers finish_actual over finish_planned
- ValueError when neither start date is set
- activities list matches what ingest_from_p6 would return
- plant_id / unit_id copied correctly (including None unit_id)
- Keyword args forwarded to ingest_from_p6
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from outage_model.models import Outage
from outage_uncertainty.domain.activity import ActivityCase
from outage_uncertainty.domain.outage import OutageRecord
from outage_uncertainty.services.activity_service import ActivityService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_outage(
    outage_id="RF-22",
    plant_id="PLANT-A",
    unit_id="UNIT-1",
    start_planned=None,
    start_actual=None,
    finish_planned=None,
    finish_actual=None,
) -> Outage:
    return Outage(
        outage_id=outage_id,
        plant_id=plant_id,
        unit_id=unit_id,
        outage_name=f"{outage_id} Refueling",
        start_planned=start_planned or datetime(2024, 3, 1, tzinfo=timezone.utc),
        finish_planned=finish_planned or datetime(2024, 4, 15, tzinfo=timezone.utc),
        start_actual=start_actual,
        finish_actual=finish_actual,
    )


def _make_activity(activity_id: str) -> ActivityCase:
    return ActivityCase(
        activity_id=activity_id,
        outage_id="RF-22",
        plant_id="PLANT-A",
        raw_description=activity_id,
        planned_duration_hours=8.0,
        predecessor_ids=[],
        successor_ids=[],
    )


def _service_with_activities(activities: list[ActivityCase]) -> ActivityService:
    """Return an ActivityService whose ingest_from_p6 returns the given list."""
    workflow = MagicMock()
    workflow.run.return_value = activities
    return ActivityService(ingestion_workflow=workflow)


# ---------------------------------------------------------------------------
# Header fields
# ---------------------------------------------------------------------------

class TestOutageRecordHeaderFields:

    def test_outage_id_copied(self):
        svc = _service_with_activities([])
        outage = _make_outage(outage_id="RF-99")
        result = svc.build_outage_record(MagicMock(), outage)
        assert result.outage_id == "RF-99"

    def test_plant_id_copied(self):
        svc = _service_with_activities([])
        outage = _make_outage(plant_id="PLANT-B")
        result = svc.build_outage_record(MagicMock(), outage)
        assert result.plant_id == "PLANT-B"

    def test_unit_id_copied(self):
        svc = _service_with_activities([])
        outage = _make_outage(unit_id="UNIT-2")
        result = svc.build_outage_record(MagicMock(), outage)
        assert result.unit_id == "UNIT-2"

    def test_unit_id_none_allowed(self):
        svc = _service_with_activities([])
        outage = _make_outage(unit_id=None)
        result = svc.build_outage_record(MagicMock(), outage)
        assert result.unit_id is None

    def test_returns_outage_record_instance(self):
        svc = _service_with_activities([])
        result = svc.build_outage_record(MagicMock(), _make_outage())
        assert isinstance(result, OutageRecord)


# ---------------------------------------------------------------------------
# start_date resolution
# ---------------------------------------------------------------------------

class TestStartDateResolution:

    def test_start_actual_preferred_over_planned(self):
        actual = datetime(2024, 3, 3, tzinfo=timezone.utc)
        planned = datetime(2024, 3, 1, tzinfo=timezone.utc)
        svc = _service_with_activities([])
        outage = _make_outage(start_planned=planned, start_actual=actual)
        result = svc.build_outage_record(MagicMock(), outage)
        assert result.start_date == actual

    def test_start_planned_used_when_actual_absent(self):
        planned = datetime(2024, 3, 1, tzinfo=timezone.utc)
        svc = _service_with_activities([])
        outage = _make_outage(start_planned=planned, start_actual=None)
        result = svc.build_outage_record(MagicMock(), outage)
        assert result.start_date == planned

    def test_raises_when_neither_start_set(self):
        svc = _service_with_activities([])
        outage = Outage(
            outage_id="RF-22",
            outage_name="RF-22",
            start_planned=None,
            start_actual=None,
        )
        with pytest.raises(ValueError, match="start"):
            svc.build_outage_record(MagicMock(), outage)


# ---------------------------------------------------------------------------
# end_date resolution
# ---------------------------------------------------------------------------

class TestEndDateResolution:

    def test_finish_actual_preferred_over_planned(self):
        actual = datetime(2024, 4, 20, tzinfo=timezone.utc)
        planned = datetime(2024, 4, 15, tzinfo=timezone.utc)
        svc = _service_with_activities([])
        outage = _make_outage(finish_planned=planned, finish_actual=actual)
        result = svc.build_outage_record(MagicMock(), outage)
        assert result.end_date == actual

    def test_finish_planned_used_when_actual_absent(self):
        planned = datetime(2024, 4, 15, tzinfo=timezone.utc)
        svc = _service_with_activities([])
        outage = _make_outage(finish_planned=planned, finish_actual=None)
        result = svc.build_outage_record(MagicMock(), outage)
        assert result.end_date == planned

    def test_end_date_none_when_no_finish_set(self):
        svc = _service_with_activities([])
        outage = Outage(
            outage_id="RF-22",
            outage_name="RF-22",
            start_planned=datetime(2024, 3, 1, tzinfo=timezone.utc),
            finish_planned=None,
            finish_actual=None,
        )
        result = svc.build_outage_record(MagicMock(), outage)
        assert result.end_date is None


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------

class TestActivities:

    def test_activities_list_populated(self):
        acts = [_make_activity("T-001"), _make_activity("T-002")]
        svc = _service_with_activities(acts)
        result = svc.build_outage_record(MagicMock(), _make_outage())
        assert len(result.activities) == 2

    def test_activity_ids_correct(self):
        acts = [_make_activity("T-001"), _make_activity("T-002")]
        svc = _service_with_activities(acts)
        result = svc.build_outage_record(MagicMock(), _make_outage())
        ids = {a.activity_id for a in result.activities}
        assert ids == {"T-001", "T-002"}

    def test_empty_outage_gives_empty_activities(self):
        svc = _service_with_activities([])
        result = svc.build_outage_record(MagicMock(), _make_outage())
        assert result.activities == []


# ---------------------------------------------------------------------------
# Keyword-arg forwarding to ingest_from_p6
# ---------------------------------------------------------------------------

class TestKwargForwarding:

    def test_skip_milestones_forwarded(self):
        svc = ActivityService(ingestion_workflow=MagicMock())
        dataset = MagicMock()
        outage = _make_outage()

        with patch.object(svc, "ingest_from_p6", return_value=[]) as mock_ingest:
            svc.build_outage_record(dataset, outage, skip_milestones=False)
            mock_ingest.assert_called_once_with(
                dataset, outage,
                skip_milestones=False,
                emergent_change_types=None,
                contractor_org_units=None,
            )

    def test_emergent_change_types_forwarded(self):
        svc = ActivityService(ingestion_workflow=MagicMock())
        dataset = MagicMock()
        outage = _make_outage()
        custom_types = {"emergent", "scope_addition", "corrective"}

        with patch.object(svc, "ingest_from_p6", return_value=[]) as mock_ingest:
            svc.build_outage_record(dataset, outage, emergent_change_types=custom_types)
            _, kwargs = mock_ingest.call_args
            assert kwargs["emergent_change_types"] == custom_types

    def test_contractor_org_units_forwarded(self):
        svc = ActivityService(ingestion_workflow=MagicMock())
        dataset = MagicMock()
        outage = _make_outage()
        org_units = {"VENDOR-A", "CONTRACTOR-B"}

        with patch.object(svc, "ingest_from_p6", return_value=[]) as mock_ingest:
            svc.build_outage_record(dataset, outage, contractor_org_units=org_units)
            _, kwargs = mock_ingest.call_args
            assert kwargs["contractor_org_units"] == org_units


# ---------------------------------------------------------------------------
# OutageRiskWorkflow integration smoke-test
# ---------------------------------------------------------------------------

class TestOutageRiskWorkflowIntegration:

    def test_outage_record_accepted_by_workflow(self):
        """build_outage_record() output can be passed directly to OutageRiskWorkflow."""
        from outage_uncertainty.adapters.schedule_network_builder import (
            OutageRecordScheduleBuilder,
        )
        from outage_uncertainty.domain.duration import DurationDistribution
        from outage_uncertainty.domain.result_types import ActivityEstimate
        from outage_uncertainty.schedule_risk.cp_analyzer import CriticalPathRiskAnalyzer
        from outage_uncertainty.schedule_risk.scenario_runner import ScenarioRunner
        from outage_uncertainty.workflows.outage_risk_workflow import OutageRiskWorkflow

        # Stub estimator that always returns a trivial distribution
        class _StubEstimator:
            def run(self, activity, historical_activities):
                dur = activity.planned_duration_hours or 8.0
                dist = DurationDistribution(
                    distribution_type="empirical",
                    samples=[dur],
                    p50=dur,
                )
                return ActivityEstimate(
                    activity_id=activity.activity_id,
                    estimated_distribution=dist,
                    confidence_score=0.9,
                    confidence_tier="data_supported",
                )

        acts = [
            _make_activity("T-001"),
            _make_activity("T-002"),
        ]
        # Wire T-001 → T-002 edges
        acts[0].successor_ids = ["T-002"]
        acts[1].predecessor_ids = ["T-001"]
        acts[0].planned_duration_hours = 6.0
        acts[1].planned_duration_hours = 10.0

        svc = _service_with_activities(acts)
        outage_record = svc.build_outage_record(MagicMock(), _make_outage())

        workflow = OutageRiskWorkflow(
            estimator_workflow=_StubEstimator(),
            schedule_builder=OutageRecordScheduleBuilder(),
            scenario_runner=ScenarioRunner(analyzer=CriticalPathRiskAnalyzer()),
        )
        result = workflow.run(outage_record, historical_activities=[])

        assert "activity_estimates" in result
        assert "risk_summary" in result
        assert set(result["activity_estimates"].keys()) == {"T-001", "T-002"}
        risk = result["risk_summary"]
        assert risk["p50_finish"] == pytest.approx(16.0)
