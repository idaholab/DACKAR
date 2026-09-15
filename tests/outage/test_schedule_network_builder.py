"""
Tests for OutageRecordScheduleBuilder.

Covers:
- Basic conversion: ActivityCase → ScheduleActivity (all fields)
- Duration distribution attached when Stage D estimate exists
- Deterministic fallback (None distribution) when no estimate
- Baseline CP time: linear chain, parallel paths, single activity
- Dangling-edge pruning (prune_dangling_edges=True / False)
- None planned_duration_hours → baseline 0.0
- Empty outage returns (empty ScheduleNetwork, 0.0)
- OutageRiskWorkflow integration smoke-test
"""
from __future__ import annotations

from datetime import datetime

import pytest

from outage_uncertainty.adapters.schedule_network_builder import OutageRecordScheduleBuilder
from outage_uncertainty.domain.activity import ActivityCase
from outage_uncertainty.domain.duration import DurationDistribution
from outage_uncertainty.domain.outage import OutageRecord
from outage_uncertainty.domain.result_types import ActivityEstimate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _outage(activities) -> OutageRecord:
    return OutageRecord(
        outage_id="RF-22",
        plant_id="PLANT-A",
        unit_id="UNIT-1",
        start_date=datetime(2024, 3, 1),
        activities=activities,
    )


def _case(
    activity_id: str,
    planned_duration_hours: float | None = 8.0,
    predecessors: list[str] | None = None,
    successors: list[str] | None = None,
    raw_description: str = "",
) -> ActivityCase:
    return ActivityCase(
        activity_id=activity_id,
        outage_id="RF-22",
        plant_id="PLANT-A",
        raw_description=raw_description or activity_id,
        planned_duration_hours=planned_duration_hours,
        predecessor_ids=predecessors or [],
        successor_ids=successors or [],
    )


def _estimate(activity_id: str, samples: list[float]) -> ActivityEstimate:
    dist = DurationDistribution(
        distribution_type="empirical",
        samples=samples,
        p50=sorted(samples)[len(samples) // 2],
    )
    return ActivityEstimate(
        activity_id=activity_id,
        estimated_distribution=dist,
        confidence_score=0.8,
        confidence_tier="high",
    )


# ---------------------------------------------------------------------------
# 1. Basic field conversion
# ---------------------------------------------------------------------------

class TestFieldConversion:
    def test_activity_id_preserved(self):
        outage = _outage([_case("T-001")])
        builder = OutageRecordScheduleBuilder()
        network, _ = builder.build(outage, {})
        assert "T-001" in network.activities

    def test_name_from_raw_description(self):
        outage = _outage([_case("T-001", raw_description="Replace pump seal")])
        builder = OutageRecordScheduleBuilder()
        network, _ = builder.build(outage, {})
        assert network.activities["T-001"].name == "Replace pump seal"

    def test_baseline_duration_from_planned(self):
        outage = _outage([_case("T-001", planned_duration_hours=12.0)])
        builder = OutageRecordScheduleBuilder()
        network, _ = builder.build(outage, {})
        assert network.activities["T-001"].baseline_duration_hours == 12.0

    def test_none_planned_duration_gives_zero(self):
        outage = _outage([_case("T-001", planned_duration_hours=None)])
        builder = OutageRecordScheduleBuilder()
        network, _ = builder.build(outage, {})
        assert network.activities["T-001"].baseline_duration_hours == 0.0

    def test_predecessor_and_successor_edges(self):
        outage = _outage([
            _case("T-001", successors=["T-002"]),
            _case("T-002", predecessors=["T-001"]),
        ])
        builder = OutageRecordScheduleBuilder()
        network, _ = builder.build(outage, {})
        assert network.activities["T-001"].successors == ["T-002"]
        assert network.activities["T-002"].predecessors == ["T-001"]


# ---------------------------------------------------------------------------
# 2. Duration distribution
# ---------------------------------------------------------------------------

class TestDurationDistribution:
    def test_distribution_attached_when_estimate_exists(self):
        outage = _outage([_case("T-001")])
        estimates = {"T-001": _estimate("T-001", [8.0, 10.0, 12.0])}
        builder = OutageRecordScheduleBuilder()
        network, _ = builder.build(outage, estimates)
        assert network.activities["T-001"].duration_distribution is not None

    def test_distribution_is_none_when_no_estimate(self):
        outage = _outage([_case("T-001")])
        builder = OutageRecordScheduleBuilder()
        network, _ = builder.build(outage, {})
        assert network.activities["T-001"].duration_distribution is None

    def test_distribution_samples_are_correct(self):
        outage = _outage([_case("T-001")])
        estimates = {"T-001": _estimate("T-001", [8.0, 10.0, 12.0])}
        builder = OutageRecordScheduleBuilder()
        network, _ = builder.build(outage, estimates)
        dist = network.activities["T-001"].duration_distribution
        assert dist.samples == [8.0, 10.0, 12.0]

    def test_partial_estimates_mixed_correctly(self):
        """Activities with and without estimates in the same network."""
        outage = _outage([
            _case("T-001"),
            _case("T-002"),
        ])
        estimates = {"T-001": _estimate("T-001", [6.0, 8.0, 10.0])}
        builder = OutageRecordScheduleBuilder()
        network, _ = builder.build(outage, estimates)
        assert network.activities["T-001"].duration_distribution is not None
        assert network.activities["T-002"].duration_distribution is None


# ---------------------------------------------------------------------------
# 3. Baseline CP time
# ---------------------------------------------------------------------------

class TestBaselineCPTime:
    def test_linear_chain(self):
        # T-001 (8h) → T-002 (4h) → T-003 (6h) : CP = 18h
        outage = _outage([
            _case("T-001", planned_duration_hours=8.0, successors=["T-002"]),
            _case("T-002", planned_duration_hours=4.0, predecessors=["T-001"], successors=["T-003"]),
            _case("T-003", planned_duration_hours=6.0, predecessors=["T-002"]),
        ])
        builder = OutageRecordScheduleBuilder()
        _, baseline_cp = builder.build(outage, {})
        assert abs(baseline_cp - 18.0) < 1e-9

    def test_parallel_paths_longest_wins(self):
        # T-001 (5h) and T-002 (9h) both start from project start → CP = 9h
        outage = _outage([
            _case("T-001", planned_duration_hours=5.0),
            _case("T-002", planned_duration_hours=9.0),
        ])
        builder = OutageRecordScheduleBuilder()
        _, baseline_cp = builder.build(outage, {})
        assert abs(baseline_cp - 9.0) < 1e-9

    def test_single_activity(self):
        outage = _outage([_case("T-001", planned_duration_hours=24.0)])
        builder = OutageRecordScheduleBuilder()
        _, baseline_cp = builder.build(outage, {})
        assert abs(baseline_cp - 24.0) < 1e-9

    def test_empty_outage_gives_zero(self):
        outage = _outage([])
        builder = OutageRecordScheduleBuilder()
        network, baseline_cp = builder.build(outage, {})
        assert baseline_cp == 0.0
        assert len(network.activities) == 0

    def test_critical_path_contains_longest_chain_activities(self):
        # Long chain: T-A(10h) → T-B(10h) = 20h
        # Short chain: T-C(5h)             = 5h
        outage = _outage([
            _case("T-A", planned_duration_hours=10.0, successors=["T-B"]),
            _case("T-B", planned_duration_hours=10.0, predecessors=["T-A"]),
            _case("T-C", planned_duration_hours=5.0),
        ])
        builder = OutageRecordScheduleBuilder()
        network, baseline_cp = builder.build(outage, {})
        assert abs(baseline_cp - 20.0) < 1e-9
        # Verify CP includes the long chain
        baseline_durations = {aid: a.baseline_duration_hours for aid, a in network.activities.items()}
        cp_result = network.compute_critical_path(baseline_durations)
        assert "T-A" in cp_result["cp_path"]
        assert "T-B" in cp_result["cp_path"]
        assert "T-C" not in cp_result["cp_path"]


# ---------------------------------------------------------------------------
# 4. Dangling-edge pruning
# ---------------------------------------------------------------------------

class TestDanglingEdgePruning:
    def test_dangling_successor_pruned_by_default(self):
        """Successor references an activity not in the outage → silently removed."""
        outage = _outage([
            _case("T-001", successors=["EXTERNAL-MILESTONE"]),
        ])
        builder = OutageRecordScheduleBuilder()
        network, _ = builder.build(outage, {})
        # No ValueError; dangling edge removed
        assert network.activities["T-001"].successors == []

    def test_dangling_predecessor_pruned_by_default(self):
        outage = _outage([
            _case("T-001", predecessors=["EXTERNAL-PREDECESSOR"]),
        ])
        builder = OutageRecordScheduleBuilder()
        network, _ = builder.build(outage, {})
        assert network.activities["T-001"].predecessors == []

    def test_valid_edges_not_pruned(self):
        outage = _outage([
            _case("T-001", successors=["T-002"]),
            _case("T-002", predecessors=["T-001"]),
        ])
        builder = OutageRecordScheduleBuilder()
        network, _ = builder.build(outage, {})
        assert network.activities["T-001"].successors == ["T-002"]

    def test_prune_false_raises_on_dangling_edge(self):
        outage = _outage([
            _case("T-001", successors=["EXTERNAL-MILESTONE"]),
        ])
        builder = OutageRecordScheduleBuilder(prune_dangling_edges=False)
        with pytest.raises(ValueError, match="EXTERNAL-MILESTONE"):
            builder.build(outage, {})


# ---------------------------------------------------------------------------
# 5. MonteCarloSimulator integration smoke-test
# ---------------------------------------------------------------------------

class TestMonteCarloIntegration:
    def test_mc_runs_on_built_network(self):
        """End-to-end: OutageRecord → ScheduleNetwork → MonteCarloSimulator → SimulationResult."""
        from outage_uncertainty.schedule_risk.monte_carlo import MonteCarloSimulator

        outage = _outage([
            _case("T-001", planned_duration_hours=8.0,  successors=["T-003"]),
            _case("T-002", planned_duration_hours=12.0, successors=["T-003"]),
            _case("T-003", planned_duration_hours=4.0,  predecessors=["T-001", "T-002"]),
        ])
        estimates = {
            "T-001": _estimate("T-001", [6.0, 8.0, 10.0]),
            "T-002": _estimate("T-002", [10.0, 12.0, 16.0]),
        }

        builder = OutageRecordScheduleBuilder()
        network, baseline_cp_time = builder.build(outage, estimates)

        # T-002(12) → T-003(4) is the critical path: CP = 16h
        assert abs(baseline_cp_time - 16.0) < 1e-9

        sim = MonteCarloSimulator(network, n_samples=200)
        result = sim.run()

        assert len(result.cp_times) == 200
        assert all(t > 0 for t in result.cp_times)
        # T-002 should appear on CP most often (it's the bottleneck)
        assert "T-002" in result.activity_criticality
        assert result.activity_criticality.get("T-002", 0) > result.activity_criticality.get("T-001", 0)

    def test_deterministic_activities_give_constant_cp(self):
        """Activities with no distribution always use baseline_duration → constant CP time."""
        from outage_uncertainty.schedule_risk.monte_carlo import MonteCarloSimulator

        outage = _outage([
            _case("T-001", planned_duration_hours=10.0, successors=["T-002"]),
            _case("T-002", planned_duration_hours=5.0,  predecessors=["T-001"]),
        ])

        builder = OutageRecordScheduleBuilder()
        network, baseline_cp_time = builder.build(outage, {})  # no estimates → deterministic

        sim = MonteCarloSimulator(network, n_samples=100)
        result = sim.run()

        # All CP times must equal the baseline (no randomness)
        assert all(abs(t - baseline_cp_time) < 1e-9 for t in result.cp_times)


# ---------------------------------------------------------------------------
# 6. OutageRiskWorkflow integration smoke-test
# ---------------------------------------------------------------------------

class TestOutageRiskWorkflowIntegration:
    def test_workflow_runs_end_to_end(self):
        """Smoke-test: schedule_builder plugs into OutageRiskWorkflow correctly."""
        from outage_uncertainty.workflows.outage_risk_workflow import OutageRiskWorkflow
        from outage_uncertainty.schedule_risk.scenario_runner import ScenarioRunner
        from outage_uncertainty.schedule_risk.cp_analyzer import CriticalPathRiskAnalyzer

        # Minimal estimator_workflow stub that returns a pre-built ActivityEstimate
        class _StubEstimatorWorkflow:
            def run(self, activity, historical_activities):
                return _estimate(activity.activity_id, [activity.planned_duration_hours or 8.0])

        workflow = OutageRiskWorkflow(
            estimator_workflow=_StubEstimatorWorkflow(),
            schedule_builder=OutageRecordScheduleBuilder(),
            scenario_runner=ScenarioRunner(
                analyzer=CriticalPathRiskAnalyzer()
            ),
        )

        outage = _outage([
            _case("T-001", planned_duration_hours=6.0,  successors=["T-003"]),
            _case("T-002", planned_duration_hours=10.0, successors=["T-003"]),
            _case("T-003", planned_duration_hours=3.0,  predecessors=["T-001", "T-002"]),
        ])

        result = workflow.run(outage, historical_activities=[])

        assert "activity_estimates" in result
        assert "risk_summary" in result
        assert set(result["activity_estimates"].keys()) == {"T-001", "T-002", "T-003"}

        risk = result["risk_summary"]
        assert "robustness" in risk
        assert "p50_finish" in risk
        assert "p80_finish" in risk
        assert "criticality_index" in risk
        # T-002 (10h) is the bottleneck → should appear on CP
        assert "T-002" in risk["criticality_index"]
