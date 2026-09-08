"""
Tests for ScheduleNetwork lag and mobilization-lead CPM support (Gap 2).

Covers:
- Lag-free baseline: behaviour unchanged from original implementation
- FS lag shifts EF and CP time correctly (forward pass)
- Lag correctly tightens LF of predecessor (backward pass / float)
- Mobilization lead for source activities (no predecessors)
- Mobilization lead for mid-network activities
- Lag + lead combined on the same edge
- Parallel paths: lag only affects the lagged branch
- Critical path changes when lag makes a previously non-critical path critical
- Lag validation: negative lag raises ValueError
- Lag validation: non-finite lag raises ValueError
- Lags for non-existent edges are silently ignored
- ScheduleActivity.mobilization_lead_hours defaults to 0.0 (backwards-compatible)
- OutageRecordScheduleBuilder passes lags through to ScheduleNetwork
- MonteCarloSimulator samples correct CP times with lags (smoke-test)
"""
from __future__ import annotations

import pytest

from outage_uncertainty.domain.schedule import ScheduleActivity
from outage_uncertainty.schedule_risk.schedule_graph import ScheduleNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _act(
    activity_id: str,
    predecessors: list[str] | None = None,
    successors: list[str] | None = None,
    baseline: float = 0.0,
    lead: float = 0.0,
) -> ScheduleActivity:
    return ScheduleActivity(
        activity_id=activity_id,
        name=activity_id,
        predecessors=predecessors or [],
        successors=successors or [],
        baseline_duration_hours=baseline,
        mobilization_lead_hours=lead,
    )


def _cp(network: ScheduleNetwork, durations: dict[str, float] | None = None) -> dict:
    if durations is None:
        durations = {aid: a.baseline_duration_hours for aid, a in network.activities.items()}
    return network.compute_critical_path(durations)


# ---------------------------------------------------------------------------
# 1. Lag-free baseline — unchanged behaviour
# ---------------------------------------------------------------------------

class TestLagFreeBehaviour:
    def test_linear_chain_no_lag(self):
        # T1(8) → T2(4) → T3(6) : CP = 18
        net = ScheduleNetwork([
            _act("T1", successors=["T2"], baseline=8.0),
            _act("T2", predecessors=["T1"], successors=["T3"], baseline=4.0),
            _act("T3", predecessors=["T2"], baseline=6.0),
        ])
        result = _cp(net)
        assert abs(result["cp_time"] - 18.0) < 1e-9
        assert result["cp_path"] == ["T1", "T2", "T3"]

    def test_parallel_no_lag(self):
        # T1(5) and T2(9) run in parallel → CP = 9
        net = ScheduleNetwork([
            _act("T1", baseline=5.0),
            _act("T2", baseline=9.0),
        ])
        result = _cp(net)
        assert abs(result["cp_time"] - 9.0) < 1e-9
        assert "T2" in result["cp_path"]
        assert "T1" not in result["cp_path"]

    def test_empty_network(self):
        net = ScheduleNetwork([])
        result = _cp(net, {})
        assert result["cp_time"] == 0.0
        assert result["cp_path"] == []


# ---------------------------------------------------------------------------
# 2. FS lag — forward pass
# ---------------------------------------------------------------------------

class TestLagForwardPass:
    def test_lag_extends_cp_time(self):
        # T1(8) --lag=4h--> T2(6) : CP = 8 + 4 + 6 = 18  (vs 14 without lag)
        net = ScheduleNetwork(
            [
                _act("T1", successors=["T2"], baseline=8.0),
                _act("T2", predecessors=["T1"], baseline=6.0),
            ],
            lags={("T1", "T2"): 4.0},
        )
        result = _cp(net)
        assert abs(result["cp_time"] - 18.0) < 1e-9

    def test_zero_lag_same_as_no_lag(self):
        net_with = ScheduleNetwork(
            [
                _act("T1", successors=["T2"], baseline=10.0),
                _act("T2", predecessors=["T1"], baseline=5.0),
            ],
            lags={("T1", "T2"): 0.0},
        )
        net_without = ScheduleNetwork([
            _act("T1", successors=["T2"], baseline=10.0),
            _act("T2", predecessors=["T1"], baseline=5.0),
        ])
        assert abs(_cp(net_with)["cp_time"] - _cp(net_without)["cp_time"]) < 1e-9

    def test_lag_only_on_one_branch(self):
        # Two branches merge into T3.
        # Branch A: T1(6) --lag=4h--> T3(3) : arrives at T3 at t=10+3=13
        # Branch B: T2(12) ----------> T3(3) : arrives at T3 at t=12+3=15
        # T3 EF = 15, CP = 15 (branch B drives it)
        net = ScheduleNetwork(
            [
                _act("T1", successors=["T3"], baseline=6.0),
                _act("T2", successors=["T3"], baseline=12.0),
                _act("T3", predecessors=["T1", "T2"], baseline=3.0),
            ],
            lags={("T1", "T3"): 4.0},
        )
        result = _cp(net)
        assert abs(result["cp_time"] - 15.0) < 1e-9
        assert "T2" in result["cp_path"]
        assert "T3" in result["cp_path"]

    def test_lag_makes_lagged_branch_critical(self):
        # Without lag: T1(6) → T3(3) CP = 9;  T2(7) → T3 CP = 10, so T2 path critical
        # With lag=5h on T1→T3: T1 branch = 6+5+3=14, T2 branch = 7+3=10 → T1 path now critical
        net = ScheduleNetwork(
            [
                _act("T1", successors=["T3"], baseline=6.0),
                _act("T2", successors=["T3"], baseline=7.0),
                _act("T3", predecessors=["T1", "T2"], baseline=3.0),
            ],
            lags={("T1", "T3"): 5.0},
        )
        result = _cp(net)
        assert abs(result["cp_time"] - 14.0) < 1e-9
        assert "T1" in result["cp_path"]
        assert "T2" not in result["cp_path"]


# ---------------------------------------------------------------------------
# 3. FS lag — backward pass / float
# ---------------------------------------------------------------------------

class TestLagBackwardPass:
    def test_lag_tightens_predecessor_float(self):
        # T1(8) --lag=4h--> T2(6)
        # Without lag: float[T1] = 0 (on CP)
        # With lag: T1 is still on CP — LF[T1] should be 8 (= 18 - 4 - 6)
        net = ScheduleNetwork(
            [
                _act("T1", successors=["T2"], baseline=8.0),
                _act("T2", predecessors=["T1"], baseline=6.0),
            ],
            lags={("T1", "T2"): 4.0},
        )
        result = _cp(net)
        # T1 on CP → LF[T1] - EF[T1] = 0
        assert "T1" in result["cp_path"]
        assert "T2" in result["cp_path"]

    def test_non_critical_activity_has_positive_float_with_lag(self):
        # T1(2) --lag=0h--> T3(5)
        # T2(4) --lag=3h--> T3(5)
        # Branch T1: 2+0+5=7; Branch T2: 4+3+5=12 → T2 critical, T1 has float = 12-7=5
        net = ScheduleNetwork(
            [
                _act("T1", successors=["T3"], baseline=2.0),
                _act("T2", successors=["T3"], baseline=4.0),
                _act("T3", predecessors=["T1", "T2"], baseline=5.0),
            ],
            lags={("T2", "T3"): 3.0},
        )
        result = _cp(net)
        assert abs(result["cp_time"] - 12.0) < 1e-9
        assert "T1" not in result["cp_path"]
        assert "T2" in result["cp_path"]


# ---------------------------------------------------------------------------
# 4. Mobilization lead — forward pass
# ---------------------------------------------------------------------------

class TestMobilizationLead:
    def test_source_activity_lead_delays_start(self):
        # T1 has lead=3h, duration=7h → EF=10, CP=10
        net = ScheduleNetwork([
            _act("T1", baseline=7.0, lead=3.0),
        ])
        result = _cp(net)
        assert abs(result["cp_time"] - 10.0) < 1e-9

    def test_mid_network_lead_adds_to_es(self):
        # T1(8) → T2 with lead=2h, duration=6h
        # ES[T2] = EF[T1] + lead[T2] = 8 + 2 = 10; EF[T2] = 10+6=16
        net = ScheduleNetwork([
            _act("T1", successors=["T2"], baseline=8.0),
            _act("T2", predecessors=["T1"], baseline=6.0, lead=2.0),
        ])
        result = _cp(net)
        assert abs(result["cp_time"] - 16.0) < 1e-9

    def test_zero_lead_unchanged(self):
        net_with  = ScheduleNetwork([_act("T1", baseline=8.0, lead=0.0)])
        net_without = ScheduleNetwork([_act("T1", baseline=8.0)])
        assert abs(_cp(net_with)["cp_time"] - _cp(net_without)["cp_time"]) < 1e-9

    def test_lead_on_source_only_no_predecessors(self):
        # Two sources: T1(5, lead=0) and T2(5, lead=3) → CPs 5 and 8
        net = ScheduleNetwork([
            _act("T1", baseline=5.0, lead=0.0),
            _act("T2", baseline=5.0, lead=3.0),
        ])
        result = _cp(net)
        assert abs(result["cp_time"] - 8.0) < 1e-9
        assert "T2" in result["cp_path"]


# ---------------------------------------------------------------------------
# 5. Lag + lead combined
# ---------------------------------------------------------------------------

class TestLagAndLeadCombined:
    def test_lag_and_lead_both_applied(self):
        # T1(10) --lag=2h--> T2(lead=3h, dur=5h)
        # ES[T2] = EF[T1] + lag + lead = 10 + 2 + 3 = 15; EF[T2] = 20
        net = ScheduleNetwork(
            [
                _act("T1", successors=["T2"], baseline=10.0),
                _act("T2", predecessors=["T1"], baseline=5.0, lead=3.0),
            ],
            lags={("T1", "T2"): 2.0},
        )
        result = _cp(net)
        assert abs(result["cp_time"] - 20.0) < 1e-9

    def test_three_node_chain_lag_and_lead(self):
        # T1(6) --lag=1h--> T2(lead=2h, dur=4h) --lag=3h--> T3(lead=1h, dur=5h)
        # ES[T2] = 6+1+2=9,  EF[T2]=13
        # ES[T3] = 13+3+1=17, EF[T3]=22
        net = ScheduleNetwork(
            [
                _act("T1", successors=["T2"], baseline=6.0),
                _act("T2", predecessors=["T1"], successors=["T3"], baseline=4.0, lead=2.0),
                _act("T3", predecessors=["T2"], baseline=5.0, lead=1.0),
            ],
            lags={("T1", "T2"): 1.0, ("T2", "T3"): 3.0},
        )
        result = _cp(net)
        assert abs(result["cp_time"] - 22.0) < 1e-9
        assert result["cp_path"] == ["T1", "T2", "T3"]


# ---------------------------------------------------------------------------
# 6. Lag validation
# ---------------------------------------------------------------------------

class TestLagValidation:
    def test_negative_lag_raises(self):
        with pytest.raises(ValueError, match="negative"):
            ScheduleNetwork(
                [
                    _act("T1", successors=["T2"], baseline=8.0),
                    _act("T2", predecessors=["T1"], baseline=4.0),
                ],
                lags={("T1", "T2"): -2.0},
            )

    def test_nan_lag_raises(self):
        import math
        with pytest.raises(ValueError, match="non-finite"):
            ScheduleNetwork(
                [
                    _act("T1", successors=["T2"], baseline=8.0),
                    _act("T2", predecessors=["T1"], baseline=4.0),
                ],
                lags={("T1", "T2"): math.nan},
            )

    def test_inf_lag_raises(self):
        import math
        with pytest.raises(ValueError, match="non-finite"):
            ScheduleNetwork(
                [
                    _act("T1", successors=["T2"], baseline=8.0),
                    _act("T2", predecessors=["T1"], baseline=4.0),
                ],
                lags={("T1", "T2"): math.inf},
            )

    def test_lag_for_nonexistent_edge_ignored(self):
        # Lag defined for T1→T3 which is not a real edge; should not raise
        net = ScheduleNetwork(
            [
                _act("T1", successors=["T2"], baseline=5.0),
                _act("T2", predecessors=["T1"], baseline=5.0),
                _act("T3", baseline=3.0),
            ],
            lags={("T1", "T3"): 2.0},  # T3 is not a successor of T1
        )
        result = _cp(net)
        # T1→T2 chain: CP=10; T3: 3 → CP=10
        assert abs(result["cp_time"] - 10.0) < 1e-9

    def test_lag_for_unknown_activity_id_ignored(self):
        # Lag dict references "GHOST" which is not in the network — silently ignored
        net = ScheduleNetwork(
            [
                _act("T1", successors=["T2"], baseline=6.0),
                _act("T2", predecessors=["T1"], baseline=4.0),
            ],
            lags={("T1", "T2"): 1.0, ("GHOST", "T2"): 5.0},
        )
        result = _cp(net)
        # Only the real lag T1→T2 (1h) applies: 6+1+4=11
        assert abs(result["cp_time"] - 11.0) < 1e-9


# ---------------------------------------------------------------------------
# 7. Backwards compatibility: mobilization_lead_hours defaults to 0
# ---------------------------------------------------------------------------

class TestBackwardsCompatibility:
    def test_schedule_activity_lead_defaults_to_zero(self):
        act = ScheduleActivity(activity_id="T1", name="T1")
        assert act.mobilization_lead_hours == 0.0

    def test_schedule_network_no_lags_arg(self):
        # Existing construction pattern still works
        net = ScheduleNetwork([
            _act("T1", successors=["T2"], baseline=5.0),
            _act("T2", predecessors=["T1"], baseline=3.0),
        ])
        result = _cp(net)
        assert abs(result["cp_time"] - 8.0) < 1e-9


# ---------------------------------------------------------------------------
# 8. OutageRecordScheduleBuilder passes lags through
# ---------------------------------------------------------------------------

class TestBuilderLagPassthrough:
    def test_lags_reach_schedule_network(self):
        from datetime import datetime
        from outage_uncertainty.adapters.schedule_network_builder import OutageRecordScheduleBuilder
        from outage_uncertainty.domain.activity import ActivityCase
        from outage_uncertainty.domain.outage import OutageRecord

        def _case(aid, dur, preds=None, succs=None):
            return ActivityCase(
                activity_id=aid, outage_id="RF-22", plant_id="P",
                raw_description=aid,
                planned_duration_hours=dur,
                predecessor_ids=preds or [],
                successor_ids=succs or [],
            )

        outage = OutageRecord(
            outage_id="RF-22", plant_id="P", unit_id=None,
            start_date=datetime(2024, 3, 1),
            activities=[
                _case("T1", 8.0, succs=["T2"]),
                _case("T2", 6.0, preds=["T1"]),
            ],
        )

        # Without lag: CP = 14; with lag=3h: CP = 17
        builder = OutageRecordScheduleBuilder()
        _, cp_no_lag = builder.build(outage, {})
        _, cp_with_lag = builder.build(outage, {}, lags={("T1", "T2"): 3.0})

        assert abs(cp_no_lag  - 14.0) < 1e-9
        assert abs(cp_with_lag - 17.0) < 1e-9


# ---------------------------------------------------------------------------
# 9. MonteCarloSimulator with lags — smoke-test
# ---------------------------------------------------------------------------

class TestMCSmokeWithLag:
    def test_mc_cp_times_shifted_by_lag(self):
        from outage_uncertainty.domain.duration import DurationDistribution
        from outage_uncertainty.schedule_risk.monte_carlo import MonteCarloSimulator

        dist = DurationDistribution(samples=[4.0, 6.0, 8.0])

        # Network: T1(fixed=10) --lag=5h--> T2(dist)
        net = ScheduleNetwork(
            [
                ScheduleActivity(
                    activity_id="T1", name="T1",
                    successors=["T2"],
                    baseline_duration_hours=10.0,
                    duration_distribution=None,  # deterministic
                ),
                ScheduleActivity(
                    activity_id="T2", name="T2",
                    predecessors=["T1"],
                    baseline_duration_hours=6.0,
                    duration_distribution=dist,
                ),
            ],
            lags={("T1", "T2"): 5.0},
        )

        sim = MonteCarloSimulator(net, n_samples=300)
        result = sim.run()

        # T2 is sampled from {4, 6, 8}; T1 is fixed=10
        # CP = 10 + 5 + T2_sample ∈ {19, 21, 23}
        assert all(t in {19.0, 21.0, 23.0} for t in result.cp_times)
        # Both T1 and T2 always on CP (single chain)
        assert result.activity_criticality.get("T1", 0) == 300
        assert result.activity_criticality.get("T2", 0) == 300
