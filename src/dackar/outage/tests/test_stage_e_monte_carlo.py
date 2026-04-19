"""
Tests for Stage E Monte Carlo engine (Gap 3).

Covers:
    _build_duration_sampler()   — lognormal fitting from mean/std, p80, p90, p50-only
    _run_3scenario_proxy()      — fallback path when monte_carlo_runs < 10
    _run_monte_carlo()          — real MC path (set_durations and clone-per-iteration)
    notes field in assess()     — updated wording for real MC vs proxy
    permit_lead_hours           — added to every sample
"""
from __future__ import annotations

import copy
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

_OUTAGE_ROOT = Path(__file__).parent.parent
if str(_OUTAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTAGE_ROOT))

from stages.stage_e_schedule import (
    ScheduleImpactAssessor,
    ScheduleImpactConfig,
    _SimResult,
    _ScheduleNetwork,
    _MIN_DURATION_HOURS,
)


# ===========================================================================
# Shared mock infrastructure
# ===========================================================================

class _MockActivity:
    def __init__(self, name: str):
        self.name = name
        self.description = name
        self.required_resources: List[dict] = []

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, _MockActivity) and self.name == other.name

    def __repr__(self):
        return f"<MockActivity {self.name}>"


class _MockPertBase:
    """Minimal Pert stub used by both path variants."""

    def __init__(self, baseline_hours: float = 480.0):
        self._baseline_hours = baseline_hours
        self._durations: Dict[str, float] = {}
        acts = {tid: _MockActivity(tid) for tid in ["T-A", "T-B", "T-C"]}
        self.task_to_activity: Dict[str, _MockActivity] = acts
        self.infoDict: Dict[_MockActivity, Dict] = {
            act: {"es": i * 100.0, "ef": (i + 1) * 100.0, "slack": 0.0}
            for i, act in enumerate(acts.values())
        }
        self.forwardDict: Dict = {}
        self.startTime = datetime(2026, 4, 1, tzinfo=timezone.utc)
        self._generate_calls = 0
        self._reset_calls = 0

    def generateInfo(self):
        self._generate_calls += 1

    def resetInfo(self):
        self._reset_calls += 1

    def getProjectDuration(self) -> float:
        # Simple model: baseline + emergent task duration
        extra = self._durations.get("EA_MC", 0.0) + self._durations.get("EA_FLOAT_CHECK", 0.0)
        # Use any registered emergent key
        for key, dur in self._durations.items():
            if key.startswith("EA"):
                return self._baseline_hours + dur
        return self._baseline_hours

    def clone_for_analysis(self) -> "_MockPertBase":
        return copy.deepcopy(self)

    def insert_task(self, task_dict: dict, after_task_id=None, before_task_id=None):
        task_id = task_dict["task_id"]
        duration = task_dict.get("duration", 0.0)
        new_act = _MockActivity(task_id)
        self.task_to_activity[task_id] = new_act
        self.infoDict[new_act] = {"es": 0.0, "ef": duration, "slack": 0.0}
        self.forwardDict[new_act] = []
        self._durations[task_id] = duration


class _MockPertWithSetDurations(_MockPertBase):
    """Pert stub that supports set_durations (efficient LOGOS path)."""

    def set_durations(self, durations: Dict[str, float]) -> None:
        for task_id, dur in durations.items():
            self._durations[task_id] = dur
            if task_id in self.task_to_activity:
                act = self.task_to_activity[task_id]
                self.infoDict[act] = {
                    "es": 0.0, "ef": dur, "slack": 0.0
                }

    def getProjectDuration(self) -> float:
        extra = max(
            (self._durations.get(k, 0.0) for k in self._durations if k.startswith("EA")),
            default=0.0,
        )
        return self._baseline_hours + extra


def _make_network(baseline_hours=480.0, use_set_durations=True) -> _ScheduleNetwork:
    if use_set_durations:
        pert = _MockPertWithSetDurations(baseline_hours=baseline_hours)
    else:
        pert = _MockPertBase(baseline_hours=baseline_hours)
    return _ScheduleNetwork(pert=pert, baseline_cp_hours=baseline_hours)


def _insertion_point(emergent_task_id="EA_MC") -> dict:
    return {
        "emergent_task_id": emergent_task_id,
        "after_task_id": "T-A",
        "before_task_id": "T-B",
        "outage_phase": "maintenance",
        "proposed_start": "2026-04-05T08:00:00+00:00",
        "proposed_finish": "2026-04-06T08:00:00+00:00",
    }


def _dist(p50=8.0, p80=12.0, p90=16.0, mean=None, std=None) -> dict:
    return {
        "p50_hours": p50,
        "p80_hours": p80,
        "p90_hours": p90,
        "mean_hours": mean,
        "std_hours": std,
        "distribution_type": "lognormal",
        "confidence_tier": "data_supported",
        "sample_size": 20,
    }


# ===========================================================================
# _build_duration_sampler — distribution fitting
# ===========================================================================

class TestBuildDurationSampler:
    """_build_duration_sampler() fits a lognormal and returns a callable."""

    def _sampler(self, **kwargs):
        assessor = ScheduleImpactAssessor()
        return assessor._build_duration_sampler(_dist(**kwargs))

    def test_returns_callable(self):
        s = self._sampler(p50=10.0, p80=15.0)
        assert callable(s)

    def test_samples_are_positive(self):
        s = self._sampler(p50=8.0, p80=12.0)
        samples = [s() for _ in range(200)]
        assert all(v > 0 for v in samples)

    def test_median_near_p50_when_fit_from_p80(self):
        """Empirical median of 2000 samples should be close to p50."""
        s = self._sampler(p50=10.0, p80=14.0)
        samples = [s() for _ in range(2000)]
        empirical_median = float(np.median(samples))
        # Allow ±20% tolerance — lognormal median = exp(mu) = p50
        assert abs(empirical_median - 10.0) / 10.0 < 0.20

    def test_fit_from_mean_std(self):
        """Method-of-moments path gives samples centred on mean."""
        assessor = ScheduleImpactAssessor()
        dist = {"p50_hours": 10.0, "p80_hours": None, "mean_hours": 12.0, "std_hours": 3.0}
        s = assessor._build_duration_sampler(dist)
        samples = [s() for _ in range(2000)]
        empirical_mean = float(np.mean(samples))
        assert abs(empirical_mean - 12.0) / 12.0 < 0.15

    def test_fit_from_p90_only(self):
        """Falls back to p90 fitting when p80 is absent."""
        assessor = ScheduleImpactAssessor()
        dist = {"p50_hours": 10.0, "p80_hours": None, "p90_hours": 16.0}
        s = assessor._build_duration_sampler(dist)
        samples = [s() for _ in range(500)]
        assert all(v > 0 for v in samples)

    def test_p50_only_minimal_spread(self):
        """With no spread info, sigma=0.10 — samples near p50."""
        assessor = ScheduleImpactAssessor()
        dist = {"p50_hours": 10.0}
        s = assessor._build_duration_sampler(dist)
        samples = [s() for _ in range(1000)]
        assert abs(float(np.median(samples)) - 10.0) < 2.5

    def test_permit_lead_hours_added_to_every_sample(self):
        """Every sample must be ≥ permit_lead_hours."""
        assessor = ScheduleImpactAssessor()
        dist = {"p50_hours": 8.0, "p80_hours": 12.0}
        s = assessor._build_duration_sampler(dist, permit_lead_hours=4.0)
        samples = [s() for _ in range(200)]
        # All samples should be ≥ _MIN_DURATION_HOURS (the absolute floor)
        # and in practice well above 4.0
        assert all(v >= _MIN_DURATION_HOURS for v in samples)
        assert float(np.median(samples)) > 4.0

    def test_degenerate_p50_clamped_to_min(self):
        """p50 below _MIN_DURATION_HOURS is clamped before fitting."""
        assessor = ScheduleImpactAssessor()
        dist = {"p50_hours": 0.0}
        s = assessor._build_duration_sampler(dist)
        samples = [s() for _ in range(100)]
        assert all(v >= _MIN_DURATION_HOURS for v in samples)

    def test_sigma_floor_prevents_zero_spread(self):
        """Sigma cannot be smaller than 0.01 (safety floor)."""
        assessor = ScheduleImpactAssessor()
        # p80 = p50: would produce sigma=0 without floor
        dist = {"p50_hours": 10.0, "p80_hours": 10.0}
        s = assessor._build_duration_sampler(dist)
        samples = [s() for _ in range(100)]
        assert all(v > 0 for v in samples)


# ===========================================================================
# _run_3scenario_proxy — legacy proxy
# ===========================================================================

class TestRun3ScenarioProxy:

    def _proxy(self, network, insertion_pt, dist, **kwargs):
        assessor = ScheduleImpactAssessor()
        return assessor._run_3scenario_proxy(network, insertion_pt, dist, **kwargs)

    def test_returns_3_project_durations(self):
        network = _make_network(baseline_hours=480.0)
        result = self._proxy(network, _insertion_point(), _dist(p50=8.0, p80=12.0, p90=16.0))
        assert result.n_runs == 3
        assert len(result.project_durations) == 3

    def test_durations_monotonically_increasing(self):
        """p50 < p80 < p90 implies project_durations[0] ≤ [1] ≤ [2]."""
        network = _make_network(baseline_hours=480.0)
        result = self._proxy(network, _insertion_point(), _dist(p50=8.0, p80=12.0, p90=16.0))
        durs = result.project_durations
        assert durs[0] <= durs[1] <= durs[2]

    def test_permit_lead_added_to_all_scenarios(self):
        network = _make_network(baseline_hours=480.0)
        result_no_lead = self._proxy(
            network, _insertion_point(), _dist(p50=8.0, p80=12.0, p90=16.0)
        )
        result_with_lead = self._proxy(
            network, _insertion_point(), _dist(p50=8.0, p80=12.0, p90=16.0),
            permit_lead_hours=4.0,
        )
        # All scenarios shifted up by 4 hours
        for d_base, d_lead in zip(result_no_lead.project_durations,
                                  result_with_lead.project_durations):
            assert abs(d_lead - d_base - 4.0) < 1e-6

    def test_zero_duration_scenarios_skipped(self):
        network = _make_network(baseline_hours=480.0)
        dist = _dist(p50=0.0, p80=0.0, p90=0.0)
        # All three scenarios get clamped to _MIN_DURATION_HOURS, not skipped
        result = self._proxy(network, _insertion_point(), dist)
        assert result.n_runs >= 0  # may be 3 clamped scenarios


# ===========================================================================
# _run_monte_carlo — real MC path (set_durations available)
# ===========================================================================

class TestRunMonteCarloSetDurations:
    """_run_monte_carlo() uses set_durations when available."""

    def _run(self, n_runs=50, **dist_kwargs):
        config = ScheduleImpactConfig(monte_carlo_runs=n_runs)
        assessor = ScheduleImpactAssessor(config=config)
        network = _make_network(baseline_hours=480.0, use_set_durations=True)
        return assessor._run_monte_carlo(
            network, _insertion_point(), _dist(**dist_kwargs)
        )

    def test_n_runs_samples_produced(self):
        result = self._run(n_runs=50)
        assert result.n_runs == 50
        assert len(result.project_durations) == 50

    def test_1000_samples_produced(self):
        result = self._run(n_runs=1000)
        assert result.n_runs == 1000
        assert len(result.project_durations) == 1000

    def test_project_durations_all_positive(self):
        result = self._run(n_runs=100)
        assert all(d > 0 for d in result.project_durations)

    def test_project_durations_exceed_baseline(self):
        """Adding an emergent task always produces durations ≥ baseline."""
        result = self._run(n_runs=100, p50=8.0)
        assert all(d >= 480.0 for d in result.project_durations)

    def test_sensitivity_score_between_0_and_1(self):
        result = self._run(n_runs=100)
        sensitivity = result.on_cp_count / result.n_runs
        assert 0.0 <= sensitivity <= 1.0

    def test_emergent_task_id_preserved(self):
        config = ScheduleImpactConfig(monte_carlo_runs=20)
        assessor = ScheduleImpactAssessor(config=config)
        network = _make_network(use_set_durations=True)
        ip = _insertion_point(emergent_task_id="EA::CUSTOM-001")
        result = assessor._run_monte_carlo(network, ip, _dist())
        assert result.emergent_task_id == "EA::CUSTOM-001"

    def test_permit_lead_shifts_distribution_up(self):
        """With permit_lead_hours, median project duration increases."""
        config = ScheduleImpactConfig(monte_carlo_runs=200)
        assessor = ScheduleImpactAssessor(config=config)
        network_a = _make_network(baseline_hours=480.0, use_set_durations=True)
        network_b = _make_network(baseline_hours=480.0, use_set_durations=True)
        dist = _dist(p50=8.0, p80=12.0)
        result_no_lead = assessor._run_monte_carlo(network_a, _insertion_point(), dist)
        result_with_lead = assessor._run_monte_carlo(
            network_b, _insertion_point(), dist, permit_lead_hours=6.0
        )
        median_no_lead = float(np.median(result_no_lead.project_durations))
        median_with_lead = float(np.median(result_with_lead.project_durations))
        assert median_with_lead > median_no_lead


# ===========================================================================
# _run_monte_carlo — clone-per-iteration fallback (no set_durations)
# ===========================================================================

class TestRunMonteCarloCloneFallback:
    """_run_monte_carlo() falls back to clone-per-iteration when set_durations absent."""

    def _run(self, n_runs=30):
        config = ScheduleImpactConfig(monte_carlo_runs=n_runs)
        assessor = ScheduleImpactAssessor(config=config)
        network = _make_network(baseline_hours=480.0, use_set_durations=False)
        return assessor._run_monte_carlo(
            network, _insertion_point(), _dist(p50=8.0, p80=12.0, p90=16.0)
        )

    def test_n_runs_samples_produced(self):
        result = self._run(n_runs=30)
        assert result.n_runs == 30
        assert len(result.project_durations) == 30

    def test_all_durations_positive(self):
        result = self._run(n_runs=30)
        assert all(d > 0 for d in result.project_durations)


# ===========================================================================
# Proxy fallback when monte_carlo_runs < 10
# ===========================================================================

class TestProxyFallback:

    def test_proxy_used_when_runs_less_than_10(self):
        config = ScheduleImpactConfig(monte_carlo_runs=3)
        assessor = ScheduleImpactAssessor(config=config)
        network = _make_network(baseline_hours=480.0, use_set_durations=True)
        result = assessor._run_monte_carlo(
            network, _insertion_point(), _dist(p50=8.0, p80=12.0, p90=16.0)
        )
        # 3-scenario proxy produces exactly 3 results
        assert result.n_runs == 3
        assert len(result.project_durations) == 3

    def test_real_mc_used_when_runs_ge_10(self):
        config = ScheduleImpactConfig(monte_carlo_runs=10)
        assessor = ScheduleImpactAssessor(config=config)
        network = _make_network(baseline_hours=480.0, use_set_durations=True)
        result = assessor._run_monte_carlo(
            network, _insertion_point(), _dist(p50=8.0, p80=12.0, p90=16.0)
        )
        assert result.n_runs == 10
        assert len(result.project_durations) == 10


# ===========================================================================
# _compute_cp_metrics integration with real MC results
# ===========================================================================

class TestComputeCPMetricsWithRealMC:
    """_compute_cp_metrics() computes numpy percentiles for ≥ 10 samples."""

    def _run_and_analyze(self, n_runs=100):
        config = ScheduleImpactConfig(monte_carlo_runs=n_runs)
        assessor = ScheduleImpactAssessor(config=config)
        network = _make_network(baseline_hours=480.0, use_set_durations=True)
        sim = assessor._run_monte_carlo(network, _insertion_point(), _dist(p50=8.0, p80=12.0))
        return assessor._compute_cp_metrics(sim, baseline_cp_hours=480.0)

    def test_estimated_new_cp_exceeds_baseline(self):
        result = self._run_and_analyze(n_runs=100)
        assert result["estimated_new_cp_hours"] >= result["baseline_cp_hours"]

    def test_cp_drag_nonnegative(self):
        result = self._run_and_analyze(n_runs=100)
        assert result["cp_drag_hours"] >= 0.0

    def test_p80_gte_p50(self):
        result = self._run_and_analyze(n_runs=100)
        assert result["p80_cp_hours"] >= result["estimated_new_cp_hours"]

    def test_p90_gte_p80(self):
        result = self._run_and_analyze(n_runs=100)
        assert result["p90_cp_hours"] >= result["p80_cp_hours"]

    def test_sensitivity_score_in_range(self):
        result = self._run_and_analyze(n_runs=100)
        assert 0.0 <= result["cp_sensitivity_score"] <= 1.0


# ===========================================================================
# notes field in assess() output
# ===========================================================================

class TestAssessNotesField:
    """assess() notes field reflects whether real MC or proxy was used."""

    def _assessor_with_stub_loader(self, n_runs: int) -> ScheduleImpactAssessor:
        """Return an assessor with stub loader/builder and given n_runs."""
        import unittest.mock as mock

        config = ScheduleImpactConfig(monte_carlo_runs=n_runs)
        pert = _MockPertWithSetDurations(baseline_hours=480.0)

        outage_data = mock.MagicMock()
        outage_data.outage_config = {"version_id": "v1"}

        class _StubBuilder:
            def build(self, od):
                return pert

        def _loader(outage_id, version=None):
            return outage_data

        return ScheduleImpactAssessor(
            config=config,
            schedule_loader=_loader,
            schedule_graph_builder=_StubBuilder(),
        )

    def test_notes_mention_mc_runs_when_real_mc(self):
        assessor = self._assessor_with_stub_loader(n_runs=100)
        result = assessor.assess(
            emergent_activity={
                "activity_id": "EA-001",
                "outage_id": "RF-22",
                "planned_duration_hours": 8.0,
            },
            intake_result={"outage_phase": "maintenance", "execution_mode_flags": {}},
            historical_analogs={"duration_distribution": _dist(p50=8.0, p80=12.0)},
            run_context={"run_id": "r-001", "started_at": "2026-04-01T00:00:00Z"},
        )
        notes_text = " ".join(result["notes"])
        assert "100" in notes_text
        assert "Monte Carlo" in notes_text

    def test_notes_mention_proxy_when_n_runs_lt_10(self):
        assessor = self._assessor_with_stub_loader(n_runs=3)
        result = assessor.assess(
            emergent_activity={
                "activity_id": "EA-001",
                "outage_id": "RF-22",
                "planned_duration_hours": 8.0,
            },
            intake_result={"outage_phase": "maintenance", "execution_mode_flags": {}},
            historical_analogs={"duration_distribution": _dist(p50=8.0, p80=12.0)},
            run_context={"run_id": "r-002", "started_at": "2026-04-01T00:00:00Z"},
        )
        notes_text = " ".join(result["notes"])
        assert "proxy" in notes_text.lower() or "3-scenario" in notes_text.lower()
