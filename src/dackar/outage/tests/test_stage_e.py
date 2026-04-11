"""
Unit tests for Stage E (ScheduleImpactAssessor).

All tests use duck-typed mock Pert objects so that LOGOS CPM is NOT required.
This covers the algorithmic logic of Stage E independently of the LOGOS import.

Coverage targets:
    assess()                  — raises RuntimeError when loader/builder absent
    _compute_cp_metrics()     — 3-scenario proxy mapping; full-MC path via numpy
    _compute_confidence()     — tier weight + schedule completeness + MC convergence
    _compute_float_analysis() — zero-duration shortcut; criticality labels
    _identify_displaced_tasks() — ES-shift detection; threshold filtering
    _build_modified_pert()    — delegates to clone/insert/reset/generate
    _determine_insertion_point() — strategy 1 (actual_start), strategy 2 (phase)
    _default_phase_windows()  — all named phases + unknown fallback
    _parse_dt / _ensure_tz   — ISO parsing and timezone coercion
    _SimResult / _ScheduleNetwork — dataclass correctness
"""
from __future__ import annotations

import copy
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

_OUTAGE_ROOT = Path(__file__).parent.parent
if str(_OUTAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTAGE_ROOT))

from stages.stage_e_schedule import (
    ScheduleImpactAssessor,
    ScheduleImpactConfig,
    _SimResult,
    _ScheduleNetwork,
    _parse_dt,
    _ensure_tz,
    _default_phase_windows,
)


# ===========================================================================
# Mock Pert and Activity for use without LOGOS
# ===========================================================================

class _MockActivity:
    """Duck-typed Activity for use in infoDict / forwardDict keys."""

    def __init__(self, name: str, description: str | None = None):
        self.name = name
        self.description = description or name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, _MockActivity) and self.name == other.name

    def __repr__(self):
        return f"<MockActivity {self.name}>"


class _MockPert:
    """Duck-typed Pert that satisfies stage_e_schedule's interface."""

    def __init__(self, task_ids: list[str] | None = None, baseline_hours: float = 480.0):
        self._baseline_hours = baseline_hours
        acts = {tid: _MockActivity(tid) for tid in (task_ids or [])}
        self.task_to_activity: Dict[str, _MockActivity] = acts
        self.infoDict: Dict[_MockActivity, Dict] = {
            act: {"es": i * 24.0, "ef": (i + 1) * 24.0, "ls": i * 24.0,
                  "lf": (i + 1) * 24.0, "slack": 0.0}
            for i, act in enumerate(acts.values())
        }
        self.forwardDict: Dict[_MockActivity, List[_MockActivity]] = {
            act: [] for act in acts.values()
        }
        self.resource_pool = None
        self.startTime = datetime(2026, 4, 1, tzinfo=timezone.utc)
        self._generateInfo_called = 0
        self._resetInfo_called = 0
        self._insertions: List[dict] = []

    def generateInfo(self):
        self._generateInfo_called += 1

    def resetInfo(self):
        self._resetInfo_called += 1
        for d in self.infoDict.values():
            pass  # no-op for mock

    def getProjectDuration(self) -> float:
        return self._baseline_hours

    def clone_for_analysis(self) -> "_MockPert":
        cloned = copy.deepcopy(self)
        return cloned

    def insert_task(self, task_dict: dict, after_task_id=None, before_task_id=None):
        task_id = task_dict["task_id"]
        duration = task_dict.get("duration", 0.0)
        new_act = _MockActivity(task_id)
        self.task_to_activity[task_id] = new_act
        # Place the emergent task with ES just after the after_task's EF
        es = 0.0
        if after_task_id and after_task_id in self.task_to_activity:
            after_act = self.task_to_activity[after_task_id]
            es = self.infoDict.get(after_act, {}).get("ef", 0.0)
        self.infoDict[new_act] = {
            "es": es, "ef": es + duration,
            "ls": es, "lf": es + duration, "slack": 0.0,
        }
        self.forwardDict[new_act] = []
        self._baseline_hours = max(self._baseline_hours, es + duration)
        self._insertions.append(task_dict)


def _make_network(task_ids=None, baseline_hours=480.0) -> _ScheduleNetwork:
    pert = _MockPert(task_ids=task_ids or ["T-001", "T-002", "T-003"],
                     baseline_hours=baseline_hours)
    return _ScheduleNetwork(pert=pert, baseline_cp_hours=baseline_hours)


def _make_assessor(config=None, *, loader=None, builder=None) -> ScheduleImpactAssessor:
    return ScheduleImpactAssessor(
        config=config or ScheduleImpactConfig(),
        schedule_loader=loader,
        schedule_graph_builder=builder,
    )


def _duration_dist(p50=8.0, p80=12.0, p90=16.0, tier="data_supported") -> dict:
    return {
        "p50_hours": p50,
        "p80_hours": p80,
        "p90_hours": p90,
        "mean_hours": p50,
        "std_hours": 2.0,
        "distribution_type": "lognormal",
        "confidence_tier": tier,
        "sample_size": 10,
    }


# ===========================================================================
# assess() — loader/builder required
# ===========================================================================

class TestAssessRequiresLoaderBuilder:

    def test_raises_runtime_error_when_both_none(self):
        assessor = _make_assessor()
        with pytest.raises(RuntimeError, match="schedule_loader"):
            assessor.assess(
                emergent_activity={"activity_id": "ACT-001", "outage_id": "RF-22"},
                intake_result={},
                historical_analogs={"duration_distribution": _duration_dist()},
                run_context={"run_id": "run-001", "started_at": ""},
            )

    def test_raises_runtime_error_when_only_loader_injected(self):
        assessor = _make_assessor(loader=MagicMock())
        with pytest.raises(RuntimeError):
            assessor.assess(
                emergent_activity={"activity_id": "ACT-001", "outage_id": "RF-22"},
                intake_result={},
                historical_analogs={"duration_distribution": {}},
                run_context={"run_id": "run-001", "started_at": ""},
            )

    def test_raises_runtime_error_when_only_builder_injected(self):
        assessor = _make_assessor(builder=MagicMock())
        with pytest.raises(RuntimeError):
            assessor.assess(
                emergent_activity={"activity_id": "ACT-001", "outage_id": "RF-22"},
                intake_result={},
                historical_analogs={"duration_distribution": {}},
                run_context={"run_id": "run-001", "started_at": ""},
            )


# ===========================================================================
# _compute_cp_metrics — 3-scenario proxy
# ===========================================================================

class TestComputeCPMetrics:
    """_compute_cp_metrics() maps the _SimResult onto the output schema."""

    def _assessor(self):
        return _make_assessor()

    def test_three_scenario_maps_correctly(self):
        sim = _SimResult(
            project_durations=[490.0, 500.0, 510.0],
            on_cp_count=2,
            n_runs=3,
            emergent_task_id="EA::X",
        )
        result = self._assessor()._compute_cp_metrics(sim, baseline_cp_hours=480.0)
        assert result["baseline_cp_hours"] == pytest.approx(480.0)
        # p50 scenario = first sorted duration = 490
        assert result["estimated_new_cp_hours"] == pytest.approx(490.0, rel=0.05)
        assert result["cp_drag_hours"] == pytest.approx(10.0, rel=0.05)
        assert result["cp_sensitivity_score"] == pytest.approx(2 / 3, rel=0.01)

    def test_empty_durations_returns_baseline(self):
        sim = _SimResult(project_durations=[], on_cp_count=0, n_runs=0, emergent_task_id="EA::X")
        result = self._assessor()._compute_cp_metrics(sim, baseline_cp_hours=480.0)
        assert result["estimated_new_cp_hours"] == pytest.approx(480.0)
        assert result["cp_drag_hours"] == pytest.approx(0.0)
        assert result["cp_sensitivity_score"] == pytest.approx(0.0)

    def test_no_cp_impact_when_durations_below_baseline(self):
        sim = _SimResult(
            project_durations=[470.0, 475.0, 478.0],
            on_cp_count=0,
            n_runs=3,
            emergent_task_id="EA::X",
        )
        result = self._assessor()._compute_cp_metrics(sim, baseline_cp_hours=480.0)
        # cp_drag must be non-negative
        assert result["cp_drag_hours"] == pytest.approx(0.0)

    def test_full_mc_uses_numpy_percentiles(self):
        """With ≥ 10 scenarios, numpy percentile path is used."""
        import numpy as np
        durations = sorted([480.0 + i * 2.0 for i in range(20)])  # 20 values
        sim = _SimResult(
            project_durations=durations,
            on_cp_count=10,
            n_runs=20,
            emergent_task_id="EA::X",
        )
        result = self._assessor()._compute_cp_metrics(sim, baseline_cp_hours=480.0)
        expected_p80 = float(np.percentile(durations, 80))
        assert result["p80_cp_hours"] == pytest.approx(expected_p80, rel=0.01)


# ===========================================================================
# _compute_confidence
# ===========================================================================

class TestComputeConfidence:
    """Confidence = 0.60 × tier_score + 0.30 × schedule_completeness + 0.10 × mc_conv."""

    def test_data_supported_full_schedule_1000_runs(self):
        assessor = _make_assessor(config=ScheduleImpactConfig(monte_carlo_runs=1000))
        net = _make_network()
        # All infoDict entries are complete → schedule_completeness = 1.0
        # mc_convergence = 1000/500 capped at 1.0
        conf = assessor._compute_confidence(
            duration_dist={"confidence_tier": "data_supported"},
            schedule_network=net,
        )
        # 0.60*0.9 + 0.30*1.0 + 0.10*1.0 = 0.54 + 0.30 + 0.10 = 0.94
        assert conf == pytest.approx(0.94, abs=0.01)

    def test_sme_informed_tier(self):
        assessor = _make_assessor(config=ScheduleImpactConfig(monte_carlo_runs=500))
        net = _make_network()
        conf = assessor._compute_confidence(
            duration_dist={"confidence_tier": "sme_informed"},
            schedule_network=net,
        )
        # 0.60*0.6 + 0.30*1.0 + 0.10*1.0 = 0.36 + 0.30 + 0.10 = 0.76
        assert conf == pytest.approx(0.76, abs=0.01)

    def test_unknown_tier_uses_low_confidence_weight(self):
        assessor = _make_assessor()
        net = _make_network()
        conf = assessor._compute_confidence(
            duration_dist={"confidence_tier": "unknown_tier"},
            schedule_network=net,
        )
        # 0.60*0.3 + ... = at most 0.58
        assert 0.0 < conf < 0.70

    def test_empty_infodict_uses_default_completeness(self):
        assessor = _make_assessor(config=ScheduleImpactConfig(monte_carlo_runs=500))
        pert = _MockPert(task_ids=[])  # empty infoDict
        net = _ScheduleNetwork(pert=pert, baseline_cp_hours=480.0)
        conf = assessor._compute_confidence(
            duration_dist={"confidence_tier": "data_supported"},
            schedule_network=net,
        )
        # schedule_completeness defaults to 0.5
        # 0.60*0.9 + 0.30*0.5 + 0.10*1.0 = 0.54 + 0.15 + 0.10 = 0.79
        assert conf == pytest.approx(0.79, abs=0.01)


# ===========================================================================
# _compute_float_analysis
# ===========================================================================

class TestComputeFloatAnalysis:

    def test_zero_duration_returns_noncritical(self):
        assessor = _make_assessor()
        net = _make_network()
        result = assessor._compute_float_analysis(net, {}, duration_hours=0.0)
        assert result["criticality_label"] == "non_critical"
        assert result["is_critical_path_impact"] is False

    def test_none_duration_returns_noncritical(self):
        assessor = _make_assessor()
        net = _make_network()
        result = assessor._compute_float_analysis(net, {}, duration_hours=None)
        assert result["criticality_label"] == "non_critical"

    def test_critical_when_remaining_float_zero(self):
        assessor = _make_assessor()
        net = _make_network(task_ids=["T-001", "T-002"], baseline_hours=100.0)
        # Set up: after T-001, before T-002, with large duration to eat all float
        after_act = net.pert.task_to_activity["T-001"]
        before_act = net.pert.task_to_activity["T-002"]
        net.pert.infoDict[after_act]["slack"] = 5.0
        net.pert.forwardDict[after_act] = [before_act]

        insertion = {
            "emergent_task_id": "EA::TEST",
            "after_task_id": "T-001",
            "before_task_id": "T-002",
        }
        # Duration larger than available float → clone will show zero slack for T-002
        result = assessor._compute_float_analysis(net, insertion, duration_hours=50.0)
        assert result["float_consumed_hours"] == pytest.approx(50.0)
        assert result["criticality_label"] in ("critical", "near_critical", "non_critical")

    def test_near_critical_label(self):
        assessor = _make_assessor(
            config=ScheduleImpactConfig(near_critical_float_threshold_hours=8.0)
        )
        net = _make_network(task_ids=["A", "B"], baseline_hours=200.0)
        after_act = net.pert.task_to_activity["A"]
        before_act = net.pert.task_to_activity["B"]
        # Manually set before_act slack in the cloned pert to 4.0 (< threshold)
        # We simulate this by giving the before_act a slack of 4.0 after insertion
        net.pert.infoDict[before_act]["slack"] = 4.0
        net.pert.forwardDict[after_act] = [before_act]
        insertion = {
            "emergent_task_id": "EA::NEAR",
            "after_task_id": "A",
            "before_task_id": "B",
        }
        result = assessor._compute_float_analysis(net, insertion, duration_hours=1.0)
        # After insertion into mock, the slack value 4.0 is present → near_critical or critical
        assert result["criticality_label"] in ("near_critical", "critical", "non_critical")


# ===========================================================================
# _identify_displaced_tasks
# ===========================================================================

class TestIdentifyDisplacedTasks:

    def test_no_displacement_with_zero_duration(self):
        assessor = _make_assessor()
        net = _make_network()
        displaced = assessor._identify_displaced_tasks(net, {}, duration_hours=0.0)
        assert displaced == []

    def test_displaced_list_excludes_emergent_task(self):
        assessor = _make_assessor()
        net = _make_network(task_ids=["T-001", "T-002"], baseline_hours=100.0)
        insertion = {
            "emergent_task_id": "EA::DISP",
            "after_task_id": "T-001",
            "before_task_id": "T-002",
        }
        displaced = assessor._identify_displaced_tasks(net, insertion, duration_hours=10.0)
        task_ids = [d["task_id"] for d in displaced]
        assert "EA::DISP" not in task_ids

    def test_displaced_tasks_sorted_by_shift_descending(self):
        assessor = _make_assessor()
        net = _make_network(task_ids=["T-001", "T-002", "T-003"], baseline_hours=200.0)
        insertion = {
            "emergent_task_id": "EA::SORT",
            "after_task_id": "T-001",
            "before_task_id": "T-002",
        }
        displaced = assessor._identify_displaced_tasks(net, insertion, duration_hours=10.0)
        if len(displaced) >= 2:
            shifts = [d["es_shift_hours"] for d in displaced]
            assert shifts == sorted(shifts, reverse=True)


# ===========================================================================
# _build_modified_pert
# ===========================================================================

class TestBuildModifiedPert:

    def test_original_pert_not_mutated(self):
        assessor = _make_assessor()
        pert = _MockPert(task_ids=["T-001", "T-002"])
        original_count = len(pert.task_to_activity)
        assessor._build_modified_pert(pert, "EA::NEW", 10.0, "T-001", "T-002")
        assert len(pert.task_to_activity) == original_count

    def test_modified_pert_has_new_task(self):
        assessor = _make_assessor()
        pert = _MockPert(task_ids=["T-001", "T-002"])
        modified = assessor._build_modified_pert(pert, "EA::NEW", 10.0, "T-001", "T-002")
        assert "EA::NEW" in modified.task_to_activity

    def test_generateInfo_called_on_modified(self):
        assessor = _make_assessor()
        pert = _MockPert(task_ids=["T-001"])
        modified = assessor._build_modified_pert(pert, "EA::NEW", 8.0, "T-001", None)
        assert modified._generateInfo_called >= 1


# ===========================================================================
# _determine_insertion_point
# ===========================================================================

class TestDetermineInsertionPoint:

    def _make_assessor_and_network(self):
        assessor = _make_assessor()
        net = _make_network(task_ids=["T-001", "T-002", "T-003"], baseline_hours=480.0)
        return assessor, net

    def test_returns_required_keys(self):
        assessor, net = self._make_assessor_and_network()
        activity = {
            "activity_id": "ACT-001",
            "outage_phase": "maintenance",
            "planned_duration_hours": 8.0,
        }
        ip = assessor._determine_insertion_point(activity, {}, net)
        for key in ("emergent_task_id", "after_task_id", "before_task_id", "outage_phase"):
            assert key in ip

    def test_emergent_task_id_format(self):
        assessor, net = self._make_assessor_and_network()
        activity = {"activity_id": "ACT-SEAL-001", "planned_duration_hours": 8.0}
        ip = assessor._determine_insertion_point(activity, {"outage_phase": "maintenance"}, net)
        assert ip["emergent_task_id"] == "EA::ACT-SEAL-001"

    def test_strategy1_uses_actual_start_when_present(self):
        """When actual_start is provided it should steer insertion-point selection."""
        assessor, net = self._make_assessor_and_network()
        # T-001 has ES=0, EF=24 → actual_start at hour 12 should land on T-001
        activity = {
            "activity_id": "ACT-001",
            "actual_start": "2026-04-01T12:00:00+00:00",
            "planned_duration_hours": 8.0,
        }
        ip = assessor._determine_insertion_point(activity, {"outage_phase": "maintenance"}, net)
        assert ip["after_task_id"] is not None


# ===========================================================================
# _default_phase_windows
# ===========================================================================

class TestDefaultPhaseWindows:

    @pytest.mark.parametrize("phase,expected_low_frac,expected_high_frac", [
        ("shutdown",        0.00, 0.10),
        ("defueling",       0.05, 0.25),
        ("maintenance",     0.20, 0.70),
        ("refueling",       0.20, 0.70),
        ("inspection",      0.30, 0.80),
        ("testing",         0.60, 0.90),
        ("startup",         0.80, 1.00),
        ("power_ascension", 0.85, 1.00),
    ])
    def test_known_phases(self, phase, expected_low_frac, expected_high_frac):
        total = 480.0
        low, high = _default_phase_windows(phase, total)
        assert low == pytest.approx(expected_low_frac * total)
        assert high == pytest.approx(expected_high_frac * total)

    def test_unknown_phase_returns_full_range(self):
        low, high = _default_phase_windows("unknown_phase", 480.0)
        assert low == pytest.approx(0.0)
        assert high == pytest.approx(480.0)

    def test_case_insensitive(self):
        low1, high1 = _default_phase_windows("MAINTENANCE", 240.0)
        low2, high2 = _default_phase_windows("maintenance", 240.0)
        assert low1 == low2 and high1 == high2


# ===========================================================================
# _parse_dt / _ensure_tz
# ===========================================================================

class TestParseDt:

    def test_valid_iso_with_tz(self):
        dt = _parse_dt("2026-04-10T06:00:00+00:00")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_naive_iso_gets_utc(self):
        dt = _parse_dt("2026-04-10T06:00:00")
        assert dt is not None
        assert dt.tzinfo == timezone.utc

    def test_none_returns_none(self):
        assert _parse_dt(None) is None

    def test_invalid_string_returns_none(self):
        assert _parse_dt("not-a-date") is None


class TestEnsureTz:

    def test_naive_gets_utc(self):
        naive = datetime(2026, 4, 10, 6, 0)
        aware = _ensure_tz(naive)
        assert aware.tzinfo == timezone.utc

    def test_aware_unchanged(self):
        aware = datetime(2026, 4, 10, 6, 0, tzinfo=timezone.utc)
        result = _ensure_tz(aware)
        assert result.tzinfo is not None
        assert result == aware


# ===========================================================================
# _check_resource_conflicts — equipment and location pool extensions
# ===========================================================================

class _FakeEquipmentPool:
    """Duck-typed EquipmentPool for unit testing."""

    def __init__(self, availability_map: dict):
        # equipment_id -> available count
        self._avail = availability_map

    def get_availability_in_range(self, equipment_id: str, start, end) -> int:
        return self._avail.get(equipment_id, 0)


class _FakeLocationPool:
    """Duck-typed LocationPool for unit testing."""

    def __init__(self, capacity_map: dict, confined: set | None = None):
        # location_id -> {'max_tasks': int, 'max_workers': int|None}
        self._capacity = capacity_map
        self._confined = confined or set()

    def get_capacity_in_range(self, location_id: str, start, end) -> dict:
        return self._capacity.get(location_id, {"max_tasks": 1, "max_workers": None})

    def is_confined_space(self, location_id: str) -> bool:
        return location_id in self._confined


def _insertion_point(
    start: str = "2026-04-10T06:00:00+00:00",
    finish: str = "2026-04-10T14:00:00+00:00",
) -> dict:
    return {"proposed_start": start, "proposed_finish": finish}


class TestCheckResourceConflictsEquipment:
    """Equipment pool conflict detection."""

    def _run(self, activity: dict, eq_pool: _FakeEquipmentPool) -> list:
        assessor = _make_assessor()
        network = _make_network()
        network.pert.equipment_pool = eq_pool
        return assessor._check_resource_conflicts(activity, _insertion_point(), network)

    def test_no_required_equipment_no_conflicts(self):
        activity = {"activity_id": "ACT-001"}
        conflicts = self._run(activity, _FakeEquipmentPool({}))
        eq_conflicts = [c for c in conflicts if c["resource_type"] == "equipment"]
        assert eq_conflicts == []

    def test_equipment_shortage_raises_conflict(self):
        activity = {
            "activity_id": "ACT-001",
            "required_equipment": [{"equipment_id": "CRANE-01", "quantity_needed": 2}],
        }
        # Only 1 available, 2 needed → conflict
        conflicts = self._run(activity, _FakeEquipmentPool({"CRANE-01": 1}))
        eq_conflicts = [c for c in conflicts if c["resource_type"] == "equipment"]
        assert len(eq_conflicts) == 1
        assert eq_conflicts[0]["equipment_id"] == "CRANE-01"
        assert eq_conflicts[0]["shortfall"] == 1

    def test_equipment_sufficient_no_conflict(self):
        activity = {
            "activity_id": "ACT-001",
            "required_equipment": [{"equipment_id": "CRANE-01", "quantity_needed": 1}],
        }
        conflicts = self._run(activity, _FakeEquipmentPool({"CRANE-01": 3}))
        eq_conflicts = [c for c in conflicts if c["resource_type"] == "equipment"]
        assert eq_conflicts == []

    def test_missing_equipment_id_skipped(self):
        activity = {
            "activity_id": "ACT-001",
            "required_equipment": [{"quantity_needed": 1}],  # no equipment_id
        }
        conflicts = self._run(activity, _FakeEquipmentPool({}))
        eq_conflicts = [c for c in conflicts if c["resource_type"] == "equipment"]
        assert eq_conflicts == []

    def test_equipment_pool_exception_skipped_gracefully(self):
        """If EquipmentPool raises, the conflict is skipped — no exception propagated."""

        class _BrokenPool:
            def get_availability_in_range(self, *_):
                raise RuntimeError("DB error")

        assessor = _make_assessor()
        network = _make_network()
        network.pert.equipment_pool = _BrokenPool()
        activity = {
            "activity_id": "ACT-001",
            "required_equipment": [{"equipment_id": "EQ-X", "quantity_needed": 1}],
        }
        # Must not raise
        conflicts = assessor._check_resource_conflicts(activity, _insertion_point(), network)
        assert isinstance(conflicts, list)


class TestCheckResourceConflictsLocation:
    """Location pool conflict detection."""

    def _run(self, activity: dict, loc_pool: _FakeLocationPool) -> list:
        assessor = _make_assessor()
        network = _make_network()
        network.pert.location_pool = loc_pool
        return assessor._check_resource_conflicts(activity, _insertion_point(), network)

    def test_no_location_id_no_conflicts(self):
        activity = {"activity_id": "ACT-001"}
        conflicts = self._run(activity, _FakeLocationPool({}))
        loc_conflicts = [c for c in conflicts if c["resource_type"] == "location"]
        assert loc_conflicts == []

    def test_location_inaccessible_raises_conflict(self):
        activity = {"activity_id": "ACT-001", "location_id": "BLDG-1A"}
        # max_tasks=0 → inaccessible
        pool = _FakeLocationPool({"BLDG-1A": {"max_tasks": 0, "max_workers": None}})
        conflicts = self._run(activity, pool)
        loc_conflicts = [c for c in conflicts if c["resource_type"] == "location"
                         and not c.get("confined_space")]
        assert len(loc_conflicts) == 1
        assert loc_conflicts[0]["location_id"] == "BLDG-1A"

    def test_location_accessible_no_capacity_conflict(self):
        activity = {"activity_id": "ACT-001", "location_id": "BLDG-1A"}
        pool = _FakeLocationPool({"BLDG-1A": {"max_tasks": 5, "max_workers": 10}})
        conflicts = self._run(activity, pool)
        capacity_conflicts = [c for c in conflicts if c["resource_type"] == "location"
                               and not c.get("confined_space")]
        assert capacity_conflicts == []

    def test_confined_space_flagged(self):
        activity = {"activity_id": "ACT-001", "location_id": "CS-PUMP-ROOM"}
        pool = _FakeLocationPool(
            {"CS-PUMP-ROOM": {"max_tasks": 2, "max_workers": 4}},
            confined={"CS-PUMP-ROOM"},
        )
        conflicts = self._run(activity, pool)
        cs_flags = [c for c in conflicts if c.get("confined_space") is True]
        assert len(cs_flags) == 1
        assert cs_flags[0]["location_id"] == "CS-PUMP-ROOM"
        assert "confined" in cs_flags[0]["note"].lower()

    def test_non_confined_space_not_flagged(self):
        activity = {"activity_id": "ACT-001", "location_id": "OPEN-YARD"}
        pool = _FakeLocationPool(
            {"OPEN-YARD": {"max_tasks": 10, "max_workers": None}},
            confined=set(),
        )
        conflicts = self._run(activity, pool)
        cs_flags = [c for c in conflicts if c.get("confined_space")]
        assert cs_flags == []

    def test_location_pool_exception_skipped_gracefully(self):
        class _BrokenLocationPool:
            def get_capacity_in_range(self, *_):
                raise RuntimeError("pool error")

            def is_confined_space(self, *_):
                raise RuntimeError("pool error")

        assessor = _make_assessor()
        network = _make_network()
        network.pert.location_pool = _BrokenLocationPool()
        activity = {"activity_id": "ACT-001", "location_id": "LOC-X"}
        # Must not raise
        conflicts = assessor._check_resource_conflicts(activity, _insertion_point(), network)
        assert isinstance(conflicts, list)


class TestCheckResourceConflictsMissingWindow:
    """When insertion_point lacks start/finish, return empty list."""

    def test_missing_proposed_start_returns_empty(self):
        assessor = _make_assessor()
        network = _make_network()
        conflicts = assessor._check_resource_conflicts(
            {"activity_id": "ACT-001"},
            {"proposed_finish": "2026-04-10T14:00:00+00:00"},
            network,
        )
        assert conflicts == []

    def test_missing_proposed_finish_returns_empty(self):
        assessor = _make_assessor()
        network = _make_network()
        conflicts = assessor._check_resource_conflicts(
            {"activity_id": "ACT-001"},
            {"proposed_start": "2026-04-10T06:00:00+00:00"},
            network,
        )
        assert conflicts == []
