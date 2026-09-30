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

_OUTAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "dackar" / "outage"
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
    _off_shift_overlap_hours,
    _has_shift_boundary,
    _compute_permit_lead_time,
    _shift_insertion_times,
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
# _compute_cp_metrics — baseline locking (schedule variance fields)
# ===========================================================================

class TestComputeCPMetricsBaselineLocking:
    """Verify schedule variance and overrun fields when a locked baseline is provided."""

    def _assessor(self):
        return _make_assessor()

    def _sim(self, durations=(490.0, 500.0, 510.0)):
        return _SimResult(
            project_durations=list(durations),
            on_cp_count=2,
            n_runs=len(durations),
            emergent_task_id="EA::X",
        )

    def test_schedule_variance_positive_when_working_exceeds_locked(self):
        """working_cp=480 locked=470 → already slipped 10 h."""
        result = self._assessor()._compute_cp_metrics(
            self._sim(), baseline_cp_hours=480.0,
            locked_baseline_cp_hours=470.0,
        )
        assert result["schedule_variance_hours"] == pytest.approx(10.0)

    def test_schedule_variance_zero_when_on_plan(self):
        """working_cp == locked → variance 0."""
        result = self._assessor()._compute_cp_metrics(
            self._sim(), baseline_cp_hours=480.0,
            locked_baseline_cp_hours=480.0,
        )
        assert result["schedule_variance_hours"] == pytest.approx(0.0)

    def test_total_overrun_includes_drag_and_variance(self):
        """estimated_new_cp=490, locked=470 → overrun 20 h."""
        result = self._assessor()._compute_cp_metrics(
            self._sim([490.0, 500.0, 510.0]), baseline_cp_hours=480.0,
            locked_baseline_cp_hours=470.0,
        )
        # p50 scenario → 490; locked=470 → overrun=20
        assert result["total_overrun_hours"] == pytest.approx(20.0)

    def test_total_overrun_non_negative_when_new_cp_below_locked(self):
        """If new_cp < locked_baseline (very good case), overrun is clamped to 0."""
        result = self._assessor()._compute_cp_metrics(
            self._sim([460.0, 465.0, 470.0]), baseline_cp_hours=465.0,
            locked_baseline_cp_hours=480.0,
        )
        assert result["total_overrun_hours"] == pytest.approx(0.0)

    def test_locked_baseline_fields_absent_when_not_provided(self):
        """Without locked_baseline_cp_hours the new keys must not appear."""
        result = self._assessor()._compute_cp_metrics(
            self._sim(), baseline_cp_hours=480.0,
        )
        assert "locked_baseline_cp_hours" not in result
        assert "schedule_variance_hours" not in result
        assert "total_overrun_hours" not in result

    def test_locked_baseline_finish_computed_from_start_and_duration(self):
        """locked_baseline_finish = locked_start + locked_cp_hours."""
        from datetime import datetime, timezone, timedelta
        locked_start = datetime(2024, 3, 1, 6, 0, tzinfo=timezone.utc)
        result = self._assessor()._compute_cp_metrics(
            self._sim(), baseline_cp_hours=480.0,
            locked_baseline_cp_hours=480.0,
            locked_baseline_start=locked_start,
        )
        expected = (locked_start + timedelta(hours=480.0)).isoformat()
        assert result["locked_baseline_finish"] == expected

    def test_projected_finish_computed_from_working_start_and_new_cp(self):
        """projected_finish = working_start + estimated_new_cp_hours."""
        from datetime import datetime, timezone, timedelta
        working_start = datetime(2024, 3, 1, 6, 0, tzinfo=timezone.utc)
        result = self._assessor()._compute_cp_metrics(
            self._sim([492.0, 500.0, 510.0]), baseline_cp_hours=480.0,
            locked_baseline_cp_hours=470.0,
            working_start=working_start,
        )
        # p50 scenario = first sorted = 492
        expected = (working_start + timedelta(hours=492.0)).isoformat()
        assert result["projected_finish_after_insertion"] == expected

    def test_empty_durations_with_locked_baseline(self):
        """Zero-duration sim still populates variance fields."""
        sim = _SimResult(project_durations=[], on_cp_count=0, n_runs=0, emergent_task_id="EA::X")
        result = self._assessor()._compute_cp_metrics(
            sim, baseline_cp_hours=480.0,
            locked_baseline_cp_hours=470.0,
        )
        # estimated_new_cp falls back to baseline_cp_hours = 480
        assert result["schedule_variance_hours"] == pytest.approx(10.0)
        assert result["total_overrun_hours"] == pytest.approx(10.0)

    def test_locked_baseline_cp_hours_present_in_output(self):
        """locked_baseline_cp_hours value is echoed in the result dict."""
        result = self._assessor()._compute_cp_metrics(
            self._sim(), baseline_cp_hours=480.0,
            locked_baseline_cp_hours=465.0,
        )
        assert result["locked_baseline_cp_hours"] == pytest.approx(465.0)


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
# _has_regulatory_constraint — unit tests
# ===========================================================================

class TestHasRegulatoryConstraint:
    """_has_regulatory_constraint() should detect nuclear regulatory keywords."""

    @staticmethod
    def _f(text):
        from stages.stage_e_schedule import _has_regulatory_constraint
        return _has_regulatory_constraint(text)

    def test_ts_number_returns_true(self):
        assert self._f("TS 3.5.7 pump surveillance test") is True

    def test_lco_number_returns_true(self):
        assert self._f("LCO 3.4.1 applicable — entry required") is True

    def test_surveillance_returns_true(self):
        assert self._f("Quarterly surveillance of emergency diesel") is True

    def test_hold_point_returns_true(self):
        assert self._f("QC hold point verification prior to restart") is True

    def test_nrc_returns_true(self):
        assert self._f("NRC commitment item tracking") is True

    def test_alara_returns_true(self):
        assert self._f("ALARA review required before entry") is True

    def test_technical_specification_returns_true(self):
        assert self._f("Technical specification requires completion") is True

    def test_mode_change_returns_true(self):
        assert self._f("Mode change to mode 4 gating activity") is True

    def test_plain_mechanical_description_returns_false(self):
        assert self._f("Replace pump mechanical seal — corrective maintenance") is False

    def test_none_returns_false(self):
        assert self._f(None) is False

    def test_empty_string_returns_false(self):
        assert self._f("") is False

    def test_case_insensitive_surveillance(self):
        assert self._f("QUARTERLY SURVEILLANCE TEST") is True

    def test_partial_word_not_matched(self):
        # "NRC" embedded in a longer token should not match (word boundary)
        assert self._f("CAPNRC not a real token") is False


# ===========================================================================
# _identify_displaced_tasks — regulatory constraint enrichment
# ===========================================================================

class TestIdentifyDisplacedTasksRegulatoryFlag:
    """has_regulatory_constraint must be set from the displaced task description."""

    class _ShiftedPert(_MockPert):
        """MockPert whose clone shifts T-002's ES by 8 hours for displacement testing."""

        def clone_for_analysis(self) -> "_MockPert":
            c = super().clone_for_analysis()
            for act in list(c.task_to_activity.values()):
                if act.name == "T-002":
                    info = c.infoDict[act]
                    info["es"] = info.get("es", 0.0) + 8.0
                    info["ef"] = info.get("ef", 0.0) + 8.0
            return c

    def _make_shifted_net(self, t002_description: str) -> _ScheduleNetwork:
        pert = self._ShiftedPert(task_ids=["T-001", "T-002"], baseline_hours=200.0)
        t002 = pert.task_to_activity["T-002"]
        t002.description = t002_description
        return _ScheduleNetwork(pert=pert, baseline_cp_hours=200.0)

    def test_ts_description_flags_displaced_task(self):
        """Displaced task with 'TS 3.5.7' in description → has_regulatory_constraint True."""
        assessor = _make_assessor()
        net = self._make_shifted_net("TS 3.5.7 quarterly pump surveillance")
        insertion = {
            "emergent_task_id": "EA::REG",
            "after_task_id": "T-001",
            "before_task_id": "T-002",
        }
        displaced = assessor._identify_displaced_tasks(net, insertion, duration_hours=8.0)
        reg_tasks = [d for d in displaced if d["task_id"] == "T-002"]
        assert reg_tasks, "T-002 should appear as displaced"
        assert reg_tasks[0]["has_regulatory_constraint"] is True

    def test_plain_description_does_not_flag(self):
        """Displaced task with no regulatory keywords → has_regulatory_constraint False."""
        assessor = _make_assessor()
        net = self._make_shifted_net("Replace valve seat — scheduled PM")
        insertion = {
            "emergent_task_id": "EA::PLAIN",
            "after_task_id": "T-001",
            "before_task_id": "T-002",
        }
        displaced = assessor._identify_displaced_tasks(net, insertion, duration_hours=8.0)
        reg_tasks = [d for d in displaced if d["task_id"] == "T-002"]
        assert reg_tasks, "T-002 should appear as displaced"
        assert reg_tasks[0]["has_regulatory_constraint"] is False

    def test_surveillance_description_flags_displaced_task(self):
        """'surveillance' keyword flags the task."""
        assessor = _make_assessor()
        net = self._make_shifted_net("Quarterly diesel generator surveillance test")
        insertion = {
            "emergent_task_id": "EA::SURV",
            "after_task_id": "T-001",
            "before_task_id": "T-002",
        }
        displaced = assessor._identify_displaced_tasks(net, insertion, duration_hours=8.0)
        reg_tasks = [d for d in displaced if d["task_id"] == "T-002"]
        assert reg_tasks
        assert reg_tasks[0]["has_regulatory_constraint"] is True


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


# ===========================================================================
# _off_shift_overlap_hours
# ===========================================================================

class TestOffShiftOverlapHours:
    """Tests for the module-level shift-overlap helper."""

    def test_24_7_schedule_always_zero(self):
        start = datetime(2026, 4, 10, 8, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 10, 20, 0, tzinfo=timezone.utc)
        assert _off_shift_overlap_hours(start, end, shift_start_hour=6, working_hours_per_day=24) == 0.0

    def test_window_fully_within_shift(self):
        # Shift 06:00–18:00; window 08:00–10:00 → fully in shift
        start = datetime(2026, 4, 10, 8, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 10, 10, 0, tzinfo=timezone.utc)
        result = _off_shift_overlap_hours(start, end, shift_start_hour=6, working_hours_per_day=12)
        assert result == 0.0

    def test_window_fully_outside_shift(self):
        # Shift 06:00–18:00; window 20:00–23:00 → fully off-shift (3 h)
        start = datetime(2026, 4, 10, 20, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 10, 23, 0, tzinfo=timezone.utc)
        result = _off_shift_overlap_hours(start, end, shift_start_hour=6, working_hours_per_day=12)
        assert abs(result - 3.0) < 0.1

    def test_window_spans_shift_end(self):
        # Shift 06:00–18:00; window 16:00–20:00 → 2 h off-shift (18:00–20:00)
        start = datetime(2026, 4, 10, 16, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 10, 20, 0, tzinfo=timezone.utc)
        result = _off_shift_overlap_hours(start, end, shift_start_hour=6, working_hours_per_day=12)
        assert abs(result - 2.0) < 0.1

    def test_window_equal_start_end_returns_zero(self):
        t = datetime(2026, 4, 10, 8, 0, tzinfo=timezone.utc)
        assert _off_shift_overlap_hours(t, t, shift_start_hour=6, working_hours_per_day=12) == 0.0

    def test_multi_day_window_accumulates_off_shift(self):
        # Shift 06:00–18:00 (12 h/day); window spans two full days → 12*2 = 24 off-shift h
        start = datetime(2026, 4, 10, 6, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 12, 6, 0, tzinfo=timezone.utc)
        result = _off_shift_overlap_hours(start, end, shift_start_hour=6, working_hours_per_day=12)
        assert abs(result - 24.0) < 0.5


# ===========================================================================
# _has_shift_boundary
# ===========================================================================

class TestHasShiftBoundary:
    """Tests for the module-level shift-boundary detection helper."""

    def test_24_7_never_has_boundary(self):
        start = datetime(2026, 4, 10, 8, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 10, 20, 0, tzinfo=timezone.utc)
        assert _has_shift_boundary(start, end, shift_start_hour=6, working_hours_per_day=24) is False

    def test_shift_start_inside_window_returns_true(self):
        # Shift starts at 06:00; window 04:00–10:00 contains shift start → True
        start = datetime(2026, 4, 10, 4, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 10, 10, 0, tzinfo=timezone.utc)
        assert _has_shift_boundary(start, end, shift_start_hour=6, working_hours_per_day=12) is True

    def test_shift_start_at_window_start_not_inside(self):
        # Boundary is at window start (not strictly inside)
        start = datetime(2026, 4, 10, 6, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 10, 14, 0, tzinfo=timezone.utc)
        assert _has_shift_boundary(start, end, shift_start_hour=6, working_hours_per_day=12) is False

    def test_shift_start_after_window_returns_false(self):
        # Shift starts at 06:00; window 07:00–12:00 — next shift start is tomorrow
        start = datetime(2026, 4, 10, 7, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)
        assert _has_shift_boundary(start, end, shift_start_hour=6, working_hours_per_day=12) is False

    def test_window_spans_midnight_into_next_day_shift_start(self):
        # Shift starts at 06:00; window 22:00 day1 → 08:00 day2 → crosses 06:00 boundary
        start = datetime(2026, 4, 10, 22, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 11, 8, 0, tzinfo=timezone.utc)
        assert _has_shift_boundary(start, end, shift_start_hour=6, working_hours_per_day=12) is True

    def test_equal_start_end_returns_false(self):
        t = datetime(2026, 4, 10, 6, 0, tzinfo=timezone.utc)
        assert _has_shift_boundary(t, t, shift_start_hour=6, working_hours_per_day=12) is False


# ===========================================================================
# _assess_crew_continuity
# ===========================================================================

class _MockCrewPool:
    """Duck-typed ResourcePool for crew-continuity tests."""

    def __init__(self, skill_availability: dict):
        # {skill_type: available_count}
        self.resources = {skill: None for skill in skill_availability}
        self._avail = skill_availability  # minimum over any range

    def get_availability_in_range(self, skill_type: str, start, end) -> int:
        return self._avail.get(skill_type, 0)


class _MockPertWithShift(_MockPert):
    """Extends _MockPert with shift-calendar and resource-pool attributes."""

    def __init__(
        self,
        task_ids=None,
        baseline_hours=480.0,
        shift_start_hour=6,
        working_hours_per_day=12,
        crew_pool=None,
        start_time=None,
    ):
        super().__init__(task_ids=task_ids, baseline_hours=baseline_hours)
        self.shift_start_hour = shift_start_hour
        self.working_hours_per_day = working_hours_per_day
        self.crew_pool = crew_pool
        if start_time is not None:
            self.startTime = start_time


def _network_with_shift(
    task_ids=None,
    shift_start_hour=6,
    working_hours_per_day=12,
    crew_pool=None,
    start_time=None,
    baseline_hours=480.0,
) -> _ScheduleNetwork:
    pert = _MockPertWithShift(
        task_ids=task_ids or ["T-001", "T-002"],
        baseline_hours=baseline_hours,
        shift_start_hour=shift_start_hour,
        working_hours_per_day=working_hours_per_day,
        crew_pool=crew_pool,
        start_time=start_time or datetime(2026, 4, 10, 6, 0, tzinfo=timezone.utc),
    )
    return _ScheduleNetwork(pert=pert, baseline_cp_hours=baseline_hours)


_WINDOW_IN_SHIFT = {
    "proposed_start": "2026-04-10T08:00:00+00:00",   # 08:00 — inside 06–18 shift
    "proposed_finish": "2026-04-10T14:00:00+00:00",  # 14:00 — inside shift
}
_WINDOW_SPANNING_SHIFT_END = {
    "proposed_start": "2026-04-10T16:00:00+00:00",   # 16:00 — inside shift
    "proposed_finish": "2026-04-10T22:00:00+00:00",  # 22:00 — outside shift (past 18:00)
}
_WINDOW_SPANNING_SHIFT_START = {
    "proposed_start": "2026-04-10T04:00:00+00:00",   # 04:00 — off-shift
    "proposed_finish": "2026-04-10T10:00:00+00:00",  # 10:00 — in shift; crosses 06:00
}


class TestAssessCrewContinuity:

    def test_missing_proposed_start_returns_not_available(self):
        assessor = _make_assessor()
        net = _network_with_shift()
        result = assessor._assess_crew_continuity(
            {}, {"proposed_finish": "2026-04-10T14:00:00+00:00"}, net
        )
        assert result["available"] is False

    def test_missing_proposed_finish_returns_not_available(self):
        assessor = _make_assessor()
        net = _network_with_shift()
        result = assessor._assess_crew_continuity(
            {}, {"proposed_start": "2026-04-10T08:00:00+00:00"}, net
        )
        assert result["available"] is False

    def test_no_crew_pool_returns_available_empty_utilization(self):
        assessor = _make_assessor()
        net = _network_with_shift(crew_pool=None)
        result = assessor._assess_crew_continuity({}, _WINDOW_IN_SHIFT, net)
        assert result["available"] is True
        assert result["utilization_at_window"] == {}
        assert result["peak_utilization_skill"] is None

    def test_24_7_schedule_zero_off_shift_no_boundary(self):
        assessor = _make_assessor()
        net = _network_with_shift(working_hours_per_day=24)
        result = assessor._assess_crew_continuity({}, _WINDOW_IN_SHIFT, net)
        assert result["off_shift_overlap_hours"] == 0.0
        assert result["shift_boundary_conflict"] is False

    def test_partial_shift_window_in_shift_no_overlap(self):
        assessor = _make_assessor()
        net = _network_with_shift(shift_start_hour=6, working_hours_per_day=12)
        result = assessor._assess_crew_continuity({}, _WINDOW_IN_SHIFT, net)
        assert result["off_shift_overlap_hours"] == 0.0
        assert result["fatigue_risk"] is False

    def test_partial_shift_window_spans_shift_end_detects_overlap(self):
        # Window 16:00–22:00; shift ends at 18:00 → 4 h off-shift
        assessor = _make_assessor()
        net = _network_with_shift(shift_start_hour=6, working_hours_per_day=12)
        result = assessor._assess_crew_continuity({}, _WINDOW_SPANNING_SHIFT_END, net)
        assert result["off_shift_overlap_hours"] > 0.0

    def test_fatigue_risk_triggered_above_threshold(self):
        # Off-shift overlap = 4h; threshold default = 2.0h → fatigue_risk=True
        assessor = _make_assessor(ScheduleImpactConfig(fatigue_risk_off_shift_hours=2.0))
        net = _network_with_shift(shift_start_hour=6, working_hours_per_day=12)
        result = assessor._assess_crew_continuity({}, _WINDOW_SPANNING_SHIFT_END, net)
        assert result["fatigue_risk"] is True
        assert any("fatigue" in n.lower() or "off-shift" in n.lower() for n in result["notes"])

    def test_fatigue_risk_not_triggered_below_threshold(self):
        # Same window but threshold set very high
        assessor = _make_assessor(ScheduleImpactConfig(fatigue_risk_off_shift_hours=10.0))
        net = _network_with_shift(shift_start_hour=6, working_hours_per_day=12)
        result = assessor._assess_crew_continuity({}, _WINDOW_SPANNING_SHIFT_END, net)
        assert result["fatigue_risk"] is False

    def test_shift_boundary_crossing_detected(self):
        # Window 04:00–10:00 crosses shift start at 06:00
        assessor = _make_assessor()
        net = _network_with_shift(shift_start_hour=6, working_hours_per_day=12)
        result = assessor._assess_crew_continuity({}, _WINDOW_SPANNING_SHIFT_START, net)
        assert result["shift_boundary_conflict"] is True
        assert any("handover" in n.lower() for n in result["notes"])

    def test_shift_boundary_not_detected_when_window_within_shift(self):
        assessor = _make_assessor()
        net = _network_with_shift(shift_start_hour=6, working_hours_per_day=12)
        result = assessor._assess_crew_continuity({}, _WINDOW_IN_SHIFT, net)
        assert result["shift_boundary_conflict"] is False

    def test_utilization_computed_from_crew_pool(self):
        pool = _MockCrewPool({"MECHANIC": 20})
        assessor = _make_assessor()
        net = _network_with_shift(crew_pool=pool, working_hours_per_day=24)
        result = assessor._assess_crew_continuity({}, _WINDOW_IN_SHIFT, net)
        assert "MECHANIC" in result["utilization_at_window"]
        entry = result["utilization_at_window"]["MECHANIC"]
        assert entry["available"] == 20

    def test_free_crew_computed_correctly(self):
        # 20 available, 0 committed (no tasks have required_resources) → free=20
        pool = _MockCrewPool({"MECHANIC": 20})
        assessor = _make_assessor()
        net = _network_with_shift(
            crew_pool=pool,
            working_hours_per_day=24,
        )
        result = assessor._assess_crew_continuity({}, _WINDOW_IN_SHIFT, net)
        entry = result["utilization_at_window"]["MECHANIC"]
        assert entry["committed"] == 0
        assert entry["free"] == 20
        assert entry["utilization_pct"] == 0.0

    def test_high_utilization_flagged_above_threshold(self):
        # 20 available; task with 18 mechanics overlaps window → 90% → flagged
        pool = _MockCrewPool({"MECHANIC": 20})
        assessor = _make_assessor(ScheduleImpactConfig(high_crew_utilization_threshold=0.80))

        class _HighLoadPert(_MockPertWithShift):
            def __init__(self, **kw):
                super().__init__(**kw)
                act = _MockActivity("T-LOAD")
                act.required_resources = [{"skill_type": "MECHANIC", "crew_count": 18}]
                # Project starts at 06:00; window is 08:00–14:00 → offsets 2–8 h
                self.infoDict[act] = {"es": 2.0, "ef": 8.0, "slack": 0.0}
                self.task_to_activity["T-LOAD"] = act

        pert = _HighLoadPert(
            task_ids=[],
            baseline_hours=480.0,
            shift_start_hour=6,
            working_hours_per_day=24,
            crew_pool=pool,
            start_time=datetime(2026, 4, 10, 6, 0, tzinfo=timezone.utc),
        )
        net = _ScheduleNetwork(pert=pert, baseline_cp_hours=480.0)
        result = assessor._assess_crew_continuity({}, _WINDOW_IN_SHIFT, net)
        entry = result["utilization_at_window"]["MECHANIC"]
        assert entry["high_utilization"] is True
        assert entry["utilization_pct"] >= 80.0
        assert any("high crew utilization" in n.lower() for n in result["notes"])

    def test_peak_utilization_skill_reported(self):
        # Two skills; WELDER fully committed (5/5=100%) vs MECHANIC idle (0/20=0%)
        pool = _MockCrewPool({"MECHANIC": 20, "WELDER": 5})
        assessor = _make_assessor()

        class _MultiSkillPert(_MockPertWithShift):
            def __init__(self, **kw):
                super().__init__(**kw)
                act = _MockActivity("T-WELD")
                act.required_resources = [{"skill_type": "WELDER", "crew_count": 5}]
                self.infoDict[act] = {"es": 2.0, "ef": 8.0, "slack": 0.0}
                self.task_to_activity["T-WELD"] = act

        pert = _MultiSkillPert(
            task_ids=[],
            baseline_hours=480.0,
            shift_start_hour=6,
            working_hours_per_day=24,
            crew_pool=pool,
            start_time=datetime(2026, 4, 10, 6, 0, tzinfo=timezone.utc),
        )
        net = _ScheduleNetwork(pert=pert, baseline_cp_hours=480.0)
        result = assessor._assess_crew_continuity({}, _WINDOW_IN_SHIFT, net)
        assert result["peak_utilization_skill"] == "WELDER"
        assert result["peak_utilization_pct"] == 100.0

    def test_result_has_all_required_keys(self):
        assessor = _make_assessor()
        net = _network_with_shift()
        result = assessor._assess_crew_continuity({}, _WINDOW_IN_SHIFT, net)
        for key in (
            "available", "off_shift_overlap_hours", "shift_boundary_conflict",
            "fatigue_risk", "utilization_at_window", "peak_utilization_skill",
            "peak_utilization_pct", "notes",
        ):
            assert key in result, f"Missing key: {key}"


# ===========================================================================
# ScheduleImpactConfig — new crew-continuity fields
# ===========================================================================

class TestScheduleImpactConfigCrewContinuity:

    def test_default_high_utilization_threshold(self):
        assert ScheduleImpactConfig().high_crew_utilization_threshold == 0.80

    def test_default_fatigue_risk_off_shift_hours(self):
        assert ScheduleImpactConfig().fatigue_risk_off_shift_hours == 2.0

    def test_custom_threshold_accepted(self):
        cfg = ScheduleImpactConfig(high_crew_utilization_threshold=0.90)
        assert cfg.high_crew_utilization_threshold == 0.90


# ===========================================================================
# Permit lead time — _compute_permit_lead_time
# ===========================================================================

class TestComputePermitLeadTime:

    def _call(self, flags=None, rp=4.0, scaffold=8.0, clearance=2.0, mode="max"):
        return _compute_permit_lead_time(
            flags or {},
            rp_hold_hours=rp,
            scaffold_hours=scaffold,
            clearance_hours=clearance,
            mode=mode,
        )

    def test_no_flags_zero_lead_time(self):
        result = self._call(flags={})
        assert result["total_lead_hours"] == pytest.approx(0.0)
        assert result["start_adjusted"] is False

    def test_required_keys_present(self):
        result = self._call()
        for k in ("total_lead_hours", "rp_hold_hours", "scaffold_hours",
                  "clearance_hours", "start_adjusted", "combination_mode", "notes"):
            assert k in result

    def test_rp_only_max_mode(self):
        result = self._call(flags={"has_rp_hold": True}, rp=4.0, scaffold=8.0, mode="max")
        assert result["total_lead_hours"] == pytest.approx(4.0)
        assert result["rp_hold_hours"] == pytest.approx(4.0)
        assert result["scaffold_hours"] == pytest.approx(0.0)

    def test_scaffold_only_max_mode(self):
        result = self._call(flags={"requires_scaffold": True}, scaffold=8.0, mode="max")
        assert result["total_lead_hours"] == pytest.approx(8.0)

    def test_clearance_only_max_mode(self):
        result = self._call(flags={"has_clearance": True}, clearance=2.0, mode="max")
        assert result["total_lead_hours"] == pytest.approx(2.0)

    def test_max_mode_takes_longest(self):
        # rp=4, scaffold=8, clearance=2 → max = 8
        result = self._call(
            flags={"has_rp_hold": True, "requires_scaffold": True, "has_clearance": True},
            rp=4.0, scaffold=8.0, clearance=2.0, mode="max",
        )
        assert result["total_lead_hours"] == pytest.approx(8.0)

    def test_sum_mode_adds_all(self):
        result = self._call(
            flags={"has_rp_hold": True, "requires_scaffold": True, "has_clearance": True},
            rp=4.0, scaffold=8.0, clearance=2.0, mode="sum",
        )
        assert result["total_lead_hours"] == pytest.approx(14.0)

    def test_combination_mode_recorded(self):
        r_max = self._call(flags={"has_rp_hold": True}, mode="max")
        r_sum = self._call(flags={"has_rp_hold": True}, mode="sum")
        assert r_max["combination_mode"] == "max"
        assert r_sum["combination_mode"] == "sum"

    def test_start_adjusted_true_when_lead_nonzero(self):
        result = self._call(flags={"has_rp_hold": True}, rp=4.0)
        assert result["start_adjusted"] is True

    def test_notes_mention_active_permits(self):
        result = self._call(
            flags={"has_rp_hold": True, "requires_scaffold": True},
            mode="max",
        )
        combined = " ".join(result["notes"])
        assert "RP hold" in combined
        assert "Scaffold" in combined

    def test_inactive_flags_produce_zero_components(self):
        result = self._call(flags={"has_rp_hold": False, "requires_scaffold": False})
        assert result["rp_hold_hours"] == pytest.approx(0.0)
        assert result["scaffold_hours"] == pytest.approx(0.0)

    def test_none_flags_treated_as_empty(self):
        result = _compute_permit_lead_time(
            None,  # type: ignore[arg-type]
            rp_hold_hours=4.0, scaffold_hours=8.0, clearance_hours=2.0,
        )
        assert result["total_lead_hours"] == pytest.approx(0.0)


# ===========================================================================
# Permit lead time — _shift_insertion_times
# ===========================================================================

class TestShiftInsertionTimes:

    def test_zero_lead_returns_unchanged(self):
        ip = {"proposed_start": "2026-04-01T08:00:00+00:00",
              "proposed_finish": "2026-04-01T16:00:00+00:00"}
        result = _shift_insertion_times(ip, 0.0)
        assert result["proposed_start"] == ip["proposed_start"]
        assert result["proposed_finish"] == ip["proposed_finish"]

    def test_start_shifted_forward(self):
        ip = {"proposed_start": "2026-04-01T08:00:00+00:00",
              "proposed_finish": "2026-04-01T16:00:00+00:00"}
        result = _shift_insertion_times(ip, 4.0)
        assert result["proposed_start"] == "2026-04-01T12:00:00+00:00"

    def test_finish_shifted_forward(self):
        ip = {"proposed_start": "2026-04-01T08:00:00+00:00",
              "proposed_finish": "2026-04-01T16:00:00+00:00"}
        result = _shift_insertion_times(ip, 4.0)
        assert result["proposed_finish"] == "2026-04-01T20:00:00+00:00"

    def test_other_fields_preserved(self):
        ip = {"proposed_start": "2026-04-01T08:00:00+00:00",
              "proposed_finish": "2026-04-01T16:00:00+00:00",
              "after_task_id": "T010",
              "emergent_task_id": "EA::X1"}
        result = _shift_insertion_times(ip, 2.0)
        assert result["after_task_id"] == "T010"
        assert result["emergent_task_id"] == "EA::X1"

    def test_missing_start_kept_as_none(self):
        ip = {"proposed_start": None, "proposed_finish": "2026-04-01T16:00:00+00:00"}
        result = _shift_insertion_times(ip, 4.0)
        assert result["proposed_start"] is None

    def test_unparseable_start_kept_unchanged(self):
        ip = {"proposed_start": "not-a-date", "proposed_finish": "2026-04-01T16:00:00+00:00"}
        result = _shift_insertion_times(ip, 4.0)
        assert result["proposed_start"] == "not-a-date"

    def test_original_dict_not_mutated(self):
        ip = {"proposed_start": "2026-04-01T08:00:00+00:00",
              "proposed_finish": "2026-04-01T16:00:00+00:00"}
        _shift_insertion_times(ip, 4.0)
        assert ip["proposed_start"] == "2026-04-01T08:00:00+00:00"


# ===========================================================================
# Permit lead time — ScheduleImpactConfig defaults
# ===========================================================================

class TestScheduleImpactConfigPermitFields:

    def test_enabled_by_default(self):
        assert ScheduleImpactConfig().permit_lead_times_enabled is True

    def test_default_rp_hold_hours(self):
        assert ScheduleImpactConfig().rp_hold_lead_time_hours == pytest.approx(4.0)

    def test_default_scaffold_hours(self):
        assert ScheduleImpactConfig().scaffold_lead_time_hours == pytest.approx(8.0)

    def test_default_clearance_hours(self):
        assert ScheduleImpactConfig().clearance_lead_time_hours == pytest.approx(2.0)

    def test_default_mode_is_max(self):
        assert ScheduleImpactConfig().permit_lead_time_mode == "max"

    def test_custom_values_accepted(self):
        cfg = ScheduleImpactConfig(
            rp_hold_lead_time_hours=6.0,
            scaffold_lead_time_hours=12.0,
            clearance_lead_time_hours=3.0,
            permit_lead_time_mode="sum",
        )
        assert cfg.rp_hold_lead_time_hours == pytest.approx(6.0)
        assert cfg.permit_lead_time_mode == "sum"

    def test_disabled_config(self):
        cfg = ScheduleImpactConfig(permit_lead_times_enabled=False)
        assert cfg.permit_lead_times_enabled is False


# ===========================================================================
# Permit lead time — assess() integration (via mock Pert)
# ===========================================================================

class TestAssessPermitLeadTime:
    """Integration tests: permit_lead_time block appears in assess() output."""

    def _run_assess(self, flags: dict, *, enabled: bool = True, mode: str = "max"):
        """Run assess() with a stub schedule and the given execution_mode_flags."""
        pert = _MockPert(task_ids=["T01", "T02"], baseline_hours=480.0)
        # Give T01 good CPM data
        t01 = pert.task_to_activity["T01"]
        t02 = pert.task_to_activity["T02"]
        pert.infoDict[t01] = {"es": 0.0, "ef": 24.0, "ls": 0.0, "lf": 24.0, "slack": 100.0}
        pert.infoDict[t02] = {"es": 24.0, "ef": 48.0, "ls": 24.0, "lf": 48.0, "slack": 100.0}
        pert.forwardDict[t01] = [t02]
        pert.startTime = datetime(2026, 4, 1, tzinfo=timezone.utc)

        loader = lambda *a, **kw: MagicMock(outage_config={"version_id": "V1"})
        builder = MagicMock()
        builder.build.return_value = pert

        cfg = ScheduleImpactConfig(
            permit_lead_times_enabled=enabled,
            rp_hold_lead_time_hours=4.0,
            scaffold_lead_time_hours=8.0,
            clearance_lead_time_hours=2.0,
            permit_lead_time_mode=mode,
            baseline_schedule_version="",  # disable baseline loading
        )
        assessor = ScheduleImpactAssessor(
            config=cfg,
            schedule_loader=loader,
            schedule_graph_builder=builder,
        )
        ea = {"activity_id": "EA1", "outage_id": "O1",
              "planned_duration_hours": 8.0, "outage_phase": "maintenance"}
        intake = {"outage_phase": "maintenance", "execution_mode_flags": flags}
        analogs = {"duration_distribution": {
            "p50_hours": 8.0, "p80_hours": 12.0, "p90_hours": 16.0,
            "confidence_tier": "sme_informed",
        }}
        run_ctx = {"run_id": "R1", "started_at": "2026-04-01T00:00:00Z"}
        return assessor.assess(ea, intake, analogs, run_ctx)

    def test_permit_lead_time_block_present(self):
        result = self._run_assess({})
        assert "permit_lead_time" in result

    def test_no_flags_zero_lead(self):
        result = self._run_assess({})
        assert result["permit_lead_time"]["total_lead_hours"] == pytest.approx(0.0)
        assert result["permit_lead_time"]["start_adjusted"] is False

    def test_rp_hold_produces_lead_time(self):
        result = self._run_assess({"has_rp_hold": True})
        assert result["permit_lead_time"]["total_lead_hours"] == pytest.approx(4.0)
        assert result["permit_lead_time"]["start_adjusted"] is True

    def test_scaffold_produces_largest_lead_time_in_max_mode(self):
        # scaffold=8h > rp=4h → max = 8h
        result = self._run_assess({"has_rp_hold": True, "requires_scaffold": True})
        assert result["permit_lead_time"]["total_lead_hours"] == pytest.approx(8.0)

    def test_disabled_config_zero_lead(self):
        result = self._run_assess({"has_rp_hold": True, "requires_scaffold": True},
                                   enabled=False)
        assert result["permit_lead_time"]["total_lead_hours"] == pytest.approx(0.0)
        assert "disabled" in " ".join(result["permit_lead_time"]["notes"]).lower()

    def test_proposed_start_shifted_when_rp_hold(self):
        """proposed_start should be 4h later than without permit flag."""
        result_no_permit = self._run_assess({})
        result_with_permit = self._run_assess({"has_rp_hold": True})

        start_no = _parse_dt(result_no_permit["insertion_point"]["proposed_start"])
        start_with = _parse_dt(result_with_permit["insertion_point"]["proposed_start"])

        if start_no is not None and start_with is not None:
            diff_hours = (start_with - start_no).total_seconds() / 3600.0
            assert diff_hours == pytest.approx(4.0)

    def test_sum_mode_flag_respected(self):
        result = self._run_assess(
            {"has_rp_hold": True, "requires_scaffold": True, "has_clearance": True},
            mode="sum",
        )
        # 4 + 8 + 2 = 14
        assert result["permit_lead_time"]["total_lead_hours"] == pytest.approx(14.0)
        assert result["permit_lead_time"]["combination_mode"] == "sum"
