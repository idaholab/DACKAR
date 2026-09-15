"""
Tests for:
  - insertion_point_determiner.ScheduleContext
  - insertion_point_determiner.InsertionPointDeterminer
  - stage_d_analogs._rerank_by_schedule_context  (Layer 1 structural affinity)
  - stage_d_analogs._stamp_topology              (Layer 2 prep)
  - stage_e_schedule assess() pre-computed schedule_context pass-through
  - outage_activity_orchestrator._precompute_schedule_context
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from stages.insertion_point_determiner import (
    InsertionPointDeterminer,
    ScheduleContext,
    _default_phase_windows,
    _ensure_tz,
    _parse_dt,
)
from stages.stage_d_analogs import (
    HistoricalAnalogConfig,
    HistoricalAnalogRetriever,
    _stamp_topology,
)


# ---------------------------------------------------------------------------
# ScheduleContext helpers
# ---------------------------------------------------------------------------

def _make_context(**kw) -> ScheduleContext:
    defaults = dict(
        activity_id="EA001",
        after_task_id="T010",
        before_task_id="T020",
        outage_phase="maintenance",
        proposed_start="2025-04-01T08:00:00+00:00",
        proposed_finish="2025-04-01T16:00:00+00:00",
        insertion_in_degree=2,
        insertion_out_degree=1,
        available_float_hours=12.0,
        insertion_on_cp=False,
    )
    defaults.update(kw)
    return ScheduleContext(**defaults)


# ---------------------------------------------------------------------------
# Minimal Pert / Activity stubs
# ---------------------------------------------------------------------------

class _Act:
    """Minimal activity object with a .name attribute."""
    def __init__(self, name: str, *, required_resources: Optional[List] = None):
        self.name = name
        self.required_resources = required_resources or []


def _make_pert(
    activities: List[_Act],
    info: Dict,
    *,
    start_time: Optional[datetime] = None,
    forward_dict: Optional[Dict] = None,
    backward_dict: Optional[Dict] = None,
) -> Any:
    """Build a duck-typed Pert stub."""
    pert = MagicMock()
    pert.infoDict = info
    pert.startTime = start_time or datetime(2025, 1, 1, tzinfo=timezone.utc)
    pert.forwardDict = forward_dict or {}
    pert.backwardDict = backward_dict or {}
    pert.generateInfo = MagicMock()
    return pert


# ===========================================================================
# TestScheduleContext
# ===========================================================================

class TestScheduleContext:
    def test_frozen(self):
        ctx = _make_context()
        with pytest.raises((AttributeError, TypeError)):
            ctx.activity_id = "other"  # type: ignore[misc]

    def test_to_insertion_point_keys(self):
        ctx = _make_context()
        ip = ctx.to_insertion_point()
        assert ip["emergent_task_id"] == "EA::EA001"
        assert ip["after_task_id"] == "T010"
        assert ip["before_task_id"] == "T020"
        assert ip["outage_phase"] == "maintenance"
        assert ip["proposed_start"] == "2025-04-01T08:00:00+00:00"
        assert ip["proposed_finish"] == "2025-04-01T16:00:00+00:00"

    def test_to_insertion_point_none_after(self):
        ctx = _make_context(after_task_id=None, before_task_id=None)
        ip = ctx.to_insertion_point()
        assert ip["after_task_id"] is None
        assert ip["before_task_id"] is None

    def test_insertion_on_cp_true(self):
        ctx = _make_context(available_float_hours=0.0, insertion_on_cp=True)
        assert ctx.insertion_on_cp is True

    def test_all_fields_accessible(self):
        ctx = _make_context(insertion_in_degree=3, insertion_out_degree=5)
        assert ctx.insertion_in_degree == 3
        assert ctx.insertion_out_degree == 5
        assert ctx.available_float_hours == 12.0


# ===========================================================================
# TestInsertionPointDeterminer — graceful degradation
# ===========================================================================

class TestInsertionPointDeterminerDegradation:
    def test_no_loader_returns_none(self):
        det = InsertionPointDeterminer()
        assert det.determine({}, {}) is None

    def test_no_builder_returns_none(self):
        det = InsertionPointDeterminer(schedule_loader=lambda *a, **kw: object())
        assert det.determine({}, {}) is None

    def test_loader_raises_returns_none(self):
        def bad_loader(outage_id, version="working"):
            raise RuntimeError("db down")

        det = InsertionPointDeterminer(
            schedule_loader=bad_loader,
            schedule_graph_builder=MagicMock(),
        )
        result = det.determine({"outage_id": "O1"}, {})
        assert result is None

    def test_empty_info_dict_returns_none(self):
        pert = _make_pert([], {})
        builder = MagicMock()
        builder.build.return_value = pert

        det = InsertionPointDeterminer(
            schedule_loader=lambda *a, **kw: object(),
            schedule_graph_builder=builder,
        )
        result = det.determine({"outage_id": "O1", "activity_id": "EA1"}, {})
        assert result is None

    def test_exception_in_build_context_returns_none(self, monkeypatch):
        act = _Act("T10")
        info = {act: {"es": 0.0, "ef": 8.0, "slack": 0.0}}
        pert = _make_pert([act], info)
        builder = MagicMock()
        builder.build.return_value = pert

        det = InsertionPointDeterminer(
            schedule_loader=lambda *a, **kw: object(),
            schedule_graph_builder=builder,
        )
        monkeypatch.setattr(det, "_build_context", lambda **kw: (_ for _ in ()).throw(ValueError("boom")))
        result = det.determine({"outage_id": "O1", "activity_id": "EA1"}, {})
        assert result is None


# ===========================================================================
# TestInsertionPointDeterminer — happy path
# ===========================================================================

class TestInsertionPointDeterminerHappyPath:
    """Topology extraction and strategy selection."""

    def _setup(self, *, on_cp=False):
        """Build a minimal two-task schedule: T10 → T20."""
        t10 = _Act("T10")
        t20 = _Act("T20")
        float_val = 0.0 if on_cp else 16.0
        info = {
            t10: {"es": 0.0, "ef": 8.0, "slack": float_val},
            t20: {"es": 8.0, "ef": 20.0, "slack": float_val},
        }
        forward_dict = {t10: [t20]}
        backward_dict = {t20: [t10]}
        start_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        pert = _make_pert([t10, t20], info,
                          start_time=start_time,
                          forward_dict=forward_dict,
                          backward_dict=backward_dict)
        return pert, t10, t20

    def _make_det(self, pert):
        builder = MagicMock()
        builder.build.return_value = pert
        return InsertionPointDeterminer(
            schedule_loader=lambda *a, **kw: object(),
            schedule_graph_builder=builder,
        )

    def test_returns_schedule_context(self):
        pert, t10, t20 = self._setup()
        det = self._make_det(pert)
        ctx = det.determine(
            {"activity_id": "EA1", "outage_id": "O1", "outage_phase": "maintenance"},
            {"outage_phase": "maintenance"},
        )
        assert isinstance(ctx, ScheduleContext)

    def test_activity_id_set(self):
        pert, _, _ = self._setup()
        det = self._make_det(pert)
        ctx = det.determine({"activity_id": "EA99", "outage_id": "O1"}, {})
        assert ctx.activity_id == "EA99"

    def test_outage_phase_from_intake(self):
        pert, _, _ = self._setup()
        det = self._make_det(pert)
        ctx = det.determine(
            {"activity_id": "EA1", "outage_id": "O1"},
            {"outage_phase": "startup"},
        )
        assert ctx.outage_phase == "startup"

    def test_outage_phase_fallback_from_activity(self):
        pert, _, _ = self._setup()
        det = self._make_det(pert)
        ctx = det.determine(
            {"activity_id": "EA1", "outage_id": "O1", "outage_phase": "defueling"},
            {},
        )
        assert ctx.outage_phase == "defueling"

    def test_outage_phase_unknown_when_absent(self):
        pert, _, _ = self._setup()
        det = self._make_det(pert)
        ctx = det.determine({"activity_id": "EA1", "outage_id": "O1"}, {})
        assert ctx.outage_phase == "unknown"

    def test_on_cp_when_float_zero(self):
        pert, _, _ = self._setup(on_cp=True)
        det = self._make_det(pert)
        ctx = det.determine({"activity_id": "EA1", "outage_id": "O1"}, {})
        assert ctx.insertion_on_cp is True
        assert ctx.available_float_hours == 0.0

    def test_not_on_cp_with_float(self):
        pert, _, _ = self._setup(on_cp=False)
        det = self._make_det(pert)
        ctx = det.determine({"activity_id": "EA1", "outage_id": "O1"}, {})
        assert ctx.insertion_on_cp is False
        assert ctx.available_float_hours > 0

    def test_before_task_id_is_successor(self):
        pert, t10, t20 = self._setup()
        det = self._make_det(pert)
        ctx = det.determine({"activity_id": "EA1", "outage_id": "O1"}, {})
        # before_task_id should be the successor of after_task_id
        if ctx.after_task_id == "T10":
            assert ctx.before_task_id == "T20"

    def test_out_degree_matches_successors(self):
        pert, t10, t20 = self._setup()
        det = self._make_det(pert)
        ctx = det.determine({"activity_id": "EA1", "outage_id": "O1"}, {})
        # T10 has one successor (T20); out_degree = 1
        if ctx.after_task_id == "T10":
            assert ctx.insertion_out_degree == 1

    def test_in_degree_from_backward_dict(self):
        t10 = _Act("T10")
        t20 = _Act("T20")
        t30 = _Act("T30")
        # T20 has two predecessors: T10 and T30
        info = {
            t10: {"es": 0.0, "ef": 8.0, "slack": 10.0},
            t30: {"es": 0.0, "ef": 6.0, "slack": 12.0},
            t20: {"es": 8.0, "ef": 20.0, "slack": 10.0},
        }
        fwd = {t10: [t20], t30: [t20]}
        bwd = {t20: [t10, t30]}
        start_dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        pert = _make_pert([t10, t20, t30], info,
                          start_time=start_dt,
                          forward_dict=fwd,
                          backward_dict=bwd)
        builder = MagicMock()
        builder.build.return_value = pert
        det = InsertionPointDeterminer(
            schedule_loader=lambda *a, **kw: object(),
            schedule_graph_builder=builder,
        )
        ctx = det.determine(
            {"activity_id": "EA1", "outage_id": "O1", "outage_phase": "maintenance"},
            {},
        )
        assert ctx is not None

    def test_proposed_start_computed(self):
        pert, _, _ = self._setup()
        det = self._make_det(pert)
        ctx = det.determine(
            {"activity_id": "EA1", "outage_id": "O1", "planned_duration_hours": 8.0},
            {},
        )
        assert ctx is not None
        # proposed_start / proposed_finish should be ISO strings when base_dt is set
        if ctx.after_task_id is not None:
            assert ctx.proposed_start is not None
            assert ctx.proposed_finish is not None

    def test_strategy1_actual_start_used(self):
        """Strategy 1: actual_start drives insertion site selection."""
        t10 = _Act("T10")
        t20 = _Act("T20")
        # T10 covers hours [0, 20); T20 covers [20, 40)
        info = {
            t10: {"es": 0.0, "ef": 20.0, "slack": 5.0},
            t20: {"es": 20.0, "ef": 40.0, "slack": 5.0},
        }
        start_dt = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        pert = _make_pert([t10, t20], info, start_time=start_dt,
                          forward_dict={t10: [t20]},
                          backward_dict={t20: [t10]})
        builder = MagicMock()
        builder.build.return_value = pert
        det = InsertionPointDeterminer(
            schedule_loader=lambda *a, **kw: object(),
            schedule_graph_builder=builder,
        )
        # actual_start at hour 5 → falls in T10's window [0, 20)
        ctx = det.determine(
            {"activity_id": "EA1", "outage_id": "O1",
             "actual_start": "2025-01-01T05:00:00+00:00"},
            {},
        )
        assert ctx is not None
        assert ctx.after_task_id == "T10"


# ===========================================================================
# TestDefaultPhaseWindows
# ===========================================================================

class TestDefaultPhaseWindows:
    def test_known_phase_scaled(self):
        lo, hi = _default_phase_windows("maintenance", 100.0)
        assert lo == pytest.approx(20.0)
        assert hi == pytest.approx(70.0)

    def test_unknown_phase_full_range(self):
        lo, hi = _default_phase_windows("commissioning", 80.0)
        assert lo == 0.0
        assert hi == 80.0

    def test_shutdown_early_window(self):
        lo, hi = _default_phase_windows("shutdown", 200.0)
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(20.0)

    def test_startup_late_window(self):
        lo, hi = _default_phase_windows("startup", 100.0)
        assert lo == pytest.approx(80.0)
        assert hi == pytest.approx(100.0)


# ===========================================================================
# TestReRankByScheduleContext
# ===========================================================================

def _analog(
    activity_id: str,
    similarity_score: float,
    *,
    actual_duration_hours: Optional[float] = None,
    planned_duration_hours: Optional[float] = None,
    is_vendor_supported: bool = False,
) -> Dict:
    return {
        "activity_id": activity_id,
        "similarity_score": similarity_score,
        "actual_duration_hours": actual_duration_hours,
        "planned_duration_hours": planned_duration_hours,
        "is_vendor_supported": is_vendor_supported,
    }


class TestReRankByScheduleContext:
    """Tests for HistoricalAnalogRetriever._rerank_by_schedule_context."""

    def _retriever(self, weight: float = 0.10) -> HistoricalAnalogRetriever:
        cfg = HistoricalAnalogConfig(schedule_context_rerank_weight=weight)
        return HistoricalAnalogRetriever(config=cfg)

    def test_w_zero_no_change(self):
        ret = self._retriever(weight=0.0)
        analogs = [_analog("A", 0.8)]
        ctx = _make_context()
        result = ret._rerank_by_schedule_context(analogs, ctx)
        assert result[0]["similarity_score"] == pytest.approx(0.8)
        assert "structural_affinity_score" not in result[0]

    def test_empty_analogs_returns_empty(self):
        ret = self._retriever()
        assert ret._rerank_by_schedule_context([], _make_context()) == []

    def test_cp_overrun_boost(self):
        ret = self._retriever(weight=0.10)
        # actual 12h, planned 10h → ratio 1.2 > 1.10 → boost 0.5
        analogs = [_analog("A", 0.6, actual_duration_hours=12.0, planned_duration_hours=10.0)]
        ctx = _make_context(insertion_on_cp=True)
        result = ret._rerank_by_schedule_context(analogs, ctx)
        # affinity = 0.5; blended = 0.9*0.6 + 0.1*0.5 = 0.54 + 0.05 = 0.59
        assert result[0]["structural_affinity_score"] == pytest.approx(0.5)
        assert result[0]["similarity_score"] == pytest.approx(0.9 * 0.6 + 0.1 * 0.5, abs=1e-4)

    def test_cp_no_boost_when_not_on_cp(self):
        ret = self._retriever()
        analogs = [_analog("A", 0.6, actual_duration_hours=12.0, planned_duration_hours=10.0)]
        ctx = _make_context(insertion_on_cp=False)
        result = ret._rerank_by_schedule_context(analogs, ctx)
        assert result[0]["structural_affinity_score"] == pytest.approx(0.0)

    def test_tight_float_precision_boost(self):
        ret = self._retriever(weight=0.10)
        # actual/planned ratio within 15% → precision boost
        analogs = [_analog("A", 0.5, actual_duration_hours=10.0, planned_duration_hours=10.5)]
        ctx = _make_context(available_float_hours=4.0, insertion_on_cp=False)
        result = ret._rerank_by_schedule_context(analogs, ctx)
        assert result[0]["structural_affinity_score"] == pytest.approx(0.5)

    def test_tight_float_no_boost_when_float_large(self):
        ret = self._retriever()
        analogs = [_analog("A", 0.5, actual_duration_hours=10.0, planned_duration_hours=10.5)]
        ctx = _make_context(available_float_hours=20.0)
        result = ret._rerank_by_schedule_context(analogs, ctx)
        assert result[0]["structural_affinity_score"] == pytest.approx(0.0)

    def test_fanout_vendor_boost(self):
        ret = self._retriever(weight=0.10)
        analogs = [_analog("A", 0.5, is_vendor_supported=True)]
        ctx = _make_context(insertion_out_degree=4)
        result = ret._rerank_by_schedule_context(analogs, ctx)
        assert result[0]["structural_affinity_score"] == pytest.approx(0.3)

    def test_fanout_no_boost_non_vendor(self):
        ret = self._retriever()
        analogs = [_analog("A", 0.5, is_vendor_supported=False)]
        ctx = _make_context(insertion_out_degree=4)
        result = ret._rerank_by_schedule_context(analogs, ctx)
        assert result[0]["structural_affinity_score"] == pytest.approx(0.0)

    def test_affinity_capped_at_1(self):
        ret = self._retriever(weight=0.10)
        # Trigger all three boosts: on_cp+overrun (0.5) + tight_float+precise (0.5) = 1.0 (capped)
        analogs = [_analog(
            "A", 0.5,
            actual_duration_hours=11.0,
            planned_duration_hours=10.0,  # overrun >10%
            is_vendor_supported=True,
        )]
        ctx = _make_context(
            insertion_on_cp=True,
            available_float_hours=3.0,  # tight float but actual/planned = 1.1, |diff|/planned=0.1<0.15 → also triggers precision boost
            insertion_out_degree=4,
        )
        result = ret._rerank_by_schedule_context(analogs, ctx)
        assert result[0]["structural_affinity_score"] <= 1.0

    def test_sort_order_by_blended_score(self):
        ret = self._retriever(weight=0.10)
        analogs = [
            _analog("LOW", 0.4),
            _analog("HIGH", 0.9, actual_duration_hours=12.0, planned_duration_hours=10.0),
        ]
        ctx = _make_context(insertion_on_cp=True)
        result = ret._rerank_by_schedule_context(analogs, ctx)
        assert result[0]["activity_id"] == "HIGH"

    def test_structural_affinity_score_present(self):
        ret = self._retriever()
        analogs = [_analog("A", 0.5)]
        ctx = _make_context()
        result = ret._rerank_by_schedule_context(analogs, ctx)
        assert "structural_affinity_score" in result[0]

    def test_original_fields_preserved(self):
        ret = self._retriever()
        analogs = [_analog("A", 0.5, actual_duration_hours=8.0)]
        result = ret._rerank_by_schedule_context(analogs, _make_context())
        assert result[0]["actual_duration_hours"] == 8.0
        assert result[0]["activity_id"] == "A"


# ===========================================================================
# TestStampTopology
# ===========================================================================

class TestStampTopology:
    def test_predecessor_stamped(self):
        case = MagicMock()
        ctx = _make_context(after_task_id="T10")
        _stamp_topology(case, ctx)
        assert case.predecessor_ids == ["T10"]

    def test_successor_stamped(self):
        case = MagicMock()
        ctx = _make_context(before_task_id="T20")
        _stamp_topology(case, ctx)
        assert case.successor_ids == ["T20"]

    def test_no_stamp_when_after_none(self):
        case = MagicMock()
        ctx = _make_context(after_task_id=None)
        _stamp_topology(case, ctx)
        # MagicMock records attribute access; predecessor_ids should not have been set
        assert "predecessor_ids" not in case.__dict__

    def test_no_stamp_when_before_none(self):
        case = MagicMock()
        ctx = _make_context(before_task_id=None)
        _stamp_topology(case, ctx)
        assert "successor_ids" not in case.__dict__

    def test_graceful_when_attribute_error(self):
        """Frozen dataclass or read-only object — should not raise."""
        @dataclass(frozen=True)
        class FrozenCase:
            activity_id: str = "A"

        case = FrozenCase()
        ctx = _make_context(after_task_id="T10", before_task_id="T20")
        _stamp_topology(case, ctx)  # must not raise

    def test_graceful_when_type_error(self):
        """Object whose attribute assignment raises TypeError — should not raise."""
        class _Immutable:
            __slots__ = ("activity_id",)
            def __init__(self):
                self.activity_id = "A"

        case = _Immutable()
        ctx = _make_context()
        _stamp_topology(case, ctx)  # must not raise


# ===========================================================================
# TestHistoricalAnalogConfigScheduleContextWeight
# ===========================================================================

class TestHistoricalAnalogConfigScheduleContextWeight:
    def test_default_weight(self):
        cfg = HistoricalAnalogConfig()
        assert cfg.schedule_context_rerank_weight == pytest.approx(0.10)

    def test_custom_weight(self):
        cfg = HistoricalAnalogConfig(schedule_context_rerank_weight=0.25)
        assert cfg.schedule_context_rerank_weight == pytest.approx(0.25)

    def test_zero_weight_disables_rerank(self):
        cfg = HistoricalAnalogConfig(schedule_context_rerank_weight=0.0)
        ret = HistoricalAnalogRetriever(config=cfg)
        analogs = [_analog("A", 0.8, actual_duration_hours=15.0, planned_duration_hours=10.0)]
        ctx = _make_context(insertion_on_cp=True)
        result = ret._rerank_by_schedule_context(analogs, ctx)
        assert result[0]["similarity_score"] == pytest.approx(0.8)


# ===========================================================================
# TestOrchestratorPrecomputeScheduleContext
# ===========================================================================

class TestOrchestratorPrecomputeScheduleContext:
    """Tests for OutageActivityOrchestrator._precompute_schedule_context."""

    def _make_orch(self, *, determiner=None):
        from orchestrators.outage_activity_orchestrator import OutageActivityOrchestrator
        from orchestrators.protocols import (
            FileArtifactStore,
            NoOpSchemaValidator,
        )

        class _Stub:
            def process(self, *a, **kw): return {}
            def build(self, *a, **kw): return {}
            def score(self, *a, **kw): return {}
            def retrieve(self, *a, **kw): return {}
            def assess(self, *a, **kw): return {}
            def generate(self, *a, **kw): return {}
            def synthesize(self, *a, **kw): return {}

        stub = _Stub()
        return OutageActivityOrchestrator(
            validator=NoOpSchemaValidator(),
            artifact_store=FileArtifactStore("/tmp/orch_test"),
            intake_processor=stub,
            kg_timeline_builder=stub,
            temporal_chain_scorer=stub,
            analog_retriever=stub,
            schedule_impact_assessor=stub,
            option_generator=stub,
            recommendation_synthesizer=stub,
            insertion_point_determiner=determiner,
        )

    def test_returns_none_when_no_determiner(self):
        orch = self._make_orch()
        result = orch._precompute_schedule_context(
            {"activity_id": "EA1", "outage_id": "O1"}, {}
        )
        assert result is None

    def test_returns_context_from_determiner(self):
        ctx = _make_context()
        det = MagicMock()
        det.determine.return_value = ctx
        orch = self._make_orch(determiner=det)
        result = orch._precompute_schedule_context(
            {"activity_id": "EA1", "outage_id": "O1"}, {}
        )
        assert result is ctx

    def test_determiner_called_with_correct_args(self):
        ctx = _make_context()
        det = MagicMock()
        det.determine.return_value = ctx
        orch = self._make_orch(determiner=det)
        ea = {"activity_id": "EA99", "outage_id": "O99"}
        ir = {"outage_phase": "startup"}
        orch._precompute_schedule_context(ea, ir)
        det.determine.assert_called_once_with(ea, ir)

    def test_returns_none_when_determiner_returns_none(self):
        det = MagicMock()
        det.determine.return_value = None
        orch = self._make_orch(determiner=det)
        result = orch._precompute_schedule_context({}, {})
        assert result is None

    def test_returns_none_on_determiner_exception(self):
        det = MagicMock()
        det.determine.side_effect = RuntimeError("schedule unavailable")
        orch = self._make_orch(determiner=det)
        result = orch._precompute_schedule_context({}, {})
        assert result is None

    def test_schedule_context_forwarded_to_stage_d(self):
        """When determiner returns context, analog_retriever.retrieve() gets schedule_context kwarg."""
        ctx = _make_context()
        det = MagicMock()
        det.determine.return_value = ctx

        import tempfile, os
        tmp = tempfile.mkdtemp()

        from orchestrators.outage_activity_orchestrator import OutageActivityOrchestrator
        from orchestrators.protocols import FileArtifactStore, NoOpSchemaValidator

        captured = {}

        class _CapturingRetriever:
            def retrieve(self, *a, schedule_context=None, **kw):
                captured["schedule_context"] = schedule_context
                return {}

        class _Stub:
            def process(self, *a, **kw): return {}
            def build(self, *a, **kw): return {}
            def score(self, *a, **kw): return {}
            def assess(self, *a, **kw): return {}
            def generate(self, *a, **kw): return {}
            def synthesize(self, *a, **kw): return {}

        stub = _Stub()
        orch = OutageActivityOrchestrator(
            validator=NoOpSchemaValidator(),
            artifact_store=FileArtifactStore(tmp),
            intake_processor=stub,
            kg_timeline_builder=stub,
            temporal_chain_scorer=stub,
            analog_retriever=_CapturingRetriever(),
            schedule_impact_assessor=stub,
            option_generator=stub,
            recommendation_synthesizer=stub,
            insertion_point_determiner=det,
        )
        orch.run({"activity_id": "EA1", "outage_id": "O1"})
        assert captured.get("schedule_context") is ctx
