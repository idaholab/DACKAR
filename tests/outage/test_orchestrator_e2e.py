"""
End-to-end pipeline tests: A → B → C → D → E(stub) → F → G.

Uses run_pipeline() from the demo module with the two pre-defined scenarios.
Stages A, C, D, F, G run their real production logic.
Stage B runs with a stub KG driver.
Stage E uses the pre-built stub artifact (no LOGOS required).

Coverage targets:
    run_pipeline()           — produces all seven artifact keys
    RCP seal scenario        — ESCALATE decision, regulatory flag set,
                               float_analysis.is_critical_path_impact = True
    Snubber scenario         — PROCEED decision, no regulatory block,
                               non-critical path impact
    Artifact schemas         — required keys present in each stage output
    Stage E fallback         — when schedule_data_root is None, pre-built stub used
    Stage E LOGOS path       — when schedule_data_root is non-existent dir,
                               FileNotFoundError caught, stub returned
    run_id propagation       — custom run_id threaded through all artifacts
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_OUTAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "dackar" / "outage"
_DEMO_ROOT = _OUTAGE_ROOT / "demos" / "unexpected_act_workflow_1"
for _p in (_OUTAGE_ROOT, _DEMO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytest.importorskip("demo_scenarios", reason="outage demos (src/dackar/outage/demos) arrive in MR #17")
from demo_scenarios import (
    run_pipeline,
    SCENARIO_RCP_SEAL,
    SCENARIO_SNUBBER_EXT,
    SCENARIO_UNKNOWN_COMPONENT,
)

# ===========================================================================
# Helpers
# ===========================================================================

_EXPECTED_ARTIFACT_KEYS = {
    "intake": ("activity_id", "run_id", "emergence_type", "regulatory_drivers",
               "extracted_entities", "data_quality_score"),
    "timeline": ("activity_id", "run_id", "component_id", "events",
                 "recurrence_indicators", "data_coverage"),
    "temporal": ("activity_id", "run_id", "chain_links"),
    "analogs": ("activity_id", "run_id", "analogs", "duration_distribution"),
    "schedule": ("activity_id", "run_id",),
    "options": ("activity_id", "run_id", "options"),
    "recommendation": ("activity_id", "run_id", "decision_status", "executive_summary"),
}


def _check_artifact_keys(result: Dict[str, Any], artifact: str):
    obj = result[artifact]
    for key in _EXPECTED_ARTIFACT_KEYS.get(artifact, ()):
        assert key in obj, f"{artifact} missing key '{key}'"


# ===========================================================================
# RCP Seal scenario (ESCALATE)
# ===========================================================================

class TestRCPSealScenario:

    @pytest.fixture(scope="class")
    def result(self):
        return run_pipeline(SCENARIO_RCP_SEAL)

    def test_top_level_keys(self, result):
        for key in ("scenario_label", "run_id", "intake", "timeline",
                    "temporal", "analogs", "schedule", "options", "recommendation"):
            assert key in result, f"missing top-level key '{key}'"

    def test_intake_artifact_keys(self, result):
        _check_artifact_keys(result, "intake")

    def test_timeline_artifact_keys(self, result):
        _check_artifact_keys(result, "timeline")

    def test_temporal_artifact_keys(self, result):
        _check_artifact_keys(result, "temporal")

    def test_analogs_artifact_keys(self, result):
        _check_artifact_keys(result, "analogs")

    def test_schedule_artifact_keys(self, result):
        _check_artifact_keys(result, "schedule")

    def test_options_artifact_keys(self, result):
        _check_artifact_keys(result, "options")

    def test_recommendation_artifact_keys(self, result):
        _check_artifact_keys(result, "recommendation")

    def test_decision_status_is_escalate(self, result):
        status = result["recommendation"]["decision_status"]
        assert status == "ESCALATE", f"expected ESCALATE, got {status}"

    def test_regulatory_flag_set(self, result):
        """RCP seal has TS 3.4.6 — at least one defer_prohibited driver."""
        drivers = result["intake"].get("regulatory_drivers", [])
        assert any(d.get("defer_prohibited") for d in drivers), \
            "expected at least one defer_prohibited regulatory driver for RCP seal"

    def test_emergence_type_is_regulatory_driven(self, result):
        # RCP seal description contains TS 3.4.6 — rule 1 fires → regulatory_driven
        et = result["intake"]["emergence_type"]
        assert et == "regulatory_driven", f"expected regulatory_driven, got {et}"

    def test_critical_path_impact(self, result):
        fa = result["schedule"].get("float_analysis", {})
        assert fa.get("is_critical_path_impact") is True, \
            "RCP seal scenario should have critical path impact"

    def test_scenario_label(self, result):
        assert "RCP" in result["scenario_label"] or "Seal" in result["scenario_label"] \
            or "Scenario 1" in result["scenario_label"]

    def test_options_list_not_empty(self, result):
        assert len(result["options"]["options"]) > 0

    def test_run_id_consistent_across_artifacts(self, result):
        run_id = result["run_id"]
        for artifact in ("intake", "timeline", "temporal", "analogs", "schedule",
                         "options", "recommendation"):
            assert result[artifact].get("run_id") == run_id, \
                f"{artifact}.run_id mismatch"


# ===========================================================================
# Snubber scope expansion scenario (PROCEED)
# ===========================================================================

class TestSnubberScenario:

    @pytest.fixture(scope="class")
    def result(self):
        return run_pipeline(SCENARIO_SNUBBER_EXT)

    def test_top_level_keys(self, result):
        for key in ("scenario_label", "run_id", "intake", "timeline",
                    "temporal", "analogs", "schedule", "options", "recommendation"):
            assert key in result

    def test_decision_status_is_defer_not_escalate(self, result):
        """Snubber scope expansion: non-regulatory, non-critical path → DEFER.

        After the M2 fix, deferred_labor_cost_usd is excluded from total_cost_usd
        to avoid inflating in-outage cost scoring.  With symmetric urgency
        (causal_posture="partial", urgency=0.50) the cost tiebreaker favours
        defer (total_cost=0) over insert_now (total_cost=$2400) by 0.003 risk
        points.  DEFER is a valid recommendation here: the additional snubbers
        are non-safety-critical and can be planned for post-outage maintenance.
        ESCALATE is the one outcome that must never appear for a non-regulatory
        activity with zero CP drag and ample float.
        """
        status = result["recommendation"]["decision_status"]
        assert status != "ESCALATE", (
            f"non-regulatory, non-critical snubber must not produce ESCALATE, got {status}"
        )
        assert status in ("DEFER", "PROCEED"), (
            f"expected DEFER or PROCEED for snubber scope expansion, got {status}"
        )

    def test_no_defer_prohibited_driver(self, result):
        drivers = result["intake"].get("regulatory_drivers", [])
        assert not any(d.get("defer_prohibited") for d in drivers), \
            "snubber scenario should not have defer_prohibited regulatory drivers"

    def test_non_critical_path(self, result):
        fa = result["schedule"].get("float_analysis", {})
        assert fa.get("is_critical_path_impact") is False, \
            "snubber scenario should not have critical path impact"

    def test_options_include_insert_now(self, result):
        option_types = [o.get("option_type") for o in result["options"]["options"]]
        assert "insert_now" in option_types, \
            "snubber scenario should have insert_now as a feasible option"

    def test_run_id_consistent_across_artifacts(self, result):
        run_id = result["run_id"]
        for artifact in ("intake", "timeline", "schedule", "recommendation"):
            assert result[artifact].get("run_id") == run_id


# ===========================================================================
# Custom run_id propagation
# ===========================================================================

class TestCustomRunId:

    def test_custom_run_id_propagated(self):
        custom_id = "CUSTOM-RUN-XYZ-42"
        result = run_pipeline(SCENARIO_RCP_SEAL, run_id=custom_id)
        assert result["run_id"] == custom_id
        assert result["intake"]["run_id"] == custom_id
        assert result["recommendation"]["run_id"] == custom_id

    def test_auto_generated_run_id_has_demo_prefix(self):
        result = run_pipeline(SCENARIO_SNUBBER_EXT)
        assert result["run_id"].startswith("DEMO::")


# ===========================================================================
# Stage E fallback behaviour
# ===========================================================================

class TestStageEFallback:

    def test_no_schedule_root_uses_pre_built_stub(self):
        """Passing schedule_data_root=None must return the pre-built artifact."""
        result = run_pipeline(SCENARIO_RCP_SEAL, schedule_data_root=None)
        # Pre-built artifact has 'schedule_version_id' set to RF-24-WRK-003
        assert result["schedule"].get("schedule_version_id") is not None

    def test_nonexistent_schedule_root_falls_back_to_stub(self):
        """A non-existent data root must trigger FileNotFoundError and return stub."""
        result = run_pipeline(
            SCENARIO_RCP_SEAL,
            schedule_data_root="/nonexistent/path/that/does/not/exist",
        )
        # Should not raise; should return a valid schedule artifact
        assert "activity_id" in result["schedule"]


# ===========================================================================
# Confidence and data quality
# ===========================================================================

class TestDataQuality:

    def test_data_quality_score_in_range(self):
        result = run_pipeline(SCENARIO_RCP_SEAL)
        score = result["intake"]["data_quality_score"]
        assert 0.0 <= score <= 1.0

    def test_analogs_confidence_tier_present(self):
        result = run_pipeline(SCENARIO_SNUBBER_EXT)
        tier = result["analogs"]["duration_distribution"].get("confidence_tier")
        assert tier in ("data_supported", "sme_informed", "low_confidence", None)

    def test_duration_distribution_has_percentiles(self):
        result = run_pipeline(SCENARIO_RCP_SEAL)
        dd = result["analogs"]["duration_distribution"]
        assert dd.get("p50_hours") is not None
        assert dd.get("p80_hours") is not None


# ===========================================================================
# N13 — Stage E RuntimeError → stub artifact + optional_failures
# ===========================================================================

class TestN13StageEStubDegradation:
    """Verify that a RuntimeError from Stage E is caught, a stub artifact is
    produced, and optional_failures is populated — so Stages F and G still run.

    N13 fix: the orchestrator's _stage_e_schedule() wraps the assessor call in
    try/except RuntimeError.  On failure it calls _stub_schedule_impact_artifact()
    and appends to optional_failures.
    """

    def _make_orchestrator(self, *, assessor_raises: bool = False):
        """Build a minimal orchestrator with all stubs except the assessor."""
        import sys
        _ORCH_ROOT = Path(__file__).resolve().parents[2] / "src" / "dackar" / "outage"
        if str(_ORCH_ROOT) not in sys.path:
            sys.path.insert(0, str(_ORCH_ROOT))

        from orchestrators.outage_activity_orchestrator import OutageActivityOrchestrator
        from orchestrators.protocols import (
            NoOpSchemaValidator,
            FileArtifactStore,
            OutageOrchestratorConfig,
        )
        import tempfile

        class _NoOpStage:
            def process(self, *a, **kw): return {}
            def build(self, *a, **kw): return {}
            def score(self, *a, **kw): return {}
            def retrieve(self, *a, **kw): return {}
            def assess(self, *a, **kw):
                if assessor_raises:
                    raise RuntimeError("schedule_loader not injected")
                return {}
            def generate(self, *a, **kw): return {}
            def synthesize(self, *a, **kw): return {}

        stub = _NoOpStage()
        cfg = OutageOrchestratorConfig(persist_intermediate_artifacts=False)
        return OutageActivityOrchestrator(
            validator=NoOpSchemaValidator(),
            artifact_store=FileArtifactStore(tempfile.mkdtemp()),
            intake_processor=stub,
            kg_timeline_builder=stub,
            temporal_chain_scorer=stub,
            analog_retriever=stub,
            schedule_impact_assessor=stub if not assessor_raises else _NoOpStage(),
            option_generator=stub,
            recommendation_synthesizer=stub,
            config=cfg,
        )

    def test_n13_runtime_error_produces_stub_artifact(self):
        """RuntimeError from assessor.assess() must not propagate; stub returned."""
        orch = self._make_orchestrator(assessor_raises=True)
        optional_failures: list = []
        stub_result = orch._stage_e_schedule(
            run_id="TEST-001",
            emergent_activity={"activity_id": "ACT-001"},
            intake_result={},
            historical_analogs={},
            run_context={},
            precomputed=None,
            optional_failures=optional_failures,
        )
        assert isinstance(stub_result, dict), "Stub must return a dict"
        assert stub_result.get("schedule_version_id") == "STUB::no_schedule", (
            "Stub artifact must identify itself via schedule_version_id='STUB::no_schedule' (Y8 fix)"
        )

    def test_n13_optional_failures_populated_on_runtime_error(self):
        """optional_failures must receive one entry when Stage E raises RuntimeError."""
        orch = self._make_orchestrator(assessor_raises=True)
        optional_failures: list = []
        orch._stage_e_schedule(
            run_id="TEST-001",
            emergent_activity={"activity_id": "ACT-001"},
            intake_result={},
            historical_analogs={},
            run_context={},
            precomputed=None,
            optional_failures=optional_failures,
        )
        assert len(optional_failures) == 1, (
            "Exactly one optional_failure entry expected for Stage E RuntimeError"
        )
        failure = optional_failures[0]
        assert failure["stage"] == "stage_e_schedule"
        assert failure["optional"] is True

    def test_n13_stub_artifact_has_null_cp_metrics(self):
        """Stub float_analysis fields must be None so Stage F degrades gracefully."""
        orch = self._make_orchestrator(assessor_raises=True)
        optional_failures: list = []
        stub_result = orch._stage_e_schedule(
            run_id="TEST-001",
            emergent_activity={"activity_id": "ACT-001"},
            intake_result={},
            historical_analogs={},
            run_context={},
            precomputed=None,
            optional_failures=optional_failures,
        )
        float_analysis = stub_result.get("float_analysis", {})
        assert float_analysis.get("float_consumed_hours") == 0.0
        assert float_analysis.get("remaining_float_after_hours") is None  # Y8 fix: schema field name
        assert float_analysis.get("is_critical_path_impact") is False

    def test_y8_stub_artifact_passes_schema_field_names(self):
        """Y8 fix: stub must use schema-valid field names and values throughout."""
        orch = self._make_orchestrator(assessor_raises=True)
        stub_result = orch._stage_e_schedule(
            run_id="TEST-Y8",
            emergent_activity={"activity_id": "ACT-Y8"},
            intake_result={},
            historical_analogs={},
            run_context={},
            precomputed=None,
            optional_failures=[],
        )
        # Non-schema fields that were present before Y8 fix must be absent
        assert "schedule_loader_unavailable" not in stub_result, (
            "'schedule_loader_unavailable' is not in schema (Y8 fix)"
        )
        assert "crew_continuity" not in stub_result, (
            "'crew_continuity' is not in schema (Y8 fix)"
        )
        assert "permit_lead_time" not in stub_result, (
            "'permit_lead_time' is not in schema (Y8 fix)"
        )
        # float_analysis: old wrong field names must be absent
        fa = stub_result.get("float_analysis", {})
        assert "available_float_before" not in fa, (
            "'available_float_before' is wrong field name — schema uses 'available_float_before_hours' (Y8 fix)"
        )
        assert "remaining_float_hours" not in fa, (
            "'remaining_float_hours' is wrong field name — schema uses 'remaining_float_after_hours' (Y8 fix)"
        )
        assert "near_critical_float_threshold" not in fa, (
            "'near_critical_float_threshold' is not in schema (Y8 fix)"
        )
        # criticality_label must be in schema enum
        assert fa.get("criticality_label") in {"critical", "near_critical", "non_critical"}, (
            f"criticality_label '{fa.get('criticality_label')}' not in schema enum (Y8 fix: 'unknown' was invalid)"
        )
        # cp_impact must have required fields
        cp = stub_result.get("cp_impact", {})
        assert "estimated_new_cp_hours" in cp, (
            "cp_impact must include 'estimated_new_cp_hours' (Y8 fix)"
        )
        # duration_estimate must have required fields
        de = stub_result.get("duration_estimate", {})
        for req_field in ("p50_hours", "p80_hours", "p90_hours", "confidence_tier"):
            assert req_field in de, (
                f"duration_estimate missing required field '{req_field}' (Y8 fix)"
            )
        # schedule_version_id must be a non-null string
        assert isinstance(stub_result.get("schedule_version_id"), str), (
            "schedule_version_id must be a string, not None (Y8 fix)"
        )


# ===========================================================================
# N7 — Orchestrator record_completion() no-op path when feedback_writer=None
# ===========================================================================

class TestN7RecordCompletionNoOp:
    """Verify the orchestrator's record_completion() no-op path.

    N7 fix: when feedback_writer is None, record_completion() logs a warning
    and returns a CompletionRecord with index_updated=False, persisted=False,
    and a validation_warning rather than raising AttributeError.
    """

    def _make_orchestrator(self):
        import tempfile
        from orchestrators.outage_activity_orchestrator import OutageActivityOrchestrator
        from orchestrators.protocols import (
            NoOpSchemaValidator,
            FileArtifactStore,
            OutageOrchestratorConfig,
        )

        class _NoOpStage:
            def process(self, *a, **kw): return {}
            def build(self, *a, **kw): return {}
            def score(self, *a, **kw): return {}
            def retrieve(self, *a, **kw): return {}
            def assess(self, *a, **kw): return {}
            def generate(self, *a, **kw): return {}
            def synthesize(self, *a, **kw): return {}

        stub = _NoOpStage()
        return OutageActivityOrchestrator(
            validator=NoOpSchemaValidator(),
            artifact_store=FileArtifactStore(tempfile.mkdtemp()),
            intake_processor=stub,
            kg_timeline_builder=stub,
            temporal_chain_scorer=stub,
            analog_retriever=stub,
            schedule_impact_assessor=stub,
            option_generator=stub,
            recommendation_synthesizer=stub,
            feedback_writer=None,     # ← no feedback writer
            config=OutageOrchestratorConfig(persist_intermediate_artifacts=False),
        )

    def test_n7_record_completion_does_not_raise(self):
        """record_completion() must not raise when feedback_writer is None."""
        orch = self._make_orchestrator()
        # Must not raise
        record = orch.record_completion(
            activity_id="ACT-001",
            run_id="RUN-001",
            actual_duration_hours=8.0,
        )
        assert record is not None

    def test_n7_no_op_record_index_not_updated(self):
        """No-op record must have index_updated=False."""
        orch = self._make_orchestrator()
        record = orch.record_completion(
            activity_id="ACT-001",
            run_id="RUN-001",
            actual_duration_hours=8.0,
        )
        assert record.index_updated is False, (
            "index_updated must be False when no feedback_writer is injected"
        )

    def test_n7_no_op_record_not_persisted(self):
        """No-op record must have persisted=False."""
        orch = self._make_orchestrator()
        record = orch.record_completion(
            activity_id="ACT-001",
            run_id="RUN-001",
            actual_duration_hours=8.0,
        )
        assert record.persisted is False

    def test_n7_no_op_record_carries_validation_warning(self):
        """No-op record must carry a validation_warning describing why."""
        orch = self._make_orchestrator()
        record = orch.record_completion(
            activity_id="ACT-001",
            run_id="RUN-001",
            actual_duration_hours=8.0,
        )
        warnings = record.validation_warnings
        assert len(warnings) > 0, "validation_warnings must be non-empty for no-op path"
        assert any("feedback_writer" in str(w).lower() for w in warnings), (
            "At least one warning must mention 'feedback_writer'"
        )


# ===========================================================================
# X4 — Stage B ValueError → stub timeline + optional_failures
# ===========================================================================

class TestX4StageBValueErrorDegradation:
    """Verify that a ValueError from Stage B (unresolvable component_id) is
    caught, a stub timeline is produced, and optional_failures is populated —
    so Stages C, D, F, and G still run.

    X4 fix: _stage_b_kg_timeline() wraps build() in try/except ValueError.
    On failure it calls _stub_component_event_timeline() and appends to
    optional_failures.
    """

    def _make_orchestrator(self, *, builder_raises: bool = False):
        import sys
        import tempfile
        _ORCH_ROOT = Path(__file__).resolve().parents[2] / "src" / "dackar" / "outage"
        if str(_ORCH_ROOT) not in sys.path:
            sys.path.insert(0, str(_ORCH_ROOT))

        from orchestrators.outage_activity_orchestrator import OutageActivityOrchestrator
        from orchestrators.protocols import (
            NoOpSchemaValidator,
            FileArtifactStore,
            OutageOrchestratorConfig,
        )

        class _NoOpStage:
            def process(self, *a, **kw): return {}
            def build(self, *a, **kw):
                if builder_raises:
                    raise ValueError("cannot determine component_id: "
                                     "no resolved_component_ids and no known_component_id")
                return {}
            def score(self, *a, **kw): return {}
            def retrieve(self, *a, **kw): return {}
            def assess(self, *a, **kw): return {}
            def generate(self, *a, **kw): return {}
            def synthesize(self, *a, **kw): return {}

        stub = _NoOpStage()
        cfg = OutageOrchestratorConfig(persist_intermediate_artifacts=False)
        return OutageActivityOrchestrator(
            validator=NoOpSchemaValidator(),
            artifact_store=FileArtifactStore(tempfile.mkdtemp()),
            intake_processor=stub,
            kg_timeline_builder=stub,
            temporal_chain_scorer=stub,
            analog_retriever=stub,
            schedule_impact_assessor=stub,
            option_generator=stub,
            recommendation_synthesizer=stub,
            config=cfg,
        )

    def test_x4_value_error_produces_stub_timeline(self):
        """ValueError from build() must not propagate; stub timeline returned."""
        orch = self._make_orchestrator(builder_raises=True)
        optional_failures: list = []
        stub_result = orch._stage_b_kg_timeline(
            run_id="TEST-001",
            emergent_activity={"activity_id": "ACT-001"},
            intake_result={},
            run_context={},
            precomputed=None,
            optional_failures=optional_failures,
        )
        assert isinstance(stub_result, dict), "Stub must return a dict"
        assert stub_result.get("kg_driver_available") is False, (
            "Stub artifact must set kg_driver_available=False"
        )
        assert stub_result.get("events") == [], (
            "Stub artifact must have empty events list"
        )

    def test_x4_optional_failures_populated_on_value_error(self):
        """optional_failures must receive one entry when Stage B raises ValueError."""
        orch = self._make_orchestrator(builder_raises=True)
        optional_failures: list = []
        orch._stage_b_kg_timeline(
            run_id="TEST-001",
            emergent_activity={"activity_id": "ACT-001"},
            intake_result={},
            run_context={},
            precomputed=None,
            optional_failures=optional_failures,
        )
        assert len(optional_failures) == 1, (
            "Exactly one optional_failure entry expected for Stage B ValueError"
        )
        failure = optional_failures[0]
        assert failure["stage"] == "stage_b_kg_timeline"
        assert failure["optional"] is True

    def test_x4_stub_timeline_has_empty_recurrence_indicators(self):
        """Stub recurrence_indicators must be zero/null so Stage C degrades gracefully."""
        orch = self._make_orchestrator(builder_raises=True)
        optional_failures: list = []
        stub_result = orch._stage_b_kg_timeline(
            run_id="TEST-001",
            emergent_activity={"activity_id": "ACT-001"},
            intake_result={},
            run_context={},
            precomputed=None,
            optional_failures=optional_failures,
        )
        ri = stub_result.get("recurrence_indicators", {})
        assert ri.get("repeat_failure_count") == 0
        assert ri.get("trend") == "insufficient_data"

    def test_x4_no_value_error_when_build_succeeds(self):
        """Normal build() call must not touch optional_failures."""
        orch = self._make_orchestrator(builder_raises=False)
        optional_failures: list = []
        result = orch._stage_b_kg_timeline(
            run_id="TEST-001",
            emergent_activity={"activity_id": "ACT-001"},
            intake_result={},
            run_context={},
            precomputed=None,
            optional_failures=optional_failures,
        )
        assert optional_failures == [], "No failures should be appended on success"
        assert isinstance(result, dict)

    # ── Y7 fix: stub must produce a schema-valid ComponentEventTimeline ───────

    def _get_stub(self, run_id: str = "TEST-Y7") -> dict:
        orch = self._make_orchestrator(builder_raises=True)
        return orch._stage_b_kg_timeline(
            run_id=run_id,
            emergent_activity={"activity_id": "ACT-Y7"},
            intake_result={},
            run_context={},
            precomputed=None,
            optional_failures=[],
        )

    def test_y7_data_coverage_is_object_not_scalar(self):
        """Y7 fix: data_coverage must be an object, not 0.0."""
        stub = self._get_stub()
        dc = stub.get("data_coverage")
        assert isinstance(dc, dict), (
            f"data_coverage must be a dict, got {type(dc).__name__} (Y7 fix)"
        )
        assert "total_events" in dc

    def test_y7_no_non_schema_top_level_fields(self):
        """Y7 fix: stub must not include component_type (not in schema, not in build())."""
        stub = self._get_stub()
        assert "component_type" not in stub, (
            "'component_type' is not in schema and not set by build() (Y7 fix)"
        )

    def test_y7_recurrence_indicators_no_non_schema_fields(self):
        """Y7 fix: stub recurrence_indicators must not include last_failure_date or inter_event_period_days."""
        stub = self._get_stub()
        ri = stub.get("recurrence_indicators", {})
        assert "last_failure_date" not in ri, (
            "'last_failure_date' is not in schema — use 'last_cm_date' (Y7 fix)"
        )
        assert "inter_event_period_days" not in ri, (
            "'inter_event_period_days' is not in schema (Y7 fix)"
        )
        # schema-valid replacement fields must be present
        assert "last_cm_date" in ri
        assert "mean_inter_event_days" in ri


# ===========================================================================
# MONITOR decision status — full pipeline path (E2E-03)
# ===========================================================================

class TestMonitorScenario:
    """Full end-to-end pipeline that must produce MONITOR decision status.

    Scenario: unknown component with zero analog history and a non-critical
    schedule impact.  Three conditions that jointly trigger MONITOR in Stage G:

        1. analog_count == 0  (empty retrieval index → Stage D returns nothing)
        2. dist_tier == "low_confidence"  (zero analogs forces low_confidence tier)
        3. criticality_label == "non_critical"  (Stage E: 0 CP drag, ample float)

    A fourth guard prevents an earlier status from firing first:
        4. safety_related=True  →  defer_to_post_outage is infeasible in Stage F,
           so the primary option is insert_now (not escalate, not defer).

    Documents E2E-03 from §7.4 of deep_review_and_test_strategy.md.
    """

    @pytest.fixture(scope="class")
    def result(self):
        return run_pipeline(SCENARIO_UNKNOWN_COMPONENT)

    # ── basic structure ────────────────────────────────────────────────────────

    def test_top_level_keys(self, result):
        for key in ("scenario_label", "run_id", "intake", "timeline",
                    "temporal", "analogs", "schedule", "options", "recommendation"):
            assert key in result, f"missing top-level key '{key}'"

    def test_recommendation_keys(self, result):
        _check_artifact_keys(result, "recommendation")

    # ── MONITOR decision ───────────────────────────────────────────────────────

    def test_decision_status_is_monitor(self, result):
        """Primary assertion: this pipeline path must produce MONITOR."""
        status = result["recommendation"]["decision_status"]
        assert status == "MONITOR", (
            f"expected MONITOR for zero-analog, non-critical unknown component, got {status}"
        )

    # ── MONITOR preconditions — verify the three trigger fields ───────────────

    def test_analog_count_is_zero(self, result):
        """Precondition 1: zero analogs in the retrieval summary."""
        rs = result["analogs"].get("retrieval_summary", {})
        count = rs.get("analog_count", -1)
        assert count == 0, (
            f"MONITOR requires analog_count == 0, got {count}"
        )

    def test_confidence_tier_is_low_confidence(self, result):
        """Precondition 2: low_confidence tier from Stage D."""
        tier = result["analogs"]["duration_distribution"].get("confidence_tier")
        assert tier == "low_confidence", (
            f"MONITOR requires low_confidence tier, got '{tier}'"
        )

    def test_schedule_is_non_critical(self, result):
        """Precondition 3: non_critical schedule from Stage E."""
        label = result["schedule"].get("float_analysis", {}).get("criticality_label")
        assert label == "non_critical", (
            f"MONITOR requires non_critical schedule, got '{label}'"
        )

    # ── MONITOR guard — verify escalate/defer are not the primary option ──────

    def test_no_defer_option_is_recommended(self, result):
        """Guard: primary option must not be defer_to_post_outage (safety_related=True blocks it)."""
        rec_id = result["options"].get("recommended_option_id")
        if rec_id is not None:
            opts = {o["option_id"]: o for o in result["options"].get("options", [])}
            primary = opts.get(rec_id, {})
            assert primary.get("option_type") != "defer_to_post_outage", (
                "defer_to_post_outage must be infeasible when safety_related=True"
            )

    def test_no_critical_path_impact(self, result):
        """Guard: zero CP drag so escalate option is not generated."""
        fa = result["schedule"].get("float_analysis", {})
        assert fa.get("is_critical_path_impact") is False

    # ── analyst review flags expected on MONITOR ──────────────────────────────

    def test_analyst_review_required(self, result):
        """MONITOR path must set analyst_review.required = True."""
        ar = result["recommendation"].get("analyst_review", {})
        assert ar.get("required") is True, (
            "analyst_review.required must be True for MONITOR status"
        )

    def test_low_confidence_attention_flag_raised(self, result):
        """low_confidence_recommendation attention flag must be present on MONITOR.

        Flags live in executive_summary.analyst_attention_flags (not analyst_review).
        """
        flags = result["recommendation"].get("executive_summary", {}).get(
            "analyst_attention_flags", []
        )
        assert "low_confidence_recommendation" in flags, (
            f"expected 'low_confidence_recommendation' in analyst_attention_flags, got {flags}"
        )

    def test_low_analog_count_attention_flag_raised(self, result):
        """low_analog_count attention flag must be present when analog_count == 0.

        Flags live in executive_summary.analyst_attention_flags (not analyst_review).
        """
        flags = result["recommendation"].get("executive_summary", {}).get(
            "analyst_attention_flags", []
        )
        assert "low_analog_count" in flags, (
            f"expected 'low_analog_count' in analyst_attention_flags, got {flags}"
        )

    # ── run_id propagation ────────────────────────────────────────────────────

    def test_run_id_consistent_across_artifacts(self, result):
        run_id = result["run_id"]
        for artifact in ("intake", "timeline", "temporal", "analogs",
                         "schedule", "options", "recommendation"):
            assert result[artifact].get("run_id") == run_id, (
                f"{artifact}.run_id mismatch: expected {run_id}, "
                f"got {result[artifact].get('run_id')}"
            )
