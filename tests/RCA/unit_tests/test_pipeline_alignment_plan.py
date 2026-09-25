"""
Integration-oriented tests for RCA pipeline alignment updates.

Run directly:
  python test_pipeline_alignment_plan.py
"""
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock
from types import SimpleNamespace

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.artifact_store import NoOpSchemaValidator
from orchestrators.rca_reasoning_orchestrator import (
    OrchestratorConfig,
    RCAReasoningOrchestrator,
    build_dev_orchestrator,
)
from validation.schema_validator import RCAArtifactValidator


class _MemoryArtifactStore:
    def __init__(self) -> None:
        self.saved: Dict[str, Dict[str, Any]] = {}

    def save(self, run_id: str, artifact_name: str, payload: Dict[str, Any]) -> str:
        self.saved.setdefault(run_id, {})[artifact_name] = payload
        return f"{run_id}/{artifact_name}.json"

    def save_list(self, run_id: str, artifact_name: str, payload):
        self.saved.setdefault(run_id, {})[artifact_name] = payload
        return f"{run_id}/{artifact_name}.json"


class _KGBuilder:
    client = None
    database = None

    def build(self, event, telemetry_summary, operational_context, pm_compliance, run_context, focus_component_ids=None):
        return {
            "event_id": event.get("event_id"),
            "asset_id": event.get("asset_id"),
            "subgraph_id": "KGCTX::1",
            "components": [{"component_id": "CMP-1"}],
            "failure_modes": [{"fm_id": "FM-1", "component_id": "CMP-1"}],
            "past_events": [],
            "seed_context": {},
            "documents": [],
        }


class _CausalityEngine:
    def generate(self, event, telemetry_summary, kg_context, tskr_patterns, operational_context, pm_compliance, run_context):
        return {
            "event_id": event.get("event_id"),
            "subgraph_id": kg_context.get("subgraph_id"),
            "candidates": [
                {
                    "candidate_id": "FM::FM-1",
                    "cause_node_id": "FM-1",
                    "hypothesis_type": "failure_mode",
                    "scores": {"evidence": 0.2, "governance": 0.5},
                    "composite_score": 0.6,
                    "temporal_evidence": {},
                    "evidence_posture": "supported",
                    "temporal_posture": "supported",
                    "confidence_label": "medium",
                }
            ],
            "filtered_out_candidates": [],
            "pipeline_health": {"status": "green", "issues": []},
            "provenance": {},
            "summary": {},
        }

    def refine_with_evidence(self, causality_candidates, evidence_bundle, signal_evidence=None):
        return dict(causality_candidates)


class _EvidenceRetriever:
    store = object()

    def retrieve(self, event, kg_context, causality_candidates, operational_context, run_context):
        return {
            "retrieval_scope": {"asset_id": event.get("asset_id")},
            "results": [],
            "candidate_evidence_summary": [],
            "pipeline_health": {"status": "green", "issues": []},
        }


class _Synthesizer:
    def synthesize(self, event, telemetry_summary, kg_context, tskr_patterns, causality_candidates, evidence_bundle, operational_context, pm_compliance, ishikawa_matrix, cmms_context, run_context, **kwargs):
        return {
            "event_id": event.get("event_id"),
            "asset_id": event.get("asset_id"),
            "executive_summary": {"decision_status": "candidate_ready", "analyst_attention_flags": []},
            "primary_hypothesis": {
                "candidate_id": "FM::FM-1",
                "cause_label": "FM-1",
                "confidence_label": "medium",
            },
            "validation_status": {
                "schema_valid": True,
                "all_claims_cited": True,
                "passed_minimum_evidence_gate": True,
                "fallback_used": False,
            },
            "analyst_review": {
                "decision_required": False,
                "writeback_recommendation": "ready_if_accepted",
            },
            "recommended_actions": [],
            "contributing_causes": [],
        }


def _make_orchestrator(extra: Optional[Dict[str, Any]] = None) -> RCAReasoningOrchestrator:
    cfg = OrchestratorConfig(
        enable_ishikawa=False,
        persist_intermediate_artifacts=True,
        stop_on_validation_error=False,
        extra={
            "strict_red_state_governance": False,
            "hard_abort_on_kg_red_state": False,
            "enable_chroma_archive_stage": False,
            "hard_fail_on_chroma_archive_error": False,
            "causality_engine_version": "v32",
            **(extra or {}),
        },
    )
    return RCAReasoningOrchestrator(
        validator=NoOpSchemaValidator(),
        artifact_store=_MemoryArtifactStore(),
        kg_context_builder=_KGBuilder(),
        tskr_temporal_scorer=None,
        causality_engine=_CausalityEngine(),
        evidence_retriever=_EvidenceRetriever(),
        rca_synthesizer=_Synthesizer(),
        config=cfg,
    )


class _TSKRScorer:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            simultaneous_epsilon_hours=0.5,
            min_confidence_for_support=0.35,
        )

    def score(self, event, telemetry_summary, kg_context, operational_context, run_context, signal_evidence=None):
        return {
            "event_id": event.get("event_id"),
            "asset_id": event.get("asset_id"),
            "patterns": [],
            "summary": {"has_temporal_support": False, "mode": "deterministic_v1"},
            "provenance": {"generated_by": "test_tskr", "generated_at": "2026-01-01T12:00:00+00:00"},
        }


def test_pm_auto_build_path_and_override():
    event = {
        "event_id": "EVT-1",
        "asset_id": "ASSET-1",
        "timestamp_start": "2026-01-01T12:00:00+00:00",
    }
    telemetry = {"asset_id": "ASSET-1", "signals": []}
    export_rows = [
        {
            "asset_id": "ASSET-1",
            "check_id": "CHK-1",
            "task_code": "PM-1",
            "check_type": "lubrication",
            "component_id": "CMP-1",
            "scheduled_date": "2025-12-20T00:00:00+00:00",
            "completed_date": "2025-12-20T12:00:00+00:00",
            "compliance_status": "compliant",
            "applicable_fm_ids": ["FM-1"],
        }
    ]

    o = _make_orchestrator()
    result = o.run(
        event=event,
        telemetry_summary=telemetry,
        operational_context={"pm_export_rows": export_rows},
    )
    assert result["pm_compliance"] is not None
    assert result["run_context"]["pipeline_runtime"]["pm_compliance"]["source"] == "auto_built"
    assert result["run_context"]["input_refs"]["has_pm_compliance"] is True

    explicit_pm = {
        "asset_id": "ASSET-1",
        "window": {"start": "2025-01-01T00:00:00+00:00", "end": "2026-01-01T12:00:00+00:00"},
        "checks": [],
        "summary": {"total_checks": 0, "passed": 0, "failed": 0, "unknown": 0, "overdue_count": 0},
    }
    o2 = _make_orchestrator()
    result2 = o2.run(
        event=event,
        telemetry_summary=telemetry,
        operational_context={"pm_export_rows": export_rows},
        pm_compliance=explicit_pm,
    )
    assert result2["pm_compliance"] == explicit_pm
    assert result2["run_context"]["pipeline_runtime"]["pm_compliance"]["source"] == "provided"
    print("  PASS test_pm_auto_build_path_and_override")


def test_build_dev_orchestrator_defaults_to_v32():
    with tempfile.TemporaryDirectory() as td:
        o = build_dev_orchestrator(output_dir=td, client=MagicMock())
    assert o.config.extra.get("causality_engine_version") == "v32"
    assert type(o.causality_engine).__name__ == "RuleBasedCausalityEngineV32"
    print("  PASS test_build_dev_orchestrator_defaults_to_v32")


def test_validator_extended_bundle_and_semantics():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    signal_evidence = {
        "run_id": "RUN-1",
        "augmented_anomaly_set": [],
        "propagation_chains": [],
        "per_candidate_chain_score": {},
        "chain_coverage": 0.0,
        "augmented_anomaly_count": 0,
        "historian_anomaly_count": 0,
    }
    barrier = {
        "analysis_id": "BARR-1",
        "event_id": "EVT-1",
        "generated_at": "2026-01-01T12:00:00+00:00",
        "barriers": [],
        "summary": {"overall_status": "green", "barrier_count": 0, "degraded_barrier_count": 0},
        "provenance": {"generated_by": "test"},
    }
    report = validator.validate_run_bundle(
        event={
            "event_id": "EVT-1",
            "asset_id": "ASSET-1",
            "timestamp_start": "2026-01-01T12:00:00+00:00",
            "timestamp_end": "2026-01-01T12:05:00+00:00",
            "severity": "HIGH",
            "event_type": "FAILURE",
            "symptom_signature": {"description": "reactor trip"},
        },
        signal_evidence=signal_evidence,
        barrier_analysis=barrier,
        cmms_context={
            "cmms_context_id": "CMMSCTX::1",
            "run_id": "RUN-1",
            "event_id": "EVT-1",
            "asset_id": "ASSET-1",
            "generated_at": "2026-01-01T12:00:00+00:00",
            "adapter": "TestAdapter",
            "lookback_anchor": "custom",
            "lookback_from": "2025-01-01T00:00:00+00:00",
            "lookback_to": "2026-01-01T12:00:00+00:00",
            "cr_records": [],
            "wo_records": [],
            "recurrence_summary": {
                "cr_count_primary": 0,
                "cr_count_sister": 0,
                "open_wo_count": 0,
                "open_cr_count": 0,
            },
            "provenance": {"generated_by": "test"},
        },
    )
    assert report.ok is True

    bad_signal = dict(signal_evidence)
    bad_signal["augmented_anomaly_set"] = [{"sensor_id": "S1", "timestamp_start": "2026-01-01T00:00:00+00:00", "timestamp_end": "2026-01-01T00:10:00+00:00", "pattern": "spike", "severity": 0.4, "source": "telemetry_summary"}]
    bad_signal["augmented_anomaly_count"] = 0
    r2 = validator.validate_artifact("signal_evidence", bad_signal)
    assert r2.ok is False
    assert any(i.code == "augmented_anomaly_count_mismatch" for i in r2.issues)
    print("  PASS test_validator_extended_bundle_and_semantics")


def test_validator_flags_unknown_recommended_action_target_component():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    report = validator.validate_run_bundle(
        event={
            "event_id": "EVT-2",
            "asset_id": "ASSET-2",
            "timestamp_start": "2026-01-01T12:00:00+00:00",
            "timestamp_end": "2026-01-01T12:05:00+00:00",
            "severity": "HIGH",
            "event_type": "FAILURE",
            "symptom_signature": {"description": "pump trip"},
        },
        kg_context={
            "event_id": "EVT-2",
            "asset_id": "ASSET-2",
            "subgraph_id": "KGCTX::2",
            "components": [{"component_id": "CMP-VALID"}],
            "failure_modes": [],
            "past_events": [],
            "seed_context": {},
            "documents": [],
        },
        rca_card={
            "rca_id": "RCA::EVT-2::1",
            "event_id": "EVT-2",
            "generated_at": "2026-01-01T12:10:00+00:00",
            "llm_model": "stub",
            "input_artifacts": {"event_id": "EVT-2", "bundle_id": "B-2", "candidates_ref": "RUN-2"},
            "validation_status": {
                "schema_valid": True,
                "all_claims_cited": True,
                "passed_minimum_evidence_gate": True,
                "validation_errors": [],
                "retry_count": 0,
                "fallback_used": False,
            },
            "executive_summary": {
                "decision_status": "review_required",
                "primary_conclusion": "Needs analyst review",
                "confidence_label": "medium",
                "analyst_attention_flags": [],
            },
            "primary_hypothesis": {
                "candidate_id": "NONE",
                "cause_label": "No supported hypothesis",
                "hypothesis_type": "external_cause",
                "fm_id": None,
                "narrative": "n",
                "why_primary": ["w"],
                "uncertainties": ["u"],
                "composite_score": 0.0,
                "confidence_label": "speculative",
                "citations": [],
            },
            "contributing_causes": [],
            "alternatives": [],
            "evidence": [],
            "recommended_actions": [
                {
                    "action_id": "ACT-1",
                    "action_type": "immediate_corrective",
                    "description": "Inspect sister component",
                    "priority": "high",
                    "target_component_id": "CMP-UNKNOWN",
                }
            ],
            "analyst_review": {
                "decision_required": True,
                "questions_to_resolve": ["Confirm target component identity."],
                "writeback_recommendation": "hold_until_review",
            },
            "provenance": {
                "source_bundle_id": "B-2",
                "pipeline_version": "rca_orchestrator_v3_1",
                "generated_by": "unit_test",
                "card_version": 1,
            },
        },
    )
    assert any(
        i.code == "recommended_action_target_component_not_in_kg_context"
        for i in report.issues
    )
    print("  PASS test_validator_flags_unknown_recommended_action_target_component")


def test_out_of_boundary_anomalies_surface_analyst_attention_flags():
    rca_card = {"executive_summary": {"analyst_attention_flags": []}}
    kg_context = {
        "out_of_boundary_anomalies": [
            {"sensor_id": "S-1", "component_id": "C-OUT-1", "allen_relation": "precedes", "not_in_kg": False},
            {"sensor_id": "S-2", "component_id": None, "allen_relation": "precedes", "not_in_kg": True},
        ]
    }
    RCAReasoningOrchestrator._apply_out_of_boundary_attention_flags(rca_card, kg_context)
    flags = (rca_card.get("executive_summary") or {}).get("analyst_attention_flags") or []
    assert any("out-of-boundary anomaly signal" in f.lower() for f in flags)
    assert any("not_in_kg=true" in f for f in flags)
    print("  PASS test_out_of_boundary_anomalies_surface_analyst_attention_flags")


def test_tskr_epsilon_runtime_override_and_manifest_snapshot():
    event = {
        "event_id": "EVT-3",
        "asset_id": "ASSET-3",
        "timestamp_start": "2026-01-01T12:00:00+00:00",
    }
    telemetry = {"asset_id": "ASSET-3", "signals": []}
    scorer = _TSKRScorer()
    cfg = OrchestratorConfig(
        enable_ishikawa=False,
        persist_intermediate_artifacts=True,
        stop_on_validation_error=False,
        extra={
            "strict_red_state_governance": False,
            "hard_abort_on_kg_red_state": False,
            "enable_chroma_archive_stage": False,
            "hard_fail_on_chroma_archive_error": False,
            "causality_engine_version": "v32",
            "tskr_simultaneous_epsilon_hours": 1.25,
        },
    )
    o = RCAReasoningOrchestrator(
        validator=NoOpSchemaValidator(),
        artifact_store=_MemoryArtifactStore(),
        kg_context_builder=_KGBuilder(),
        tskr_temporal_scorer=scorer,
        causality_engine=_CausalityEngine(),
        evidence_retriever=_EvidenceRetriever(),
        rca_synthesizer=_Synthesizer(),
        config=cfg,
    )
    result = o.run(
        event=event,
        telemetry_summary=telemetry,
        operational_context={},
    )
    assert abs(float(scorer.config.simultaneous_epsilon_hours) - 1.25) < 1e-9
    tskr_runtime = ((result.get("run_manifest") or {}).get("pipeline_config") or {}).get("tskr_runtime") or {}
    assert abs(float(tskr_runtime.get("simultaneous_epsilon_hours")) - 1.25) < 1e-9
    print("  PASS test_tskr_epsilon_runtime_override_and_manifest_snapshot")


def test_run_context_initializes_scope_revision_lifecycle():
    event = {
        "event_id": "EVT-S0-1",
        "asset_id": "ASSET-S0-1",
        "component_id": "CMP-S0-1",
        "timestamp_start": "2026-01-01T12:00:00+00:00",
        "timestamp_end": "2026-01-01T12:10:00+00:00",
        "severity": "HIGH",
        "event_type": "FAILURE",
        "actuation_type": "anomalous",
        "trigger_source": "alarm",
    }
    telemetry = {"asset_id": "ASSET-S0-1", "signals": []}
    operational_context = {
        "asset_id": "ASSET-S0-1",
        "mode": "steady",
        "percent_rated_power": 98.5,
        "train_configuration": {"train_id": "Train-A", "in_service": True},
        "recent_alarms": [
            {"alarm_id": "A-1", "system_affected": "RCS", "timestamp": "2026-01-01T12:00:00+00:00", "priority": "high"},
            {"alarm_id": "A-2", "system_affected": "AuxFeedwater", "timestamp": "2026-01-01T12:01:00+00:00", "priority": "medium"},
        ]
    }
    o = _make_orchestrator()
    result = o.run(
        event=event,
        telemetry_summary=telemetry,
        operational_context=operational_context,
    )
    scope = (result.get("run_context") or {}).get("scope_management") or {}
    input_refs = (result.get("run_context") or {}).get("input_refs") or {}
    assert scope.get("active_scope_version") == 0
    assert input_refs.get("active_scope_version") == 0
    # event fields surfaced in input_refs
    assert input_refs.get("event_severity") == "HIGH"
    assert input_refs.get("event_type") == "FAILURE"
    assert input_refs.get("actuation_type") == "anomalous"
    assert input_refs.get("trigger_source") == "alarm"
    assert input_refs.get("has_operational_context") is True
    assert input_refs.get("has_soe_log") is False
    assert input_refs.get("has_alarm_log") is False
    assert input_refs.get("has_protection_logic_context") is False
    revisions = scope.get("scope_revisions") or []
    assert len(revisions) == 1
    first = revisions[0]
    assert first.get("trigger") == "initial_intake"
    assert first.get("analyst_decision") == "accepted"
    snapshot = first.get("scope_snapshot") or {}
    assert snapshot.get("asset_ids") == ["ASSET-S0-1"]
    assert "CMP-S0-1" in (snapshot.get("component_ids") or [])
    assert "RCS" in (snapshot.get("system_boundary") or [])
    # operating context captured
    op_ctx = snapshot.get("operating_context") or {}
    assert op_ctx.get("mode") == "steady"
    assert abs(float(op_ctx.get("percent_rated_power") or 0) - 98.5) < 0.01
    assert op_ctx.get("train_id") == "Train-A"
    assert op_ctx.get("train_in_service") is True
    # event context captured
    ev_ctx = snapshot.get("event_context") or {}
    assert ev_ctx.get("severity") == "HIGH"
    assert ev_ctx.get("event_type") == "FAILURE"
    assert ev_ctx.get("actuation_type") == "anomalous"
    # data availability flags
    da = snapshot.get("data_availability") or {}
    assert da.get("has_operational_context") is True
    assert da.get("has_soe_log") is False
    manifest_scope = ((result.get("run_manifest") or {}).get("scope_revision_summary") or {})
    assert manifest_scope.get("active_scope_version") == 0
    assert manifest_scope.get("revision_count") == 1
    assert manifest_scope.get("latest_analyst_decision") == "accepted"
    print("  PASS test_run_context_initializes_scope_revision_lifecycle")


def test_run_context_captures_alarm_log_and_soe_system_boundary():
    event = {
        "event_id": "EVT-S0-3",
        "asset_id": "ASSET-S0-3",
        "timestamp_start": "2026-01-01T12:00:00+00:00",
        "severity": "CRITICAL",
        "event_type": "FAILURE",
    }
    telemetry = {"asset_id": "ASSET-S0-3", "signals": []}
    alarm_log = {
        "alarm_log_id": "ALM-1",
        "event_id": "EVT-S0-3",
        "asset_id": "ASSET-S0-3",
        "generated_at": "2026-01-01T12:00:00+00:00",
        "window": {"start": "2026-01-01T11:50:00+00:00", "end": "2026-01-01T12:00:00+00:00"},
        "alarms": [
            {"alarm_id": "ALM-001", "timestamp": "2026-01-01T11:55:00+00:00",
             "priority": "critical", "state": "active", "system": "ECCS"},
            {"alarm_id": "ALM-002", "timestamp": "2026-01-01T11:57:00+00:00",
             "priority": "high", "state": "active", "system": "ReactorCoolant"},
        ],
        "provenance": {"generated_by": "test"},
    }
    soe_log = {
        "soe_id": "SOE-1",
        "event_id": "EVT-S0-3",
        "asset_id": "ASSET-S0-3",
        "generated_at": "2026-01-01T12:00:00+00:00",
        "window": {"start": "2026-01-01T11:50:00+00:00", "end": "2026-01-01T12:00:00+00:00"},
        "records": [
            {"record_id": "R1", "timestamp": "2026-01-01T11:55:00+00:00",
             "signal_id": "SIG-1", "transition": "trip", "component_id": "CMP-RCP-1"},
            {"record_id": "R2", "timestamp": "2026-01-01T11:56:00+00:00",
             "signal_id": "SIG-2", "transition": "assert", "component_id": "CMP-HIS-1"},
        ],
        "provenance": {"generated_by": "test"},
    }
    o = _make_orchestrator()
    # Call _stage_a_build_run_context directly (not run()) to test the new params
    import uuid
    run_id = str(uuid.uuid4())
    rc = o._stage_a_build_run_context(
        run_id=run_id,
        event=event,
        telemetry_summary=telemetry,
        operational_context=None,
        pm_compliance=None,
        alarm_log=alarm_log,
        soe_log=soe_log,
    )
    snapshot = rc["scope_management"]["scope_revisions"][0]["scope_snapshot"]
    assert "ECCS" in snapshot["system_boundary"]
    assert "ReactorCoolant" in snapshot["system_boundary"]
    assert "CMP-RCP-1" in snapshot["component_ids"]
    assert "CMP-HIS-1" in snapshot["component_ids"]
    assert rc["input_refs"]["has_alarm_log"] is True
    assert rc["input_refs"]["has_soe_log"] is True
    assert rc["input_refs"]["has_protection_logic_context"] is False
    da = snapshot["data_availability"]
    assert da["has_alarm_log"] is True
    assert da["has_soe_log"] is True
    assert da["has_protection_logic_context"] is False
    print("  PASS test_run_context_captures_alarm_log_and_soe_system_boundary")


def test_apply_scope_revision_tracks_accepted_and_rejected_revisions():
    o = _make_orchestrator()
    run_id = "RUN-S0-2"
    run_context = {
        "run_id": run_id,
        "input_refs": {"event_id": "EVT-S0-2"},
        "scope_management": {
            "active_scope_version": 0,
            "latest_approved_revision_id": "SCOPE::EVT-S0-2::0",
            "scope_revisions": [
                {
                    "revision_id": "SCOPE::EVT-S0-2::0",
                    "scope_version": 0,
                    "trigger": "initial_intake",
                    "changed_boundary": {"window_delta": "initial"},
                    "analyst_decision": "accepted",
                    "decision_timestamp": "2026-01-01T00:00:00+00:00",
                    "scope_snapshot": {"asset_ids": ["ASSET-S0-2"]},
                }
            ],
        },
    }
    accepted = o.apply_scope_revision(
        run_id=run_id,
        run_context=run_context,
        revision_input={
            "trigger": "candidate_coverage_gap",
            "analyst_decision": "accepted",
            "changed_boundary": {"added_component_ids": ["CMP-2"]},
            "scope_snapshot": {"asset_ids": ["ASSET-S0-2"], "component_ids": ["CMP-2"]},
        },
        persist=False,
    )
    scope1 = accepted["scope_management"]
    refs1 = accepted["input_refs"]
    assert scope1["active_scope_version"] == 1
    assert refs1["active_scope_version"] == 1
    assert len(scope1["scope_revisions"]) == 2
    assert scope1["scope_revisions"][-1]["analyst_decision"] == "accepted"

    rejected = o.apply_scope_revision(
        run_id=run_id,
        run_context=accepted,
        revision_input={
            "trigger": "analyst_override",
            "analyst_decision": "rejected",
            "changed_boundary": {"removed_component_ids": ["CMP-2"]},
        },
        persist=False,
    )
    scope2 = rejected["scope_management"]
    assert scope2["active_scope_version"] == 1
    assert len(scope2["scope_revisions"]) == 3
    assert scope2["scope_revisions"][-1]["analyst_decision"] == "rejected"
    print("  PASS test_apply_scope_revision_tracks_accepted_and_rejected_revisions")


ALL_TESTS = [
    test_pm_auto_build_path_and_override,
    test_build_dev_orchestrator_defaults_to_v32,
    test_validator_extended_bundle_and_semantics,
    test_validator_flags_unknown_recommended_action_target_component,
    test_out_of_boundary_anomalies_surface_analyst_attention_flags,
    test_tskr_epsilon_runtime_override_and_manifest_snapshot,
    test_run_context_initializes_scope_revision_lifecycle,
    test_run_context_captures_alarm_log_and_soe_system_boundary,
    test_apply_scope_revision_tracks_accepted_and_rejected_revisions,
]


def run_all() -> bool:
    print(f"\n=== test_pipeline_alignment_plan ({len(ALL_TESTS)} tests) ===")
    passed = 0
    failed = 0
    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
            import traceback
            print(f"  FAIL {fn.__name__}: {exc}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    raise SystemExit(0 if ok else 1)
