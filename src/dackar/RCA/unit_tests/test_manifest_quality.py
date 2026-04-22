"""
test_manifest_quality.py — unit tests for run-manifest quality helpers

Run directly:   python test_manifest_quality.py
Or via pytest:  pytest test_manifest_quality.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator, OrchestratorConfig


def test_pipeline_health_green():
    h = RCAReasoningOrchestrator._compute_pipeline_health(
        output_validation={"ok": True},
        causality_candidates={"pipeline_health": {"status": "green", "issues": []}},
        evidence_bundle={"pipeline_health": {"status": "green", "issues": []}},
        optional_artifact_failures=[],
        kg_governance={"status": "green", "issues": []},
    )
    assert h["status"] == "green"
    assert h["issues"] == []


def test_pipeline_health_yellow_when_dense_only_or_optional_failure():
    h = RCAReasoningOrchestrator._compute_pipeline_health(
        output_validation={"ok": True},
        causality_candidates={"pipeline_health": {"status": "yellow", "issues": ["Candidates filtered."]}},
        evidence_bundle={"pipeline_health": {"status": "green", "issues": []}},
        optional_artifact_failures=[{"artifact": "ishikawa_matrix", "error": "x"}],
        kg_governance={"status": "green", "issues": []},
    )
    assert h["status"] == "yellow"
    assert any("optional artifacts" in i.lower() for i in h["issues"])


def test_pipeline_health_red_when_validation_fails():
    h = RCAReasoningOrchestrator._compute_pipeline_health(
        output_validation={"ok": False},
        causality_candidates={"pipeline_health": {"status": "green", "issues": []}},
        evidence_bundle={"pipeline_health": {"status": "red", "issues": ["No evidence hits were retrieved."]}},
        optional_artifact_failures=[],
        kg_governance={"status": "green", "issues": []},
    )
    assert h["status"] == "red"
    assert any("validation failed" in i.lower() for i in h["issues"])


def test_pipeline_health_yellow_when_kg_governance_warns():
    h = RCAReasoningOrchestrator._compute_pipeline_health(
        output_validation={"ok": True},
        causality_candidates={"pipeline_health": {"status": "green", "issues": []}},
        evidence_bundle={"pipeline_health": {"status": "green", "issues": []}},
        optional_artifact_failures=[],
        kg_governance={"status": "yellow", "issues": ["KG stale"]},
    )
    assert h["status"] == "yellow"
    assert any("kg stale" in i.lower() for i in h["issues"])


def test_kg_governance_detects_low_coverage_and_snapshot_mismatch():
    o = RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
    )
    block = o._compute_kg_governance(
        event={"timestamp_start": "2024-01-10T00:00:00Z", "asset_class": "default"},
        kg_context={
            "kg_snapshot_version": "neo4j:5.13|modified:2025-01-01T00:00:00+00:00",
            "failure_modes": [],
        },
    )
    assert block["status"] in {"yellow", "red"}
    assert block["too_few_failure_modes"] is True
    assert block["snapshot_newer_than_event"] is True


def test_barrier_analysis_marks_safety_barrier_degraded():
    o = RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
    )
    block = o._compute_barrier_analysis(
        event={"event_id": "EVT-1"},
        kg_context={
            "safety_functions": [
                {"sf_id": "SF::RPS", "sf_name": "Reactor Protection"},
                {"sf_id": "SF::OTHER", "sf_name": "Other"},
            ]
        },
        causality_candidates={
            "candidates": [
                {
                    "candidate_id": "FM::A",
                    "composite_score": 0.72,
                    "evidence_posture": "supported",
                    "affected_safety_functions": [{"sf_id": "SF::RPS"}],
                }
            ]
        },
        evidence_bundle={"results": []},
        ishikawa_matrix=None,
    )
    assert block["summary"]["degraded_barrier_count"] == 1
    assert block["summary"]["overall_status"] == "yellow"


def test_barrier_summary_for_card_shape():
    summary = RCAReasoningOrchestrator._barrier_summary_for_card(
        {
            "barriers": [
                {
                    "barrier_id": "SF::A",
                    "barrier_label": "Core Cooling",
                    "barrier_type": "safety_function",
                    "status": "degraded",
                }
            ],
            "summary": {"overall_status": "yellow"},
        }
    )
    assert summary["overall_status"] == "yellow"
    assert summary["degraded_barrier_count"] == 1
    assert len(summary["key_degraded_barriers"]) == 1


def test_ap913_completeness_flags():
    block = RCAReasoningOrchestrator._compute_ap913_completeness(
        rca_card={
            "primary_hypothesis": {"candidate_id": "FM::A"},
            "contributing_causes": [{"candidate_id": "FM::B"}],
            "recommended_actions": [{"action_type": "monitoring"}],
        },
        causality_candidates={"recurrence_summary": {"candidate_count_with_recurrence": 1}},
        cmms_context={"sister_components": []},
    )
    assert block["root_cause_identified"] is True
    assert block["direct_cause_identified"] is True
    assert block["contributing_causes_identified"] is True
    assert block["extent_of_condition_assessed"] is True
    assert block["effectiveness_review_defined"] is True


def test_hard_abort_policy_enabled_for_red_governance():
    o = RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
        config=OrchestratorConfig(extra={"strict_red_state_governance": True, "hard_abort_on_kg_red_state": True}),
    )
    assert o._should_hard_abort_for_kg_governance({"status": "red"}) is True
    assert o._should_hard_abort_for_kg_governance({"status": "yellow"}) is False


def test_hard_abort_policy_disabled_by_config():
    o = RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
        config=OrchestratorConfig(extra={"strict_red_state_governance": True, "hard_abort_on_kg_red_state": False}),
    )
    assert o._should_hard_abort_for_kg_governance({"status": "red"}) is False


def test_manifest_includes_reentry_execution_artifact_block():
    o = RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
    )
    manifest = o._stage_g_finalize_manifest(
        run_context={"run_id": "RUN-1", "input_refs": {"event_id": "EVT-1", "asset_id": "ASSET-1"}},
        kg_context={"subgraph_id": "KGCTX::1", "components": [], "failure_modes": []},
        tskr_patterns={"patterns": []},
        causality_candidates={"candidates": [], "provenance": {}},
        causality_candidates_pre_refine=None,
        evidence_bundle={"results": []},
        ishikawa_matrix=None,
        cmms_context=None,
        rca_card={
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
            "executive_summary": {
                "decision_status": "candidate_ready",
            },
            "primary_hypothesis": {"candidate_id": "NONE"},
            "recommended_actions": [],
            "contributing_causes": [],
        },
        input_validation={"ok": True},
        output_validation={"ok": True},
        optional_artifact_failures=[],
        kg_governance={"status": "green", "issues": [], "failure_mode_count": 0, "min_failure_modes_required": 0},
        barrier_analysis={"barriers": [], "summary": {"overall_status": "green", "barrier_count": 0, "degraded_barrier_count": 0}},
        reentry_execution={
            "auto_reentry_enabled": True,
            "attempt_count": 1,
            "attempts": [{"attempt_index": 1, "status": "completed"}],
            "reentry_hook": {"should_reenter": False, "reason": "no_rank_inversion"},
        },
        reentry_hook={"should_reenter": False, "reason": "no_rank_inversion"},
    )
    assert manifest["artifacts"]["reentry_execution"]["present"] is True
    assert manifest["artifacts"]["reentry_execution"]["attempt_count"] == 1
    assert manifest["pipeline_config"]["reentry_execution"]["attempt_count"] == 1


def test_stage_health_marks_missing_stage_outputs():
    stage_health = RCAReasoningOrchestrator._compute_stage_health(
        kg_context={"components": [], "failure_modes": [], "past_events": []},
        tskr_patterns={"patterns": []},
        causality_candidates={"candidates": [], "filtered_out_candidates": []},
        evidence_bundle={"results": [], "pipeline_health": {"issues": []}},
        ishikawa_matrix=None,
        optional_artifact_failures=[],
    )
    assert stage_health["stage_b_kg_context"]["status"] == "red"
    assert stage_health["stage_d_causality"]["status"] == "red"
    assert stage_health["stage_e_evidence"]["status"] == "red"
    assert stage_health["stage_i_archive"]["status"] == "yellow"


def test_cmms_records_inject_synthetic_past_events():
    o = RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
    )
    out = o._augment_kg_context_with_cmms_past_events(
        kg_context={"past_events": [], "seed_context": {}},
        cmms_context={
            "cr_records": [{"cr_id": "CR-1", "component_id": "CMP-A", "created_date": "2024-01-01T00:00:00Z", "status": "open"}],
            "wo_records": [{"wo_id": "WO-1", "component_id": "CMP-B", "created_date": "2024-01-02T00:00:00Z", "status": "closed"}],
        },
        event={"event_id": "EVT-1", "asset_id": "ASSET-1"},
    )
    assert len(out["past_events"]) == 2
    assert out["seed_context"]["cmms_past_events_injected"] == 2
    assert out["seed_context"]["historical_support_channels"]["mode"] == "support_channel_only"
    assert out["seed_context"]["canonical_event_graph"]["edge_count"] >= 2
    ids = {x.get("event_id") for x in out["past_events"]}
    assert "CMMS::CR::CR-1" in ids
    assert "CMMS::WO::WO-1" in ids


def test_stage_i_archive_health_red_on_archive_exception():
    class _FailingStore:
        def archive_run_scope(self, *args, **kwargs):
            raise RuntimeError("disk full")

    retriever = MagicMock()
    retriever.store = _FailingStore()
    o = RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=retriever,
        rca_synthesizer=MagicMock(),
        config=OrchestratorConfig(extra={"hard_fail_on_chroma_archive_error": True}),
    )
    block = o._stage_i_archive_chroma(run_id="RUN-1", run_context={"run_id": "RUN-1"})
    assert block["status"] == "red"
    assert block["attempted"] is True
    assert "disk full" in (block.get("error") or "")


def test_missing_archive_hook_is_yellow_not_red():
    retriever = MagicMock()
    retriever.store = object()
    o = RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=retriever,
        rca_synthesizer=MagicMock(),
        config=OrchestratorConfig(extra={"hard_fail_on_chroma_archive_error": True}),
    )
    block = o._stage_i_archive_chroma(run_id="RUN-1", run_context={"run_id": "RUN-1"})
    assert block["status"] == "yellow"
    assert block["attempted"] is False


def test_hard_abort_policy_enabled_for_chroma_archive_red_state():
    o = RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
        config=OrchestratorConfig(extra={"hard_fail_on_chroma_archive_error": True}),
    )
    assert o._should_hard_abort_for_chroma_archive({"status": "red"}) is True
    assert o._should_hard_abort_for_chroma_archive({"status": "yellow"}) is False


def test_recurrence_match_quality_attention_flag_added():
    card = {"executive_summary": {"analyst_attention_flags": []}}
    RCAReasoningOrchestrator._apply_recurrence_match_quality_attention_flags(
        rca_card=card,
        tskr_patterns={
            "summary": {
                "high_cr_match_failure_rate": True,
                "unmatched_cr_count": 4,
                "total_cr_count": 10,
                "unmatched_cr_rate": 0.4,
            }
        },
    )
    flags = card["executive_summary"]["analyst_attention_flags"]
    assert any("high cr-to-failure-mode match failure rate" in f.lower() for f in flags)


def test_workflow_dispatch_builds_queue_payload_from_next_step():
    o = RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
    )
    payload = o._build_workflow_dispatch(
        run_context={"run_id": "RUN-1", "input_refs": {"event_id": "EVT-1", "asset_id": "ASSET-1"}},
        rca_card={"executive_summary": {"decision_status": "review_required"}},
        review_hooks={"next_step": "analyst_review", "writeback_ready": False, "requires_human_review": True},
    )
    assert payload["dispatch_enabled"] is True
    assert payload["dispatched"] is True
    assert payload["target_queue"] == "rca_analyst_review_queue"


def test_workflow_dispatch_transport_sent_when_adapter_present():
    class _Adapter:
        def dispatch(self, payload):
            return {"dispatch_ref": "EXT-123"}

    o = RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
        workflow_dispatch_adapter=_Adapter(),
    )
    payload = o._build_workflow_dispatch(
        run_context={"run_id": "RUN-1", "input_refs": {"event_id": "EVT-1", "asset_id": "ASSET-1"}},
        rca_card={"executive_summary": {"decision_status": "review_required"}},
        review_hooks={"next_step": "analyst_review", "writeback_ready": False, "requires_human_review": True},
    )
    sent = o._execute_workflow_dispatch_transport(payload)
    assert sent["transport_status"] == "sent"
    assert sent["transport_ref"] == "EXT-123"


def test_cmms_documents_are_injected_into_kg_context_scope():
    out = RCAReasoningOrchestrator._augment_kg_context_with_cmms_documents(
        kg_context={"documents": [], "seed_context": {}},
        cmms_context={
            "cr_records": [{"cr_id": "CR-1", "component_id": "CMP-A"}],
            "wo_records": [{"wo_id": "WO-1", "component_id": "CMP-B"}],
        },
    )
    doc_ids = {d.get("doc_id") for d in out.get("documents", [])}
    assert "CMMS::CR::CR-1" in doc_ids
    assert "CMMS::WO::WO-1" in doc_ids
    assert out["seed_context"]["cmms_documents_injected"] == 2

