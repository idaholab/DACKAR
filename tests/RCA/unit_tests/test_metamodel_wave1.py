"""
Wave 1 metamodel compatibility tests.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator
from validation.schema_validator import RCAArtifactValidator


def _validator():
    return RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")


def test_causality_candidates_warn_when_wave1_metadata_missing():
    report = _validator().validate_artifact(
        "causality_candidates",
        {
            "event_id": "EVT-1",
            "generated_at": "2026-04-25T12:00:00Z",
            "scoring_config": {"weights": {"structural": 0.3, "temporal": 0.2, "telemetry": 0.2, "evidence": 0.2, "governance": 0.1}},
            "screening": {
                "minimum_evidence_threshold": 0.35,
                "minimum_composite_threshold": 0.3,
                "requires_both_thresholds": True,
                "top_k_candidates": 3,
                "retention_mode": "threshold_then_top_k",
            },
            "summary": {
                "generated_candidate_count": 1,
                "retained_candidate_count": 1,
                "filtered_out_candidate_count": 0,
                "top_retained_composite_score": 0.8,
                "retention_mode": "threshold_then_top_k",
            },
            "candidates": [
                {
                    "candidate_id": "FM::1",
                    "hypothesis_type": "failure_mode",
                    "cause_node_id": "FM-1",
                    "cause_label": "Bearing wear",
                    "target_event_id": "EVT-1",
                    "kg_path": [{"node_id": "N1", "node_type": "failure_mode"}],
                    "scores": {"structural": 0.8, "temporal": 0.8, "telemetry": 0.8, "evidence": 0.7, "governance": 0.6},
                    "composite_score": 0.75,
                    "meets_evidence_threshold": True,
                }
            ],
            "pipeline_health": {"status": "green", "issues": []},
        },
    )
    codes = {i.code for i in report.issues}
    assert "metamodel_compliance_missing" in codes
    assert "category_coverage_missing" in codes
    assert "applicability_assessment_missing" in codes
    assert report.ok is True


def test_invalid_ruleout_reason_code_fails_semantic_validation():
    payload = {
        "event_id": "EVT-2",
        "generated_at": "2026-04-25T12:00:00Z",
        "scoring_config": {"weights": {"structural": 0.3, "temporal": 0.2, "telemetry": 0.2, "evidence": 0.2, "governance": 0.1}},
        "screening": {
            "minimum_evidence_threshold": 0.35,
            "minimum_composite_threshold": 0.3,
            "requires_both_thresholds": True,
            "top_k_candidates": 3,
            "retention_mode": "threshold_then_top_k",
        },
        "summary": {
            "generated_candidate_count": 1,
            "retained_candidate_count": 1,
            "filtered_out_candidate_count": 0,
            "top_retained_composite_score": 0.8,
            "retention_mode": "threshold_then_top_k",
        },
        "metamodel_compliance": {"level": "partial", "version": "wave1"},
        "category_coverage": {"A": {"status": "candidate_scored", "candidate_count": 1}},
        "applicability_assessment": {"A": {"status": "applicable"}},
        "candidates": [
            {
                "candidate_id": "FM::2",
                "canonical_candidate_key": "cmp::fm::A::initiating::EVT-2",
                "primary_causal_category": "A",
                "chain_position": "initiating",
                "category_assignment_method": "deterministic",
                "category_applicability": "applicable",
                "ruleout": {"reason_code": "bad_reason"},
                "hypothesis_type": "failure_mode",
                "cause_node_id": "FM-2",
                "cause_label": "Seal leak",
                "target_event_id": "EVT-2",
                "kg_path": [{"node_id": "N1", "node_type": "failure_mode"}],
                "scores": {"structural": 0.8, "temporal": 0.8, "telemetry": 0.8, "evidence": 0.7, "governance": 0.6},
                "composite_score": 0.75,
                "meets_evidence_threshold": True,
            }
        ],
        "pipeline_health": {"status": "green", "issues": []},
    }
    report = _validator().validate_artifact("causality_candidates", payload)
    assert any(i.code == "ruleout_reason_code_invalid" for i in report.issues)
    assert report.ok is False


def test_manifest_carries_wave1_metamodel_config():
    orchestrator = RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
    )
    manifest = orchestrator._stage_g_finalize_manifest(
        run_context={"run_id": "RUN-W1", "input_refs": {"event_id": "EVT-W1", "asset_id": "ASSET-1"}},
        kg_context={"subgraph_id": "KGCTX::1", "components": [], "failure_modes": []},
        tskr_patterns={"patterns": []},
        causality_candidates={
            "candidates": [],
            "provenance": {},
            "metamodel_compliance": {"level": "partial", "version": "wave1"},
            "category_coverage": {"A": {"status": "unknown", "candidate_count": 0}},
        },
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
            "analyst_review": {"decision_required": False, "writeback_recommendation": "ready_if_accepted"},
            "executive_summary": {"decision_status": "candidate_ready", "analyst_attention_flags": []},
            "primary_hypothesis": {"candidate_id": "NONE"},
            "recommended_actions": [],
            "contributing_causes": [],
        },
        input_validation={"ok": True},
        output_validation={"ok": True},
        optional_artifact_failures=[],
        kg_governance={"status": "green", "issues": [], "failure_mode_count": 0, "min_failure_modes_required": 0},
        barrier_analysis={"barriers": [], "summary": {"overall_status": "green", "barrier_count": 0, "degraded_barrier_count": 0}},
    )
    cfg = manifest["pipeline_config"]
    assert cfg["metamodel_compliance_level"] == "partial"
    assert cfg["near_tie_delta"] == 0.05
    assert cfg["critical_stream_floor"] == 0.30
    assert cfg["oe_reinstatement_threshold"] == 0.65
    assert manifest["coverage_summary"]["category_coverage"]["A"]["status"] == "unknown"
    assert "source_families" in manifest["coverage_summary"]
