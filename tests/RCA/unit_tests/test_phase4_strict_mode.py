"""
Phase 4 strict-mode validation tests.
"""
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from validation.schema_validator import RCAArtifactValidator


def _base_payload():
    payload = {
        "event_id": "EVT-STRICT-1",
        "generated_at": "2026-04-25T12:00:00Z",
        "metamodel_compliance": {"level": "full", "version": "wave4"},
        "category_coverage": {
            k: {"status": "ruled_out", "candidate_count": 0, "rationale": "No supporting data."}
            for k in "ABCDEFGHIJKL"
        },
        "applicability_assessment": {
            k: {"status": "unknown", "rationale": "Insufficient boundary evidence."}
            for k in "ABCDEFGHIJKL"
        },
        "applicability_summary": {"applicable": 0, "not_applicable": 0, "unknown": 12},
        "uncertainty_summary": {"candidate_count": 1, "average_quality_multiplier": 0.6},
        "decision_posture": {
            "recommended_decision_status": "review_required",
            "near_tie": False,
            "contradiction_blocked_count": 0,
            "eligible_primary_candidate_ids": ["C1"],
            "blocked_candidate_ids": [],
        },
        "scoring_config": {
            "weights": {
                "structural": 0.3,
                "temporal": 0.2,
                "telemetry": 0.2,
                "evidence": 0.2,
                "governance": 0.1,
            }
        },
        "screening": {
            "minimum_evidence_threshold": 0.35,
            "minimum_composite_threshold": 0.30,
            "requires_both_thresholds": True,
            "top_k_candidates": 5,
            "retention_mode": "threshold_then_top_k",
        },
        "summary": {
            "generated_candidate_count": 1,
            "retained_candidate_count": 1,
            "filtered_out_candidate_count": 0,
            "top_retained_composite_score": 0.66,
            "retention_mode": "threshold_then_top_k",
        },
        "candidates": [
            {
                "candidate_id": "C1",
                "canonical_tuple": {
                    "component": "CMP-1",
                    "failure_mode": "FM-1",
                    "causal_category": "A",
                    "chain_position": "initiating",
                },
                "canonical_candidate_key": "CMP-1::FM-1::A::initiating::EVT-STRICT-1",
                "component_id": "CMP-1",
                "failure_mode_id": "FM-1",
                "primary_causal_category": "A",
                "chain_position": "initiating",
                "chain_position_confidence": 0.7,
                "chain_position_rationale": "Precedes trigger",
                "event_scope_id": "EVT-STRICT-1",
                "category_assignment_method": "deterministic",
                "category_assignment_confidence": 0.8,
                "category_applicability": "unknown",
                "hypothesis_type": "failure_mode",
                "cause_node_id": "FM-1",
                "cause_label": "Bearing wear",
                "target_event_id": "EVT-STRICT-1",
                "kg_path": [{"node_id": "N1", "node_type": "failure_mode"}],
                "scores": {
                    "structural": 0.8,
                    "temporal": 0.6,
                    "telemetry": 0.5,
                    "evidence": 0.4,
                    "governance": 0.3,
                },
                "composite_score": 0.66,
                "meets_evidence_threshold": True,
                "stream_quality": {"temporal": 0.7, "logical": 0.6, "documentary": 0.5, "oe": 0.4},
                "quality_multiplier": 0.6,
                "primary_eligibility": "eligible",
                "primary_block_reasons": [],
                "reinstatement_status": "none",
                "hard_gates": {
                    "physical_plausibility": {
                        "passed": True,
                        "rationale": "PASS: structural=0.800; component='CMP-1'; failure_mode='FM-1'.",
                        "gate_order": 1,
                    },
                    "timeline_consistency": {
                        "passed": True,
                        "degraded_mode": False,
                        "rationale": "PASS: latency_violation_type=none; observed_lag_hours=1.0; expected_range=(0.5, 2.0).",
                        "gate_order": 2,
                    },
                    "barrier_logic": {
                        "passed": True,
                        "degraded_mode": True,
                        "rationale": "PASS (degraded): barrier/protection inputs unavailable.",
                        "gate_order": 3,
                    },
                },
            }
        ],
        "pipeline_health": {"status": "green", "issues": []},
    }
    payload["category_coverage"]["A"] = {
        "status": "candidate_scored",
        "candidate_count": 1,
        "rationale": "Category represented by retained candidate.",
    }
    return payload


def test_full_mode_missing_required_candidate_field_fails():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_payload()
    del payload["candidates"][0]["chain_position_rationale"]
    report = validator.validate_artifact("causality_candidates", payload)
    assert any(i.code == "full_mode_candidate_field_missing" for i in report.issues)
    assert report.ok is False


def test_full_mode_payload_passes_strict_semantic_requirements():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    report = validator.validate_artifact("causality_candidates", _base_payload())
    assert report.ok is True, [i.to_dict() for i in report.issues if i.severity == "error"]


def test_full_mode_candidate_canonical_tuple_mismatch_fails():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_payload()
    payload["candidates"][0]["canonical_tuple"]["causal_category"] = "B"
    report = validator.validate_artifact("causality_candidates", payload)
    assert any(i.code == "canonical_tuple_category_mismatch" for i in report.issues)
    assert report.ok is False


def test_full_mode_missing_timeline_gate_fails():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_payload()
    del payload["candidates"][0]["hard_gates"]["timeline_consistency"]
    report = validator.validate_artifact("causality_candidates", payload)
    assert any(i.code == "timeline_consistency_gate_missing" for i in report.issues)
    assert report.ok is False


def test_full_mode_missing_barrier_gate_fails():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_payload()
    del payload["candidates"][0]["hard_gates"]["barrier_logic"]
    report = validator.validate_artifact("causality_candidates", payload)
    assert any(i.code == "barrier_logic_gate_missing" for i in report.issues)
    assert report.ok is False


def test_full_mode_applicable_category_requires_candidate_or_ruleout():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_payload()
    payload["applicability_assessment"]["A"]["status"] = "applicable"
    payload["category_coverage"]["A"]["status"] = "unknown"
    report = validator.validate_artifact("causality_candidates", payload)
    assert any(
        i.code in {
            "category_coverage_unknown_forbidden_full_mode",
            "category_coverage_applicable_status_invalid_full_mode",
        }
        for i in report.issues
    )
    assert report.ok is False


def test_full_mode_requires_all_category_coverage_rows():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_payload()
    del payload["category_coverage"]["B"]
    report = validator.validate_artifact("causality_candidates", payload)
    assert any(i.code == "category_coverage_category_missing_full_mode" for i in report.issues)
    assert report.ok is False


def _base_run_manifest_full():
    return {
        "run_id": "RUN-STRICT-1",
        "completed_at": "2026-04-25T12:00:00Z",
        "input_refs": {"event_id": "EVT-STRICT-1", "asset_id": "ASSET-1"},
        "pipeline_config": {
            "metamodel_compliance_level": "full",
            "metamodel_migration": {"phase": "wave4", "compatibility_mode": False},
        },
        "artifacts": {},
        "review_hooks": {
            "writeback_ready": False,
            "next_step": "analyst_review",
            "coverage_status": "partial",
            "coverage_degraded": True,
            "coverage_acknowledgement_required": True,
            "coverage_acknowledged": False,
        },
        "pipeline_health": {"status": "green", "issues": []},
        "ap913_completeness": {
            "root_cause_identified": True,
            "direct_cause_identified": True,
            "contributing_causes_identified": True,
            "extent_of_condition_assessed": True,
            "effectiveness_review_defined": True,
        },
        "coverage_summary": {
            "overall_status": "partial",
            "source_families": {
                "kg_context": {"status": "complete", "metrics": {"component_count": 1, "failure_mode_count": 1}},
                "chroma_corpus": {"status": "partial", "metrics": {"evidence_result_count": 1}},
                "upstream_anomaly_inputs": {"status": "missing", "metrics": {"pattern_count": 0}},
            },
            "category_coverage": {},
        },
        "applicability_summary": {},
        "uncertainty_summary": {
            "candidate_count": 1,
            "average_quality_multiplier": 0.55,
            "data_limited_candidate_count": 1,
            "average_coverage_quality_factor": 0.85,
            "coverage_degraded_candidate_count": 1,
            "coverage_flagged_source_families": ["chroma_corpus", "upstream_anomaly_inputs"],
            "critical_stream_floor": 0.30,
        },
        "decision_posture": {},
        "replayability_signature": {
            "algorithm": "sha256",
            "digest": "a" * 64,
            "candidate_count": 1,
            "canonical_payload_version": "v1",
        },
        "analyst_checkpoints": [
            {"step_id": "0", "step_name": "scoping", "status": "completed", "decision_required": False, "decision_state": "completed"},
            {"step_id": "1", "step_name": "data_management", "status": "completed", "decision_required": False, "decision_state": "completed"},
            {"step_id": "2", "step_name": "kg_expansion", "status": "completed", "decision_required": False, "decision_state": "completed"},
            {"step_id": "3", "step_name": "pattern_recognition_documentary", "status": "completed", "decision_required": False, "decision_state": "completed"},
            {"step_id": "3.5", "step_name": "pattern_recognition_signal", "status": "completed", "decision_required": False, "decision_state": "completed"},
            {"step_id": "4", "step_name": "candidate_generation", "status": "completed", "decision_required": False, "decision_state": "completed"},
            {"step_id": "5", "step_name": "ranking_and_evidence_assessment", "status": "completed", "decision_required": True, "decision_state": "hold_until_review"},
            {"step_id": "6", "step_name": "conclusion", "status": "completed", "decision_required": True, "decision_state": "hold_until_review"},
        ],
        "decision_trail": [
            {
                "event_type": "final_decision",
                "candidate_id": "C1",
                "decision_status": "review_required",
                "confidence_label": "medium",
            }
        ],
    }


def test_full_mode_run_manifest_missing_checkpoints_fails():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_run_manifest_full()
    del payload["analyst_checkpoints"]
    report = validator.validate_artifact("run_manifest", payload)
    assert any(i.code == "run_manifest_analyst_checkpoints_missing" for i in report.issues)
    assert report.ok is False


def test_full_mode_run_manifest_missing_source_family_coverage_fails():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_run_manifest_full()
    del payload["coverage_summary"]["source_families"]
    report = validator.validate_artifact("run_manifest", payload)
    assert any(i.code == "run_manifest_coverage_summary_source_families_missing" for i in report.issues)
    assert report.ok is False


def test_full_mode_run_manifest_missing_replayability_signature_fails():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_run_manifest_full()
    del payload["replayability_signature"]
    report = validator.validate_artifact("run_manifest", payload)
    assert any(i.code == "run_manifest_full_mode_field_missing" and i.path == ["replayability_signature"] for i in report.issues)
    assert report.ok is False


def test_full_mode_run_manifest_degraded_coverage_requires_ack_for_writeback():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_run_manifest_full()
    payload["review_hooks"]["writeback_ready"] = True
    payload["review_hooks"]["coverage_acknowledgement_required"] = False
    payload["review_hooks"]["coverage_acknowledged"] = False
    report = validator.validate_artifact("run_manifest", payload)
    assert any(i.code == "run_manifest_coverage_acknowledgement_missing_for_writeback" for i in report.issues)
    assert report.ok is False


def test_full_mode_run_manifest_degraded_coverage_requires_uncertainty_coverage_signals():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_run_manifest_full()
    payload["uncertainty_summary"] = {"candidate_count": 1}
    report = validator.validate_artifact("run_manifest", payload)
    assert any(i.code == "run_manifest_uncertainty_coverage_factor_missing" for i in report.issues)
    assert any(i.code == "run_manifest_uncertainty_coverage_degraded_count_invalid" for i in report.issues)
    assert any(i.code == "run_manifest_uncertainty_coverage_flags_missing" for i in report.issues)
    assert report.ok is False


def test_full_mode_run_manifest_degraded_coverage_requires_review_hook_degraded_flag():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_run_manifest_full()
    payload["review_hooks"]["coverage_degraded"] = False
    payload["review_hooks"]["coverage_status"] = "complete"
    report = validator.validate_artifact("run_manifest", payload)
    assert any(i.code == "run_manifest_coverage_degraded_flag_missing" for i in report.issues)
    assert any(i.code == "run_manifest_coverage_status_mismatch" for i in report.issues)
    assert report.ok is False


def test_full_mode_run_manifest_missing_decision_trail_fails():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_run_manifest_full()
    payload["decision_trail"] = []
    report = validator.validate_artifact("run_manifest", payload)
    assert any(i.code == "run_manifest_decision_trail_empty" for i in report.issues)
    assert report.ok is False


def test_full_mode_run_manifest_invalid_decision_trail_event_type_fails():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_run_manifest_full()
    payload["decision_trail"] = [
        {"event_type": "unknown_event", "candidate_id": "C1"}
    ]
    report = validator.validate_artifact("run_manifest", payload)
    assert any(i.code == "run_manifest_decision_trail_event_type_invalid" for i in report.issues)
    assert report.ok is False


def test_full_mode_run_manifest_ruleout_entry_requires_fields():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_run_manifest_full()
    payload["decision_trail"] = [
        {"event_type": "ruleout", "candidate_id": "C1", "reason_code": "no_supporting_data"}
    ]
    report = validator.validate_artifact("run_manifest", payload)
    assert any(i.code == "run_manifest_decision_trail_ruleout_field_missing" for i in report.issues)
    assert report.ok is False


def test_full_mode_run_manifest_reinstatement_entry_requires_rationale_refs_and_timestamp():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_run_manifest_full()
    payload["decision_trail"] = [
        {
            "event_type": "reinstatement_status",
            "candidate_id": "C1",
            "status": "reinstated_by_analyst",
            "reason_detail": "",
            "evidence_refs": [],
            "reinstated_at": "",
        }
    ]
    report = validator.validate_artifact("run_manifest", payload)
    assert any(i.code == "run_manifest_decision_trail_reinstatement_reason_missing" for i in report.issues)
    assert any(i.code == "run_manifest_decision_trail_reinstatement_evidence_refs_missing" for i in report.issues)
    assert any(i.code == "run_manifest_decision_trail_reinstatement_timestamp_missing" for i in report.issues)
    assert report.ok is False


def test_full_mode_run_manifest_analyst_checkpoint_invalid_state_fails():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_run_manifest_full()
    payload["analyst_checkpoints"][0]["status"] = "done"
    report = validator.validate_artifact("run_manifest", payload)
    assert any(i.code == "run_manifest_analyst_checkpoint_status_invalid" for i in report.issues)
    assert report.ok is False


def test_full_mode_run_manifest_analyst_checkpoint_state_consistency_fails():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_run_manifest_full()
    payload["analyst_checkpoints"][0]["decision_required"] = False
    payload["analyst_checkpoints"][0]["decision_state"] = "hold_until_review"
    report = validator.validate_artifact("run_manifest", payload)
    assert any(i.code == "run_manifest_analyst_checkpoint_state_inconsistent" for i in report.issues)
    assert report.ok is False


def _base_rca_card_for_full_mode_bundle():
    return {
        "rca_id": "RCA::EVT-STRICT-1::1",
        "event_id": "EVT-STRICT-1",
        "generated_at": "2026-04-25T12:10:00Z",
        "llm_model": "stub",
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
            "primary_conclusion": "Primary mechanism selected.",
            "confidence_label": "medium",
            "analyst_attention_flags": [],
            "causal_depth_summary": {
                "proximate_cause": "Bearing wear",
                "contributing_causes": ["PM interval mismatch"],
                "root_cause": "Program gap",
                "depth_complete": True,
            },
            "unresolved_gaps": ["Need confirmation inspection."],
            "effectiveness_monitoring_plan": [
                {
                    "linked_action_id": "ACT-001",
                    "indicator": "Repeat precursor anomaly",
                    "threshold": "No repeat in 90d",
                    "review_horizon": "90d",
                }
            ],
        },
        "primary_hypothesis": {
            "candidate_id": "C1",
            "cause_label": "Bearing wear",
            "hypothesis_type": "failure_mode",
            "narrative": "n",
            "why_primary": ["w"],
            "uncertainties": ["u"],
            "composite_score": 0.61,
            "confidence_label": "medium",
            "citations": [{"claim_summary": "c", "source_type": "kg_path", "source_id": "C1", "excerpt": "x"}],
        },
        "contributing_causes": [],
        "alternatives": [],
        "evidence": [],
        "recommended_actions": [
            {
                "action_id": "ACT-001",
                "action_type": "immediate_corrective",
                "description": "Address proximate mechanism",
                "priority": "high",
                "target_causal_depth": "proximate",
            },
            {
                "action_id": "ACT-002",
                "action_type": "preventive",
                "description": "Address contributing factors",
                "priority": "medium",
                "target_causal_depth": "contributing",
            },
            {
                "action_id": "ACT-003",
                "action_type": "long_term_corrective",
                "description": "Address systemic root driver",
                "priority": "high",
                "target_causal_depth": "root",
            },
        ],
        "analyst_review": {"decision_required": True, "questions_to_resolve": ["q"], "writeback_recommendation": "hold_until_review"},
        "provenance": {"pipeline_version": "v", "generated_by": "unit_test"},
    }


def test_full_mode_bundle_missing_depth_summary_fails():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_payload()
    rca_card = _base_rca_card_for_full_mode_bundle()
    del rca_card["executive_summary"]["causal_depth_summary"]
    report = validator.validate_run_bundle(
        event={"event_id": "EVT-STRICT-1", "asset_id": "ASSET-1"},
        causality_candidates=payload,
        rca_card=rca_card,
    )
    assert any(i.code == "full_mode_causal_depth_summary_missing" for i in report.issues)
    assert report.ok is False


def test_full_mode_bundle_missing_effectiveness_plan_fails():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_payload()
    rca_card = _base_rca_card_for_full_mode_bundle()
    rca_card["executive_summary"]["effectiveness_monitoring_plan"] = []
    report = validator.validate_run_bundle(
        event={"event_id": "EVT-STRICT-1", "asset_id": "ASSET-1"},
        causality_candidates=payload,
        rca_card=rca_card,
    )
    assert any(i.code == "full_mode_effectiveness_plan_missing" for i in report.issues)
    assert report.ok is False


def test_full_mode_bundle_requires_action_mapping_for_all_resolved_depth_layers():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_payload()
    rca_card = _base_rca_card_for_full_mode_bundle()
    rca_card["recommended_actions"] = [
        {
            "action_id": "ACT-001",
            "action_type": "immediate_corrective",
            "description": "Only proximate mapping",
            "priority": "high",
            "target_causal_depth": "proximate",
        }
    ]
    report = validator.validate_run_bundle(
        event={"event_id": "EVT-STRICT-1", "asset_id": "ASSET-1"},
        causality_candidates=payload,
        rca_card=rca_card,
    )
    assert any(i.code == "full_mode_recommended_actions_depth_mapping_incomplete" for i in report.issues)
    assert report.ok is False


def _base_run_context_full_mode():
    return {
        "run_id": "RUN-STRICT-S0-1",
        "run_label": "strict-step0",
        "started_at": "2026-04-25T12:00:00Z",
        "config": {
            "metamodel_compliance_level": "full",
            "top_k_candidates": 5,
            "top_k_evidence": 8,
        },
        "input_refs": {
            "event_id": "EVT-STRICT-S0-1",
            "asset_id": "ASSET-S0-1",
            "telemetry_asset_id": "ASSET-S0-1",
            "has_operational_context": True,
            "has_pm_compliance": True,
        },
        "validation": {"inputs": {"ok": True}},
        "scope_management": {
            "active_scope_version": 0,
            "latest_approved_revision_id": "SCOPE::EVT-STRICT-S0-1::0",
            "scope_revisions": [
                {
                    "revision_id": "SCOPE::EVT-STRICT-S0-1::0",
                    "scope_version": 0,
                    "trigger": "initial_intake",
                    "changed_boundary": {"window_delta": "initial"},
                    "analyst_decision": "accepted",
                    "decision_timestamp": "2026-04-25T12:00:00Z",
                    "scope_snapshot": {
                        "asset_ids": ["ASSET-S0-1"],
                        "component_ids": ["CMP-1"],
                        "system_boundary": ["RCS"],
                        "time_window": {
                            "start": "2026-04-25T11:50:00Z",
                            "end": "2026-04-25T12:00:00Z",
                        },
                        "safety_function_map": [],
                    },
                }
            ],
        },
    }


def test_full_mode_run_context_requires_scope_management():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_run_context_full_mode()
    del payload["scope_management"]
    report = validator.validate_artifact("run_context", payload)
    assert any(i.code == "scope_management_missing_full_mode" for i in report.issues)
    assert report.ok is False


def test_full_mode_run_context_active_version_must_match_accepted_revision():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_run_context_full_mode()
    payload["scope_management"]["active_scope_version"] = 2
    report = validator.validate_artifact("run_context", payload)
    assert any(i.code == "active_scope_version_not_accepted" for i in report.issues)
    assert report.ok is False


def test_full_mode_run_context_input_refs_scope_markers_must_match_scope_management():
    validator = RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")
    payload = _base_run_context_full_mode()
    payload["input_refs"]["active_scope_version"] = 1
    payload["input_refs"]["active_scope_revision_id"] = "SCOPE::EVT-STRICT-S0-1::999"
    report = validator.validate_artifact("run_context", payload)
    assert any(i.code == "input_refs_active_scope_version_mismatch" for i in report.issues)
    assert any(i.code == "input_refs_active_scope_revision_id_mismatch" for i in report.issues)
    assert report.ok is False
