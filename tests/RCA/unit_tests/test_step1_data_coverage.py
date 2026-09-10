"""
test_step1_data_coverage.py — Step 1 Data Management hardening tests

Covers:
- Coverage report contains all 8 source families
- Per-artifact quality fields drive telemetry/SOE/alarm status
- Paired-data check (SOE ↔ protection logic context)
- Coverage quality factor weighted across new families
- Strict-mode semantic validation (paired requirement, overall status consistency,
  telemetry missing, paired_data_checks block presence)
- Paired-data warning surfaces in review_hooks.degraded_reasons

Run:  pytest test_step1_data_coverage.py -v
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
from orchestrators.causality_engine_v32 import RuleBasedCausalityEngineV32 as CausalityEngineV32
from validation.schema_validator import RCAArtifactValidator

ALL_EXPECTED_FAMILIES = {
    "kg_context", "chroma_corpus", "upstream_anomaly_inputs",
    "telemetry_detail", "soe_log", "alarm_log",
    "protection_logic_context", "configuration_change_records",
    "environmental_monitoring", "vendor_supply_chain_records", "training_records",
}

_VALID_STATUSES = {"complete", "partial", "missing", "not_assessed"}


def _base_kg():
    return {
        "subgraph_id": "KG::S1",
        "components": [{"component_id": "C1"}],
        "failure_modes": [{"fm_id": "FM-1"}],
        "past_events": [],
    }


def _base_tskr():
    return {"patterns": [{"pattern_id": "P1"}]}


def _base_evidence():
    return {"results": [{"snippet_id": "S1"}, {"snippet_id": "S2"}, {"snippet_id": "S3"}]}


def _base_candidates():
    return {
        "candidates": [],
        "category_coverage": {"A": {"status": "ruled_out"}},
        "provenance": {},
    }


def _base_telemetry(degraded: bool = False):
    if degraded:
        return {
            "asset_id": "A1",
            "signals": [
                {"tag_id": "T1", "data_quality": {"missing_fraction": 0.30, "flatline_detected": True, "outlier_fraction": 0.0}},
            ],
        }
    return {
        "asset_id": "A1",
        "signals": [
            {"tag_id": "T1", "data_quality": {"missing_fraction": 0.02, "flatline_detected": False, "outlier_fraction": 0.0}},
        ],
    }


def _base_soe_log(degraded: bool = False):
    if degraded:
        return {"records": [{"record_id": "R1"}], "quality": {"clock_sync_ok": False, "dropped_record_count": 5, "duplicate_record_count": 0}}
    return {"records": [{"record_id": "R1"}], "quality": {"clock_sync_ok": True, "dropped_record_count": 0, "duplicate_record_count": 0}}


def _base_alarm_log(degraded: bool = False):
    if degraded:
        return {"alarms": [], "quality": {"clock_sync_ok": True, "missing_fraction": 0.35}}
    return {"alarms": [{"alarm_id": "AL1", "system": "RCS"}], "quality": {"clock_sync_ok": True, "missing_fraction": 0.0}}


def _base_plc():
    return {"actuation_records": [{"record_id": "PLC-1"}]}


def _base_run_context(has_soe=False, has_alarm=False, has_plc=False, has_ccr=False):
    return {
        "run_id": "RUN-S1",
        "input_refs": {
            "event_id": "EVT-S1",
            "has_soe_log": has_soe,
            "has_alarm_log": has_alarm,
            "has_protection_logic_context": has_plc,
            "has_configuration_change_records": has_ccr,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Coverage report structure
# ─────────────────────────────────────────────────────────────────────────────

def test_coverage_report_contains_all_eight_families_minimal():
    """All 8 families present even when only core inputs provided."""
    cov = RCAReasoningOrchestrator._build_data_coverage_summary(
        kg_context=_base_kg(),
        tskr_patterns=_base_tskr(),
        evidence_bundle=_base_evidence(),
        causality_candidates=_base_candidates(),
        telemetry_summary=_base_telemetry(),
    )
    families = set((cov.get("source_families") or {}).keys())
    assert ALL_EXPECTED_FAMILIES.issubset(families), f"Missing families: {ALL_EXPECTED_FAMILIES - families}"


def test_coverage_report_family_statuses_are_valid_values():
    cov = RCAReasoningOrchestrator._build_data_coverage_summary(
        kg_context=_base_kg(),
        tskr_patterns=_base_tskr(),
        evidence_bundle=_base_evidence(),
        causality_candidates=_base_candidates(),
        telemetry_summary=_base_telemetry(),
        soe_log=_base_soe_log(),
        alarm_log=_base_alarm_log(),
        protection_logic_context=_base_plc(),
    )
    families = cov.get("source_families") or {}
    for name, row in families.items():
        status = row.get("status")
        assert status in _VALID_STATUSES, f"Family '{name}' has invalid status: {status}"


def test_coverage_report_optional_families_are_not_assessed_when_absent():
    cov = RCAReasoningOrchestrator._build_data_coverage_summary(
        kg_context=_base_kg(),
        tskr_patterns=_base_tskr(),
        evidence_bundle=_base_evidence(),
        causality_candidates=_base_candidates(),
        telemetry_summary=_base_telemetry(),
    )
    fam = cov["source_families"]
    assert fam["soe_log"]["status"] == "not_assessed"
    assert fam["alarm_log"]["status"] == "not_assessed"
    assert fam["protection_logic_context"]["status"] == "not_assessed"
    assert fam["configuration_change_records"]["status"] == "not_assessed"


def test_coverage_report_overall_status_not_degraded_by_not_assessed_families():
    """Overall status must not count not_assessed families as degraded."""
    cov = RCAReasoningOrchestrator._build_data_coverage_summary(
        kg_context=_base_kg(),
        tskr_patterns=_base_tskr(),
        evidence_bundle=_base_evidence(),
        causality_candidates=_base_candidates(),
        telemetry_summary=_base_telemetry(),
    )
    # All optional families are not_assessed; core families are complete
    assert cov["overall_status"] == "complete"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-artifact quality field consumption
# ─────────────────────────────────────────────────────────────────────────────

def test_telemetry_degraded_signals_produce_partial_status():
    cov = RCAReasoningOrchestrator._build_data_coverage_summary(
        kg_context=_base_kg(),
        tskr_patterns=_base_tskr(),
        evidence_bundle=_base_evidence(),
        causality_candidates=_base_candidates(),
        telemetry_summary=_base_telemetry(degraded=True),
    )
    assert cov["source_families"]["telemetry_detail"]["status"] == "partial"
    metrics = cov["source_families"]["telemetry_detail"]["metrics"]
    assert metrics["degraded_signal_count"] > 0


def test_telemetry_healthy_signals_produce_complete_status():
    cov = RCAReasoningOrchestrator._build_data_coverage_summary(
        kg_context=_base_kg(),
        tskr_patterns=_base_tskr(),
        evidence_bundle=_base_evidence(),
        causality_candidates=_base_candidates(),
        telemetry_summary=_base_telemetry(degraded=False),
    )
    assert cov["source_families"]["telemetry_detail"]["status"] == "complete"


def test_soe_log_quality_fields_drive_partial_status():
    cov = RCAReasoningOrchestrator._build_data_coverage_summary(
        kg_context=_base_kg(),
        tskr_patterns=_base_tskr(),
        evidence_bundle=_base_evidence(),
        causality_candidates=_base_candidates(),
        soe_log=_base_soe_log(degraded=True),
    )
    assert cov["source_families"]["soe_log"]["status"] == "partial"
    metrics = cov["source_families"]["soe_log"]["metrics"]
    assert metrics["clock_sync_ok"] is False
    assert metrics["dropped_record_count"] == 5


def test_soe_log_healthy_produces_complete_status():
    cov = RCAReasoningOrchestrator._build_data_coverage_summary(
        kg_context=_base_kg(),
        tskr_patterns=_base_tskr(),
        evidence_bundle=_base_evidence(),
        causality_candidates=_base_candidates(),
        soe_log=_base_soe_log(degraded=False),
    )
    assert cov["source_families"]["soe_log"]["status"] == "complete"


def test_alarm_log_high_missing_fraction_produces_partial_status():
    cov = RCAReasoningOrchestrator._build_data_coverage_summary(
        kg_context=_base_kg(),
        tskr_patterns=_base_tskr(),
        evidence_bundle=_base_evidence(),
        causality_candidates=_base_candidates(),
        alarm_log=_base_alarm_log(degraded=True),
    )
    assert cov["source_families"]["alarm_log"]["status"] == "partial"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Paired-data checks
# ─────────────────────────────────────────────────────────────────────────────

def test_paired_check_ok_when_both_soe_and_plc_present():
    cov = RCAReasoningOrchestrator._build_data_coverage_summary(
        kg_context=_base_kg(),
        tskr_patterns=_base_tskr(),
        evidence_bundle=_base_evidence(),
        causality_candidates=_base_candidates(),
        soe_log=_base_soe_log(),
        protection_logic_context=_base_plc(),
    )
    assert cov["paired_data_checks"]["soe_protection_logic_pairing"] == "ok"


def test_paired_check_warning_when_soe_present_plc_absent():
    cov = RCAReasoningOrchestrator._build_data_coverage_summary(
        kg_context=_base_kg(),
        tskr_patterns=_base_tskr(),
        evidence_bundle=_base_evidence(),
        causality_candidates=_base_candidates(),
        soe_log=_base_soe_log(),
        protection_logic_context=None,
    )
    assert cov["paired_data_checks"]["soe_protection_logic_pairing"] == "violated"
    assert cov["source_families"]["protection_logic_context"]["status"] == "missing"


def test_paired_check_not_applicable_when_both_absent():
    cov = RCAReasoningOrchestrator._build_data_coverage_summary(
        kg_context=_base_kg(),
        tskr_patterns=_base_tskr(),
        evidence_bundle=_base_evidence(),
        causality_candidates=_base_candidates(),
    )
    assert cov["paired_data_checks"]["soe_protection_logic_pairing"] == "not_applicable"


def test_paired_check_warning_surfaces_in_review_hooks():
    o = RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
    )
    coverage_with_warning = {
        "overall_status": "complete",
        "source_families": {},
        "paired_data_checks": {"soe_protection_logic_pairing": "violated"},
    }
    review_hooks = o._compute_review_hooks(
        rca_card={
            "validation_status": {"schema_valid": True, "all_claims_cited": True, "passed_minimum_evidence_gate": True, "fallback_used": False},
            "analyst_review": {"decision_required": False, "writeback_recommendation": "ready_if_accepted"},
            "executive_summary": {"decision_status": "candidate_ready"},
            "primary_hypothesis": {"candidate_id": "C1", "composite_score": 0.7},
            "recommended_actions": [],
        },
        output_validation={"ok": True},
        coverage_summary=coverage_with_warning,
    )
    degraded_reasons = review_hooks.get("degraded_reasons") or []
    assert any("paired-data" in r.lower() or "protection logic" in r.lower() for r in degraded_reasons), (
        f"Expected paired-data warning in degraded_reasons; got: {degraded_reasons}"
    )
    analyst_decisions = review_hooks.get("analyst_decisions_required") or []
    assert any("protection_logic_context" in d.lower() or "plc" in d.lower() for d in analyst_decisions), (
        f"Expected PLC pairing violation in analyst_decisions_required; got: {analyst_decisions}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Coverage quality factor — weighted engine profile
# ─────────────────────────────────────────────────────────────────────────────

def test_coverage_factor_1_when_all_complete():
    cov = {
        "overall_status": "complete",
        "source_families": {
            "kg_context": {"status": "complete"},
            "chroma_corpus": {"status": "complete"},
            "upstream_anomaly_inputs": {"status": "complete"},
            "telemetry_detail": {"status": "complete"},
            "soe_log": {"status": "complete"},
            "alarm_log": {"status": "complete"},
            "protection_logic_context": {"status": "complete"},
            "configuration_change_records": {"status": "complete"},
        },
    }
    factor, flags = CausalityEngineV32._coverage_quality_profile(cov)
    assert factor == 1.0
    assert flags == []


def test_coverage_factor_below_1_when_kg_partial():
    cov = {
        "overall_status": "partial",
        "source_families": {
            "kg_context": {"status": "partial"},
            "chroma_corpus": {"status": "complete"},
            "upstream_anomaly_inputs": {"status": "complete"},
            "telemetry_detail": {"status": "complete"},
        },
    }
    factor, flags = CausalityEngineV32._coverage_quality_profile(cov)
    assert factor < 1.0
    assert "kg_context" in flags


def test_coverage_factor_not_assessed_families_not_penalized():
    """not_assessed optional families must not reduce the coverage factor."""
    cov_all_not_assessed = {
        "overall_status": "complete",
        "source_families": {
            "kg_context": {"status": "complete"},
            "chroma_corpus": {"status": "complete"},
            "upstream_anomaly_inputs": {"status": "complete"},
            "telemetry_detail": {"status": "complete"},
            "soe_log": {"status": "not_assessed"},
            "alarm_log": {"status": "not_assessed"},
            "protection_logic_context": {"status": "not_assessed"},
            "configuration_change_records": {"status": "not_assessed"},
        },
    }
    cov_three_core = {
        "overall_status": "complete",
        "source_families": {
            "kg_context": {"status": "complete"},
            "chroma_corpus": {"status": "complete"},
            "upstream_anomaly_inputs": {"status": "complete"},
            "telemetry_detail": {"status": "complete"},
        },
    }
    factor_with_not_assessed, flags1 = CausalityEngineV32._coverage_quality_profile(cov_all_not_assessed)
    factor_core_only, flags2 = CausalityEngineV32._coverage_quality_profile(cov_three_core)
    assert factor_with_not_assessed == factor_core_only == 1.0
    assert flags1 == [] and flags2 == []


def test_coverage_factor_optional_assessed_and_degraded_reduces_score():
    cov = {
        "overall_status": "partial",
        "source_families": {
            "kg_context": {"status": "complete"},
            "chroma_corpus": {"status": "complete"},
            "upstream_anomaly_inputs": {"status": "complete"},
            "telemetry_detail": {"status": "complete"},
            "soe_log": {"status": "missing"},
            "alarm_log": {"status": "missing"},
            "protection_logic_context": {"status": "missing"},
            "configuration_change_records": {"status": "missing"},
        },
    }
    factor, flags = CausalityEngineV32._coverage_quality_profile(cov)
    assert factor < 1.0
    assert set(flags) >= {"soe_log", "alarm_log", "protection_logic_context", "configuration_change_records"}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Strict-mode semantic validation
# ─────────────────────────────────────────────────────────────────────────────

def _validator():
    return RCAArtifactValidator(schema_dir=_RCA_ROOT / "schemas", mode="compat")


def _base_manifest_full_coverage():
    """A run_manifest with a complete, valid Step 1 coverage_summary in full mode."""
    return {
        "run_id": "RUN-S1-VAL",
        "completed_at": "2026-04-25T12:00:00Z",
        "input_refs": {"event_id": "EVT-S1-VAL", "asset_id": "ASSET-S1"},
        "pipeline_config": {
            "metamodel_compliance_level": "full",
            "metamodel_migration": {"phase": "wave4", "compatibility_mode": False},
        },
        "artifacts": {},
        "coverage_summary": {
            "overall_status": "complete",
            "source_families": {fam: {"status": "complete", "metrics": {}} for fam in ALL_EXPECTED_FAMILIES},
            "paired_data_checks": {"soe_protection_logic_pairing": "not_applicable"},
            "category_coverage": {},
        },
        "review_hooks": {
            "coverage_degraded": False,
            "coverage_status": "complete",
            "coverage_acknowledgement_required": False,
            "coverage_acknowledged": False,
            "writeback_ready": True,
        },
        "pipeline_health": {"status": "green", "issues": []},
        "ap913_completeness": {
            "root_cause_identified": True,
            "direct_cause_identified": True,
            "contributing_causes_identified": True,
            "extent_of_condition_assessed": True,
            "effectiveness_review_defined": True,
        },
        "applicability_summary": {},
        "uncertainty_summary": {
            "candidate_count": 1,
            "average_quality_multiplier": 0.95,
            "average_coverage_quality_factor": 1.0,
            "coverage_degraded_candidate_count": 0,
            "coverage_flagged_source_families": [],
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
            {"event_type": "final_decision", "candidate_id": "C1", "decision_status": "review_required", "confidence_label": "medium"},
        ],
    }


def test_full_mode_passes_with_valid_step1_coverage_summary():
    v = _validator()
    manifest = _base_manifest_full_coverage()
    report = v.validate_artifact("run_manifest", manifest)
    errors = [i for i in report.issues if i.severity == "error"]
    step1_errors = [e for e in errors if "step1" in str(e.code or "").lower() or "paired" in str(e.code or "").lower()]
    assert step1_errors == [], f"Unexpected Step 1 errors: {[(e.code, e.message) for e in step1_errors]}"


def test_full_mode_fails_when_step1_families_missing_from_coverage():
    v = _validator()
    manifest = _base_manifest_full_coverage()
    for fam in ("telemetry_detail", "soe_log", "alarm_log", "protection_logic_context", "configuration_change_records",
                "environmental_monitoring", "vendor_supply_chain_records", "training_records"):
        del manifest["coverage_summary"]["source_families"][fam]
    report = v.validate_artifact("run_manifest", manifest)
    error_codes = {i.code for i in report.issues if i.severity == "error"}
    assert "run_manifest_coverage_summary_step1_family_missing" in error_codes


def test_full_mode_fails_when_telemetry_is_missing():
    v = _validator()
    manifest = _base_manifest_full_coverage()
    manifest["coverage_summary"]["source_families"]["telemetry_detail"]["status"] = "missing"
    manifest["coverage_summary"]["overall_status"] = "missing"
    manifest["review_hooks"]["coverage_degraded"] = True
    manifest["review_hooks"]["coverage_acknowledgement_required"] = True
    manifest["review_hooks"]["coverage_acknowledged"] = False
    manifest["review_hooks"]["writeback_ready"] = False
    manifest["review_hooks"]["coverage_status"] = "missing"
    manifest["uncertainty_summary"]["coverage_degraded_candidate_count"] = 0
    manifest["uncertainty_summary"]["average_coverage_quality_factor"] = 0.85
    report = v.validate_artifact("run_manifest", manifest)
    error_codes = {i.code for i in report.issues if i.severity == "error"}
    assert "run_manifest_telemetry_detail_missing" in error_codes


def test_full_mode_fails_when_soe_present_plc_missing():
    v = _validator()
    manifest = _base_manifest_full_coverage()
    manifest["coverage_summary"]["source_families"]["soe_log"]["status"] = "complete"
    manifest["coverage_summary"]["source_families"]["protection_logic_context"]["status"] = "missing"
    manifest["coverage_summary"]["paired_data_checks"]["soe_protection_logic_pairing"] = "violated"
    report = v.validate_artifact("run_manifest", manifest)
    error_codes = {i.code for i in report.issues if i.severity == "error"}
    assert "run_manifest_paired_data_soe_plc_violated" in error_codes


def test_full_mode_fails_when_overall_complete_but_families_degraded():
    v = _validator()
    manifest = _base_manifest_full_coverage()
    manifest["coverage_summary"]["source_families"]["chroma_corpus"]["status"] = "partial"
    # overall_status still "complete" — inconsistency
    report = v.validate_artifact("run_manifest", manifest)
    error_codes = {i.code for i in report.issues if i.severity == "error"}
    assert "run_manifest_coverage_overall_status_inconsistent" in error_codes


def test_full_mode_fails_when_paired_data_checks_block_absent():
    v = _validator()
    manifest = _base_manifest_full_coverage()
    del manifest["coverage_summary"]["paired_data_checks"]
    report = v.validate_artifact("run_manifest", manifest)
    error_codes = {i.code for i in report.issues if i.severity == "error"}
    assert "run_manifest_paired_data_checks_missing" in error_codes


def test_not_assessed_status_is_accepted_without_error():
    """not_assessed is valid for optional families; no status_invalid error raised."""
    v = _validator()
    manifest = _base_manifest_full_coverage()
    for fam in ("soe_log", "alarm_log", "protection_logic_context", "configuration_change_records"):
        manifest["coverage_summary"]["source_families"][fam]["status"] = "not_assessed"
    report = v.validate_artifact("run_manifest", manifest)
    status_invalid_errors = [
        i for i in report.issues
        if i.severity == "error" and i.code == "run_manifest_coverage_summary_status_invalid"
    ]
    assert status_invalid_errors == [], f"Unexpected status_invalid errors: {[(e.code, e.message) for e in status_invalid_errors]}"


ALL_TESTS = [
    test_coverage_report_contains_all_eight_families_minimal,
    test_coverage_report_family_statuses_are_valid_values,
    test_coverage_report_optional_families_are_not_assessed_when_absent,
    test_coverage_report_overall_status_not_degraded_by_not_assessed_families,
    test_telemetry_degraded_signals_produce_partial_status,
    test_telemetry_healthy_signals_produce_complete_status,
    test_soe_log_quality_fields_drive_partial_status,
    test_soe_log_healthy_produces_complete_status,
    test_alarm_log_high_missing_fraction_produces_partial_status,
    test_paired_check_ok_when_both_soe_and_plc_present,
    test_paired_check_warning_when_soe_present_plc_absent,
    test_paired_check_not_applicable_when_both_absent,
    test_paired_check_warning_surfaces_in_review_hooks,
    test_coverage_factor_1_when_all_complete,
    test_coverage_factor_below_1_when_kg_partial,
    test_coverage_factor_not_assessed_families_not_penalized,
    test_coverage_factor_optional_assessed_and_degraded_reduces_score,
    test_full_mode_passes_with_valid_step1_coverage_summary,
    test_full_mode_fails_when_step1_families_missing_from_coverage,
    test_full_mode_fails_when_telemetry_is_missing,
    test_full_mode_fails_when_soe_present_plc_missing,
    test_full_mode_fails_when_overall_complete_but_families_degraded,
    test_full_mode_fails_when_paired_data_checks_block_absent,
    test_not_assessed_status_is_accepted_without_error,
]

if __name__ == "__main__":
    passed = 0
    failed = 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            print(f"  PASS  {test_fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {test_fn.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
