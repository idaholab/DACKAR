"""
test_review_hooks.py — standalone unit tests for
RCAReasoningOrchestrator._compute_review_hooks

Run directly:   python test_review_hooks.py
Or via pytest:  pytest test_review_hooks.py

_compute_review_hooks builds writeback_ready (8-condition AND gate) and next_step:

  writeback_ready requires ALL of:
    1. outputs_ok            (output_validation.ok == True; missing → False)
    2. schema_valid          (rca_card.validation_status.schema_valid)
    3. all_claims_cited      (rca_card.validation_status.all_claims_cited)
    4. passed_minimum_evidence_gate
    5. not decision_required
    6. writeback_recommendation == "ready_if_accepted"
    7. decision_status == "candidate_ready"

  Note: fallback_used no longer blocks writeback — the deterministic path is
  production-quality and should be writeback-eligible when all other gates pass.

  next_step:
    "writeback"              → writeback_ready=True
    "analyst_review"         → outputs_ok=True but not writeback_ready
    "validation_remediation" → outputs_ok=False

  outputs_ok default: missing output_validation → False (safe default)
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

# Stub heavy optional dependencies that are not needed for unit-testing
# _compute_review_hooks (which is a pure dict-manipulation method).
for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator, OrchestratorConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_orchestrator(strict_red_state_governance=True, hard_abort_on_kg_red_state=True):
    return RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
        config=OrchestratorConfig(
            extra={
                "strict_red_state_governance": strict_red_state_governance,
                "hard_abort_on_kg_red_state": hard_abort_on_kg_red_state,
            }
        ),
    )


def make_clean_card(decision_required=False, writeback_recommendation="ready_if_accepted",
                    decision_status="candidate_ready", fallback_used=False,
                    schema_valid=True, all_claims_cited=True,
                    passed_minimum_evidence_gate=True):
    return {
        "validation_status": {
            "schema_valid": schema_valid,
            "all_claims_cited": all_claims_cited,
            "passed_minimum_evidence_gate": passed_minimum_evidence_gate,
            "fallback_used": fallback_used,
        },
        "analyst_review": {
            "decision_required": decision_required,
            "writeback_recommendation": writeback_recommendation,
        },
        "executive_summary": {
            "decision_status": decision_status,
        },
    }


def make_output_validation(ok=True):
    return {"ok": ok}


# ── _compute_review_hooks tests ───────────────────────────────────────────

def test_writeback_ready_when_all_conditions_met():
    """All 8 conditions satisfied → writeback_ready=True, next_step='writeback'."""
    o = make_orchestrator()
    result = o._compute_review_hooks(
        make_clean_card(),
        make_output_validation(ok=True),
        pipeline_health={"status": "green", "issues": []},
        reentry_hook={"should_reenter": False},
    )
    assert result["writeback_ready"] is True
    assert result["next_step"] == "writeback"
    print("  PASS test_writeback_ready_when_all_conditions_met")


def test_writeback_blocked_when_outputs_not_ok():
    """outputs_ok=False → writeback_ready=False, next_step='validation_remediation'."""
    o = make_orchestrator()
    result = o._compute_review_hooks(
        make_clean_card(),
        make_output_validation(ok=False),
        pipeline_health={"status": "green", "issues": []},
        reentry_hook={"should_reenter": False},
    )
    assert result["writeback_ready"] is False
    assert result["next_step"] == "validation_remediation"
    print("  PASS test_writeback_blocked_when_outputs_not_ok")


def test_writeback_blocked_when_output_validation_missing():
    """
    output_validation=None → outputs_ok defaults to False (safe default).
    next_step='validation_remediation'.
    """
    o = make_orchestrator()
    result = o._compute_review_hooks(
        make_clean_card(),
        None,
        pipeline_health={"status": "green", "issues": []},
        reentry_hook={"should_reenter": False},
    )
    assert result["outputs_ok"] is False
    assert result["writeback_ready"] is False
    assert result["next_step"] == "validation_remediation"
    print("  PASS test_writeback_blocked_when_output_validation_missing")


def test_writeback_blocked_when_schema_invalid():
    """schema_valid=False → writeback_ready=False."""
    o = make_orchestrator()
    card = make_clean_card(schema_valid=False)
    result = o._compute_review_hooks(
        card,
        make_output_validation(ok=True),
        pipeline_health={"status": "green", "issues": []},
        reentry_hook={"should_reenter": False},
    )
    assert result["writeback_ready"] is False
    assert result["next_step"] == "analyst_review"   # outputs_ok still True
    print("  PASS test_writeback_blocked_when_schema_invalid")


def test_writeback_blocked_when_claims_not_cited():
    """all_claims_cited=False → writeback_ready=False."""
    o = make_orchestrator()
    card = make_clean_card(all_claims_cited=False)
    result = o._compute_review_hooks(
        card,
        make_output_validation(ok=True),
        pipeline_health={"status": "green", "issues": []},
        reentry_hook={"should_reenter": False},
    )
    assert result["writeback_ready"] is False
    print("  PASS test_writeback_blocked_when_claims_not_cited")


def test_writeback_blocked_when_evidence_gate_failed():
    """passed_minimum_evidence_gate=False → writeback_ready=False."""
    o = make_orchestrator()
    card = make_clean_card(passed_minimum_evidence_gate=False)
    result = o._compute_review_hooks(
        card,
        make_output_validation(ok=True),
        pipeline_health={"status": "green", "issues": []},
        reentry_hook={"should_reenter": False},
    )
    assert result["writeback_ready"] is False
    print("  PASS test_writeback_blocked_when_evidence_gate_failed")


def test_writeback_allowed_when_fallback_used_and_all_conditions_met():
    """fallback_used=True no longer blocks writeback — the deterministic path is production-quality."""
    o = make_orchestrator()
    card = make_clean_card(fallback_used=True)
    result = o._compute_review_hooks(
        card,
        make_output_validation(ok=True),
        pipeline_health={"status": "green", "issues": []},
        reentry_hook={"should_reenter": False},
    )
    assert result["writeback_ready"] is True
    print("  PASS test_writeback_allowed_when_fallback_used_and_all_conditions_met")


def test_writeback_blocked_when_decision_required():
    """decision_required=True → writeback_ready=False."""
    o = make_orchestrator()
    card = make_clean_card(decision_required=True)
    result = o._compute_review_hooks(
        card,
        make_output_validation(ok=True),
        pipeline_health={"status": "green", "issues": []},
        reentry_hook={"should_reenter": False},
    )
    assert result["writeback_ready"] is False
    print("  PASS test_writeback_blocked_when_decision_required")


def test_writeback_blocked_when_recommendation_not_ready():
    """writeback_recommendation != 'ready_if_accepted' → writeback_ready=False."""
    o = make_orchestrator()
    card = make_clean_card(writeback_recommendation="hold_until_review")
    result = o._compute_review_hooks(
        card,
        make_output_validation(ok=True),
        pipeline_health={"status": "green", "issues": []},
        reentry_hook={"should_reenter": False},
    )
    assert result["writeback_ready"] is False
    print("  PASS test_writeback_blocked_when_recommendation_not_ready")


def test_writeback_blocked_when_decision_status_not_candidate_ready():
    """decision_status != 'candidate_ready' → writeback_ready=False."""
    o = make_orchestrator()
    card = make_clean_card(decision_status="review_required")
    result = o._compute_review_hooks(
        card,
        make_output_validation(ok=True),
        pipeline_health={"status": "green", "issues": []},
        reentry_hook={"should_reenter": False},
    )
    assert result["writeback_ready"] is False
    print("  PASS test_writeback_blocked_when_decision_status_not_candidate_ready")


def test_next_step_is_analyst_review_when_outputs_ok_but_not_writeback():
    """outputs_ok=True but one condition fails → next_step='analyst_review'."""
    o = make_orchestrator()
    card = make_clean_card(all_claims_cited=False)   # one condition fails
    result = o._compute_review_hooks(
        card,
        make_output_validation(ok=True),
        pipeline_health={"status": "green", "issues": []},
        reentry_hook={"should_reenter": False},
    )
    assert result["next_step"] == "analyst_review"
    print("  PASS test_next_step_is_analyst_review_when_outputs_ok_but_not_writeback")


def test_requires_human_review_computed_from_conditions():
    """requires_human_review is False when all quality gates pass, True otherwise."""
    o = make_orchestrator()
    # All conditions met → no human review needed
    result = o._compute_review_hooks(
        make_clean_card(),
        make_output_validation(ok=True),
        pipeline_health={"status": "green", "issues": []},
        reentry_hook={"should_reenter": False},
    )
    assert result["requires_human_review"] is False, "should not require review when all gates pass"
    # outputs_ok=False → human review required
    result = o._compute_review_hooks(
        make_clean_card(),
        make_output_validation(ok=False),
        pipeline_health={"status": "green", "issues": []},
        reentry_hook={"should_reenter": False},
    )
    assert result["requires_human_review"] is True, "should require review when outputs are invalid"
    # decision_required=True → human review required
    result = o._compute_review_hooks(
        make_clean_card(decision_required=True),
        make_output_validation(ok=True),
        pipeline_health={"status": "green", "issues": []},
        reentry_hook={"should_reenter": False},
    )
    assert result["requires_human_review"] is True, "should require review when decision_required"
    print("  PASS test_requires_human_review_computed_from_conditions")


def test_result_includes_all_diagnostic_fields():
    """Hook result exposes all individual flags for analyst diagnostics."""
    o = make_orchestrator()
    result = o._compute_review_hooks(
        make_clean_card(),
        make_output_validation(ok=True),
        pipeline_health={"status": "green", "issues": []},
        reentry_hook={"should_reenter": False},
    )
    expected_keys = {
        "requires_human_review", "writeback_ready", "next_step",
        "outputs_ok", "schema_valid", "all_claims_cited",
        "fallback_used", "passed_minimum_evidence_gate",
        "decision_required", "decision_status", "writeback_recommendation",
        "degraded_run", "degraded_reasons", "reentry_hook",
    }
    missing = expected_keys - set(result.keys())
    assert not missing, f"Missing keys in hook result: {missing}"
    print("  PASS test_result_includes_all_diagnostic_fields")


def test_writeback_blocked_when_degraded_pipeline_health():
    o = make_orchestrator()
    result = o._compute_review_hooks(
        make_clean_card(),
        make_output_validation(ok=True),
        pipeline_health={"status": "yellow", "issues": ["KG stale"]},
        reentry_hook={"should_reenter": False},
    )
    assert result["writeback_ready"] is False
    assert result["degraded_run"] is True
    assert any("kg stale" in x.lower() for x in result["degraded_reasons"])
    print("  PASS test_writeback_blocked_when_degraded_pipeline_health")


def test_writeback_blocked_when_reentry_recommended():
    o = make_orchestrator()
    result = o._compute_review_hooks(
        make_clean_card(),
        make_output_validation(ok=True),
        pipeline_health={"status": "green", "issues": []},
        reentry_hook={"should_reenter": True, "reason": "rank_inversion_detected"},
    )
    assert result["writeback_ready"] is False
    assert result["degraded_run"] is True
    assert any("rank inversion" in x.lower() for x in result["degraded_reasons"])
    print("  PASS test_writeback_blocked_when_reentry_recommended")


def test_strict_red_state_forces_validation_remediation():
    o = make_orchestrator(strict_red_state_governance=True)
    result = o._compute_review_hooks(
        make_clean_card(),
        make_output_validation(ok=True),
        pipeline_health={"status": "red", "issues": ["KG failure mode count 0 is below minimum 1."]},
        reentry_hook={"should_reenter": False},
    )
    assert result["next_step"] == "validation_remediation"
    assert result["strict_red_state_governance"] is True
    assert result["hard_abort_required"] is True
    print("  PASS test_strict_red_state_forces_validation_remediation")


def test_non_strict_red_state_allows_analyst_review():
    o = make_orchestrator(strict_red_state_governance=False)
    result = o._compute_review_hooks(
        make_clean_card(),
        make_output_validation(ok=True),
        pipeline_health={"status": "red", "issues": ["KG failure mode count 0 is below minimum 1."]},
        reentry_hook={"should_reenter": False},
    )
    assert result["next_step"] == "analyst_review"
    assert result["strict_red_state_governance"] is False
    assert result["hard_abort_required"] is False
    print("  PASS test_non_strict_red_state_allows_analyst_review")


def test_strict_red_without_hard_abort_does_not_mark_abort_required():
    o = make_orchestrator(strict_red_state_governance=True, hard_abort_on_kg_red_state=False)
    result = o._compute_review_hooks(
        make_clean_card(),
        make_output_validation(ok=True),
        pipeline_health={"status": "red", "issues": ["KG failure mode count 0 is below minimum 1."]},
        reentry_hook={"should_reenter": False},
    )
    assert result["next_step"] == "validation_remediation"
    assert result["hard_abort_on_kg_red_state"] is False
    assert result["hard_abort_required"] is False
    print("  PASS test_strict_red_without_hard_abort_does_not_mark_abort_required")


def test_stage_policy_hook_forces_validation_remediation():
    o = make_orchestrator()
    result = o._compute_review_hooks(
        make_clean_card(),
        make_output_validation(ok=True),
        pipeline_health={"status": "green", "issues": []},
        reentry_hook={"should_reenter": False},
        stage_health={
            "stage_e_evidence": {
                "status": "red",
                "issues": ["No evidence results retrieved."],
            }
        },
    )
    assert result["next_step"] == "validation_remediation"
    assert result["stage_hard_stop_required"] is True
    assert len(result["stage_policy_violations"]) >= 1
    print("  PASS test_stage_policy_hook_forces_validation_remediation")


# ── Main runner ───────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_writeback_ready_when_all_conditions_met,
    test_writeback_blocked_when_outputs_not_ok,
    test_writeback_blocked_when_output_validation_missing,
    test_writeback_blocked_when_schema_invalid,
    test_writeback_blocked_when_claims_not_cited,
    test_writeback_blocked_when_evidence_gate_failed,
    test_writeback_allowed_when_fallback_used_and_all_conditions_met,
    test_writeback_blocked_when_decision_required,
    test_writeback_blocked_when_recommendation_not_ready,
    test_writeback_blocked_when_decision_status_not_candidate_ready,
    test_next_step_is_analyst_review_when_outputs_ok_but_not_writeback,
    test_requires_human_review_computed_from_conditions,
    test_result_includes_all_diagnostic_fields,
    test_writeback_blocked_when_degraded_pipeline_health,
    test_writeback_blocked_when_reentry_recommended,
    test_strict_red_state_forces_validation_remediation,
    test_non_strict_red_state_allows_analyst_review,
    test_strict_red_without_hard_abort_does_not_mark_abort_required,
    test_stage_policy_hook_forces_validation_remediation,
]


def run_all():
    print(f"\n=== test_review_hooks ({len(ALL_TESTS)} tests) ===")
    passed, failed = 0, 0
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
    sys.exit(0 if ok else 1)
