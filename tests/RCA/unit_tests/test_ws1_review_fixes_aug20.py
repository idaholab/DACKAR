"""
test_ws1_review_fixes_aug20.py — Workstream 1 fixes from the Aug-20 causal-soundness review.

Covers:
  F-1  physical_plausibility gate honestly labels itself a minimum-structural-score
       screen (check_basis / operating_state_checked) instead of overclaiming a
       physical-state check.
  P-1  KG context builder records per-family truncation stats into
       kg_context.provenance so silently-dropped candidates are visible.
  P-6  Optional-phase failures (here: CMMS context build) are recorded in
       run_manifest.pipeline_warnings rather than swallowed with a log line.

No live Neo4j, Chroma, or LLM required.

Run:  pytest test_ws1_review_fixes_aug20.py -v
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
_SCENARIO_ROOT = Path(__file__).resolve().parents[1] / "scenario"
_TESTS_SHARED = _SCENARIO_ROOT / "shared"
for _p in (str(_RCA_ROOT), str(_TESTS_SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _mod in (
    "neo4j", "py2neo", "chromadb", "langchain_community",
    "langchain_community.vectorstores", "langchain_community.embeddings",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest  # noqa: E402

from orchestrators.causality_engine_v32 import (  # noqa: E402
    RuleBasedCausalityEngineV32,
    CausalityEngineConfigV32,
)
from orchestrators.kg_context_builder import (  # noqa: E402
    Neo4jKGContextBuilder,
    KGContextBuilderConfig,
)
import pytest  # noqa: E402
pytest.importorskip("run_helpers", reason="scenario shared helpers (tests/RCA/scenario/shared) arrive in MR #12")
from run_helpers import build_fixture_orchestrator, load_fixtures, run_rca  # noqa: E402

_TC8_FIXTURES = _SCENARIO_ROOT / "test_case_8" / "fixtures"


# ═══════════════════════════════════════════════════════════════════════
# F-1 — physical_plausibility gate is honestly labelled
# ═══════════════════════════════════════════════════════════════════════

def _engine() -> RuleBasedCausalityEngineV32:
    return RuleBasedCausalityEngineV32(config=CausalityEngineConfigV32())


def test_f1_gate_pass_declares_structural_screen_basis():
    cand = {"scores": {"structural": 0.80}, "component_id": "C1", "failure_mode_id": "FM1"}
    _engine()._apply_physical_plausibility_gate(cand)
    gate = cand["hard_gates"]["physical_plausibility"]
    assert gate["passed"] is True
    assert gate["check_basis"] == "minimum_structural_score"
    assert gate["operating_state_checked"] is False


def test_f1_gate_rationale_disclaims_operating_state_check():
    cand = {"scores": {"structural": 0.80}, "component_id": "C1", "failure_mode_id": "FM1"}
    _engine()._apply_physical_plausibility_gate(cand)
    rationale = cand["hard_gates"]["physical_plausibility"]["rationale"].lower()
    assert "operating-state" in rationale
    assert "not evaluated" in rationale


def test_f1_gate_fail_below_floor_still_labels_basis():
    cand = {"scores": {"structural": 0.10}, "component_id": "C1", "failure_mode_id": "FM1"}
    _engine()._apply_physical_plausibility_gate(cand)
    gate = cand["hard_gates"]["physical_plausibility"]
    assert gate["passed"] is False
    assert gate["check_basis"] == "minimum_structural_score"
    assert gate["operating_state_checked"] is False


# ═══════════════════════════════════════════════════════════════════════
# P-1 — KG builder records truncation stats
# ═══════════════════════════════════════════════════════════════════════

def _bare_builder() -> Neo4jKGContextBuilder:
    b = Neo4jKGContextBuilder.__new__(Neo4jKGContextBuilder)
    b._truncation_stats = {}
    return b


def test_p1_record_truncation_known_total():
    b = _bare_builder()
    b._record_truncation(
        family="past_events", total_matched=15, cap=10, dropped_ids=["e11", "e12"]
    )
    stat = b._truncation_stats["past_events"]
    assert stat["truncated"] is True
    assert stat["cap"] == 10
    assert stat["total_matched"] == 15
    assert stat["retained"] == 10
    assert stat["dropped_count"] == 5
    assert stat["dropped_ids"] == ["e11", "e12"]


def test_p1_record_truncation_no_drop():
    b = _bare_builder()
    b._record_truncation(family="documents", total_matched=3, cap=20, dropped_ids=[])
    stat = b._truncation_stats["documents"]
    assert stat["truncated"] is False
    assert stat["dropped_count"] == 0
    assert stat["retained"] == 3


def test_p1_record_truncation_unknown_total_at_cap():
    b = _bare_builder()
    b._record_truncation(
        family="oe_documents", total_matched=None, cap=10, dropped_ids=[], retained=10
    )
    stat = b._truncation_stats["oe_documents"]
    assert stat["truncated"] is True
    assert stat["total_matched"] is None
    assert stat["dropped_count"] is None


def test_p1_build_populates_provenance_expansion_and_truncation():
    client = MagicMock()
    client.query.return_value = []
    builder = Neo4jKGContextBuilder(
        client=client, database="testdb", config=KGContextBuilderConfig()
    )
    kg_context = builder.build(
        event={"event_id": "E1", "asset_id": "A1"},
        telemetry_summary={"asset_id": "A1", "signals": []},
        operational_context=None,
        pm_compliance=None,
        run_context={"run_id": "R1"},
    )
    prov = kg_context["provenance"]
    assert "expansion" in prov and prov["expansion"]["max_hops"] == KGContextBuilderConfig().max_hops
    assert "truncation" in prov
    assert prov["truncation_occurred"] is False  # empty KG → nothing truncated
    # past_events and documents families are always assessed when included
    assert "past_events" in prov["truncation"]
    assert "documents" in prov["truncation"]


# ═══════════════════════════════════════════════════════════════════════
# P-6 — optional-phase (CMMS) failure surfaces in the manifest
# ═══════════════════════════════════════════════════════════════════════

def test_p6_cmms_build_failure_recorded_in_pipeline_warnings():
    if not _TC8_FIXTURES.exists():
        pytest.skip(f"TC-8 fixtures not found at {_TC8_FIXTURES}")

    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = build_fixture_orchestrator(tmp)
        # Force the Stage 5B CMMS branch to execute, then make it fail.
        orchestrator.cmms_adapter = object()

        def _boom(*_args, **_kwargs):
            raise RuntimeError("Simulated CMMS adapter outage (P-6 test).")

        orchestrator.build_cmms_context = _boom  # type: ignore[assignment]

        fixtures = load_fixtures(_TC8_FIXTURES)
        result = run_rca(orchestrator, fixtures)

    manifest = result.get("run_manifest") or {}
    warnings = manifest.get("pipeline_warnings") or []
    assert warnings, "P-6 FAIL: CMMS failure not recorded in pipeline_warnings."
    phases = {str((w or {}).get("phase")) for w in warnings if isinstance(w, dict)}
    assert "cmms_context_build" in phases, (
        f"P-6 FAIL: expected a 'cmms_context_build' warning entry; got phases={phases}"
    )
    # And the run still completes.
    assert result.get("rca_card"), "Pipeline must still produce an rca_card."
