"""
test_f3_prevention_analysis_aug20.py — F-3 'why was it not prevented?' output.

The metamodel requires the RCA card to state which barriers failed, which held,
and why (a first-class output). The pre-existing structural `barrier_analysis`
only maps which safety functions a candidate impacts; it never answers *why the
failure was not prevented*. F-3 adds a deterministic, additive `prevention_analysis`
card block assessing three defense-in-depth layers for the primary cause from data
already on hand (PM/surveillance compliance, condition-monitoring detection, and the
primary's barrier-logic hard gate), honestly marking layers without inputs as
`not_evaluated` rather than asserting a failure.

Run:  pytest test_f3_prevention_analysis_aug20.py -v
"""
from __future__ import annotations

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

from orchestrators.llm_clients import DummyLLMClient  # noqa: E402
from synthesis.rca_synthesizer_v31 import (  # noqa: E402
    RuleValidatedRCASynthesizerV31,
    RCASynthesizerConfig,
)


def _synth() -> RuleValidatedRCASynthesizerV31:
    return RuleValidatedRCASynthesizerV31(llm_client=DummyLLMClient(), config=RCASynthesizerConfig())


def _card(primary_id="FM::P"):
    return {"primary_hypothesis": {"candidate_id": primary_id}}


def _cc(primary_id="FM::P", *, barrier_gate=None, safety_functions=None):
    cand = {"candidate_id": primary_id}
    if barrier_gate is not None:
        cand["hard_gates"] = {"barrier_logic": barrier_gate}
    if safety_functions is not None:
        cand["affected_safety_functions"] = safety_functions
    return {"candidates": [cand]}


def _by_type(block, btype):
    return next(b for b in block["barriers"] if b["barrier_type"] == btype)


# ── preventive maintenance layer ────────────────────────────────────────────

def test_pm_failed_check_is_a_gap():
    s = _synth()
    block = s._build_prevention_analysis(
        card=_card(), causality_candidates=_cc(),
        pm_compliance={"checks": [{"check_id": "PM-1", "status": "fail"},
                                  {"check_id": "PM-2", "status": "pass"}]},
        telemetry_summary=None,
    )
    pm = _by_type(block, "preventive_maintenance")
    assert pm["status"] == "gap"
    assert "PM-1" in pm.get("detail", "")
    assert "preventive_maintenance" in block["failed_or_missing_barriers"]
    assert block["applicable"] is True
    assert "not prevented because" in block["why_not_prevented"].lower()


def test_pm_all_pass_holds():
    s = _synth()
    block = s._build_prevention_analysis(
        card=_card(), causality_candidates=_cc(),
        pm_compliance={"checks": [{"check_id": "PM-1", "status": "pass"}]},
        telemetry_summary=None,
    )
    assert _by_type(block, "preventive_maintenance")["status"] == "held"


def test_pm_absent_not_evaluated():
    s = _synth()
    block = s._build_prevention_analysis(
        card=_card(), causality_candidates=_cc(), pm_compliance=None, telemetry_summary=None,
    )
    assert _by_type(block, "preventive_maintenance")["status"] == "not_evaluated"


# ── condition monitoring layer ──────────────────────────────────────────────

def test_detection_holds_when_anomalies_present():
    s = _synth()
    block = s._build_prevention_analysis(
        card=_card(), causality_candidates=_cc(), pm_compliance=None,
        telemetry_summary={"signals": [{"sensor_id": "S-1", "anomalies": [{"pattern": "spike"}]}]},
    )
    assert _by_type(block, "condition_monitoring")["status"] == "held"


def test_detection_gap_when_no_precursor():
    s = _synth()
    block = s._build_prevention_analysis(
        card=_card(), causality_candidates=_cc(), pm_compliance=None,
        telemetry_summary={"signals": [{"sensor_id": "S-1", "anomalies": []}]},
    )
    cm = _by_type(block, "condition_monitoring")
    assert cm["status"] == "gap"
    assert "condition_monitoring" in block["failed_or_missing_barriers"]


# ── protection-logic layer ──────────────────────────────────────────────────

def test_protection_gap_when_gate_passed_with_safety_functions():
    s = _synth()
    block = s._build_prevention_analysis(
        card=_card(),
        causality_candidates=_cc(
            barrier_gate={"passed": True, "degraded_mode": False},
            safety_functions=[{"sf_name": "Reactor Protection", "sf_id": "SF-1"}],
        ),
        pm_compliance=None, telemetry_summary=None,
    )
    pl = _by_type(block, "protection_logic")
    assert pl["status"] == "gap"
    assert "Reactor Protection" in pl.get("detail", "")


def test_protection_not_evaluated_when_gate_degraded():
    s = _synth()
    block = s._build_prevention_analysis(
        card=_card(),
        causality_candidates=_cc(barrier_gate={"passed": True, "degraded_mode": True},
                                 safety_functions=[{"sf_name": "X", "sf_id": "SF-1"}]),
        pm_compliance=None, telemetry_summary=None,
    )
    assert _by_type(block, "protection_logic")["status"] == "not_evaluated"


def test_protection_not_applicable_without_safety_functions():
    s = _synth()
    block = s._build_prevention_analysis(
        card=_card(),
        causality_candidates=_cc(barrier_gate={"passed": True, "degraded_mode": False}),
        pm_compliance=None, telemetry_summary=None,
    )
    assert _by_type(block, "protection_logic")["status"] == "not_applicable"


# ── aggregate behaviour ─────────────────────────────────────────────────────

def test_none_primary_not_applicable():
    s = _synth()
    block = s._build_prevention_analysis(
        card=_card("NONE"), causality_candidates={"candidates": []},
        pm_compliance={"checks": [{"check_id": "PM-1", "status": "fail"}]},
        telemetry_summary=None,
    )
    assert block["applicable"] is False
    assert block["failed_or_missing_barriers"] == []


def test_all_inputs_absent_is_not_applicable_with_data_note():
    s = _synth()
    block = s._build_prevention_analysis(
        card=_card(), causality_candidates=_cc(), pm_compliance=None, telemetry_summary=None,
    )
    # PM not_evaluated, detection not_evaluated, protection not_evaluated → nothing assessable
    assert block["applicable"] is False
    assert "insufficient" in block["why_not_prevented"].lower()
    assert all(b["status"] == "not_evaluated" for b in block["barriers"])


def test_schema_shape_is_complete():
    s = _synth()
    block = s._build_prevention_analysis(
        card=_card(), causality_candidates=_cc(),
        pm_compliance={"checks": [{"check_id": "PM-1", "status": "fail"}]},
        telemetry_summary={"signals": [{"sensor_id": "S-1", "anomalies": [{"pattern": "spike"}]}]},
    )
    for key in ("applicable", "barriers", "failed_or_missing_barriers", "why_not_prevented", "provenance_note"):
        assert key in block
    for b in block["barriers"]:
        assert set(b).issubset({"barrier_type", "status", "basis", "detail"})
        assert {"barrier_type", "status", "basis"}.issubset(b)
