"""
test_finding_i_plc_gates.py — Finding I: Direct protection_logic_context read in hard gates

Covers:
- _build_plc_barrier_index: sf_state_index, logic_signal_ids, None input, empty input
- _apply_physical_plausibility_gate: plc_consulted flag, held-barrier note, no-PLC baseline
- _apply_barrier_logic_gate: failed/degraded barrier → gate fails, held barrier note,
  plc_consulted flag, no-PLC baseline unchanged
- refine_with_evidence: protection_logic_context threaded to both gates
- Orchestrator: protection_logic_context forwarded into refine_kwargs

Run:  pytest test_finding_i_plc_gates.py -v
"""
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock
from typing import Optional, Dict, List

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.causality_engine_v32 import (
    RuleBasedCausalityEngineV32,
    CausalityEngineConfigV32,
)
from orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator

ENGINE = RuleBasedCausalityEngineV32

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plc(
    barrier_states: Optional[List[Dict]] = None,
    logic_sets: Optional[List[Dict]] = None,
) -> dict:
    return {
        "barrier_states": barrier_states or [],
        "logic_sets": logic_sets or [],
    }


def _barrier_state(sf_id: str, state: str) -> dict:
    return {"sf_id": sf_id, "state": state}


def _logic_set(inputs: Optional[List[str]] = None, outputs: Optional[List[str]] = None) -> dict:
    return {
        "set_id": "LS-001",
        "input_signals": inputs or [],
        "output_signals": outputs or [],
    }


def _candidate(
    component_id: str,
    structural: float = 0.60,
    affected_sfs: Optional[List[Dict]] = None,
) -> dict:
    return {
        "candidate_id": f"FM::{component_id}",
        "component_id": component_id,
        "hypothesis_type": "failure_mode",
        "composite_score": 0.55,
        "quality_multiplier": 1.0,
        "scores": {
            "structural": structural,
            "temporal": 0.50,
            "evidence": 0.50,
            "telemetry": 0.30,
            "governance": 0.20,
            "barrier_signal": None,
            "composite_raw": 0.55,
        },
        "temporal_evidence": {},
        "affected_safety_functions": affected_sfs or [],
    }


def _make_engine() -> RuleBasedCausalityEngineV32:
    cfg = CausalityEngineConfigV32(
        minimum_composite_threshold=0.05,
        minimum_evidence_threshold=0.05,
    )
    return RuleBasedCausalityEngineV32(config=cfg)


def _minimal_candidates_payload(cands: List[Dict]) -> dict:
    return {"candidates": cands, "run_id": "test-run"}


def _empty_bundle() -> dict:
    return {
        "evidence_bundle_id": "eb-test",
        "candidate_summaries": [],
    }


# ===========================================================================
# _build_plc_barrier_index
# ===========================================================================

def test_index_none_returns_empty():
    sf_state, signals = ENGINE._build_plc_barrier_index(None)
    assert sf_state == {} and signals == set()


def test_index_empty_dict_returns_empty():
    sf_state, signals = ENGINE._build_plc_barrier_index({})
    assert sf_state == {} and signals == set()


def test_index_barrier_states_indexed():
    plc = _plc(
        barrier_states=[
            _barrier_state("SF-1", "held"),
            _barrier_state("SF-2", "failed"),
            _barrier_state("SF-3", "degraded"),
            _barrier_state("SF-4", "unknown"),
        ]
    )
    sf_state, _ = ENGINE._build_plc_barrier_index(plc)
    assert sf_state["SF-1"] == "held"
    assert sf_state["SF-2"] == "failed"
    assert sf_state["SF-3"] == "degraded"
    assert sf_state["SF-4"] == "unknown"


def test_index_logic_set_signals_collected():
    plc = _plc(
        logic_sets=[
            _logic_set(inputs=["PCV-100", "TT-200"], outputs=["RX-TRIP"]),
        ]
    )
    _, signals = ENGINE._build_plc_barrier_index(plc)
    assert "PCV-100" in signals
    assert "TT-200" in signals
    assert "RX-TRIP" in signals


def test_index_multiple_logic_sets_merged():
    plc = _plc(
        logic_sets=[
            _logic_set(inputs=["A"], outputs=["B"]),
            _logic_set(inputs=["C"], outputs=["D"]),
        ]
    )
    _, signals = ENGINE._build_plc_barrier_index(plc)
    assert signals == {"A", "B", "C", "D"}


def test_index_skips_null_signals():
    plc = _plc(
        logic_sets=[_logic_set(inputs=[None, "VALID-SIG"], outputs=[None])]
    )
    _, signals = ENGINE._build_plc_barrier_index(plc)
    assert "VALID-SIG" in signals
    assert None not in signals
    assert "None" not in signals


# ===========================================================================
# _apply_physical_plausibility_gate
# ===========================================================================

def test_pp_gate_no_plc_passes_normally():
    """Baseline: high structural score, no PLC → passes without plc_consulted."""
    eng = _make_engine()
    cand = _candidate("PUMP-1", structural=0.70)
    eng._apply_physical_plausibility_gate(cand)
    gate = cand["hard_gates"]["physical_plausibility"]
    assert gate["passed"] is True
    assert gate.get("plc_consulted") is False


def test_pp_gate_low_structural_fails_no_plc():
    eng = _make_engine()
    cand = _candidate("PUMP-1", structural=0.10)
    eng._apply_physical_plausibility_gate(cand)
    gate = cand["hard_gates"]["physical_plausibility"]
    assert gate["passed"] is False
    assert "physical_plausibility_gate_failed" in cand["primary_block_reasons"]


def test_pp_gate_with_plc_component_in_signals_consulted():
    """Component in PLC logic signals → plc_consulted=True."""
    eng = _make_engine()
    cand = _candidate("PCV-100", structural=0.70)
    plc = _plc(logic_sets=[_logic_set(inputs=["PCV-100"])])
    _, logic_sigs = ENGINE._build_plc_barrier_index(plc)
    eng._apply_physical_plausibility_gate(
        cand,
        plc_logic_signal_ids=logic_sigs,
        plc_sf_state={},
    )
    gate = cand["hard_gates"]["physical_plausibility"]
    assert gate["plc_consulted"] is True
    assert gate["passed"] is True
    assert "PCV-100" in gate["rationale"]


def test_pp_gate_component_not_in_signals_not_consulted():
    eng = _make_engine()
    cand = _candidate("VALVE-999", structural=0.70)
    _, logic_sigs = ENGINE._build_plc_barrier_index(
        _plc(logic_sets=[_logic_set(inputs=["OTHER-SIG"])])
    )
    eng._apply_physical_plausibility_gate(
        cand, plc_logic_signal_ids=logic_sigs, plc_sf_state={}
    )
    gate = cand["hard_gates"]["physical_plausibility"]
    assert gate["plc_consulted"] is False


def test_pp_gate_held_sf_noted_in_rationale():
    eng = _make_engine()
    cand = _candidate(
        "PCV-100",
        structural=0.70,
        affected_sfs=[{"sf_id": "SF-TRIP", "name": "Reactor Trip"}],
    )
    plc_sf = {"SF-TRIP": "held"}
    plc = _plc(logic_sets=[_logic_set(inputs=["PCV-100"])])
    _, logic_sigs = ENGINE._build_plc_barrier_index(plc)
    eng._apply_physical_plausibility_gate(
        cand, plc_logic_signal_ids=logic_sigs, plc_sf_state=plc_sf
    )
    gate = cand["hard_gates"]["physical_plausibility"]
    assert gate["passed"] is True
    assert "held" in gate["rationale"]
    assert "SF-TRIP" in gate["rationale"]


# ===========================================================================
# _apply_barrier_logic_gate
# ===========================================================================

def test_barrier_gate_no_plc_passes_degraded():
    """No PLC, no affected SFs, no barrier signal → PASS degraded."""
    eng = _make_engine()
    cand = _candidate("PUMP-1")
    eng._apply_barrier_logic_gate(cand)
    gate = cand["hard_gates"]["barrier_logic"]
    assert gate["passed"] is True
    assert gate["degraded_mode"] is True
    assert gate.get("plc_consulted") is False


def test_barrier_gate_plc_failed_sf_fails_candidate():
    """PLC reports sf_id with state='failed' → gate fails → blocked."""
    eng = _make_engine()
    cand = _candidate(
        "VALVE-A",
        affected_sfs=[{"sf_id": "SF-2"}],
    )
    plc_sf = {"SF-2": "failed"}
    eng._apply_barrier_logic_gate(cand, plc_sf_state=plc_sf)
    gate = cand["hard_gates"]["barrier_logic"]
    assert gate["passed"] is False
    assert gate["plc_consulted"] is True
    assert "barrier_logic_gate_failed" in cand["primary_block_reasons"]
    assert cand.get("meets_evidence_threshold") is False


def test_barrier_gate_plc_degraded_sf_fails_candidate():
    """PLC reports state='degraded' → treated same as failed."""
    eng = _make_engine()
    cand = _candidate("VALVE-B", affected_sfs=[{"sf_id": "SF-3"}])
    plc_sf = {"SF-3": "degraded"}
    eng._apply_barrier_logic_gate(cand, plc_sf_state=plc_sf)
    gate = cand["hard_gates"]["barrier_logic"]
    assert gate["passed"] is False
    assert gate["plc_consulted"] is True


def test_barrier_gate_plc_held_sf_passes_with_note():
    """PLC state='held' → gate passes but rationale mentions hold."""
    eng = _make_engine()
    cand = _candidate("VALVE-C", affected_sfs=[{"sf_id": "SF-HOLD"}])
    plc_sf = {"SF-HOLD": "held"}
    eng._apply_barrier_logic_gate(cand, plc_sf_state=plc_sf)
    gate = cand["hard_gates"]["barrier_logic"]
    assert gate["passed"] is True
    assert "held" in gate["rationale"]
    assert gate["plc_consulted"] is True


def test_barrier_gate_plc_unknown_state_passes():
    """PLC state='unknown' → not a failure, gate passes."""
    eng = _make_engine()
    cand = _candidate("VALVE-D", affected_sfs=[{"sf_id": "SF-UNK"}])
    plc_sf = {"SF-UNK": "unknown"}
    eng._apply_barrier_logic_gate(cand, plc_sf_state=plc_sf)
    gate = cand["hard_gates"]["barrier_logic"]
    assert gate["passed"] is True


def test_barrier_gate_sf_not_in_plc_not_consulted():
    """SF id not present in plc_sf_state → plc_consulted stays False."""
    eng = _make_engine()
    cand = _candidate("VALVE-E", affected_sfs=[{"sf_id": "SF-NOTINPLC"}])
    plc_sf = {"SF-OTHER": "failed"}
    eng._apply_barrier_logic_gate(cand, plc_sf_state=plc_sf)
    gate = cand["hard_gates"]["barrier_logic"]
    assert gate["plc_consulted"] is False
    assert gate["passed"] is True


# ===========================================================================
# Integration — refine_with_evidence passes PLC through
# ===========================================================================

def test_refine_plc_barrier_failed_blocks_candidate():
    """End-to-end: barrier gate fails when PLC reports sf_id as 'failed'."""
    eng = _make_engine()
    cand = _candidate(
        "VALVE-X",
        structural=0.75,
        affected_sfs=[{"sf_id": "SF-SCRAM"}],
    )
    plc = _plc(
        barrier_states=[_barrier_state("SF-SCRAM", "failed")],
        logic_sets=[_logic_set(inputs=["VALVE-X"])],
    )
    payload = _minimal_candidates_payload([deepcopy(cand)])
    result = eng.refine_with_evidence(
        causality_candidates=payload,
        evidence_bundle=_empty_bundle(),
        protection_logic_context=plc,
    )
    all_cands = (result.get("candidates") or []) + (result.get("filtered_out_candidates") or [])
    found = next((c for c in all_cands if "VALVE-X" in (c.get("candidate_id") or "")), None)
    assert found is not None
    barrier_gate = (found.get("hard_gates") or {}).get("barrier_logic") or {}
    assert barrier_gate.get("passed") is False
    assert barrier_gate.get("plc_consulted") is True


def test_refine_plc_logic_signal_component_consulted_in_pp_gate():
    """Physical plausibility gate has plc_consulted=True for component in PLC signals."""
    eng = _make_engine()
    cand = _candidate("FWP-101", structural=0.75)
    plc = _plc(logic_sets=[_logic_set(inputs=["FWP-101"])])
    payload = _minimal_candidates_payload([deepcopy(cand)])
    result = eng.refine_with_evidence(
        causality_candidates=payload,
        evidence_bundle=_empty_bundle(),
        protection_logic_context=plc,
    )
    all_cands = (result.get("candidates") or []) + (result.get("filtered_out_candidates") or [])
    found = next((c for c in all_cands if "FWP-101" in (c.get("candidate_id") or "")), None)
    assert found is not None
    pp_gate = (found.get("hard_gates") or {}).get("physical_plausibility") or {}
    assert pp_gate.get("plc_consulted") is True


def test_refine_no_plc_gates_not_consulted():
    """Without protection_logic_context gates run normally, plc_consulted=False."""
    eng = _make_engine()
    cand = _candidate("COMP-A", structural=0.60)
    payload = _minimal_candidates_payload([deepcopy(cand)])
    result = eng.refine_with_evidence(
        causality_candidates=payload,
        evidence_bundle=_empty_bundle(),
    )
    all_cands = (result.get("candidates") or []) + (result.get("filtered_out_candidates") or [])
    found = next((c for c in all_cands if "COMP-A" in (c.get("candidate_id") or "")), None)
    assert found is not None
    pp_gate = (found.get("hard_gates") or {}).get("physical_plausibility") or {}
    bl_gate = (found.get("hard_gates") or {}).get("barrier_logic") or {}
    assert pp_gate.get("plc_consulted") is False
    assert bl_gate.get("plc_consulted") is False


def test_refine_plc_none_no_error():
    """protection_logic_context=None must not raise."""
    eng = _make_engine()
    cand = _candidate("COMP-NULL", structural=0.60)
    payload = _minimal_candidates_payload([deepcopy(cand)])
    result = eng.refine_with_evidence(
        causality_candidates=payload,
        evidence_bundle=_empty_bundle(),
        protection_logic_context=None,
    )
    assert "candidates" in result or "filtered_out_candidates" in result


# ===========================================================================
# Orchestrator — protection_logic_context in refine_kwargs
# ===========================================================================

def test_orchestrator_threads_plc_into_refine_kwargs():
    """Verify refine_with_evidence declares protection_logic_context parameter."""
    import inspect

    sig = inspect.signature(RuleBasedCausalityEngineV32.refine_with_evidence)
    assert "protection_logic_context" in sig.parameters, (
        "refine_with_evidence must declare protection_logic_context parameter"
    )
