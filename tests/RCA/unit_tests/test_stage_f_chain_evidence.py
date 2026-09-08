"""
Contract checks for Stage B.5 -> Stage F evidence blending.

Run:
  python test_stage_f_chain_evidence.py
"""
import sys
from pathlib import Path
from typing import Any, Dict

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from orchestrators.causality_engine_v32 import CausalityEngineConfigV32, RuleBasedCausalityEngineV32


def _base_candidates() -> Dict[str, Any]:
    return {
        "candidates": [
            {
                "candidate_id": "FM::FM-A",
                "hypothesis_type": "failure_mode",
                "cause_node_id": "FM-A",
                "scores": {"evidence": 0.2, "governance": 0.5},
                "composite_score": 0.5,
            }
        ],
        "summary": {},
    }


def _base_evidence_bundle() -> Dict[str, Any]:
    return {
        "candidate_evidence_summary": [
            {
                "candidate_id": "FM::FM-A",
                "best_support_score": 0.8,
                "best_contradiction_score": 0.0,
                "best_context_score": 0.2,
                "hit_count": 2,
            }
        ]
    }


def test_chain_score_present_increases_evidence_for_root_candidate():
    engine = RuleBasedCausalityEngineV32(
        CausalityEngineConfigV32(
            minimum_evidence_threshold=0.0,
            minimum_pre_evidence_threshold=0.0,
            minimum_composite_threshold=0.0,
            top_k_candidates=5,
        )
    )
    out = engine.refine_with_evidence(
        causality_candidates=_base_candidates(),
        evidence_bundle=_base_evidence_bundle(),
        kg_context={"failure_modes": []},
        signal_evidence={
            "per_candidate_chain_score": {
                "FM-A": {"chain_position_score": 1.0, "position_type": "root"}
            }
        },
    )
    cand = out["candidates"][0]
    e_doc = float(cand["scores"]["evidence_doc"])
    e_new = float(cand["scores"]["evidence"])
    assert e_new > e_doc
    assert float(cand["scores"]["evidence_chain"]) == 1.0
    print("  PASS test_chain_score_present_increases_evidence_for_root_candidate")


def test_chain_score_absent_keeps_evidence_equal_to_doc():
    engine = RuleBasedCausalityEngineV32(
        CausalityEngineConfigV32(
            minimum_evidence_threshold=0.0,
            minimum_pre_evidence_threshold=0.0,
            minimum_composite_threshold=0.0,
            top_k_candidates=5,
        )
    )
    out = engine.refine_with_evidence(
        causality_candidates=_base_candidates(),
        evidence_bundle=_base_evidence_bundle(),
        kg_context={"failure_modes": []},
        signal_evidence={"per_candidate_chain_score": {}},
    )
    cand = out["candidates"][0]
    e_doc = float(cand["scores"]["evidence_doc"])
    e_new = float(cand["scores"]["evidence"])
    assert abs(e_new - e_doc) < 1e-9
    assert float(cand["scores"]["evidence_chain"]) == 0.0
    print("  PASS test_chain_score_absent_keeps_evidence_equal_to_doc")


def test_contributing_cause_passthrough_stage_f():
    engine = RuleBasedCausalityEngineV32(
        CausalityEngineConfigV32(
            minimum_evidence_threshold=0.0,
            minimum_pre_evidence_threshold=0.0,
            minimum_composite_threshold=0.0,
            top_k_candidates=5,
        )
    )
    out = engine.refine_with_evidence(
        causality_candidates=_base_candidates(),
        evidence_bundle=_base_evidence_bundle(),
        kg_context={"failure_modes": []},
        signal_evidence={
            "per_candidate_chain_score": {
                "FM-A": {
                    "chain_position_score": 0.4,
                    "position_type": "intermediate",
                    "contributing_cause_role": "concurrent_cause_candidate",
                    "confluence_component_id": "C-B",
                }
            }
        },
    )
    cand = out["candidates"][0]
    assert cand.get("is_contributing_cause_candidate") is True
    assert cand.get("confluence_component_id") == "C-B"
    print("  PASS test_contributing_cause_passthrough_stage_f")


ALL_TESTS = [
    test_chain_score_present_increases_evidence_for_root_candidate,
    test_chain_score_absent_keeps_evidence_equal_to_doc,
    test_contributing_cause_passthrough_stage_f,
]


def run_all() -> bool:
    print(f"\n=== test_stage_f_chain_evidence ({len(ALL_TESTS)} tests) ===")
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
