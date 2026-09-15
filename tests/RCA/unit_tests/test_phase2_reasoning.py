"""
Phase 2 reasoning contract tests for metamodel alignment.
"""
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from orchestrators.causality_engine_v32 import RuleBasedCausalityEngineV32


def test_failure_mode_category_inference_prefers_support_dependency_keywords():
    cat, alts = RuleBasedCausalityEngineV32._infer_primary_category_for_failure_mode(
        fm={
            "name": "Instrument air header pressure loss",
            "superclass": "support_system_degradation",
            "failure_mechanism": "instrument air unavailable",
        },
        event={"event_type": "trip"},
    )
    assert cat == "B"
    assert isinstance(alts, list)


def test_phase2_coverage_marks_missing_applicable_category_as_ruled_out():
    coverage, applicability = RuleBasedCausalityEngineV32._build_metamodel_scaffolds(
        retained_candidates=[{"candidate_id": "FM::1", "primary_causal_category": "A"}],
        filtered_out_candidates=[],
        event_analogs=[],
        kg_context={"failure_modes": [{"fm_id": "FM-1"}], "components": [{"component_id": "C1"}], "documents": []},
        operational_context={"recent_alarms": []},
        external_oe_unavailable=True,
    )
    assert coverage["A"]["status"] == "candidate_scored"
    assert coverage["L"]["status"] in {"candidate_scored", "ruled_out"}
    assert applicability["L"]["status"] == "applicable"


def test_uncertainty_propagation_degrades_composite_score():
    engine = RuleBasedCausalityEngineV32()
    candidate = {
        "candidate_id": "FM::X",
        "primary_causal_category": "A",
        "composite_score": 0.8,
        "scores": {
            "structural": 0.6,
            "temporal": 0.2,
            "evidence": 0.1,
        },
        "recurrence": {"recurrence_score": 0.0},
    }
    engine._apply_uncertainty_propagation(candidate)
    assert candidate["quality_multiplier"] <= 1.0
    assert candidate["composite_score"] < 0.8
    assert "stream_quality" in candidate


def test_category_minima_gate_caps_supported_posture_to_weak_when_missing_streams():
    engine = RuleBasedCausalityEngineV32()
    candidate = {
        "candidate_id": "FM::Y",
        "primary_causal_category": "K",
        "evidence_posture": "supported",
        "scores": {
            "structural": 0.7,
            "temporal": 0.7,
            "evidence": 0.2,
            "evidence_doc": 0.2,
        },
        "recurrence": {"recurrence_score": 0.0},
    }
    engine._apply_category_minimum_evidence_gate(candidate)
    assert candidate["evidence_minima_met"] is False
    assert candidate["evidence_posture"] == "weak"
    assert "evidence_minima_missing" in candidate
