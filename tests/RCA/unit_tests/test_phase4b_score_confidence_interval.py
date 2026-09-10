"""
test_phase4b_score_confidence_interval.py — unit tests for Issue 14 (Phase 4b):
per-candidate score_confidence_interval derived from data-degradation signals.

Tests:
  TestApplyScoreConfidenceInterval (14 tests) — _apply_score_confidence_interval
  TestScoreConfidenceIntervalIntegration (3 tests) — field present after refine_with_evidence
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

from orchestrators.causality_engine_v32 import (
    RuleBasedCausalityEngineV32,
    CausalityEngineConfigV32,
)

ENGINE = RuleBasedCausalityEngineV32


# ── helpers ───────────────────────────────────────────────────────────────────

def _candidate(
    composite: float = 0.60,
    *,
    structural_degraded: bool = False,
    temporal_quality: str = "full_allen",
    telemetry: float = 0.40,
    observationally_ungrounded: bool = False,
    governance_degraded: bool = False,
) -> dict:
    return {
        "candidate_id": "FM::TEST",
        "component_id": "COMP-X",
        "composite_score": composite,
        "observationally_ungrounded": observationally_ungrounded,
        "scores": {
            "temporal_score_quality": temporal_quality,
            "telemetry": telemetry,
        },
        "hard_gates": {
            "physical_plausibility": {"degraded_mode": structural_degraded},
            "barrier_logic": {"degraded_mode": governance_degraded},
        },
    }


def _sci(candidate: dict) -> dict:
    ENGINE._apply_score_confidence_interval(candidate)
    return candidate["score_confidence_interval"]


# ── TestApplyScoreConfidenceInterval ─────────────────────────────────────────

class TestApplyScoreConfidenceInterval:

    def test_zero_degraded_width_zero(self):
        sci = _sci(_candidate(0.60, temporal_quality="full_allen", telemetry=0.40))
        assert sci["width"] == 0.0
        assert sci["degraded_dimension_count"] == 0
        assert sci["lower"] == 0.60
        assert sci["upper"] == 0.60

    def test_all_five_degraded_width_one(self):
        sci = _sci(_candidate(
            0.50,
            structural_degraded=True,
            temporal_quality="proxy",
            telemetry=0.0,
            observationally_ungrounded=True,
            governance_degraded=True,
        ))
        assert sci["width"] == 1.0
        assert sci["degraded_dimension_count"] == 5

    def test_one_degraded_temporal_proxy(self):
        sci = _sci(_candidate(0.60, temporal_quality="proxy"))
        assert sci["degraded_dimension_count"] == 1
        assert abs(sci["width"] - 0.2) < 1e-6

    def test_two_degraded_width_point_four(self):
        sci = _sci(_candidate(0.50, temporal_quality="proxy", observationally_ungrounded=True))
        assert sci["degraded_dimension_count"] == 2
        assert abs(sci["width"] - 0.4) < 1e-6

    def test_structural_degraded_counted(self):
        sci = _sci(_candidate(0.50, structural_degraded=True, temporal_quality="full_allen"))
        assert "structural" in sci["degraded_dimensions"]
        assert sci["degraded_dimension_count"] >= 1

    def test_temporal_proxy_counted(self):
        sci = _sci(_candidate(0.50, temporal_quality="proxy"))
        assert "temporal" in sci["degraded_dimensions"]

    def test_temporal_full_allen_not_counted(self):
        sci = _sci(_candidate(0.50, temporal_quality="full_allen"))
        assert "temporal" not in sci["degraded_dimensions"]

    def test_telemetry_zero_counted(self):
        sci = _sci(_candidate(0.50, telemetry=0.0, temporal_quality="full_allen"))
        assert "telemetry" in sci["degraded_dimensions"]

    def test_telemetry_nonzero_not_counted(self):
        sci = _sci(_candidate(0.50, telemetry=0.35, temporal_quality="full_allen"))
        assert "telemetry" not in sci["degraded_dimensions"]

    def test_evidence_ungrounded_counted(self):
        sci = _sci(_candidate(0.50, observationally_ungrounded=True, temporal_quality="full_allen"))
        assert "evidence" in sci["degraded_dimensions"]

    def test_governance_degraded_counted(self):
        sci = _sci(_candidate(0.50, governance_degraded=True, temporal_quality="full_allen"))
        assert "governance" in sci["degraded_dimensions"]

    def test_lower_clamps_at_zero(self):
        # composite=0.05, all five degraded → width=1.0, naive lower=-0.45
        sci = _sci(_candidate(
            0.05,
            structural_degraded=True, temporal_quality="proxy",
            telemetry=0.0, observationally_ungrounded=True, governance_degraded=True,
        ))
        assert sci["lower"] >= 0.0

    def test_upper_clamps_at_one(self):
        # composite=0.95, all five degraded → width=1.0, naive upper=1.45
        sci = _sci(_candidate(
            0.95,
            structural_degraded=True, temporal_quality="proxy",
            telemetry=0.0, observationally_ungrounded=True, governance_degraded=True,
        ))
        assert sci["upper"] <= 1.0

    def test_degraded_dimensions_list_contents(self):
        # telemetry=0.40 (default, non-zero) → not degraded; only temporal + evidence
        sci = _sci(_candidate(0.50, temporal_quality="proxy", observationally_ungrounded=True))
        assert set(sci["degraded_dimensions"]) == {"temporal", "evidence"}

    def test_interval_symmetric_when_unclamped(self):
        # composite=0.50, 2 degraded → width=0.4, lower=0.30, upper=0.70
        sci = _sci(_candidate(0.50, temporal_quality="proxy", observationally_ungrounded=True))
        assert abs(sci["lower"] - 0.30) < 1e-5
        assert abs(sci["upper"] - 0.70) < 1e-5


# ── TestScoreConfidenceIntervalIntegration ────────────────────────────────────

def _make_candidates(component_id: str = "COMP-A") -> dict:
    return {
        "event_id": "EVT-001",
        "candidates": [
            {
                "candidate_id": f"FM::{component_id}",
                "component_id": component_id,
                "hypothesis_type": "failure_mode",
                "composite_score": 0.55,
                "quality_multiplier": 1.0,
                "primary_causal_category": "A",
                "chain_position": "proximate",
                "canonical_tuple": {},
                "canonical_candidate_key": f"A::proximate::{component_id}",
                "scores": {
                    "temporal": 0.30,
                    "structural": 0.80,
                    "evidence": 0.40,
                    "telemetry": 0.30,
                    "governance": 0.20,
                    "composite_raw": 0.55,
                },
                "temporal_evidence": {},
            }
        ],
        "filtered_out_candidates": [],
        "event_analogs": [],
        "summary": {},
        "category_coverage": {},
        "applicability_assessment": {},
    }


def _make_evidence_bundle() -> dict:
    return {"per_candidate_summary": [], "retrieval_run_id": "RUN-001"}


class TestScoreConfidenceIntervalIntegration:

    def _engine(self):
        return RuleBasedCausalityEngineV32(CausalityEngineConfigV32(
            minimum_evidence_threshold=0.05,
            minimum_composite_threshold=0.20,
        ))

    def test_field_present_after_refine(self):
        result = self._engine().refine_with_evidence(
            causality_candidates=_make_candidates(),
            evidence_bundle=_make_evidence_bundle(),
        )
        all_cands = (result.get("candidates") or []) + (result.get("filtered_out_candidates") or [])
        assert len(all_cands) > 0
        assert "score_confidence_interval" in all_cands[0]

    def test_sci_keys_present(self):
        result = self._engine().refine_with_evidence(
            causality_candidates=_make_candidates(),
            evidence_bundle=_make_evidence_bundle(),
        )
        all_cands = (result.get("candidates") or []) + (result.get("filtered_out_candidates") or [])
        sci = all_cands[0]["score_confidence_interval"]
        for key in ("lower", "upper", "width", "degraded_dimension_count", "degraded_dimensions"):
            assert key in sci, f"Missing key: {key}"

    def test_width_bounded_zero_to_one(self):
        result = self._engine().refine_with_evidence(
            causality_candidates=_make_candidates(),
            evidence_bundle=_make_evidence_bundle(),
        )
        all_cands = (result.get("candidates") or []) + (result.get("filtered_out_candidates") or [])
        sci = all_cands[0]["score_confidence_interval"]
        assert 0.0 <= sci["width"] <= 1.0
        assert 0.0 <= sci["lower"] <= sci["upper"] <= 1.0
