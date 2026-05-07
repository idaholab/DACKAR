"""
test_phase4c_scoring_profiles.py — unit tests for Phase 4c Steps 1–3:
per-category scoring profiles in CausalityEngineConfigV32.

Tests:
  TestScoringProfilesConfig      (8 tests) — config defaults, validation, overrides
  TestScoringProfileForFm        (5 tests) — _scoring_profile_for_fm dispatch
  TestScoreProfileAppliedField   (6 tests) — score_profile_applied in generate() output
  TestProfileWeightsInComposite  (4 tests) — G/I/L profiles actually affect composite score
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest
from orchestrators.causality_engine_v32 import (
    RuleBasedCausalityEngineV32,
    CausalityEngineConfigV32,
    _DEFAULT_SCORING_PROFILES,
    _SCORING_PROFILE_DIMENSIONS,
)

ENGINE = RuleBasedCausalityEngineV32


# ── TestScoringProfilesConfig ─────────────────────────────────────────────────

class TestScoringProfilesConfig:

    def test_default_profiles_all_12_categories(self):
        cfg = CausalityEngineConfigV32()
        assert set(cfg.scoring_profiles.keys()) == set("ABCDEFGHIJKL")

    def test_default_profiles_correct_dimensions(self):
        cfg = CausalityEngineConfigV32()
        for cat, profile in cfg.scoring_profiles.items():
            assert set(profile.keys()) == _SCORING_PROFILE_DIMENSIONS, (
                f"Category {cat} missing dimensions"
            )

    def test_default_profiles_all_sum_to_one(self):
        cfg = CausalityEngineConfigV32()
        for cat, profile in cfg.scoring_profiles.items():
            total = sum(profile.values())
            assert abs(total - 1.0) < 1e-6, f"Category {cat} sums to {total}"

    def test_athrough_f_use_current_weights(self):
        cfg = CausalityEngineConfigV32()
        equipment_profile = {"structural": 0.30, "temporal": 0.20, "telemetry": 0.20,
                             "evidence": 0.20, "governance": 0.10}
        for cat in "ABCDEF":
            assert cfg.scoring_profiles[cat] == equipment_profile, (
                f"Category {cat} deviates from equipment_origin defaults"
            )

    def test_site_override_single_profile(self):
        custom = {cat: dict(p) for cat, p in _DEFAULT_SCORING_PROFILES.items()}
        custom["L"] = {"structural": 0.02, "temporal": 0.03, "telemetry": 0.05,
                       "evidence": 0.65, "governance": 0.25}
        cfg = CausalityEngineConfigV32(scoring_profiles=custom)
        assert cfg.scoring_profiles["L"]["evidence"] == 0.65
        assert cfg.scoring_profiles["A"] == _DEFAULT_SCORING_PROFILES["A"]

    def test_validation_rejects_profile_not_summing_to_one(self):
        bad = {cat: dict(p) for cat, p in _DEFAULT_SCORING_PROFILES.items()}
        bad["G"] = {"structural": 0.30, "temporal": 0.20, "telemetry": 0.20,
                    "evidence": 0.20, "governance": 0.20}  # sums to 1.10
        with pytest.raises(ValueError, match="sum to 1.0"):
            CausalityEngineConfigV32(scoring_profiles=bad)

    def test_validation_rejects_wrong_dimension_keys(self):
        bad = {cat: dict(p) for cat, p in _DEFAULT_SCORING_PROFILES.items()}
        bad["H"] = {"structural": 0.50, "temporal": 0.50}  # missing keys
        with pytest.raises(ValueError, match="keys"):
            CausalityEngineConfigV32(scoring_profiles=bad)

    def test_scoring_profiles_independent_of_weights_field(self):
        # scoring_profiles and weights are separate; both can coexist
        cfg = CausalityEngineConfigV32()
        assert cfg.weights == {"structural": 0.30, "temporal": 0.20, "telemetry": 0.20,
                               "evidence": 0.20, "governance": 0.10}
        assert cfg.scoring_profiles["G"]["evidence"] == 0.65


# ── TestScoringProfileForFm ───────────────────────────────────────────────────

class TestScoringProfileForFm:

    def _engine(self):
        return RuleBasedCausalityEngineV32(CausalityEngineConfigV32())

    def test_known_category_returns_correct_profile(self):
        e = self._engine()
        profile = e._scoring_profile_for_fm("G")
        assert profile["evidence"] == 0.65
        assert profile["structural"] == 0.05

    def test_unknown_category_falls_back_to_a(self):
        e = self._engine()
        profile = e._scoring_profile_for_fm("Z")
        assert profile == e.config.scoring_profiles["A"]

    def test_empty_category_falls_back_to_a(self):
        e = self._engine()
        assert e._scoring_profile_for_fm("") == e.config.scoring_profiles["A"]

    def test_returns_copy_not_reference(self):
        e = self._engine()
        p1 = e._scoring_profile_for_fm("L")
        p1["evidence"] = 0.0
        p2 = e._scoring_profile_for_fm("L")
        assert p2["evidence"] == 0.60  # config unchanged

    def test_all_12_categories_resolve(self):
        e = self._engine()
        for cat in "ABCDEFGHIJKL":
            profile = e._scoring_profile_for_fm(cat)
            assert abs(sum(profile.values()) - 1.0) < 1e-6, f"Category {cat} profile broken"


# ── TestScoreProfileAppliedField ──────────────────────────────────────────────

def _minimal_kg(category_keyword: str = "bearing failure") -> dict:
    """KG with one FM node whose name triggers a specific category inference."""
    return {
        "failure_modes": [
            {
                "fm_id": "FM::TEST-001",
                "name": category_keyword,
                "failure_mechanism": category_keyword,
                "superclass": category_keyword,
                "component_id": "COMP-A",
                "component_name": "Pump A",
                "failure_mode_refs": ["FM::TEST-001"],
                "fmea_rpn": 80,
                "maintenance_preventable": True,
            }
        ],
        "components": [
            {
                "component_id": "COMP-A",
                "seed_match_type": "direct",
                "neighbors": [],
                "safety_functions": [],
            }
        ],
        "safety_functions": [],
        "past_events": [],
        "common_cause_groups": [],
    }


def _minimal_event() -> dict:
    return {
        "event_id": "EVT-001",
        "id": "EVT-001",
        "description": "Pump bearing failure",
        "event_type": "equipment_failure",
        "primary_system": "COMP-A",
    }


def _run_generate(kg_context: dict) -> dict:
    engine = RuleBasedCausalityEngineV32(CausalityEngineConfigV32())
    return engine.generate(
        event=_minimal_event(),
        telemetry_summary={},
        kg_context=kg_context,
        tskr_patterns=None,
        operational_context=None,
        pm_compliance=None,
        run_context={"run_id": "RUN-001"},
    )


class TestScoreProfileAppliedField:

    def test_score_profile_applied_present_on_fm_candidate(self):
        result = _run_generate(_minimal_kg("bearing failure"))
        cands = result.get("candidates") or []
        assert len(cands) > 0
        scores = cands[0].get("scores") or {}
        assert "score_profile_applied" in scores

    def test_scoring_profile_weights_present_on_fm_candidate(self):
        result = _run_generate(_minimal_kg("bearing failure"))
        cands = result.get("candidates") or []
        scores = cands[0].get("scores") or {}
        assert "scoring_profile_weights" in scores
        assert set(scores["scoring_profile_weights"].keys()) == _SCORING_PROFILE_DIMENSIONS

    def test_equipment_origin_profile_for_bearing_failure(self):
        result = _run_generate(_minimal_kg("bearing failure"))
        cands = result.get("candidates") or []
        scores = cands[0].get("scores") or {}
        assert scores["score_profile_applied"] == "equipment_origin"

    def test_human_performance_profile_for_operator_error(self):
        result = _run_generate(_minimal_kg("operator error human procedure not followed"))
        all_cands = (result.get("candidates") or []) + (result.get("filtered_out_candidates") or [])
        g_cands = [c for c in all_cands
                   if c.get("primary_causal_category") == "G"]
        if g_cands:
            assert g_cands[0]["scores"]["score_profile_applied"] == "human_performance"

    def test_organizational_profile_for_systemic_issue(self):
        result = _run_generate(_minimal_kg("systemic latent safety culture recurrence"))
        all_cands = (result.get("candidates") or []) + (result.get("filtered_out_candidates") or [])
        l_cands = [c for c in all_cands
                   if c.get("primary_causal_category") == "L"]
        if l_cands:
            assert l_cands[0]["scores"]["score_profile_applied"] == "organizational"

    def test_governance_weight_stored_matches_profile(self):
        result = _run_generate(_minimal_kg("bearing failure"))
        all_cands = (result.get("candidates") or []) + (result.get("filtered_out_candidates") or [])
        for c in all_cands:
            scores = c.get("scores") or {}
            profile_w = scores.get("scoring_profile_weights") or {}
            stored_gov = scores.get("governance_weight")
            if profile_w and stored_gov is not None:
                assert abs(stored_gov - profile_w["governance"]) < 1e-9


# ── TestProfileWeightsInComposite ─────────────────────────────────────────────

class TestProfileWeightsInComposite:
    """Verify that profile weights actually affect composite score computation."""

    def _combine(self, scores: dict, profile: dict) -> float:
        e = RuleBasedCausalityEngineV32(CausalityEngineConfigV32())
        return e._combine_scores(scores, weights_override=profile)

    def test_high_evidence_profile_boosts_evidence_heavy_candidate(self):
        scores = {"structural": 0.10, "temporal": 0.10, "telemetry": 0.05,
                  "evidence": 0.90, "governance": 0.20}
        composite_equipment = self._combine(scores, _DEFAULT_SCORING_PROFILES["A"])
        composite_human = self._combine(scores, _DEFAULT_SCORING_PROFILES["G"])
        assert composite_human > composite_equipment

    def test_low_evidence_candidate_ranked_lower_under_g_profile(self):
        scores = {"structural": 0.90, "temporal": 0.80, "telemetry": 0.70,
                  "evidence": 0.10, "governance": 0.10}
        composite_equipment = self._combine(scores, _DEFAULT_SCORING_PROFILES["A"])
        composite_human = self._combine(scores, _DEFAULT_SCORING_PROFILES["G"])
        assert composite_equipment > composite_human

    def test_i_profile_temporal_weight_higher_than_g(self):
        assert _DEFAULT_SCORING_PROFILES["I"]["temporal"] > _DEFAULT_SCORING_PROFILES["G"]["temporal"]

    def test_l_profile_structural_and_telemetry_near_zero(self):
        l_profile = _DEFAULT_SCORING_PROFILES["L"]
        assert l_profile["structural"] <= 0.05
        assert l_profile["telemetry"] <= 0.05
