"""
test_phase4c_causal_category_dispatch.py — unit tests for Phase 4c Steps 4a-4c:

  * assign_causal_category() in kg_population_helpers
  * Engine _infer_primary_category_for_failure_mode reads curated causal_category
    from FM node before falling through to keyword inference
  * generate() output reflects curated causal_category for TC-6 FM nodes

Tests:
  TestAssignCausalCategory         (10 tests) — helpers function
  TestEngineReadsCuratedCategory   ( 8 tests) — engine dispatch with/without field
  TestTC6CuratedCategoryDispatch   ( 5 tests) — integration with TC-6 fixtures
"""
import json
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
from demos.kg_population_helpers import assign_causal_category
from orchestrators.causality_engine_v32 import (
    RuleBasedCausalityEngineV32,
    CausalityEngineConfigV32,
)

_TC6_KG_PATH = _RCA_ROOT / "tests" / "test_case_6" / "fixtures" / "kg_context.json"
_TC3_KG_PATH = _RCA_ROOT / "tests" / "test_case_3" / "fixtures" / "kg_context.json"

ENGINE = RuleBasedCausalityEngineV32


# ── TestAssignCausalCategory ──────────────────────────────────────────────────

class TestAssignCausalCategory:

    def test_curated_valid_category_returned_as_curated(self):
        fm = {"causal_category": "G", "name": "operator error"}
        cat, src = assign_causal_category(fm)
        assert cat == "G"
        assert src == "curated"

    def test_curated_overrides_keyword_inference(self):
        # name has B-category keywords but causal_category says I
        fm = {"causal_category": "I", "name": "power support cooling", "superclass": ""}
        cat, src = assign_causal_category(fm)
        assert cat == "I"
        assert src == "curated"

    def test_curated_lowercase_normalized(self):
        fm = {"causal_category": "l", "name": "systemic latent"}
        cat, src = assign_causal_category(fm)
        assert cat == "L"
        assert src == "curated"

    def test_curated_with_whitespace_normalized(self):
        fm = {"causal_category": "  G  ", "name": "human operator"}
        cat, src = assign_causal_category(fm)
        assert cat == "G"
        assert src == "curated"

    def test_invalid_causal_category_falls_through_to_inference(self):
        fm = {"causal_category": "Z", "name": "operator error human procedure not followed"}
        cat, src = assign_causal_category(fm)
        assert cat == "G"
        assert src == "inferred"

    def test_missing_causal_category_infers_from_keywords(self):
        fm = {"name": "systemic latent safety culture recurrence", "superclass": ""}
        cat, src = assign_causal_category(fm)
        assert cat == "L"
        assert src == "inferred"

    def test_no_keywords_defaults_to_a(self):
        fm = {"name": "bearing failure", "superclass": "mechanical_wear", "failure_mechanism": "fatigue"}
        cat, src = assign_causal_category(fm)
        assert cat == "A"
        assert src == "inferred"

    def test_b_category_keyword_match(self):
        fm = {"name": "cooling water supply loss", "superclass": "support_system_failure"}
        cat, src = assign_causal_category(fm)
        assert cat == "B"
        assert src == "inferred"

    def test_empty_fm_node_defaults_to_a(self):
        cat, src = assign_causal_category({})
        assert cat == "A"
        assert src == "inferred"

    def test_all_valid_curated_categories_accepted(self):
        for letter in "ABCDEFGHIJKL":
            cat, src = assign_causal_category({"causal_category": letter})
            assert cat == letter
            assert src == "curated"


# ── TestEngineReadsCuratedCategory ────────────────────────────────────────────

class TestEngineReadsCuratedCategory:

    def _infer(self, fm: dict, event: dict = None) -> tuple:
        event = event or {"event_id": "EVT-001", "event_type": "equipment_failure"}
        return ENGINE._infer_primary_category_for_failure_mode(fm=fm, event=event)

    def test_curated_g_bypasses_keyword_inference(self):
        fm = {
            "fm_id": "FM-TEST",
            "causal_category": "G",
            "name": "bearing failure",          # would infer A without curated field
            "superclass": "mechanical_wear",
            "failure_mechanism": "fatigue",
        }
        cat, alts = self._infer(fm)
        assert cat == "G"
        assert alts == []

    def test_curated_i_bypasses_keyword_inference(self):
        fm = {
            "fm_id": "FM-TEST",
            "causal_category": "I",
            "name": "procedure acceptance criteria gap",
            "superclass": "procedure_acceptance_criteria_gap",
            "failure_mechanism": "procedural",
        }
        cat, alts = self._infer(fm)
        assert cat == "I"
        assert alts == []

    def test_no_curated_category_uses_keyword_inference(self):
        fm = {
            "fm_id": "FM-TEST",
            "name": "operator error human procedure not followed",
            "superclass": "human_performance",
            "failure_mechanism": "human",
        }
        cat, alts = self._infer(fm)
        assert cat == "G"

    def test_curated_a_is_accepted_not_treated_as_missing(self):
        # "A" is a valid curated category — must short-circuit like any other
        fm = {
            "fm_id": "FM-TEST",
            "causal_category": "A",
            "name": "systemic latent safety culture",  # would infer L without curated
            "superclass": "organizational",
        }
        cat, alts = self._infer(fm)
        assert cat == "A"
        assert alts == []

    def test_curated_b_returns_empty_alternatives(self):
        fm = {"causal_category": "B", "name": "power supply failure"}
        cat, alts = self._infer(fm)
        assert cat == "B"
        assert alts == []

    def test_invalid_curated_value_falls_through(self):
        # "X" is not a valid category — inference runs
        fm = {
            "causal_category": "X",
            "name": "systemic latent recurrence",
            "superclass": "",
            "failure_mechanism": "",
        }
        cat, alts = self._infer(fm)
        assert cat == "L"

    def test_empty_causal_category_falls_through_to_inference(self):
        fm = {
            "causal_category": "",
            "name": "operator human maintenance error",
            "superclass": "",
            "failure_mechanism": "human",
        }
        cat, alts = self._infer(fm)
        assert cat == "G"

    def test_none_causal_category_falls_through_to_inference(self):
        fm = {
            "causal_category": None,
            "name": "seismic disturbance fire environment",
            "superclass": "",
            "failure_mechanism": "",
        }
        cat, alts = self._infer(fm)
        assert cat == "F"


# ── TestTC6CuratedCategoryDispatch ────────────────────────────────────────────

def _load_kg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _minimal_event_tc6() -> dict:
    return {
        "event_id": "EVT-TC6",
        "id": "EVT-TC6",
        "description": "Main feedwater pump B bearing failure following lube oil maintenance",
        "event_type": "equipment_failure",
        "primary_system": "FM-MFPB-BEARING-WEAR",
    }


def _run_tc6_generate() -> dict:
    if not _TC6_KG_PATH.exists():
        pytest.skip(f"TC-6 fixture not found: {_TC6_KG_PATH}")
    kg = _load_kg(_TC6_KG_PATH)
    engine = RuleBasedCausalityEngineV32(CausalityEngineConfigV32())
    return engine.generate(
        event=_minimal_event_tc6(),
        telemetry_summary={},
        kg_context=kg,
        tskr_patterns=None,
        operational_context=None,
        pm_compliance=None,
        run_context={"run_id": "RUN-TC6"},
    )


@pytest.fixture(scope="module")
def tc6_result():
    return _run_tc6_generate()


class TestTC6CuratedCategoryDispatch:

    def test_tc6_fixtures_have_causal_category(self):
        if not _TC6_KG_PATH.exists():
            pytest.skip(f"TC-6 fixture not found: {_TC6_KG_PATH}")
        kg = _load_kg(_TC6_KG_PATH)
        for fm in kg["failure_modes"]:
            assert "causal_category" in fm, f"{fm['fm_id']} missing causal_category"
            assert "causal_category_source" in fm, f"{fm['fm_id']} missing causal_category_source"

    def test_tc6_curated_lube_oil_omission_is_g(self):
        if not _TC6_KG_PATH.exists():
            pytest.skip(f"TC-6 fixture not found: {_TC6_KG_PATH}")
        kg = _load_kg(_TC6_KG_PATH)
        fm = next(f for f in kg["failure_modes"] if f["fm_id"] == "FM-MFPB-LUBE-OIL-OMISSION")
        assert fm["causal_category"] == "G"
        assert fm["causal_category_source"] == "curated"

    def test_tc6_curated_proc_criterion_gap_is_i(self):
        if not _TC6_KG_PATH.exists():
            pytest.skip(f"TC-6 fixture not found: {_TC6_KG_PATH}")
        kg = _load_kg(_TC6_KG_PATH)
        fm = next(f for f in kg["failure_modes"] if f["fm_id"] == "FM-MFPB-PROC-CRITERION-GAP")
        assert fm["causal_category"] == "I"
        assert fm["causal_category_source"] == "curated"

    def test_tc6_generate_produces_g_category_candidate(self, tc6_result):
        all_cands = (tc6_result.get("candidates") or []) + (tc6_result.get("filtered_out_candidates") or [])
        g_cands = [c for c in all_cands if c.get("primary_causal_category") == "G"]
        assert len(g_cands) > 0, "Expected at least one G-category candidate from TC-6"

    def test_tc6_g_candidate_uses_human_performance_profile(self, tc6_result):
        all_cands = (tc6_result.get("candidates") or []) + (tc6_result.get("filtered_out_candidates") or [])
        g_cands = [c for c in all_cands if c.get("primary_causal_category") == "G"]
        if g_cands:
            scores = g_cands[0].get("scores") or {}
            assert scores.get("score_profile_applied") == "human_performance"
