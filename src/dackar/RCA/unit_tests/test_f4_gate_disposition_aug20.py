"""
test_f4_gate_disposition_aug20.py — F-4 elimination-first audit.

Hard gates run *after* composite scoring, so a candidate can be gate-eliminated
yet still carry a high composite_score, with the elimination scattered across
`hard_gates` / `primary_eligibility` / `ruleout`. F-4 adds an additive, ranking-
neutral `gate_disposition` card block that (a) states hard gates are dispositive
and override the retained score, and (b) surfaces any high-scoring gate-eliminated
candidate so it cannot be silently outranked-then-ignored. No pipeline reorder.

Run:  pytest test_f4_gate_disposition_aug20.py -v
"""
from __future__ import annotations

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

from orchestrators.llm_clients import DummyLLMClient  # noqa: E402
from synthesis.rca_synthesizer_v31 import (  # noqa: E402
    RuleValidatedRCASynthesizerV31,
    RCASynthesizerConfig,
)


def _synth() -> RuleValidatedRCASynthesizerV31:
    return RuleValidatedRCASynthesizerV31(llm_client=DummyLLMClient(), config=RCASynthesizerConfig())


def _pass_gates():
    return {
        "physical_plausibility": {"passed": True},
        "timeline_consistency": {"passed": True},
        "barrier_logic": {"passed": True},
    }


def _card(primary_id="FM::PRIMARY"):
    return {"primary_hypothesis": {"candidate_id": primary_id}}


# ── eliminating-gate derivation ─────────────────────────────────────────────

def test_eliminating_gates_reads_failed_hard_gate():
    cand = {"candidate_id": "X", "hard_gates": {**_pass_gates(), "barrier_logic": {"passed": False}}}
    assert RuleValidatedRCASynthesizerV31._eliminating_gates_for(cand) == ["barrier_logic"]


def test_eliminating_gates_includes_contradiction_postures():
    cand = {
        "candidate_id": "X",
        "hard_gates": _pass_gates(),
        "primary_block_reasons": ["documentary_contradiction", "temporal_contradiction"],
    }
    egates = RuleValidatedRCASynthesizerV31._eliminating_gates_for(cand)
    assert "documentary_contradiction" in egates and "temporal_contradiction" in egates


def test_clean_candidate_has_no_eliminating_gates():
    cand = {"candidate_id": "X", "hard_gates": _pass_gates()}
    assert RuleValidatedRCASynthesizerV31._eliminating_gates_for(cand) == []


# ── card block assembly ─────────────────────────────────────────────────────

def test_high_scoring_gate_eliminated_candidate_is_surfaced():
    s = _synth()
    cc = {
        "candidates": [
            {"candidate_id": "FM::PRIMARY", "composite_score": 0.61, "hard_gates": _pass_gates()},
        ],
        "filtered_out_candidates": [
            {"candidate_id": "FM::BLOCKED", "composite_score": 0.83,
             "hard_gates": {**_pass_gates(), "barrier_logic": {"passed": False}}},
        ],
    }
    block = s._build_gate_disposition(card=_card(), causality_candidates=cc)
    assert block["hard_gates_are_dispositive"] is True
    assert block["primary_gate_status"] == "passed_all_gates"
    assert [e["candidate_id"] for e in block["eliminated_candidates"]] == ["FM::BLOCKED"]
    elim = block["eliminated_candidates"][0]
    assert elim["eliminating_gates"] == ["barrier_logic"]
    assert elim["composite_score"] == 0.83  # high score preserved + visible
    assert "override" in elim["note"].lower()


def test_eliminated_candidates_sorted_by_descending_score():
    s = _synth()
    cc = {
        "candidates": [{"candidate_id": "P", "composite_score": 0.5, "hard_gates": _pass_gates()}],
        "filtered_out_candidates": [
            {"candidate_id": "A", "composite_score": 0.40,
             "hard_gates": {**_pass_gates(), "timeline_consistency": {"passed": False}}},
            {"candidate_id": "B", "composite_score": 0.77,
             "hard_gates": {**_pass_gates(), "physical_plausibility": {"passed": False}}},
        ],
    }
    block = s._build_gate_disposition(card=_card("P"), causality_candidates=cc)
    assert [e["candidate_id"] for e in block["eliminated_candidates"]] == ["B", "A"]


def test_primary_status_none_when_no_primary():
    s = _synth()
    block = s._build_gate_disposition(
        card=_card("NONE"), causality_candidates={"candidates": []},
    )
    assert block["primary_gate_status"] == "no_primary_established"
    assert block["eliminated_candidates"] == []


def test_primary_status_eliminated_when_primary_failed_a_gate():
    s = _synth()
    cc = {
        "candidates": [
            {"candidate_id": "FM::PRIMARY", "composite_score": 0.7,
             "hard_gates": {**_pass_gates(), "barrier_logic": {"passed": False}}},
        ],
    }
    block = s._build_gate_disposition(card=_card(), causality_candidates=cc)
    assert block["primary_gate_status"] == "eliminated"


def test_gate_order_is_reported():
    s = _synth()
    block = s._build_gate_disposition(card=_card("NONE"), causality_candidates={"candidates": []})
    assert block["gate_order"] == ["physical_plausibility", "timeline_consistency", "barrier_logic"]


def test_schema_shape_is_complete_and_bounded():
    s = _synth()
    cc = {"candidates": [], "filtered_out_candidates": [
        {"candidate_id": "A", "composite_score": 0.9,
         "hard_gates": {**_pass_gates(), "barrier_logic": {"passed": False}}}]}
    block = s._build_gate_disposition(card=_card("NONE"), causality_candidates=cc)
    for key in ("hard_gates_are_dispositive", "gate_order", "primary_gate_status",
                "eliminated_candidates", "note"):
        assert key in block
    for e in block["eliminated_candidates"]:
        assert set(e).issubset({"candidate_id", "eliminating_gates", "composite_score", "note"})
        assert {"candidate_id", "eliminating_gates", "composite_score"}.issubset(e)
