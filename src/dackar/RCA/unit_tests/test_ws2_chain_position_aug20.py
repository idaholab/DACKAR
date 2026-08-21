"""
test_ws2_chain_position_aug20.py — Workstream 2 (Part A) chain-position eligibility.

Covers the two Part-A behaviours (decision-log §6; metamodel "initiating candidates
ranked above contributing"; closes F-6 for the near-tie case, reinforces N-1/P-5):

  1. `_promote_initiator_over_consequence` / `_select_candidates`:
     a near-tie `initiating` candidate is promoted ahead of a top `consequence`;
     a clearly-stronger consequence is left in place; non-consequence tops untouched.
  2. `_apply_chain_position_review_flag`:
     a `consequence`-as-primary raises an analyst attention flag pointing at the
     upstream initiator, plus an uncertainty note; an `initiating` primary does not.

Depth labelling stays category-based (Part A scope) — not asserted here.

Run:  pytest test_ws2_chain_position_aug20.py -v
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

_MARGIN = RuleValidatedRCASynthesizerV31._CHAIN_POSITION_PRIMARY_TIE_MARGIN


def _synth() -> RuleValidatedRCASynthesizerV31:
    return RuleValidatedRCASynthesizerV31(llm_client=DummyLLMClient(), config=RCASynthesizerConfig())


def _cand(cid: str, chain_position: str, score: float) -> dict:
    return {
        "candidate_id": cid,
        "chain_position": chain_position,
        "composite_score": score,
        "cause_label": f"cause_{cid}",
        "primary_causal_category": "A",
    }


# ── 1. eligibility / promotion ─────────────────────────────────────────────

def test_near_tie_initiator_promoted_over_consequence():
    s = _synth()
    ranked = [
        _cand("CONS", "consequence", 0.80),
        _cand("INIT", "initiating", 0.80 - _MARGIN + 0.01),  # within margin
        _cand("OTH", "contributing", 0.50),
    ]
    out = s._promote_initiator_over_consequence(ranked)
    assert out[0]["candidate_id"] == "INIT"
    assert out[1]["candidate_id"] == "CONS"


def test_clearly_stronger_consequence_not_overridden():
    s = _synth()
    ranked = [
        _cand("CONS", "consequence", 0.90),
        _cand("INIT", "initiating", 0.90 - _MARGIN - 0.05),  # outside margin
    ]
    out = s._promote_initiator_over_consequence(ranked)
    assert out[0]["candidate_id"] == "CONS"  # not promoted


def test_initiating_top_untouched():
    s = _synth()
    ranked = [
        _cand("INIT", "initiating", 0.70),
        _cand("CONS", "consequence", 0.69),
    ]
    out = s._promote_initiator_over_consequence(ranked)
    assert out[0]["candidate_id"] == "INIT"


def test_no_initiator_present_leaves_consequence():
    s = _synth()
    ranked = [
        _cand("CONS", "consequence", 0.70),
        _cand("CONTRIB", "contributing", 0.69),
    ]
    out = s._promote_initiator_over_consequence(ranked)
    assert out[0]["candidate_id"] == "CONS"


def test_select_candidates_applies_promotion():
    s = _synth()
    cc = {
        "candidates": [
            _cand("CONS", "consequence", 0.80),
            _cand("INIT", "initiating", 0.79),
            _cand("C3", "contributing", 0.40),
        ]
    }
    selected = s._select_candidates(cc)
    assert selected[0]["candidate_id"] == "INIT"


# ── 2. consequence-as-primary flag ─────────────────────────────────────────

def test_consequence_primary_raises_flag_and_points_to_initiator():
    s = _synth()
    card = {
        "primary_hypothesis": {"candidate_id": "CONS", "uncertainties": []},
        "executive_summary": {"analyst_attention_flags": []},
    }
    cc = {"candidates": [_cand("CONS", "consequence", 0.90), _cand("INIT", "initiating", 0.60)]}
    s._apply_chain_position_review_flag(card, cc)
    flags = card["executive_summary"]["analyst_attention_flags"]
    assert any("consequence" in f.lower() for f in flags)
    assert any("INIT" in f for f in flags), "flag should point to the upstream initiator"
    assert any("downstream consequence" in u.lower() for u in card["primary_hypothesis"]["uncertainties"])


def test_consequence_primary_flag_when_no_initiator():
    s = _synth()
    card = {
        "primary_hypothesis": {"candidate_id": "CONS", "uncertainties": []},
        "executive_summary": {"analyst_attention_flags": []},
    }
    cc = {"candidates": [_cand("CONS", "consequence", 0.90), _cand("C2", "contributing", 0.60)]}
    s._apply_chain_position_review_flag(card, cc)
    flags = card["executive_summary"]["analyst_attention_flags"]
    assert any("outside the current candidate set" in f for f in flags)


def test_initiating_primary_raises_no_flag():
    s = _synth()
    card = {
        "primary_hypothesis": {"candidate_id": "INIT", "uncertainties": []},
        "executive_summary": {"analyst_attention_flags": []},
    }
    cc = {"candidates": [_cand("INIT", "initiating", 0.90), _cand("CONS", "consequence", 0.60)]}
    s._apply_chain_position_review_flag(card, cc)
    assert card["executive_summary"]["analyst_attention_flags"] == []
    assert card["primary_hypothesis"]["uncertainties"] == []
