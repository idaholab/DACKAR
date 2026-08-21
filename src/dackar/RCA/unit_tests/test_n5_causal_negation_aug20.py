"""
test_n5_causal_negation_aug20.py — N-5 negated/reversed causal extraction as non-evidence.

The extraction layer is empirically weak on the causally hard cases (direction
accuracy 56%, negated_causality F1 0.000), so a negated causal claim
("X did NOT cause Y") could still index as a positive X->Y attribution and
inflate the evidence/recurrence score of the wrong failure mode. N-5:

  1. Source propagation — `CausalSentence` already detects sentence negation
     (`isNegation`); it is now carried through `to_stage5_dict()` as a `negated`
     flag (default False). The adapter's existing `_route_negated_statements` /
     recurrence-skip plumbing already acts on that flag, so negated tuples stop
     entering positive support/recurrence.
  2. Evidence backstop — `_assess_hit_against_candidate` now detects a negated
     causal LINK ("did not cause", "not attributable to") relevant to the
     candidate, drops the causal-attribution boost, and counts it as contradiction
     (parity with the P-3 negated-state refutation).

Run:  pytest test_n5_causal_negation_aug20.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb",
             "langchain_community", "langchain_community.vectorstores",
             "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.evidence_retriever import (  # noqa: E402
    ChromaEvidenceRetriever,
    EvidenceRetrieverConfig,
)
from ner.causal_condition_adapter import (  # noqa: E402
    _tuple_to_causal_dict,
    _build_causal_statement,
    _route_negated_statements,
)


# ── evidence-scoring backstop ───────────────────────────────────────────────

def _retriever():
    return ChromaEvidenceRetriever(store=MagicMock(), config=EvidenceRetrieverConfig())


def _hit(snippet: str, vector_score: float = 0.7, doc_type: str = "ECA"):
    return {
        "snippet": snippet,
        "metadata": {
            "_vector_score": vector_score,
            "doc_type": doc_type,
            "epistemic_class": "analyzes_past_degradation",
            "authority_level": "mandatory",
        },
    }


_CANDIDATE_PLAN = {
    "query_type": "candidate",
    "cause_label": "bearing wear",
    "hypothesis_type": "mechanical degradation",
}


def test_negated_causal_link_is_not_supporting():
    r = _retriever()
    res = r._assess_hit_against_candidate(
        _hit("the bearing wear did not cause the pump vibration"),
        _CANDIDATE_PLAN,
    )
    assert res["causal_link_negation_hit"] is True
    assert res["support_role"] != "supporting"


def test_not_attributable_to_is_contradiction():
    r = _retriever()
    res = r._assess_hit_against_candidate(
        _hit("the failure was not attributable to bearing wear"),
        _CANDIDATE_PLAN,
    )
    assert res["causal_link_negation_hit"] is True
    assert res["contradiction_score"] >= res["support_score"]


def test_positive_attribution_still_supporting():
    r = _retriever()
    res = r._assess_hit_against_candidate(
        _hit("the pump failure was caused by bearing wear and confirmed as degraded"),
        _CANDIDATE_PLAN,
    )
    assert res["causal_link_negation_hit"] is False
    assert res["support_role"] == "supporting"


def test_irrelevant_causal_negation_not_attributed():
    r = _retriever()
    # negated causal link about a DIFFERENT component, zero relevance to bearing wear
    res = r._assess_hit_against_candidate(
        _hit("the gasket leak was not caused by seal misalignment", vector_score=0.0),
        _CANDIDATE_PLAN,
    )
    assert res["causal_link_negation_hit"] is False


# ── source propagation (deterministic, no spaCy needed) ─────────────────────

def test_tuple_to_causal_dict_maps_negation_index():
    # 8-element legacy tuple: [...conjecture(6), negated(7)]
    row = ("bearing wear", "degraded", "caused", "vibration", "high", "sent", False, True)
    out = _tuple_to_causal_dict(row)
    assert out["negated"] is True


def test_build_causal_statement_reads_negated_from_row():
    stmt = _build_causal_statement(
        doc_id="D1", chunk_index=0, i=0,
        row={"cause_text": "bearing wear", "effect_text": "vibration",
             "connector": "caused", "sentence": "bearing wear did not cause vibration",
             "negated": True},
        source="CausalSentence",
    )
    assert stmt["negated"] is True


def test_negated_statement_is_routed_out_of_active_evidence():
    result = {
        "extracted_causal_statements": [
            {"cause_text": "bearing wear", "effect_text": "vibration", "negated": False},
            {"cause_text": "fouling", "effect_text": "heat rise", "negated": True},
        ]
    }
    _route_negated_statements(result)
    active = result["extracted_causal_statements"]
    ruled_out = result.get("ruled_out_mechanisms") or []
    assert [s["cause_text"] for s in active] == ["bearing wear"]
    assert [s["cause_text"] for s in ruled_out] == ["fouling"]
