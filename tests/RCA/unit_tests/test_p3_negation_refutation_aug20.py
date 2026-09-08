"""
test_p3_negation_refutation_aug20.py — P-3 semantic-ish negation/refutation.

Contradiction detection was phrase-list only, so a refutation expressed as a negated
degradation state outside the fixed cue list ("bearing showed no wear", "did not
exhibit degradation", "ruled out fouling") was missed and the snippet mis-classed as
merely contextual. P-3 adds a deterministic, tightly-scoped negation detector that
flags a negation trigger followed by a degradation/failure-state term, gated on
semantic relevance so it is attributed to the right candidate.

Covers:
  - `_negation_refutation_hit` unit behaviour (positives, tight-scope negatives,
    absence-as-cause negatives, multiword triggers).
  - `_assess_hit_against_candidate` integration: a relevant refutation snippet is
    classified `contradicting`; an irrelevant one is not; a plain supporting snippet
    is unaffected; explicit causal attribution is not flipped.

Run:  pytest test_p3_negation_refutation_aug20.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
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
    _negation_refutation_hit,
    _NEGATABLE_STATE_TERMS,
    _norm_text,
)


# ── helper-level unit tests ────────────────────────────────────────────────

def _neg(text: str) -> bool:
    return _negation_refutation_hit(_norm_text(text), _NEGATABLE_STATE_TERMS)


def test_negation_positive_showed_no_damage():
    assert _neg("the pump showed no damage during inspection") is True


def test_negation_positive_did_not_exhibit_wear():
    assert _neg("the bearing did not exhibit wear") is True


def test_negation_positive_not_degraded():
    assert _neg("component was not degraded") is True


def test_negation_positive_ruled_out_fouling():
    assert _neg("heat exchanger fouling was ruled out fouling by inspection") is True


def test_negation_positive_no_signs_of_drift():
    assert _neg("there were no signs of drift in the transmitter") is True


def test_absence_as_cause_is_not_refutation():
    # 'no lubrication' is a cause, and 'wear' is 4+ tokens away → outside tight window
    assert _neg("no lubrication eventually led to bearing wear") is False


def test_plain_degradation_statement_is_not_refutation():
    assert _neg("the bearing exhibited significant wear and damage") is False


def test_negation_far_from_state_term_is_not_refutation():
    # 'not' present but no state term within the 3-token window
    assert _neg("this was not the primary consideration for the review board") is False


# ── integration: _assess_hit_against_candidate ─────────────────────────────

def _retriever():
    return ChromaEvidenceRetriever(store=MagicMock(), config=EvidenceRetrieverConfig())


def _hit(snippet: str, vector_score: float = 0.6, doc_type: str = "ECA"):
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


def test_integration_relevant_refutation_is_contradicting():
    r = _retriever()
    res = r._assess_hit_against_candidate(
        _hit("the bearing showed no wear and no damage on inspection"),
        _CANDIDATE_PLAN,
    )
    assert res["negation_refutation_hit"] is True
    assert res["support_role"] == "contradicting"


def test_integration_irrelevant_refutation_not_flagged():
    r = _retriever()
    # no lexical/semantic overlap with "bearing wear" → not attributed to this candidate
    res = r._assess_hit_against_candidate(
        _hit("the gasket showed no corrosion", vector_score=0.0),
        _CANDIDATE_PLAN,
    )
    assert res["negation_refutation_hit"] is False


def test_integration_supporting_snippet_unaffected():
    r = _retriever()
    res = r._assess_hit_against_candidate(
        _hit("bearing wear was caused by loss of lubrication and confirmed as degraded"),
        _CANDIDATE_PLAN,
    )
    assert res["negation_refutation_hit"] is False
    assert res["support_role"] == "supporting"


def test_integration_causal_attribution_not_flipped_by_unrelated_negation():
    r = _retriever()
    # explicit attribution to this candidate + a negation elsewhere → stays supporting
    res = r._assess_hit_against_candidate(
        _hit("failure was caused by bearing wear; seal showed no damage", vector_score=0.7),
        _CANDIDATE_PLAN,
    )
    assert res["support_role"] == "supporting"
