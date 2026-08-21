"""
test_p4_supersession_relevance_aug20.py — P-4 supersession relevance gate.

Supersession previously zeroed a lower-authority hit's support based on authority
(+ recency) ALONE, ignoring relevance. A weakly-relevant high-authority document
could therefore erase the most on-point lower-authority evidence. The P-4 gate blocks
that: a higher-authority hit only supersedes a lower-authority hit when it is at least
nearly as on-point (relevance within `_RELEVANCE_SUPERSEDE_MARGIN`).

Run:  pytest test_p4_supersession_relevance_aug20.py -v
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

from orchestrators.supersession import (  # noqa: E402
    resolve_supersession,
    _RELEVANCE_SUPERSEDE_MARGIN,
)


def _hit(candidate_id, doc_type, support_score=0.6, finding_status="",
         event_date="", snippet_id="", extra_meta=None):
    meta = {
        "linked_candidate_id": candidate_id,
        "doc_type": doc_type,
        "support_role": "supporting" if support_score > 0 else "contextual",
    }
    if finding_status:
        meta["finding_status"] = finding_status
    if event_date:
        meta["event_date"] = event_date
    if extra_meta:
        meta.update(extra_meta)
    return {
        "snippet_id": snippet_id or f"{doc_type}::{candidate_id}",
        "support_score": support_score,
        "contradiction_score": 0.0,
        "context_score": 0.0,
        "metadata": meta,
    }


def _bundle(*hits):
    return {"results": list(hits), "candidate_evidence_summary": []}


# ── the P-4 scenario: off-point authority must not erase on-point support ──

def test_offpoint_high_authority_does_not_supersede_onpoint_lower():
    """Weakly-relevant RCA (0.40) must NOT erase strongly on-point CR (0.85)."""
    b = _bundle(
        _hit("C1", "RCA", finding_status="formal_conclusion", support_score=0.40, snippet_id="rca"),
        _hit("C1", "CR", finding_status="preliminary_assessment", support_score=0.85, snippet_id="cr"),
    )
    resolve_supersession(b)
    cr = next(h for h in b["results"] if h["snippet_id"] == "cr")
    assert cr.get("superseded") is not True
    assert cr["support_score"] == 0.85
    assert cr.get("supersession_relevance_retained") is True
    assert b["supersession_count"] == 0
    assert b["supersession_relevance_retained_count"] == 1


def test_onpoint_high_authority_still_supersedes():
    """Comparable relevance → authority hierarchy still governs (regression)."""
    b = _bundle(
        _hit("C1", "RCA", finding_status="formal_conclusion", support_score=0.80, snippet_id="rca"),
        _hit("C1", "CR", finding_status="preliminary_assessment", support_score=0.70, snippet_id="cr"),
    )
    resolve_supersession(b)
    cr = next(h for h in b["results"] if h["snippet_id"] == "cr")
    assert cr.get("superseded") is True
    assert cr["support_score"] == 0.0
    assert b["supersession_relevance_retained_count"] == 0


def test_margin_boundary_supersedes():
    """Exactly at the margin boundary → still supersedes (>= comparison)."""
    lower = 0.80
    higher = lower - _RELEVANCE_SUPERSEDE_MARGIN  # exactly on the boundary
    b = _bundle(
        _hit("C1", "RCA", finding_status="formal_conclusion", support_score=higher, snippet_id="rca"),
        _hit("C1", "CR", finding_status="preliminary_assessment", support_score=lower, snippet_id="cr"),
    )
    resolve_supersession(b)
    cr = next(h for h in b["results"] if h["snippet_id"] == "cr")
    assert cr.get("superseded") is True


def test_explicit_relevance_score_overrides_support_score():
    """metadata.relevance_score is used for the gate when present."""
    b = _bundle(
        # RCA has high support_score but explicitly low relevance → off-point
        _hit("C1", "RCA", finding_status="formal_conclusion", support_score=0.90,
             snippet_id="rca", extra_meta={"relevance_score": 0.30}),
        _hit("C1", "CR", finding_status="preliminary_assessment", support_score=0.60,
             snippet_id="cr", extra_meta={"relevance_score": 0.85}),
    )
    resolve_supersession(b)
    cr = next(h for h in b["results"] if h["snippet_id"] == "cr")
    assert cr.get("supersession_relevance_retained") is True
    assert cr["support_score"] == 0.60


def test_recency_scoped_to_same_rank_does_not_reerase_retained():
    """A newer off-point higher-authority hit must not erase a relevance-retained
    older lower-authority on-point hit via the recency tiebreak."""
    b = _bundle(
        _hit("C1", "RCA", finding_status="formal_conclusion", support_score=0.35,
             event_date="2024-01-01", snippet_id="rca_new"),
        _hit("C1", "CR", finding_status="preliminary_assessment", support_score=0.85,
             event_date="2020-01-01", snippet_id="cr_old"),
    )
    resolve_supersession(b)
    cr = next(h for h in b["results"] if h["snippet_id"] == "cr_old")
    assert cr.get("superseded") is not True
    assert cr["support_score"] == 0.85
