"""
test_cross_pattern_summary.py — Phase 4 unit tests for cross_pattern/summary.py

Coverage:
  - format_rca_card_cross_pattern_summary: exact wording for all outcomes
  - get_cross_pattern_attention_flags: conflicting top candidate, multiple_consistent,
    no_data, stale provenance, non-top conflicting not raised
  - build_manifest_cross_pattern_summary: precedence distribution, temporal_link_skipped
    count, per-candidate summaries
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in (
    "neo4j", "py2neo", "chromadb",
    "langchain_chroma", "langchain_community",
    "langchain_community.vectorstores", "langchain_community.embeddings",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from cross_pattern.models import CandidateCrossPatternEvidence, CrossPatternLink
from cross_pattern.summary import (
    format_rca_card_cross_pattern_summary,
    build_manifest_cross_pattern_summary,
    get_cross_pattern_attention_flags,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evidence(
    candidate_id: str = "CAND-001",
    fm_id: str = "FM-001",
    linkage_outcome: str = "linked",
    support_posture: str = "reinforcing",
    reinforcement_strength: Optional[str] = "single",
    best_link_score: float = 0.75,
    evidence_paths: Optional[List[CrossPatternLink]] = None,
) -> CandidateCrossPatternEvidence:
    return CandidateCrossPatternEvidence(
        candidate_id=candidate_id,
        component_id="COMP-1",
        fm_id=fm_id,
        linked_episode_ids=["ep1"] if linkage_outcome == "linked" else [],
        linked_doc_ids=["DOC-001"] if linkage_outcome == "linked" else [],
        best_link_score=best_link_score,
        support_posture=support_posture,
        reinforcement_strength=reinforcement_strength,
        linkage_outcome=linkage_outcome,
        evidence_paths=evidence_paths or [],
    )


def _make_link(
    linkage_precedence_level: int = 3,
    temporal_link_skipped: bool = False,
    ep_index_status: str = "indexed",
) -> CrossPatternLink:
    return CrossPatternLink(
        link_id="ep1::DOC-001::3",
        episode_id="ep1",
        doc_id="DOC-001",
        asset_match=True,
        time_overlap_hours=None,
        temporal_link_skipped=temporal_link_skipped,
        linkage_precedence_level=linkage_precedence_level,
        component_overlap=[],
        fm_alignment_score=1.0,
        signal_similarity_score=0.8,
        document_similarity_score=None,
        link_confidence=0.7,
        provenance={"ep_index_status": ep_index_status},
    )


def _make_candidate_dict(candidate_id: str) -> Dict[str, Any]:
    return {"candidate_id": candidate_id, "fm_id": "FM-001"}


# ═══════════════════════════════════════════════════════════════════════════════
# RCA card wording (§4.7 exact strings)
# ═══════════════════════════════════════════════════════════════════════════════

def test_wording_no_data():
    """outcome='no_data' → narrative contains 'No historical signal or document data available'."""
    ev = _make_evidence(linkage_outcome="no_data", support_posture="unresolved",
                        reinforcement_strength=None)
    dist = {"no_data": 1, "linked": 0, "no_match": 0, "below_threshold": 0}
    narrative = format_rca_card_cross_pattern_summary([ev], dist)
    assert "No historical signal or document data available" in narrative


def test_wording_no_match():
    """outcome='no_match' → narrative contains 'No historically similar signal episodes were found'."""
    ev = _make_evidence(linkage_outcome="no_match", support_posture="unresolved",
                        reinforcement_strength=None)
    dist = {"no_data": 0, "linked": 0, "no_match": 1, "below_threshold": 0}
    narrative = format_rca_card_cross_pattern_summary([ev], dist)
    assert "No historically similar signal episodes were found" in narrative


def test_wording_below_threshold():
    """outcome='below_threshold' → narrative contains 'could not be reliably linked'."""
    ev = _make_evidence(linkage_outcome="below_threshold", support_posture="unresolved",
                        reinforcement_strength=None)
    dist = {"no_data": 0, "linked": 0, "no_match": 0, "below_threshold": 1}
    narrative = format_rca_card_cross_pattern_summary([ev], dist)
    assert "could not be reliably linked" in narrative


def test_wording_linked_reinforcing_single():
    """linked+reinforcing+single → narrative contains 'reinforces'."""
    ev = _make_evidence(
        linkage_outcome="linked",
        support_posture="reinforcing",
        reinforcement_strength="single",
    )
    dist = {"no_data": 0, "linked": 1, "no_match": 0, "below_threshold": 0}
    narrative = format_rca_card_cross_pattern_summary([ev], dist)
    assert "reinforces" in narrative


def test_wording_linked_conflicting():
    """linked+conflicting → narrative contains 'conflicts' and 'analyst review'."""
    ev = _make_evidence(
        linkage_outcome="linked",
        support_posture="conflicting",
        reinforcement_strength=None,
    )
    dist = {"no_data": 0, "linked": 1, "no_match": 0, "below_threshold": 0}
    narrative = format_rca_card_cross_pattern_summary([ev], dist)
    assert "conflicts" in narrative
    assert "analyst review" in narrative


# ═══════════════════════════════════════════════════════════════════════════════
# Attention flags
# ═══════════════════════════════════════════════════════════════════════════════

def test_attention_flag_conflicting_top_candidate():
    """top candidate has support_posture='conflicting' → flag with 'conflicts' and 'analyst review required'."""
    ev = _make_evidence(
        candidate_id="CAND-001",
        linkage_outcome="linked",
        support_posture="conflicting",
        reinforcement_strength=None,
    )
    candidates = [_make_candidate_dict("CAND-001")]
    flags = get_cross_pattern_attention_flags([ev], candidates, top_n_candidates=3)
    assert any("conflicts" in f and "analyst review required" in f for f in flags)


def test_attention_flag_not_raised_for_non_top_candidate():
    """conflicting candidate is outside top-N → no conflict flag."""
    ev_top = _make_evidence(
        candidate_id="CAND-001",
        linkage_outcome="linked",
        support_posture="reinforcing",
        reinforcement_strength="single",
    )
    ev_outside = _make_evidence(
        candidate_id="CAND-004",
        linkage_outcome="linked",
        support_posture="conflicting",
        reinforcement_strength=None,
    )
    # Only CAND-001 is in top-3
    candidates = [_make_candidate_dict("CAND-001")]
    flags = get_cross_pattern_attention_flags([ev_top, ev_outside], candidates, top_n_candidates=3)
    # Should not have a conflict flag for CAND-004 since it's not in top_n
    assert not any("CAND-004" in f and "conflicts" in f for f in flags)


def test_attention_flag_multiple_consistent():
    """top candidate has reinforcement_strength='multiple_consistent' → flag with 'Multiple consistent'."""
    ev = _make_evidence(
        candidate_id="CAND-001",
        linkage_outcome="linked",
        support_posture="reinforcing",
        reinforcement_strength="multiple_consistent",
    )
    candidates = [_make_candidate_dict("CAND-001")]
    flags = get_cross_pattern_attention_flags([ev], candidates, top_n_candidates=3)
    assert any("Multiple consistent" in f for f in flags)


def test_attention_flag_no_data():
    """any candidate has linkage_outcome='no_data' → flag with 'No historical signal or document data'."""
    ev = _make_evidence(
        candidate_id="CAND-001",
        linkage_outcome="no_data",
        support_posture="unresolved",
        reinforcement_strength=None,
    )
    candidates = [_make_candidate_dict("CAND-001")]
    flags = get_cross_pattern_attention_flags([ev], candidates, top_n_candidates=3)
    assert any("No historical signal or document data" in f for f in flags)


def test_attention_flag_stale_from_provenance():
    """link provenance has ep_index_status='stale' → flag with 'stale'."""
    stale_link = _make_link(ep_index_status="stale")
    ev = _make_evidence(
        candidate_id="CAND-001",
        linkage_outcome="linked",
        support_posture="reinforcing",
        reinforcement_strength="single",
        evidence_paths=[stale_link],
    )
    candidates = [_make_candidate_dict("CAND-001")]
    flags = get_cross_pattern_attention_flags([ev], candidates, top_n_candidates=3)
    assert any("stale" in f.lower() for f in flags)


# ═══════════════════════════════════════════════════════════════════════════════
# Manifest summary
# ═══════════════════════════════════════════════════════════════════════════════

def test_manifest_has_precedence_distribution():
    """links at levels 1, 2, 3 → linkage_precedence_distribution has correct counts."""
    link1 = _make_link(linkage_precedence_level=1)
    link2 = _make_link(linkage_precedence_level=2)
    link3 = _make_link(linkage_precedence_level=3)
    ev = _make_evidence(
        candidate_id="CAND-001",
        linkage_outcome="linked",
        evidence_paths=[link1, link2, link3],
    )
    manifest = build_manifest_cross_pattern_summary(
        candidate_evidences=[ev],
        total_episodes=3,
        total_docs=3,
        total_links=3,
        links_above_threshold=3,
    )
    dist = manifest["linkage_precedence_distribution"]
    assert dist["1"] == 1
    assert dist["2"] == 1
    assert dist["3"] == 1


def test_manifest_temporal_link_skipped_count():
    """two links with temporal_link_skipped=True → count=2."""
    link_a = _make_link(temporal_link_skipped=True)
    link_b = _make_link(temporal_link_skipped=True)
    link_c = _make_link(temporal_link_skipped=False)
    ev = _make_evidence(
        candidate_id="CAND-001",
        linkage_outcome="linked",
        evidence_paths=[link_a, link_b, link_c],
    )
    manifest = build_manifest_cross_pattern_summary(
        candidate_evidences=[ev],
        total_episodes=1,
        total_docs=1,
        total_links=3,
        links_above_threshold=3,
    )
    assert manifest["temporal_link_skipped_count"] == 2


def test_manifest_per_candidate_summaries():
    """candidate_summaries list has correct candidate_id and linkage_outcome fields."""
    ev1 = _make_evidence(candidate_id="CAND-001", linkage_outcome="linked")
    ev2 = _make_evidence(candidate_id="CAND-002", linkage_outcome="no_data",
                         support_posture="unresolved", reinforcement_strength=None)
    manifest = build_manifest_cross_pattern_summary(
        candidate_evidences=[ev1, ev2],
        total_episodes=2,
        total_docs=2,
        total_links=1,
        links_above_threshold=1,
    )
    summaries = {s["candidate_id"]: s for s in manifest["candidate_summaries"]}
    assert "CAND-001" in summaries
    assert summaries["CAND-001"]["linkage_outcome"] == "linked"
    assert "CAND-002" in summaries
    assert summaries["CAND-002"]["linkage_outcome"] == "no_data"
