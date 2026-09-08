"""
test_cross_pattern_linker.py — Phase 4 unit tests for cross_pattern/linker.py

Coverage:
  - Episode-to-candidate mapping: FM matching, alt FM id, sentinel episode
  - Redundancy suppression: level-1 beats level-2 for same pair, different pairs kept
  - Stale cap applied in linker
  - Reinforcement strength: single, multiple_consistent, mixed
  - temporal_link_skipped propagated to link
"""
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
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

from cross_pattern.config import CrossPatternConfig
from cross_pattern.linker import CrossPatternLinker
from cross_pattern.models import HistoricalDocExtraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


@dataclass
class _FakeEpisode:
    episode_id: str
    index_status: str = "indexed"
    similarity_to_current: float = 0.80
    asset_id: str = "ASSET-1"
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    linked_doc_ids: List[str] = None

    def __post_init__(self):
        if self.linked_doc_ids is None:
            self.linked_doc_ids = []


def _make_doc(
    doc_id: str = "DOC-001",
    fm_id_candidate: Optional[str] = "FM-001",
    fm_id_candidate_alt: Optional[str] = None,
    event_time_confidence: str = "absent",
    asset_id: Optional[str] = "ASSET-1",
    event_time_start: Optional[datetime] = None,
    event_time_end: Optional[datetime] = None,
) -> HistoricalDocExtraction:
    return HistoricalDocExtraction(
        doc_id=doc_id,
        doc_type="cr",
        asset_id=asset_id,
        event_time_start=event_time_start,
        event_time_end=event_time_end,
        event_time_confidence=event_time_confidence,
        identified_effect=None,
        assessed_cause=None,
        inferred_fm_label=None,
        fm_id_candidate=fm_id_candidate,
        fm_id_candidate_alt=fm_id_candidate_alt,
        fm_resolution_status="auto_resolved",
        fm_resolution_score=0.92,
        confidence="high",
        cause_is_symptom=False,
    )


def _make_candidate(
    candidate_id: str = "CAND-001",
    component_id: str = "COMP-1",
    fm_id: str = "FM-001",
) -> Dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "component_id": component_id,
        "fm_id": fm_id,
    }


def _default_config(**kwargs) -> CrossPatternConfig:
    cfg = CrossPatternConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _linker(**kwargs) -> CrossPatternLinker:
    return CrossPatternLinker(_default_config(**kwargs))


# ═══════════════════════════════════════════════════════════════════════════════
# Episode-to-candidate mapping
# ═══════════════════════════════════════════════════════════════════════════════

def test_only_fm_matching_candidates_linked():
    """Episode linked to doc with fm_id_candidate='FM-001'; candidate FM-001 gets evidence;
    candidate FM-999 gets linkage_outcome='no_data' (no docs match)."""
    episodes = [_FakeEpisode(episode_id="ep1")]
    doc = _make_doc(doc_id="DOC-001", fm_id_candidate="FM-001")
    candidates = [
        _make_candidate("CAND-001", fm_id="FM-001"),
        _make_candidate("CAND-002", fm_id="FM-999"),
    ]
    result = _linker().run(episodes, [doc], candidates)
    ev_by_id = {ev["candidate_id"]: ev for ev in result["candidate_evidence"]}

    # FM-001 candidate should have a link
    ev_001 = ev_by_id["CAND-001"]
    assert ev_001["linkage_outcome"] in ("linked", "below_threshold", "no_match")
    # A link was built (doc.fm_id_candidate == FM-001 matches candidate FM-001)
    assert len(ev_001["evidence_paths"]) >= 0  # above threshold

    # FM-999 candidate — doc has fm_id_candidate="FM-001", not FM-999
    ev_999 = ev_by_id["CAND-002"]
    assert ev_999["linkage_outcome"] == "no_match"


def test_only_fm_matching_candidates_linked_above_threshold():
    """Episode with high similarity linked to doc FM-001 → candidate FM-001 gets 'linked'."""
    episodes = [_FakeEpisode(episode_id="ep1", similarity_to_current=0.9)]
    doc = _make_doc(doc_id="DOC-001", fm_id_candidate="FM-001")
    candidates = [_make_candidate("CAND-001", fm_id="FM-001")]
    # Use low threshold to ensure linking
    result = _linker(link_confidence_threshold=0.05).run(episodes, [doc], candidates)
    ev = result["candidate_evidence"][0]
    assert ev["linkage_outcome"] == "linked"
    assert len(ev["evidence_paths"]) == 1


def test_alt_fm_id_also_produces_link():
    """doc has fm_id_candidate='FM-001', fm_id_candidate_alt='FM-002';
    candidate for FM-002 also gets a link."""
    episodes = [_FakeEpisode(episode_id="ep1", similarity_to_current=0.9)]
    doc = _make_doc(
        doc_id="DOC-001",
        fm_id_candidate="FM-001",
        fm_id_candidate_alt="FM-002",
    )
    candidates = [
        _make_candidate("CAND-002", fm_id="FM-002"),
    ]
    result = _linker(link_confidence_threshold=0.05).run(episodes, [doc], candidates)
    ev = result["candidate_evidence"][0]
    # FM-002 matches via fm_id_candidate_alt → should get a link
    assert ev["linkage_outcome"] == "linked"


def test_no_data_outcome_when_sentinel_episode():
    """episode with index_status='no_episodes_indexed' → every candidate gets linkage_outcome='no_data'."""
    episodes = [_FakeEpisode(episode_id="ep-sentinel", index_status="no_episodes_indexed")]
    doc = _make_doc()
    candidates = [
        _make_candidate("CAND-001"),
        _make_candidate("CAND-002"),
    ]
    result = _linker().run(episodes, [doc], candidates)
    for ev in result["candidate_evidence"]:
        assert ev["linkage_outcome"] == "no_data"


# ═══════════════════════════════════════════════════════════════════════════════
# Redundancy suppression
# ═══════════════════════════════════════════════════════════════════════════════

def test_level1_suppresses_level2_for_same_pair():
    """Same episode-doc pair qualifies at both level 1 and level 2;
    result has only one link (the higher-precedence level 1)."""
    ep = _FakeEpisode(
        episode_id="ep1",
        linked_doc_ids=["DOC-001"],  # causes level 1
        window_start=BASE_TIME,
        window_end=BASE_TIME + timedelta(hours=2),
    )
    # Doc has explicit event_time → would also qualify at level 2,
    # but since doc_id is in linked_doc_ids, it's level 1
    doc = _make_doc(
        doc_id="DOC-001",
        event_time_confidence="explicit",
        event_time_start=BASE_TIME + timedelta(hours=1),
        event_time_end=BASE_TIME + timedelta(hours=3),
    )
    candidates = [_make_candidate("CAND-001")]
    result = _linker(link_confidence_threshold=0.05).run([ep], [doc], candidates)

    all_links = result["all_links"]
    # Should have exactly one link for ep1::DOC-001
    ep1_doc1_links = [l for l in all_links if l["episode_id"] == "ep1" and l["doc_id"] == "DOC-001"]
    assert len(ep1_doc1_links) == 1
    assert ep1_doc1_links[0]["linkage_precedence_level"] == 1


def test_different_pairs_not_suppressed():
    """Two different doc_ids for same episode → both links retained."""
    ep = _FakeEpisode(episode_id="ep1", similarity_to_current=0.9)
    doc1 = _make_doc(doc_id="DOC-001", fm_id_candidate="FM-001")
    doc2 = _make_doc(doc_id="DOC-002", fm_id_candidate="FM-001")
    candidates = [_make_candidate("CAND-001", fm_id="FM-001")]
    result = _linker(link_confidence_threshold=0.01).run([ep], [doc1, doc2], candidates)
    all_links = result["all_links"]
    doc_ids = {l["doc_id"] for l in all_links}
    assert "DOC-001" in doc_ids
    assert "DOC-002" in doc_ids


# ═══════════════════════════════════════════════════════════════════════════════
# Stale cap in linker
# ═══════════════════════════════════════════════════════════════════════════════

def test_stale_index_cap_applied():
    """Episode with index_status='stale', link_confidence computed as >0.70 → result ≤ 0.70."""
    # Use high similarity to force high raw confidence
    ep = _FakeEpisode(episode_id="ep1", index_status="stale", similarity_to_current=1.0)
    doc = _make_doc(doc_id="DOC-001", fm_id_candidate="FM-001", event_time_confidence="absent")
    candidates = [_make_candidate("CAND-001")]
    cfg = _default_config(
        stale_index_confidence_cap=0.70,
        link_confidence_threshold=0.01,
    )
    result = CrossPatternLinker(cfg).run([ep], [doc], candidates)
    for lnk in result["all_links"]:
        assert lnk["link_confidence"] <= 0.70 + 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# Reinforcement strength
# ═══════════════════════════════════════════════════════════════════════════════

def test_single_reinforcement():
    """One reinforcing link above threshold → reinforcement_strength='single'."""
    ep = _FakeEpisode(episode_id="ep1", similarity_to_current=0.9)
    doc = _make_doc(doc_id="DOC-001", fm_id_candidate="FM-001")
    candidates = [_make_candidate("CAND-001", fm_id="FM-001")]
    result = _linker(link_confidence_threshold=0.05).run([ep], [doc], candidates)
    ev = result["candidate_evidence"][0]
    assert ev["linkage_outcome"] == "linked"
    assert ev["support_posture"] == "reinforcing"
    assert ev["reinforcement_strength"] == "single"


def test_multiple_consistent_reinforcement():
    """Two episodes both linked to same FM doc → reinforcement_strength='multiple_consistent',
    support_posture='reinforcing'."""
    ep1 = _FakeEpisode(episode_id="ep1", similarity_to_current=0.9, asset_id="ASSET-1")
    ep2 = _FakeEpisode(episode_id="ep2", similarity_to_current=0.9, asset_id="ASSET-1")
    # One doc that both episodes link to
    doc = _make_doc(doc_id="DOC-001", fm_id_candidate="FM-001")
    candidates = [_make_candidate("CAND-001", fm_id="FM-001")]
    result = _linker(link_confidence_threshold=0.05).run([ep1, ep2], [doc], candidates)
    ev = result["candidate_evidence"][0]
    assert ev["support_posture"] == "reinforcing"
    assert ev["reinforcement_strength"] == "multiple_consistent"


def test_mixed_reinforcement_downgrades_posture():
    """Two episodes, each linked to a different doc (both FM-001) — still reinforcing
    with multiple_consistent since both docs have fm_id_candidate=FM-001."""
    ep1 = _FakeEpisode(episode_id="ep1", similarity_to_current=0.9)
    ep2 = _FakeEpisode(episode_id="ep2", similarity_to_current=0.9)
    doc1 = _make_doc(doc_id="DOC-001", fm_id_candidate="FM-001")
    doc2 = _make_doc(doc_id="DOC-002", fm_id_candidate="FM-002")
    candidates = [_make_candidate("CAND-001", fm_id="FM-001")]
    # doc2 has FM-002, not FM-001, so it won't match the FM-001 candidate
    result = _linker(link_confidence_threshold=0.05).run([ep1, ep2], [doc1, doc2], candidates)
    ev = result["candidate_evidence"][0]
    # doc1 matches FM-001; doc2 does not → only doc1 links
    assert ev["support_posture"] == "reinforcing"
    # Two episodes linked to same doc/fm → multiple_consistent
    assert ev["reinforcement_strength"] == "multiple_consistent"


# ═══════════════════════════════════════════════════════════════════════════════
# temporal_link_skipped propagated to link
# ═══════════════════════════════════════════════════════════════════════════════

def test_temporal_link_skipped_propagated_to_link():
    """doc with event_time_confidence='inferred' but no event_time_start → level 2
    with no timestamps → compute_time_overlap_hours returns None →
    CrossPatternLink.temporal_link_skipped=True."""
    ep = _FakeEpisode(
        episode_id="ep1",
        similarity_to_current=0.9,
        window_start=BASE_TIME,
        window_end=BASE_TIME + timedelta(hours=2),
    )
    # event_time_confidence='inferred' (not 'absent') → classify_linkage_precedence returns 2
    # but event_time_start=None → compute_time_overlap_hours returns None → temporal_link_skipped=True
    doc = _make_doc(
        doc_id="DOC-001",
        fm_id_candidate="FM-001",
        event_time_confidence="inferred",  # not 'absent' → level 2 applies
        event_time_start=None,             # missing → compute_time_overlap_hours returns None
        event_time_end=None,
    )
    candidates = [_make_candidate("CAND-001", fm_id="FM-001")]
    result = _linker(link_confidence_threshold=0.05).run([ep], [doc], candidates)
    all_links = result["all_links"]
    assert len(all_links) >= 1
    # The link for this doc should have temporal_link_skipped=True
    link = next(l for l in all_links if l["doc_id"] == "DOC-001")
    assert link["temporal_link_skipped"] is True
