"""
test_cross_pattern_rules.py — Phase 4 unit tests for cross_pattern/rules.py

Coverage:
  - compute_link_confidence: weighted renormalization, missing dimensions, provenance
  - compute_time_overlap_hours: absent confidence → None, explicit timestamps → positive
  - apply_stale_confidence_cap: cap applied / not applied
  - classify_linkage_precedence: levels 1, 2, 3
  - classify_support_posture: all posture branches
  - classify_linkage_outcome: all outcome branches
"""
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

# Stub heavy optional dependencies
for _mod in (
    "neo4j", "py2neo", "chromadb",
    "langchain_chroma", "langchain_community",
    "langchain_community.vectorstores", "langchain_community.embeddings",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from cross_pattern.rules import (
    compute_link_confidence,
    compute_time_overlap_hours,
    apply_stale_confidence_cap,
    classify_linkage_precedence,
    classify_support_posture,
    classify_linkage_outcome,
)
from cross_pattern.models import CrossPatternLink, HistoricalDocExtraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(
    doc_id: str = "DOC-001",
    event_time_confidence: str = "explicit",
    event_time_start: Optional[datetime] = None,
    event_time_end: Optional[datetime] = None,
    fm_id_candidate: Optional[str] = "FM-001",
    fm_id_candidate_alt: Optional[str] = None,
    asset_id: Optional[str] = "ASSET-1",
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


def _make_link(
    link_confidence: float = 0.50,
    linkage_precedence_level: int = 3,
    temporal_link_skipped: bool = False,
    provenance: Optional[Dict[str, Any]] = None,
    ep_index_status: str = "indexed",
) -> CrossPatternLink:
    prov = dict(provenance or {})
    prov.setdefault("ep_index_status", ep_index_status)
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
        link_confidence=link_confidence,
        provenance=prov,
    )


class _FakeEpisode:
    def __init__(self, index_status: str = "indexed", similarity_to_current: float = 0.8):
        self.index_status = index_status
        self.similarity_to_current = similarity_to_current
        self.episode_id = "ep1"


# ═══════════════════════════════════════════════════════════════════════════════
# compute_link_confidence — §4.2 normalization
# ═══════════════════════════════════════════════════════════════════════════════

def test_link_confidence_all_four_dimensions():
    """signal=0.8, temporal=0.9, fm=0.7, doc=0.6 → weighted renormalized result."""
    prov: Dict[str, Any] = {}
    # All weights present: 0.30 + 0.20 + 0.20 + 0.30 = 1.0
    result = compute_link_confidence(
        signal_similarity_score=0.8,
        time_overlap_hours=1.0,          # non-None → temporal contributes
        temporal_compatibility_score=0.9,
        fm_alignment_score=0.7,
        document_similarity_score=0.6,
        provenance=prov,
    )
    expected = (0.30 * 0.8 + 0.20 * 0.9 + 0.20 * 0.7 + 0.30 * 0.6) / 1.0
    assert abs(result - expected) < 1e-6
    assert prov["link_confidence_terms"]["total_weight"] == 1.0


def test_link_confidence_temporal_absent():
    """temporal=None → only signal+fm+doc contribute; total_weight=0.80."""
    prov: Dict[str, Any] = {}
    # signal=0.30, fm=0.20, doc=0.30; temporal=0 → total=0.80
    result = compute_link_confidence(
        signal_similarity_score=0.6,
        time_overlap_hours=None,        # temporal absent
        temporal_compatibility_score=None,
        fm_alignment_score=0.5,
        document_similarity_score=0.4,
        provenance=prov,
    )
    expected = (0.30 * 0.6 + 0.20 * 0.5 + 0.30 * 0.4) / 0.80
    assert abs(result - expected) < 1e-6
    terms = prov["link_confidence_terms"]
    assert terms["temporal_weight"] == 0.0
    assert abs(terms["total_weight"] - 0.80) < 1e-6


def test_link_confidence_fm_and_doc_absent():
    """only signal → result == signal_score (total_weight=0.30, 0.30*s/0.30)."""
    prov: Dict[str, Any] = {}
    result = compute_link_confidence(
        signal_similarity_score=0.75,
        time_overlap_hours=None,
        temporal_compatibility_score=None,
        fm_alignment_score=None,
        document_similarity_score=None,
        provenance=prov,
    )
    assert abs(result - 0.75) < 1e-6
    terms = prov["link_confidence_terms"]
    assert terms["fm_weight"] == 0.0
    assert terms["document_weight"] == 0.0
    assert abs(terms["total_weight"] - 0.30) < 1e-6


def test_link_confidence_provenance_records_missing_dims():
    """When fm=None, provenance should note fm was absent (fm_alignment_score=None)."""
    prov: Dict[str, Any] = {}
    compute_link_confidence(
        signal_similarity_score=0.7,
        time_overlap_hours=None,
        temporal_compatibility_score=None,
        fm_alignment_score=None,
        document_similarity_score=0.5,
        provenance=prov,
    )
    terms = prov["link_confidence_terms"]
    assert terms["fm_alignment_score"] is None
    assert terms["fm_weight"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# compute_time_overlap_hours — temporal_link_skipped logic
# ═══════════════════════════════════════════════════════════════════════════════

BASE_TIME = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


def test_temporal_link_skipped_when_event_time_confidence_absent():
    """event_time_confidence='absent' → compute_time_overlap_hours returns None."""
    doc = _make_doc(event_time_confidence="absent")
    result = compute_time_overlap_hours(
        episode_window_start=BASE_TIME,
        episode_window_end=BASE_TIME + timedelta(hours=2),
        doc=doc,
        max_gap_days=180.0,
    )
    assert result is None


def test_temporal_link_not_skipped_when_explicit():
    """valid overlapping timestamps with event_time_confidence='explicit' → positive float hours."""
    doc = _make_doc(
        event_time_confidence="explicit",
        event_time_start=BASE_TIME + timedelta(hours=1),
        event_time_end=BASE_TIME + timedelta(hours=3),
    )
    result = compute_time_overlap_hours(
        episode_window_start=BASE_TIME,
        episode_window_end=BASE_TIME + timedelta(hours=2),
        doc=doc,
        max_gap_days=180.0,
    )
    assert result is not None
    assert result > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# apply_stale_confidence_cap
# ═══════════════════════════════════════════════════════════════════════════════

def test_stale_cap_reduces_link_confidence():
    """link with confidence=0.90 and stale cap=0.70 → result confidence=0.70, stale_index_cap_applied=True."""
    link = _make_link(link_confidence=0.90)
    capped = apply_stale_confidence_cap(link, cap=0.70)
    assert abs(capped.link_confidence - 0.70) < 1e-9
    assert capped.provenance["stale_index_cap_applied"] is True
    assert capped.provenance["stale_index_cap_value"] == 0.70
    assert "pre_cap_link_confidence" in capped.provenance


def test_stale_cap_not_applied_when_below_cap():
    """link with confidence=0.50 and cap=0.70 → result confidence=0.50 (unchanged)."""
    link = _make_link(link_confidence=0.50)
    result = apply_stale_confidence_cap(link, cap=0.70)
    assert abs(result.link_confidence - 0.50) < 1e-9
    assert result.provenance["stale_index_cap_applied"] is False
    assert "pre_cap_link_confidence" not in result.provenance


# ═══════════════════════════════════════════════════════════════════════════════
# classify_linkage_precedence
# ═══════════════════════════════════════════════════════════════════════════════

def test_precedence_level1_when_doc_id_in_episode_refs():
    """doc_id in episode_source_refs → level 1."""
    doc = _make_doc(doc_id="DOC-001", event_time_confidence="explicit")
    level = classify_linkage_precedence(
        episode_id="ep1",
        doc=doc,
        episode_source_refs=["DOC-001", "DOC-002"],
    )
    assert level == 1


def test_precedence_level2_when_asset_matches_and_event_time_known():
    """event_time_confidence='explicit', asset matches → level 2."""
    doc = _make_doc(doc_id="DOC-002", event_time_confidence="explicit")
    level = classify_linkage_precedence(
        episode_id="ep1",
        doc=doc,
        episode_source_refs=[],  # doc_id not in refs
    )
    assert level == 2


def test_precedence_level3_fallback():
    """no direct ref, event_time_confidence='absent' → level 3."""
    doc = _make_doc(doc_id="DOC-003", event_time_confidence="absent")
    level = classify_linkage_precedence(
        episode_id="ep1",
        doc=doc,
        episode_source_refs=[],
    )
    assert level == 3


# ═══════════════════════════════════════════════════════════════════════════════
# classify_support_posture
# ═══════════════════════════════════════════════════════════════════════════════

def test_support_posture_reinforcing_single():
    """one reinforcing FM → ('reinforcing', 'single')."""
    posture, strength = classify_support_posture(
        reinforcing_fm_ids=["FM-001"],
        conflicting_fm_ids=[],
    )
    assert posture == "reinforcing"
    assert strength == "single"


def test_support_posture_reinforcing_multiple_consistent():
    """two reinforcing, same FM → ('reinforcing', 'multiple_consistent')."""
    posture, strength = classify_support_posture(
        reinforcing_fm_ids=["FM-001", "FM-001"],
        conflicting_fm_ids=[],
    )
    assert posture == "reinforcing"
    assert strength == "multiple_consistent"


def test_support_posture_mixed_downgrades():
    """two reinforcing different FMs → ('weakly_supporting', 'mixed')."""
    posture, strength = classify_support_posture(
        reinforcing_fm_ids=["FM-001", "FM-002"],
        conflicting_fm_ids=[],
    )
    assert posture == "weakly_supporting"
    assert strength == "mixed"


def test_support_posture_conflicting():
    """any conflicting → ('conflicting', None)."""
    posture, strength = classify_support_posture(
        reinforcing_fm_ids=["FM-001"],
        conflicting_fm_ids=["FM-999"],
    )
    assert posture == "conflicting"
    assert strength is None


def test_support_posture_unresolved_when_empty():
    """no reinforcing, no conflicting → ('unresolved', None)."""
    posture, strength = classify_support_posture(
        reinforcing_fm_ids=[],
        conflicting_fm_ids=[],
    )
    assert posture == "unresolved"
    assert strength is None


# ═══════════════════════════════════════════════════════════════════════════════
# classify_linkage_outcome
# ═══════════════════════════════════════════════════════════════════════════════

def test_outcome_no_data_when_all_episodes_no_episodes_indexed():
    """all episodes have index_status='no_episodes_indexed' → 'no_data'."""
    episodes = [_FakeEpisode(index_status="no_episodes_indexed")]
    docs = [_make_doc()]
    outcome = classify_linkage_outcome(
        episodes=episodes,
        candidate_links=[],
        doc_extractions=docs,
        link_confidence_threshold=0.25,
    )
    assert outcome == "no_data"


def test_outcome_no_data_when_no_doc_extractions():
    """episodes present but doc_extractions=[] → 'no_data'."""
    episodes = [_FakeEpisode(index_status="indexed")]
    outcome = classify_linkage_outcome(
        episodes=episodes,
        candidate_links=[],
        doc_extractions=[],
        link_confidence_threshold=0.25,
    )
    assert outcome == "no_data"


def test_outcome_no_match_when_no_links():
    """episodes and docs present but candidate_links=[] → 'no_match'."""
    episodes = [_FakeEpisode(index_status="indexed")]
    docs = [_make_doc()]
    outcome = classify_linkage_outcome(
        episodes=episodes,
        candidate_links=[],
        doc_extractions=docs,
        link_confidence_threshold=0.25,
    )
    assert outcome == "no_match"


def test_outcome_below_threshold_when_all_links_below():
    """all links below threshold → 'below_threshold'."""
    episodes = [_FakeEpisode(index_status="indexed")]
    docs = [_make_doc()]
    links = [_make_link(link_confidence=0.10)]
    outcome = classify_linkage_outcome(
        episodes=episodes,
        candidate_links=links,
        doc_extractions=docs,
        link_confidence_threshold=0.25,
    )
    assert outcome == "below_threshold"


def test_outcome_linked_when_one_link_above_threshold():
    """at least one link above threshold → 'linked'."""
    episodes = [_FakeEpisode(index_status="indexed")]
    docs = [_make_doc()]
    links = [_make_link(link_confidence=0.60)]
    outcome = classify_linkage_outcome(
        episodes=episodes,
        candidate_links=links,
        doc_extractions=docs,
        link_confidence_threshold=0.25,
    )
    assert outcome == "linked"
