from __future__ import annotations

"""Business-logic functions for cross-pattern linkage.

All functions here are pure (no I/O).  The linker calls them; callers are
responsible for managing state.
"""

import dataclasses
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .models import CrossPatternLink, HistoricalDocExtraction

# Import lazily to avoid circular dependency at module level; functions that
# need HistoricalSignalEpisode accept it as Any and use attribute access.


def compute_link_confidence(
    signal_similarity_score: float,
    time_overlap_hours: Optional[float],
    temporal_compatibility_score: Optional[float],
    fm_alignment_score: Optional[float],
    document_similarity_score: Optional[float],
    provenance: Dict[str, Any],  # mutated to record which terms contributed
) -> float:
    """Renormalized weighted formula from §4.2.

    Always present: signal (weight 0.30).
    Temporal (0.20), FM (0.20), document (0.30) contribute only if non-None.
    The formula is renormalized over the sum of present weights so that a
    missing dimension does not silently deflate confidence relative to other
    links.

    ``provenance`` is mutated to record the contributing terms, their raw
    weights, and the normalization factor.
    """
    present_weights: Dict[str, float] = {}
    present_weights["signal"] = 0.30  # always present
    present_weights["temporal"] = 0.20 if time_overlap_hours is not None else 0.0
    present_weights["fm"] = 0.20 if fm_alignment_score is not None else 0.0
    present_weights["document"] = 0.30 if document_similarity_score is not None else 0.0

    total_weight = sum(present_weights.values())
    # total_weight is always > 0 because signal is always present

    numerator = (
        present_weights["signal"] * float(signal_similarity_score)
        + present_weights["temporal"] * float(temporal_compatibility_score or 0.0)
        + present_weights["fm"] * float(fm_alignment_score or 0.0)
        + present_weights["document"] * float(document_similarity_score or 0.0)
    )

    confidence = numerator / total_weight

    # Record provenance
    provenance["link_confidence_terms"] = {
        "signal_weight": present_weights["signal"],
        "temporal_weight": present_weights["temporal"],
        "fm_weight": present_weights["fm"],
        "document_weight": present_weights["document"],
        "total_weight": round(total_weight, 4),
        "signal_similarity_score": round(float(signal_similarity_score), 4),
        "temporal_compatibility_score": (
            round(float(temporal_compatibility_score), 4)
            if temporal_compatibility_score is not None
            else None
        ),
        "fm_alignment_score": (
            round(float(fm_alignment_score), 4) if fm_alignment_score is not None else None
        ),
        "document_similarity_score": (
            round(float(document_similarity_score), 4)
            if document_similarity_score is not None
            else None
        ),
        "raw_confidence": round(confidence, 4),
    }

    return float(max(0.0, min(1.0, confidence)))


def classify_linkage_precedence(
    episode_id: str,
    doc: HistoricalDocExtraction,
    episode_source_refs: List[str],
) -> int:
    """Return the precedence level for an (episode, doc) pair.

    Level 1 — Direct reference: doc.doc_id appears in episode_source_refs.
    Level 2 — Temporal + asset: event_time_confidence != "absent" AND asset
               information is available (not "absent" confidence alone implies
               temporal data is usable).
    Level 3 — Fallback (semantic/FM): used when neither level 1 nor 2 applies.

    Parameters
    ----------
    episode_id:
        The episode identifier (carried for provenance; not used in logic).
    doc:
        The HistoricalDocExtraction being evaluated.
    episode_source_refs:
        Raw event/document references carried on the episode (e.g. linked
        CR IDs, work-order IDs, or source_event_id values).
    """
    if doc.doc_id in episode_source_refs:
        return 1
    if doc.event_time_confidence != "absent":
        return 2
    return 3


def compute_time_overlap_hours(
    episode_window_start: Optional[datetime],
    episode_window_end: Optional[datetime],
    doc: HistoricalDocExtraction,
    max_gap_days: float,
) -> Optional[float]:
    """Compute temporal overlap (in hours) between episode window and doc event window.

    Returns
    -------
    float
        Positive → windows overlap; value is overlap duration in hours.
        Negative → gap between windows (negative float); the caller can apply
                   the max_gap_days gate.
    None
        Returned when:
        - ``doc.event_time_confidence == "absent"``
        - any required timestamp is missing (episode or doc)
    """
    if doc.event_time_confidence == "absent":
        return None

    ep_start = episode_window_start
    ep_end = episode_window_end
    doc_start = doc.event_time_start
    doc_end = doc.event_time_end

    if ep_start is None or doc_start is None:
        return None

    # Use end = start when end is absent
    ep_end_safe = ep_end if ep_end is not None else ep_start
    doc_end_safe = doc_end if doc_end is not None else doc_start

    # Ensure timezone-aware comparison (treat naive as UTC)
    def _ensure_tz(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    ep_start = _ensure_tz(ep_start)
    ep_end_safe = _ensure_tz(ep_end_safe)
    doc_start = _ensure_tz(doc_start)
    doc_end_safe = _ensure_tz(doc_end_safe)

    # Overlap: max(0, min(end_a, end_b) - max(start_a, start_b))
    overlap_start = max(ep_start, doc_start)
    overlap_end = min(ep_end_safe, doc_end_safe)
    overlap_delta = overlap_end - overlap_start

    if overlap_delta.total_seconds() >= 0:
        return overlap_delta.total_seconds() / 3600.0

    # No overlap — compute gap
    if ep_end_safe <= doc_start:
        gap = doc_start - ep_end_safe
    else:
        gap = ep_start - doc_end_safe

    gap_days = gap.total_seconds() / 86400.0
    if gap_days > max_gap_days:
        # Gap exceeds gate — signal that this should be suppressed (caller decides)
        return -(gap.total_seconds() / 3600.0)

    return -(gap.total_seconds() / 3600.0)


def classify_support_posture(
    reinforcing_fm_ids: List[str],
    conflicting_fm_ids: List[str],
) -> Tuple[str, Optional[str]]:
    """Classify the support posture for a candidate based on its linked FM IDs.

    Parameters
    ----------
    reinforcing_fm_ids:
        FM IDs from links where doc.fm_id_candidate matches the candidate FM.
    conflicting_fm_ids:
        FM IDs from links where doc.fm_id_candidate differs from the candidate FM.

    Returns
    -------
    (support_posture, reinforcement_strength)

    Rules (in precedence order):
    1. Any conflicting links → ("conflicting", None)
    2. Both lists empty → ("unresolved", None)
    3. Exactly one reinforcing → ("reinforcing", "single")
    4. Two+ reinforcing, all same fm_id → ("reinforcing", "multiple_consistent")
    5. Two+ reinforcing, mixed fm_ids → ("weakly_supporting", "mixed")
    6. reinforcing empty, conflicting empty but some docs existed → ("weakly_supporting", None)
       (Note: caller passes empty lists when no links at all; rule 2 applies first.)
    """
    if conflicting_fm_ids:
        return ("conflicting", None)

    if not reinforcing_fm_ids and not conflicting_fm_ids:
        return ("unresolved", None)

    if len(reinforcing_fm_ids) == 1:
        return ("reinforcing", "single")

    # len >= 2
    unique_fms = set(reinforcing_fm_ids)
    if len(unique_fms) == 1:
        return ("reinforcing", "multiple_consistent")

    return ("weakly_supporting", "mixed")


def classify_linkage_outcome(
    episodes: List[Any],              # list[HistoricalSignalEpisode]
    candidate_links: List[CrossPatternLink],
    doc_extractions: List[HistoricalDocExtraction],
    link_confidence_threshold: float,
) -> str:
    """Determine the linkage outcome for one candidate.

    Parameters
    ----------
    episodes:
        All episodes considered for this candidate.
    candidate_links:
        All CrossPatternLink objects built before threshold filtering (may be empty).
    doc_extractions:
        All HistoricalDocExtraction objects available.
    link_confidence_threshold:
        Minimum confidence for "linked" outcome.

    Returns
    -------
    "no_data"
        All episodes have index_status "no_episodes_indexed", or no doc_extractions exist.
    "no_match"
        Index is populated and episodes were found, but no candidate_links were built.
    "below_threshold"
        Links exist but none exceeds link_confidence_threshold.
    "linked"
        At least one link exceeds link_confidence_threshold.
    """
    # Check data availability
    eligible_statuses = {"indexed", "stale"}
    has_eligible_episodes = any(
        getattr(ep, "index_status", "no_episodes_indexed") in eligible_statuses
        for ep in episodes
    )
    if not has_eligible_episodes or not doc_extractions:
        return "no_data"

    if not candidate_links:
        return "no_match"

    if any(lnk.link_confidence >= link_confidence_threshold for lnk in candidate_links):
        return "linked"

    return "below_threshold"


def apply_stale_confidence_cap(link: CrossPatternLink, cap: float) -> CrossPatternLink:
    """Return a new CrossPatternLink with link_confidence capped at ``cap``.

    Records in ``provenance`` whether the cap was applied and at what value.
    Returns the link unchanged (but as a new copy) when confidence is already
    below the cap.
    """
    new_provenance = dict(link.provenance)
    was_capped = link.link_confidence > cap
    new_provenance["stale_index_cap_applied"] = was_capped
    new_provenance["stale_index_cap_value"] = cap
    if was_capped:
        new_provenance["pre_cap_link_confidence"] = round(link.link_confidence, 4)

    new_confidence = min(link.link_confidence, cap)

    return CrossPatternLink(
        link_id=link.link_id,
        episode_id=link.episode_id,
        doc_id=link.doc_id,
        asset_match=link.asset_match,
        time_overlap_hours=link.time_overlap_hours,
        temporal_link_skipped=link.temporal_link_skipped,
        linkage_precedence_level=link.linkage_precedence_level,
        component_overlap=list(link.component_overlap),
        fm_alignment_score=link.fm_alignment_score,
        signal_similarity_score=link.signal_similarity_score,
        document_similarity_score=link.document_similarity_score,
        link_confidence=new_confidence,
        provenance=new_provenance,
    )
