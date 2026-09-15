from __future__ import annotations

"""Summary and attention-flag utilities for cross-pattern linkage.

These functions consume CandidateCrossPatternEvidence objects (or their dict
representations) and produce text or structured summaries for injection into
rca_card and run_manifest.
"""

from typing import Any, Dict, List, Optional

from .models import CandidateCrossPatternEvidence

# RCA-card wording from §4.7 linkage_outcome table
_OUTCOME_WORDING: Dict[str, str] = {
    "linked": "",  # normal cross-pattern summary; handled by format function
    "no_data": (
        "No historical signal or document data available for cross-pattern assessment."
    ),
    "no_match": (
        "No historically similar signal episodes were found for this event."
    ),
    "below_threshold": (
        "Historical episodes and documents were found but could not be reliably linked."
    ),
}


def format_rca_card_cross_pattern_summary(
    candidate_evidences: List[CandidateCrossPatternEvidence],
    linkage_outcome_distribution: Dict[str, int],
) -> str:
    """Return a brief narrative string for injection into rca_card.

    Uses exact wording from §4.7 linkage_outcome table for each outcome type.
    When linked candidates exist, summarizes support posture across them.
    """
    if not candidate_evidences:
        return _OUTCOME_WORDING["no_data"]

    linked = [ev for ev in candidate_evidences if ev.linkage_outcome == "linked"]
    no_data = linkage_outcome_distribution.get("no_data", 0)
    no_match = linkage_outcome_distribution.get("no_match", 0)
    below = linkage_outcome_distribution.get("below_threshold", 0)
    total = len(candidate_evidences)

    if not linked:
        # Pick the most informative outcome wording
        if no_data == total:
            return _OUTCOME_WORDING["no_data"]
        if no_match > 0:
            return _OUTCOME_WORDING["no_match"]
        if below > 0:
            return _OUTCOME_WORDING["below_threshold"]
        return _OUTCOME_WORDING["no_data"]

    # Summarize linked candidates
    reinforcing = [ev for ev in linked if ev.support_posture == "reinforcing"]
    conflicting = [ev for ev in linked if ev.support_posture == "conflicting"]
    weakly = [ev for ev in linked if ev.support_posture == "weakly_supporting"]

    parts: List[str] = []

    if reinforcing:
        strong = [ev for ev in reinforcing if ev.reinforcement_strength == "multiple_consistent"]
        single = [ev for ev in reinforcing if ev.reinforcement_strength == "single"]
        if strong:
            parts.append(
                f"Cross-pattern evidence strongly reinforces {len(strong)} candidate(s) "
                f"with multiple consistent historical links."
            )
        if single:
            parts.append(
                f"Cross-pattern evidence reinforces {len(single)} candidate(s) "
                f"with a single historical link."
            )

    if conflicting:
        parts.append(
            f"Cross-pattern evidence conflicts with {len(conflicting)} candidate(s); "
            f"analyst review is required."
        )

    if weakly:
        parts.append(
            f"Weak or mixed cross-pattern support found for {len(weakly)} candidate(s)."
        )

    if no_match > 0:
        parts.append(_OUTCOME_WORDING["no_match"])
    elif below > 0:
        parts.append(_OUTCOME_WORDING["below_threshold"])

    return "  ".join(parts) if parts else (
        f"Cross-pattern linkage completed for {len(linked)} candidate(s)."
    )


def build_manifest_cross_pattern_summary(
    candidate_evidences: List[CandidateCrossPatternEvidence],
    total_episodes: int,
    total_docs: int,
    total_links: int,
    links_above_threshold: int,
) -> Dict[str, Any]:
    """Return structured summary for run_manifest.artifacts.

    Includes per-candidate linkage outcome and support posture breakdowns.
    """
    outcome_dist: Dict[str, int] = {
        "linked": 0,
        "no_data": 0,
        "no_match": 0,
        "below_threshold": 0,
    }
    posture_dist: Dict[str, int] = {}
    precedence_dist: Dict[int, int] = {}
    temporal_skipped_count = 0

    for ev in candidate_evidences:
        outcome_dist[ev.linkage_outcome] = outcome_dist.get(ev.linkage_outcome, 0) + 1
        posture_dist[ev.support_posture] = posture_dist.get(ev.support_posture, 0) + 1
        for lnk in ev.evidence_paths:
            lvl = lnk.linkage_precedence_level
            precedence_dist[lvl] = precedence_dist.get(lvl, 0) + 1
            if lnk.temporal_link_skipped:
                temporal_skipped_count += 1

    candidate_summaries = [
        {
            "candidate_id": ev.candidate_id,
            "fm_id": ev.fm_id,
            "linkage_outcome": ev.linkage_outcome,
            "support_posture": ev.support_posture,
            "reinforcement_strength": ev.reinforcement_strength,
            "best_link_score": ev.best_link_score,
            "linked_episode_count": len(ev.linked_episode_ids),
            "linked_doc_count": len(ev.linked_doc_ids),
            "evidence_path_count": len(ev.evidence_paths),
        }
        for ev in candidate_evidences
    ]

    return {
        "present": True,
        "total_episodes_retrieved": total_episodes,
        "total_doc_extractions": total_docs,
        "total_links_built": total_links,
        "links_above_threshold": links_above_threshold,
        "linkage_outcome_distribution": outcome_dist,
        "support_posture_distribution": posture_dist,
        "linkage_precedence_distribution": {
            str(k): v for k, v in sorted(precedence_dist.items())
        },
        "temporal_link_skipped_count": temporal_skipped_count,
        "candidate_summaries": candidate_summaries,
    }


def get_cross_pattern_attention_flags(
    candidate_evidences: List[CandidateCrossPatternEvidence],
    candidates: List[Dict[str, Any]],       # ranked candidates (index 0 = top)
    top_n_candidates: int = 3,
) -> List[str]:
    """Return attention flag strings for injection into analyst_attention_flags.

    Flags raised:
    - "conflicting" support posture on any of top_n_candidates
    - "multiple_consistent" reinforcement (positive signal) on top candidates
    - "no_data" outcome (data coverage gap) for any candidate
    - "stale" index (detected via provenance on any link)
    """
    flags: List[str] = []

    # Build lookup by candidate_id
    evidence_by_id: Dict[str, CandidateCrossPatternEvidence] = {
        ev.candidate_id: ev for ev in candidate_evidences
    }

    # Determine top-N candidate IDs (by order in candidates list)
    top_ids = [
        str(c.get("candidate_id") or "")
        for c in candidates[:top_n_candidates]
        if c.get("candidate_id")
    ]

    for cand_id in top_ids:
        ev = evidence_by_id.get(cand_id)
        if ev is None:
            continue

        if ev.support_posture == "conflicting":
            flags.append(
                f"Cross-pattern evidence conflicts with candidate {cand_id} "
                f"(fm_id={ev.fm_id}); analyst review required."
            )

        if (
            ev.support_posture == "reinforcing"
            and ev.reinforcement_strength == "multiple_consistent"
        ):
            flags.append(
                f"Multiple consistent cross-pattern links reinforce candidate {cand_id} "
                f"(fm_id={ev.fm_id})."
            )

    # Check all candidates for no_data
    no_data_cands = [ev for ev in candidate_evidences if ev.linkage_outcome == "no_data"]
    if no_data_cands:
        fm_ids_str = ", ".join(sorted(set(ev.fm_id for ev in no_data_cands))[:5])
        flags.append(
            "No historical signal or document data available for cross-pattern "
            f"assessment on {len(no_data_cands)} candidate(s) ({fm_ids_str})."
        )

    # Check for stale index via provenance on any link
    stale_detected = False
    for ev in candidate_evidences:
        for lnk in ev.evidence_paths:
            if (lnk.provenance or {}).get("ep_index_status") == "stale":
                stale_detected = True
                break
        if stale_detected:
            break

    if stale_detected:
        flags.append(
            "Signal episode index is stale; cross-pattern results may not reflect "
            "recent plant history."
        )

    return flags
