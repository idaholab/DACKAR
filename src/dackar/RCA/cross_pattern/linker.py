from __future__ import annotations

"""CrossPatternLinker — builds cross-pattern evidence for the RCA pipeline.

Phase 2 of the cross-pattern linkage layer.  Consumes historical signal
episodes (from PatternSearcher) and historical doc extractions (from
DocExtractionStore) and produces CandidateCrossPatternEvidence for each
causality candidate.
"""

import dataclasses
import logging
from typing import Any, Dict, List, Optional

from .config import CrossPatternConfig
from .models import (
    CandidateCrossPatternEvidence,
    CrossPatternLink,
    HistoricalDocExtraction,
)
from .rules import (
    apply_stale_confidence_cap,
    classify_linkage_outcome,
    classify_linkage_precedence,
    classify_support_posture,
    compute_link_confidence,
    compute_time_overlap_hours,
)

LOGGER = logging.getLogger(__name__)


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses (and nested structures) to plain dicts."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result: Dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            result[f.name] = _dataclass_to_dict(getattr(obj, f.name))
        return result
    if isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, frozenset):
        return sorted(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return obj


class CrossPatternLinker:
    """Links historical signal episodes to doc extractions for each RCA candidate.

    Usage
    -----
    linker = CrossPatternLinker(config)
    result = linker.run(episodes, doc_extractions, candidates)

    The returned dict is JSON-serializable and contains:
    - "candidate_evidence": list of CandidateCrossPatternEvidence as dicts
    - "all_links": all CrossPatternLink as dicts
    - "summary": top-level counts and distribution
    """

    def __init__(self, config: CrossPatternConfig) -> None:
        self.config = config

    def run(
        self,
        episodes: List[Any],                  # list[HistoricalSignalEpisode]
        doc_extractions: List[HistoricalDocExtraction],
        candidates: List[Dict[str, Any]],     # each has candidate_id, component_id, fm_id
    ) -> Dict[str, Any]:
        """Build cross-pattern evidence.

        Algorithm per candidate
        -----------------------
        1. Check index_status on all episodes.  If all are "no_episodes_indexed"
           or no doc_extractions exist → outcome = "no_data".
        2. Filter episodes by signal_similarity_floor.
        3. For each surviving episode × doc_extraction pair:
           a. Check asset compatibility (same asset_id).
           b. Determine precedence level.
           c. Compute temporal overlap (level ≤ 2) or skip (level 3).
           d. Apply temporal gate if mode == "gate".
           e. Compute fm_alignment_score: 1.0 when fm_id_candidate matches
              candidate.fm_id, else None.
           f. Compute document_similarity_score: None (Phase 2 placeholder).
           g. Compute temporal_compatibility_score from overlap hours.
           h. Compute link_confidence.
           i. Apply stale cap when episode.index_status == "stale".
        4. Redundancy suppression: for each (episode_id, doc_id) pair keep only
           the highest-precedence link.
        5. Filter links by link_confidence_threshold.
        6. Build CandidateCrossPatternEvidence.
        7. Mutate doc.source_episode_ids for each linked doc.
        """
        cfg = self.config

        all_links_for_result: List[CrossPatternLink] = []
        candidate_evidences: List[CandidateCrossPatternEvidence] = []
        outcome_distribution: Dict[str, int] = {
            "linked": 0,
            "no_data": 0,
            "no_match": 0,
            "below_threshold": 0,
        }

        for candidate in candidates:
            cand_id = str(candidate.get("candidate_id") or "")
            comp_id = str(candidate.get("component_id") or "")
            fm_id = str(candidate.get("fm_id") or "")

            # Step 1 — check data availability early
            eligible_statuses = {"indexed", "stale"}
            has_eligible = any(
                getattr(ep, "index_status", "no_episodes_indexed") in eligible_statuses
                for ep in episodes
            )
            if not has_eligible or not doc_extractions:
                outcome = "no_data"
                evidence = CandidateCrossPatternEvidence(
                    candidate_id=cand_id,
                    component_id=comp_id,
                    fm_id=fm_id,
                    linked_episode_ids=[],
                    linked_doc_ids=[],
                    best_link_score=0.0,
                    support_posture="unresolved",
                    reinforcement_strength=None,
                    linkage_outcome=outcome,
                    evidence_paths=[],
                )
                candidate_evidences.append(evidence)
                outcome_distribution[outcome] = outcome_distribution.get(outcome, 0) + 1
                continue

            # Step 2 — filter episodes by similarity floor
            passing_episodes = [
                ep for ep in episodes
                if (
                    getattr(ep, "index_status", "no_episodes_indexed") in eligible_statuses
                    and getattr(ep, "similarity_to_current", 0.0) >= cfg.signal_similarity_floor
                )
            ]

            # Build all candidate links before redundancy suppression
            raw_links: List[CrossPatternLink] = []

            for ep in passing_episodes:
                ep_id = str(getattr(ep, "episode_id", ""))
                ep_asset = str(getattr(ep, "asset_id", ""))
                ep_window_start = getattr(ep, "window_start", None)
                ep_window_end = getattr(ep, "window_end", None)
                ep_sim = float(getattr(ep, "similarity_to_current", 0.0))
                ep_index_status = str(getattr(ep, "index_status", "no_episodes_indexed"))

                # Collect episode source references for level-1 check
                # (linked_doc_ids on episode if pre-populated, plus any known refs)
                ep_source_refs: List[str] = list(getattr(ep, "linked_doc_ids", []) or [])

                for doc in doc_extractions:
                    # Episode-to-candidate mapping: link only when doc fm matches candidate
                    doc_fm = doc.fm_id_candidate or ""
                    doc_fm_alt = doc.fm_id_candidate_alt or ""
                    fm_matches = (
                        doc_fm == fm_id
                        or doc_fm_alt == fm_id
                    )

                    # For level-3 fallback (no temporal link), still require FM match
                    if not fm_matches:
                        continue

                    # Check asset compatibility
                    doc_asset = doc.asset_id or ""
                    asset_match = (ep_asset != "" and doc_asset != "" and ep_asset == doc_asset)

                    # Determine precedence level
                    level = classify_linkage_precedence(ep_id, doc, ep_source_refs)

                    # Compute temporal overlap
                    time_overlap: Optional[float] = None
                    temporal_link_skipped = False
                    temporal_compatibility_score: Optional[float] = None
                    temporal_gate_failed = False

                    if level <= 2:
                        time_overlap = compute_time_overlap_hours(
                            ep_window_start,
                            ep_window_end,
                            doc,
                            cfg.temporal_compatibility_max_gap_days,
                        )
                        if time_overlap is None:
                            temporal_link_skipped = True
                            # Falls through to level 3 behavior
                            if level == 2:
                                # Reclassify to level 3 since temporal check cannot run
                                level = 3
                        else:
                            # Apply temporal gate
                            if cfg.temporal_compatibility_mode == "gate":
                                gap_hours = cfg.temporal_compatibility_max_gap_days * 24.0
                                if time_overlap < -gap_hours:
                                    temporal_gate_failed = True
                                else:
                                    # Temporal score: 1.0 for overlap, decaying for gap
                                    if time_overlap >= 0:
                                        temporal_compatibility_score = 1.0
                                    else:
                                        # Partial credit in gate mode: not used in score
                                        # since gate suppresses the link
                                        temporal_compatibility_score = None
                            else:
                                # Formula mode: score from overlap
                                if time_overlap >= 0:
                                    temporal_compatibility_score = 1.0
                                else:
                                    # Negative overlap = gap; score decays linearly
                                    gap_days = abs(time_overlap) / 24.0
                                    max_gap = cfg.temporal_compatibility_max_gap_days
                                    temporal_compatibility_score = max(0.0, 1.0 - gap_days / max_gap)

                    if temporal_gate_failed:
                        # Suppress link due to temporal gate
                        continue

                    # FM alignment score
                    fm_alignment_score: Optional[float] = None
                    if doc_fm == fm_id or doc_fm_alt == fm_id:
                        fm_alignment_score = 1.0

                    # Document similarity score: Phase 2 placeholder
                    document_similarity_score: Optional[float] = None

                    # Component overlap (placeholder — both episode and doc carry asset_id
                    # but not component lists; populated empty for now)
                    component_overlap: List[str] = []

                    # Compute link confidence
                    prov: Dict[str, Any] = {
                        "linkage_precedence_level": level,
                        "temporal_link_skipped": temporal_link_skipped,
                        "asset_match": asset_match,
                        "ep_index_status": ep_index_status,
                        "fm_id_candidate": doc_fm,
                        "fm_id_candidate_alt": doc_fm_alt,
                        "candidate_fm_id": fm_id,
                    }

                    link_confidence = compute_link_confidence(
                        signal_similarity_score=ep_sim,
                        time_overlap_hours=time_overlap if (time_overlap is not None and time_overlap >= 0) else None,
                        temporal_compatibility_score=temporal_compatibility_score,
                        fm_alignment_score=fm_alignment_score,
                        document_similarity_score=document_similarity_score,
                        provenance=prov,
                    )

                    link_id = f"{ep_id}::{doc.doc_id}::{level}"
                    link = CrossPatternLink(
                        link_id=link_id,
                        episode_id=ep_id,
                        doc_id=doc.doc_id,
                        asset_match=asset_match,
                        time_overlap_hours=time_overlap,
                        temporal_link_skipped=temporal_link_skipped,
                        linkage_precedence_level=level,
                        component_overlap=component_overlap,
                        fm_alignment_score=fm_alignment_score,
                        signal_similarity_score=ep_sim,
                        document_similarity_score=document_similarity_score,
                        link_confidence=link_confidence,
                        provenance=prov,
                    )

                    # Apply stale cap if needed
                    if ep_index_status == "stale":
                        link = apply_stale_confidence_cap(link, cfg.stale_index_confidence_cap)

                    raw_links.append(link)

            # Step 4 — Redundancy suppression: for each (episode_id, doc_id) pair,
            # keep only the highest-precedence (lowest level number) link.
            best_pair: Dict[str, CrossPatternLink] = {}
            for lnk in raw_links:
                pair_key = f"{lnk.episode_id}::{lnk.doc_id}"
                existing = best_pair.get(pair_key)
                if existing is None:
                    best_pair[pair_key] = lnk
                elif lnk.linkage_precedence_level < existing.linkage_precedence_level:
                    best_pair[pair_key] = lnk
                elif (
                    lnk.linkage_precedence_level == existing.linkage_precedence_level
                    and lnk.link_confidence > existing.link_confidence
                ):
                    best_pair[pair_key] = lnk

            deduplicated_links = list(best_pair.values())

            # Step 5 — classify linkage outcome using pre-threshold links
            outcome = classify_linkage_outcome(
                episodes=episodes,
                candidate_links=deduplicated_links,
                doc_extractions=doc_extractions,
                link_confidence_threshold=cfg.link_confidence_threshold,
            )
            outcome_distribution[outcome] = outcome_distribution.get(outcome, 0) + 1

            # Filter to above-threshold links for evidence_paths
            above_threshold = [
                lnk for lnk in deduplicated_links
                if lnk.link_confidence >= cfg.link_confidence_threshold
            ]

            # Record all_links (deduplicated, pre-threshold — for audit)
            all_links_for_result.extend(deduplicated_links)

            # Step 6 — classify support posture
            reinforcing_fm_ids: List[str] = []
            conflicting_fm_ids: List[str] = []
            for lnk in above_threshold:
                # Find the corresponding doc
                doc_match = next(
                    (d for d in doc_extractions if d.doc_id == lnk.doc_id), None
                )
                if doc_match is None:
                    continue
                linked_fm = doc_match.fm_id_candidate or ""
                if linked_fm == fm_id or (doc_match.fm_id_candidate_alt or "") == fm_id:
                    reinforcing_fm_ids.append(linked_fm or fm_id)
                else:
                    conflicting_fm_ids.append(linked_fm)

            support_posture, reinforcement_strength = classify_support_posture(
                reinforcing_fm_ids=reinforcing_fm_ids,
                conflicting_fm_ids=conflicting_fm_ids,
            )

            # Build episode/doc id lists
            linked_episode_ids = sorted(set(lnk.episode_id for lnk in above_threshold))
            linked_doc_ids = sorted(set(lnk.doc_id for lnk in above_threshold))
            best_link_score = (
                max(lnk.link_confidence for lnk in above_threshold)
                if above_threshold
                else 0.0
            )

            # Mutate doc.source_episode_ids for linked docs
            for doc in doc_extractions:
                if doc.doc_id in linked_doc_ids:
                    for ep_id_ref in linked_episode_ids:
                        if ep_id_ref not in doc.source_episode_ids:
                            doc.source_episode_ids.append(ep_id_ref)

            evidence = CandidateCrossPatternEvidence(
                candidate_id=cand_id,
                component_id=comp_id,
                fm_id=fm_id,
                linked_episode_ids=linked_episode_ids,
                linked_doc_ids=linked_doc_ids,
                best_link_score=round(best_link_score, 4),
                support_posture=support_posture,
                reinforcement_strength=reinforcement_strength,
                linkage_outcome=outcome,
                evidence_paths=above_threshold,
            )
            candidate_evidences.append(evidence)

        links_above_threshold = sum(
            1 for lnk in all_links_for_result
            if lnk.link_confidence >= cfg.link_confidence_threshold
        )

        summary: Dict[str, Any] = {
            "total_episodes": len([ep for ep in episodes if getattr(ep, "episode_id", "")]),
            "total_doc_extractions": len(doc_extractions),
            "total_links_built": len(all_links_for_result),
            "links_above_threshold": links_above_threshold,
            "linkage_outcome_distribution": outcome_distribution,
        }

        return {
            "candidate_evidence": [_dataclass_to_dict(ev) for ev in candidate_evidences],
            "all_links": [_dataclass_to_dict(lnk) for lnk in all_links_for_result],
            "summary": summary,
        }
