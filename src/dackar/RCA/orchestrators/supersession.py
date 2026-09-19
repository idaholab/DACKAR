from __future__ import annotations

"""Phase C — Evidence bundle supersession pass (ADR-1 + ADR-2, 2026-04-30).

resolve_supersession() runs between EvidenceRetriever.retrieve() and
CausalityEngine.refine_with_evidence().  It zeroes the support_score of
analyzes-class evidence hits that are superseded by a higher-authority hit
covering the same candidate, then patches candidate_evidence_summary so that
refine_with_evidence() sees only the post-supersession best scores.

Authority hierarchy (ADR-2):
  1. plant RCA with formal_conclusion
  2. plant ECA with formal_conclusion  /  any RCA or ECA
  3. plant CR with preliminary_assessment
  4. fleet OE  (finding_status == "fleet_experience")
  5. industry OE  (doc_type OE, other finding_status)
  6. any other analyzes-class (LER, etc.)

Rules:
  - Only analyzes_past_degradation hits participate in supersession.
  - Higher authority supersedes lower authority for the same candidate.
  - Equal authority + different known recency → most recent wins; older zeroed.
  - Equal authority + recency unknown or tied → both contribute (no supersession).
  - Cross-class hits (monitors, affects, characterizes) are never superseded
    and never supersede.
"""

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

JsonDict = Dict[str, Any]

# String constants — mirror schema.EpistemicClass without importing across packages
_ANALYZES_CLASS = "analyzes_past_degradation"
_AFFECTS_CLASS = "affects_performance"

# P-4 — relevance gate margin.
# Authority and relevance are different axes. A higher-authority hit may only
# supersede a lower-authority hit when it is at least *nearly as on-point*: its
# relevance must be within this margin of the hit it would erase. A higher-authority
# but off-point hit (relevance more than the margin below the on-point lower-authority
# hit) does NOT supersede it — the most on-point evidence survives. Kept modest so the
# authority hierarchy still governs the normal case (comparable relevance).
_RELEVANCE_SUPERSEDE_MARGIN = 0.15

# doc_type sets used as fallback when epistemic_class annotation is absent
_ANALYZES_DOC_TYPES: Set[str] = {"RCA", "ECA", "OE", "LER"}
_AFFECTS_DOC_TYPES: Set[str] = {"WO"}
# CR is analyzes-class only when finding_status indicates a formal assessment
_CR_ANALYZES_STATUSES: Set[str] = {"preliminary_assessment", "formal_conclusion"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _epistemic_class_from_meta(meta: JsonDict) -> str:
    """Return epistemic class string, preferring Phase A annotation over doc_type fallback."""
    ep = str(meta.get("epistemic_class") or "").strip().lower()
    if ep:
        return ep
    doc_type = str(meta.get("doc_type") or "").upper().strip()
    if doc_type == "CR":
        # CR earns analyzes-class only with an explicit assessment finding_status
        finding_status = str(meta.get("finding_status") or "").lower().strip()
        if finding_status in _CR_ANALYZES_STATUSES:
            return _ANALYZES_CLASS
        return ""
    if doc_type in _ANALYZES_DOC_TYPES:
        return _ANALYZES_CLASS
    if doc_type in _AFFECTS_DOC_TYPES:
        return _AFFECTS_CLASS
    return ""


def _is_analyzes_class(meta: JsonDict) -> bool:
    return _epistemic_class_from_meta(meta) == _ANALYZES_CLASS


def _authority_rank(meta: JsonDict) -> int:
    """Return authority rank — lower integer = higher authority."""
    doc_type = str(meta.get("doc_type") or "").upper().strip()
    finding_status = str(meta.get("finding_status") or "").lower().strip()
    if doc_type == "RCA" and finding_status == "formal_conclusion":
        return 1
    if doc_type == "ECA" and finding_status == "formal_conclusion":
        return 2
    if doc_type in ("RCA", "ECA"):
        return 2
    if doc_type == "CR" and finding_status == "preliminary_assessment":
        return 3
    if doc_type == "OE" and finding_status == "fleet_experience":
        return 4
    if doc_type == "OE":
        return 5
    return 6  # other analyzes-class (LER, etc.)


def _relevance_of(hit: JsonDict) -> float:
    """Relevance (on-pointness) proxy for the P-4 supersession gate.

    Prefers an explicit relevance / semantic-overlap annotation when the retriever
    supplies one, else falls back to the retrieval ``support_score`` (the pre-
    supersession value if a prior pass already zeroed it). This is the axis
    supersession must respect so a higher-authority but *off-point* hit does not erase
    strongly on-point lower-authority support.
    """
    meta = hit.get("metadata") or {}
    for field in ("relevance_score", "semantic_overlap", "on_point_score"):
        val = meta.get(field)
        if isinstance(val, (int, float)):
            return float(val)
    val = hit.get("support_score_original", hit.get("support_score"))
    try:
        return float(val or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _recency_dt(meta: JsonDict) -> Optional[datetime]:
    """Extract best available recency timestamp from hit metadata.  Returns None when absent."""
    for field in (
        "event_time_start", "event_date", "document_date",
        "event_start", "created_at", "timestamp_start",
    ):
        val = meta.get(field)
        if val and isinstance(val, str):
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except Exception:
                pass
    return None


def _patch_candidate_summary(
    summary: List[JsonDict],
    results: List[JsonDict],
) -> None:
    """Recompute score-affecting fields of candidate_evidence_summary after supersession.

    Only patches best_support_score, supporting_count, supporting_snippet_ids, and
    best_source_tier — all fields touched by support_score zeroing.  spaCy aggregates
    and NER entity lists are metadata-derived and unaffected.
    """
    by_cid: Dict[str, List[JsonDict]] = {}
    for hit in results:
        meta = hit.get("metadata") or {}
        cid = meta.get("linked_candidate_id") or meta.get("candidate_id")
        if cid:
            by_cid.setdefault(str(cid), []).append(hit)

    for row in summary:
        cid = str(row.get("candidate_id") or "")
        hits = by_cid.get(cid, [])
        best_score = 0.0
        count = 0
        ids: List[str] = []
        best_tier: Optional[str] = None
        best_tier_sc = -1.0
        for hit in hits:
            meta = hit.get("metadata") or {}
            if meta.get("support_role") == "supporting":
                sc = float(hit.get("support_score") or 0.0)
                best_score = max(best_score, sc)
                count += 1
                sid = hit.get("snippet_id")
                if sid:
                    ids.append(sid)
                if sc >= best_tier_sc:
                    tier = meta.get("source_tier")
                    best_tier = str(tier) if tier else None
                    best_tier_sc = sc
        row["best_support_score"] = best_score
        row["supporting_count"] = count
        row["supporting_snippet_ids"] = ids
        if best_tier is not None:
            row["best_source_tier"] = best_tier


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_supersession(
    bundle: JsonDict,
    epistemics_policy_version: Optional[str] = None,
) -> JsonDict:
    """Zero support_score of analyzes-class hits superseded by higher-authority hits.

    Modifies bundle in-place.  On each superseded hit sets:
      ``superseded = True``, ``support_score_original`` (original value),
      ``support_score = 0.0``, and downgrades ``metadata.support_role`` from
      "supporting" to "contextual".

    Patches ``bundle["candidate_evidence_summary"]`` so that
    ``refine_with_evidence`` sees the post-supersession best scores.

    Adds provenance fields to the bundle:
      ``supersession_applied``, ``supersession_count``,
      ``supersession_policy_version`` (when policy_version is supplied).
    """
    results: Optional[List[JsonDict]] = bundle.get("results")
    if not results:
        bundle["supersession_applied"] = False
        bundle["supersession_count"] = 0
        return bundle

    # Group analyzes-class hit indices by candidate_id
    by_candidate: Dict[str, List[int]] = {}
    for idx, hit in enumerate(results):
        meta = hit.get("metadata") or {}
        cid = meta.get("linked_candidate_id") or meta.get("candidate_id")
        if not cid:
            continue
        if not _is_analyzes_class(meta):
            continue
        by_candidate.setdefault(str(cid), []).append(idx)

    superseded: Set[int] = set()
    relevance_retained = 0

    for cid, indices in by_candidate.items():
        if len(indices) <= 1:
            continue

        group = [(idx, results[idx].get("metadata") or {}) for idx in indices]

        # Step 1 — authority rank filter, gated by relevance (P-4).
        # A lower-authority hit is only superseded when at least one best-rank
        # (would-be superseding) hit is nearly as on-point (within the relevance
        # margin). If every higher-authority hit is materially *less* relevant than
        # this hit, the hit is the most on-point evidence and is retained despite its
        # lower authority — authority must not erase relevance.
        best_rank = min(_authority_rank(meta) for _, meta in group)
        best_rank_relevance = max(
            _relevance_of(results[idx])
            for idx, meta in group
            if _authority_rank(meta) == best_rank
        )
        for idx, meta in group:
            if _authority_rank(meta) <= best_rank:
                continue
            this_relevance = _relevance_of(results[idx])
            if best_rank_relevance >= this_relevance - _RELEVANCE_SUPERSEDE_MARGIN:
                superseded.add(idx)
            else:
                # Higher-authority evidence is off-point relative to this hit — retain.
                results[idx]["supersession_relevance_retained"] = True
                results[idx]["supersession_relevance"] = round(this_relevance, 6)
                results[idx]["supersession_superior_relevance"] = round(best_rank_relevance, 6)
                relevance_retained += 1

        # Step 2 — recency tiebreak, scoped to survivors of the *same* authority rank.
        # (Relevance-retained lower-authority survivors must not be re-erased by a
        # newer higher-authority hit — that would defeat the Step-1 relevance gate.)
        survivors = [(idx, meta) for idx, meta in group if idx not in superseded]
        if len(survivors) <= 1:
            continue
        by_rank: Dict[int, List] = defaultdict(list)
        for idx, meta in survivors:
            by_rank[_authority_rank(meta)].append((idx, meta))
        for _rank, bucket in by_rank.items():
            if len(bucket) <= 1:
                continue
            dated = [(idx, _recency_dt(meta)) for idx, meta in bucket]
            known = [(idx, dt) for idx, dt in dated if dt is not None]
            if not known:
                continue  # no recency info — both contribute
            latest_dt = max(dt for _, dt in known)
            for idx, dt in dated:
                if dt is not None and dt < latest_dt:
                    superseded.add(idx)
            # Ties or unknown recency: contribute without supersession

    # Apply supersession: zero support_score, downgrade role
    for idx in superseded:
        hit = results[idx]
        hit["superseded"] = True
        hit["support_score_original"] = hit.get("support_score", 0.0)
        hit["support_score"] = 0.0
        meta = hit.get("metadata") or {}
        if meta.get("support_role") == "supporting":
            meta["support_role"] = "contextual"
            meta["superseded_downgrade"] = True

    # Patch candidate_evidence_summary to reflect zeroed scores
    summary = bundle.get("candidate_evidence_summary")
    if isinstance(summary, list) and superseded:
        _patch_candidate_summary(summary, results)

    bundle["supersession_applied"] = bool(superseded)
    bundle["supersession_count"] = len(superseded)
    bundle["supersession_relevance_retained_count"] = relevance_retained
    if epistemics_policy_version:
        bundle["supersession_policy_version"] = epistemics_policy_version

    return bundle
