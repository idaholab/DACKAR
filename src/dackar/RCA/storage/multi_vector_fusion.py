# multi_vector_fusion.py
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    per_view_hits: Dict[str, List[Dict[str, Any]]],
    k: int = 10,
    rrf_k: int = 60,
    view_weights: Optional[Dict[str, float]] = None,
    min_votes: int = 1,
) -> List[Dict[str, Any]]:
    """
    Fuse ranked hit lists from multiple retrieval views using Reciprocal Rank
    Fusion (RRF).

    RRF is rank-only — it does not rely on raw similarity scores — making it
    robust when different views return scores on incompatible scales (e.g.
    L2 distance vs. BM25 score).  Each document's contribution from a view
    at rank *r* (0-indexed) is ``weight / (rrf_k + r + 1)``.

    Args:
        per_view_hits:
            Mapping of ``view_name -> list[hit]``.  Each hit must have at
            least a ``"record_id"`` key.  Optional keys ``"score"``
            (raw similarity), ``"metadata"``, and ``"document"`` are
            preserved on the first occurrence.
        k:
            Maximum number of results to return after fusion.
        rrf_k:
            Smoothing constant (default 60 per the original RRF paper).
            Higher values reduce the penalty for lower ranks.
        view_weights:
            Optional per-view scaling factors (default 1.0 for all views).
            Use to up-weight a higher-quality view, e.g.
            ``{"bm25": 1.5, "dense": 1.0}``.
        min_votes:
            Minimum number of views that must have returned a chunk for it
            to appear in the output.  Chunks retrieved by only one view are
            often lower-confidence matches; set ``min_votes=2`` to enforce
            cross-view agreement.

    Returns:
        List[dict]: Up to *k* fused hit dicts, sorted by descending RRF score.
        Each dict contains:
        - ``"record_id"`` (str)
        - ``"score"``    (float) — cumulative RRF score
        - ``"votes"``    (list[str]) — view names that returned this chunk
        - ``"views"``    (dict) — per-view rank and raw score
        - ``"metadata"`` (dict | None) — from the first view that provided it
        - ``"document"`` (Any | None)  — LC Document object if available
    """
    view_weights = view_weights or {}

    # Accumulate RRF scores per chunk
    agg: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "record_id": None,
        "score": 0.0,
        "votes": [],
        "views": {},
        "metadata": None,
        "document": None,
    })

    for view, hits in per_view_hits.items():
        w = view_weights.get(view, 1.0)
        for rank, hit in enumerate(hits):
            cid = hit.get("record_id")
            if not cid:
                continue

            entry = agg[cid]
            entry["record_id"] = cid
            entry["score"] += w * (1.0 / (rrf_k + rank + 1))
            entry["votes"].append(view)
            entry["views"][view] = {"rank": rank, "score_raw": hit.get("score")}

            # Preserve metadata / document from the first view that supplies them
            if entry["metadata"] is None and hit.get("metadata"):
                entry["metadata"] = hit["metadata"]
            if entry["document"] is None and hit.get("document"):
                entry["document"] = hit["document"]

    # Filter by vote threshold, then sort by score descending
    results = [
        entry for entry in agg.values()
        if len(entry["votes"]) >= min_votes
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]


# ---------------------------------------------------------------------------
# Weighted Distance Inversion
# ---------------------------------------------------------------------------

def weighted_distance_inversion(
    per_view_hits: Dict[str, List[Dict[str, Any]]],
    k: int = 10,
    view_weights: Optional[Dict[str, float]] = None,
    min_votes: int = 1,
) -> List[Dict[str, Any]]:
    """
    Fuse hit lists from multiple retrieval views by inverting raw distance
    scores into similarity-like contributions.

    Assumes *score* is a non-negative distance (lower = more similar), as
    returned by Chroma's ``similarity_search_with_score`` under L2 or cosine
    distance metrics.  The contribution of each hit is
    ``weight / (1 + distance)``, mapping distance 0 → contribution 1.0.

    Hits whose ``"score"`` is ``None`` (e.g. from the BM25 fallback path
    which does not produce a numeric distance) are **skipped** rather than
    contributing 0.0, which would incorrectly penalise them relative to
    scored hits from other views.  (Issue 13)

    Args:
        per_view_hits:
            Mapping of ``view_name -> list[hit]``.  Each hit must have at
            least a ``"record_id"`` key.  ``"score"`` should be a non-negative
            float distance; ``None`` is explicitly handled.
        k:
            Maximum number of results to return.
        view_weights:
            Optional per-view scaling factors (default 1.0).
        min_votes:
            Minimum number of views that must contribute a non-None score for
            a chunk to appear in the output.

    Returns:
        List[dict]: Up to *k* fused hit dicts sorted by descending score.
        Same schema as :func:`reciprocal_rank_fusion`.

    Note:
        If you switch Chroma to return cosine *similarity* (higher = better),
        this function's inversion logic will become incorrect.  The metric
        assumption (distance, not similarity) must match the Chroma collection
        configuration.
    """
    view_weights = view_weights or {}

    agg: Dict[str, Dict[str, Any]] = {}

    for view, hits in per_view_hits.items():
        w = view_weights.get(view, 1.0)
        for hit in hits:
            cid = hit.get("record_id")
            if not cid:
                continue

            raw = hit.get("score")

            # Issue 13 fix: skip None scores rather than adding 0 contribution.
            # A None score means the retrieval path (e.g. BM25) did not produce
            # a numeric distance; treating it as 0 would under-rank these hits
            # relative to chunks with real scores from other views.
            if raw is None:
                continue

            # Clip to [0, ∞) as a defensive measure against unexpected negative
            # values (e.g. numerical noise in inner-product spaces).
            distance = max(0.0, float(raw))
            contribution = w * (1.0 / (1.0 + distance))

            entry = agg.setdefault(cid, {
                "record_id": cid,
                "score": 0.0,
                "votes": [],
                "views": {},
                "metadata": None,
                "document": None,
            })
            entry["score"] += contribution
            entry["votes"].append(view)
            entry["views"][view] = {"score_raw": raw}

            if entry["metadata"] is None and hit.get("metadata"):
                entry["metadata"] = hit["metadata"]
            if entry["document"] is None and hit.get("document"):
                entry["document"] = hit["document"]

    results = [
        entry for entry in agg.values()
        if len(entry["votes"]) >= min_votes
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]