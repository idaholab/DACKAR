from __future__ import annotations

"""Phase D — EpistemicsDigest builder (§7.4, 2026-04-30).

build_epistemics_digests() runs post-refine_with_evidence(), before synthesis.
It produces a per-candidate structured digest from:
  - causality_candidates (for observationally_ungrounded flag set by Phase C)
  - evidence_bundle["results"] (for analyzes/affects hit enumeration)

The digest is the contract between the epistemics module and the synthesizer.
Both the LLM prompt and the deterministic fallback path must consume it.
"""

from typing import Any, Dict, List, Optional

# Borrow constants and helpers from the supersession module — same package,
# no circular dependency risk.
from orchestrators.supersession import (
    _authority_rank,
    _epistemic_class_from_meta,
    _ANALYZES_CLASS,
    _AFFECTS_CLASS,
)

JsonDict = Dict[str, Any]


def _authority_level(meta: JsonDict) -> str:
    """Map authority rank to §7.4 level string."""
    rank = _authority_rank(meta)
    if rank <= 2:
        return "mandatory"
    if rank <= 4:
        return "guidance"
    return "informational"


def _classification_resolution_level(meta: JsonDict) -> str:
    """How the epistemic class was determined for this hit."""
    if meta.get("epistemic_class"):
        return "annotation"
    doc_type = str(meta.get("doc_type") or "").upper().strip()
    if doc_type:
        return "doc_type_fallback"
    return "default"


def build_epistemics_digests(
    causality_candidates: JsonDict,
    results: List[JsonDict],
) -> Dict[str, JsonDict]:
    """Return {candidate_id → EpistemicsDigest dict} for all candidates.

    Digest schema per §7.4:
        candidate_id, analyzes_support_count, analyzes_support_items,
        affects_support_present, affects_support_items,
        observationally_ungrounded, causal_grounding_absent,
        degraded_classification_count, confidence_cap
    """
    candidates = causality_candidates.get("candidates") or []

    by_cid: Dict[str, List[JsonDict]] = {}
    for hit in results:
        meta = hit.get("metadata") or {}
        cid = meta.get("linked_candidate_id") or meta.get("candidate_id")
        if cid:
            by_cid.setdefault(str(cid), []).append(hit)

    digests: Dict[str, JsonDict] = {}

    for cand in candidates:
        cid = str(cand.get("candidate_id") or "")
        if not cid:
            continue

        hits = by_cid.get(cid, [])
        analyzes_items: List[JsonDict] = []
        affects_items: List[JsonDict] = []
        degraded_count = 0

        for hit in hits:
            meta = hit.get("metadata") or {}
            ep_class = _epistemic_class_from_meta(meta)
            if not meta.get("epistemic_class"):
                degraded_count += 1

            if ep_class == _ANALYZES_CLASS:
                analyzes_items.append({
                    "source_id": hit.get("snippet_id") or "",
                    "authority_level": _authority_level(meta),
                    "superseded": bool(hit.get("superseded", False)),
                })
            elif ep_class == _AFFECTS_CLASS:
                affects_items.append({
                    "source_id": hit.get("snippet_id") or "",
                    "component_id": str(meta.get("component_id") or ""),
                    "within_precursor_window": bool(meta.get("within_precursor_window", True)),
                })

        analyzes_count = sum(1 for i in analyzes_items if not i["superseded"])
        affects_present = len(affects_items) > 0
        observationally_ungrounded = bool(cand.get("observationally_ungrounded", False))
        causal_grounding_absent = analyzes_count == 0
        confidence_cap: Optional[str] = "medium" if observationally_ungrounded else None

        digests[cid] = {
            "candidate_id": cid,
            "analyzes_support_count": analyzes_count,
            "analyzes_support_items": analyzes_items,
            "affects_support_present": affects_present,
            "affects_support_items": affects_items,
            "observationally_ungrounded": observationally_ungrounded,
            "causal_grounding_absent": causal_grounding_absent,
            "degraded_classification_count": degraded_count,
            "confidence_cap": confidence_cap,
        }

    return digests


def build_epistemics_run_summary(
    causality_candidates: JsonDict,
    results: List[JsonDict],
    evidence_bundle: JsonDict,
    calibration_profile_name: Optional[str] = None,
    calibration_profile_version: Optional[str] = None,
) -> JsonDict:
    """Build run_manifest.epistemics_summary (§7.4 manifest consequence).

    Records: hit counts by epistemic class, supersession edges, degraded
    classification counts by artifact type, classification resolution level
    distribution, and calibration profile reference.
    """
    candidates = causality_candidates.get("candidates") or []

    class_counts: Dict[str, int] = {}
    degraded_by_doc_type: Dict[str, int] = {}
    resolution_level_counts: Dict[str, int] = {}

    for hit in results:
        meta = hit.get("metadata") or {}
        ep_class = _epistemic_class_from_meta(meta) or "unknown"
        class_counts[ep_class] = class_counts.get(ep_class, 0) + 1

        res_level = _classification_resolution_level(meta)
        resolution_level_counts[res_level] = resolution_level_counts.get(res_level, 0) + 1

        if not meta.get("epistemic_class"):
            doc_type = str(meta.get("doc_type") or "unknown").upper().strip()
            degraded_by_doc_type[doc_type] = degraded_by_doc_type.get(doc_type, 0) + 1

    supersession_count = int(evidence_bundle.get("supersession_count", 0) or 0)

    per_candidate = []
    for cand in candidates:
        cid = str(cand.get("candidate_id") or "")
        if not cid:
            continue
        per_candidate.append({
            "candidate_id": cid,
            "observationally_ungrounded": bool(cand.get("observationally_ungrounded", False)),
            "confidence_label": cand.get("confidence_label"),
            "confidence_label_cap_reason": cand.get("confidence_label_cap_reason"),
        })

    return {
        "hit_counts_by_epistemic_class": class_counts,
        "supersession_edge_count": supersession_count,
        "supersession_policy_version": evidence_bundle.get("supersession_policy_version"),
        "degraded_classification_by_doc_type": degraded_by_doc_type,
        "classification_resolution_level_distribution": resolution_level_counts,
        "per_candidate_grounding": per_candidate,
        "calibration_profile": {
            "name": calibration_profile_name,
            "version": calibration_profile_version,
        },
    }
