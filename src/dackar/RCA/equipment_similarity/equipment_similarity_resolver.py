"""
equipment_similarity_resolver — EquipmentSimilarityResolver.

Identifies sister equipment using two complementary tiers:

  Tier 2 — Failure mode overlap
      Derived from kg_context.failure_modes[].  Components that share
      ≥ fm_overlap_min_shared failure modes with the target are flagged.
      No Chroma, no KG re-query — purely from the already-built artifact.

  Tier 3 — Spec embedding similarity
      Queries the EquipmentSpecStore (Chroma ``equipment_specs`` collection).
      Query text is built from kg_context fields — no additional KG call.
      Skipped silently if spec_store is None or unpopulated.

Results from both tiers are merged, deduplicated (same component_id in
multiple tiers → combined match_type), and returned as a ranked list of
SisterComponent objects.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]
NON_EMBEDDING_DISTANCE = 1.0


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class SisterComponent:
    """
    A single sister equipment candidate.

    Attributes
    ----------
    component_id:
        KG element_usage node ID.
    component_label:
        Human-readable name, if available from kg_context.
    match_type:
        How this component was identified.  Possible values:
        ``"failure_mode_overlap"``, ``"spec_embedding"``,
        ``"fm_overlap+spec_embedding"``.
    shared_fm_count:
        Number of shared failure modes (Tier 2).  0 for Tier 3-only matches.
    embedding_score:
        Chroma similarity score (Tier 3). ``NON_EMBEDDING_DISTANCE`` for
        Tier 2-only matches.
        Lower score = more similar (Chroma uses distance by default).
    """

    component_id: str
    component_label: Optional[str] = None
    match_type: str = "spec_embedding"
    shared_fm_count: int = 0
    embedding_score: float = NON_EMBEDDING_DISTANCE

    def to_dict(self) -> JsonDict:
        return {
            "component_id":    self.component_id,
            "component_label": self.component_label,
            "match_type":      self.match_type,
            "shared_fm_count": self.shared_fm_count,
            "embedding_score": self.embedding_score,
        }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EquipmentSimilarityConfig:
    """
    Configuration for EquipmentSimilarityResolver.

    Parameters
    ----------
    fm_overlap_min_shared:
        Minimum number of shared failure modes for Tier 2 inclusion.
        Default: 2.  Set to 1 for more inclusive matching.
    embedding_top_k:
        Number of candidates to request from Chroma (Tier 3).
        After exclude_ids filtering, up to this many are returned.
    embedding_min_score:
        Maximum acceptable Chroma distance score.  Chroma returns
        L2 distances (lower = more similar); results above this
        threshold are filtered out.  Default: 0.8 (fairly permissive).
        Reduce to 0.4–0.6 for stricter matching.
    include_fm_overlap:
        Enable Tier 2 (failure mode overlap).
    include_spec_embedding:
        Enable Tier 3 (spec embedding).  Has no effect if spec_store is None.
    """

    fm_overlap_min_shared: int = 2
    embedding_top_k: int = 10
    embedding_min_score: float = 0.8
    include_fm_overlap: bool = True
    include_spec_embedding: bool = True


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

class EquipmentSimilarityResolver:
    """
    Resolves sister equipment using failure mode overlap and spec embeddings.

    Parameters
    ----------
    spec_store:
        ``EquipmentSpecStore`` instance (or ``None`` to disable Tier 3).
    config:
        ``EquipmentSimilarityConfig`` — defaults are conservative.
    """

    def __init__(
        self,
        spec_store: Optional[Any] = None,
        config: Optional[EquipmentSimilarityConfig] = None,
    ) -> None:
        self.spec_store = spec_store
        self.config = config or EquipmentSimilarityConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_similar(
        self,
        target_component_ids: List[str],
        kg_context: JsonDict,
    ) -> List[SisterComponent]:
        """
        Return sister equipment candidates for the given target components.

        Parameters
        ----------
        target_component_ids:
            KG component IDs of the primary asset's components.
            These are excluded from results.
        kg_context:
            KG context artifact from Stage 5A.  Provides failure modes and
            component labels for query text construction.

        Returns
        -------
        List[SisterComponent]
            Ranked by (embedding_score ASC, shared_fm_count DESC).
            Empty list if no sisters found or both tiers are disabled.
        """
        target_set: Set[str] = set(target_component_ids)
        candidates: Dict[str, SisterComponent] = {}

        # Tier 2: failure mode overlap
        if self.config.include_fm_overlap:
            fm_sisters = self._resolve_by_fm_overlap(target_set, kg_context)
            for s in fm_sisters:
                candidates[s.component_id] = s

        # Tier 3: spec embedding
        if self.config.include_spec_embedding and self.spec_store is not None:
            query_text = self._build_query_text(target_component_ids, kg_context)
            if query_text.strip():
                emb_sisters = self._resolve_by_embedding(query_text, target_set)
                for s in emb_sisters:
                    if s.component_id in candidates:
                        # Promote match_type to combined
                        existing = candidates[s.component_id]
                        existing.match_type = "fm_overlap+spec_embedding"
                        existing.embedding_score = s.embedding_score
                    else:
                        candidates[s.component_id] = s

        # Sort: lower embedding_score (closer) first; break ties by shared FM count desc
        result = sorted(
            candidates.values(),
            key=lambda s: (s.embedding_score, -s.shared_fm_count),
        )
        return result

    # ------------------------------------------------------------------
    # Tier 2 — failure mode overlap
    # ------------------------------------------------------------------

    def _resolve_by_fm_overlap(
        self,
        target_set: Set[str],
        kg_context: JsonDict,
    ) -> List[SisterComponent]:
        """
        Find components that share ≥ fm_overlap_min_shared failure modes
        with any target component.
        """
        failure_modes = kg_context.get("failure_modes") or []
        if not failure_modes:
            return []

        # Build: component_id → set of fm_ids
        comp_to_fms: Dict[str, Set[str]] = {}
        for fm in failure_modes:
            if not isinstance(fm, dict):
                continue
            # Support both field name variants
            comp_id = fm.get("component_id") or fm.get("applies_to_component_id") or ""
            fm_id   = fm.get("fm_id") or fm.get("name") or ""
            if comp_id and fm_id:
                comp_to_fms.setdefault(comp_id, set()).add(fm_id)

        # Target FM set (union across all target components)
        target_fms: Set[str] = set()
        for tid in target_set:
            target_fms |= comp_to_fms.get(tid, set())

        if not target_fms:
            return []

        # Label lookup
        label_map = self._build_label_map(kg_context)

        sisters: List[SisterComponent] = []
        for comp_id, fm_ids in comp_to_fms.items():
            if comp_id in target_set:
                continue
            shared = fm_ids & target_fms
            if len(shared) >= self.config.fm_overlap_min_shared:
                sisters.append(SisterComponent(
                    component_id=comp_id,
                    component_label=label_map.get(comp_id),
                    match_type="failure_mode_overlap",
                    shared_fm_count=len(shared),
                    embedding_score=NON_EMBEDDING_DISTANCE,
                ))

        return sorted(sisters, key=lambda s: -s.shared_fm_count)

    # ------------------------------------------------------------------
    # Tier 3 — spec embedding
    # ------------------------------------------------------------------

    def _resolve_by_embedding(
        self,
        query_text: str,
        target_set: Set[str],
    ) -> List[SisterComponent]:
        """
        Query EquipmentSpecStore and return candidates above the score threshold.
        """
        try:
            hits = self.spec_store.find_similar(
                query_text=query_text,
                top_k=self.config.embedding_top_k,
                exclude_ids=list(target_set),
            )
        except Exception as exc:
            logger.warning("EquipmentSimilarityResolver: spec_store query failed: %s", exc)
            return []

        sisters: List[SisterComponent] = []
        for doc in hits:
            meta  = doc.metadata or {}
            score = float(meta.get("_score") or meta.get("_vector_score") or 1.0)
            if score > self.config.embedding_min_score:
                continue  # too dissimilar
            comp_id = meta.get("component_id") or meta.get("doc_id") or ""
            if not comp_id or comp_id in target_set:
                continue
            sisters.append(SisterComponent(
                component_id=comp_id,
                component_label=meta.get("component_label"),
                match_type="spec_embedding",
                shared_fm_count=0,
                embedding_score=score,
            ))

        return sisters

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_query_text(
        self,
        target_component_ids: List[str],
        kg_context: JsonDict,
    ) -> str:
        """
        Build a query string from kg_context for the target components.

        Uses component label/type from ``kg_context.components[]`` and
        failure mode names from ``kg_context.failure_modes[]``.
        No KG re-query required.
        """
        target_set = set(target_component_ids)
        lines: List[str] = []

        # Component labels and types
        for comp in kg_context.get("components") or []:
            if not isinstance(comp, dict):
                continue
            if comp.get("component_id") not in target_set:
                continue
            label = comp.get("component_label") or ""
            ctype = comp.get("component_type") or ""
            if label:
                lines.append(f"Equipment: {label}")
            if ctype:
                lines.append(f"Type: {ctype}")

        # Failure mode names for the target components
        fm_names: List[str] = []
        mechanisms: List[str] = []
        for fm in kg_context.get("failure_modes") or []:
            if not isinstance(fm, dict):
                continue
            comp_id = fm.get("component_id") or fm.get("applies_to_component_id") or ""
            if comp_id not in target_set:
                continue
            name = fm.get("name") or fm.get("fm_label") or ""
            mech = fm.get("failure_mechanism") or ""
            if name:
                fm_names.append(name)
            if mech:
                mechanisms.append(mech)

        if fm_names:
            lines.append(f"Failure modes: {', '.join(list(dict.fromkeys(fm_names))[:8])}")
        if mechanisms:
            lines.append(f"Failure mechanisms: {', '.join(list(dict.fromkeys(mechanisms))[:6])}")

        return "\n".join(lines)

    def _build_label_map(self, kg_context: JsonDict) -> Dict[str, Optional[str]]:
        """Build component_id → component_label from kg_context.components[]."""
        label_map: Dict[str, Optional[str]] = {}
        for comp in kg_context.get("components") or []:
            if isinstance(comp, dict) and comp.get("component_id"):
                label_map[comp["component_id"]] = comp.get("component_label")
        return label_map
