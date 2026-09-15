from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .lc_retriever_processed import LCProcessedRetriever


@dataclass
class ProcessedEvidenceStoreAdapter:
    retriever: LCProcessedRetriever
    default_doc_types: Optional[List[str]] = None
    top_k_per_doc_type: int = 8
    k_final: int = 10
    fusion: str = "rrf"
    hybrid_weight: float = 0.5
    snippet_preference: str = "raw_text"

    def query(
        self,
        query_text: str,
        *,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        filters = dict(filters or {})

        doc_types = filters.get("doc_type") or self.default_doc_types or [
            "CR", "WO", "SOP", "RCA", "ECA", "FMEA", "MANUAL"
        ]

        if isinstance(doc_types, str):
            doc_types = [doc_types]

        filter_meta: Dict[str, Any] = {}
        if "doc_ids" in filters:
            filter_meta["doc_ids"] = filters["doc_ids"]
        if "doc_type" in filters:
            filter_meta["doc_types"] = doc_types

        # asset_id and component_ids both address equipment identity. Records persist component
        # identity (primary_component_id / component_ids) but no dedicated asset_id metadata key,
        # so an asset_id filter is routed through the component_ids path (asset == component for
        # retrieval). When both are supplied their values are unioned (order-preserving de-dup).
        component_filter: List[str] = []
        for key in ("component_ids", "asset_id"):
            if key not in filters:
                continue
            val = filters[key]
            if isinstance(val, (list, tuple, set)):
                component_filter.extend(str(x) for x in val if x is not None and str(x).strip())
            elif val is not None and str(val).strip():
                component_filter.append(str(val))
        if component_filter:
            seen: set = set()
            deduped: List[str] = []
            for cid in component_filter:
                if cid not in seen:
                    seen.add(cid)
                    deduped.append(cid)
            filter_meta["component_ids"] = deduped

        ctx = self.retriever.query_doc_types(
            doc_types=list(doc_types),
            query_text=query_text,
            top_k_per_doc_type=self.top_k_per_doc_type,
            k_final=min(top_k, self.k_final),
            filter_meta=filter_meta,
            fusion=self.fusion,
            hybrid_weight=self.hybrid_weight,
            snippet_preference=self.snippet_preference,
        )

        out: List[Dict[str, Any]] = []
        for r in ctx.results:
            meta = dict(r.metadata or {})
            vec_meta = dict(meta.get("_vector_metadata") or {})
            rec = r.record or {}

            out.append({
                "record_id": r.record_id,
                "chunk_id": (
                    ((rec.get("provenance") or {}).get("chunk_id"))
                    or vec_meta.get("chunk_id")
                    or r.record_id
                ),
                "snippet": r.snippet,
                "score": r.score,
                "metadata": meta,
                "doc_id": rec.get("doc_id") or meta.get("doc_id") or vec_meta.get("doc_id"),
                "doc_type": rec.get("doc_type") or meta.get("doc_type") or vec_meta.get("doc_type"),
                "component_id": meta.get("component_id") or vec_meta.get("component_id"),
                "record": rec,
            })

        return out