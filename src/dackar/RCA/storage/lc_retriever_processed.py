from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .chroma_store import ChromaRecordStore
from .multi_vector_fusion import reciprocal_rank_fusion, weighted_distance_inversion

LOGGER = logging.getLogger(__name__)

from .processed_record_store import ProcessedRecordStore, select_processed_snippet

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RetrieveResult:
    record_id: str
    snippet: str
    metadata: Dict[str, Any]
    per_view: Dict[str, Any]
    score: Optional[float]
    record: Optional[Dict[str, Any]]


@dataclass
class ContextPack:
    query_texts: Dict[str, str]
    results: List[RetrieveResult]


# ---------------------------------------------------------------------------
# Retriever / orchestrator
# ---------------------------------------------------------------------------

class LCProcessedRetriever:
    """
    Light adaptation of the earlier retrieval layer for processed_text_record.

    Differences from the original lc_retriever.py:
    - Uses ChromaRecordStore collections keyed by document type.
    - Treats record_id as the vector/document identity.
    - Hydrates canonical processed_text_record objects instead of raw mdParser chunks.
    """

    def __init__(self, manager: ChromaRecordStore, doc_store: ProcessedRecordStore) -> None:
        self.manager = manager
        self.store = doc_store

    def query_doc_types(
        self,
        *,
        doc_types: List[str],
        query_text: str,
        top_k_per_doc_type: int = 8,
        k_final: int = 10,
        filter_meta: Optional[Dict[str, Any]] = None,
        fusion: str = "rrf",
        view_weights: Optional[Dict[str, float]] = None,
        hybrid_weight: float = 0.5,
        snippet_preference: str = "raw_text",
    ) -> ContextPack:
        per_view: Dict[str, List[Dict[str, Any]]] = {}

        for doc_type in doc_types:
            # Ensure persisted collections are opened even if this process did
            # not perform the original ingest.
            try:
                self.manager.load_collection(doc_type=doc_type)
            except Exception as exc:
                LOGGER.warning(
                    "Failed to load collection for doc_type=%s before query: %s",
                    doc_type,
                    exc,
                )

            docs = self.manager.query_doc_type(
                doc_type=doc_type,
                query_text=query_text,
                top_k=top_k_per_doc_type,
                filter_meta=filter_meta,
                hybrid_weight=hybrid_weight,
            )
            hits: List[Dict[str, Any]] = []
            for doc in docs:
                rid = self._extract_record_id(doc)
                if not rid:
                    LOGGER.warning(
                        "Skipping vector hit with no stable record_id for doc_type=%s query=%r.",
                        doc_type,
                        query_text,
                    )
                    continue
                hits.append(
                    {
                        "record_id": rid,
                        "score": doc.metadata.get("_score"),
                        "document": doc,
                        "metadata": doc.metadata,
                    }
                )
            per_view[doc_type] = hits

        fused = self._fuse(per_view, fusion=fusion, k_final=k_final, view_weights=view_weights)

        if not fused:
            LOGGER.warning("No fused hits returned for query '%s'.", query_text)
        
        record_ids = [h["record_id"] for h in fused]
        records = self.store.get_many(record_ids)
        rec_by_id = {r["record_id"]: r for r in records}

        results: List[RetrieveResult] = []
        for hit in fused:
            rid = hit["record_id"]
            rec = rec_by_id.get(rid)
            snippet = select_processed_snippet(rec, prefer=snippet_preference)
            canonical_meta = (rec.get("metadata") or {}) if isinstance(rec, dict) else {}
            result_meta = {
                **canonical_meta,
                "_vector_metadata": hit.get("metadata") or {},
            }
            results.append(
                RetrieveResult(
                    record_id=rid,
                    snippet=snippet,
                    metadata=result_meta,
                    per_view=hit.get("views") or {},
                    score=hit.get("score"),
                    record=rec,
                )
            )

        return ContextPack(query_texts={"query": query_text}, results=results)

    def _fuse(
        self,
        per_view_hits: Dict[str, List[Dict[str, Any]]],
        *,
        fusion: str,
        k_final: int,
        view_weights: Optional[Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        if fusion == "rrf":
            return reciprocal_rank_fusion(per_view_hits, k=k_final, view_weights=view_weights)
        return weighted_distance_inversion(per_view_hits, k=k_final, view_weights=view_weights)
    
    def _extract_record_id(self, doc: Any) -> Optional[str]:
        """
        Best-effort stable identifier extraction from a Chroma hit.
        Never synthesize record_id from page_content hash, because hydration
        must be based on the canonical processed_text_record identity.
        """
        metadata = getattr(doc, "metadata", {}) or {}
        for key in ("record_id", "id", "_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for attr in ("id", "record_id"):
            value = getattr(doc, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return None
