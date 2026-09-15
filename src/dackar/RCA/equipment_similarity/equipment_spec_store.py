"""
equipment_spec_store — EquipmentSpecStore.

Thin wrapper around ChromaRecordStore for the ``equipment_specs`` doc_type.
One document per plant component; ``embedding_text`` is the natural-language
spec string produced by EquipmentSpecBuilder.

The full Stage 1-6 enrichment pipeline (NLP, NER, summarization) is not used
here — equipment specs are structured, not unstructured prose, so a minimal
record with just the spec text and identity metadata is sufficient.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]

# Sentinel doc_type — must be consistent with collection naming
EQUIPMENT_SPECS_DOC_TYPE = "EQUIPMENT_SPECS"


class EquipmentSpecStore:
    """
    Manages the ``equipment_specs`` Chroma collection.

    Parameters
    ----------
    chroma_store:
        A ``ChromaRecordStore`` instance (from ``storage/chroma_store.py``).
        The store is used as-is; no modifications are made to its configuration.
    """

    def __init__(self, chroma_store: Any) -> None:
        self.chroma_store = chroma_store
        self.doc_type = EQUIPMENT_SPECS_DOC_TYPE

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_component(
        self,
        component_id: str,
        spec_text: str,
        metadata: Optional[JsonDict] = None,
    ) -> None:
        """
        Upsert one component spec into the Chroma collection.

        Parameters
        ----------
        component_id:
            KG element_usage node ID.  Used as ``doc_id`` and for stable
            deduplication (same component_id → same Chroma record_id).
        spec_text:
            Natural-language spec string from ``EquipmentSpecBuilder``.
        metadata:
            Optional extra metadata (component_label, component_type, etc.)
            stored alongside the embedding for retrieval context.
        """
        if not spec_text or not spec_text.strip():
            logger.warning("EquipmentSpecStore: skipping empty spec_text for component_id=%s", component_id)
            return

        record_id = f"equip_spec::{component_id}"
        meta = dict(metadata or {})
        meta["component_id"] = component_id

        record: JsonDict = {
            "record_id":      record_id,
            "doc_id":         component_id,
            "doc_type":       self.doc_type,
            "chunk_index":    0,
            "embedding_text": spec_text,
            "metadata":       meta,
            "provenance": {
                "chunk_id":           f"{record_id}::0",
                "authority_level":    "informational",
                "section_role":       "equipment_specification",
                "index_in_vector_store": True,
                "page_start":         None,
                "page_end":           None,
            },
            "enrichment": {},
        }

        self.chroma_store.upsert_records([record], doc_type=self.doc_type)

    def upsert_batch(self, components: List[JsonDict]) -> int:
        """
        Upsert a batch of component spec dicts.

        Each dict must have keys: ``component_id``, ``spec_text``, and
        optionally ``metadata``.

        Returns the number of records upserted.
        """
        records: List[JsonDict] = []
        for comp in components:
            component_id = comp.get("component_id") or ""
            spec_text    = comp.get("spec_text") or ""
            metadata     = comp.get("metadata") or {}

            if not component_id or not spec_text.strip():
                continue

            record_id = f"equip_spec::{component_id}"
            meta = dict(metadata)
            meta["component_id"] = component_id

            records.append({
                "record_id":      record_id,
                "doc_id":         component_id,
                "doc_type":       self.doc_type,
                "chunk_index":    0,
                "embedding_text": spec_text,
                "metadata":       meta,
                "provenance": {
                    "chunk_id":              f"{record_id}::0",
                    "authority_level":       "informational",
                    "section_role":          "equipment_specification",
                    "index_in_vector_store": True,
                    "page_start":            None,
                    "page_end":              None,
                },
                "enrichment": {},
            })

        if not records:
            return 0
        return self.chroma_store.upsert_records(records, doc_type=self.doc_type)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def find_similar(
        self,
        query_text: str,
        top_k: int = 10,
        exclude_ids: Optional[List[str]] = None,
    ) -> list:
        """
        Find components with specs most similar to ``query_text``.

        Parameters
        ----------
        query_text:
            Natural-language description of the target equipment (built from
            kg_context by ``EquipmentSimilarityResolver._build_query_text()``).
        top_k:
            Maximum number of results to return (after exclude_ids filtering).
        exclude_ids:
            component_ids to exclude from results (the target component itself).

        Returns
        -------
        List[langchain_core.documents.Document]
            Each Document has ``page_content`` (spec text) and ``metadata``
            including ``component_id``, ``_score``, and other stored properties.
            Returns ``[]`` if the collection has not been populated yet.
        """
        exclude = set(exclude_ids or [])

        try:
            fetch_k = top_k + len(exclude) + 5  # over-fetch to absorb exclusions
            hits = self.chroma_store.query_doc_type(
                self.doc_type,
                query_text,
                top_k=fetch_k,
            )
        except ValueError:
            # Collection not yet initialized (no specs populated)
            logger.debug("EquipmentSpecStore: collection not initialised — returning empty.")
            return []
        except Exception as exc:
            logger.warning("EquipmentSpecStore.find_similar() failed: %s", exc)
            return []

        filtered = [
            doc for doc in hits
            if doc.metadata.get("component_id") not in exclude
        ]
        return filtered[:top_k]

    def component_count(self) -> int:
        """
        Return the number of component specs currently in the collection.
        Returns 0 if the collection has not been populated.
        """
        try:
            state = self.chroma_store.get_or_create_collection(self.doc_type)
            return state.vectorstore._collection.count()
        except Exception:
            return 0
