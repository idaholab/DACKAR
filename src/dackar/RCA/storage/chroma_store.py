from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _iter_jsonl(jsonl_path: str) -> Iterable[Dict[str, Any]]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("_iter_jsonl: parse error in %s line %d: %s", jsonl_path, lineno, exc)
                continue
            if isinstance(obj, dict):
                yield obj


def _collapse_ws(s: Optional[str]) -> str:
    return " ".join((s or "").split())


# ---------------------------------------------------------------------------
# Processed record helpers
# ---------------------------------------------------------------------------

def _looks_like_processed_text_record(obj: Dict[str, Any]) -> bool:
    return (
        isinstance(obj, dict)
        and all(k in obj for k in ("record_id", "doc_id", "doc_type", "chunk_index", "embedding_text", "metadata", "provenance"))
        and isinstance(obj.get("metadata"), dict)
        and isinstance(obj.get("provenance"), dict)
        and isinstance(obj.get("record_id"), str)
        and bool(obj.get("record_id", "").strip())
    )


def extract_processed_text_record(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    rec = obj.get("processed_text_record")
    if isinstance(rec, dict) and _looks_like_processed_text_record(rec):
        return rec
    if _looks_like_processed_text_record(obj):
        return obj
    return None

def _stable_record_id_from_doc(doc: Document) -> Optional[str]:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    for key in ("record_id", "id", "_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for attr in ("id", "record_id"):
        value = getattr(doc, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None

LIST_FIELDS = {
    "equipment_ids",
    "system_names",
    "component_names",
    "mechanisms",
    "failure_outcomes",
    "maintenance_actions",
    "surveillance_actions",
    "tools_methods",
    "properties_or_limits",
}


def _sanitize_meta_value(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, list):
        # Chroma metadata is primitive-only in the common LangChain path.
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)



def _sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for k, v in meta.items():
        vv = _sanitize_meta_value(v)
        if vv is not None:
            clean[k] = vv
    return clean



def build_chroma_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    provenance = dict(record.get("provenance") or {})

    # Add a few identity / traceability keys that are useful for retrieval.
    metadata.setdefault("record_id", record.get("record_id"))
    metadata.setdefault("doc_id", record.get("doc_id"))
    metadata.setdefault("doc_type", record.get("doc_type"))
    metadata.setdefault("chunk_index", record.get("chunk_index"))
    metadata.setdefault("chunk_id", provenance.get("chunk_id"))
    metadata.setdefault("page_start", provenance.get("page_start", metadata.get("page_start")))
    metadata.setdefault("page_end", provenance.get("page_end", metadata.get("page_end")))
    metadata.setdefault("authority_level", provenance.get("authority_level", metadata.get("authority_level")))
    metadata.setdefault("section_role", provenance.get("section_role", metadata.get("section_role")))

    # Add string versions of list metadata for display / debugging. The raw list-valued
    # keys remain too, but are stringified by _sanitize_meta because Chroma metadata
    # typically expects primitive values only.
    for key in LIST_FIELDS:
        if key in metadata and isinstance(metadata[key], list):
            metadata[f"{key}_text"] = " | ".join(str(x) for x in metadata[key])

    return _sanitize_meta(metadata)



def to_chroma_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(record["record_id"]),
        "document": _collapse_ws(record.get("embedding_text") or ""),
        "metadata": build_chroma_metadata(record),
    }



def collection_name_for_doc_type(doc_type: str, prefix: str = "processed") -> str:
    dt = (doc_type or "OTHER").strip().upper().replace("/", "_").replace(" ", "_")
    return f"{prefix}_{dt}"


@dataclass
class _CollectionState:
    vectorstore: Chroma
    bm25_docs: List[Document] = field(default_factory=list)
    bm25: Optional[BM25Retriever] = None


class ChromaRecordStore:
    """
    Stage-6 store for canonical processed_text_record objects.

    One Chroma collection is used per document class (CR, MR, SOP, ECA, ...).
    Every indexed vector corresponds to one validated processed_text_record.
    """

    def __init__(
        self,
        persist_directory: str,
        *,
        embed_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        collection_prefix: str = "processed",
        bm25_k: int = 20,
    ) -> None:
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        self.embed_model = embed_model or os.environ.get("OLLAMA_EMBED_MODEL", "mxbai-embed-large:335m")
        self.ollama_base_url = ollama_base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.collection_prefix = collection_prefix
        self.bm25_k = bm25_k

        self.embedder = OllamaEmbeddings(base_url=self.ollama_base_url, model=self.embed_model)
        self._states: Dict[str, _CollectionState] = {}

    def _collection_name(self, doc_type: str) -> str:
        safe_model = self.embed_model.replace(":", "_").replace("/", "_")
        base = collection_name_for_doc_type(doc_type=doc_type, prefix=self.collection_prefix)
        return f"{base}_{safe_model}"

    def get_or_create_collection(self, doc_type: str, collection_name: Optional[str] = None) -> _CollectionState:
        cname = collection_name or self._collection_name(doc_type)
        if cname in self._states:
            return self._states[cname]
        vs = Chroma(
            collection_name=cname,
            embedding_function=self.embedder,
            persist_directory=self.persist_directory,
        )
        state = _CollectionState(vectorstore=vs)
        self._states[cname] = state
        LOGGER.info("ChromaRecordStore: opened/created collection '%s'.", cname)
        return state

    def load_collection(self, doc_type: str, collection_name: Optional[str] = None) -> _CollectionState:
        state = self.get_or_create_collection(doc_type=doc_type, collection_name=collection_name)
        if state.bm25 is None:
            LOGGER.warning(
                "Collection '%s' loaded from disk — BM25 is unavailable until re-ingest.",
                collection_name or self._collection_name(doc_type),
            )
        return state

    def _get_or_build_bm25(self, state: _CollectionState) -> Optional[BM25Retriever]:
        if state.bm25 is not None:
            return state.bm25
        if not state.bm25_docs:
            return None
        state.bm25 = BM25Retriever.from_documents(state.bm25_docs, k=self.bm25_k)
        return state.bm25

    def upsert_records(
        self,
        records: Iterable[Dict[str, Any]],
        *,
        doc_type: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> int:
        records = list(records)
        if not records:
            return 0

        inferred_doc_type = (doc_type or str(records[0].get("doc_type") or "OTHER")).strip().upper()
        bad_doc_types: List[str] = []
        for rec in records:
            rec_dt = str(rec.get("doc_type") or inferred_doc_type).strip().upper()
            if rec_dt != inferred_doc_type:
                bad_doc_types.append(str(rec.get("record_id") or "UNKNOWN"))
        if bad_doc_types:
            raise ValueError(
                f"upsert_records received mixed doc_type batch for collection doc_type={inferred_doc_type}. "
                f"Offending record_ids={bad_doc_types[:10]}"
            )

        state = self.get_or_create_collection(doc_type=inferred_doc_type, collection_name=collection_name)
        vs = state.vectorstore

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        bm25_docs: List[Document] = []

        for rec in records:
            if not _looks_like_processed_text_record(rec):
                LOGGER.warning("Skipping malformed processed_text_record during Chroma upsert: keys=%s", sorted(rec.keys()))
                continue
            payload = to_chroma_payload(rec)
            if not payload["document"]:
                continue
            ids.append(payload["id"])
            docs.append(payload["document"])
            metas.append(payload["metadata"])
            bm25_docs.append(Document(page_content=payload["document"], metadata=payload["metadata"]))

        if not ids:
            return 0

        embeddings = self.embedder.embed_documents(docs)
        vs._collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

        # Keep a fresh BM25 corpus for collections ingested in-process.
        existing_by_id = {d.metadata.get("record_id"): d for d in state.bm25_docs}
        for d in bm25_docs:
            rid = d.metadata.get("record_id")
            if rid:
                existing_by_id[rid] = d
        state.bm25_docs = list(existing_by_id.values())
        state.bm25 = None

        LOGGER.info("ChromaRecordStore: upserted %d records into '%s'.", len(ids), vs._collection.name)
        return len(ids)

    def upsert_jsonl(
        self,
        jsonl_path: str,
        *,
        doc_type_override: Optional[str] = None,
    ) -> Dict[str, int]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for obj in _iter_jsonl(jsonl_path):
            rec = extract_processed_text_record(obj)
            if not rec:
                continue
            doc_type = (doc_type_override or rec.get("doc_type") or "OTHER").strip().upper()
            grouped.setdefault(doc_type, []).append(rec)

        counts: Dict[str, int] = {}
        for doc_type, records in grouped.items():
            counts[doc_type] = self.upsert_records(records, doc_type=doc_type)
        return counts

    def query_doc_type(
        self,
        doc_type: str,
        query_text: str,
        *,
        top_k: int = 8,
        filter_meta: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        hybrid_weight: float = 0.5,
    ) -> List[Document]:
        cname = collection_name or self._collection_name(doc_type)
        if cname not in self._states:
            raise ValueError(f"Collection '{cname}' is not initialised. Call upsert_jsonl() or load_collection() first.")

        state = self._states[cname]
        vs = state.vectorstore
        filter_sane = {k: v for k, v in (filter_meta or {}).items() if v is not None}

        if not filter_sane:
            chroma_where = None
        elif len(filter_sane) == 1:
            k, v = next(iter(filter_sane.items()))
            chroma_where = {k: {"$eq": v}}
        else:
            chroma_where = {"$and": [{k: {"$eq": v}} for k, v in filter_sane.items()]}

        dense_hits: List[Dict[str, Any]] = []
        fetch_k = top_k * 2
        try:
            scored = vs.similarity_search_with_score(query_text, k=fetch_k, filter=chroma_where)
            for doc, score in scored:
                doc.metadata = dict(doc.metadata or {})
                doc.metadata["_score"] = float(score)
                rid = _stable_record_id_from_doc(doc)
                if not rid:
                    LOGGER.warning("Dense retrieval returned hit with no stable record_id; skipping.")
                    continue
                dense_hits.append({
                    "record_id": rid,
                    "score": float(score),
                    "document": doc,
                    "metadata": doc.metadata,
                })
        except Exception as exc:
            LOGGER.warning("Dense retrieval failed for doc_type '%s': %s", doc_type, exc)

        bm25_hits: List[Dict[str, Any]] = []
        bm25 = self._get_or_build_bm25(state)
        if bm25 is not None:
            try:
                bm25.k = fetch_k
                docs = bm25.invoke(query_text)
                if filter_sane:
                    docs = [d for d in docs if all(d.metadata.get(k) == v for k, v in filter_sane.items())]
                for doc in docs:
                    rid = _stable_record_id_from_doc(doc)
                    if not rid:
                        LOGGER.warning("BM25 retrieval returned hit with no stable record_id; skipping.")
                        continue
                    bm25_hits.append({
                        "record_id": rid,
                        "score": None,
                        "document": doc,
                        "metadata": doc.metadata,
                    })
            except Exception as exc:
                LOGGER.warning("BM25 retrieval failed for doc_type '%s': %s", doc_type, exc)

        fused = _reciprocal_rank_fusion(
            {"dense": dense_hits, "bm25": bm25_hits},
            k=top_k,
            view_weights={"dense": hybrid_weight, "bm25": 1.0 - hybrid_weight},
            key_field="record_id",
        )

        out: List[Document] = []
        for entry in fused:
            doc = entry.get("document")
            if doc is None:
                continue
            doc.metadata = dict(doc.metadata or {})
            doc.metadata["_score"] = entry["score"]
            out.append(doc)
        return out


# ---------------------------------------------------------------------------
# Small local fusion helper (keeps chroma_store self-contained)
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    per_view: Dict[str, List[Dict[str, Any]]],
    *,
    k: int,
    view_weights: Optional[Dict[str, float]] = None,
    key_field: str = "record_id",
) -> List[Dict[str, Any]]:
    view_weights = view_weights or {}
    acc: Dict[str, Dict[str, Any]] = {}
    for view_name, hits in per_view.items():
        weight = float(view_weights.get(view_name, 1.0))
        for rank, hit in enumerate(hits, start=1):
            hid = str(hit.get(key_field) or "")
            if not hid:
                continue
            score = weight * (1.0 / (60 + rank))
            if hid not in acc:
                acc[hid] = {
                    key_field: hid,
                    "score": 0.0,
                    "document": hit.get("document"),
                    "metadata": hit.get("metadata") or {},
                    "views": {},
                }
            acc[hid]["score"] += score
            acc[hid]["views"][view_name] = {"rank": rank, "raw_score": hit.get("score")}
            if acc[hid].get("document") is None:
                acc[hid]["document"] = hit.get("document")
    ranked = sorted(acc.values(), key=lambda x: x.get("score", 0.0), reverse=True)
    return ranked[:k]
