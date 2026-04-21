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
    "doc_refs",
    "alarm_ids",
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

def _normalize_filter_meta(filter_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize high-level retrieval filters into the scalar metadata keys that
    are actually stored in Chroma.

    Examples:
      - doc_ids   -> doc_id
      - doc_types -> doc_type
      - component_ids is handled via post-filtering (not passable to Chroma where clause)
    """
    raw = dict(filter_meta or {})
    norm: Dict[str, Any] = {}

    alias_map = {
        "doc_ids": "doc_id",
        "doc_types": "doc_type",
    }

    for key, value in raw.items():
        if value is None:
            continue

        target_key = alias_map.get(key, key)

        # component_ids is a list field stored as a JSON string in Chroma metadata.
        # Chroma's where clause cannot query inside a JSON-encoded list, so we skip
        # it here and apply it as a Python post-filter in query_doc_type instead.
        if target_key == "component_ids":
            continue

        # Drop empty containers
        if isinstance(value, (list, tuple, set)) and not value:
            continue

        norm[target_key] = value

    return norm


def _doc_matches_component_ids(doc: Document, wanted: set) -> bool:
    """Return True if *doc* is associated with at least one of the *wanted* component IDs.

    Checks:
      1. The scalar ``component_id`` metadata field (populated for single-component records).
      2. The ``component_ids`` field, which may be a JSON-encoded list (how Chroma stores
         list-valued metadata) or a plain Python list (BM25 in-memory path).
    """
    meta = doc.metadata or {}
    scalar = meta.get("component_id")
    if scalar and scalar in wanted:
        return True
    raw = meta.get("component_ids")
    if isinstance(raw, str):
        try:
            ids = json.loads(raw)
            if isinstance(ids, list) and any(c in wanted for c in ids):
                return True
        except (json.JSONDecodeError, TypeError):
            pass
    elif isinstance(raw, (list, tuple)):
        if any(c in wanted for c in raw):
            return True
    return False

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

    # Flatten condition_assessment nested object into scalar keys so Chroma can
    # store and return them.  Source priority: record-level > metadata-level.
    # These are most meaningful on WO records but are preserved for all doc types.
    ca = record.get("condition_assessment") or metadata.get("condition_assessment")
    if isinstance(ca, dict):
        for field in ("as_found_condition", "as_left_condition", "as_found_text", "as_left_text"):
            val = ca.get(field)
            if val is not None:
                metadata.setdefault(f"ca_{field}", val)
        # Flatten measurements as a compact summary string (first out-of-spec item wins).
        measurements = ca.get("measurements")
        if isinstance(measurements, list) and measurements:
            oos = [m for m in measurements if isinstance(m, dict) and m.get("in_spec") is False]
            summary = oos[0] if oos else measurements[0]
            if isinstance(summary, dict):
                metadata.setdefault(
                    "ca_measurement_summary",
                    f"{summary.get('parameter','?')}={summary.get('value','?')}{summary.get('unit','')} "
                    f"({'OOS' if not summary.get('in_spec', True) else 'in-spec'})",
                )

    # Flatten ECA structured fields into scalar keys.
    # `finding_status` encodes the epistemic status of the document's causal claims:
    #   "confirmed"    — ECA with a formally validated root cause conclusion
    #   "preliminary"  — CR with an initial engineer assessment (not validated)
    #   "observational"— WO, SOP, MANUAL, SPEC, BULLETIN, or unknown — neither
    doc_type_upper = str(metadata.get("doc_type") or record.get("doc_type") or "").strip().upper()
    if doc_type_upper == "ECA":
        metadata.setdefault("finding_status", "confirmed")
    elif doc_type_upper == "CR":
        metadata.setdefault("finding_status", "preliminary")
    elif doc_type_upper == "OE":
        metadata.setdefault("finding_status", "fleet_experience")
    else:
        metadata.setdefault("finding_status", "observational")

    eca_block = record.get("eca") or {}
    if isinstance(eca_block, dict) and eca_block:
        eca_conf = eca_block.get("confidence")
        if isinstance(eca_conf, (int, float)):
            metadata.setdefault("eca_confidence", float(eca_conf))
        causal_factors = eca_block.get("causal_factors")
        if isinstance(causal_factors, list) and causal_factors:
            metadata.setdefault(
                "eca_causal_factors_text",
                " | ".join(str(f) for f in causal_factors if f),
            )
        rationale = eca_block.get("rationale")
        if rationale:
            metadata.setdefault("eca_rationale_excerpt", str(rationale)[:300])

    # Flatten OE metadata into scalar oe_* prefixed keys for Chroma pre-filtering.
    # Soft filter fields (plant_scope, applicable_system_types, applicable_component_types)
    # are stored as strings so evidence_retriever can use them as advisory hints.
    oe_block = record.get("oe_metadata") or {}
    if isinstance(oe_block, dict) and oe_block:
        for scalar_field in ("issuing_body", "oe_number", "plant_scope", "similarity_basis"):
            val = oe_block.get(scalar_field)
            if val is not None:
                metadata.setdefault(f"oe_{scalar_field}", str(val))
        for list_field in ("applicable_system_types", "applicable_component_types"):
            val = oe_block.get(list_field)
            if isinstance(val, list) and val:
                metadata.setdefault(f"oe_{list_field}", val)
                metadata.setdefault(f"oe_{list_field}_text", " | ".join(str(x) for x in val))

    # Flatten NER-enriched entities from the enrichment block.
    # These are populated by augment_chunks.build_processed_text_record via NERSeed.
    ner = (record.get("enrichment") or {}).get("extracted_entities") or {}
    if isinstance(ner, dict) and ner:
        # List fields — kept as lists; LIST_FIELDS loop will also add *_text companions.
        for list_key in ("doc_refs", "alarm_ids"):
            val = ner.get(list_key)
            if isinstance(val, list) and val:
                metadata.setdefault(list_key, val)

        # Temporal refs — raw strings, join for text search.
        temporal_refs = ner.get("temporal_refs")
        if isinstance(temporal_refs, list) and temporal_refs:
            metadata.setdefault("temporal_refs_text", " | ".join(str(r) for r in temporal_refs))

        # Measurements — compact "value unit" pairs.
        measurements = ner.get("measurements")
        if isinstance(measurements, list) and measurements:
            parts = []
            for m in measurements:
                if not isinstance(m, dict):
                    continue
                v = m.get("value", m.get("parameter", ""))
                u = m.get("unit", "")
                if v or u:
                    parts.append(f"{v}{u}".strip())
            if parts:
                metadata.setdefault("ner_measurements_text", " | ".join(parts))

        # Conjectures — hedge strings.
        conjectures = ner.get("conjectures")
        if isinstance(conjectures, list) and conjectures:
            metadata.setdefault("conjectures_text", " | ".join(str(c) for c in conjectures))

        # Locations — dict with text + sub_label.
        locations = ner.get("locations")
        if isinstance(locations, list) and locations:
            loc_parts = []
            for loc in locations:
                if isinstance(loc, dict):
                    loc_parts.append(str(loc.get("text") or loc.get("sub_label") or ""))
                elif isinstance(loc, str):
                    loc_parts.append(loc)
            if loc_parts:
                metadata.setdefault("locations_text", " | ".join(p for p in loc_parts if p))

        # Dominant temporal relation for pre-filter hints.
        temporal_relations = ner.get("temporal_relations")
        if isinstance(temporal_relations, list) and temporal_relations:
            from collections import Counter
            votes = Counter(
                r.get("sub_label") for r in temporal_relations
                if isinstance(r, dict) and r.get("sub_label")
            )
            if votes:
                metadata.setdefault("dominant_temporal_relation", votes.most_common(1)[0][0])

    # Flatten Stage 5 causal statements for BM25 / text search and confidence filtering.
    # Boolean flags (has_explicit_causal_statement etc.) are already in record["metadata"]
    # from build_processed_text_record; here we add the actual span text and max confidence.
    stage5 = (record.get("enrichment") or {}).get("stage5_causal_condition") or {}
    if isinstance(stage5, dict) and stage5:
        statements = stage5.get("extracted_causal_statements") or []
        if isinstance(statements, list) and statements:
            causal_parts = []
            max_conf = 0.0
            for stmt in statements[:8]:
                if not isinstance(stmt, dict):
                    continue
                c = str(stmt.get("cause_text") or "").strip()
                e = str(stmt.get("effect_text") or "").strip()
                k = str(stmt.get("connector") or "").strip()
                conf = float(stmt.get("confidence", 0.0))
                max_conf = max(max_conf, conf)
                line = " ".join(p for p in [c, k, e] if p)
                if line:
                    causal_parts.append(line)
            if causal_parts:
                metadata.setdefault("causal_statements_text", " | ".join(causal_parts))
            if max_conf > 0.0:
                metadata.setdefault("causal_confidence_max", round(max_conf, 3))

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
        filter_sane = _normalize_filter_meta(filter_meta)

        # component_ids cannot be passed to the Chroma where clause (list stored as JSON
        # string); extract it here for Python post-filtering after retrieval.
        wanted_component_ids: set = set((filter_meta or {}).get("component_ids") or [])

        if not filter_sane:
            chroma_where = None
        else:
            clauses = []
            for k, v in filter_sane.items():
                if isinstance(v, (list, tuple, set)):
                    vals = [x for x in v if x is not None]
                    if not vals:
                        continue
                    clauses.append({k: {"$in": list(vals)}})
                else:
                    clauses.append({k: {"$eq": v}})

            if not clauses:
                chroma_where = None
            elif len(clauses) == 1:
                chroma_where = clauses[0]
            else:
                chroma_where = {"$and": clauses}

        if wanted_component_ids:
            LOGGER.warning(
                "ChromaRecordStore: component_ids filter applied as Python post-filter "
                "on top-%d dense hits — not as a Chroma index-level filter.  "
                "Documents outside the top-%d similarity results are not scanned.  "
                "Increase top_k or re-ingest with component_id as a scalar metadata field "
                "to improve component-level retrieval precision.",
                top_k * 4,
                top_k * 4,
            )

        dense_hits: List[Dict[str, Any]] = []
        # Fetch extra candidates when component post-filtering is active so that
        # attrition from the filter does not starve downstream consumers.
        fetch_k = top_k * (4 if wanted_component_ids else 2)
        try:
            scored = vs.similarity_search_with_score(query_text, k=fetch_k, filter=chroma_where)
            for doc, score in scored:
                doc.metadata = dict(doc.metadata or {})
                doc.metadata["_score"] = float(score)
                # Preserve raw vector similarity before RRF fusion overwrites _score.
                # Downstream consumers (e.g. evidence assessment) use _vector_score
                # as a semantic relevance signal that survives RRF re-ranking.
                doc.metadata["_vector_score"] = float(score)
                rid = _stable_record_id_from_doc(doc)
                if not rid:
                    LOGGER.warning("Dense retrieval returned hit with no stable record_id; skipping.")
                    continue
                if wanted_component_ids and not _doc_matches_component_ids(doc, wanted_component_ids):
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
        bm25_available = bm25 is not None
        if bm25_available:
            try:
                bm25.k = fetch_k
                docs = bm25.invoke(query_text)
                if filter_sane or wanted_component_ids:
                    docs = [
                        d for d in docs
                        if all(d.metadata.get(k) == v for k, v in filter_sane.items())
                        and (not wanted_component_ids or _doc_matches_component_ids(d, wanted_component_ids))
                    ]
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
                bm25_available = False
                LOGGER.warning("BM25 retrieval failed for doc_type '%s': %s", doc_type, exc)
        else:
            LOGGER.warning(
                "BM25 unavailable for collection '%s' (loaded from disk without in-process ingest); "
                "falling back to dense-only retrieval.  hybrid_weight=%.2f has no effect.",
                cname,
                hybrid_weight,
            )

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
            doc.metadata["_bm25_available"] = bm25_available
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
