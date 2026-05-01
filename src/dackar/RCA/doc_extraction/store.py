from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .schema import ConfidenceLevel, DocExtractionRecord

logger = logging.getLogger(__name__)

# Stable Chroma record ID format: "{doc_id}::chain::{chain_index}"
_ID_SEP = "::chain::"


def _make_record_id(doc_id: str, chain_index: int) -> str:
    return f"{doc_id}{_ID_SEP}{chain_index}"


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


@dataclass
class SemanticMatch:
    """One deduplicated result from a DocExtractionStore.query() call."""
    record_id: str
    doc_id: str
    chain_index: int
    identified_effect: Optional[str]
    assessed_cause: Optional[str]
    inferred_fm_label: Optional[str]
    fm_id_candidate: Optional[str]
    confidence: ConfidenceLevel
    cause_is_symptom: bool
    similarity_score: float  # cosine similarity [0, 1]; higher = more similar
    fm_resolution_status: Optional[str] = None  # FMResolutionStatus value; None = not yet resolved
    # Epistemic annotation fields — populated by EpistemicClassifier when a
    # classifier is attached to DocExtractionStore (Phase A).
    doc_type: str = ""
    finding_status: Optional[str] = None
    authority_level: Optional[str] = None
    epistemic_class: Optional[str] = None
    classification_resolution_level: Optional[str] = None
    degraded_classification: bool = False

    @property
    def confidence_weight(self) -> float:
        return {ConfidenceLevel.HIGH: 1.0, ConfidenceLevel.MEDIUM: 0.7, ConfidenceLevel.LOW: 0.3}[self.confidence]

    @property
    def cause_is_symptom_factor(self) -> float:
        return 0.5 if self.cause_is_symptom else 1.0

    @property
    def semantic_contribution(self) -> float:
        """Fractional recurrence contribution for effective_recurrence_count (§4.3)."""
        return self.similarity_score * self.confidence_weight * self.cause_is_symptom_factor


class EmbeddingModelVersionError(RuntimeError):
    """Raised when the query-time embedding model does not match the collection's stored model."""


class DocExtractionStore:
    """Chroma-backed store for DocExtractionRecord objects.

    One collection (``"doc_extractions"``) stores all extraction records across
    all document types.  Each record's embed_text (§4.1) is computed and stored
    at upsert time; similarity queries run against these pre-computed vectors.

    Key guarantees:
    - Embedding model version is written into every record's metadata.
      A mismatch between query-time model and stored model raises EmbeddingModelVersionError.
    - fm_id_candidate is always null at ingestion; resolve_fm_candidates() writes it
      back in batch at the start of an RCA run (§3.3 Step C).
    - query() deduplicates by doc_id, returning only the best-scoring chain per document.
    """

    COLLECTION_NAME = "doc_extractions"
    # Cosine distance is used for the collection (hnsw:space = cosine).
    # Chroma returns cosine *distance* = 1 - cosine_similarity, so scores are inverted below.
    _HNSW_SPACE = "cosine"

    def __init__(
        self,
        persist_directory: str,
        embed_model: str = "nomic-embed-text-v1.5",
        ollama_base_url: Optional[str] = None,
        fm_resolution_threshold: float = 0.88,
        epistemics_classifier: Optional[Any] = None,
    ) -> None:
        self.persist_directory = persist_directory
        self.embed_model = embed_model
        self.ollama_base_url = ollama_base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.fm_resolution_threshold = fm_resolution_threshold  # 0.88 = auto_resolved boundary
        self.epistemics_classifier = epistemics_classifier  # EpistemicClassifier | None
        self._collection = None  # lazy-initialized on first use

    @property
    def embedding_model_version(self) -> str:
        return self.embed_model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embedder(self):
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(base_url=self.ollama_base_url, model=self.embed_model)

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        from langchain_chroma import Chroma
        os.makedirs(self.persist_directory, exist_ok=True)
        vs = Chroma(
            collection_name=self.COLLECTION_NAME,
            embedding_function=self._get_embedder(),
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": self._HNSW_SPACE},
        )
        self._collection = vs
        return vs

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._get_embedder().embed_documents(texts)

    def _embed_query(self, text: str) -> List[float]:
        return self._get_embedder().embed_query(text)

    def _chroma_collection(self):
        return self._get_collection()._collection

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert(self, record: DocExtractionRecord) -> str:
        """Embed and store one extraction record. Returns the Chroma record_id.

        Records with no embed_text are stored with a single-space document to avoid
        Chroma rejecting empty strings; they are retrievable by metadata but will not
        surface in similarity queries.
        """
        record_id = _make_record_id(record.doc_id, record.chain_index)
        embed_text = record.embed_text() or " "

        meta = record.as_chroma_metadata()
        # Overwrite embedding_model_version with the store's current model
        meta["embedding_model_version"] = self.embed_model

        embedding = self._embed_texts([embed_text])[0]
        self._chroma_collection().upsert(
            ids=[record_id],
            documents=[embed_text],
            metadatas=[meta],
            embeddings=[embedding],
        )

        # Write embedding_model_version back to the record for caller's reference
        record.embedding_model_version = self.embed_model
        logger.debug("DocExtractionStore: upserted %s", record_id)
        return record_id

    def upsert_batch(self, records: List[DocExtractionRecord]) -> int:
        """Embed and store multiple extraction records in one batch call."""
        if not records:
            return 0

        record_ids = [_make_record_id(r.doc_id, r.chain_index) for r in records]
        embed_texts = [r.embed_text() or " " for r in records]
        metadatas = []
        for r in records:
            meta = r.as_chroma_metadata()
            meta["embedding_model_version"] = self.embed_model
            metadatas.append(meta)

        embeddings = self._embed_texts(embed_texts)
        self._chroma_collection().upsert(
            ids=record_ids,
            documents=embed_texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        for r in records:
            r.embedding_model_version = self.embed_model

        logger.info("DocExtractionStore: upserted batch of %d records.", len(records))
        return len(records)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.75,
        near_match_window: float = 0.10,
        filter_meta: Optional[Dict[str, Any]] = None,
        exact_doc_ids: Optional[set] = None,
    ) -> Tuple[List[SemanticMatch], List[SemanticMatch]]:
        """Query for semantically similar extraction records.

        The embedding model used at query time must match the collection's stored
        ``embedding_model_version``.  A mismatch raises EmbeddingModelVersionError.

        Deduplication: only the highest-scoring chain per doc_id is returned.

        Args:
            query_text: The query string (e.g. ``fm.name | fm.expected_symptoms | event.symptom_description``).
            top_k: Maximum number of doc_id-deduplicated results to return.
            similarity_threshold: Minimum cosine similarity for inclusion in the main result set.
            near_match_window: Width of the soft zone below threshold that populates near_matches.
            filter_meta: Optional Chroma metadata pre-filter (e.g. ``{"doc_type": "CR"}``).
            exact_doc_ids: Set of doc_ids already counted via exact-match recurrence (kg_context.past_events).
                           Matching records are excluded from both matches and near_matches to prevent
                           double-counting.  None or empty set disables the guard.

        Returns:
            (matches, near_matches) where:
              - matches: similarity >= similarity_threshold, deduplicated, top_k max, exact_doc_ids excluded
              - near_matches: similarity in [similarity_threshold - near_match_window, similarity_threshold)
        """
        self._assert_model_version()

        collection = self._chroma_collection()
        fetch_n = top_k * 4  # over-fetch to account for deduplication

        chroma_where = _build_where_clause(filter_meta) if filter_meta else None

        query_embedding = self._embed_query(query_text)

        try:
            kwargs: Dict[str, Any] = {"query_embeddings": [query_embedding], "n_results": fetch_n}
            if chroma_where:
                kwargs["where"] = chroma_where
            raw = collection.query(**kwargs)
        except Exception as exc:
            logger.error("DocExtractionStore.query failed: %s", exc)
            return [], []

        ids = (raw.get("ids") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]

        # Chroma cosine distance = 1 - cosine_similarity → convert
        candidates: List[Tuple[float, SemanticMatch]] = []
        for record_id, distance, meta in zip(ids, distances, metadatas):
            if meta is None:
                continue
            sim = max(0.0, min(1.0, 1.0 - float(distance)))
            match = _meta_to_semantic_match(record_id, sim, meta)
            if match is not None:
                candidates.append((sim, match))

        # Deduplicate by doc_id: keep best similarity per document;
        # exclude doc_ids already counted via exact-match recurrence (double-counting guard).
        _exact = exact_doc_ids or set()
        best_per_doc: Dict[str, Tuple[float, SemanticMatch]] = {}
        for sim, match in candidates:
            if match.doc_id in _exact:
                continue
            existing = best_per_doc.get(match.doc_id)
            if existing is None or sim > existing[0]:
                best_per_doc[match.doc_id] = (sim, match)

        sorted_unique = sorted(best_per_doc.values(), key=lambda t: t[0], reverse=True)

        matches: List[SemanticMatch] = []
        near_matches: List[SemanticMatch] = []
        near_lower = similarity_threshold - near_match_window

        for sim, match in sorted_unique:
            if sim >= similarity_threshold:
                if len(matches) < top_k:
                    matches.append(match)
            elif sim >= near_lower:
                near_matches.append(match)

        # Apply epistemic classifier when present
        if self.epistemics_classifier is not None:
            for m in matches:
                self.epistemics_classifier.annotate_record(m)
            for m in near_matches:
                self.epistemics_classifier.annotate_record(m)

        return matches, near_matches

    def _assert_model_version(self) -> None:
        """Raise EmbeddingModelVersionError if stored records use a different model."""
        try:
            collection = self._chroma_collection()
            sample = collection.get(limit=1, include=["metadatas"])
            metas = sample.get("metadatas") or []
            if metas:
                stored_model = (metas[0] or {}).get("embedding_model_version", "")
                if stored_model and stored_model != self.embed_model:
                    raise EmbeddingModelVersionError(
                        f"Embedding model mismatch: query uses '{self.embed_model}' "
                        f"but collection was embedded with '{stored_model}'. "
                        f"Re-embed the collection before querying."
                    )
        except EmbeddingModelVersionError:
            raise
        except Exception:
            pass  # empty collection or Chroma unavailable — let the query attempt proceed

    # ------------------------------------------------------------------
    # fm_id_candidate batch resolution (§3.3 Step C)
    # ------------------------------------------------------------------

    def resolve_fm_candidates(
        self,
        fm_list: List[Tuple[str, str]],
        *,
        resolution_threshold: Optional[float] = None,
    ) -> int:
        """Resolve fm_id_candidate for unresolved extraction records.

        Called once per RCA run before Step 3 / Step 2d.  For each record with
        fm_id_candidate == "" (unresolved), embeds the stored inferred_fm_label
        and compares against the KG FM list for the current asset neighborhood.
        Writes fm_id_candidate (and fm_id_candidate_alt) back to the collection
        if cosine similarity >= resolution_threshold.

        Args:
            fm_list: List of (fm_id, fm_label) tuples from the KG.
            resolution_threshold: Override default; defaults to self.fm_resolution_threshold.

        Returns:
            Number of records updated.
        """
        if not fm_list:
            return 0

        threshold = resolution_threshold if resolution_threshold is not None else self.fm_resolution_threshold

        collection = self._chroma_collection()

        # Fetch all unresolved records (fm_id_candidate stored as "")
        try:
            result = collection.get(
                where={"fm_id_candidate": {"$eq": ""}},
                include=["metadatas"],
            )
        except Exception as exc:
            logger.warning("resolve_fm_candidates: failed to fetch unresolved records: %s", exc)
            return 0

        ids = result.get("ids") or []
        metas = result.get("metadatas") or []
        if not ids:
            logger.debug("resolve_fm_candidates: no unresolved records found.")
            return 0

        # Group record IDs by inferred_fm_label to avoid redundant embedding calls
        label_to_ids: Dict[str, List[str]] = {}
        for record_id, meta in zip(ids, metas):
            label = (meta or {}).get("inferred_fm_label", "")
            if not label:
                continue
            label_to_ids.setdefault(label, []).append(record_id)

        if not label_to_ids:
            return 0

        # Embed all unique labels and all KG FM labels
        unique_labels = list(label_to_ids.keys())
        fm_ids = [fm[0] for fm in fm_list]
        fm_labels = [fm[1] for fm in fm_list]

        logger.info(
            "resolve_fm_candidates: resolving %d unique labels against %d KG FMs.",
            len(unique_labels), len(fm_labels),
        )

        label_embeddings = self._embed_texts(unique_labels)
        fm_embeddings = self._embed_texts(fm_labels)

        updated_ids: List[str] = []
        updated_metas: List[Dict[str, Any]] = []
        resolved_count: int = 0  # only auto_resolved + ambiguous (non-empty fm_id_candidate)

        # Ambiguity boundary: [ambiguity_floor, threshold) → "ambiguous"; < ambiguity_floor → "unresolved"
        _AMBIGUITY_FLOOR = 0.80

        for label, label_emb in zip(unique_labels, label_embeddings):
            sims = [_cosine_similarity(label_emb, fm_emb) for fm_emb in fm_embeddings]
            sorted_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)

            best_idx = sorted_indices[0]
            best_sim = sims[best_idx]

            # Determine resolution status per three-tier rule (§4.10)
            if best_sim >= threshold:
                resolution_status = "auto_resolved"
                best_fm_id = fm_ids[best_idx]
                alt_fm_id = ""
                if len(sorted_indices) > 1:
                    alt_idx = sorted_indices[1]
                    if sims[alt_idx] >= _AMBIGUITY_FLOOR:
                        alt_fm_id = fm_ids[alt_idx]
            elif best_sim >= _AMBIGUITY_FLOOR:
                resolution_status = "ambiguous"
                best_fm_id = fm_ids[best_idx]
                alt_fm_id = ""
                if len(sorted_indices) > 1:
                    alt_idx = sorted_indices[1]
                    if sims[alt_idx] >= _AMBIGUITY_FLOOR:
                        alt_fm_id = fm_ids[alt_idx]
            else:
                resolution_status = "unresolved"
                best_fm_id = ""  # fm_id_candidate remains None/empty per spec
                alt_fm_id = ""

            for record_id in label_to_ids[label]:
                updated_ids.append(record_id)
                updated_metas.append({
                    "fm_id_candidate": best_fm_id,
                    "fm_id_candidate_alt": alt_fm_id,
                    "fm_resolution_status": resolution_status,
                    "fm_resolution_score": round(best_sim, 6),
                })
                if best_fm_id:  # non-empty: auto_resolved or ambiguous
                    resolved_count += 1

        if updated_ids:
            collection.update(ids=updated_ids, metadatas=updated_metas)
            logger.info("resolve_fm_candidates: resolved %d records.", resolved_count)

        return resolved_count

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return total number of extraction records in the collection."""
        try:
            return self._chroma_collection().count()
        except Exception:
            return 0

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all extraction records for a given source document (re-ingestion path)."""
        try:
            self._chroma_collection().delete(where={"doc_id": {"$eq": doc_id}})
            logger.info("DocExtractionStore: deleted records for doc_id '%s'.", doc_id)
        except Exception as exc:
            logger.warning("DocExtractionStore.delete_by_doc_id failed for '%s': %s", doc_id, exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_where_clause(filter_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    clauses = []
    for k, v in filter_meta.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple, set)):
            vals = [x for x in v if x is not None]
            if vals:
                clauses.append({k: {"$in": list(vals)}})
        else:
            clauses.append({k: {"$eq": v}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _meta_to_semantic_match(
    record_id: str,
    similarity: float,
    meta: Dict[str, Any],
) -> Optional[SemanticMatch]:
    doc_id = meta.get("doc_id", "")
    chain_index = int(meta.get("chain_index", 0))
    confidence_raw = meta.get("confidence", "low")
    try:
        confidence = ConfidenceLevel(confidence_raw)
    except ValueError:
        confidence = ConfidenceLevel.LOW
    fm_resolution_status = meta.get("fm_resolution_status") or None
    return SemanticMatch(
        record_id=record_id,
        doc_id=doc_id,
        chain_index=chain_index,
        identified_effect=meta.get("identified_effect") or None,
        assessed_cause=meta.get("assessed_cause") or None,
        inferred_fm_label=meta.get("inferred_fm_label") or None,
        fm_id_candidate=meta.get("fm_id_candidate") or None,
        confidence=confidence,
        cause_is_symptom=bool(meta.get("cause_is_symptom", False)),
        similarity_score=similarity,
        fm_resolution_status=fm_resolution_status,
        # Epistemic fields — pass through from stored Chroma metadata when present
        doc_type=meta.get("doc_type") or "",
        finding_status=meta.get("finding_status") or None,
        authority_level=meta.get("authority_level") or None,
        epistemic_class=meta.get("epistemic_class") or None,
        classification_resolution_level=meta.get("classification_resolution_level") or None,
        degraded_classification=bool(meta.get("degraded_classification", False)),
    )
