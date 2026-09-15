"""Unit tests for DocExtractionStore (Phase 2).

All tests use a pure-Python mock Chroma collection and a deterministic mock
embedder, so no Ollama, langchain-chroma, or chromadb installation is required.
"""
from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub out Chroma / LangChain before any store import touches them
# ---------------------------------------------------------------------------
for _mod in ("langchain_chroma", "langchain_community", "langchain_community.embeddings",
             "langchain_core", "langchain_core.documents"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from dackar.RCA.doc_extraction.schema import ConfidenceLevel, DocExtractionRecord
from dackar.RCA.doc_extraction.store import (
    DocExtractionStore,
    EmbeddingModelVersionError,
    SemanticMatch,
    _build_where_clause,
    _cosine_similarity,
    _make_record_id,
    _meta_to_semantic_match,
)


# ---------------------------------------------------------------------------
# Deterministic test vectors
# ---------------------------------------------------------------------------

DIM = 8


def _unit_vec(direction: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in direction))
    return [x / norm for x in direction] if norm else direction


_VEC_BEARING   = _unit_vec([1, 0, 0, 0, 0, 0, 0, 0])
_VEC_CORROSION = _unit_vec([0.9, 0.44, 0, 0, 0, 0, 0, 0])   # cos(bearing) ≈ 0.90
_VEC_PUMP_TRIP = _unit_vec([0, 0, 1, 0, 0, 0, 0, 0])          # orthogonal
_VEC_VALVE     = _unit_vec([0, 0, 0, 1, 0, 0, 0, 0])           # orthogonal
_VEC_NULL      = _unit_vec([0, 0, 0, 0, 0, 1, 0, 0])
_VEC_UNKNOWN   = _unit_vec([0, 0, 0, 0, 0, 0, 1, 0])

def _keyword_vec(text: str) -> List[float]:
    """Return a deterministic vector based on keywords in text, order-insensitive."""
    t = text.lower()
    if "bearing" in t:
        return _VEC_BEARING
    if "corrosion" in t or "chloride" in t or "wall thinning" in t:
        return _VEC_CORROSION
    if "pump trip" in t or "overspeed" in t or "system shutdown" in t:
        return _VEC_PUMP_TRIP
    if "valve" in t:
        return _VEC_VALVE
    return _VEC_UNKNOWN


def _mock_embed(texts: List[str]) -> List[List[float]]:
    return [_keyword_vec(t) for t in texts]


def _mock_embed_query(text: str) -> List[float]:
    return _keyword_vec(text)


# ---------------------------------------------------------------------------
# Pure-Python mock Chroma collection
# ---------------------------------------------------------------------------

class _MockChromaCollection:
    """In-memory store that mimics the chromadb Collection API used by DocExtractionStore."""

    def __init__(self) -> None:
        self._ids: List[str] = []
        self._docs: List[str] = []
        self._metas: List[Dict[str, Any]] = []
        self._embeddings: List[List[float]] = []

    def _index(self, record_id: str) -> Optional[int]:
        try:
            return self._ids.index(record_id)
        except ValueError:
            return None

    def upsert(self, ids, documents, metadatas, embeddings=None) -> None:
        for i, (rid, doc, meta) in enumerate(zip(ids, documents, metadatas)):
            emb = embeddings[i] if embeddings else _VEC_UNKNOWN
            idx = self._index(rid)
            if idx is None:
                self._ids.append(rid)
                self._docs.append(doc)
                self._metas.append(dict(meta))
                self._embeddings.append(emb)
            else:
                self._docs[idx] = doc
                self._metas[idx] = dict(meta)
                self._embeddings[idx] = emb

    def add(self, ids, documents, metadatas, embeddings=None) -> None:
        self.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def update(self, ids, metadatas) -> None:
        for rid, meta_update in zip(ids, metadatas):
            idx = self._index(rid)
            if idx is not None:
                self._metas[idx].update(meta_update)

    def get(self, ids=None, where=None, limit=None, include=None) -> Dict[str, Any]:
        if ids is not None:
            indices = [self._index(rid) for rid in ids if self._index(rid) is not None]
        else:
            indices = list(range(len(self._ids)))

        if where is not None:
            indices = [i for i in indices if self._matches_where(self._metas[i], where)]

        if limit is not None:
            indices = indices[:limit]

        return {
            "ids": [self._ids[i] for i in indices],
            "documents": [self._docs[i] for i in indices],
            "metadatas": [self._metas[i] for i in indices],
        }

    def query(self, query_embeddings, n_results, where=None) -> Dict[str, Any]:
        q_emb = query_embeddings[0]
        candidates = list(range(len(self._ids)))
        if where is not None:
            candidates = [i for i in candidates if self._matches_where(self._metas[i], where)]

        # Cosine distance = 1 - cosine_similarity
        scored = sorted(
            candidates,
            key=lambda i: _cosine_similarity(q_emb, self._embeddings[i]),
            reverse=True,
        )[:n_results]

        distances = [1.0 - _cosine_similarity(q_emb, self._embeddings[i]) for i in scored]
        return {
            "ids": [[self._ids[i] for i in scored]],
            "distances": [distances],
            "metadatas": [[self._metas[i] for i in scored]],
        }

    def delete(self, where=None) -> None:
        if where is None:
            self._ids.clear()
            self._docs.clear()
            self._metas.clear()
            self._embeddings.clear()
            return
        keep = [i for i in range(len(self._ids)) if not self._matches_where(self._metas[i], where)]
        self._ids = [self._ids[i] for i in keep]
        self._docs = [self._docs[i] for i in keep]
        self._metas = [self._metas[i] for i in keep]
        self._embeddings = [self._embeddings[i] for i in keep]

    def count(self) -> int:
        return len(self._ids)

    @staticmethod
    def _matches_where(meta: Dict[str, Any], clause: Dict[str, Any]) -> bool:
        """Minimal Chroma where-clause evaluator for $eq / $in / $and."""
        if "$and" in clause:
            return all(_MockChromaCollection._matches_where(meta, c) for c in clause["$and"])
        for key, condition in clause.items():
            if key.startswith("$"):
                continue
            val = meta.get(key)
            if isinstance(condition, dict):
                op = next(iter(condition))
                operand = condition[op]
                if op == "$eq" and val != operand:
                    return False
                if op == "$in" and val not in operand:
                    return False
                if op == "$ne" and val == operand:
                    return False
            else:
                if val != condition:
                    return False
        return True


def _make_store() -> DocExtractionStore:
    """Build a DocExtractionStore with all external dependencies mocked."""
    store = DocExtractionStore(
        persist_directory="/tmp/test_doc_extraction",
        embed_model="test-model-v1",
    )
    mock_collection = _MockChromaCollection()
    store._embed_texts = _mock_embed
    store._embed_query = _mock_embed_query
    store._chroma_collection = MagicMock(return_value=mock_collection)
    return store


def _make_record(
    doc_id: str,
    chain_index: int = 0,
    identified_effect: str = "pump trip",
    assessed_cause: str = "bearing wear",
    inferred_fm_label: str = "bearing wear — lubrication starvation",
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    cause_is_symptom: bool = False,
    as_found: Optional[str] = "degraded",
    fm_id_candidate: Optional[str] = None,
) -> DocExtractionRecord:
    return DocExtractionRecord(
        doc_id=doc_id,
        chain_index=chain_index,
        identified_effect=identified_effect,
        assessed_cause=assessed_cause,
        inferred_fm_label=inferred_fm_label,
        fm_id_candidate=fm_id_candidate,
        fm_id_candidate_alt=None,
        confidence=confidence,
        cause_is_symptom=cause_is_symptom,
        as_found=as_found,
        as_left=None,
        procedural_deviation_score=0.0,
        extraction_version="ner-v1.0_gaz-v1_llm-none",
        embedding_model_version=None,
    )


# ---------------------------------------------------------------------------
# Tests — SemanticMatch computed properties
# ---------------------------------------------------------------------------

class TestSemanticMatchProperties:
    def test_confidence_weight_high(self):
        m = SemanticMatch(record_id="r", doc_id="CR-001", chain_index=0,
                          identified_effect=None, assessed_cause=None, inferred_fm_label=None,
                          fm_id_candidate=None, confidence=ConfidenceLevel.HIGH,
                          cause_is_symptom=False, similarity_score=0.85)
        assert m.confidence_weight == 1.0

    def test_confidence_weight_medium(self):
        m = SemanticMatch(record_id="r", doc_id="CR-001", chain_index=0,
                          identified_effect=None, assessed_cause=None, inferred_fm_label=None,
                          fm_id_candidate=None, confidence=ConfidenceLevel.MEDIUM,
                          cause_is_symptom=False, similarity_score=0.80)
        assert m.confidence_weight == 0.7

    def test_confidence_weight_low(self):
        m = SemanticMatch(record_id="r", doc_id="CR-001", chain_index=0,
                          identified_effect=None, assessed_cause=None, inferred_fm_label=None,
                          fm_id_candidate=None, confidence=ConfidenceLevel.LOW,
                          cause_is_symptom=False, similarity_score=0.80)
        assert m.confidence_weight == 0.3

    def test_cause_is_symptom_factor_true(self):
        m = SemanticMatch(record_id="r", doc_id="CR-001", chain_index=0,
                          identified_effect=None, assessed_cause=None, inferred_fm_label=None,
                          fm_id_candidate=None, confidence=ConfidenceLevel.HIGH,
                          cause_is_symptom=True, similarity_score=0.80)
        assert m.cause_is_symptom_factor == 0.5

    def test_cause_is_symptom_factor_false(self):
        m = SemanticMatch(record_id="r", doc_id="CR-001", chain_index=0,
                          identified_effect=None, assessed_cause=None, inferred_fm_label=None,
                          fm_id_candidate=None, confidence=ConfidenceLevel.HIGH,
                          cause_is_symptom=False, similarity_score=0.80)
        assert m.cause_is_symptom_factor == 1.0

    def test_semantic_contribution_formula(self):
        m = SemanticMatch(record_id="r", doc_id="CR-001", chain_index=0,
                          identified_effect=None, assessed_cause=None, inferred_fm_label=None,
                          fm_id_candidate=None, confidence=ConfidenceLevel.MEDIUM,
                          cause_is_symptom=True, similarity_score=0.80)
        # 0.80 * 0.7 * 0.5 = 0.28
        assert abs(m.semantic_contribution - 0.28) < 1e-6

    def test_semantic_contribution_high_no_symptom(self):
        m = SemanticMatch(record_id="r", doc_id="CR-001", chain_index=0,
                          identified_effect=None, assessed_cause=None, inferred_fm_label=None,
                          fm_id_candidate=None, confidence=ConfidenceLevel.HIGH,
                          cause_is_symptom=False, similarity_score=0.90)
        assert abs(m.semantic_contribution - 0.90) < 1e-6


# ---------------------------------------------------------------------------
# Tests — upsert and record_id
# ---------------------------------------------------------------------------

class TestUpsert:
    def test_upsert_sets_embedding_model_version_on_record(self):
        store = _make_store()
        record = _make_record("CR-001")
        assert record.embedding_model_version is None
        store.upsert(record)
        assert record.embedding_model_version == "test-model-v1"

    def test_upsert_stores_record_retrievable_by_id(self):
        store = _make_store()
        record = _make_record("CR-002")
        record_id = store.upsert(record)
        assert record_id == "CR-002::chain::0"
        result = store._chroma_collection().get(ids=[record_id])
        assert record_id in (result.get("ids") or [])

    def test_upsert_batch_returns_count(self):
        store = _make_store()
        records = [
            _make_record("CR-010", chain_index=0),
            _make_record("CR-010", chain_index=1),
            _make_record("CR-011", chain_index=0),
        ]
        count = store.upsert_batch(records)
        assert count == 3

    def test_upsert_batch_all_have_embedding_model_version(self):
        store = _make_store()
        records = [_make_record(f"CR-{i:03d}") for i in range(3)]
        store.upsert_batch(records)
        for r in records:
            assert r.embedding_model_version == "test-model-v1"

    def test_upsert_null_record_stored_without_error(self):
        store = _make_store()
        null_record = DocExtractionRecord(
            doc_id="CR-020", chain_index=0,
            identified_effect=None, assessed_cause=None, inferred_fm_label=None,
            fm_id_candidate=None, fm_id_candidate_alt=None,
            confidence=ConfidenceLevel.LOW, cause_is_symptom=False,
            as_found=None, as_left=None, procedural_deviation_score=0.0,
            extraction_version="ner-v1.0", embedding_model_version=None,
            needs_human_review=True,
        )
        record_id = store.upsert(null_record)
        assert record_id == "CR-020::chain::0"

    def test_upsert_overwrites_existing_record(self):
        store = _make_store()
        record = _make_record("CR-030")
        store.upsert(record)
        assert store.count() == 1
        # Upsert again (same id) — count should remain 1
        store.upsert(record)
        assert store.count() == 1


# ---------------------------------------------------------------------------
# Tests — query, deduplication, near_match
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_returns_matches_above_threshold(self):
        store = _make_store()
        # Insert a bearing-related record (high sim) and an unrelated one (low sim)
        store.upsert(_make_record(
            "CR-100", chain_index=0,
            identified_effect="pump trip",
            assessed_cause="bearing wear",
            inferred_fm_label="bearing wear — lubrication starvation",
        ))
        store.upsert(_make_record(
            "CR-101", chain_index=0,
            identified_effect="valve leak",
            assessed_cause="valve seal failure",
            inferred_fm_label="valve seal failure",
            confidence=ConfidenceLevel.MEDIUM,
        ))

        query = "bearing failure | low oil pressure | bearing degradation"
        matches, near_matches = store.query(query, similarity_threshold=0.80)

        # CR-100 should appear (bearing vectors are identical → sim=1.0)
        assert any(m.doc_id == "CR-100" for m in matches)

    def test_query_deduplicates_by_doc_id(self):
        store = _make_store()
        # Two chains for same document — both bearing vectors
        for chain_idx in range(2):
            store.upsert(_make_record("CR-200", chain_index=chain_idx))

        query = "bearing failure | low oil pressure | bearing degradation"
        matches, _ = store.query(query, top_k=10, similarity_threshold=0.50)

        doc_ids = [m.doc_id for m in matches]
        assert len(doc_ids) == len(set(doc_ids)), "Duplicate doc_id in matches"
        assert doc_ids.count("CR-200") == 1

    def test_query_near_match_populated_below_threshold(self):
        store = _make_store()
        # Insert a corrosion record (cos(bearing) ≈ 0.9)
        store.upsert(_make_record(
            "CR-300", chain_index=0,
            identified_effect="wall thinning",
            assessed_cause="corrosion",
            inferred_fm_label="pipe corrosion — chloride attack",
            confidence=ConfidenceLevel.MEDIUM,
        ))

        # Query with bearing vector, threshold=0.95, near_window=0.10
        # cos(bearing, corrosion) ≈ 0.90, which falls in [0.85, 0.95)
        query = "bearing failure | low oil pressure | bearing degradation"
        matches, near_matches = store.query(
            query,
            similarity_threshold=0.95,
            near_match_window=0.10,
        )

        # Should be in near_matches, not matches
        assert not any(m.doc_id == "CR-300" for m in matches)
        assert any(m.doc_id == "CR-300" for m in near_matches)

    def test_query_top_k_limits_results(self):
        store = _make_store()
        for i in range(10):
            store.upsert(_make_record(f"CR-{400+i}"))

        matches, _ = store.query(
            "bearing failure | low oil pressure | bearing degradation",
            top_k=3,
            similarity_threshold=0.50,
        )
        assert len(matches) <= 3

    def test_query_semantic_text_fields_returned_in_match(self):
        """identified_effect, assessed_cause, inferred_fm_label are stored in metadata
        and round-trip back through the query result."""
        store = _make_store()
        store.upsert(_make_record(
            "CR-450", chain_index=0,
            identified_effect="pump trip",
            assessed_cause="bearing wear",
            inferred_fm_label="bearing wear — lubrication starvation",
        ))
        matches, _ = store.query(
            "bearing failure | low oil pressure | bearing degradation",
            similarity_threshold=0.50,
        )
        assert any(m.doc_id == "CR-450" for m in matches)
        match = next(m for m in matches if m.doc_id == "CR-450")
        assert match.identified_effect == "pump trip"
        assert match.assessed_cause == "bearing wear"
        assert match.inferred_fm_label == "bearing wear — lubrication starvation"


# ---------------------------------------------------------------------------
# Tests — embedding model version guard
# ---------------------------------------------------------------------------

class TestModelVersionGuard:
    def test_raises_on_model_mismatch(self):
        store = _make_store()
        collection = store._chroma_collection()
        collection.add(
            ids=["CR-500::chain::0"],
            documents=["bearing wear | pump trip"],
            metadatas=[{"embedding_model_version": "other-model-v2", "doc_id": "CR-500",
                         "chain_index": 0, "confidence": "high", "cause_is_symptom": False,
                         "fm_id_candidate": "", "fm_id_candidate_alt": "",
                         "as_found": "degraded", "as_left": "",
                         "procedural_deviation_score": 0.0,
                         "extraction_version": "ner-v1.0", "needs_human_review": False}],
        )
        with pytest.raises(EmbeddingModelVersionError, match="mismatch"):
            store._assert_model_version()

    def test_no_error_when_model_matches(self):
        store = _make_store()
        collection = store._chroma_collection()
        collection.add(
            ids=["CR-501::chain::0"],
            documents=["bearing wear | pump trip"],
            metadatas=[{"embedding_model_version": "test-model-v1", "doc_id": "CR-501",
                         "chain_index": 0, "confidence": "high", "cause_is_symptom": False,
                         "fm_id_candidate": "", "fm_id_candidate_alt": "",
                         "as_found": "", "as_left": "", "procedural_deviation_score": 0.0,
                         "extraction_version": "ner-v1.0", "needs_human_review": False}],
        )
        store._assert_model_version()  # must not raise

    def test_no_error_on_empty_collection(self):
        store = _make_store()
        store._assert_model_version()  # must not raise


# ---------------------------------------------------------------------------
# Tests — fm_id_candidate batch resolution
# ---------------------------------------------------------------------------

class TestFmCandidateResolution:
    def _store_with_unresolved(self, inferred_label: str, embed_vec: List[float]) -> DocExtractionStore:
        store = _make_store()

        def custom_embed(texts):
            result = []
            for t in texts:
                # Return exact embed_vec for the label, keyword-based for everything else
                if t == inferred_label:
                    result.append(embed_vec)
                else:
                    result.append(_keyword_vec(t))
            return result

        store._embed_texts = custom_embed
        collection = store._chroma_collection()
        collection.add(
            ids=[f"CR-600::chain::0"],
            documents=[inferred_label],
            metadatas=[{
                "doc_id": "CR-600", "chain_index": 0,
                "fm_id_candidate": "",
                "fm_id_candidate_alt": "",
                "inferred_fm_label": inferred_label,
                "confidence": "high", "cause_is_symptom": False,
                "as_found": "degraded", "as_left": "",
                "procedural_deviation_score": 0.0,
                "extraction_version": "ner-v1.0",
                "embedding_model_version": "test-model-v1",
                "needs_human_review": False,
            }],
        )
        return store

    def test_resolves_matching_label_above_threshold(self):
        store = self._store_with_unresolved(
            "bearing wear — lubrication starvation", _VEC_BEARING
        )
        fm_list = [
            ("FM-BRG-001", "bearing wear — lubrication starvation"),
            ("FM-CORR-002", "pipe corrosion — chloride attack"),
        ]
        count = store.resolve_fm_candidates(fm_list, resolution_threshold=0.80)

        assert count == 1
        result = store._chroma_collection().get(ids=["CR-600::chain::0"])
        assert result["metadatas"][0]["fm_id_candidate"] == "FM-BRG-001"

    def test_does_not_resolve_below_threshold(self):
        store = self._store_with_unresolved("pump trip", _VEC_PUMP_TRIP)
        fm_list = [
            ("FM-BRG-001", "bearing wear — lubrication starvation"),
            ("FM-CORR-002", "pipe corrosion — chloride attack"),
        ]
        count = store.resolve_fm_candidates(fm_list, resolution_threshold=0.80)

        assert count == 0
        result = store._chroma_collection().get(ids=["CR-600::chain::0"])
        assert result["metadatas"][0]["fm_id_candidate"] == ""

    def test_empty_fm_list_returns_zero(self):
        store = _make_store()
        assert store.resolve_fm_candidates([]) == 0

    def test_alt_fm_id_set_when_second_also_above_threshold(self):
        store = self._store_with_unresolved(
            "bearing wear — lubrication starvation", _VEC_BEARING
        )
        # Both FM labels are close to the bearing vector in our test space
        fm_list = [
            ("FM-BRG-001", "bearing wear — lubrication starvation"),
            ("FM-CORR-002", "pipe corrosion — chloride attack"),
        ]
        # Use a low threshold so corrosion (cos~0.9) also qualifies
        count = store.resolve_fm_candidates(fm_list, resolution_threshold=0.75)
        assert count == 1

        result = store._chroma_collection().get(ids=["CR-600::chain::0"])
        meta = result["metadatas"][0]
        assert meta["fm_id_candidate"] == "FM-BRG-001"
        assert meta["fm_id_candidate_alt"] == "FM-CORR-002"


# ---------------------------------------------------------------------------
# Tests — count and delete
# ---------------------------------------------------------------------------

class TestCountAndDelete:
    def test_count_zero_on_empty_store(self):
        store = _make_store()
        assert store.count() == 0

    def test_count_increases_after_upsert(self):
        store = _make_store()
        store.upsert(_make_record("CR-700"))
        assert store.count() == 1
        store.upsert(_make_record("CR-701"))
        assert store.count() == 2

    def test_delete_by_doc_id_removes_records(self):
        store = _make_store()
        store.upsert(_make_record("CR-800", chain_index=0))
        store.upsert(_make_record("CR-800", chain_index=1))
        store.upsert(_make_record("CR-801", chain_index=0))
        assert store.count() == 3

        store.delete_by_doc_id("CR-800")
        assert store.count() == 1


# ---------------------------------------------------------------------------
# Tests — helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_make_record_id(self):
        assert _make_record_id("CR-001", 0) == "CR-001::chain::0"
        assert _make_record_id("WO-XYZ", 3) == "WO-XYZ::chain::3"

    def test_build_where_clause_single_field(self):
        assert _build_where_clause({"doc_type": "CR"}) == {"doc_type": {"$eq": "CR"}}

    def test_build_where_clause_multiple_fields(self):
        clause = _build_where_clause({"doc_type": "CR", "as_found": "degraded"})
        assert "$and" in clause
        assert len(clause["$and"]) == 2

    def test_build_where_clause_list_field(self):
        assert _build_where_clause({"doc_type": ["CR", "WO"]}) == {"doc_type": {"$in": ["CR", "WO"]}}

    def test_build_where_clause_none_value_skipped(self):
        clause = _build_where_clause({"doc_type": "CR", "as_found": None})
        assert "as_found" not in str(clause)

    def test_build_where_clause_empty_returns_none(self):
        assert _build_where_clause({}) is None

    def test_meta_to_semantic_match_valid(self):
        meta = {"doc_id": "CR-001", "chain_index": "2", "confidence": "medium",
                 "cause_is_symptom": True, "fm_id_candidate": "FM-001"}
        m = _meta_to_semantic_match("CR-001::chain::2", 0.82, meta)
        assert m is not None
        assert m.doc_id == "CR-001"
        assert m.chain_index == 2
        assert m.confidence == ConfidenceLevel.MEDIUM
        assert m.cause_is_symptom is True
        assert m.similarity_score == 0.82
        assert m.fm_id_candidate == "FM-001"

    def test_meta_to_semantic_match_bad_confidence_defaults_to_low(self):
        meta = {"doc_id": "CR-001", "chain_index": 0, "confidence": "garbage", "cause_is_symptom": False}
        m = _meta_to_semantic_match("CR-001::chain::0", 0.75, meta)
        assert m.confidence == ConfidenceLevel.LOW

    def test_meta_to_semantic_match_empty_fm_id_candidate_is_none(self):
        meta = {"doc_id": "CR-001", "chain_index": 0, "confidence": "high",
                "cause_is_symptom": False, "fm_id_candidate": ""}
        m = _meta_to_semantic_match("CR-001::chain::0", 0.80, meta)
        assert m.fm_id_candidate is None

    def test_cosine_similarity_identical(self):
        v = _unit_vec([1, 2, 3, 4, 5, 6, 7, 8])
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        assert abs(_cosine_similarity([1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0])) < 1e-6

    def test_cosine_similarity_near_similar(self):
        assert _cosine_similarity(_VEC_BEARING, _VEC_CORROSION) > 0.85

    def test_cosine_similarity_zero_vector(self):
        assert _cosine_similarity([0] * 8, _VEC_BEARING) == 0.0
