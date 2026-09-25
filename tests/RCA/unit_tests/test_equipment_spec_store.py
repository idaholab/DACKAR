"""
Unit tests for equipment_similarity.equipment_spec_store.EquipmentSpecStore.

Locks the I2 fix: find_similar() opens the persisted collection via
load_collection() before querying, so a disk-backed store queried from a fresh
process (which has performed no upsert and so has no in-memory collection state)
still resolves Tier 3 instead of raising ValueError and silently returning [].
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from equipment_similarity.equipment_spec_store import (
    EQUIPMENT_SPECS_DOC_TYPE,
    EquipmentSpecStore,
)


class _DiskBackedChromaStub:
    """Mimics ChromaRecordStore for a collection that exists on disk but has not
    been touched in this process: query_doc_type() raises ValueError until
    load_collection() has registered it — exactly the fresh-process failure mode
    the I2 fix addresses."""

    def __init__(self, hit_component_id: str) -> None:
        self._loaded = False
        self._hit = hit_component_id
        self.calls = []

    def load_collection(self, doc_type, collection_name=None):
        self.calls.append(("load", doc_type))
        self._loaded = True

    def query_doc_type(self, doc_type, query_text, top_k=10):
        self.calls.append(("query", doc_type))
        if not self._loaded:
            raise ValueError("Call upsert_jsonl() or load_collection() first")
        doc = MagicMock()
        doc.metadata = {"component_id": self._hit, "_vector_score": 0.1}
        return [doc]


def test_find_similar_opens_persisted_collection_before_query():
    store = _DiskBackedChromaStub(hit_component_id="C-500")
    es = EquipmentSpecStore(store)

    hits = es.find_similar("main feed pump", top_k=5)

    # Tier 3 resolves: load_collection() ran first, so the disk-backed query
    # succeeds instead of raising ValueError and yielding an empty result.
    assert [h.metadata["component_id"] for h in hits] == ["C-500"]
    # ...and it happened in the right order.
    assert store.calls == [
        ("load", EQUIPMENT_SPECS_DOC_TYPE),
        ("query", EQUIPMENT_SPECS_DOC_TYPE),
    ]


def test_find_similar_graceful_when_collection_absent():
    """If the collection genuinely cannot be opened, ValueError degrades to an
    empty result rather than propagating."""

    class _Absent:
        def load_collection(self, doc_type, collection_name=None):
            raise ValueError("no such collection")

        def query_doc_type(self, *args, **kwargs):
            raise AssertionError("query must not run when the collection is absent")

    es = EquipmentSpecStore(_Absent())
    assert es.find_similar("q") == []
