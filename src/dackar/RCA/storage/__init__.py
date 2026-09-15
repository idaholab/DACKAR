"""Stage-6 storage & retrieval for canonical processed_text_record objects.

Public API:
    ChromaRecordStore
        Per-doc-type Chroma store (dense + in-memory BM25) for processed_text_records.
    ProcessedRecordStore
        Corpus-level in-memory index for hydrating records by record_id / chunk_id.
    LCProcessedRetriever
        Cross-doc-type hybrid retriever/orchestrator built on the two stores above.
    ProcessedEvidenceStoreAdapter
        Thin adapter exposing the retriever through the evidence-store query interface.
"""

from .chroma_store import ChromaRecordStore
from .lc_retriever_processed import LCProcessedRetriever
from .processed_evidence_store_adapter import ProcessedEvidenceStoreAdapter
from .processed_record_store import ProcessedRecordStore

__all__ = [
    "ChromaRecordStore",
    "ProcessedRecordStore",
    "LCProcessedRetriever",
    "ProcessedEvidenceStoreAdapter",
]
