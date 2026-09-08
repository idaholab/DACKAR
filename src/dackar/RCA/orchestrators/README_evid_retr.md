
# Evidence Retriever

Implements a first deterministic `EvidenceRetriever` for the RCA workflow.

## Responsibilities
- build a small set of KG-guided retrieval queries
- apply metadata filters from `kg_context`
- query a retrieval backend (e.g., Chroma/LangChain)
- normalize, deduplicate, and rank hits
- emit `evidence_bundle.json`

## Files
- `evidence_retriever.py`
