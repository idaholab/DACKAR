from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class DocExtractionRecord:
    """One extraction record per identified causal chain within a source document.

    A single CR/WO may produce multiple records (one per causal chain).
    Records with all semantic fields null are stored but flagged for human review.
    `fm_id_candidate` is null at ingestion time; resolved via batch KG lookup at RCA run time (§3.3 Step C).
    `embedding_model_version` is null until the record is embedded in Stage 2.
    """
    doc_id: str
    chain_index: int

    identified_effect: Optional[str]
    assessed_cause: Optional[str]
    inferred_fm_label: Optional[str]

    fm_id_candidate: Optional[str]
    fm_id_candidate_alt: Optional[str]

    confidence: ConfidenceLevel
    cause_is_symptom: bool

    as_found: Optional[str]
    as_left: Optional[str]
    procedural_deviation_score: float

    extraction_version: str
    embedding_model_version: Optional[str]

    needs_human_review: bool = field(default=False)

    def embed_text(self) -> str:
        """Concatenation of semantic fields used for embedding (null-safe)."""
        parts = [self.identified_effect, self.assessed_cause, self.inferred_fm_label]
        return " | ".join(p for p in parts if p)

    def is_null_record(self) -> bool:
        return not any([self.identified_effect, self.assessed_cause, self.inferred_fm_label])

    def as_chroma_metadata(self) -> dict:
        """Flat metadata dict for Chroma upsert (all values must be str/int/float/bool)."""
        return {
            "doc_id": self.doc_id,
            "chain_index": self.chain_index,
            # Semantic text fields stored in metadata for audit retrieval via SemanticMatch
            "identified_effect": self.identified_effect or "",
            "assessed_cause": self.assessed_cause or "",
            "inferred_fm_label": self.inferred_fm_label or "",
            "fm_id_candidate": self.fm_id_candidate or "",
            "fm_id_candidate_alt": self.fm_id_candidate_alt or "",
            "confidence": self.confidence.value,
            "cause_is_symptom": self.cause_is_symptom,
            "as_found": self.as_found or "",
            "as_left": self.as_left or "",
            "procedural_deviation_score": self.procedural_deviation_score,
            "extraction_version": self.extraction_version,
            "embedding_model_version": self.embedding_model_version or "",
            "needs_human_review": self.needs_human_review,
        }
