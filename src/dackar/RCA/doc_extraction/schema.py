from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EventTimeConfidence(str, Enum):
    """Confidence level for the event_time_start / event_time_end fields on a DocExtractionRecord.

    Used by the cross-pattern linkage layer (§4.1) to decide whether temporal
    matching can be attempted.  "absent" causes temporal_link_skipped = True.
    """
    EXPLICIT = "explicit"    # populated from a structured timestamp field in the source document
    INFERRED = "inferred"    # derived from surrounding document context; less reliable
    ABSENT   = "absent"      # no event-time information available; temporal linkage skipped


class FMResolutionStatus(str, Enum):
    """Resolution status of fm_id_candidate after batch KG embedding lookup.

    Thresholds (cosine similarity vs. KG FM embeddings):
      >= 0.88  → AUTO_RESOLVED  (eligible for recurrence counting)
      [0.80, 0.88) → AMBIGUOUS  (candidate stored; requires analyst promotion)
      < 0.80   → UNRESOLVED    (fm_id_candidate remains None)
    """
    AUTO_RESOLVED = "auto_resolved"
    AMBIGUOUS     = "ambiguous"
    UNRESOLVED    = "unresolved"


class EpistemicClass(str, Enum):
    """Four-way epistemic classification for data elements (§2 of epistemics_notes_4.md).

    Answers: what relationship does this data element have to equipment performance?

    AFFECTS_PERFORMANCE     — things that act on the equipment; candidate causes
                              (WOs, config changes, PM compliance, operational context)
    MONITORS_PERFORMANCE    — things that observe the equipment's state; evidence of
                              condition, not cause (CRs, telemetry, alarms, SOE)
    ANALYZES_PAST_DEGRADATION — things whose primary purpose is causal interpretation
                              of a specific past event (ECAs, RCAs, OE/LER documents)
    CHARACTERIZES_THE_SYSTEM  — things that define the reference frame: KG, FMEA,
                              protection logic, SOPs
    """
    AFFECTS_PERFORMANCE       = "affects_performance"
    MONITORS_PERFORMANCE      = "monitors_performance"
    ANALYZES_PAST_DEGRADATION = "analyzes_past_degradation"
    CHARACTERIZES_THE_SYSTEM  = "characterizes_the_system"


class ClassificationResolutionLevel(str, Enum):
    """Which level of the priority chain resolved the epistemic classification (§3.3).

    Priority order: FINDING_STATUS → AUTHORITY_LEVEL → DOC_TYPE → DEFAULT

    FINDING_STATUS  — resolved via semantic finding_status field; not degraded
    AUTHORITY_LEVEL — resolved via semantic authority_level field; not degraded
    DOC_TYPE        — resolved via syntactic doc_type proxy; degraded_classification=True
    DEFAULT         — no metadata available; degraded_classification=True
    """
    FINDING_STATUS  = "finding_status"
    AUTHORITY_LEVEL = "authority_level"
    DOC_TYPE        = "doc_type"
    DEFAULT         = "default"


class FindingStatus(str, Enum):
    """Semantic status of a document's finding — primary input to the classifier (Level 1).

    Populated at document ingestion when the source document carries a structured
    finding_status field (e.g., CR closure code, ECA conclusion type).

    FORMAL_CONCLUSION      — document is a finalized causal interpretation
                             → routes to analyzes_past_degradation
    PRELIMINARY_ASSESSMENT — document contains a preliminary cause observation
                             → routes to monitors_performance (CR) or
                               analyzes_past_degradation (ECA/RCA) per doc_type
    OBSERVATION_ONLY       — document records a condition with no causal assessment
                             → routes to monitors_performance
    """
    FORMAL_CONCLUSION      = "formal_conclusion"
    PRELIMINARY_ASSESSMENT = "preliminary_assessment"
    OBSERVATION_ONLY       = "observation_only"


class AuthorityLevel(str, Enum):
    """Epistemic authority of a document — secondary input to the classifier (Level 2).

    Populated at ingestion for OE/industry documents and formal plant documents.
    Captures the tier-distance discount described in §2.3 (plant 1.0 / fleet 0.80 /
    industry 0.60).

    MANDATORY     — plant-level formal conclusion; highest authority
    GUIDANCE      — fleet or industry analogy; discounted authority
    INFORMATIONAL — background or preliminary; no causal authority
    """
    MANDATORY     = "mandatory"
    GUIDANCE      = "guidance"
    INFORMATIONAL = "informational"


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

    # Ruled-out failure mechanism hypotheses from negated causal statements
    # (e.g. "X was NOT caused by Y").  Stored as cause_text strings so Step 2d
    # can penalise FM candidates whose label matches a ruled-out cause.
    ruled_out_mechanisms: List[str] = field(default_factory=list)

    # Event-time fields (§4.1 Phase 0) — capture when the plant event occurred,
    # not document creation time.  Required for temporal cross-pattern linkage.
    event_time_start: Optional[datetime] = None
    event_time_end: Optional[datetime] = None
    event_time_confidence: Optional[str] = None   # EventTimeConfidence values

    # Source reference fields — stable IDs linking this record back to the
    # originating CR / WO / event.  Used by the double-counting guard in
    # DocExtractionStore.query() (exact_doc_ids exclusion).
    source_cr_id: Optional[str] = None
    source_wo_id: Optional[str] = None
    source_event_id: Optional[str] = None

    # FM resolution fields — populated by DocExtractionStore.resolve_fm_candidates().
    # fm_resolution_status governs recurrence eligibility (see is_recurrence_eligible()).
    fm_resolution_status: Optional[str] = None   # FMResolutionStatus values
    fm_resolution_score: Optional[float] = None  # cosine similarity of best FM match

    # Epistemic classifier input fields — populated at document ingestion time.
    # These feed the priority chain: finding_status → authority_level → doc_type → default.
    doc_type: str = ""                    # CR | WO | ECA | RCA | FMEA | SOP | OE | LER | ...
    finding_status: Optional[str] = None  # FindingStatus values; Level 1 classifier input
    authority_level: Optional[str] = None # AuthorityLevel values; Level 2 classifier input

    # Epistemic annotation fields — populated by EpistemicClassifier (Phase A).
    # epistemic_class is None until the classifier has been run on this record.
    epistemic_class: Optional[str] = None              # EpistemicClass values
    classification_resolution_level: Optional[str] = None  # ClassificationResolutionLevel values
    degraded_classification: bool = False               # True when resolved via doc_type or default

    def embed_text(self) -> str:
        """Concatenation of semantic fields used for embedding (null-safe)."""
        parts = [self.identified_effect, self.assessed_cause, self.inferred_fm_label]
        return " | ".join(p for p in parts if p)

    def is_null_record(self) -> bool:
        return not any([self.identified_effect, self.assessed_cause, self.inferred_fm_label])

    def is_recurrence_eligible(self) -> bool:
        """Return True when this record may contribute to recurrence counting.

        Records with fm_resolution_status == "ambiguous" require analyst promotion
        before being eligible; all others (auto_resolved, unresolved, or not yet
        resolved) are treated as eligible by default.

        Phase C will add an additional gate: records whose epistemic_class is not
        "analyzes_past_degradation" will be ineligible regardless of fm_resolution_status.
        That gate is intentionally off here to avoid a scoring change before calibration.
        """
        return self.fm_resolution_status != FMResolutionStatus.AMBIGUOUS

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
            # Event-time fields
            "event_time_start": self.event_time_start.isoformat() if self.event_time_start else "",
            "event_time_end": self.event_time_end.isoformat() if self.event_time_end else "",
            "event_time_confidence": self.event_time_confidence or "",
            # Source reference fields
            "source_cr_id": self.source_cr_id or "",
            "source_wo_id": self.source_wo_id or "",
            "source_event_id": self.source_event_id or "",
            # FM resolution fields
            "fm_resolution_status": self.fm_resolution_status or "",
            "fm_resolution_score": self.fm_resolution_score if self.fm_resolution_score is not None else -1.0,
            # Epistemic classifier input fields
            "doc_type": self.doc_type or "",
            "finding_status": self.finding_status or "",
            "authority_level": self.authority_level or "",
            # Epistemic annotation fields (populated by EpistemicClassifier)
            "epistemic_class": self.epistemic_class or "",
            "classification_resolution_level": self.classification_resolution_level or "",
            "degraded_classification": self.degraded_classification,
            # Semicolon-joined ruled-out cause texts for Step 2d FM penalisation
            "ruled_out_mechanisms": "; ".join(self.ruled_out_mechanisms),
        }
