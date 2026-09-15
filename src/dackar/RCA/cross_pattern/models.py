from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class HistoricalDocExtraction:
    """Canonical cross-pattern representation of a single doc extraction record.

    Produced by converting a SemanticMatch (from DocExtractionStore.query()) into
    a richer object that carries temporal and FM resolution metadata needed for
    the linkage rules in rules.py.

    source_episode_ids is populated by CrossPatternLinker.run() after linkage.
    """
    doc_id: str
    doc_type: str
    asset_id: Optional[str]
    event_time_start: Optional[datetime]
    event_time_end: Optional[datetime]
    event_time_confidence: str           # "explicit" | "inferred" | "absent"
    identified_effect: Optional[str]
    assessed_cause: Optional[str]
    inferred_fm_label: Optional[str]
    fm_id_candidate: Optional[str]
    fm_id_candidate_alt: Optional[str]
    fm_resolution_status: str            # "auto_resolved" | "ambiguous" | "unresolved"
    fm_resolution_score: Optional[float]
    confidence: str                      # "high" | "medium" | "low"
    cause_is_symptom: bool
    source_episode_ids: List[str] = field(default_factory=list)  # populated by linker
    # Epistemic annotation fields — carried from SemanticMatch (Phase A)
    epistemic_class: Optional[str] = None
    classification_resolution_level: Optional[str] = None
    degraded_classification: bool = False


@dataclass
class CrossPatternLink:
    """A single link between one HistoricalSignalEpisode and one HistoricalDocExtraction.

    Carries full provenance so analysts can trace exactly which checks passed
    (direct reference, temporal+asset, or semantic/FM).
    """
    link_id: str
    episode_id: str
    doc_id: str
    asset_match: bool
    time_overlap_hours: Optional[float]
    temporal_link_skipped: bool
    linkage_precedence_level: int        # 1=direct, 2=temporal+asset, 3=semantic/FM
    component_overlap: List[str]
    fm_alignment_score: Optional[float]
    signal_similarity_score: float
    document_similarity_score: Optional[float]
    link_confidence: float
    provenance: Dict[str, Any]


@dataclass
class CandidateCrossPatternEvidence:
    """Aggregated cross-pattern evidence for a single causality candidate.

    Produced by CrossPatternLinker.run() — one instance per candidate in
    causality_candidates["candidates"].
    """
    candidate_id: str
    component_id: str
    fm_id: str
    linked_episode_ids: List[str]
    linked_doc_ids: List[str]
    best_link_score: float
    support_posture: str                # "reinforcing" | "conflicting" | "weakly_supporting" | "unresolved"
    reinforcement_strength: Optional[str]   # "single" | "multiple_consistent" | "mixed" | None
    linkage_outcome: str                # "linked" | "no_data" | "no_match" | "below_threshold"
    evidence_paths: List[CrossPatternLink]
