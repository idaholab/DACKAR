from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """
    Represents a single reliability text document (condition report, work order, etc.).

    Offsets are always defined on `text` (the original authoritative string).
    Normalization can be added later using `normalized_text` + `offset_map`.
    """
    doc_id: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)
    normalized_text: Optional[str] = None
    offset_map: Any = None


@dataclass
class SourceHit:
    """
    Provenance record for how a candidate span was generated.

    Examples:
      - source_type="regex", source_id="REGEX_FAIL_START"
      - source_type="gazetteer_exact", source_id="deg_mech_list"
      - source_type="noun_chunk", source_id="spacy_np"
    """
    source_type: str
    source_id: str
    score: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LabelHypothesis:
    """
    A proposed label for a candidate span before final resolution.

    `group` is typically filled by schema lookup (label->group).
    `score` can be used as a prior confidence or later overwritten by an ML classifier.
    """
    label: str
    group: Optional[str] = None
    score: Optional[float] = None
    rationale: Optional[str] = None


@dataclass
class CandidateSpan:
    """
    A proposed entity mention span before final decision.

    Fields:
      - start/end: character offsets into Document.text
      - text: cached substring for convenience
      - sources: provenance (where did this candidate come from)
      - proposed_labels: potentially multiple, potentially conflicting
    """
    span_id: str
    doc_id: str
    start: int
    end: int
    text: str

    sources: List[SourceHit] = field(default_factory=list)
    proposed_labels: List[LabelHypothesis] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    is_nested_allowed: bool = True


@dataclass
class ResolvedSpan:
    """
    A finalized span after conflict resolution / compatibility enforcement.

    `labels` may contain multiple labels only if allowed by schema/rules (e.g., G5+G6).
    """
    span_id: str
    doc_id: str
    start: int
    end: int
    text: str

    labels: List[str]
    groups: List[str]
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """
    Records how one or more CandidateSpan(s) are resolved into final output spans.

    action:
      - "accept": accept as-is (possibly with refined label list)
      - "reject": discard
      - "split": output_spans contains >1 span
      - "nest": output contains nested spans
      - "defer": unresolved; keep for ML or human review
    """
    decision_id: str
    doc_id: str

    input_span_ids: List[str]
    output_spans: List[ResolvedSpan]

    action: str
    triggered_rule_ids: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    confidence: Optional[float] = None


@dataclass
class RelationProposal:
    """
    Optional suggested relation between resolved spans. This is not required for v0.1,
    but the compatibility schema can recommend link types like:
      - "causes", "affects", "made_of", "has_outcome"
    """
    rel_id: str
    doc_id: str

    relation_type: str
    head_span_id: str
    tail_span_id: str

    confidence: Optional[float] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    triggered_rule_ids: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """
    Final result for a document.
      - decisions: full traceability of how spans were resolved
      - entities: flattened accepted spans (for typical downstream use)
      - relations: optional relation proposals
      - diagnostics: counters/timings/debug info
    """
    doc_id: str
    decisions: List[Decision]
    entities: List[ResolvedSpan]
    relations: List[RelationProposal]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
