from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from dackar.RCA.ner.hybrid_ner.models import Document, ResolvedSpan
from .schema import ConfidenceLevel, DocExtractionRecord

# causal_condition_adapter is NOT imported at module level — it pulls in spacy transitively.
# The adapter receives the extractor as an injected callable (see _causal_extractor param)
# or loads it lazily on first use via _get_causal_extractor().

logger = logging.getLogger(__name__)

# Entity group IDs as defined in the hybrid NER schema
_GROUP_MECHANISM = "G4_MECHANISM_PROCESS"
_GROUP_OUTCOME = "G5_FAILURE_OUTCOME"
_GROUP_COMPONENT = "G1_PHYSICAL_COMPONENT"

# Document types in scope for extraction (§2.3)
EXTRACTABLE_DOC_TYPES = frozenset({"CR", "WO", "RCA", "ECA"})


class DocExtractionAdapter:
    """Wraps HybridNERPipeline + causal_condition_adapter to produce DocExtractionRecords.

    Produces one record per identified causal chain in the document.
    A document with no extractable causal language produces one null record (confidence=low,
    needs_human_review=True).

    fm_id_candidate is always None at extraction time; resolved via batch KG lookup
    at RCA run time (see DocExtractionStore.resolve_fm_candidates).
    """

    def __init__(
        self,
        ner_pipeline: Any,
        nlp: Any = None,
        llm_cfg: Optional[Dict[str, Any]] = None,
        extraction_version: str = "ner-v1.0_gaz-v1_llm-none",
        _causal_extractor: Optional[Any] = None,
    ) -> None:
        self._ner = ner_pipeline
        self._nlp = nlp
        self._llm_cfg = llm_cfg
        self.extraction_version = extraction_version
        self.__causal_extractor = _causal_extractor  # injectable; None → lazy-loaded on first use

    def _get_causal_extractor(self) -> Any:
        if self.__causal_extractor is not None:
            return self.__causal_extractor
        from dackar.RCA.ner.causal_condition_adapter import extract_stage5_causal_condition  # noqa: PLC0415
        return extract_stage5_causal_condition

    def extract(
        self,
        doc_id: str,
        text: str,
        doc_type: str,
        section_role: str = "body",
    ) -> List[DocExtractionRecord]:
        """Extract all causal chains from a single document text.

        Args:
            doc_id: Source document identifier (e.g. "CR-2026-00123").
            text: Full document text (single chunk; multi-chunk handling is a future extension).
            doc_type: Must be one of EXTRACTABLE_DOC_TYPES.
            section_role: Hint for condition-state extraction ("body", "as_found", "as_left", etc.).

        Returns:
            List of DocExtractionRecord, one per causal chain. Never empty — a document
            with no causal language returns a single null record.

        Raises:
            ValueError: if doc_type is not in EXTRACTABLE_DOC_TYPES.
        """
        if doc_type not in EXTRACTABLE_DOC_TYPES:
            raise ValueError(
                f"doc_type '{doc_type}' is outside extraction scope. "
                f"Allowed: {sorted(EXTRACTABLE_DOC_TYPES)}"
            )

        # Step A: NER entity extraction
        doc = Document(doc_id=doc_id, text=text)
        try:
            pipeline_result = self._ner.run(doc)
            mechanism_spans = [e for e in pipeline_result.entities if _GROUP_MECHANISM in e.groups]
            outcome_spans = [e for e in pipeline_result.entities if _GROUP_OUTCOME in e.groups]
        except Exception as exc:
            logger.warning("NER pipeline failed for %s: %s — falling back to empty entities", doc_id, exc)
            mechanism_spans = []
            outcome_spans = []

        # Step B: causal relation + condition state extraction
        try:
            stage5 = self._get_causal_extractor()(
                doc_id=doc_id,
                chunk_index=0,
                chunk_text=text,
                doc_type=doc_type,
                section_role=section_role,
                nlp=self._nlp,
                llm_cfg=self._llm_cfg,
            )
        except Exception as exc:
            logger.warning("causal_condition_adapter failed for %s: %s — null record produced", doc_id, exc)
            return [self._null_record(doc_id=doc_id, as_found=None, as_left=None, proc_score=0.0)]

        as_found: Optional[str] = stage5["condition_state"].get("as_found")
        as_left: Optional[str] = stage5["condition_state"].get("as_left")
        proc_score = float(stage5["procedural_deviation"].get("confidence", 0.0))
        extractor_used: str = stage5["extractor"].get("used", "")
        causal_statements: List[Dict[str, Any]] = stage5.get("extracted_causal_statements", [])

        if not causal_statements:
            return [self._null_record(doc_id=doc_id, as_found=as_found, as_left=as_left, proc_score=proc_score)]

        records: List[DocExtractionRecord] = []
        for i, stmt in enumerate(causal_statements):
            cause_text: Optional[str] = (stmt.get("cause_text") or "").strip() or None
            effect_text: Optional[str] = (stmt.get("effect_text") or "").strip() or None
            stmt_confidence = float(stmt.get("confidence", 0.5))
            negated = bool(stmt.get("negated", False))
            conjectural = bool(stmt.get("conjectural", False))

            # Skip negated/conjectural statements — they indicate absence, not occurrence
            if negated or conjectural:
                logger.debug("Skipping negated/conjectural causal statement %d in %s", i, doc_id)
                continue

            identified_effect = effect_text or _best_entity_text(outcome_spans)
            assessed_cause = cause_text
            inferred_fm_label = (
                _best_overlapping_entity(cause_text, mechanism_spans)
                or cause_text  # free-text fallback when no G4 entity overlaps
            )

            cause_is_symptom = _text_overlaps_any(cause_text, outcome_spans) if cause_text else False

            # Step C: fm_id_candidate deferred — always None at ingestion time
            confidence = _assign_confidence(
                extractor_used=extractor_used,
                stmt_confidence=stmt_confidence,
                has_mechanism=bool(_best_overlapping_entity(cause_text, mechanism_spans)),
                mechanism_spans=mechanism_spans,
                outcome_spans=outcome_spans,
            )

            records.append(DocExtractionRecord(
                doc_id=doc_id,
                chain_index=i,
                identified_effect=identified_effect,
                assessed_cause=assessed_cause,
                inferred_fm_label=inferred_fm_label,
                fm_id_candidate=None,
                fm_id_candidate_alt=None,
                confidence=confidence,
                cause_is_symptom=cause_is_symptom,
                as_found=as_found,
                as_left=as_left,
                procedural_deviation_score=proc_score,
                extraction_version=self.extraction_version,
                embedding_model_version=None,
                needs_human_review=(confidence == ConfidenceLevel.LOW),
            ))

        if not records:
            # All statements were negated/conjectural
            return [self._null_record(doc_id=doc_id, as_found=as_found, as_left=as_left, proc_score=proc_score)]

        return records

    def _null_record(
        self,
        doc_id: str,
        as_found: Optional[str],
        as_left: Optional[str],
        proc_score: float,
    ) -> DocExtractionRecord:
        return DocExtractionRecord(
            doc_id=doc_id,
            chain_index=0,
            identified_effect=None,
            assessed_cause=None,
            inferred_fm_label=None,
            fm_id_candidate=None,
            fm_id_candidate_alt=None,
            confidence=ConfidenceLevel.LOW,
            cause_is_symptom=False,
            as_found=as_found,
            as_left=as_left,
            procedural_deviation_score=proc_score,
            extraction_version=self.extraction_version,
            embedding_model_version=None,
            needs_human_review=True,
        )


# ---------------------------------------------------------------------------
# Entity span helpers
# ---------------------------------------------------------------------------

def _best_entity_text(spans: List[ResolvedSpan]) -> Optional[str]:
    """Return the text of the highest-scoring span, or None if list is empty."""
    if not spans:
        return None
    best = max(
        spans,
        key=lambda s: max(
            (float(src.get("score") or 0.0) for src in s.provenance.get("sources", [])),
            default=0.0,
        ),
    )
    return best.text.strip() or None


def _best_overlapping_entity(target_text: Optional[str], spans: List[ResolvedSpan]) -> Optional[str]:
    """Return the text of the first span that overlaps with target_text (substring either way)."""
    if not target_text or not spans:
        return None
    target_lower = target_text.lower()
    for span in spans:
        span_lower = span.text.strip().lower()
        if span_lower and (span_lower in target_lower or target_lower in span_lower):
            return span.text.strip()
    return None


def _text_overlaps_any(target_text: Optional[str], spans: List[ResolvedSpan]) -> bool:
    """Return True if target_text overlaps (substring) with any span in the list."""
    if not target_text or not spans:
        return False
    target_lower = target_text.lower()
    return any(
        span.text.strip().lower() in target_lower or target_lower in span.text.strip().lower()
        for span in spans
        if span.text.strip()
    )


def _has_gazetteer_source(spans: List[ResolvedSpan], min_score: float = 0.85) -> bool:
    """Return True if any span has a gazetteer source with score >= min_score."""
    for span in spans:
        for src in span.provenance.get("sources", []):
            src_type = src.get("source_type", "")
            src_score = float(src.get("score") or 0.0)
            if src_type.startswith("gazetteer") and src_score >= min_score:
                return True
    return False


def _assign_confidence(
    extractor_used: str,
    stmt_confidence: float,
    has_mechanism: bool,
    mechanism_spans: List[ResolvedSpan],
    outcome_spans: List[ResolvedSpan],
) -> ConfidenceLevel:
    """Assign confidence level per §3.4 of the design plan.

    Rules (in priority order):
      LOW  — LLM fallback used
      LOW  — no G4 mechanism entity found (cause is symptom-only or null)
      HIGH — gazetteer hit (score ≥ 0.85) on any G4/G5 span + CausalSentence + stmt_confidence ≥ 0.60
      MEDIUM — CausalSentence or CausalSimple with mechanism present but no strong gazetteer hit
      LOW  — fallback
    """
    if extractor_used == "LLM_implicit":
        return ConfidenceLevel.LOW
    if not has_mechanism:
        return ConfidenceLevel.LOW

    all_relevant_spans = mechanism_spans + outcome_spans
    if (
        extractor_used == "CausalSentence"
        and _has_gazetteer_source(all_relevant_spans)
        and stmt_confidence >= 0.60
    ):
        return ConfidenceLevel.HIGH

    if extractor_used in {"CausalSentence", "CausalSimple"}:
        return ConfidenceLevel.MEDIUM

    return ConfidenceLevel.LOW
