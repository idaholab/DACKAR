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

# Jaccard threshold for token-set and lemma-set entity linking (Stage 3).
# Calibrate against the annotated test dataset; 0.4 is a conservative starting point.
_ENTITY_LINK_JACCARD_THRESHOLD = 0.4

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
            outcome_spans   = [e for e in pipeline_result.entities if _GROUP_OUTCOME   in e.groups]
            component_spans = [e for e in pipeline_result.entities if _GROUP_COMPONENT in e.groups]
        except Exception as exc:
            logger.warning("NER pipeline failed for %s: %s — falling back to empty entities", doc_id, exc)
            mechanism_spans = []
            outcome_spans   = []
            component_spans = []

        # Build a causal_sentence_factory that injects NER-extracted entities as SSC
        # patterns so CausalSentence fires on the same entities the NER pipeline found.
        # Without this bridge, CausalSentence._matchedSents is always empty (no SSC
        # annotations) and every document falls through to dep_fallback.
        cs_factory = _make_ner_cs_factory(mechanism_spans, outcome_spans, component_spans)

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
                causal_sentence_factory=cs_factory,
            )
        except Exception as exc:
            logger.warning("causal_condition_adapter failed for %s: %s — null record produced", doc_id, exc)
            return [self._null_record(doc_id=doc_id, as_found=None, as_left=None, proc_score=0.0)]

        as_found: Optional[str] = stage5["condition_state"].get("as_found")
        as_left: Optional[str] = stage5["condition_state"].get("as_left")
        proc_score = float(stage5["procedural_deviation"].get("confidence", 0.0))
        extractor_used: str = stage5["extractor"].get("used", "")
        causal_statements: List[Dict[str, Any]] = stage5.get("extracted_causal_statements", [])
        causal_chains: List[Dict[str, Any]] = stage5.get("causal_chain", [])
        # Ruled-out mechanisms: cause_text strings from negated statements (14.4).
        # Stored on every record from this document for Step 2d FM penalisation.
        ruled_out: List[str] = [
            s.get("cause_text", "")
            for s in stage5.get("ruled_out_mechanisms", [])
            if s.get("cause_text")
        ]

        if not causal_statements:
            return [self._null_record(doc_id=doc_id, as_found=as_found, as_left=as_left, proc_score=proc_score)]

        records: List[DocExtractionRecord] = []
        covered_stmt_ids: set = set()

        # Build a lookup from statement_id → source for Improvement E chain confidence.
        stmt_source_by_id: dict = {
            s.get("statement_id", ""): s.get("source", "")
            for s in causal_statements
        }

        # Step C-chain: one record per multi-hop chain (nodes ≥ 2).
        # Chain fields: root node → inferred_fm_label (root cause);
        # proximate node ([-2]) → assessed_cause; terminal node ([-1]) → identified_effect.
        # cause_is_symptom = True when intermediate nodes exist (chain length > 2).
        for ci, chain_dict in enumerate(causal_chains):
            nodes = chain_dict.get("nodes", [])
            if len(nodes) < 2:
                continue
            source_ids = chain_dict.get("source_statement_ids", [])
            covered_stmt_ids.update(source_ids)

            root_text = nodes[0]
            terminal_text = nodes[-1]
            proximate_text = nodes[-2]

            inferred_fm = (
                _best_overlapping_entity(root_text, mechanism_spans, nlp=self._nlp)
                or root_text
            )
            has_mech = bool(_best_overlapping_entity(root_text, mechanism_spans, nlp=self._nlp))
            min_conf = float(chain_dict.get("min_confidence", 0.5))

            # Improvement E: if all contributing statements came from dep_fallback,
            # pass that source so _assign_confidence caps them at MEDIUM.
            chain_sources = {stmt_source_by_id.get(sid, "") for sid in source_ids}
            chain_stmt_source = "dep_fallback" if chain_sources <= {"dep_fallback", ""} else ""

            confidence = _assign_confidence(
                extractor_used=extractor_used,
                stmt_confidence=min_conf,
                has_mechanism=has_mech,
                mechanism_spans=mechanism_spans,
                outcome_spans=outcome_spans,
                stmt_source=chain_stmt_source,
            )

            records.append(DocExtractionRecord(
                doc_id=doc_id,
                chain_index=ci,
                identified_effect=terminal_text or None,
                assessed_cause=proximate_text or None,
                inferred_fm_label=inferred_fm,
                fm_id_candidate=None,
                fm_id_candidate_alt=None,
                confidence=confidence,
                cause_is_symptom=len(nodes) > 2,
                as_found=as_found,
                as_left=as_left,
                procedural_deviation_score=proc_score,
                extraction_version=self.extraction_version,
                embedding_model_version=None,
                needs_human_review=(confidence == ConfidenceLevel.LOW),
                ruled_out_mechanisms=ruled_out,
            ))

        # Step C-stmt: one record per statement not covered by any chain above.
        chain_record_count = len(records)
        for i, stmt in enumerate(causal_statements):
            if stmt.get("statement_id") in covered_stmt_ids:
                continue

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
                _best_overlapping_entity(cause_text, mechanism_spans, nlp=self._nlp)
                or cause_text  # free-text fallback when no G4 entity overlaps
            )

            cause_is_symptom = _text_overlaps_any(cause_text, outcome_spans) if cause_text else False

            # fm_id_candidate deferred — always None at ingestion time
            # Improvement E: pass per-statement source so dep_fallback statements
            # are not promoted to HIGH via the CausalSentence rules.
            confidence = _assign_confidence(
                extractor_used=extractor_used,
                stmt_confidence=stmt_confidence,
                has_mechanism=bool(_best_overlapping_entity(cause_text, mechanism_spans, nlp=self._nlp)),
                mechanism_spans=mechanism_spans,
                outcome_spans=outcome_spans,
                stmt_source=stmt.get("source", ""),
            )

            records.append(DocExtractionRecord(
                doc_id=doc_id,
                chain_index=chain_record_count + i,
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
                ruled_out_mechanisms=ruled_out,
            ))

        if not records:
            # All statements were negated/conjectural and no chains produced records
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
# NER → CausalSentence bridge
# ---------------------------------------------------------------------------

# Lemma-based causal keyword patterns injected into CausalSentence via EntityRuler.
# CausalSentence selects sentences that contain BOTH an SSC entity AND a causal
# keyword; without the keyword patterns the matched-sentence buffer is always empty.
# These lemmas cover the core causal vocabulary; the dep_fallback path handles the
# full extended set from cause_effect_keywords_full.csv.
_CAUSAL_KEYWORD_PATTERNS: List[Dict[str, Any]] = [
    {"label": "causal", "pattern": [{"LEMMA": lemma}], "id": "causal"}
    for lemma in [
        "cause", "result", "lead", "trigger", "induce", "produce", "create",
        "force", "drive", "contribute", "attribute", "accelerate", "activate",
        "affect", "damage", "facilitate", "generate", "initiate", "originate",
        "precipitate", "prevent", "promote", "prompt", "propagate", "spark",
        "stimulate",
    ]
]


def _make_ner_cs_factory(
    mechanism_spans: List[ResolvedSpan],
    outcome_spans: List[ResolvedSpan],
    component_spans: List[ResolvedSpan],
) -> Optional[Any]:
    """Build a causal_sentence_factory that injects NER entities as SSC patterns.

    CausalSentence requires two types of EntityRuler patterns to fire:
      1. SSC entity patterns — G4 mechanism, G5 outcome, G1 component spans from NER
      2. Causal keyword patterns — lemma-based verb patterns for sentence selection

    Without both, _matchedSents is empty and the extractor silently falls back to
    dep_fallback on every document. This factory bridges the NER pipeline output
    into the CausalSentence input contract.

    Note: EntityRuler patterns are added to the shared nlp pipeline and persist
    across documents. Pattern accumulation is benign for nuclear domain entities
    (all terms remain valid SSC candidates) but grows linearly with corpus size.

    Returns None when no entity spans are available (CausalSentence would return
    empty anyway; dep_fallback remains the active extractor).
    """
    all_spans = mechanism_spans + outcome_spans + component_spans
    ssc_texts = [s.text.strip() for s in all_spans if s.text.strip()]
    if not ssc_texts:
        return None

    ssc_patterns = [{"label": "SSC", "pattern": t, "id": "SSC"} for t in dict.fromkeys(ssc_texts)]

    def factory(text: str, nlp: Any) -> Any:
        from dackar.RCA.causal.CausalSentence import CausalSentence  # noqa: PLC0415
        cs = CausalSentence(nlp)
        cs.addEntityPattern("ner_ssc_entities", ssc_patterns)
        cs.addEntityPattern("ner_causal_keywords", _CAUSAL_KEYWORD_PATTERNS)
        return cs

    return factory


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


def _best_overlapping_entity(
    target_text: Optional[str],
    spans: List[ResolvedSpan],
    nlp: Optional[Any] = None,
) -> Optional[str]:
    """Return the text of the best-matching span for target_text.

    Priority chain (stops at first hit):
      1. Exact substring match — O(n), no dependencies
      2. Token-set Jaccard ≥ _ENTITY_LINK_JACCARD_THRESHOLD
      3. Lemma match via spaCy: lemmatize both sides, re-apply (1) then (2)
         Only attempted when nlp is provided.
    """
    if not target_text or not spans:
        return None

    target_lower = target_text.lower()

    # Pass 1: exact substring
    for span in spans:
        span_lower = span.text.strip().lower()
        if span_lower and (span_lower in target_lower or target_lower in span_lower):
            return span.text.strip()

    # Shared token helpers for passes 2 and 3
    def _tok_set(text: str) -> frozenset:
        return frozenset(t for t in text.lower().split() if len(t) > 1)

    def _jaccard(a: frozenset, b: frozenset) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # Pass 2: token-set Jaccard
    target_tokens = _tok_set(target_text)
    best_span: Optional[str] = None
    best_score = 0.0
    for span in spans:
        score = _jaccard(target_tokens, _tok_set(span.text))
        if score >= _ENTITY_LINK_JACCARD_THRESHOLD and score > best_score:
            best_score = score
            best_span = span.text.strip()
    if best_span:
        return best_span

    # Pass 3: lemma match — lemmatize both sides, re-apply substring then Jaccard
    if nlp is not None:
        try:
            def _lemma_tokens(text: str) -> frozenset:
                return frozenset(
                    tok.lemma_.lower()
                    for tok in nlp(text)
                    if not tok.is_punct and not tok.is_space
                )

            target_lemmas = _lemma_tokens(target_text)
            target_lemma_str = " ".join(sorted(target_lemmas))
            best_score = 0.0  # reset for lemma-Jaccard ranking
            best_span = None
            for span in spans:
                span_lemmas = _lemma_tokens(span.text)
                span_lemma_str = " ".join(sorted(span_lemmas))
                # Re-apply pass 1 on lemmatized strings
                if span_lemma_str and (
                    span_lemma_str in target_lemma_str or target_lemma_str in span_lemma_str
                ):
                    return span.text.strip()
                # Re-apply pass 2 on lemma token sets
                score = _jaccard(target_lemmas, span_lemmas)
                if score >= _ENTITY_LINK_JACCARD_THRESHOLD and score > best_score:
                    best_score = score
                    best_span = span.text.strip()
            if best_span:
                return best_span
        except Exception:
            pass

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
    stmt_source: str = "",
) -> ConfidenceLevel:
    """Assign confidence level per §3.4 of the design plan.

    Rules (in priority order):
      LOW    — LLM fallback used
      LOW    — no G4 mechanism entity found (cause is symptom-only or null)
      MEDIUM — dep_fallback source with mechanism present (Improvement E)
               dep_fallback spans have lower linguistic quality than CausalSentence;
               they are never promoted to HIGH regardless of gazetteer hits.
      HIGH   — CausalSentence + gazetteer hit (score ≥ 0.85) + stmt_confidence ≥ 0.60
      MEDIUM — CausalSentence or CausalSimple with mechanism present, no strong gazetteer hit
      LOW    — fallback

    Args:
        stmt_source: per-statement ``source`` field (e.g. ``"dep_fallback"``,
            ``"CausalSentence"``).  When provided, takes precedence over
            ``extractor_used`` for the dep_fallback rule so that dep-tree
            statements embedded inside a CausalSentence result dict are not
            incorrectly promoted to MEDIUM/HIGH via the CausalSentence path
            (Improvement E).
    """
    if extractor_used in {"LLM_implicit", "LLM_extract_all"}:
        return ConfidenceLevel.LOW
    if not has_mechanism:
        return ConfidenceLevel.LOW

    # Improvement E: dep_fallback statements are capped at MEDIUM.
    # stmt_source is set per-statement; extractor_used is the top-level label
    # which is "CausalSentence" even when dep_fallback provided the statements.
    effective_source = stmt_source or extractor_used
    if "dep_fallback" in effective_source:
        return ConfidenceLevel.MEDIUM

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
