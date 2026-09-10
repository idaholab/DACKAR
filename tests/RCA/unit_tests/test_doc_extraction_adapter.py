"""Unit tests for DocExtractionAdapter (Phase 1 — extraction only).

All tests use lightweight mocks so no NLP models or Chroma instance is required.
The causal extractor (which pulls in spacy) is injected as a mock callable via the
_causal_extractor parameter — no spacy import occurs at collection time.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from dackar.RCA.doc_extraction.adapter import (
    DocExtractionAdapter,
    EXTRACTABLE_DOC_TYPES,
    _assign_confidence,
    _best_overlapping_entity,
    _has_gazetteer_source,
    _text_overlaps_any,
)
from dackar.RCA.doc_extraction.schema import ConfidenceLevel, DocExtractionRecord
from dackar.RCA.ner.hybrid_ner.models import PipelineResult, ResolvedSpan


# ---------------------------------------------------------------------------
# Fixtures — minimal mock objects
# ---------------------------------------------------------------------------

def _resolved_span(
    text: str,
    groups: List[str],
    source_type: str = "gazetteer_exact",
    score: float = 0.90,
) -> ResolvedSpan:
    return ResolvedSpan(
        span_id="test-span",
        doc_id="test-doc",
        start=0,
        end=len(text),
        text=text,
        labels=[],
        groups=groups,
        provenance={"sources": [{"source_type": source_type, "score": score, "source_id": "test"}]},
    )


def _pipeline_result(entities: List[ResolvedSpan]) -> PipelineResult:
    return PipelineResult(
        doc_id="test-doc",
        decisions=[],
        entities=entities,
        relations=[],
    )


def _stage5_output(
    cause_text: str = "",
    effect_text: str = "",
    extractor_used: str = "CausalSentence",
    stmt_confidence: float = 0.75,
    as_found: Optional[str] = None,
    as_left: Optional[str] = None,
    proc_confidence: float = 0.0,
    negated: bool = False,
    conjectural: bool = False,
    n_statements: int = 1,
) -> Dict[str, Any]:
    statements = [
        {
            "statement_id": f"doc::0::cause::{i}",
            "cause_text": cause_text,
            "effect_text": effect_text,
            "connector": "due to",
            "confidence": stmt_confidence,
            "negated": negated,
            "conjectural": conjectural,
            "source": extractor_used,
        }
        for i in range(n_statements)
    ]
    return {
        "stage": "stage5_causal_condition_extraction",
        "status": "ok",
        "extractor": {"primary": "CausalSentence", "fallback": "CausalSimple", "used": extractor_used, "version": "v1"},
        "summary_flags": {"has_explicit_causal_statement": bool(statements)},
        "extracted_causal_statements": statements,
        "condition_state": {"as_found": as_found, "as_left": as_left, "status_mentions": [], "evidence": []},
        "procedural_deviation": {"detected": False, "evidence": [], "confidence": proc_confidence},
        "errors": [],
    }


def _empty_stage5() -> Dict[str, Any]:
    return {
        "stage": "stage5_causal_condition_extraction",
        "status": "empty",
        "extractor": {"used": "", "primary": "CausalSentence", "fallback": "CausalSimple", "version": "v1"},
        "summary_flags": {"has_explicit_causal_statement": False},
        "extracted_causal_statements": [],
        "condition_state": {"as_found": None, "as_left": None, "status_mentions": [], "evidence": []},
        "procedural_deviation": {"detected": False, "evidence": [], "confidence": 0.0},
        "errors": [],
    }


def _make_adapter(
    ner_entities: Optional[List[ResolvedSpan]] = None,
    stage5: Optional[Dict[str, Any]] = None,
    *,
    entities: Optional[List[ResolvedSpan]] = None,  # alias for ner_entities
) -> DocExtractionAdapter:
    """Build an adapter with both NER pipeline and causal extractor fully mocked."""
    resolved_entities = entities if entities is not None else (ner_entities or [])
    ner_mock = MagicMock()
    ner_mock.run.return_value = _pipeline_result(resolved_entities)

    return DocExtractionAdapter(
        ner_pipeline=ner_mock,
        nlp=None,
        llm_cfg=None,
        extraction_version="ner-v1.0_gaz-v1_llm-none",
        _causal_extractor=MagicMock(return_value=stage5),
    )


# ---------------------------------------------------------------------------
# Tests — extraction granularity
# ---------------------------------------------------------------------------

class TestExtractionGranularity:
    def test_two_causal_chains_produce_two_records(self):
        entities = [
            _resolved_span("bearing wear", groups=["G4_MECHANISM_PROCESS"]),
            _resolved_span("pump trip", groups=["G5_FAILURE_OUTCOME"]),
        ]
        stage5 = _stage5_output(cause_text="bearing wear", effect_text="pump trip", n_statements=2)
        adapter = _make_adapter(entities, stage5)

        records = adapter.extract("CR-001", "bearing wear caused pump trip", "CR")

        assert len(records) == 2
        assert records[0].chain_index == 0
        assert records[1].chain_index == 1

    def test_single_causal_chain_produces_one_record(self):
        entities = [_resolved_span("corrosion", groups=["G4_MECHANISM_PROCESS"])]
        stage5 = _stage5_output(cause_text="corrosion", effect_text="leak detected")
        adapter = _make_adapter(entities, stage5)

        records = adapter.extract("CR-002", "corrosion caused leak", "CR")

        assert len(records) == 1
        assert records[0].assessed_cause == "corrosion"
        assert records[0].identified_effect == "leak detected"

    def test_no_causal_language_produces_null_record(self):
        adapter = _make_adapter(entities=[], stage5=_empty_stage5())

        records = adapter.extract("CR-003", "pump noise observed", "CR")

        assert len(records) == 1
        assert records[0].is_null_record()
        assert records[0].confidence == ConfidenceLevel.LOW
        assert records[0].needs_human_review is True

    def test_inferred_fm_label_from_g4_entity(self):
        entities = [_resolved_span("bearing wear", groups=["G4_MECHANISM_PROCESS"])]
        stage5 = _stage5_output(cause_text="bearing wear due to lubrication loss", effect_text="pump trip")
        adapter = _make_adapter(entities, stage5)

        records = adapter.extract("CR-004", "...", "CR")

        # G4 entity "bearing wear" overlaps with cause_text → used as inferred_fm_label
        assert records[0].inferred_fm_label == "bearing wear"

    def test_inferred_fm_label_falls_back_to_cause_text(self):
        # No G4 entity → inferred_fm_label = cause_text itself
        stage5 = _stage5_output(cause_text="lubrication loss", effect_text="pump trip")
        adapter = _make_adapter(entities=[], stage5=stage5)

        records = adapter.extract("CR-005", "...", "CR")

        assert records[0].inferred_fm_label == "lubrication loss"


# ---------------------------------------------------------------------------
# Tests — cause_is_symptom detection
# ---------------------------------------------------------------------------

class TestCauseIsSymptom:
    def test_g5_cause_text_sets_cause_is_symptom_true(self):
        entities = [_resolved_span("pump trip", groups=["G5_FAILURE_OUTCOME"])]
        stage5 = _stage5_output(cause_text="pump trip", effect_text="system shutdown")
        adapter = _make_adapter(entities, stage5)

        records = adapter.extract("CR-010", "pump trip caused system shutdown", "CR")

        assert records[0].cause_is_symptom is True

    def test_g4_cause_text_sets_cause_is_symptom_false(self):
        entities = [_resolved_span("bearing wear", groups=["G4_MECHANISM_PROCESS"])]
        stage5 = _stage5_output(cause_text="bearing wear", effect_text="pump trip")
        adapter = _make_adapter(entities, stage5)

        records = adapter.extract("CR-011", "bearing wear caused pump trip", "CR")

        assert records[0].cause_is_symptom is False

    def test_no_entity_overlap_cause_is_symptom_false(self):
        stage5 = _stage5_output(cause_text="unknown cause", effect_text="valve failure")
        adapter = _make_adapter(entities=[], stage5=stage5)

        records = adapter.extract("CR-012", "unknown cause led to valve failure", "CR")

        assert records[0].cause_is_symptom is False

    def test_g4_and_g5_same_doc_cause_from_g4_not_symptom(self):
        entities = [
            _resolved_span("corrosion", groups=["G4_MECHANISM_PROCESS"]),
            _resolved_span("leak detected", groups=["G5_FAILURE_OUTCOME"]),
        ]
        stage5 = _stage5_output(cause_text="corrosion", effect_text="leak detected")
        adapter = _make_adapter(entities, stage5)

        records = adapter.extract("CR-013", "corrosion caused leak detected", "CR")

        assert records[0].cause_is_symptom is False


# ---------------------------------------------------------------------------
# Tests — confidence assignment
# ---------------------------------------------------------------------------

class TestConfidenceAssignment:
    def test_high_confidence_gazetteer_causal_sentence(self):
        spans = [_resolved_span("bearing wear", groups=["G4_MECHANISM_PROCESS"], source_type="gazetteer_exact", score=0.92)]
        conf = _assign_confidence(
            extractor_used="CausalSentence",
            stmt_confidence=0.75,
            has_mechanism=True,
            mechanism_spans=spans,
            outcome_spans=[],
        )
        assert conf == ConfidenceLevel.HIGH

    def test_medium_confidence_causal_simple_with_mechanism(self):
        spans = [_resolved_span("corrosion", groups=["G4_MECHANISM_PROCESS"], source_type="noun_chunk", score=0.60)]
        conf = _assign_confidence(
            extractor_used="CausalSimple",
            stmt_confidence=0.55,
            has_mechanism=True,
            mechanism_spans=spans,
            outcome_spans=[],
        )
        assert conf == ConfidenceLevel.MEDIUM

    def test_low_confidence_llm_fallback(self):
        spans = [_resolved_span("bearing wear", groups=["G4_MECHANISM_PROCESS"], source_type="gazetteer_exact", score=0.95)]
        conf = _assign_confidence(
            extractor_used="LLM_implicit",
            stmt_confidence=0.80,
            has_mechanism=True,
            mechanism_spans=spans,
            outcome_spans=[],
        )
        assert conf == ConfidenceLevel.LOW

    def test_low_confidence_no_mechanism(self):
        spans = [_resolved_span("pump trip", groups=["G5_FAILURE_OUTCOME"])]
        conf = _assign_confidence(
            extractor_used="CausalSentence",
            stmt_confidence=0.80,
            has_mechanism=False,
            mechanism_spans=[],
            outcome_spans=spans,
        )
        assert conf == ConfidenceLevel.LOW

    def test_medium_confidence_causal_sentence_no_gazetteer_hit(self):
        spans = [_resolved_span("erosion", groups=["G4_MECHANISM_PROCESS"], source_type="noun_chunk", score=0.70)]
        conf = _assign_confidence(
            extractor_used="CausalSentence",
            stmt_confidence=0.65,
            has_mechanism=True,
            mechanism_spans=spans,
            outcome_spans=[],
        )
        # CausalSentence but no gazetteer hit → MEDIUM (not HIGH)
        assert conf == ConfidenceLevel.MEDIUM

    def test_high_requires_stmt_confidence_threshold(self):
        spans = [_resolved_span("fatigue", groups=["G4_MECHANISM_PROCESS"], source_type="gazetteer_exact", score=0.88)]
        conf = _assign_confidence(
            extractor_used="CausalSentence",
            stmt_confidence=0.45,   # below 0.60 threshold
            has_mechanism=True,
            mechanism_spans=spans,
            outcome_spans=[],
        )
        assert conf == ConfidenceLevel.MEDIUM


# ---------------------------------------------------------------------------
# Tests — document scope enforcement
# ---------------------------------------------------------------------------

class TestDocumentScope:
    def test_raises_for_excluded_doc_type(self):
        adapter = DocExtractionAdapter(ner_pipeline=MagicMock(), extraction_version="test")
        with pytest.raises(ValueError, match="outside extraction scope"):
            adapter.extract("SOP-001", "some text", "SOP")

    def test_all_included_doc_types_accepted(self):
        for doc_type in EXTRACTABLE_DOC_TYPES:
            adapter = _make_adapter(entities=[], stage5=_empty_stage5())
            records = adapter.extract("DOC-001", "text", doc_type)
            assert len(records) >= 1


# ---------------------------------------------------------------------------
# Tests — negated / conjectural statements skipped
# ---------------------------------------------------------------------------

class TestNegatedConjectural:
    def test_negated_statement_produces_null_record(self):
        entities = [_resolved_span("bearing wear", groups=["G4_MECHANISM_PROCESS"])]
        stage5 = _stage5_output(cause_text="bearing wear", effect_text="pump trip", negated=True)
        adapter = _make_adapter(entities, stage5)

        records = adapter.extract("CR-020", "bearing wear did not cause pump trip", "CR")

        assert len(records) == 1
        assert records[0].is_null_record()

    def test_conjectural_statement_produces_null_record(self):
        entities = [_resolved_span("cavitation", groups=["G4_MECHANISM_PROCESS"])]
        stage5 = _stage5_output(cause_text="cavitation", effect_text="impeller damage", conjectural=True)
        adapter = _make_adapter(entities, stage5)

        records = adapter.extract("CR-021", "cavitation may have caused impeller damage", "CR")

        assert len(records) == 1
        assert records[0].is_null_record()


# ---------------------------------------------------------------------------
# Tests — fm_id_candidate always null at ingestion time (deferred to Step C)
# ---------------------------------------------------------------------------

class TestFmIdCandidateDeferred:
    def test_fm_id_candidate_is_none_at_extraction(self):
        entities = [_resolved_span("bearing wear", groups=["G4_MECHANISM_PROCESS"])]
        stage5 = _stage5_output(cause_text="bearing wear", effect_text="pump trip")
        adapter = _make_adapter(entities, stage5)

        records = adapter.extract("CR-030", "...", "CR")

        assert records[0].fm_id_candidate is None
        assert records[0].fm_id_candidate_alt is None

    def test_embedding_model_version_is_none_at_extraction(self):
        stage5 = _stage5_output(cause_text="corrosion", effect_text="wall thinning")
        adapter = _make_adapter(entities=[], stage5=stage5)

        records = adapter.extract("CR-031", "...", "WO")

        assert records[0].embedding_model_version is None


# ---------------------------------------------------------------------------
# Tests — schema / metadata helpers
# ---------------------------------------------------------------------------

class TestSchemaHelpers:
    def test_embed_text_concatenates_non_null_fields(self):
        rec = DocExtractionRecord(
            doc_id="CR-001", chain_index=0,
            identified_effect="pump trip",
            assessed_cause="bearing wear",
            inferred_fm_label="bearing wear — lubrication starvation",
            fm_id_candidate=None, fm_id_candidate_alt=None,
            confidence=ConfidenceLevel.HIGH,
            cause_is_symptom=False,
            as_found="degraded", as_left=None,
            procedural_deviation_score=0.0,
            extraction_version="ner-v1.0_gaz-v1_llm-none",
            embedding_model_version=None,
        )
        assert rec.embed_text() == "pump trip | bearing wear | bearing wear — lubrication starvation"

    def test_embed_text_skips_null_fields(self):
        rec = DocExtractionRecord(
            doc_id="CR-002", chain_index=0,
            identified_effect="leak detected",
            assessed_cause=None,
            inferred_fm_label=None,
            fm_id_candidate=None, fm_id_candidate_alt=None,
            confidence=ConfidenceLevel.LOW,
            cause_is_symptom=False,
            as_found=None, as_left=None,
            procedural_deviation_score=0.0,
            extraction_version="ner-v1.0",
            embedding_model_version=None,
        )
        assert rec.embed_text() == "leak detected"

    def test_is_null_record_true_when_all_fields_empty(self):
        rec = DocExtractionRecord(
            doc_id="CR-003", chain_index=0,
            identified_effect=None, assessed_cause=None, inferred_fm_label=None,
            fm_id_candidate=None, fm_id_candidate_alt=None,
            confidence=ConfidenceLevel.LOW, cause_is_symptom=False,
            as_found=None, as_left=None, procedural_deviation_score=0.0,
            extraction_version="ner-v1.0", embedding_model_version=None,
        )
        assert rec.is_null_record() is True

    def test_is_null_record_false_when_any_field_set(self):
        rec = DocExtractionRecord(
            doc_id="CR-003", chain_index=0,
            identified_effect="vibration", assessed_cause=None, inferred_fm_label=None,
            fm_id_candidate=None, fm_id_candidate_alt=None,
            confidence=ConfidenceLevel.LOW, cause_is_symptom=False,
            as_found=None, as_left=None, procedural_deviation_score=0.0,
            extraction_version="ner-v1.0", embedding_model_version=None,
        )
        assert rec.is_null_record() is False

    def test_as_chroma_metadata_all_primitive_types(self):
        rec = DocExtractionRecord(
            doc_id="CR-004", chain_index=1,
            identified_effect="vibration", assessed_cause=None, inferred_fm_label=None,
            fm_id_candidate="FM-001", fm_id_candidate_alt=None,
            confidence=ConfidenceLevel.MEDIUM, cause_is_symptom=True,
            as_found="degraded", as_left="acceptable",
            procedural_deviation_score=0.25,
            extraction_version="ner-v1.0_gaz-v2_llm-none",
            embedding_model_version="nomic-embed-text-v1.5",
        )
        meta = rec.as_chroma_metadata()
        for k, v in meta.items():
            assert isinstance(v, (str, int, float, bool)), f"field {k} has non-primitive type {type(v)}"
        assert meta["fm_id_candidate"] == "FM-001"
        assert meta["fm_id_candidate_alt"] == ""  # None → empty string
        assert meta["cause_is_symptom"] is True


# ---------------------------------------------------------------------------
# Tests — helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_best_overlapping_entity_substring_match(self):
        spans = [_resolved_span("bearing wear", groups=["G4_MECHANISM_PROCESS"])]
        result = _best_overlapping_entity("bearing wear due to lubrication loss", spans)
        assert result == "bearing wear"

    def test_best_overlapping_entity_no_match(self):
        spans = [_resolved_span("corrosion", groups=["G4_MECHANISM_PROCESS"])]
        result = _best_overlapping_entity("bearing failure", spans)
        assert result is None

    def test_best_overlapping_entity_none_target(self):
        spans = [_resolved_span("corrosion", groups=["G4_MECHANISM_PROCESS"])]
        result = _best_overlapping_entity(None, spans)
        assert result is None

    def test_text_overlaps_any_true(self):
        spans = [_resolved_span("pump trip", groups=["G5_FAILURE_OUTCOME"])]
        assert _text_overlaps_any("pump trip was observed", spans) is True

    def test_text_overlaps_any_false(self):
        spans = [_resolved_span("valve leak", groups=["G5_FAILURE_OUTCOME"])]
        assert _text_overlaps_any("bearing failure", spans) is False

    def test_text_overlaps_any_empty_spans(self):
        assert _text_overlaps_any("any text", []) is False

    def test_has_gazetteer_source_true(self):
        spans = [_resolved_span("wear", groups=["G4_MECHANISM_PROCESS"], source_type="gazetteer_exact", score=0.90)]
        assert _has_gazetteer_source(spans, min_score=0.85) is True

    def test_has_gazetteer_source_below_threshold(self):
        spans = [_resolved_span("wear", groups=["G4_MECHANISM_PROCESS"], source_type="gazetteer_exact", score=0.80)]
        assert _has_gazetteer_source(spans, min_score=0.85) is False

    def test_has_gazetteer_source_wrong_type(self):
        spans = [_resolved_span("wear", groups=["G4_MECHANISM_PROCESS"], source_type="noun_chunk", score=0.95)]
        assert _has_gazetteer_source(spans, min_score=0.85) is False

    def test_has_gazetteer_source_fuzzy_counts(self):
        spans = [_resolved_span("fatigue", groups=["G4_MECHANISM_PROCESS"], source_type="gazetteer_fuzzy", score=0.87)]
        assert _has_gazetteer_source(spans, min_score=0.85) is True
