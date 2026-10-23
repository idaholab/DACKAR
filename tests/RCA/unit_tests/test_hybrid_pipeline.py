"""
Integration tests for dackar.RCA.ner.hybrid_ner.pipeline.HybridNERPipeline.

The pipeline wires generator -> consolidator -> (disambiguator/classifier no-ops)
-> compatibility -> postprocess. It is fully runnable with NO spaCy model and NO
embedder/LLM (desc_gen=None, llm_disambiguator=None), by injecting the pure
RegexCandidateGenerator. These tests exercise the real end-to-end flow with zero
mocking and assert candidates flow through to flattened entities.
"""
from __future__ import annotations

from dackar.RCA.ner.hybrid_ner.pipeline import HybridNERPipeline
from dackar.RCA.ner.hybrid_ner.generators.regex_generator import RegexCandidateGenerator
from dackar.RCA.ner.hybrid_ner.models import Document, PipelineResult
from dackar.RCA.ner.hybrid_ner.schema import SchemaIndex


_SCHEMA = SchemaIndex(
    label_to_group={"outcome": "G5_FAILURE_OUTCOME", "comp": "G1_PHYSICAL_COMPONENT"},
    group_to_labels={"G5_FAILURE_OUTCOME": {"outcome"}, "G1_PHYSICAL_COMPONENT": {"comp"}},
    pair_decision={},
    conditional_rules=[],
)


def _pipeline(patterns):
    return HybridNERPipeline(
        schema=_SCHEMA,
        generators=[RegexCandidateGenerator(patterns)],
    )


def _doc(text):
    return Document(doc_id="D1", text=text)


# ---------------------------------------------------------------------------
# End-to-end flow
# ---------------------------------------------------------------------------

def test_candidates_flow_to_entities():
    pipe = _pipeline([("fail", r"failed to start", "outcome"), ("pump", r"pump", "comp")])
    result = pipe.run(_doc("the pump failed to start"))
    assert isinstance(result, PipelineResult)
    labeled = {(e.text, tuple(e.labels)) for e in result.entities}
    assert ("pump", ("comp",)) in labeled
    assert ("failed to start", ("outcome",)) in labeled


def test_result_is_document_ordered():
    pipe = _pipeline([("pump", r"pump", "comp"), ("fail", r"failed to start", "outcome")])
    result = pipe.run(_doc("the pump failed to start"))
    starts = [e.start for e in result.entities]
    assert starts == sorted(starts)


def test_duplicate_candidates_consolidated():
    # two generators proposing the same span offsets → one entity after consolidation
    pipe = HybridNERPipeline(
        schema=_SCHEMA,
        generators=[
            RegexCandidateGenerator([("pump_a", r"pump", "comp")]),
            RegexCandidateGenerator([("pump_b", r"pump", "comp")]),
        ],
    )
    result = pipe.run(_doc("pump"))
    assert len([e for e in result.entities if e.text == "pump"]) == 1


def test_unlabeled_candidate_defers_to_no_entity():
    # a candidate with no label hypothesis → compatibility defers → not an entity
    pipe = _pipeline([("bare", r"widget", "")])
    result = pipe.run(_doc("a widget here"))
    assert result.entities == []
    # but the decision is still recorded (traceability)
    assert result.diagnostics["n_decisions"] == 1


def test_no_matches_empty_result():
    pipe = _pipeline([("fail", r"failed to start", "outcome")])
    result = pipe.run(_doc("everything nominal"))
    assert result.entities == []
    assert result.diagnostics["n_decisions"] == 0


def test_diagnostics_present():
    pipe = _pipeline([("pump", r"pump", "comp")])
    result = pipe.run(_doc("pump and pump again"))
    for key in ("n_decisions", "n_entities", "n_relations"):
        assert key in result.diagnostics
