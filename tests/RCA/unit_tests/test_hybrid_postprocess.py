"""
Unit tests for dackar.RCA.ner.hybrid_ner.postprocess.Postprocessor.

Pure flattening/dedup over Decision/ResolvedSpan — no spaCy, no network. Covers:
only `accept` decisions contribute entities; exact duplicates are deduped
first-seen (the reviewer's "postprocess first-seen-wins" point); exact-text
nested spans sharing a group are pruned while distinct atomic nested spans are
kept; output is document-ordered; diagnostics counts are correct.
"""
from __future__ import annotations

from dackar.RCA.ner.hybrid_ner.models import (
    Decision,
    PipelineResult,
    ResolvedSpan,
)
from dackar.RCA.ner.hybrid_ner.postprocess import Postprocessor
from dackar.RCA.ner.hybrid_ner.schema import SchemaIndex

_SCHEMA = SchemaIndex(label_to_group={}, group_to_labels={}, pair_decision={}, conditional_rules=[])


def _rspan(span_id, start, end, text, labels, groups):
    return ResolvedSpan(
        span_id=span_id, doc_id="D1", start=start, end=end, text=text,
        labels=labels, groups=groups,
    )


def _decision(decision_id, action, output_spans):
    return Decision(
        decision_id=decision_id, doc_id="D1",
        input_span_ids=[decision_id], output_spans=output_spans, action=action,
    )


def _pp():
    return Postprocessor()


# ---------------------------------------------------------------------------
# Only accepted decisions contribute entities
# ---------------------------------------------------------------------------

def test_only_accept_decisions_flattened():
    accept = _decision("d1", "accept", [_rspan("s1", 0, 4, "pump", ["comp"], ["G1"])])
    reject = _decision("d2", "reject", [_rspan("s2", 5, 9, "junk", ["comp"], ["G1"])])
    defer = _decision("d3", "defer", [])
    result = _pp().apply("D1", [accept, reject, defer], [], _SCHEMA)
    assert isinstance(result, PipelineResult)
    texts = {e.text for e in result.entities}
    assert texts == {"pump"}


# ---------------------------------------------------------------------------
# First-seen-wins on exact duplicates (reviewer's point)
# ---------------------------------------------------------------------------

def test_exact_duplicate_deduped_first_seen():
    first = _rspan("first", 0, 4, "pump", ["comp"], ["G1"])
    dup = _rspan("second", 0, 4, "PUMP", ["comp"], ["G1"])  # same key (text_lower)
    d = _decision("d1", "accept", [first, dup])
    result = _pp().apply("D1", [d], [], _SCHEMA)
    assert len(result.entities) == 1
    assert result.entities[0].span_id == "first"  # first-seen kept


# ---------------------------------------------------------------------------
# Nested pruning: exact-text same-group dropped; distinct atomic kept
# ---------------------------------------------------------------------------

def test_exact_text_nested_same_group_dropped():
    outer = _rspan("outer", 0, 4, "pump", ["comp"], ["G1"])
    inner = _rspan("inner", 0, 4, "pump", ["comp"], ["G1"])  # identical text + group
    d = _decision("d1", "accept", [outer, inner])
    result = _pp().apply("D1", [d], [], _SCHEMA)
    assert len(result.entities) == 1


def test_distinct_atomic_nested_kept():
    outer = _rspan("outer", 0, 18, "pump shaft bearing", ["comp"], ["G1"])
    inner = _rspan("inner", 11, 18, "bearing", ["comp"], ["G1"])  # distinct text
    d = _decision("d1", "accept", [outer, inner])
    result = _pp().apply("D1", [d], [], _SCHEMA)
    texts = {e.text for e in result.entities}
    assert "pump shaft bearing" in texts
    assert "bearing" in texts  # atomic nested entity preserved


def test_nested_different_group_kept():
    outer = _rspan("outer", 0, 12, "pump failure", ["comp"], ["G1"])
    inner = _rspan("inner", 5, 12, "failure", ["outcome"], ["G5"])  # no shared group
    d = _decision("d1", "accept", [outer, inner])
    result = _pp().apply("D1", [d], [], _SCHEMA)
    assert len(result.entities) == 2


# ---------------------------------------------------------------------------
# Output ordering + diagnostics
# ---------------------------------------------------------------------------

def test_entities_returned_in_document_order():
    later = _rspan("b", 10, 14, "seal", ["comp"], ["G1"])
    earlier = _rspan("a", 0, 4, "pump", ["comp"], ["G1"])
    d = _decision("d1", "accept", [later, earlier])
    result = _pp().apply("D1", [d], [], _SCHEMA)
    assert [e.start for e in result.entities] == [0, 10]


def test_diagnostics_counts():
    d = _decision("d1", "accept", [_rspan("s1", 0, 4, "pump", ["comp"], ["G1"])])
    result = _pp().apply("D1", [d], [], _SCHEMA)
    assert result.diagnostics["n_decisions"] == 1
    assert result.diagnostics["n_entities"] == 1
    assert result.diagnostics["n_relations"] == 0
