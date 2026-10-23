"""
Unit tests for dackar.RCA.ner.hybrid_ner.consolidator.SpanConsolidator.

Pure span merging — no spaCy, no network. Covers dedupe by (start,end), source
+ label-hypothesis union, the policy toggle, and crucially the **no-mutation**
guarantee (the reviewer's point): consolidate() must not mutate the caller's
input CandidateSpan objects or their list fields.
"""
from __future__ import annotations

from dackar.RCA.ner.hybrid_ner.consolidator import (
    SpanConsolidator,
    SpanConsolidatorPolicy,
)
from dackar.RCA.ner.hybrid_ner.models import (
    CandidateSpan,
    Document,
    LabelHypothesis,
    SourceHit,
)


def _doc():
    return Document(doc_id="D1", text="pump shaft bearing failure")


def _span(span_id, start, end, text, labels=None, source=None):
    return CandidateSpan(
        span_id=span_id, doc_id="D1", start=start, end=end, text=text,
        sources=[SourceHit(source_type=source, source_id=f"{source}_1")] if source else [],
        proposed_labels=[LabelHypothesis(label=l) for l in (labels or [])],
    )


# ---------------------------------------------------------------------------
# Dedupe + merge
# ---------------------------------------------------------------------------

def test_identical_offsets_merged_to_one():
    a = _span("a", 0, 4, "pump", labels=["comp"])
    b = _span("b", 0, 4, "pump", labels=["comp"])
    out = SpanConsolidator().consolidate(_doc(), [a, b])
    assert len(out) == 1


def test_distinct_offsets_kept():
    a = _span("a", 0, 4, "pump")
    b = _span("b", 5, 10, "shaft")
    out = SpanConsolidator().consolidate(_doc(), [a, b])
    assert len(out) == 2


def test_label_hypotheses_unioned():
    a = _span("a", 0, 4, "pump", labels=["comp"])
    b = _span("b", 0, 4, "pump", labels=["outcome"])
    out = SpanConsolidator().consolidate(_doc(), [a, b])
    labels = {h.label for h in out[0].proposed_labels}
    assert labels == {"comp", "outcome"}


def test_sources_unioned():
    a = _span("a", 0, 4, "pump", source="regex")
    b = _span("b", 0, 4, "pump", source="gazetteer")
    out = SpanConsolidator().consolidate(_doc(), [a, b])
    gens = {s.source_type for s in out[0].sources}
    assert gens == {"regex", "gazetteer"}


# ---------------------------------------------------------------------------
# No-mutation guarantee (reviewer's point)
# ---------------------------------------------------------------------------

def test_input_spans_not_mutated():
    a = _span("a", 0, 4, "pump", labels=["comp"], source="regex")
    b = _span("b", 0, 4, "pump", labels=["outcome"], source="gazetteer")
    a_labels_before = list(a.proposed_labels)
    a_sources_before = list(a.sources)

    SpanConsolidator().consolidate(_doc(), [a, b])

    # the caller's original 'a' must be untouched — the merge happened on a copy
    assert a.proposed_labels == a_labels_before
    assert a.sources == a_sources_before
    assert len(a.proposed_labels) == 1
    assert len(a.sources) == 1


# ---------------------------------------------------------------------------
# Policy toggle
# ---------------------------------------------------------------------------

def test_dedupe_disabled_returns_input_unchanged():
    a = _span("a", 0, 4, "pump")
    b = _span("b", 0, 4, "pump")
    policy = SpanConsolidatorPolicy(dedupe_identical_spans=False)
    out = SpanConsolidator(policy).consolidate(_doc(), [a, b])
    assert out == [a, b]
