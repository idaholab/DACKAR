"""
Unit tests for dackar.RCA.ner.hybrid_ner.compatibility.CompatibilityEngine.

Pure group-compatibility resolution over CandidateSpan/SchemaIndex — no spaCy,
no network. Covers assign_groups, and the resolve_span decision policy:
no-label -> defer, single group -> accept, non-whitelisted multi-group ->
reduce to single best label (the whitelist/split behavior the reviewer named).
"""
from __future__ import annotations

from dackar.RCA.ner.hybrid_ner.compatibility import (
    CompatibilityEngine,
    CompatibilityResult,
)
from dackar.RCA.ner.hybrid_ner.models import (
    CandidateSpan,
    Decision,
    Document,
    LabelHypothesis,
)
from dackar.RCA.ner.hybrid_ner.schema import SchemaIndex


# A schema index with three labels mapping to three groups; the only
# whitelisted same-span pair is {G1_PHYSICAL_COMPONENT, G4_MECHANISM_PROCESS}.
_SCHEMA = SchemaIndex(
    label_to_group={
        "comp": "G1_PHYSICAL_COMPONENT",
        "deg_mech": "G4_MECHANISM_PROCESS",
        "outcome": "G5_FAILURE_OUTCOME",
    },
    group_to_labels={
        "G1_PHYSICAL_COMPONENT": {"comp"},
        "G4_MECHANISM_PROCESS": {"deg_mech"},
        "G5_FAILURE_OUTCOME": {"outcome"},
    },
    pair_decision={},
    conditional_rules=[],
)


def _doc(text="the pump failed during the run"):
    return Document(doc_id="D1", text=text)


def _cand(text, start, end, labels_scores):
    return CandidateSpan(
        span_id="c1", doc_id="D1", start=start, end=end, text=text,
        proposed_labels=[LabelHypothesis(label=l, score=s) for l, s in labels_scores],
    )


def _engine():
    return CompatibilityEngine()


# ---------------------------------------------------------------------------
# assign_groups
# ---------------------------------------------------------------------------

def test_assign_groups_fills_group_from_schema():
    cand = _cand("pump", 4, 8, [("comp", 0.9)])
    out = _engine().assign_groups([cand], _SCHEMA)
    assert out[0].proposed_labels[0].group == "G1_PHYSICAL_COMPONENT"


def test_assign_groups_preserves_explicit_group():
    cand = _cand("pump", 4, 8, [("comp", 0.9)])
    cand.proposed_labels[0].group = "ALREADY_SET"
    out = _engine().assign_groups([cand], _SCHEMA)
    assert out[0].proposed_labels[0].group == "ALREADY_SET"


# ---------------------------------------------------------------------------
# resolve_span decisions
# ---------------------------------------------------------------------------

def test_no_labels_defers():
    cand = CandidateSpan(span_id="c1", doc_id="D1", start=0, end=4, text="pump")
    dec, rels = _engine().resolve_span(_doc(), cand, _SCHEMA)
    assert dec.action == "defer"
    assert dec.output_spans == []


def test_single_group_accepts():
    cand = _cand("pump", 4, 8, [("comp", 0.9)])
    _engine().assign_groups([cand], _SCHEMA)
    dec, rels = _engine().resolve_span(_doc(), cand, _SCHEMA)
    assert dec.action == "accept"
    assert dec.output_spans[0].labels == ["comp"]


def test_g1_plus_outcome_splits_into_physical_and_outcome():
    """G1_PHYSICAL + G5_FAILURE_OUTCOME compound → R2 split, not reduce.

    A "<phys> failure" compound matches _try_split_physical_outcome, so the
    decision is a *split* yielding one component span and one outcome span.
    """
    cand = _cand("pump failure", 0, 12, [("comp", 0.6), ("outcome", 0.95)])
    _engine().assign_groups([cand], _SCHEMA)
    dec, rels = _engine().resolve_span(_doc("pump failure"), cand, _SCHEMA)
    assert dec.action == "split"
    assert [s.labels for s in dec.output_spans] == [["comp"], ["outcome"]]


def test_non_whitelisted_non_split_reduces_to_best_label():
    """G4_MECHANISM + G5_OUTCOME (no physical+outcome compound) with a clear
    score gap → reduce to the single best-scoring label (accept)."""
    cand = _cand("bearing wear event", 0, 18, [("deg_mech", 0.6), ("outcome", 0.95)])
    _engine().assign_groups([cand], _SCHEMA)
    dec, rels = _engine().resolve_span(_doc("bearing wear event"), cand, _SCHEMA)
    assert dec.action == "accept"
    assert dec.output_spans[0].labels == ["outcome"]  # higher score wins


def test_non_whitelisted_ambiguous_defers():
    """Same non-whitelisted pair but a near-tie (score gap < 0.15) → defer
    rather than force a collapse."""
    cand = _cand("bearing wear event", 0, 18, [("deg_mech", 0.85), ("outcome", 0.95)])
    _engine().assign_groups([cand], _SCHEMA)
    dec, rels = _engine().resolve_span(_doc("bearing wear event"), cand, _SCHEMA)
    assert dec.action == "defer"
    assert dec.output_spans == []


def test_best_label_per_group_keeps_highest_score():
    # two hypotheses in the SAME group → only the best is kept, single accept
    cand = _cand("pump", 4, 8, [("comp", 0.5), ("comp", 0.9)])
    _engine().assign_groups([cand], _SCHEMA)
    dec, rels = _engine().resolve_span(_doc(), cand, _SCHEMA)
    assert dec.action == "accept"
    assert dec.output_spans[0].labels == ["comp"]


# ---------------------------------------------------------------------------
# apply() end to end
# ---------------------------------------------------------------------------

def test_apply_returns_result_with_one_decision_per_candidate():
    cands = [
        _cand("pump", 4, 8, [("comp", 0.9)]),
        _cand("wear", 20, 24, [("deg_mech", 0.8)]),
    ]
    # distinct span_ids so decisions map 1:1
    cands[1].span_id = "c2"
    result = _engine().apply(_doc(), cands, _SCHEMA)
    assert isinstance(result, CompatibilityResult)
    assert len(result.decisions) == 2
    assert all(isinstance(d, Decision) for d in result.decisions)
