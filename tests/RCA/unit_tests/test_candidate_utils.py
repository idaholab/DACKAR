"""
Unit tests for dackar.RCA.ner.hybrid_ner.generators.candidate_utils.

Pure span manipulation over CandidateSpan dataclasses — no spaCy, no network.
Covers dedupe (normalize by (start,end,text_lower), prefer-labeled, substring
absorption, sort order), component/mechanism splitting, and a determinism +
mechanism-preservation safety net for the dormant split_multi_label_spans
(Plan C) so later changes to that function have a regression anchor.
"""
from __future__ import annotations

from dackar.RCA.ner.hybrid_ner.models import CandidateSpan, LabelHypothesis
from dackar.RCA.ner.hybrid_ner.generators.candidate_utils import (
    dedupe_candidate_spans,
    split_component_mechanism_spans,
    split_multi_label_spans,
)


def _span(span_id, start, end, text, labels=None):
    return CandidateSpan(
        span_id=span_id, doc_id="D1", start=start, end=end, text=text,
        proposed_labels=[LabelHypothesis(label=l) for l in (labels or [])],
    )


# ---------------------------------------------------------------------------
# dedupe_candidate_spans
# ---------------------------------------------------------------------------

def test_dedupe_identical_key_keeps_one():
    a = _span("a", 0, 5, "P-101")
    b = _span("b", 0, 5, "p-101")  # same (start,end,text_lower)
    out = dedupe_candidate_spans([a, b])
    assert len(out) == 1


def test_dedupe_prefers_labeled_over_unlabeled():
    unlabeled = _span("u", 0, 5, "P-101")
    labeled = _span("l", 0, 5, "P-101", labels=["comp"])
    out = dedupe_candidate_spans([unlabeled, labeled])
    assert len(out) == 1
    assert out[0].span_id == "l"  # labeled wins regardless of order


def test_dedupe_absorbs_substring_with_same_labels():
    # "P-101" (2..7) is contained in "AFW-P-101" (0..9) with identical (empty) labels
    inner = _span("inner", 2, 7, "P-101")
    outer = _span("outer", 0, 9, "AFW-P-101")
    out = dedupe_candidate_spans([inner, outer])
    ids = {c.span_id for c in out}
    assert "outer" in ids
    assert "inner" not in ids  # absorbed


def test_dedupe_keeps_distinct_non_overlapping():
    a = _span("a", 0, 5, "P-101")
    b = _span("b", 10, 16, "MOV-20")
    out = dedupe_candidate_spans([a, b])
    assert len(out) == 2


def test_dedupe_sorted_by_start_then_longer_first():
    a = _span("a", 10, 14, "xxxx")
    b = _span("b", 0, 3, "yyy")
    out = dedupe_candidate_spans([a, b])
    assert [c.start for c in out] == [0, 10]


# ---------------------------------------------------------------------------
# split_component_mechanism_spans
# ---------------------------------------------------------------------------

def test_split_mixed_component_mechanism_phrase():
    # "pump corrosion event": comp + mech, >2 tokens, not a protected phrase.
    c = _span("c", 0, 20, "pump corrosion event")
    out = split_component_mechanism_spans(
        [c], component_tokens={"pump"}, mechanism_tokens={"corrosion"},
    )
    # produced splits reference the original span id, and keep BOTH halves
    assert any("_split_comp_" in s.span_id for s in out)
    assert any("_split_mech_" in s.span_id for s in out)
    assert {s.text for s in out} == {"pump", "corrosion"}


def test_split_protected_phrase_kept_intact():
    c = _span("c", 0, 12, "seal leakage observed")
    out = split_component_mechanism_spans(
        [c], component_tokens={"seal"}, mechanism_tokens={"leakage"},
    )
    assert out == [c]  # protected phrase → untouched


def test_split_short_span_kept_intact():
    c = _span("c", 0, 10, "shaft wear")  # only 2 tokens
    out = split_component_mechanism_spans(
        [c], component_tokens={"shaft"}, mechanism_tokens={"wear"},
    )
    assert out == [c]


def test_split_no_mechanism_token_kept_intact():
    c = _span("c", 0, 20, "main pump assembly unit")
    out = split_component_mechanism_spans(
        [c], component_tokens={"pump"}, mechanism_tokens={"corrosion"},
    )
    assert out == [c]


# ---------------------------------------------------------------------------
# split_multi_label_spans — Plan C safety net (dormant feature)
# ---------------------------------------------------------------------------

def test_multi_label_single_label_unchanged():
    c = _span("c", 0, 10, "bearing wear", labels=["comp"])
    assert split_multi_label_spans([c]) == [c]


def test_multi_label_short_span_kept():
    c = _span("c", 0, 10, "bearing wear", labels=["comp", "deg"])  # 2 tokens
    assert split_multi_label_spans([c]) == [c]


def test_multi_label_split_emits_both_roles_exact():
    """Plan C fix: a multi-component + mechanism phrase splits into BOTH a
    component span and a mechanism span, with the earliest-in-text keyword chosen
    per role (so the result is identical under any PYTHONHASHSEED).

    For "bearing shaft seal corrosion event" the leftmost component keyword is
    "bearing" (offset 0) and the only mechanism keyword is "corrosion" (offset
    19), so the exact output is pinned.
    """
    c = _span("c", 0, 34, "bearing shaft seal corrosion event", labels=["comp", "deg"])
    out = [(s.span_id, s.start, s.end, s.text) for s in split_multi_label_spans([c])]
    assert out == [
        ("c_split_bearing_comp", 0, 7, "bearing"),
        ("c_split_corrosion_deg", 19, 28, "corrosion"),
    ]


def test_multi_label_split_role_labels_partitioned():
    """Each emitted span carries only its own role's labels."""
    c = _span("c", 0, 34, "bearing shaft seal corrosion event", labels=["comp", "deg"])
    out = split_multi_label_spans([c])
    by_role = {s.span_id.rsplit("_", 1)[1]: s for s in out}
    assert [h.label for h in by_role["comp"].proposed_labels] == ["comp"]
    assert [h.label for h in by_role["deg"].proposed_labels] == ["deg"]


def test_multi_label_split_component_choice_is_leftmost_not_seed_dependent():
    """The component span is the leftmost component keyword regardless of the
    order keywords are stored in the internal set (hash-seed independent)."""
    c = _span("c", 0, 34, "bearing shaft seal corrosion event", labels=["comp", "deg"])
    comp = [s for s in split_multi_label_spans([c]) if s.span_id.endswith("_comp")]
    assert len(comp) == 1
    assert comp[0].text == "bearing"  # offset 0 beats shaft(8)/seal(14)
