"""
Unit tests for dackar.RCA.ner.hybrid_ner.generators.anchored_np_generator.

Golden-behavior coverage for AnchoredNPGenerator, injecting a fake base
noun-phrase generator so no spaCy model / network is needed. These lock the
observable output (anchoring, punctuation/location trimming, min-token emission,
max_tokens, stop-token suppression) ahead of the Plan B perf refactor of
_has_any_token_evidence (regex compiled per token today).
"""
from __future__ import annotations

from dackar.RCA.ner.hybrid_ner.generators.anchored_np_generator import (
    AnchoredNPGenerator,
    STOP_TOKENS,
)
from dackar.RCA.ner.hybrid_ner.models import CandidateSpan, Document


class _FakeBaseNP:
    """Stand-in for NounPhraseGenerator: returns pre-built NP candidates."""

    def __init__(self, spans):
        self._spans = spans

    def generate(self, doc):
        return list(self._spans)


def _np(doc_id, start, end, text):
    return CandidateSpan(span_id=f"np-{start}-{end}", doc_id=doc_id,
                         start=start, end=end, text=text)


def _gen(doc_text, np_spans, token_index, **kw):
    doc = Document(doc_id="D1", text=doc_text)
    base = _FakeBaseNP(np_spans)
    gen = AnchoredNPGenerator(base_np=base, token_index=token_index, **kw)
    return gen.generate(doc), doc


# token_index maps token -> set of (label, term); only the keys matter here.
_TI = {"pump": {("comp", "pump")}, "bearing": {("comp", "bearing")},
       "shaft": {("comp", "shaft")}}


# ---------------------------------------------------------------------------
# Anchoring: emit only NP spans with gazetteer token evidence
# ---------------------------------------------------------------------------

def test_span_with_token_evidence_emitted():
    text = "the pump assembly"
    out, _ = _gen(text, [_np("D1", 4, 17, "pump assembly")], _TI, emit_min_tokens=False)
    assert any(c.text == "pump assembly" for c in out)


def test_span_without_token_evidence_dropped():
    text = "the widget assembly"
    out, _ = _gen(text, [_np("D1", 4, 19, "widget assembly")], _TI, emit_min_tokens=False)
    assert out == []


def test_no_token_index_emits_nothing():
    text = "the pump assembly"
    out, _ = _gen(text, [_np("D1", 4, 17, "pump assembly")], {}, emit_min_tokens=False)
    assert out == []


# ---------------------------------------------------------------------------
# Trimming: strong punctuation + trailing location tokens
# ---------------------------------------------------------------------------

def test_trim_at_strong_punctuation():
    text = "pump failed; bearing seized"
    out, _ = _gen(text, [_np("D1", 0, 27, "pump failed; bearing seized")], _TI,
                  emit_min_tokens=False)
    # trimmed at ';' -> "pump failed"
    assert any(c.text == "pump failed" for c in out)
    assert not any("bearing seized" in c.text for c in out)


def test_trailing_location_token_trimmed():
    text = "cam shaft region"
    out, _ = _gen(text, [_np("D1", 0, 16, "cam shaft region")], _TI,
                  emit_min_tokens=False)
    # "region" trailing location token removed -> "cam shaft"
    assert any(c.text == "cam shaft" for c in out)
    assert not any(c.text.endswith("region") for c in out)


# ---------------------------------------------------------------------------
# max_tokens cap + stop-token suppression
# ---------------------------------------------------------------------------

def test_max_tokens_cap_drops_long_spans():
    text = "the big heavy pump assembly unit"
    long_np = _np("D1", 0, len(text), text)  # 6 tokens
    out, _ = _gen(text, [long_np], _TI, max_tokens=4, emit_min_tokens=False)
    assert out == []


def test_stop_tokens_do_not_count_as_evidence():
    # "inspection"/"work"/"order" are STOP_TOKENS; a span of only those has no evidence
    text = "work order inspection"
    ti = {"work": {("x", "work")}, "order": {("x", "order")}}
    out, _ = _gen(text, [_np("D1", 0, 21, "work order inspection")], ti,
                  emit_min_tokens=False)
    assert out == []
    assert "work" in STOP_TOKENS and "inspection" in STOP_TOKENS


# ---------------------------------------------------------------------------
# Minimal token candidates
# ---------------------------------------------------------------------------

def test_min_tokens_emitted_for_indexed_tokens():
    text = "pump and bearing"
    out, _ = _gen(text, [_np("D1", 0, 16, "pump and bearing")], _TI,
                  emit_min_tokens=True)
    min_spans = [c for c in out if "_min_" in c.span_id]
    min_texts = {c.text.lower() for c in min_spans}
    assert "pump" in min_texts
    assert "bearing" in min_texts
    assert "and" not in min_texts  # not in token_index


def test_min_tokens_suppressed_when_flag_off():
    text = "pump and bearing"
    out, _ = _gen(text, [_np("D1", 0, 16, "pump and bearing")], _TI,
                  emit_min_tokens=False)
    assert not any("_min_" in c.span_id for c in out)


# ---------------------------------------------------------------------------
# Provenance + empty input
# ---------------------------------------------------------------------------

def test_anchored_span_carries_sourcehit():
    out, _ = _gen("pump assembly", [_np("D1", 0, 13, "pump assembly")], _TI,
                  emit_min_tokens=False)
    anchored = [c for c in out if c.text == "pump assembly"][0]
    assert any(s.source_type == "anchored_np" for s in anchored.sources)


def test_empty_base_candidates():
    out, _ = _gen("pump", [], _TI)
    assert out == []
