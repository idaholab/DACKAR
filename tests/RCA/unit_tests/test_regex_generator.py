"""
Unit tests for dackar.RCA.ner.hybrid_ner.generators.regex_generator.

Pure regex candidate generation — no spaCy, no network. Covers offsets/text,
the optional per-pattern label hypothesis, case-insensitive matching, multiple
matches/patterns, and the SourceHit provenance.
"""
from __future__ import annotations

from dackar.RCA.ner.hybrid_ner.generators.regex_generator import RegexCandidateGenerator
from dackar.RCA.ner.hybrid_ner.models import Document


def _doc(text):
    return Document(doc_id="D1", text=text)


def test_generates_candidate_with_correct_offsets_and_text():
    gen = RegexCandidateGenerator([("fail_start", r"failed to start", "outcome")])
    out = gen.generate(_doc("the pump failed to start today"))
    assert len(out) == 1
    c = out[0]
    assert c.text == "failed to start"
    assert _doc("the pump failed to start today").text[c.start:c.end] == "failed to start"


def test_optional_label_attached():
    gen = RegexCandidateGenerator([("fail_start", r"failed to start", "outcome")])
    out = gen.generate(_doc("pump failed to start"))
    assert [h.label for h in out[0].proposed_labels] == ["outcome"]


def test_empty_label_attaches_no_hypothesis():
    gen = RegexCandidateGenerator([("trip", r"trip occurred", "")])
    out = gen.generate(_doc("a trip occurred overnight"))
    assert out[0].proposed_labels == []


def test_case_insensitive():
    gen = RegexCandidateGenerator([("fail_start", r"failed to start", "outcome")])
    out = gen.generate(_doc("PUMP FAILED TO START"))
    assert len(out) == 1
    assert out[0].text == "FAILED TO START"


def test_multiple_matches_and_patterns():
    gen = RegexCandidateGenerator([
        ("fail_start", r"failed to start", "outcome"),
        ("leak", r"leak", "outcome"),
    ])
    out = gen.generate(_doc("failed to start; later a leak then another leak"))
    texts = sorted(c.text for c in out)
    assert texts == ["failed to start", "leak", "leak"]


def test_no_match_returns_empty():
    gen = RegexCandidateGenerator([("fail_start", r"failed to start", "outcome")])
    assert gen.generate(_doc("everything nominal")) == []


def test_sourcehit_provenance_recorded():
    gen = RegexCandidateGenerator([("fail_start", r"failed to start", "outcome")])
    c = gen.generate(_doc("pump failed to start"))[0]
    assert len(c.sources) == 1
    assert c.sources[0].source_type == "regex"
    assert c.sources[0].source_id == "fail_start"
