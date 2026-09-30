"""
Unit tests for reliability_summarizer (RCA summarizers package).

These cover the correctness, provenance, and validation fixes from the PR #58
review: trusted-ChunkContext provenance stamping (blocking), fm_ids
serialization, compact-identifier doc-type detection, specificity-ranked doc
hints, object-strict / brace-aware JSON parsing, and the full-contract
validators sharing one field/type schema with the empty_*() skeletons.

The prompt/transport layer is not exercised (no Ollama, no network); the one
end-to-end case fakes ollama_generate_json via monkeypatch.
"""
from __future__ import annotations

import pytest

from dackar.RCA.summarizers.reliability_summarizer import (
    ChunkContext,
    NERSeed,
    detect_doc_type,
    empty_rca_frame,
    empty_retrieval_summary,
    flatten_rca_frame_for_embedding,
    flatten_retrieval_summary_for_embedding,
    summarize_with_retry,
    validate_rca_frame_json,
    validate_retrieval_summary_json,
    _apply_trusted_provenance,
    _extract_first_json_object,
    _parse_json_strict,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _ctx(**kw):
    base = dict(
        doc_id="D1", doc_type="CR", chunk_id="C1",
        section_path="7 Procedure > 7.2 Stroke Time",
        page_start=3, page_end=4,
    )
    base.update(kw)
    return ChunkContext(**base)


def _seed(**kw):
    base = dict(
        systems=[], equipment_ids=[], components=[], mechanisms=[], outcomes=[],
        surveillance_actions=[], maintenance_actions=[], properties=[], tools=[],
    )
    base.update(kw)
    return NERSeed(**base)


def _valid_retrieval_summary(ctx):
    rs = empty_retrieval_summary(ctx)
    rs["scope"] = "a sufficiently descriptive scope statement"
    rs["keywords_synonyms"] = ["seal", "leak", "stroke-time"]
    return rs


# ---------------------------------------------------------------------------
# #3 compact-identifier detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("CR12345", "CR"),
    ("WO98765", "WO"),
    ("OP1234", "SOP"),
    ("SOP1234", "SOP"),
    ("ECA0007", "ECA"),
])
def test_compact_ids_detected(name, expected):
    # trailing boundary is after the digits, so no separator is required
    assert detect_doc_type(name, None, "") == expected


@pytest.mark.parametrize("name,expected", [
    ("CR-12345", "CR"),
    ("WO_98765", "WO"),
    ("SOP-101 Valve Procedure", "SOP"),
])
def test_separated_ids_still_detected(name, expected):
    assert detect_doc_type(name, None, "") == expected


# ---------------------------------------------------------------------------
# #4 specificity-ranked doc-type hints
# ---------------------------------------------------------------------------

def test_specific_phrase_beats_generic_word():
    # "root cause" (ECA) must win over the generic "procedure" (SOP)
    assert detect_doc_type(None, None, "the procedure identified a root cause") == "ECA"


def test_longer_sop_phrase_wins():
    assert detect_doc_type(None, None, "this standard operating procedure applies") == "SOP"


def test_no_hint_is_other():
    assert detect_doc_type(None, None, "unrelated narrative text") == "OTHER"


# ---------------------------------------------------------------------------
# #2 fm_ids serialization
# ---------------------------------------------------------------------------

def test_to_json_includes_fm_ids():
    assert _seed(fm_ids=["FM-1", "FM-2"]).to_json()["fm_ids"] == ["FM-1", "FM-2"]


def test_to_json_covers_all_public_fields():
    j = _seed().to_json()
    for k in (
        "systems", "equipment_ids", "components", "mechanisms", "outcomes",
        "surveillance_actions", "maintenance_actions", "properties", "tools",
        "fm_ids", "doc_refs", "alarm_ids", "measurements", "temporal_refs",
        "temporal_relations", "temporal_qualifiers", "locations", "conjectures",
    ):
        assert k in j and isinstance(j[k], list)


# ---------------------------------------------------------------------------
# #6 object-strict parsing / #7 brace-aware recovery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ['[{"a": 1}]', '"a string"', '42', 'true', 'null'])
def test_non_object_json_rejected(bad):
    # a valid non-object must be rejected, not mined for a nested object
    with pytest.raises(ValueError):
        _parse_json_strict(bad)


def test_empty_output_rejected():
    with pytest.raises(ValueError):
        _parse_json_strict("")


def test_first_object_recovered_amid_extra_text():
    # greedy first-to-last brace would span both objects and fail with
    # "Extra data"; raw_decode recovers the first standalone object
    assert _parse_json_strict('prefix {"a": 1} suffix {"b": 2}') == {"a": 1}


def test_direct_object_parsed():
    assert _parse_json_strict('{"x": 10, "y": [1, 2]}') == {"x": 10, "y": [1, 2]}


def test_extract_first_json_object_helper():
    assert _extract_first_json_object("no json here") is None
    assert _extract_first_json_object('junk {"ok": true} tail') == {"ok": True}


# ---------------------------------------------------------------------------
# #1 trusted-provenance stamping (blocking)
# ---------------------------------------------------------------------------

def test_apply_trusted_provenance_overrides_forged():
    ctx = _ctx()
    forged = {
        "chunk_id": "FAKE", "doc_type": "SOP", "view_type": "rca_frame",
        "citations": {"doc_id": "EVIL", "section_path": "x",
                      "page_start": 999, "page_end": 999},
    }
    _apply_trusted_provenance(forged, ctx, "retrieval_summary")
    assert forged["chunk_id"] == "C1"
    assert forged["doc_type"] == "CR"
    assert forged["view_type"] == "retrieval_summary"
    assert forged["citations"] == {
        "doc_id": "D1", "section_path": "7 Procedure > 7.2 Stroke Time",
        "page_start": 3, "page_end": 4,
    }


def test_summarize_stamps_provenance_over_model(monkeypatch):
    ctx = _ctx()
    forged = _valid_retrieval_summary(ctx)
    forged["chunk_id"] = "HALLUCINATED"
    forged["citations"] = {"doc_id": "WRONG", "section_path": "z",
                           "page_start": 1, "page_end": 1}
    monkeypatch.setattr(
        "dackar.RCA.summarizers.reliability_summarizer.ollama_generate_json",
        lambda *a, **k: dict(forged),
    )
    out = summarize_with_retry(
        "CR", "retrieval_summary", ctx, _seed(), "chunk text", sleep_sec=0,
    )
    assert out["chunk_id"] == "C1"
    assert out["citations"]["doc_id"] == "D1"
    assert out["citations"]["page_start"] == 3


# ---------------------------------------------------------------------------
# #5 comprehensive validators + nested citations
# ---------------------------------------------------------------------------

def test_retrieval_summary_skeleton_validates():
    assert validate_retrieval_summary_json(_valid_retrieval_summary(_ctx())) == []


def test_retrieval_summary_flags_wrong_types_and_missing():
    bad = _valid_retrieval_summary(_ctx())
    bad["keywords_synonyms"] = "not-a-list"
    bad["citations"]["page_start"] = "3"
    del bad["mechanisms"]
    flags = validate_retrieval_summary_json(bad)
    assert "keywords_synonyms_not_list" in flags
    assert "citations_page_start_wrong_type" in flags
    assert "missing_mechanisms" in flags


def test_retrieval_summary_flags_non_object_citations():
    bad = _valid_retrieval_summary(_ctx())
    bad["citations"] = "nope"
    assert "citations_not_object" in validate_retrieval_summary_json(bad)


def test_rca_frame_skeleton_validates():
    assert validate_rca_frame_json(empty_rca_frame(_ctx())) == []


def test_rca_frame_flags_missing_list_and_bad_citation():
    bad = empty_rca_frame(_ctx())
    del bad["hypotheses"]
    bad["citations"]["page_end"] = "4"
    flags = validate_rca_frame_json(bad)
    assert "missing_hypotheses" in flags
    assert "citations_page_end_wrong_type" in flags


# ---------------------------------------------------------------------------
# #6 embedding flatteners drop empty fields (PR #58 line-667 nit)
# ---------------------------------------------------------------------------

def test_flatten_retrieval_summary_drops_empty_fields():
    summary = {
        "scope": "",
        "entities": {"systems": ["RCS"], "equipment_ids": [], "components": []},
        "symptoms_outcomes": [],
        "keywords_synonyms": ["packing"],
    }
    out = flatten_retrieval_summary_for_embedding(summary)
    assert out == "SYSTEMS: RCS\nKEYWORDS: packing"
    # no bare "LABEL: " lines from empty sections
    assert "EQUIPMENT:" not in out
    assert "SCOPE:" not in out
    # a None scope is dropped, never rendered as "SCOPE: None"
    assert flatten_retrieval_summary_for_embedding({"scope": None}) == ""
    assert flatten_retrieval_summary_for_embedding({}) == ""


def test_flatten_retrieval_summary_keeps_populated_fields():
    summary = {
        "scope": "pump seal leak",
        "entities": {"systems": ["RCS"], "equipment_ids": ["P-1A"], "components": ["seal"]},
        "symptoms_outcomes": ["leak"],
        "mechanisms": ["wear"],
        "diagnostics": ["visual"],
        "corrective_actions": ["replace"],
        "numbers_limits": ["<1gpm"],
        "keywords_synonyms": ["packing"],
    }
    assert flatten_retrieval_summary_for_embedding(summary) == "\n".join([
        "SCOPE: pump seal leak",
        "SYSTEMS: RCS",
        "EQUIPMENT: P-1A",
        "COMPONENTS: seal",
        "SYMPTOMS/OUTCOMES: leak",
        "MECHANISMS: wear",
        "DIAGNOSTICS: visual",
        "ACTIONS: replace",
        "NUMBERS/LIMITS: <1gpm",
        "KEYWORDS: packing",
    ])


def test_flatten_rca_frame_drops_empty_fields():
    out = flatten_rca_frame_for_embedding({"observed": ["a", "b"], "hypotheses": []})
    assert out == "OBSERVED: a; b"
    assert "HYPOTHESES:" not in out
    assert flatten_rca_frame_for_embedding({}) == ""


def test_flatten_rca_frame_keeps_populated_fields():
    rca = {
        "observed": ["a", "b"],
        "hypotheses": ["h1"],
        "tests_to_confirm": ["t1"],
        "candidate_actions": ["c1"],
        "constraints": ["k1"],
    }
    assert flatten_rca_frame_for_embedding(rca) == "\n".join([
        "OBSERVED: a; b",
        "HYPOTHESES: h1",
        "TESTS: t1",
        "ACTIONS: c1",
        "CONSTRAINTS: k1",
    ])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
