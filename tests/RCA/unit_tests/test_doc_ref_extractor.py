"""
Unit tests for dackar.RCA.ner.doc_ref_extractor.

Pure, profile-driven extraction — no network, no spaCy. Covers the reviewer's
year-regex boundary case (a year fragment outside [min_year, max_year] makes the
ref a false positive), the excluded-prefix filter, normalization, the
load_doc_ref_profile loader (tmp file + bundled default), and the
extract_doc_ref_ids wrapper.
"""
from __future__ import annotations

import json

import pytest

from dackar.RCA.ner.doc_ref_extractor import (
    DocRef,
    extract_doc_refs,
    extract_doc_ref_ids,
    load_doc_ref_profile,
)


_PROFILE = {
    "doc_ref_types": [
        {
            "type": "CR",
            "label": "doc_ref_cr",
            "patterns": [r"\bCR-(?:19|20)\d{2}-\d{3,8}\b", r"\bCR-\d{5,8}\b"],
        },
        {
            "type": "WO",
            "label": "doc_ref_wo",
            "patterns": [r"\bWO-\d{4,8}\b"],
        },
    ],
    "false_positive_filters": {
        "min_year": 1960,
        "max_year": 2035,
        "excluded_prefixes_from_doc_ref": ["SOP"],
    },
}


def _ids(text, **kw):
    return extract_doc_ref_ids(text, profile=_PROFILE, **kw)


# ---------------------------------------------------------------------------
# Basic extraction + DocRef shape
# ---------------------------------------------------------------------------

def test_extracts_cr_and_wo():
    refs = extract_doc_refs("see CR-2026-00123 and WO-45101", profile=_PROFILE)
    norms = {r.norm for r in refs}
    assert "CR-2026-00123" in norms
    assert "WO-45101" in norms


def test_docref_shape():
    refs = extract_doc_refs("CR-2026-00123", profile=_PROFILE)
    r = refs[0]
    assert isinstance(r, DocRef)
    assert r.doc_type == "CR"
    assert r.label == "doc_ref_cr"
    assert r.raw == "CR-2026-00123"
    assert r.norm == "CR-2026-00123"


def test_empty_text_returns_empty():
    assert extract_doc_refs("", profile=_PROFILE) == []


# ---------------------------------------------------------------------------
# Year-regex boundary (reviewer's point): year fragment must be in [1960, 2035]
# ---------------------------------------------------------------------------

def test_year_in_range_accepted():
    assert "CR-1999-00001" in _ids("ref CR-1999-00001")
    assert "CR-2035-00002" in _ids("ref CR-2035-00002")  # inclusive upper bound


def test_year_below_min_rejected():
    # 1959 < 1960 → false positive. (pattern uses (?:19|20)\d{2}, so 19xx parses)
    assert "CR-1959-00001" not in _ids("ref CR-1959-00001")


def test_year_above_max_rejected():
    assert "CR-2099-00001" not in _ids("ref CR-2099-00001")


def test_year_boundary_exact_edges():
    assert "CR-1960-00001" in _ids("CR-1960-00001")   # == min, kept
    assert "CR-2036-00001" not in _ids("CR-2036-00001")  # just over max, dropped


# ---------------------------------------------------------------------------
# Excluded prefixes + normalization + unique
# ---------------------------------------------------------------------------

def test_excluded_prefix_dropped():
    prof = {
        "doc_ref_types": [{"type": "SOP", "label": "doc_ref_sop",
                           "patterns": [r"\bSOP-\d{3,6}\b"]}],
        "false_positive_filters": {"excluded_prefixes_from_doc_ref": ["SOP"]},
    }
    assert extract_doc_ref_ids("per SOP-123", profile=prof) == []


def test_case_insensitive_normalized():
    assert _ids("cr-2026-00123 raised") == ["CR-2026-00123"]


def test_unique_dedupes_same_norm():
    out = _ids("CR-2026-00123 then CR-2026-00123 again")
    assert out == ["CR-2026-00123"]


def test_max_refs_caps():
    text = " ".join(f"WO-{i:05d}" for i in range(1, 20))
    refs = extract_doc_refs(text, profile=_PROFILE, max_refs=4)
    assert len(refs) == 4


# ---------------------------------------------------------------------------
# load_doc_ref_profile
# ---------------------------------------------------------------------------

def test_load_profile_from_tmp_file(tmp_path):
    p = tmp_path / "prof.json"
    p.write_text(json.dumps(_PROFILE), encoding="utf-8")
    loaded = load_doc_ref_profile(str(p))
    assert loaded["doc_ref_types"][0]["type"] == "CR"


def test_load_profile_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_doc_ref_profile(str(tmp_path / "nope.json"))


def test_load_profile_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_doc_ref_profile(str(bad))


def test_default_profile_loads_and_runs():
    out = extract_doc_ref_ids("tracked under CR-2026-00123")
    assert "CR-2026-00123" in out
