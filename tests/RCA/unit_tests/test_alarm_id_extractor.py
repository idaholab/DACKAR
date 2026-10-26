"""
Unit tests for dackar.RCA.ner.alarm_id_extractor.

Pure regex extraction driven by a plant-profile dict — no network, no spaCy.
Tests use an inline ``profile=`` (deterministic, no file I/O) for behavior and a
small smoke test against the bundled default profile to prove the real schema
loads and runs.
"""
from __future__ import annotations

from dackar.RCA.ner.alarm_id_extractor import (
    AlarmRef,
    extract_alarm_refs,
    extract_alarm_ref_ids,
)


# A minimal, self-contained plant profile with two alarm patterns of different
# confidence that match the SAME normalized id, plus an excluded prefix.
_PROFILE = {
    "alarm_patterns": [
        {"name": "short_alarm_id", "pattern": r"\bALM-\d{3,6}[A-Z]?\b", "score": 0.88},
        {"name": "annunciator", "pattern": r"\bANN-\d{1,4}\b", "score": 0.70},
        # deliberately lower-confidence duplicate matcher for ALM-1234
        {"name": "loose_alm", "pattern": r"\bALM-\d+\b", "score": 0.50},
    ],
    "false_positive_filters": {"excluded_prefixes_from_alarm": ["CR", "WO"]},
}


def _ids(text, **kw):
    return extract_alarm_ref_ids(text, profile=_PROFILE, **kw)


# ---------------------------------------------------------------------------
# Basic extraction + AlarmRef shape
# ---------------------------------------------------------------------------

def test_extracts_alarm_and_annunciator():
    refs = extract_alarm_refs("ALM-1234 and ANN-7 tripped", profile=_PROFILE)
    norms = {r.norm for r in refs}
    assert "ALM-1234" in norms
    assert "ANN-7" in norms


def test_alarmref_is_namedtuple_with_fields():
    refs = extract_alarm_refs("ALM-1234", profile=_PROFILE)
    assert isinstance(refs[0], AlarmRef)
    r = refs[0]
    assert r.pattern_name and r.raw == "ALM-1234" and r.norm == "ALM-1234"
    assert 0.0 <= r.score <= 1.0


def test_empty_text_returns_empty():
    assert extract_alarm_refs("", profile=_PROFILE) == []


# ---------------------------------------------------------------------------
# Case-insensitive matching + normalization (alarms, unlike equipment, use IGNORECASE)
# ---------------------------------------------------------------------------

def test_case_insensitive_and_normalized_uppercase():
    assert _ids("alm-1234 flagged") == ["ALM-1234"]


# ---------------------------------------------------------------------------
# unique keeps the HIGHEST-score pattern hit for a duplicate norm
# ---------------------------------------------------------------------------

def test_unique_keeps_highest_score_pattern():
    refs = extract_alarm_refs("ALM-1234", profile=_PROFILE, unique=True)
    alm = [r for r in refs if r.norm == "ALM-1234"]
    assert len(alm) == 1
    assert alm[0].pattern_name == "short_alarm_id"  # 0.88 beats loose_alm 0.50


def test_unique_false_keeps_both_pattern_hits():
    refs = extract_alarm_refs("ALM-1234", profile=_PROFILE, unique=False)
    names = {r.pattern_name for r in refs if r.norm == "ALM-1234"}
    assert {"short_alarm_id", "loose_alm"} <= names


# ---------------------------------------------------------------------------
# Filtering: excluded prefixes, min_score, max_ids
# ---------------------------------------------------------------------------

def test_excluded_prefix_dropped():
    # add a pattern that would match a CR- token, then confirm it's filtered
    prof = {
        "alarm_patterns": [{"name": "x", "pattern": r"\b[A-Z]{2}-\d+\b", "score": 0.9}],
        "false_positive_filters": {"excluded_prefixes_from_alarm": ["CR"]},
    }
    out = extract_alarm_ref_ids("CR-2026 and ALM-10 and ANN-3", profile=prof)
    assert "CR-2026" not in out


def test_min_score_filters_low_confidence():
    out = _ids("ANN-7 and ALM-1234", min_score=0.80)
    assert "ALM-1234" in out    # 0.88 kept
    assert "ANN-7" not in out   # 0.70 dropped


def test_max_ids_caps_results():
    text = " ".join(f"ALM-{i:04d}" for i in range(1, 20))
    refs = extract_alarm_refs(text, profile=_PROFILE, max_ids=3)
    assert len(refs) == 3


def test_ids_wrapper_returns_plain_strings():
    out = _ids("ALM-1234 and ANN-7")
    assert all(isinstance(s, str) for s in out)


# ---------------------------------------------------------------------------
# Smoke test against the bundled default profile (proves the real schema loads)
# ---------------------------------------------------------------------------

def test_default_profile_loads_and_runs():
    out = extract_alarm_ref_ids("ALM-1234 reported during the event")
    assert "ALM-1234" in out
