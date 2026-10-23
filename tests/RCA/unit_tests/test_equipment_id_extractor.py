"""
Unit tests for dackar.RCA.ner.equipment_ID_extractor.extract_equipment_ids.

Pure regex extraction — no spaCy model, no network. Covers the reviewer's
case-sensitivity point (equipment tags are matched case-sensitively by design,
see the module comment at :81), plus the normalize/unique/max_ids flags, the
false-positive filter (dates, ranges, doc-ref prefixes), and the containment
filter (a tag that is the trailing segment of a longer tag is dropped).
"""
from __future__ import annotations

from dackar.RCA.ner.equipment_ID_extractor import extract_equipment_ids


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------

def test_extracts_simple_tag():
    assert "P-101" in extract_equipment_ids("Pump P-101 tripped.")


def test_extracts_valve_and_instrument_tags():
    out = extract_equipment_ids("MOV-204A failed to stroke; PT-1102 reading high.")
    assert "MOV-204A" in out
    assert "PT-1102" in out


def test_empty_text_returns_empty():
    assert extract_equipment_ids("") == []
    assert extract_equipment_ids(None) == []  # guarded by `if not text`


# ---------------------------------------------------------------------------
# Case-sensitivity (reviewer's point; module comment at :81)
# ---------------------------------------------------------------------------

def test_uppercase_tag_matches():
    assert "P-101" in extract_equipment_ids("inspect P-101 today")


def test_lowercase_tag_not_matched():
    """Lowercase prose mentions are out of scope by design (no re.IGNORECASE)."""
    out = extract_equipment_ids("inspect p-101 and mov-204a today")
    assert out == []


def test_mixed_case_only_uppercase_portion():
    out = extract_equipment_ids("P-101 and p-102 both checked")
    assert "P-101" in out
    assert "P-102" not in out


# ---------------------------------------------------------------------------
# normalize / unique / max_ids flags
# ---------------------------------------------------------------------------

def test_normalize_uppercases_and_strips_punctuation():
    # trailing punctuation stripped, separators normalized
    out = extract_equipment_ids("valve (MOV-204A).")
    assert "MOV-204A" in out


def test_unique_preserves_first_seen_order():
    out = extract_equipment_ids("P-101, then PT-1102, then P-101 again")
    assert out == ["P-101", "PT-1102"]


def test_unique_false_keeps_duplicates():
    out = extract_equipment_ids("P-101 and P-101", unique=False)
    assert out.count("P-101") == 2


def test_max_ids_caps_collection():
    text = " ".join(f"PT-{i:04d}" for i in range(1, 20))
    out = extract_equipment_ids(text, max_ids=5)
    assert len(out) == 5


# ---------------------------------------------------------------------------
# False-positive filtering
# ---------------------------------------------------------------------------

def test_dates_and_ranges_rejected():
    out = extract_equipment_ids("window 10-15 on 2024-01 covering P-101")
    assert "10-15" not in out
    assert "2024-01" not in out
    assert "P-101" in out


def test_doc_ref_prefixes_rejected():
    """CR-/WO-/SOP- etc. are document refs, not equipment tags."""
    out = extract_equipment_ids("per CR-2026-00123 and WO-45101, replace P-101")
    assert not any(t.startswith(("CR-", "WO-")) for t in out)
    assert "P-101" in out


def test_single_letter_tiny_number_rejected():
    assert "A-1" not in extract_equipment_ids("item A-1 in the list")


# ---------------------------------------------------------------------------
# Containment filter — a trailing-segment tag is absorbed by the longer tag
# ---------------------------------------------------------------------------

def test_containment_drops_suffix_tag():
    out = extract_equipment_ids("AFW-P-101 isolated")
    assert "AFW-P-101" in out
    assert "P-101" not in out  # absorbed as a trailing segment


# ---------------------------------------------------------------------------
# Custom patterns
# ---------------------------------------------------------------------------

def test_custom_patterns_override_defaults():
    out = extract_equipment_ids("BKR9 and P-101", patterns=[r"\b(BKR\d+)\b"])
    assert out == ["BKR9"]  # only the custom pattern applies
