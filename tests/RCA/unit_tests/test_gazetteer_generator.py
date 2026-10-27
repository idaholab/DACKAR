"""
Unit tests for dackar.RCA.ner.hybrid_ner.generators.gazetteer_generator.

Builds a tiny real gazetteer Excel in a tmp dir (pandas + openpyxl), so the full
load + match path runs with no network. Golden-behavior coverage of exact-phrase
and fuzzy-token matching ahead of the Plan B perf refactor of the fuzzy matcher.
"""
from __future__ import annotations

import pandas as pd
import pytest

from dackar.RCA.ner.hybrid_ner.generators.gazetteer_generator import (
    GazetteerGenerator,
    GazetteerConfig,
)
from dackar.RCA.ner.hybrid_ner.models import Document


def _gazetteer_xlsx(tmp_path, columns):
    """columns: {header_with_[label]: [terms...]} -> path to an .xlsx."""
    p = tmp_path / "gaz.xlsx"
    # pad columns to equal length for a rectangular frame
    n = max(len(v) for v in columns.values())
    data = {k: (v + [None] * (n - len(v))) for k, v in columns.items()}
    pd.DataFrame(data).to_excel(p, index=False)
    return str(p)


def _doc(text):
    return Document(doc_id="D1", text=text)


# ---------------------------------------------------------------------------
# Gazetteer loading (header-bracket label parsing)
# ---------------------------------------------------------------------------

def test_load_parses_label_from_bracketed_header(tmp_path):
    path = _gazetteer_xlsx(tmp_path, {
        "Components [comp]": ["pump", "ball bearing"],
        "Mechanisms [deg_mech]": ["corrosion"],
    })
    g = GazetteerGenerator(path)
    assert g.label_terms["comp"] == {"pump", "ball bearing"}
    assert g.label_terms["deg_mech"] == {"corrosion"}


def test_load_header_without_brackets_uses_full_header(tmp_path):
    path = _gazetteer_xlsx(tmp_path, {"PlainLabel": ["widget"]})
    g = GazetteerGenerator(path)
    assert "PlainLabel" in g.label_terms


# ---------------------------------------------------------------------------
# Exact-phrase matching (default mode)
# ---------------------------------------------------------------------------

def test_exact_match_single_and_multiword(tmp_path):
    path = _gazetteer_xlsx(tmp_path, {"Components [comp]": ["pump", "ball bearing"]})
    g = GazetteerGenerator(path)
    out = g.generate(_doc("the pump and a ball bearing failed"))
    found = {(c.text, c.proposed_labels[0].label) for c in out}
    assert ("pump", "comp") in found
    assert ("ball bearing", "comp") in found


def test_exact_match_is_case_insensitive(tmp_path):
    path = _gazetteer_xlsx(tmp_path, {"Components [comp]": ["pump"]})
    g = GazetteerGenerator(path)
    out = g.generate(_doc("The PUMP tripped"))
    assert [c.text for c in out] == ["PUMP"]


def test_exact_no_partial_word_match(tmp_path):
    # "pump" must not match inside "pumping"
    path = _gazetteer_xlsx(tmp_path, {"Components [comp]": ["pump"]})
    g = GazetteerGenerator(path)
    out = g.generate(_doc("pumping station online"))
    assert out == []


def test_exact_match_offsets_are_correct(tmp_path):
    path = _gazetteer_xlsx(tmp_path, {"Components [comp]": ["pump"]})
    g = GazetteerGenerator(path)
    text = "the pump tripped"
    out = g.generate(_doc(text))
    c = out[0]
    assert text[c.start:c.end] == "pump"


def test_exact_match_provenance_and_hypothesis(tmp_path):
    path = _gazetteer_xlsx(tmp_path, {"Components [comp]": ["pump"]})
    g = GazetteerGenerator(path)
    c = g.generate(_doc("pump"))[0]
    assert c.proposed_labels[0].label == "comp"
    assert c.proposed_labels[0].rationale == "gazetteer_exact"
    assert any(s.source_type == "gazetteer" for s in c.sources)


# ---------------------------------------------------------------------------
# Fuzzy-token matching (the Plan B hotspot)
# ---------------------------------------------------------------------------

def test_fuzzy_matches_token_overlap(tmp_path):
    path = _gazetteer_xlsx(tmp_path, {"Components [comp]": ["ball bearing"]})
    g = GazetteerGenerator(path, config=GazetteerConfig(match_mode="fuzzy_tokens"))
    # exact token sequence present → high Jaccard → matched
    out = g.generate(_doc("the ball bearing assembly"))
    assert any("bearing" in c.text.lower() for c in out)


def test_fuzzy_no_match_below_threshold(tmp_path):
    path = _gazetteer_xlsx(tmp_path, {"Components [comp]": ["ball bearing seal kit"]})
    g = GazetteerGenerator(
        path, config=GazetteerConfig(match_mode="fuzzy_tokens", fuzzy_jaccard_threshold=0.9))
    # only one of four term tokens present → Jaccard well below 0.9 → no match
    out = g.generate(_doc("the pump tripped overnight"))
    assert out == []


def test_unknown_match_mode_raises(tmp_path):
    path = _gazetteer_xlsx(tmp_path, {"Components [comp]": ["pump"]})
    with pytest.raises(ValueError, match="Unknown match_mode"):
        GazetteerGenerator(path, config=GazetteerConfig(match_mode="nonsense"))


# ---------------------------------------------------------------------------
# Empty / no-match
# ---------------------------------------------------------------------------

def test_no_matches_empty_output(tmp_path):
    path = _gazetteer_xlsx(tmp_path, {"Components [comp]": ["pump"]})
    g = GazetteerGenerator(path)
    assert g.generate(_doc("nothing relevant here")) == []
