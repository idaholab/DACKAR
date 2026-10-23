"""
Unit tests for dackar.RCA.ner.ner_adapter routing helpers, plus an
importorskip-gated smoke of SpacyAnnotator.

_route_entity / _uniq are pure and tested directly (no model). The spaCy
annotator needs en_core_web_sm + the custom DACKAR pipes, so its smoke is
skipped cleanly when the model is absent.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from dackar.RCA.ner.ner_adapter import _route_entity, _uniq


def _schema(label_to_group=None):
    return SimpleNamespace(label_to_group=label_to_group or {})


# ---------------------------------------------------------------------------
# _uniq — case-insensitive, first-seen order, strips blanks
# ---------------------------------------------------------------------------

def test_uniq_dedupes_case_insensitive_first_seen():
    assert _uniq(["Pump", "pump", "Valve", "PUMP", "valve"]) == ["Pump", "Valve"]


def test_uniq_strips_and_drops_blanks():
    assert _uniq(["  a ", "", "   ", "b"]) == ["a", "b"]


# ---------------------------------------------------------------------------
# _route_entity — explicit label-level routing
# ---------------------------------------------------------------------------

def test_route_system_labels():
    assert _route_entity(_schema(), ["syst"], []) == "systems"
    assert _route_entity(_schema(), ["plant"], []) == "systems"


def test_route_mechanism_label():
    assert _route_entity(_schema(), ["deg_mech"], []) == "mechanisms"


def test_route_outcome_labels():
    for lbl in ("event", "fail_type_n", "fail_type_v"):
        assert _route_entity(_schema(), [lbl], []) == "outcomes"


def test_route_action_and_tool_labels():
    assert _route_entity(_schema(), ["mnt_ops"], []) == "maintenance_actions"
    assert _route_entity(_schema(), ["surv_ops_v"], []) == "surveillance_actions"
    assert _route_entity(_schema(), ["mnt_tool"], []) == "tools"
    assert _route_entity(_schema(), ["prop"], []) == "properties"


def test_route_component_label_prefix():
    assert _route_entity(_schema(), ["comp_mech_spec"], []) == "components"
    assert _route_entity(_schema(), ["ast_pump"], []) == "components"


# ---------------------------------------------------------------------------
# _route_entity — group-level fallback + label→group derivation via schema
# ---------------------------------------------------------------------------

def test_route_group_fallback():
    assert _route_entity(_schema(), [], ["G1_PHYSICAL_COMPONENT"]) == "components"
    assert _route_entity(_schema(), [], ["G4_MECHANISM_PROCESS"]) == "mechanisms"
    assert _route_entity(_schema(), [], ["G5_FAILURE_OUTCOME"]) == "outcomes"


def test_route_derives_group_from_schema_when_group_absent():
    # label not in the explicit routing table, but schema maps it to a G4 group
    schema = _schema({"weird_mech_label": "G4_MECHANISM_PROCESS"})
    assert _route_entity(schema, ["weird_mech_label"], []) == "mechanisms"


def test_route_unknown_returns_none():
    assert _route_entity(_schema(), ["totally_unknown"], []) is None
    assert _route_entity(_schema(), [], []) is None


def test_route_label_level_beats_group_fallback():
    # explicit mechanism label wins even if a component group is also present
    assert _route_entity(_schema(), ["deg_mech"], ["G1_PHYSICAL_COMPONENT"]) == "mechanisms"


# ---------------------------------------------------------------------------
# SpacyAnnotator smoke — gated on the model being installed
# ---------------------------------------------------------------------------

def test_spacy_annotator_smoke():
    pytest.importorskip("en_core_web_sm", reason="spaCy model en_core_web_sm not installed")
    from dackar.RCA.ner.spacy_annotator import SpacyAnnotator, SpacyAnnotationResult

    annot = SpacyAnnotator()
    result = annot.annotate("The pump was inspected after 48 hours near the valve.")
    assert isinstance(result, SpacyAnnotationResult)
    # buckets exist and are lists
    assert isinstance(result.temporal_refs, list)
    assert isinstance(result.measurements, list)


def test_spacy_annotator_empty_text_returns_empty_result():
    pytest.importorskip("en_core_web_sm", reason="spaCy model en_core_web_sm not installed")
    from dackar.RCA.ner.spacy_annotator import SpacyAnnotator, SpacyAnnotationResult

    annot = SpacyAnnotator()
    result = annot.annotate("")
    assert isinstance(result, SpacyAnnotationResult)
    assert result.temporal_refs == []
