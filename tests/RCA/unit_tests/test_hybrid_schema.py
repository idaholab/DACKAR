"""
Unit tests for dackar.RCA.ner.hybrid_ner.schema.SchemaLoader / SchemaIndex.

Pure JSON → index loading. Covers both accepted schema formats, the
group-pair-matrix (stored both directions), conditional-rule parsing + priority
sort, and the loader guards that raise ValueError on missing required keys
(the reviewer's "schema-loader guards" point).
"""
from __future__ import annotations

import json

import pytest

from dackar.RCA.ner.hybrid_ner.schema import (
    ConditionalRule,
    SchemaIndex,
    SchemaLoader,
)


def _load(tmp_path, obj) -> SchemaIndex:
    p = tmp_path / "schema.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return SchemaLoader.load(str(p))


# ---------------------------------------------------------------------------
# Format B — {"groups": [{id, labels}]}
# ---------------------------------------------------------------------------

def test_format_b_groups_list(tmp_path):
    idx = _load(tmp_path, {"groups": [
        {"id": "G1_PHYSICAL_COMPONENT", "labels": ["comp_mech_spec", "comp"]},
        {"id": "G5_MECHANISMS", "labels": ["deg_mech"]},
    ]})
    assert idx.label_to_group["deg_mech"] == "G5_MECHANISMS"
    assert idx.label_to_group["comp"] == "G1_PHYSICAL_COMPONENT"
    assert idx.group_to_labels["G1_PHYSICAL_COMPONENT"] == {"comp_mech_spec", "comp"}


def test_format_b_group_missing_id_raises(tmp_path):
    with pytest.raises(ValueError, match="missing required 'id'"):
        _load(tmp_path, {"groups": [{"labels": ["x"]}]})


# ---------------------------------------------------------------------------
# Format A — label-keyed dict
# ---------------------------------------------------------------------------

def test_format_a_label_keyed(tmp_path):
    idx = _load(tmp_path, {
        "deg_mech": {"group": "G4_MECHANISM_PROCESS"},
        "comp_mech_spec": {"group": "G1_PHYSICAL_COMPONENT"},
    })
    assert idx.label_to_group["deg_mech"] == "G4_MECHANISM_PROCESS"
    assert "comp_mech_spec" in idx.group_to_labels["G1_PHYSICAL_COMPONENT"]


def test_format_a_entry_without_group_skipped(tmp_path):
    idx = _load(tmp_path, {
        "deg_mech": {"group": "G4"},
        "noise": {"description": "no group key here"},
    })
    assert "deg_mech" in idx.label_to_group
    assert "noise" not in idx.label_to_group


# ---------------------------------------------------------------------------
# Group-pair matrix — stored both directions
# ---------------------------------------------------------------------------

def test_pair_decision_both_directions(tmp_path):
    idx = _load(tmp_path, {
        "groups": [{"id": "G1", "labels": ["a"]}, {"id": "G5", "labels": ["b"]}],
        "group_pair_matrix": {"pairs": [{"g1": "G1", "g2": "G5", "decision": "C"}]},
    })
    assert idx.pair_decision[("G1", "G5")] == "C"
    assert idx.pair_decision[("G5", "G1")] == "C"  # order-insensitive


def test_pair_missing_key_raises(tmp_path):
    with pytest.raises(ValueError, match="group_pair_matrix pair missing"):
        _load(tmp_path, {
            "groups": [{"id": "G1", "labels": ["a"]}],
            "group_pair_matrix": {"pairs": [{"g1": "G1", "decision": "C"}]},
        })


# ---------------------------------------------------------------------------
# Conditional rules — parsed + sorted by descending priority
# ---------------------------------------------------------------------------

def test_conditional_rules_sorted_by_priority(tmp_path):
    idx = _load(tmp_path, {
        "groups": [{"id": "G1", "labels": ["a"]}],
        "conditional_rules": [
            {"id": "low", "priority": 1, "applies_to_pairs": [{"g1": "G1", "g2": "G1"}]},
            {"id": "high", "priority": 10, "decision_override": "D"},
        ],
    })
    assert [r.rule_id for r in idx.conditional_rules] == ["high", "low"]
    assert isinstance(idx.conditional_rules[0], ConditionalRule)
    assert idx.conditional_rules[1].applies_to_pairs == [("G1", "G1")]


def test_conditional_rule_missing_id_raises(tmp_path):
    with pytest.raises(ValueError, match="conditional_rule missing required 'id'"):
        _load(tmp_path, {
            "groups": [{"id": "G1", "labels": ["a"]}],
            "conditional_rules": [{"priority": 5}],
        })


# ---------------------------------------------------------------------------
# Returned object is a SchemaIndex
# ---------------------------------------------------------------------------

def test_returns_schema_index(tmp_path):
    idx = _load(tmp_path, {"groups": [{"id": "G1", "labels": ["a"]}]})
    assert isinstance(idx, SchemaIndex)
