"""
test_p2_common_cause_index_aug20.py — P-2 common-cause edge recognition.

The shared-support-dependency CCF signal previously matched only three exact edge-type
strings that the KG expansion does not emit, so the strongest CCF signal never fired
directly. P-2 replaces the brittle exact match with a semantic-family (case-insensitive
substring) match so support / functional-coupling relationships are recognised however
the KG names them, while pure containment (`has_part_usage`) is still excluded to avoid
over-firing CCF.

Run:  pytest test_p2_common_cause_index_aug20.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb",
             "langchain_community", "langchain_community.vectorstores",
             "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.causality_engine_v32 import RuleBasedCausalityEngineV32  # noqa: E402


def _engine():
    return RuleBasedCausalityEngineV32()


def _kg_with_edge(edge_type: str):
    return {
        "components": [
            {"component_id": "COMP-A"},
            {"component_id": "COMP-B"},
        ],
        "upstream_paths": [
            {
                "path_id": "P1",
                "nodes": ["COMP-A", "COMP-B"],
                "edges": [{"from_node": "COMP-A", "to_node": "COMP-B", "edge_type": edge_type}],
            }
        ],
    }


# ── predicate-level ────────────────────────────────────────────────────────

def test_is_support_edge_matches_family_variants():
    e = RuleBasedCausalityEngineV32
    for et in (
        "connected_support", "support_environment", "support_system", "SUPPORTS",
        "connects_port", "powered_by", "power_supply", "supplies_water",
        "cooling_supply", "cooled_by", "shared_header",
    ):
        assert e._is_support_dependency_edge(et) is True, et


def test_is_support_edge_excludes_containment_and_empty():
    e = RuleBasedCausalityEngineV32
    assert e._is_support_dependency_edge("has_part_usage") is False
    assert e._is_support_dependency_edge("") is False
    assert e._is_support_dependency_edge(None) is False
    assert e._is_support_dependency_edge("instance_of") is False


# ── index-level ────────────────────────────────────────────────────────────

def test_support_family_edge_populates_dependency_ids():
    idx = _engine()._build_common_cause_index(_kg_with_edge("powered_by"))
    assert idx["support_dependency_ids"] == {"COMP-A", "COMP-B"}


def test_legacy_exact_edge_still_recognized():
    idx = _engine()._build_common_cause_index(_kg_with_edge("connected_support"))
    assert idx["support_dependency_ids"] == {"COMP-A", "COMP-B"}


def test_connectivity_edge_recognized():
    idx = _engine()._build_common_cause_index(_kg_with_edge("connects_port"))
    assert idx["support_dependency_ids"] == {"COMP-A", "COMP-B"}


def test_containment_edge_does_not_populate_dependency_ids():
    # has_part_usage must NOT be treated as a shared dependency (avoids CCF over-fire),
    # but it still contributes to upstream adjacency.
    idx = _engine()._build_common_cause_index(_kg_with_edge("has_part_usage"))
    assert idx["support_dependency_ids"] == set()
    assert "COMP-B" in idx["upstream_adjacency"].get("COMP-A", set())
