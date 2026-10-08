"""Database-free tests for the schema-registry load lifecycle on ``KG``.

Finding 12: ``loadPredefinedGraphSchemas`` looped ``importGraphSchema``,
mutating ``self.graphSchemas`` entry by entry; a duplicate name part-way
through (e.g. one schema already loaded individually) left a partial load, and
a second whole-set load failed outright. Finding 10: a single import committed
the candidate before any whole-set check, so a misspelled endpoint or duplicate
node was retained as valid state until a later action.

The remedy is a staged batch API (``importGraphSchemas``): parse and
per-file-validate into a temporary candidate, cross-check the whole candidate,
and replace session state only once every check passes. These tests pin that it
loads atomically (no partial state on failure), idempotently (repeat load
rebuilds the same set), still supports circular cross-schema references, and
rejects a batch whose endpoints do not resolve.

Exercised on a ``KG.__new__`` instance so no Neo4j driver is opened; only the
attributes the import path reads (``baseSchema``, ``datatypes``,
``graphSchemas``) are populated.
"""
import json
from pathlib import Path

import pytest

from dackar.knowledge_graph.KGconstruction import KG
from dackar.knowledge_graph.schema_types import ALLOWED_SCHEMA_TYPES

SCHEMA_DIR = (
    Path(__file__).resolve().parents[2]
    / "src" / "dackar" / "knowledge_graph" / "schemas"
)


def _bare_kg():
    kg = KG.__new__(KG)  # bypass __init__ (no Neo4j driver, no xlsx load)
    kg.datatypes = list(ALLOWED_SCHEMA_TYPES)
    with open(SCHEMA_DIR / "baseSchema.json", "r", encoding="utf-8") as f:
        kg.baseSchema = json.load(f)
    kg.graphSchemas = {}
    return kg


def _write(tmp_path, name, body):
    p = tmp_path / f"{name}.toml"
    p.write_text(body, encoding="utf-8")
    return str(p)


# Two schemas with circular cross-references (a's relation points at b's node
# and vice versa), so neither resolves alone but the pair resolves as a set.
_A = """
title = "A"
version = "1.0"
[node.a_node]
node_description = "A"
node_properties = [{ name = "id", type = "string", optional = false, description = "id" }]
[relation.a_to_b]
relation_description = "a to b"
from_entity = "a_node"
to_entity = "b_node"
"""
_B = """
title = "B"
version = "1.0"
[node.b_node]
node_description = "B"
node_properties = [{ name = "id", type = "string", optional = false, description = "id" }]
[relation.b_to_a]
relation_description = "b to a"
from_entity = "b_node"
to_entity = "a_node"
"""
# A schema whose relation endpoint resolves to no defined node anywhere.
_BAD_ENDPOINT = """
title = "Bad"
version = "1.0"
[node.c_node]
node_description = "C"
node_properties = [{ name = "id", type = "string", optional = false, description = "id" }]
[relation.c_to_missing]
relation_description = "c to a non-existent node"
from_entity = "c_node"
to_entity = "does_not_exist"
"""


def test_batch_with_circular_cross_references_commits(tmp_path):
    kg = _bare_kg()
    files = {"A": _write(tmp_path, "A", _A), "B": _write(tmp_path, "B", _B)}
    kg.importGraphSchemas(files, crossCheck=True)
    assert set(kg.graphSchemas) == {"A", "B"}


def test_batch_failure_is_atomic_leaves_state_untouched(tmp_path):
    kg = _bare_kg()
    # Pre-load a good schema individually.
    kg.importGraphSchema("A", _write(tmp_path, "A", _A))
    before = dict(kg.graphSchemas)

    files = {"B": _write(tmp_path, "B", _B), "BAD": _write(tmp_path, "BAD", _BAD_ENDPOINT)}
    with pytest.raises(ValueError):
        kg.importGraphSchemas(files, crossCheck=True)
    # The unresolved endpoint must not be retained; state is exactly as before.
    assert kg.graphSchemas == before
    assert "BAD" not in kg.graphSchemas


def test_predefined_load_is_idempotent():
    kg = _bare_kg()
    predefined = _predefined_paths()
    kg.importGraphSchemas(predefined, crossCheck=True, replace=True)
    first = set(kg.graphSchemas)
    # A second whole-set load must not fail on existing names; it rebuilds.
    kg.importGraphSchemas(predefined, crossCheck=True, replace=True)
    assert set(kg.graphSchemas) == first


def test_single_import_still_rejects_duplicate_name(tmp_path):
    kg = _bare_kg()
    path = _write(tmp_path, "A", _A)
    kg.importGraphSchema("A", path)
    with pytest.raises(ValueError, match="already defined"):
        kg.importGraphSchema("A", path)


def _predefined_paths():
    curated = [
        "nuclearEntitySchema", "mbseSchema", "documentSchema", "conditionReportSchema",
        "fmeaSchema", "causalSchema", "safetyRiskSchema", "rootCauseAnalysisSchema",
        "hazopSchema", "stpaSchema", "workOrderSchema", "outageSchema",
        "equipmentOperationSchema", "monitoringSystemSchema", "numericPerformanceSchema",
        "supplyChainSchema", "systemSimulationSchema", "temporalRelationSchema",
        "regulatorySchema",
    ]
    return {name: str(SCHEMA_DIR / f"{name}.toml") for name in curated}
