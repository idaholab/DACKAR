"""Database-free tests for ``KG.removeGraphSchema``.

Finding 11: the Streamlit app offered a Remove control, but ``KG`` had no
``removeGraphSchema`` method, so every click raised ``AttributeError`` and left
the registry and table unchanged. Removal must also revalidate references among
the schemas that remain, so it cannot leave a relation pointing at a node label
the removal just deleted.

The remedy mirrors the atomic import path: build a candidate registry without
the named schema, cross-check the remaining set, and replace session state only
if it validates. These tests pin that a removal commits, that a removal which
would strand a cross-schema reference is rejected atomically, and that removing
an unknown name raises.

Exercised on a ``KG.__new__`` instance so no Neo4j driver is opened; only the
attributes the removal path reads are populated.
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


# A self-contained schema (its relation resolves within itself).
_SELF = """
title = "Self"
version = "1.0"
[node.s_node]
node_description = "S"
node_properties = [{ name = "id", type = "string", optional = false, description = "id" }]
[relation.s_to_s]
relation_description = "s to s"
from_entity = "s_node"
to_entity = "s_node"
"""
# A schema that supplies a node label another schema's relation points at.
_PROVIDER = """
title = "Provider"
version = "1.0"
[node.p_node]
node_description = "P"
node_properties = [{ name = "id", type = "string", optional = false, description = "id" }]
"""
# A schema whose relation resolves ONLY because _PROVIDER supplies p_node.
_CONSUMER = """
title = "Consumer"
version = "1.0"
[node.k_node]
node_description = "K"
node_properties = [{ name = "id", type = "string", optional = false, description = "id" }]
[relation.k_to_p]
relation_description = "k to p"
from_entity = "k_node"
to_entity = "p_node"
"""


def test_remove_commits_and_drops_schema(tmp_path):
    kg = _bare_kg()
    kg.importGraphSchemas(
        {"SELF": _write(tmp_path, "SELF", _SELF),
         "PROVIDER": _write(tmp_path, "PROVIDER", _PROVIDER)},
        crossCheck=True,
    )
    kg.removeGraphSchema("SELF")
    assert set(kg.graphSchemas) == {"PROVIDER"}


def test_remove_that_strands_a_reference_is_rejected_atomically(tmp_path):
    kg = _bare_kg()
    kg.importGraphSchemas(
        {"PROVIDER": _write(tmp_path, "PROVIDER", _PROVIDER),
         "CONSUMER": _write(tmp_path, "CONSUMER", _CONSUMER)},
        crossCheck=True,
    )
    before = dict(kg.graphSchemas)
    # Removing PROVIDER would leave CONSUMER.k_to_p pointing at an undefined
    # p_node; the removal must be refused and state left untouched.
    with pytest.raises(ValueError, match="is not defined"):
        kg.removeGraphSchema("PROVIDER")
    assert kg.graphSchemas == before


def test_remove_unknown_name_raises(tmp_path):
    kg = _bare_kg()
    kg.importGraphSchema("SELF", _write(tmp_path, "SELF", _SELF))
    with pytest.raises(ValueError, match="is not defined"):
        kg.removeGraphSchema("NOPE")
    assert set(kg.graphSchemas) == {"SELF"}
