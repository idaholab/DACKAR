"""Database-free tests for the schema-property accessors on ``KG``.

Finding 9: many curated relations (and, in principle, nodes) omit the
``relation_properties`` / ``node_properties`` key entirely. The accessors used
to index that key unconditionally, so a valid construction schema that referred
to a propertyless relation raised ``KeyError`` before ingestion. These tests
pin the contract that an omitted property list is treated as ``[]`` and the
returned frame still carries the columns ``_constructionSchemaValidation``
indexes, while a genuinely absent label still raises ``ValueError``.

The methods are exercised on an instance built with ``KG.__new__`` so no Neo4j
driver is opened; only ``graphSchemas`` is populated.
"""
import pandas as pd
import pytest

from dackar.knowledge_graph.KGconstruction import KG


def _kg_with_schemas(schemas):
    kg = KG.__new__(KG)  # bypass __init__ (no Neo4j driver, no xlsx load)
    kg.graphSchemas = schemas
    return kg


# A relation with no relation_properties and a node with no node_properties,
# alongside well-formed ones, in a single in-memory schema set.
_SCHEMAS = {
    "s": {
        "node": {
            "with_props": {
                "node_properties": [
                    {"name": "id", "type": "string", "optional": False},
                    {"name": "label", "type": "string", "optional": True},
                ]
            },
            "no_props": {"node_description": "a node declaring no properties"},
        },
        "relation": {
            "rel_with_props": {
                "relation_properties": [
                    {"name": "confidence", "type": "float", "optional": True},
                ]
            },
            "rel_no_props": {
                "from_entity": "with_props",
                "to_entity": "no_props",
                "relation_description": "a relation declaring no properties",
            },
        },
    }
}


def test_relation_without_properties_returns_empty_frame_not_keyerror():
    kg = _kg_with_schemas(_SCHEMAS)
    df = kg._schemaReturnRelationProperties("rel_no_props")
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    # The caller indexes these columns; they must exist even when empty.
    assert set(df["name"]) == set()
    assert set(df[df["optional"] == False]["name"]) == set()


def test_node_without_properties_returns_empty_frame_not_keyerror():
    kg = _kg_with_schemas(_SCHEMAS)
    df = kg._schemaReturnNodeProperties("no_props")
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert set(df["name"]) == set()
    assert set(df[df["optional"] == False]["name"]) == set()


def test_declared_properties_are_preserved():
    kg = _kg_with_schemas(_SCHEMAS)
    ndf = kg._schemaReturnNodeProperties("with_props")
    assert set(ndf["name"]) == {"id", "label"}
    assert set(ndf[ndf["optional"] == False]["name"]) == {"id"}

    rdf = kg._schemaReturnRelationProperties("rel_with_props")
    assert set(rdf["name"]) == {"confidence"}


def test_absent_label_still_raises_valueerror():
    kg = _kg_with_schemas(_SCHEMAS)
    with pytest.raises(ValueError):
        kg._schemaReturnNodeProperties("does_not_exist")
    with pytest.raises(ValueError):
        kg._schemaReturnRelationProperties("does_not_exist")


def test_construction_validation_accepts_a_propertyless_relation():
    """The end-to-end caller must not KeyError on a propertyless relation: a
    relation with no properties imposes no required/allowed-property constraint."""
    kg = _kg_with_schemas(_SCHEMAS)
    construction_schema = {
        "relations": {
            "rel_no_props": {"source": "with_props", "target": "no_props", "properties": {}},
        }
    }
    # Must complete without raising (no required props, empty specified set).
    kg._constructionSchemaValidation(construction_schema)
