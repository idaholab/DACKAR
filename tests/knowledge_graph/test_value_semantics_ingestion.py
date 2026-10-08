"""Database-free end-to-end test of per-value semantic validation.

Findings 17/18: the column dtype check treats enum / json_string as plain
strings and an array column as any 'mixed' column, so malformed JSON,
out-of-enum values, and heterogeneous scalar columns passed ingestion
validation. KG._checkDataframeDatatypes now runs validateValueSemantics after
the dtype check. This exercises that method directly on a KG built via
__new__ (no Neo4j driver), with an in-memory schema declaring one enum,
one json_string, and one array property.
"""
import pandas as pd
import pytest

from dackar.knowledge_graph.KGconstruction import KG


def _kg():
    kg = KG.__new__(KG)
    kg.graphSchemas = {
        "s": {
            "node": {
                "widget": {
                    "node_properties": [
                        {"name": "id", "type": "string", "optional": False},
                        {"name": "status", "type": "enum", "optional": True,
                         "enum_values": ["open", "closed"]},
                        {"name": "meta", "type": "json_string", "optional": True},
                        {"name": "tags", "type": "array", "optional": True},
                    ]
                }
            },
            "relation": {},
        }
    }
    return kg


def _schema_for(prop):
    return {"nodes": {"widget": {"id": "id_col", prop: f"{prop}_col"}}}


def test_valid_enum_json_and_array_pass():
    kg = _kg()
    data = pd.DataFrame({
        "id_col": ["w1", "w2"],
        "status_col": ["open", "closed"],
        "meta_col": ['{"a": 1}', '{"b": 2}'],
        "tags_col": [["x", "y"], ["z"]],
    })
    cs = {"nodes": {"widget": {
        "id": "id_col", "status": "status_col", "meta": "meta_col", "tags": "tags_col",
    }}}
    kg._checkDataframeDatatypes(data, cs)  # must not raise


def test_out_of_enum_value_is_rejected():
    kg = _kg()
    data = pd.DataFrame({"id_col": ["w1"], "status_col": ["ajar"]})
    with pytest.raises(ValueError, match="does not satisfy schema type enum"):
        kg._checkDataframeDatatypes(data, _schema_for("status"))


def test_malformed_json_string_is_rejected():
    kg = _kg()
    data = pd.DataFrame({"id_col": ["w1"], "meta_col": ["not json"]})
    with pytest.raises(ValueError, match="does not satisfy schema type json_string"):
        kg._checkDataframeDatatypes(data, _schema_for("meta"))


def test_heterogeneous_scalar_array_is_rejected():
    kg = _kg()
    # Scalars 1 and "text" infer as a 'mixed' column (passes the dtype check)
    # but are not list-valued cells, so the value check must reject them.
    data = pd.DataFrame({"id_col": ["w1", "w2"], "tags_col": [1, "text"]})
    with pytest.raises(ValueError, match="does not satisfy schema type array"):
        kg._checkDataframeDatatypes(data, _schema_for("tags"))
