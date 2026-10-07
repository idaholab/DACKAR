"""
End-to-end round-trip test for KG.genericWorkflow against a live Neo4j.

This is an integration test and is skipped unless a Neo4j instance is reachable
(configured via NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD, defaulting to a local
bolt endpoint). It constructs two nodes and one relation from a dataframe through
the schema-governed workflow, then reads them back with an independent driver to
confirm the finalized schema set actually builds a graph.

Spin up an ephemeral backend, e.g.:
    docker run --rm -d --name kg-neo4j -p 7687:7687 -p 7474:7474 \
        -e NEO4J_AUTH=neo4j/testpassword neo4j:5
    NEO4J_PASSWORD=testpassword pytest tests/knowledge_graph/test_generic_workflow_neo4j.py
"""
import os

import pytest

neo4j = pytest.importorskip("neo4j")
pd = pytest.importorskip("pandas")
from neo4j import GraphDatabase

URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
PWD = os.environ.get("NEO4J_PASSWORD", "testpassword")

# Distinct identifier property names (wid/gid) on purpose: the relation loader
# renames the source and target data columns to the node property names, so two
# endpoints both keyed on "id" would collide into a single dataframe column.
SCHEMA_TOML = """
title = "Round-Trip Test Schema"
version = "1.0"

[node.widget]
node_description = "Test source node."
node_properties = [
  { name = "wid",  type = "string", optional = false, description = "Widget id." },
  { name = "size", type = "float",  optional = true,  description = "Widget size." }
]

[node.gadget]
node_description = "Test target node."
node_properties = [
  { name = "gid", type = "string", optional = false, description = "Gadget id." }
]

[relation.connected_to]
relation_description = "A widget is connected to a gadget."
from_entity = "widget"
to_entity   = "gadget"
relation_properties = [
  { name = "weight", type = "float", optional = true, description = "Edge weight." }
]
"""


@pytest.fixture(scope="module")
def neo4j_driver():
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PWD), connection_timeout=5)
        driver.verify_connectivity()
    except Exception as exc:  # ServiceUnavailable, AuthError, etc.
        pytest.skip(f"No reachable Neo4j at {URI}: {exc}")
    yield driver
    driver.close()


def test_generic_workflow_round_trip(neo4j_driver, tmp_path):
    pytest.importorskip("openpyxl")  # KG.__init__ loads the entity-library xlsx
    from dackar.knowledge_graph.KGconstruction import KG

    schema_path = tmp_path / "roundtrip.toml"
    schema_path.write_text(SCHEMA_TOML, encoding="utf-8")

    # importFolderPath=None skips neo4j.conf rewriting; configFilePath is then unused.
    kg = KG(None, None, URI, PWD, USER)
    try:
        kg.resetGraph()
        kg.importGraphSchema("roundTripSchema", str(schema_path))

        data = pd.DataFrame({
            "w_id":     ["w1"],
            "w_size":   [3.5],
            "g_id":     ["g1"],
            "w_weight": [0.8],
        })
        construction_schema = {
            "nodes": {
                "widget": {"wid": "w_id", "size": "w_size"},
                "gadget": {"gid": "g_id"},
            },
            "relations": {
                "connected_to": {
                    "source": {"widget.wid": "w_id"},
                    "target": {"gadget.gid": "g_id"},
                    "properties": {"weight": "w_weight"},
                },
            },
        }

        kg.genericWorkflow(data, construction_schema)

        with neo4j_driver.session() as session:
            widgets = session.run(
                "MATCH (n:widget) RETURN n.wid AS wid, n.size AS size"
            ).data()
            gadgets = session.run(
                "MATCH (n:gadget) RETURN n.gid AS gid"
            ).data()
            rels = session.run(
                "MATCH (:widget)-[r:connected_to]->(:gadget) RETURN r.weight AS weight"
            ).data()

        assert widgets == [{"wid": "w1", "size": 3.5}]
        assert gadgets == [{"gid": "g1"}]
        assert rels == [{"weight": 0.8}]
    finally:
        kg.resetGraph()
        kg.py2neo.close()
