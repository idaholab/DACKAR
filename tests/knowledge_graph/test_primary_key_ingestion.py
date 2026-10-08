"""Production-entry-point tests for schema-declared primary keys.

Database-free: a capture client records the DDL statements and the batch
payloads the ingestion entry point would send to Neo4j, so the merge-key
contract can be asserted without a live driver.

Covers the primary-key findings end to end:
  * The schema-derived merge-key map is threaded through the production entry
    point into the batch upsert, so a declared natural key (``document_id``) is
    actually used instead of the default ``id``.
  * ``build_graph_from_workflow_artifacts`` performs one canonical identity
    transformation: the node's declared primary-key property and the edge
    endpoint key carry the same synthetic value, so Neo4j never merges on a
    null natural key.
  * ``generate_ddl_from_schema`` enforces the uniqueness constraint on the
    declared primary key for a natural-key label and on ``id`` for a default
    label.
"""
from pathlib import Path

import pytest

SCHEMA_DIR = (
    Path(__file__).resolve().parents[2]
    / "src" / "dackar" / "knowledge_graph" / "schemas"
)

# The builder resolves every label strictly when a schema is loaded, so the
# entry-point test loads the full curated set (same list as test_schema_loading).
CURATED = [
    "nuclearEntitySchema", "mbseSchema", "documentSchema", "conditionReportSchema",
    "fmeaSchema", "causalSchema", "safetyRiskSchema", "rootCauseAnalysisSchema",
    "hazopSchema", "stpaSchema", "workOrderSchema", "outageSchema",
    "equipmentOperationSchema", "monitoringSystemSchema", "numericPerformanceSchema",
    "supplyChainSchema", "systemSimulationSchema", "temporalRelationSchema",
    "regulatorySchema",
]
CURATED_PATHS = [str(SCHEMA_DIR / f"{name}.toml") for name in CURATED]


class _CaptureClient:
    """Stand-in for :class:`Py2Neo` that records calls instead of hitting a DB."""

    def __init__(self):
        self.ddl = []
        self.node_batches = []
        self.edge_batches = []

    def query(self, statement, parameters=None, db=None):
        self.ddl.append(statement)
        return []

    def upsert_nodes_batch(self, nodes, db=None, primary_keys=None):
        self.node_batches.append({"nodes": nodes, "primary_keys": primary_keys})

    def upsert_edges_batch(self, edges, db=None, primary_keys=None):
        self.edge_batches.append({"edges": edges, "primary_keys": primary_keys})


def test_entry_point_uses_declared_natural_key_for_document():
    from dackar.knowledge_graph.kg_ingest_neo4j_workflow import ingest_workflow_case_to_neo4j

    client = _CaptureClient()
    # doc_type outside {CR, WO} keeps the `document` label (which declares
    # primary_key = "document_id"); a failure_mode_ref produces a
    # document-sourced edge whose endpoint key must match the node's key.
    documents = [{
        "doc_id": "OE-INPO-2023-CND-047",
        "doc_type": "OE",
        "failure_mode_refs": [{"fm_id": "FM-CND-TUBE-LEAK", "fm_label": "tube leak", "confidence": 0.9}],
    }]
    ingest_workflow_case_to_neo4j(client, CURATED_PATHS, documents=documents)

    # Finding 4: the schema-derived merge-key map reaches the batch upsert and
    # carries the declared natural key for the document label (not the default).
    assert client.node_batches, "no node batch was ingested"
    pk_map = client.node_batches[0]["primary_keys"]
    assert pk_map["document"] == "document_id"
    # A default-id label still maps to id.
    assert pk_map.get("element_usage", "id") == "id"

    # Finding 5: one canonical identity -- the document node carries
    # document_id set to the same synthetic id, and the edge endpoint matches
    # it, so MERGE never sees a null natural key.
    nodes = client.node_batches[0]["nodes"]
    doc_nodes = [n for n in nodes if n["label"] == "document"]
    assert len(doc_nodes) == 1
    doc = doc_nodes[0]["attrs"]
    assert doc["document_id"] == doc["id"] == "DOC:OE-INPO-2023-CND-047"

    edges = client.edge_batches[0]["edges"]
    doc_edges = [e for e in edges if e["from_label"] == "document"]
    assert doc_edges, "expected a document-sourced edge"
    assert all(e["from"] == doc["document_id"] for e in doc_edges)


def test_ddl_constraint_derives_from_primary_key():
    from dackar.knowledge_graph.kg_schema_builder_workflow import (
        generate_ddl_from_schema,
        load_and_merge_schemas,
    )

    schema = load_and_merge_schemas(CURATED_PATHS)
    ddl = generate_ddl_from_schema(schema)

    # Finding 6: natural-key label constrains document_id; default label uses id.
    assert any(
        "(n:`document`)" in s and "n.`document_id` IS UNIQUE" in s for s in ddl
    ), "document uniqueness constraint is not on document_id"
    assert any(
        "(n:`element_usage`)" in s and "n.`id` IS UNIQUE" in s for s in ddl
    ), "default-id label uniqueness constraint is not on id"
    # The old hard-coded id constraint must no longer apply to a natural-key label.
    assert not any(
        "(n:`document`)" in s and "n.`id` IS UNIQUE" in s for s in ddl
    )
