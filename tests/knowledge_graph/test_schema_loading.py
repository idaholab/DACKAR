"""
Acceptance test for the curated DACKAR knowledge-graph schema set.

This test is intentionally database-free: it validates that the curated set of
TOML schemas is mutually consistent against the base meta-schema and the loader
contract, without requiring a live Neo4j instance.

Definition of done (schema finalization):
  1. Every curated schema validates against schemas/baseSchema.json.
  2. Every node/relation property uses an allowed data type.
  3. Every relation endpoint (from_entity/to_entity) resolves to a node
     defined somewhere in the curated set.
  4. No node label is defined in more than one schema.
  5. No relation is duplicated as the same (name, from_entity, to_entity) triple.
"""
import json
import tomllib
from pathlib import Path

import pytest
from jsonschema import validate as js_validate

SCHEMA_DIR = (
    Path(__file__).resolve().parents[2]
    / "src" / "dackar" / "knowledge_graph" / "schemas"
)

# The curated set that the loader (KGconstruction.predefinedGraphSchemas) manages.
# Deprecated schemas (customMbseSchema, reqTechspecSchema) are intentionally absent.
CURATED = [
    "nuclearEntitySchema",
    "mbseSchema",
    "documentSchema",
    "conditionReportSchema",
    "fmeaSchema",
    "causalSchema",
    "safetyRiskSchema",
    "rootCauseAnalysisSchema",
    "hazopSchema",
    "stpaSchema",
    "workOrderSchema",
    "outageSchema",
    "equipmentOperationSchema",
    "monitoringSystemSchema",
    "numericPerformanceSchema",
    "supplyChainSchema",
    "systemSimulationSchema",
    "temporalRelationSchema",
    "regulatorySchema",
]

# Allowed property data types (must match KGconstruction.datatypes).
ALLOWED_TYPES = {
    "string", "integer", "float", "floating", "boolean", "datetime",
    "enum", "array", "json_string",
}


def _load_base_schema():
    with open(SCHEMA_DIR / "baseSchema.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_toml(name):
    with open(SCHEMA_DIR / f"{name}.toml", "rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="module")
def schemas():
    return {name: _load_toml(name) for name in CURATED}


@pytest.mark.parametrize("name", CURATED)
def test_validates_against_base_schema(name):
    base = _load_base_schema()
    js_validate(instance=_load_toml(name), schema=base)


@pytest.mark.parametrize("name", CURATED)
def test_property_types_are_allowed(name):
    schema = _load_toml(name)
    for label, node in schema.get("node", {}).items():
        for prop in node.get("node_properties", []):
            assert prop["type"] in ALLOWED_TYPES, (
                f"{name}: node {label}.{prop['name']} uses "
                f"disallowed type {prop['type']!r}"
            )
    for label, rel in schema.get("relation", {}).items():
        for prop in rel.get("relation_properties", []):
            assert prop["type"] in ALLOWED_TYPES, (
                f"{name}: relation {label}.{prop['name']} uses "
                f"disallowed type {prop['type']!r}"
            )


@pytest.mark.parametrize("name", CURATED)
def test_primary_key_contract(name):
    """Every node's effective MERGE key (declared primary_key or 'id') must be
    a non-optional property defined on that node."""
    schema = _load_toml(name)
    for label, node in schema.get("node", {}).items():
        props = {p["name"]: p for p in node.get("node_properties", [])}
        pk = node.get("primary_key", "id")
        assert pk in props, (
            f"{name}: node {label!r} primary key {pk!r} is not a defined property"
        )
        assert props[pk].get("optional", True) is False, (
            f"{name}: node {label!r} primary key {pk!r} must be non-optional"
        )


def test_no_duplicate_node_labels(schemas):
    seen = {}
    for name, schema in schemas.items():
        for label in schema.get("node", {}):
            assert label not in seen, (
                f"Node {label!r} defined in both {seen[label]} and {name}"
            )
            seen[label] = name


def test_relation_endpoints_resolve(schemas):
    nodes = {
        label
        for schema in schemas.values()
        for label in schema.get("node", {})
    }
    for name, schema in schemas.items():
        for rel, body in schema.get("relation", {}).items():
            for side in ("from_entity", "to_entity"):
                endpoint = body[side]
                assert endpoint in nodes, (
                    f"{name}: relation {rel!r} {side}={endpoint!r} "
                    f"is not a defined node"
                )


def test_no_duplicate_relation_triples(schemas):
    seen = {}
    for name, schema in schemas.items():
        for rel, body in schema.get("relation", {}).items():
            triple = (rel, body["from_entity"], body["to_entity"])
            assert triple not in seen, (
                f"Relation triple {triple} duplicated in "
                f"{seen[triple]} and {name}"
            )
            seen[triple] = name


# ---------------------------------------------------------------------------
# Production merger acceptance tests (load the curated paths through the actual
# ingestion-path merger, not a reimplementation). The curated set reuses generic
# relation verbs (caused_by, recommends_action, targets_element,
# has_temporal_reference) for different endpoint pairs; the merger must accept
# the whole set and preserve every endpoint pair.
# ---------------------------------------------------------------------------

CURATED_PATHS = [str(SCHEMA_DIR / f"{name}.toml") for name in CURATED]


def test_curated_set_loads_through_production_merger():
    from dackar.knowledge_graph.kg_schema_builder_workflow import load_and_merge_schemas

    merged = load_and_merge_schemas(CURATED_PATHS)
    # Relation names are now list-valued (one spec per declared endpoint pair).
    for name, specs in merged["relation"].items():
        assert isinstance(specs, list) and specs, name


def test_reused_relation_names_preserve_every_endpoint_pair():
    from dackar.knowledge_graph.kg_schema_builder_workflow import (
        load_and_merge_schemas,
        relation_endpoint_map,
    )

    merged = load_and_merge_schemas(CURATED_PATHS)
    rmap = relation_endpoint_map(merged)
    # Each generic verb below is declared in two curated schemas with distinct
    # endpoint pairs; both pairs must survive the merge.
    reused = ("caused_by", "recommends_action", "targets_element", "has_temporal_reference")
    for name in reused:
        pairs = rmap.get(name, [])
        assert len(pairs) >= 2, f"relation {name!r} lost an endpoint pair: {pairs}"
        assert len(pairs) == len(set(pairs)), f"relation {name!r} has duplicate pairs: {pairs}"


def test_merger_rejects_exact_duplicate_relation_triple(tmp_path):
    from dackar.knowledge_graph.kg_schema_builder_workflow import load_and_merge_schemas

    # File 1 defines the nodes and the relation; file 2 repeats only the same
    # (name, from, to) triple, so the duplicate-triple check is what must fire
    # (not the duplicate-node check).
    nodes_and_rel = """
title = "Dup A"
version = "1.0"
[node.a]
node_description = "A"
node_properties = [{ name = "id", type = "string", optional = false, description = "id" }]
[node.b]
node_description = "B"
node_properties = [{ name = "id", type = "string", optional = false, description = "id" }]
[relation.links]
relation_description = "a links b"
from_entity = "a"
to_entity = "b"
"""
    rel_only = """
title = "Dup B"
version = "1.0"
[relation.links]
relation_description = "a links b again"
from_entity = "a"
to_entity = "b"
"""
    p1 = tmp_path / "s1.toml"
    p2 = tmp_path / "s2.toml"
    p1.write_text(nodes_and_rel, encoding="utf-8")
    p2.write_text(rel_only, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate relation definition"):
        load_and_merge_schemas([str(p1), str(p2)])
