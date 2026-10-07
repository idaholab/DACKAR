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
