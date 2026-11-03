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
  6. Every curated node carries a bfo_category drawn from the closed set the
     base meta-schema defines. The meta-schema keeps the field optional (so
     external or experimental schemas are not forced to carry it), but the
     shipped curated set must be fully and validly tagged for cross-schema
     querying (ontology-assessment A6 / cross_project_analysis.md 2d).
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


def _allowed_bfo_categories():
    """Single source of truth: the closed set the base meta-schema declares."""
    base = _load_base_schema()
    node_props = base["properties"]["node"]["additionalProperties"]["properties"]
    return set(node_props["bfo_category"]["enum"])


@pytest.mark.parametrize("name", CURATED)
def test_every_node_has_a_valid_bfo_category(name):
    """The curated set must be fully tagged with a category from the closed set.

    The meta-schema leaves bfo_category optional (a non-curated schema may omit
    it), so base-schema validation alone would pass an untagged curated node.
    This test enforces the stronger contract for the shipped set: every node is
    present and drawn from the allowed vocabulary. It is the knob that makes the
    tag effectively required for curated schemas without changing the
    meta-schema contract (ontology-assessment A6)."""
    allowed = _allowed_bfo_categories()
    schema = _load_toml(name)
    for label, node in schema.get("node", {}).items():
        assert "bfo_category" in node, (
            f"{name}: node {label!r} is missing a bfo_category tag"
        )
        assert node["bfo_category"] in allowed, (
            f"{name}: node {label!r} bfo_category "
            f"{node['bfo_category']!r} is not in the allowed set {sorted(allowed)}"
        )


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

    # File 1 defines the nodes and the relation; file 2 is itself base-valid
    # (declares its own unique node, so no duplicate-node clash) and repeats the
    # same (name, from, to) triple, so the duplicate-triple check is what must
    # fire -- not the duplicate-node check or base-schema structural validation.
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
[node.c]
node_description = "C"
node_properties = [{ name = "id", type = "string", optional = false, description = "id" }]
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


# ---------------------------------------------------------------------------
# Shared loader validation (the per-file checks KG.importGraphSchema runs, now
# applied by the production loader so malformed production schemas fail at load
# instead of at DDL/build/MERGE time).
# ---------------------------------------------------------------------------

def test_production_loader_validates_full_curated_set():
    """The shipped curated set passes the shared loader validation unchanged."""
    from dackar.knowledge_graph.kg_schema_builder_workflow import (
        load_and_merge_schemas,
        validate_schema_document,
    )

    load_and_merge_schemas(CURATED_PATHS)  # must not raise
    for name in CURATED:
        validate_schema_document(_load_toml(name), source=name)


def test_loader_rejects_disallowed_property_type(tmp_path):
    from dackar.knowledge_graph.kg_schema_builder_workflow import load_and_merge_schemas

    bad = """
title = "Bad Type"
version = "1.0"
[node.a]
node_description = "A"
node_properties = [{ name = "id", type = "uuid", optional = false, description = "id" }]
"""
    p = tmp_path / "bad.toml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(Exception):  # ValidationError (type enum) before our ValueError
        load_and_merge_schemas([str(p)])


def test_loader_rejects_primary_key_not_a_declared_property(tmp_path):
    from dackar.knowledge_graph.kg_schema_builder_workflow import load_and_merge_schemas

    bad = """
title = "Bad PK"
version = "1.0"
[node.a]
node_description = "A"
primary_key = "doc_id"
node_properties = [{ name = "id", type = "string", optional = false, description = "id" }]
"""
    p = tmp_path / "bad_pk.toml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="primary key .* is not a declared property"):
        load_and_merge_schemas([str(p)])


def test_loader_rejects_optional_primary_key(tmp_path):
    from dackar.knowledge_graph.kg_schema_builder_workflow import load_and_merge_schemas

    bad = """
title = "Optional PK"
version = "1.0"
[node.a]
node_description = "A"
primary_key = "doc_id"
node_properties = [
  { name = "id", type = "string", optional = false, description = "id" },
  { name = "doc_id", type = "string", optional = true, description = "doc id" }
]
"""
    p = tmp_path / "opt_pk.toml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="primary key .* must be non-optional"):
        load_and_merge_schemas([str(p)])


def test_loader_rejects_node_without_a_valid_default_key(tmp_path):
    """A node with no properties has effective key 'id', which is absent."""
    from dackar.knowledge_graph.kg_schema_builder_workflow import load_and_merge_schemas

    bad = """
title = "No Props"
version = "1.0"
[node.a]
node_description = "A node with no declared properties."
"""
    p = tmp_path / "no_props.toml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="primary key 'id' is not a declared property"):
        load_and_merge_schemas([str(p)])


# ---------------------------------------------------------------------------
# bfo_category ingestion map: build_bfo_category_map turns the (optional) node
# tag into a {label: bfo_category} map that the Neo4j ingestion path emits as a
# second node label (SCHEMA_FOLLOWUPS #1 / cross_project_analysis.md 2d). The
# map is what makes the shipped tag non-inert at runtime.
# ---------------------------------------------------------------------------

def test_bfo_category_map_covers_every_curated_node():
    """Over the real curated set, every node maps to its declared category, and
    every value is drawn from the closed allowed set. Pairs with the acceptance
    test that every curated node is tagged: here the *map* must not drop any."""
    from dackar.knowledge_graph.kg_schema_builder_workflow import (
        build_bfo_category_map,
        load_and_merge_schemas,
    )

    merged = load_and_merge_schemas(CURATED_PATHS)
    cat_map = build_bfo_category_map(merged)
    allowed = _allowed_bfo_categories()

    node_labels = set(merged.get("node", {}))
    assert set(cat_map) == node_labels, (
        f"bfo_category map misses {node_labels - set(cat_map)} "
        f"or has extra {set(cat_map) - node_labels}"
    )
    for label, category in cat_map.items():
        assert category in allowed, (
            f"node {label!r} mapped to {category!r}, not in {sorted(allowed)}"
        )


def test_bfo_category_map_omits_untagged_nodes():
    """The tag is optional in the meta-schema, so a node without bfo_category is
    simply absent from the map (not keyed to None) -- which keeps it out of the
    second-label SET at ingestion."""
    from dackar.knowledge_graph.kg_schema_builder_workflow import build_bfo_category_map

    schema = {
        "node": {
            "tagged": {"bfo_category": "disposition"},
            "untagged": {},
        }
    }
    cat_map = build_bfo_category_map(schema)
    assert cat_map == {"tagged": "disposition"}


def test_upsert_nodes_batch_emits_second_label_only_when_tagged():
    """The ingestion write adds `SET n:`<category>`` for a tagged label and omits
    it for an untagged one, without touching the MERGE (merge identity unchanged).
    Asserted on the generated Cypher via a fake client -- no live Neo4j."""
    from dackar.knowledge_graph.py2neo import Py2Neo

    class _FakeNeo(Py2Neo):
        def __init__(self):  # bypass driver connection
            self.queries = []

        def write(self, query, parameters=None, db=None):
            self.queries.append(query)

    client = _FakeNeo()
    client.upsert_nodes_batch(
        [
            {"label": "failure_mode", "attrs": {"id": "fm1"}},
            {"label": "widget", "attrs": {"id": "w1"}},
        ],
        primary_keys={"failure_mode": "id", "widget": "id"},
        bfo_categories={"failure_mode": "disposition"},  # widget intentionally absent
    )

    by_label = {q.split("`")[1]: q for q in client.queries}
    assert "SET n:`disposition`" in by_label["failure_mode"]
    assert "MERGE (n:`failure_mode`" in by_label["failure_mode"]
    assert "SET n:" not in by_label["widget"].replace("SET n += row", "")


def test_upsert_nodes_batch_rejects_unsafe_category():
    """A category that is not a safe Neo4j identifier is rejected (same guard the
    label/primary-key interpolation already uses), never interpolated raw."""
    from dackar.knowledge_graph.py2neo import Py2Neo

    class _FakeNeo(Py2Neo):
        def __init__(self):
            self.queries = []

        def write(self, query, parameters=None, db=None):
            self.queries.append(query)

    client = _FakeNeo()
    with pytest.raises(ValueError, match="Invalid Neo4j bfo_category"):
        client.upsert_nodes_batch(
            [{"label": "a", "attrs": {"id": "x"}}],
            primary_keys={"a": "id"},
            bfo_categories={"a": "bad label; DROP"},
        )


# ---------------------------------------------------------------------------
# require_bfo_categories: the curated-only "tag is required" contract
# (SCHEMA_FOLLOWUPS #2). Enforced for the predefined set in
# KGconstruction.loadPredefinedGraphSchemas, NOT in the meta-schema (which keeps
# the field optional so external/experimental schemas may omit it).
# ---------------------------------------------------------------------------

def test_require_bfo_categories_passes_on_curated_set():
    """The shipped curated set satisfies the required-tag contract against the
    meta-schema's allowed enum -- the whole point of the MR."""
    from dackar.knowledge_graph.kg_schema_builder_workflow import (
        load_and_merge_schemas,
        require_bfo_categories,
    )

    merged = load_and_merge_schemas(CURATED_PATHS)
    require_bfo_categories(merged, allowed=_allowed_bfo_categories())  # must not raise


def test_require_bfo_categories_raises_on_missing_tag():
    from dackar.knowledge_graph.kg_schema_builder_workflow import require_bfo_categories

    schema = {"node": {"tagged": {"bfo_category": "disposition"}, "untagged": {}}}
    with pytest.raises(ValueError, match="missing bfo_category.*untagged"):
        require_bfo_categories(schema, allowed=_allowed_bfo_categories())


def test_require_bfo_categories_raises_on_out_of_set_value():
    from dackar.knowledge_graph.kg_schema_builder_workflow import require_bfo_categories

    schema = {"node": {"a": {"bfo_category": "not_a_real_category"}}}
    with pytest.raises(ValueError, match="not in allowed set"):
        require_bfo_categories(schema, allowed=_allowed_bfo_categories())


def test_require_bfo_categories_presence_only_when_allowed_is_none():
    """With allowed=None, only presence is enforced (no membership check)."""
    from dackar.knowledge_graph.kg_schema_builder_workflow import require_bfo_categories

    require_bfo_categories({"node": {"a": {"bfo_category": "anything"}}}, allowed=None)
    with pytest.raises(ValueError, match="missing bfo_category"):
        require_bfo_categories({"node": {"a": {}}}, allowed=None)


# ---------------------------------------------------------------------------
# Numeric property bounds (SCHEMA_FOLLOWUPS #3): the meta-schema declares
# minimum/maximum on numeric properties; build_relation_bounds_map surfaces them
# and upsert_edges_batch rejects out-of-range edge property values at ingestion.
# Today the only bounds are confidence [0,1] on the resolves_to_* grounding
# relations.
# ---------------------------------------------------------------------------

def test_relation_bounds_map_captures_confidence_on_grounding_relations():
    """Over the real curated set, every resolves_to_* relation surfaces a
    confidence bound of [0, 1]."""
    from dackar.knowledge_graph.kg_schema_builder_workflow import (
        build_relation_bounds_map,
        load_and_merge_schemas,
    )

    merged = load_and_merge_schemas(CURATED_PATHS)
    bounds = build_relation_bounds_map(merged)

    grounding = [r for r in merged.get("relation", {}) if r.startswith("resolves_to_")]
    assert grounding, "expected resolves_to_* relations in the curated set"
    for rel in grounding:
        assert bounds[rel]["confidence"] == {"minimum": 0.0, "maximum": 1.0}, (
            f"{rel} confidence bound not surfaced correctly: {bounds.get(rel)}"
        )


def test_relation_bounds_map_omits_unbounded_relations():
    """A relation whose properties declare no min/max is absent from the map."""
    from dackar.knowledge_graph.kg_schema_builder_workflow import build_relation_bounds_map

    schema = {
        "relation": {
            "bounded": [{"relation_properties": [
                {"name": "p", "type": "float", "minimum": 0.0, "maximum": 1.0}
            ]}],
            "unbounded": [{"relation_properties": [{"name": "q", "type": "string"}]}],
        }
    }
    bounds = build_relation_bounds_map(schema)
    assert bounds == {"bounded": {"p": {"minimum": 0.0, "maximum": 1.0}}}


def _bounds_client():
    from dackar.knowledge_graph.py2neo import Py2Neo

    class _FakeNeo(Py2Neo):
        def __init__(self):
            self.writes = []

        def write(self, query, parameters=None, db=None):
            self.writes.append((query, parameters))

    return _FakeNeo()


_CONF_BOUNDS = {"resolves_to_asset": {"confidence": {"minimum": 0.0, "maximum": 1.0}}}


def _edge(confidence):
    return {
        "from_label": "nuclear_entity", "to_label": "element_usage",
        "type": "resolves_to_asset", "from": "m1", "to": "a1",
        "attrs": {"confidence": confidence},
    }


def test_upsert_edges_batch_accepts_in_range_value():
    client = _bounds_client()
    client.upsert_edges_batch([_edge(0.8)], relation_bounds=_CONF_BOUNDS)
    assert len(client.writes) == 1  # persisted


@pytest.mark.parametrize("bad", [87.0, 1.5, -0.1])
def test_upsert_edges_batch_rejects_out_of_range_value(bad):
    client = _bounds_client()
    with pytest.raises(ValueError, match="outside declared bounds"):
        client.upsert_edges_batch([_edge(bad)], relation_bounds=_CONF_BOUNDS)
    assert client.writes == []  # nothing persisted (fail-fast, before write)


def test_upsert_edges_batch_skips_none_and_unbounded():
    """A None value and a property with no declared bound are not checked."""
    client = _bounds_client()
    client.upsert_edges_batch([_edge(None)], relation_bounds=_CONF_BOUNDS)
    # edge whose relation type has no bounds entry at all
    other = dict(_edge(999.0), type="mentions")
    client.upsert_edges_batch([other], relation_bounds=_CONF_BOUNDS)
    assert len(client.writes) == 2
