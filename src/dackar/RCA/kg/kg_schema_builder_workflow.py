from __future__ import annotations

import json
import logging
import tomllib
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from kg.py2neo_workflow import Py2Neo

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema loading and helpers
# ---------------------------------------------------------------------------

def load_toml_schema(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a TOML schema file and return its contents as a dict.

    Args:
        path: Filesystem path to the ``.toml`` schema file.

    Returns:
        Parsed TOML document as a nested dictionary.
    """
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def load_and_merge_schemas(schema_paths: Union[str, Path, Iterable[Union[str, Path]]]) -> Dict[str, Any]:
    """Load one or more TOML schema files and merge them into a single schema dict.

    The merged dict has top-level keys ``"node"`` and ``"relation"``.  Duplicate
    keys across files raise an error to prevent silent overwrites.

    Args:
        schema_paths: A single path or an iterable of paths to ``.toml`` files.

    Returns:
        A merged schema dict with ``{"node": {...}, "relation": {...}}``.

    Raises:
        ValueError: If the same node or relation key appears in more than one file.
    """
    if isinstance(schema_paths, (str, Path)):
        paths = [Path(schema_paths)]
    else:
        paths = [Path(p) for p in schema_paths]

    merged: Dict[str, Any] = {"node": {}, "relation": {}}
    for path in paths:
        schema = load_toml_schema(path)
        for section in ("node", "relation"):
            for key, value in (schema.get(section) or {}).items():
                if key in merged[section]:
                    raise ValueError(f"Duplicate schema key {section}.{key} in {path}")
                merged[section][key] = value
    return merged


def _schema_props(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the property list from a node or relation schema spec.

    Checks both ``"node_properties"`` and ``"properties"`` keys for compatibility
    with different TOML schema conventions.

    Args:
        spec: A single node or relation entry from the merged schema.

    Returns:
        List of property descriptor dicts, or an empty list if none are defined.
    """
    return spec.get("node_properties") or spec.get("properties") or []


def _node_primary_key(spec: Dict[str, Any]) -> str:
    """Determine the primary key property name for a node spec.

    Selects the first non-optional property; falls back to ``"id"`` if present
    in the property list, then to the first listed property name, and finally
    to the hard-coded default ``"id"``.

    Args:
        spec: A single node entry from the merged schema.

    Returns:
        The property name to use as the primary key.
    """
    props = _schema_props(spec)
    for prop in props:
        if not prop.get("optional", True):
            return prop["name"]
    names = [prop["name"] for prop in props]
    if "id" in names:
        return "id"
    return names[0] if names else "id"


def _label_candidates(name: str) -> List[str]:
    """Generate a list of label name variants to try when resolving a schema label.

    Produces the original name, its lowercase form, and a PascalCase conversion
    to account for naming conventions used across different TOML schemas.

    Args:
        name: Base label name.

    Returns:
        List of candidate strings (original, lowercase, PascalCase), deduplicated
        while preserving order.
    """
    out = [name]
    low = name.lower()
    if low != name:
        out.append(low)
    pascal = "".join(part.capitalize() for part in low.split("_"))
    if pascal not in out:
        out.append(pascal)
    return out


def resolve_node_label(schema: Dict[str, Any], *candidates: str) -> str:
    """Resolve the first candidate name that exists as a node label in the schema.

    For each candidate, tries the original name and common variants (lowercase,
    PascalCase).  If none match, returns the first candidate unchanged as a
    safe fallback.

    Args:
        schema: Merged schema dict (must contain a ``"node"`` key).
        *candidates: One or more preferred label names, in priority order.

    Returns:
        The matched label string from the schema, or *candidates[0]* if no
        match is found.
    """
    available = schema.get("node") or {}
    for candidate in candidates:
        for variant in _label_candidates(candidate):
            if variant in available:
                return variant
    return candidates[0]


def relation_endpoint_map(schema: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
    """Build a lookup from relation type name to its (from_entity, to_entity) pair.

    Only relations that declare both ``from_entity`` and ``to_entity`` in the
    schema are included.

    Args:
        schema: Merged schema dict (must contain a ``"relation"`` key).

    Returns:
        Dict mapping each relation name to a ``(source_entity, target_entity)``
        tuple of entity type strings.
    """
    out: Dict[str, Tuple[str, str]] = {}
    for name, spec in (schema.get("relation") or {}).items():
        from_entity = spec.get("from_entity")
        to_entity = spec.get("to_entity")
        if from_entity and to_entity:
            out[name] = (from_entity, to_entity)
    return out


# ---------------------------------------------------------------------------
# Property sanitisation
# ---------------------------------------------------------------------------

def _is_primitive(value: Any) -> bool:
    """Return True if *value* is a Neo4j-native scalar type (str, int, float, or bool).

    Args:
        value: Any Python object.

    Returns:
        ``True`` if *value* can be stored directly as a Neo4j property scalar.
    """
    return isinstance(value, (str, int, float, bool))


def sanitize_value(value: Any) -> Any:
    """Convert an arbitrary Python value to a Neo4j-compatible property value.

    Conversion rules:
    - ``datetime`` / ``date`` → ISO-8601 string.
    - ``dict`` → JSON string (sorted keys, UTF-8).
    - ``list`` of primitives → kept as-is; mixed/complex lists → JSON string.
    - ``None`` and primitive scalars → returned unchanged.
    - Anything else → ``str(value)``.

    Args:
        value: Python value to sanitize.

    Returns:
        A value safe for storage as a Neo4j node or relationship property.
    """
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        return value if all(_is_primitive(v) for v in value) else json.dumps(value, ensure_ascii=False)
    if value is None or _is_primitive(value):
        return value
    return str(value)


def sanitize_props(props: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize all values in a property dict, dropping ``None`` entries.

    Args:
        props: Raw property dict potentially containing non-Neo4j-native types.

    Returns:
        New dict with all values converted by :func:`sanitize_value` and
        keys whose value was ``None`` removed.
    """
    return {k: sanitize_value(v) for k, v in props.items() if v is not None}


# ---------------------------------------------------------------------------
# DDL generation
# ---------------------------------------------------------------------------

def generate_ddl_from_schema(schema: Dict[str, Any]) -> List[str]:
    """Generate Neo4j DDL statements (constraints and indexes) from a merged schema.

    For every node label a ``UNIQUE`` constraint on ``id`` is created.
    Additionally, a ``CREATE INDEX`` statement is emitted for each property
    that carries ``"indexed": true`` in its spec.

    Args:
        schema: Merged schema dict as returned by :func:`load_and_merge_schemas`.

    Returns:
        List of Cypher DDL strings ready to be executed against Neo4j.
    """
    ddl: List[str] = []
    for label, spec in (schema.get("node") or {}).items():
        ddl.append(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")
        for prop in _schema_props(spec):
            if prop.get("indexed"):
                ddl.append(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{prop['name']})")
    return ddl


def apply_schema_constraints(
    client: Py2Neo,
    schema_paths: Union[str, Path, Iterable[Union[str, Path]]],
    database: Optional[str] = None,
) -> None:
    """Load schemas and apply all DDL constraints and indexes to the database.

    Combines :func:`load_and_merge_schemas`, :func:`generate_ddl_from_schema`,
    and :meth:`Py2Neo.query` into a single convenience call.

    Args:
        client: Active :class:`Py2Neo` connection.
        schema_paths: One or more paths to TOML schema files.
        database: Target database name; uses the driver default when ``None``.
    """
    schema = load_and_merge_schemas(schema_paths)
    for stmt in generate_ddl_from_schema(schema):
        client.query(stmt, db=database)


# ---------------------------------------------------------------------------
# Generic graph batch builder
# ---------------------------------------------------------------------------

class GraphBatch:
    """In-memory accumulator for nodes and edges before bulk Neo4j ingestion.

    Nodes are keyed by their ``id`` string; duplicate additions are merged.
    Edges are keyed by ``(source_id, target_id, rel_type)`` and support
    optional schema-based endpoint validation.
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """Initialise an empty batch, optionally bound to a schema.

        Args:
            schema: Merged schema dict used for label resolution and endpoint
                validation.  Defaults to an empty schema when ``None``.
        """
        self.schema = schema or {"node": {}, "relation": {}}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.relation_map = relation_endpoint_map(self.schema)

    def add_node(self, node_id: str, label: str, attrs: Optional[Dict[str, Any]] = None) -> str:
        """Add or merge a node into the batch.

        If a node with *node_id* already exists its attributes are updated with
        the new values (shallow merge after sanitization).

        Args:
            node_id: Unique identifier for the node (used as the ``id`` property).
            label: Neo4j label for the node.
            attrs: Optional property dict; ``id`` is always set from *node_id*.

        Returns:
            The *node_id* string, for convenience when chaining calls.
        """
        attrs = deepcopy(attrs or {})
        attrs["id"] = node_id
        clean = sanitize_props(attrs)
        if node_id in self.nodes:
            self.nodes[node_id]["attrs"].update(clean)
        else:
            self.nodes[node_id] = {"id": node_id, "label": label, "attrs": clean}
        return node_id

    def add_edge(
        self,
        src: str,
        dst: str,
        rel_type: str,
        attrs: Optional[Dict[str, Any]] = None,
        allow_untyped: bool = True,
    ) -> None:
        """Add or update a directed edge in the batch.

        Endpoint labels are looked up from already-added nodes.  When *rel_type*
        is declared in the schema its expected endpoint types are validated.

        Args:
            src: Node id of the source node (must already be in the batch).
            dst: Node id of the target node (must already be in the batch).
            rel_type: Relationship type name.
            attrs: Optional property dict for the relationship.
            allow_untyped: When ``True``, relationship types not declared in the
                schema are accepted.  When ``False``, an undeclared type raises
                :class:`ValueError`.

        Raises:
            KeyError: If *src* or *dst* has not been added to the batch yet.
            ValueError: If the schema declares endpoint types for *rel_type* and
                the actual node labels do not match, or if *allow_untyped* is
                ``False`` and *rel_type* is not in the schema.
        """
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError(f"Cannot create edge {rel_type}: missing node {src!r} or {dst!r}")

        src_label = self.nodes[src]["label"]
        dst_label = self.nodes[dst]["label"]
        spec = self.relation_map.get(rel_type)
        if spec:
            exp_src = resolve_node_label(self.schema, spec[0])
            exp_dst = resolve_node_label(self.schema, spec[1])
            if src_label != exp_src or dst_label != exp_dst:
                raise ValueError(
                    f"Relation {rel_type} expects ({exp_src} -> {exp_dst}), got ({src_label} -> {dst_label})"
                )
        elif not allow_untyped:
            raise ValueError(f"Relation {rel_type} is not declared in the TOML schema")

        key = (src, dst, rel_type)
        payload = sanitize_props(attrs or {})
        edge = self.edges.setdefault(
            key,
            {"from": src, "to": dst, "type": rel_type, "from_label": src_label, "to_label": dst_label, "attrs": {}},
        )
        edge["attrs"].update(payload)

    def as_lists(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return the accumulated nodes and edges as plain lists.

        Returns:
            A two-tuple ``(nodes, edges)`` where each element is a list of
            dicts suitable for passing to :meth:`Py2Neo.upsert_nodes_batch`
            and :meth:`Py2Neo.upsert_edges_batch` respectively.
        """
        return list(self.nodes.values()), list(self.edges.values())


# ---------------------------------------------------------------------------
# Workflow-specific KG construction
# ---------------------------------------------------------------------------

def _prefix(value: Optional[str], prefix: str) -> Optional[str]:
    """Build a namespaced node id by prepending *prefix* to *value*.

    Strips whitespace and skips empty values.  Already-namespaced strings
    (containing ``":"``) are returned unchanged to avoid double-prefixing.

    Args:
        value: Raw identifier string (e.g. ``"pump-101"``).
        prefix: Namespace prefix (e.g. ``"ASSET"``).

    Returns:
        A string like ``"ASSET:pump-101"``, or ``None`` if *value* is falsy
        or blank after stripping.
    """
    if not value:
        return None
    value = str(value).strip()
    if not value:
        return None
    if ":" in value:
        return value
    return f"{prefix}:{value}"

def _truthy(value: Any) -> bool:
    """Return True if *value* is non-empty and not a sentinel null-like string.

    Treats ``"unknown"``, ``"none"``, and ``"null"`` as falsy in addition to
    standard Python falsy values.

    Args:
        value: Any value to test.

    Returns:
        ``True`` if the value is considered meaningfully present.
    """
    return bool(value) and value not in ("unknown", "none", "null")

def _norm_text_key(value: Optional[str]) -> Optional[str]:
    """Normalise a text string into a stable, lowercase key.

    Collapses internal whitespace, strips leading/trailing whitespace, and
    lowercases the result.  Used to derive deterministic node ids from free-text
    labels.

    Args:
        value: Input string to normalise, or ``None``.

    Returns:
        Normalised string, or ``None`` if *value* is ``None`` or blank.
    """
    if value is None:
        return None
    text = " ".join(str(value).split()).strip().lower()
    return text or None

def _safe_rel(g: GraphBatch, src: str, dst: str, preferred: str, fallback: str, attrs: Optional[Dict[str, Any]] = None) -> None:
    """Add an edge using *preferred* relation type, falling back to *fallback* if not in schema.

    Args:
        g: The :class:`GraphBatch` to add the edge to.
        src: Source node id.
        dst: Target node id.
        preferred: Preferred relationship type name.
        fallback: Fallback relationship type name used when *preferred* is not
            declared in the schema's relation map.
        attrs: Optional property dict for the relationship.
    """
    rel = preferred if preferred in g.relation_map else fallback
    g.add_edge(src, dst, rel, attrs or {})

def _safe_ptr_entity_rel(
    g: GraphBatch,
    src: str,
    dst: str,
    attrs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    ProcessedTextRecord -> entity relation chooser.
    Avoid using document-scoped relations like 'mentions' when the schema
    types them as condition_report/work_order -> mbse_entity.
    """
    src_label = g.nodes[src]["label"]
    dst_label = g.nodes[dst]["label"]

    def _relation_matches(rel_name: str) -> bool:
        spec = g.relation_map.get(rel_name)
        if not spec:
            return False
        exp_src = resolve_node_label(g.schema, spec[0])
        exp_dst = resolve_node_label(g.schema, spec[1])
        return src_label == exp_src and dst_label == exp_dst

    # Only use typed relations if they truly match PTR -> entity endpoints.
    for rel_name in ("targets_entity", "references_entity", "mentions"):
        if _relation_matches(rel_name):
            g.add_edge(src, dst, rel_name, attrs or {})
            return

    # Otherwise use an untyped fallback relation that is safe for PTR evidence links.
    g.add_edge(src, dst, "references_entity", attrs or {}, allow_untyped=True)

def _safe_ptr_failure_mode_rel(
    g: GraphBatch,
    src: str,
    dst: str,
    attrs: Optional[Dict[str, Any]] = None,
) -> None:
    """Add a ProcessedTextRecord → failure_mode edge using the best available relation type.

    Tries ``"supports_hypothesis"``, ``"references_failure_mode"``, and
    ``"caused_by"`` in order, picking the first whose endpoint types match the
    actual node labels.  Falls back to an untyped ``"references_failure_mode"``
    edge if none match.

    Args:
        g: The :class:`GraphBatch` to add the edge to.
        src: Source node id (a ProcessedTextRecord node).
        dst: Target node id (a failure_mode node).
        attrs: Optional property dict for the relationship.
    """
    src_label = g.nodes[src]["label"]
    dst_label = g.nodes[dst]["label"]

    def _relation_matches(rel_name: str) -> bool:
        spec = g.relation_map.get(rel_name)
        if not spec:
            return False
        exp_src = resolve_node_label(g.schema, spec[0])
        exp_dst = resolve_node_label(g.schema, spec[1])
        return src_label == exp_src and dst_label == exp_dst

    for rel_name in ("supports_hypothesis", "references_failure_mode", "caused_by"):
        if _relation_matches(rel_name):
            g.add_edge(src, dst, rel_name, attrs or {})
            return

    g.add_edge(src, dst, "references_failure_mode", attrs or {}, allow_untyped=True)

def build_graph_from_workflow_artifacts(
    schema_paths: Optional[Union[str, Path, Iterable[Union[str, Path]]]] = None,
    *,
    event: Optional[Dict[str, Any]] = None,
    kg_context: Optional[Dict[str, Any]] = None,
    telemetry_summary: Optional[Dict[str, Any]] = None,
    evidence_bundle: Optional[Dict[str, Any]] = None,
    causality_candidates: Optional[Dict[str, Any]] = None,
    rca_card: Optional[Dict[str, Any]] = None,
    operational_context: Optional[Dict[str, Any]] = None,
    pm_compliance: Optional[Dict[str, Any]] = None,
    documents: Optional[Sequence[Dict[str, Any]]] = None,
    processed_text_records: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Translate all RCA workflow artifacts into a graph of nodes and edges.

    Iterates over every supplied artifact, creates typed nodes for each
    entity (assets, components, failure modes, events, telemetry signals,
    anomalies, causal candidates, evidence snippets, etc.), and links them
    with labelled directed edges.  All artifacts are optional; only those
    provided contribute nodes and edges.

    Args:
        schema_paths: Optional path(s) to TOML schema file(s) used for label
            resolution and endpoint validation.  An empty schema is used when
            ``None``.
        event: Parsed ``event`` artifact dict.
        kg_context: Parsed ``kg_context`` artifact dict (components, failure
            modes, past events).
        telemetry_summary: Parsed ``telemetry_summary`` artifact dict.
        evidence_bundle: Parsed ``evidence_bundle`` artifact dict.
        causality_candidates: Parsed ``causality_candidates`` artifact dict.
        rca_card: Parsed ``rca_card`` artifact dict.
        operational_context: Parsed ``operational_context`` artifact dict.
        pm_compliance: Parsed ``pm_compliance`` artifact dict.
        documents: List of document descriptor dicts.
        processed_text_records: List of ``processed_text_record`` dicts.

    Returns:
        A two-tuple ``(nodes, edges)`` — lists of dicts ready for bulk
        ingestion via :func:`ingest_graph_toml`.
    """
    schema = load_and_merge_schemas(schema_paths) if schema_paths else {"node": {}, "relation": {}}
    g = GraphBatch(schema=schema)

    labels = {
        "asset": resolve_node_label(schema, "mbse_entity", "asset"),
        "component": resolve_node_label(schema, "mbse_entity", "component"),
        "failure_mode": resolve_node_label(schema, "failure_mode"),
        "document": resolve_node_label(schema, "Document", "document"),
        "processed_text_record": resolve_node_label(schema, "ProcessedTextRecord", "processed_text_record"),
        "condition_report": resolve_node_label(schema, "condition_report"),
        "work_order": resolve_node_label(schema, "work_order"),
        "event": resolve_node_label(schema, "abnormal_event", "event"),
        "rca_case": resolve_node_label(schema, "rca_case"),
        "causal_factor": resolve_node_label(schema, "causal_factor", "candidate_hypothesis"),
    }

    event_id: Optional[str] = None

    if event:
        event_id = _prefix(event.get("event_id"), "EVT")
        asset_id = _prefix(event.get("asset_id"), "ASSET")
        component_id = _prefix(event.get("component_id"), "CMP")

        if asset_id:
            g.add_node(asset_id, labels["asset"], {"source_key": event.get("asset_id"), "kind": "asset"})
        if component_id:
            g.add_node(component_id, labels["component"], {"source_key": event.get("component_id"), "kind": "component"})
        if event_id:
            symptom = event.get("symptom_signature") or {}
            g.add_node(
                event_id,
                labels["event"],
                {
                    **event,
                    "event_key": event.get("event_id"),
                    "symptom_signature_json": symptom,
                    "temporal_type": event.get("temporal_type"),
                    "analysis_window_json": event.get("analysis_window"),
                },
            )
            if asset_id:
                g.add_edge(event_id, asset_id, "mentions", {"role": "asset"})
            if component_id:
                g.add_edge(event_id, component_id, "mentions", {"role": "component"})

    for doc in documents or []:
        doc_label = labels["document"]
        doc_type = (doc.get("doc_type") or "").upper()
        if doc_type == "CR":
            doc_label = labels["condition_report"]
        elif doc_type == "WO":
            doc_label = labels["work_order"]

        doc_node_id = _prefix(doc.get("doc_id"), "DOC")
        if not doc_node_id:
            continue
        g.add_node(doc_node_id, doc_label, {**doc, "doc_key": doc.get("doc_id")})

        for equipment_id in doc.get("equipment_ids") or []:
            asset_id = _prefix(equipment_id, "ASSET")
            g.add_node(asset_id, labels["asset"], {"source_key": equipment_id, "kind": "asset"})
            rel = "mentions" if doc_label == labels["condition_report"] else "targets_entity"
            if rel == "targets_entity" and rel not in g.relation_map:
                rel = "mentions"
            g.add_edge(doc_node_id, asset_id, rel, {"source": "document_equipment_ids"})

        for rel_doc in doc.get("related_docs") or []:
            target = _prefix(rel_doc.get("doc_id"), "DOC")
            if target:
                g.add_node(target, labels["document"], {"doc_key": rel_doc.get("doc_id")})
                g.add_edge(doc_node_id, target, "linked_to_report", {"relation_type": rel_doc.get("relation_type")})

        for fm in doc.get("failure_mode_refs") or []:
            fm_id = _prefix(fm.get("fm_id"), "FM")
            if fm_id:
                g.add_node(fm_id, labels["failure_mode"], {"source_key": fm.get("fm_id"), "label": fm.get("fm_label")})
                edge_type = "caused_by" if doc_label == labels["condition_report"] else "references_failure_mode"
                g.add_edge(doc_node_id, fm_id, edge_type, {"confidence": fm.get("confidence")})

    for rec in processed_text_records or []:
        rec_id = _prefix(rec.get("record_id"), "PTR")
        if not rec_id:
            continue
        provenance = rec.get("provenance") or {}
        enrichment = rec.get("enrichment") or {}
        metadata = rec.get("metadata") or {}
        stage5 = enrichment.get("stage5_causal_condition") or {}

        g.add_node(
            rec_id,
            labels["processed_text_record"],
             {
                "record_key": rec.get("record_id"),
                "doc_id": rec.get("doc_id"),
                "doc_type": rec.get("doc_type"),
                "chunk_index": rec.get("chunk_index"),
                "section_path": rec.get("section_path") or provenance.get("section_path"),
                "embedding_text": rec.get("embedding_text"),
                "metadata_json": metadata,
                "provenance_json": provenance,
                "condition_assessment_json": rec.get("condition_assessment") or stage5.get("condition_state"),
                "stage5_causal_condition_json": stage5,
            },
        )
        doc_id = _prefix(rec.get("doc_id"), "DOC")
        if doc_id:
            rec_doc_label = labels["document"]
            rec_doc_type = str(rec.get("doc_type") or "").upper()
            if rec_doc_type == "CR":
                rec_doc_label = labels["condition_report"]
            elif rec_doc_type == "WO":
                rec_doc_label = labels["work_order"]
            g.add_node(doc_id, rec_doc_label, {"doc_key": rec.get("doc_id")})
            g.add_edge(rec_id, doc_id, "derived_from_document", {})
        metadata = rec.get("metadata") or {}

        for equipment_id in metadata.get("equipment_ids") or []:
            asset_id = _prefix(equipment_id, "ASSET")
            g.add_node(asset_id, labels["asset"], {"source_key": equipment_id, "kind": "asset"})
            _safe_ptr_entity_rel(
                g,
                rec_id,
                asset_id,
                {"source": "processed_text_metadata.equipment_ids"},
            )

        for comp_name in metadata.get("component_names") or []:
            node_id = _prefix(comp_name, "CMP")
            g.add_node(node_id, labels["component"], {"source_key": comp_name, "kind": "component"})
            _safe_ptr_entity_rel(
                g,
                rec_id,
                node_id,
                {"source": "processed_text_metadata.component_names"},
            )

        for fm_raw in (metadata.get("mechanisms") or []) + (metadata.get("failure_outcomes") or []):
            fm_key = _norm_text_key(fm_raw)
            if not fm_key:
                continue
            fm_id = _prefix(fm_key, "FM")
            g.add_node(fm_id, labels["failure_mode"], {"source_key": fm_raw})
            _safe_ptr_failure_mode_rel(
                g,
                rec_id,
                fm_id,
                {"source": "processed_text_metadata.failure_semantics"},
            )

        for row in stage5.get("extracted_causal_statements") or []:
            cause_text = _norm_text_key(row.get("cause_text"))
            effect_text = _norm_text_key(row.get("effect_text"))
            confidence = row.get("confidence")
            if not cause_text or not effect_text:
                continue

            cause_id = _prefix(cause_text, "FM")
            effect_id = _prefix(effect_text, "FM")
            g.add_node(cause_id, labels["failure_mode"], {"source_key": cause_text})
            g.add_node(effect_id, labels["failure_mode"], {"source_key": effect_text})

            rel = "textual_cause_of" if "textual_cause_of" in g.relation_map else "causes"
            g.add_edge(cause_id, effect_id, rel, {
                "source": "processed_text_record.stage5",
                "record_id": rec.get("record_id"),
                "confidence": confidence,
                "connector": row.get("connector"),
            })

    if kg_context:
        subgraph_id = _prefix(kg_context.get("subgraph_id"), "KGCTX")
        if subgraph_id:
            g.add_node(subgraph_id, "KGContext", {**kg_context, "subgraph_key": kg_context.get("subgraph_id")})
        asset_id = _prefix(kg_context.get("asset_id"), "ASSET")
        if asset_id:
            g.add_node(asset_id, labels["asset"], {"source_key": kg_context.get("asset_id"), "kind": "asset"})
            if subgraph_id:
                g.add_edge(subgraph_id, asset_id, "context_for_asset", {})
        if event_id and subgraph_id:
            g.add_edge(subgraph_id, event_id, "context_for_event", {})

        for comp in kg_context.get("components") or []:
            comp_id = _prefix(comp.get("component_id"), "CMP")
            if not comp_id:
                continue
            g.add_node(comp_id, labels["component"], {**comp, "source_key": comp.get("component_id"), "kind": "component"})
            if asset_id:
                rel = comp.get("relation_to_asset") or "composition"
                if rel not in g.relation_map:
                    rel = "composition"
                g.add_edge(asset_id, comp_id, rel, {"hop_distance": comp.get("hop_distance")})
            if subgraph_id:
                g.add_edge(subgraph_id, comp_id, "contains_component", {"hop_distance": comp.get("hop_distance")})

        for fm in kg_context.get("failure_modes") or []:
            fm_id = _prefix(fm.get("fm_id"), "FM")
            comp_id = _prefix(fm.get("applies_to_component_id"), "CMP")
            if not fm_id:
                continue
            g.add_node(fm_id, labels["failure_mode"], {**fm, "source_key": fm.get("fm_id")})
            if comp_id:
                g.add_node(comp_id, labels["component"], {"source_key": fm.get("applies_to_component_id"), "kind": "component"})
                g.add_edge(comp_id, fm_id, "has_failure_mode", {"source": "kg_context"})
            if subgraph_id:
                g.add_edge(subgraph_id, fm_id, "contains_failure_mode", {})

        for past in kg_context.get("past_events") or []:
            pe_id = _prefix(past.get("event_id"), "EVT")
            if not pe_id:
                continue
            g.add_node(pe_id, labels["event"], {**past, "event_key": past.get("event_id")})
            if event_id:
                g.add_edge(pe_id, event_id, "causes", {"evidence_type": "historical_precedent"})
            fm_id = _prefix(past.get("fm_id"), "FM")
            if fm_id:
                g.add_node(fm_id, labels["failure_mode"], {"source_key": past.get("fm_id")})
                g.add_edge(pe_id, fm_id, "caused_by", {"source": "kg_context.past_events"})

    if telemetry_summary:
        tel_id = _prefix(f"{telemetry_summary.get('event_id')}:{telemetry_summary.get('generated_at')}", "TEL")
        g.add_node(
            tel_id,
            "TelemetrySummary",
            {
                **telemetry_summary,
                "window_json": telemetry_summary.get("window"),
                "overall_assessment_json": telemetry_summary.get("overall_assessment"),
            },
        )
        if event_id:
            g.add_edge(tel_id, event_id, "summarizes_event", {})

        for signal in telemetry_summary.get("signals") or []:
            sig_id = _prefix(f"{telemetry_summary.get('event_id')}:{signal.get('sensor_id')}:{signal.get('parameter')}", "SIG")
            g.add_node(
                sig_id,
                "monitored_variable",
                {
                    "sensor_id": signal.get("sensor_id"),
                    "parameter": signal.get("parameter"),
                    "unit": signal.get("unit"),
                    "stats_json": signal.get("stats"),
                    "baseline_comparison_json": signal.get("baseline_comparison"),
                    "data_quality_json": signal.get("data_quality"),
                    "within_normal_limits": signal.get("within_normal_limits"),
                },
            )
            g.add_edge(tel_id, sig_id, "monitors", {})

            for idx, anomaly in enumerate(signal.get("anomalies") or []):
                an_id = _prefix(f"{signal.get('sensor_id')}:{idx}:{anomaly.get('start')}", "ANOM")
                g.add_node(an_id, "anomaly", {**anomaly, "sensor_id": signal.get("sensor_id"), "parameter": signal.get("parameter")})
                g.add_edge(sig_id, an_id, "detects", {"source": "telemetry_summary"})
                if event_id:
                    g.add_edge(an_id, event_id, "causes", {"evidence_type": "telemetry_symptom"})

            for idx, cp in enumerate(signal.get("changepoints") or []):
                cp_id = _prefix(f"{signal.get('sensor_id')}:{idx}:{cp.get('timestamp')}", "CP")
                g.add_node(cp_id, "ChangePoint", {**cp, "sensor_id": signal.get("sensor_id"), "parameter": signal.get("parameter")})
                g.add_edge(sig_id, cp_id, "detects_changepoint", {})

    if operational_context:
        ctx_id = _prefix(f"{operational_context.get('asset_id')}:{operational_context.get('window', {}).get('start')}", "OPCTX")
        g.add_node(ctx_id, "OperationalContext", {**operational_context, "window_json": operational_context.get("window")})
        asset_id = _prefix(operational_context.get("asset_id"), "ASSET")
        if asset_id:
            g.add_node(asset_id, labels["asset"], {"source_key": operational_context.get("asset_id"), "kind": "asset"})
            g.add_edge(ctx_id, asset_id, "context_for_asset", {})
        if event_id:
            g.add_edge(ctx_id, event_id, "context_for_event", {})
        for alarm in operational_context.get("recent_alarms") or []:
            alarm_id = _prefix(alarm.get("alarm_id"), "ALARM")
            g.add_node(alarm_id, "Alarm", alarm)
            g.add_edge(ctx_id, alarm_id, "includes_alarm", {})
            if event_id:
                g.add_edge(alarm_id, event_id, "causes", {"evidence_type": "alarm_context"})
        for wo in operational_context.get("nearby_maintenance") or []:
            wo_id = _prefix(wo.get("wo_id"), "WO")
            g.add_node(wo_id, labels["work_order"], {**wo, "ID": wo.get("wo_id")})
            g.add_edge(ctx_id, wo_id, "includes_maintenance", {"proximity": wo.get("proximity")})

    if pm_compliance:
        pm_id = _prefix(f"{pm_compliance.get('asset_id')}:{pm_compliance.get('window', {}).get('start')}", "PM")
        g.add_node(pm_id, "PMCompliance", {**pm_compliance, "window_json": pm_compliance.get("window"), "summary_json": pm_compliance.get("summary")})
        asset_id = _prefix(pm_compliance.get("asset_id"), "ASSET")
        if asset_id:
            g.add_node(asset_id, labels["asset"], {"source_key": pm_compliance.get("asset_id"), "kind": "asset"})
            g.add_edge(pm_id, asset_id, "context_for_asset", {})
        for check in pm_compliance.get("checks") or []:
            check_id = _prefix(check.get("check_id"), "PMC")
            g.add_node(check_id, "PMCheck", check)
            g.add_edge(pm_id, check_id, "includes_check", {})
            if check.get("status") in {"failed", "overdue"} and event_id:
                g.add_edge(check_id, event_id, "causes", {"evidence_type": "pm_noncompliance"})

    if evidence_bundle:
        bundle_id = _prefix(evidence_bundle.get("bundle_id"), "BUNDLE")
        g.add_node(
            bundle_id,
            "EvidenceBundle",
            {
                **evidence_bundle,
                "retrieval_scope_json": evidence_bundle.get("retrieval_scope"),
                "filters_json": evidence_bundle.get("filters"),
            },
        )
        if event_id:
            g.add_edge(bundle_id, event_id, "retrieved_for_event", {})
        for result in evidence_bundle.get("results") or []:
            snippet_id = _prefix(result.get("snippet_id"), "SNIP")
            g.add_node(snippet_id, "EvidenceSnippet", {**result, "metadata_json": result.get("metadata")})
            g.add_edge(bundle_id, snippet_id, "contains_evidence", {"score": result.get("score")})
            doc_id = _prefix(result.get("doc_id"), "DOC")
            if doc_id:
                g.add_node(doc_id, labels["document"], {"doc_key": result.get("doc_id")})
                g.add_edge(snippet_id, doc_id, "derived_from_document", {})

    if causality_candidates:
        if not event_id:
            event_id = _prefix(causality_candidates.get("event_id"), "EVT")
            if event_id:
                g.add_node(event_id, labels["event"], {"event_key": causality_candidates.get("event_id")})
        for candidate in causality_candidates.get("candidates") or []:
            cand_id = _prefix(candidate.get("candidate_id"), "CAND")
            g.add_node(
                cand_id,
                labels["causal_factor"],
                {
                    **candidate,
                    "candidate_key": candidate.get("candidate_id"),
                    "scores_json": candidate.get("scores"),
                    "kg_path_json": candidate.get("kg_path"),
                    "components_involved_json": candidate.get("components_involved"),
                    "supporting_evidence_refs_json": candidate.get("supporting_evidence_refs"),
                },
            )
            if event_id:
                g.add_edge(cand_id, event_id, "causes", {"composite_score": candidate.get("composite_score")})
            cause_node_id = candidate.get("cause_node_id")
            if cause_node_id:
                # Keep original key; if it already carries a namespace do not double-prefix.
                if ":" in cause_node_id:
                    cause_id = cause_node_id
                else:
                    cause_id = _prefix(cause_node_id, "CAUSE")
                g.add_node(cause_id, labels["component"], {"source_key": candidate.get("cause_node_id"), "label": candidate.get("cause_label")})
                g.add_edge(cand_id, cause_id, "supported_by_path", {})
            fm_id = _prefix(candidate.get("cause_fm_id"), "FM")
            if fm_id:
                g.add_node(fm_id, labels["failure_mode"], {"source_key": candidate.get("cause_fm_id"), "label": candidate.get("cause_label")})
                g.add_edge(cand_id, fm_id, "caused_by", {})
            for ref in candidate.get("supporting_evidence_refs") or []:
                ref_id = _prefix(ref, "REF")
                g.add_node(ref_id, "EvidenceReference", {"source_key": ref})
                g.add_edge(cand_id, ref_id, "supported_by_evidence", {})

    if rca_card:
        rca_id = _prefix(rca_card.get("rca_id"), "RCA")
        g.add_node(
            rca_id,
            labels["rca_case"],
            {
                **rca_card,
                "rca_key": rca_card.get("rca_id"),
                "input_artifacts_json": rca_card.get("input_artifacts"),
                "validation_status_json": rca_card.get("validation_status"),
                "provenance_json": rca_card.get("provenance"),
                "human_review_json": rca_card.get("human_review"),
            },
        )
        if event_id:
            g.add_edge(rca_id, event_id, "investigates", {})

        primary = rca_card.get("primary_hypothesis") or {}
        primary_id = _prefix(primary.get("candidate_id") or f"{rca_card.get('rca_id')}:primary", "CAND")
        g.add_node(primary_id, labels["causal_factor"], {**primary, "candidate_key": primary.get("candidate_id")})
        g.add_edge(rca_id, primary_id, "identifies_causal_factor", {"is_primary": True})

        for alt in rca_card.get("alternatives") or []:
            alt_id = _prefix(alt.get("candidate_id"), "CAND")
            g.add_node(alt_id, labels["causal_factor"], {**alt, "candidate_key": alt.get("candidate_id")})
            g.add_edge(rca_id, alt_id, "identifies_causal_factor", {"is_primary": False})

        for ev in rca_card.get("evidence") or []:
            ev_id = _prefix(ev.get("evidence_id"), "EVREF")
            g.add_node(ev_id, "EvidenceReference", ev)
            g.add_edge(rca_id, ev_id, "uses_evidence", {"source_type": ev.get("source_type")})
            if _truthy(ev.get("doc_id")):
                doc_id = _prefix(ev.get("doc_id"), "DOC")
                g.add_node(doc_id, labels["document"], {"doc_key": ev.get("doc_id")})
                g.add_edge(ev_id, doc_id, "derived_from_document", {})

        for action in rca_card.get("recommended_actions") or []:
            act_id = _prefix(action.get("action_id"), "ACT")
            g.add_node(act_id, "corrective_action", action)
            g.add_edge(rca_id, act_id, "recommends_action", {"priority": action.get("priority")})
            target_component_id = _prefix(action.get("target_component_id"), "CMP")
            if target_component_id:
                g.add_node(target_component_id, labels["component"], {"source_key": action.get("target_component_id"), "kind": "component"})
                rel = "targets_entity" if "targets_entity" in g.relation_map else "mentions"
                g.add_edge(act_id, target_component_id, rel, {})
            linked_candidate_id = _prefix(action.get("linked_candidate_id"), "CAND")
            if linked_candidate_id:
                g.add_node(linked_candidate_id, labels["causal_factor"], {"candidate_key": action.get("linked_candidate_id")})
                g.add_edge(act_id, linked_candidate_id, "addresses_causal_factor", {})

    return g.as_lists()


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest_graph_toml(
    client: Py2Neo,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    database: Optional[str] = None,
) -> None:
    """Bulk-upsert a node list and edge list into Neo4j.

    A thin wrapper that calls :meth:`Py2Neo.upsert_nodes_batch` followed by
    :meth:`Py2Neo.upsert_edges_batch`, skipping each call when the
    corresponding list is empty.

    Args:
        client: Active :class:`Py2Neo` connection.
        nodes: List of node dicts as returned by :func:`build_graph_from_workflow_artifacts`.
        edges: List of edge dicts as returned by :func:`build_graph_from_workflow_artifacts`.
        database: Target database name; uses the driver default when ``None``.
    """
    if nodes:
        client.upsert_nodes_batch(nodes, db=database)
    if edges:
        client.upsert_edges_batch(edges, db=database)
