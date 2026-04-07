"""
kg_ingest_fmea_workflow.py
─────────────────────────────────────────────────────────────────────────────
Ingest parsed FMEA records (output of fmeaParser.parse_fmea_file) into Neo4j.

Graph objects created per FMEA record
──────────────────────────────────────
Nodes:
  fmea_case         one per unique (fmea_source_ref, sheet) combination
  failure_mode      one per record (keyed by failure_mode_id; merged if duplicate)
  risk_assessment   one per record when at least one of severity/occurrence/detection
                    is present
  effect            one per record when local_effect text is present

Edges:
  fmea_case    -[:IDENTIFIES_FAILURE_MODE]->  failure_mode
  failure_mode -[:HAS_RISK_ASSESSMENT]->      risk_assessment
  failure_mode -[:LEADS_TO_EFFECT]->          effect
  failure_mode -[:APPLIES_TO]->               mbse_entity  (see below)

Component-type resolution (APPLIES_TO edges)
─────────────────────────────────────────────
FMEA data is class-level: a row for "centrifugal_pump / seal degradation"
applies to *every* centrifugal pump in the plant.  During ingestion the
``component_type`` value is resolved to individual mbse_entity node IDs by
querying the live KG:

  MATCH (c:mbse_entity)
  WHERE toLower(c.component_type) = toLower($component_type)
  RETURN c.id AS component_id

An ``APPLIES_TO`` edge is created for each matched component.  If no components
are found for a type the failure_mode node is still written (with the
``component_type`` property set) and a warning is logged so the gap can be
addressed when MBSE entities are loaded.

CLI usage
─────────
  python kg_ingest_fmea_workflow.py \\
      --schema ../../knowledge_graph/schemas/fmeaSchema.toml \\
      --schema ../../knowledge_graph/schemas/customMbseSchema.toml \\
      --neo4j-uri bolt://localhost:7687 \\
      --neo4j-user neo4j --neo4j-pass secret \\
      fmea_pump.xlsx fmea_valve.xlsx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

# ---------------------------------------------------------------------------
# Resolve imports whether run as a script or imported as a module.
# ---------------------------------------------------------------------------
try:
    from kg_schema_builder_workflow import (
        GraphBatch,
        apply_schema_constraints,
        ingest_graph_toml,
        load_and_merge_schemas,
    )
    from py2neo_workflow import Py2Neo
except ModuleNotFoundError:
    # Installed / pytest path
    from dackar.RCA.kg.kg_schema_builder_workflow import (  # type: ignore
        GraphBatch,
        apply_schema_constraints,
        ingest_graph_toml,
        load_and_merge_schemas,
    )
    from dackar.RCA.kg.py2neo_workflow import Py2Neo  # type: ignore

try:
    from doc_parsers.fmeaParser import parse_fmea_files
except ModuleNotFoundError:
    from dackar.RCA.doc_parsers.fmeaParser import parse_fmea_files  # type: ignore

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    LOGGER.addHandler(_ch)
LOGGER.setLevel(logging.INFO)

JsonLike = Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Component-type → mbse_entity ID resolver
# ---------------------------------------------------------------------------

def _resolve_component_type(
    client: Py2Neo,
    component_type: str,
    database: Optional[str],
    cache: Dict[str, List[str]],
) -> List[str]:
    """Return all mbse_entity IDs whose ``component_type`` matches *component_type*.

    Results are cached per component_type string to avoid redundant queries.

    Args:
        client: Active Neo4j connection.
        component_type: Equipment class string (e.g. ``"centrifugal_pump"``).
        database: Neo4j target database; ``None`` uses the driver default.
        cache: Mutable dict used as an in-process cache across calls.

    Returns:
        List of mbse_entity ``id`` values (may be empty).
    """
    key = component_type.lower().strip()
    if key in cache:
        return cache[key]

    try:
        rows = [dict(r) for r in client.query(
            "MATCH (c:mbse_entity) "
            "WHERE toLower(c.component_type) = toLower($ct) "
            "RETURN c.id AS component_id",
            {"ct": key},
            db=database,
        )]
        ids = [r["component_id"] for r in rows if r.get("component_id")]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("KG query for component_type '%s' failed: %s", component_type, exc)
        ids = []

    if not ids:
        LOGGER.warning(
            "No mbse_entity nodes found for component_type '%s'. "
            "APPLIES_TO edges will be omitted; ensure MBSE entities are loaded first.",
            component_type,
        )

    cache[key] = ids
    return ids


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_fmea_graph(
    schema_paths: Union[str, Path, Iterable[Union[str, Path]]],
    fmea_records: Sequence[Dict[str, Any]],
    client: Optional[Py2Neo] = None,
    database: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build a Neo4j graph batch from parsed FMEA records.

    Creates ``fmea_case``, ``failure_mode``, ``risk_assessment``, and
    ``effect`` nodes with their connecting edges.  When *client* is supplied,
    ``APPLIES_TO`` edges to ``mbse_entity`` nodes are also created after
    resolving ``component_type`` against the live KG.

    Args:
        schema_paths: One or more paths to TOML schema files.
        fmea_records: Output of :func:`fmeaParser.parse_fmea_file`.
        client: Optional live :class:`Py2Neo` connection used to resolve
            component types to mbse_entity IDs.  When ``None``, APPLIES_TO
            edges are omitted.
        database: Neo4j target database; ``None`` uses the driver default.

    Returns:
        A two-tuple ``(nodes, edges)`` suitable for
        :func:`kg_schema_builder_workflow.ingest_graph_toml`.
    """
    schema = load_and_merge_schemas(schema_paths)
    g = GraphBatch(schema)

    # Cache component_type → [mbse_entity_id, …] to avoid per-row queries.
    component_type_cache: Dict[str, List[str]] = {}

    # Track which fmea_case and failure_mode nodes are already in the batch.
    seen_cases: Set[str] = set()
    seen_fms: Set[str] = set()

    for rec in fmea_records:
        source_ref = rec.get("fmea_source_ref") or "unknown_fmea"
        sheet = rec.get("_sheet") or ""
        component_type = rec.get("component_type") or ""
        fm_id = rec.get("failure_mode_id") or ""
        fm_name = rec.get("failure_mode_name") or ""

        if not fm_id or not fm_name:
            LOGGER.debug("Skipping record with missing failure_mode_id / name: %s", rec)
            continue

        # ── fmea_case node ────────────────────────────────────────────────
        case_key = f"{source_ref}::{sheet}" if sheet else source_ref
        case_id = f"FMEA_CASE:{case_key}"
        if case_id not in seen_cases:
            g.add_node(
                case_id,
                "fmea_case",
                {
                    "title": case_key,
                    "scope": component_type or None,
                    "status": "approved",
                    "fmea_source_ref": source_ref,
                    "sheet": sheet or None,
                },
            )
            seen_cases.add(case_id)

        # ── failure_mode node ─────────────────────────────────────────────
        is_new_fm = fm_id not in seen_fms
        g.add_node(
            fm_id,
            "failure_mode",
            {
                "fm_id": fm_id,
                "name": fm_name,
                "description": rec.get("local_effect") or fm_name,
                "failure_mechanism": rec.get("failure_mechanism") or None,
                "component_type": component_type or None,
                "expected_symptoms": "; ".join(rec.get("expected_symptoms") or []) or None,
                "expected_anomaly_pattern": rec.get("expected_anomaly_pattern") or None,
                "expected_latency_min_hours": rec.get("expected_latency_min_hours"),
                "expected_latency_max_hours": rec.get("expected_latency_max_hours"),
                "fmea_source_ref": source_ref,
            },
        )
        if is_new_fm:
            seen_fms.add(fm_id)

        # ── fmea_case → failure_mode ──────────────────────────────────────
        g.add_edge(case_id, fm_id, "identifies_failure_mode", allow_untyped=True)

        # ── risk_assessment node ──────────────────────────────────────────
        sev = rec.get("severity")
        occ = rec.get("occurrence")
        det = rec.get("detection")
        rpn = rec.get("rpn")
        if any(x is not None for x in (sev, occ, det, rpn)):
            ra_id = f"RA:{fm_id}"
            g.add_node(
                ra_id,
                "risk_assessment",
                {
                    "severity": sev,
                    "occurrence": occ,
                    "detection": det,
                    "RPN": rpn,
                    "notes": rec.get("notes") or None,
                },
            )
            g.add_edge(fm_id, ra_id, "has_risk_assessment", allow_untyped=True)

        # ── effect node ───────────────────────────────────────────────────
        local_effect = rec.get("local_effect")
        if local_effect:
            effect_id = f"EFF:{fm_id}"
            g.add_node(
                effect_id,
                "effect",
                {
                    "level": "local",
                    "description": local_effect,
                },
            )
            g.add_edge(fm_id, effect_id, "leads_to_effect", allow_untyped=True)

        # ── APPLIES_TO → mbse_entity nodes ────────────────────────────────
        if client is not None and component_type:
            mbse_ids = _resolve_component_type(
                client, component_type, database, component_type_cache
            )
            for mbse_id in mbse_ids:
                # mbse_entity nodes may already be in the KG; add stub here
                # so GraphBatch can track the edge endpoint.  MERGE semantics
                # in ingest_graph_toml will not overwrite existing properties.
                if mbse_id not in g.nodes:
                    g.add_node(mbse_id, "mbse_entity", {"id": mbse_id})
                try:
                    g.add_edge(fm_id, mbse_id, "applies_to", allow_untyped=False)
                except ValueError as exc:
                    # Schema endpoint mismatch — log and continue.
                    LOGGER.warning("applies_to edge skipped for %s → %s: %s", fm_id, mbse_id, exc)

    return g.as_lists()


# ---------------------------------------------------------------------------
# Top-level ingestion function
# ---------------------------------------------------------------------------

def ingest_fmea_to_neo4j(
    client: Py2Neo,
    schema_paths: Union[str, Path, Iterable[Union[str, Path]]],
    fmea_records: Sequence[Dict[str, Any]],
    *,
    database: Optional[str] = None,
    create_constraints: bool = True,
) -> Tuple[int, int]:
    """Build and ingest a FMEA graph into Neo4j.

    Args:
        client: Active :class:`Py2Neo` connection.
        schema_paths: One or more TOML schema file paths.
        fmea_records: Parsed FMEA records from :func:`fmeaParser.parse_fmea_file`.
        database: Target Neo4j database; ``None`` uses the driver default.
        create_constraints: Apply DDL constraints/indexes before ingestion.

    Returns:
        ``(node_count, edge_count)`` written to the database.
    """
    if create_constraints:
        apply_schema_constraints(client, schema_paths, database=database)

    nodes, edges = build_fmea_graph(
        schema_paths,
        fmea_records,
        client=client,
        database=database,
    )
    LOGGER.info("Built FMEA graph: %d nodes, %d edges", len(nodes), len(edges))
    ingest_graph_toml(client, nodes, edges, database=database)
    return len(nodes), len(edges)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse FMEA spreadsheet(s) and ingest into Neo4j",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "fmea_files",
        nargs="+",
        metavar="FMEA_FILE",
        help="Path(s) to .csv, .xlsx, or .xls FMEA spreadsheet(s)",
    )
    parser.add_argument(
        "--schema",
        action="append",
        required=True,
        dest="schemas",
        metavar="SCHEMA_TOML",
        help="Path to TOML schema file (repeat for multiple schemas)",
    )
    parser.add_argument("--neo4j-uri", required=True, help="Neo4j bolt URI")
    parser.add_argument("--neo4j-user", required=True, help="Neo4j username")
    parser.add_argument("--neo4j-pass", required=True, help="Neo4j password")
    parser.add_argument("--database", default=None, help="Neo4j database name")
    parser.add_argument(
        "--no-constraints",
        action="store_true",
        help="Skip DDL constraint/index creation",
    )
    parser.add_argument(
        "--sheet",
        action="append",
        dest="sheet_filter",
        metavar="SHEET_NAME",
        help="Only parse the named Excel sheet(s) (repeat for multiple; default: all)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    client = Py2Neo(args.neo4j_uri, args.neo4j_user, args.neo4j_pass)
    try:
        records = parse_fmea_files(
            [Path(p) for p in args.fmea_files],
            sheet_filter=args.sheet_filter or None,
        )
        LOGGER.info("Parsed %d total FMEA records from %d file(s).", len(records), len(args.fmea_files))
        if not records:
            LOGGER.warning("No FMEA records parsed — nothing to ingest.")
            sys.exit(0)

        nodes, edges = ingest_fmea_to_neo4j(
            client,
            schema_paths=args.schemas,
            fmea_records=records,
            database=args.database,
            create_constraints=not args.no_constraints,
        )
        LOGGER.info("Ingestion complete: %d nodes, %d edges", nodes, edges)
    finally:
        client.close()


if __name__ == "__main__":
    main()
