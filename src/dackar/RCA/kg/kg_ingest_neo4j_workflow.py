from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from py2neo_workflow import Py2Neo
from kg_schema_builder_workflow import (
    apply_schema_constraints,
    build_graph_from_workflow_artifacts,
    ingest_graph_toml,
)

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)


JsonLike = Optional[Dict[str, Any]]


def load_json(path: Optional[Union[str, Path]]) -> JsonLike:
    """Load a JSON file and return its contents as a dict, or ``None`` if no path given.

    Args:
        path: Filesystem path to a ``.json`` file, or ``None`` / empty string.

    Returns:
        Parsed JSON object as a dict, or ``None`` when *path* is falsy.
    """
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_list(paths: Optional[Sequence[Union[str, Path]]]) -> List[Dict[str, Any]]:
    """Load a sequence of JSON files, returning only non-empty dict results.

    Silently skips paths whose file parses to something other than a non-empty
    dict (e.g. a JSON array or an empty object).

    Args:
        paths: Sequence of filesystem paths, or ``None``.

    Returns:
        List of parsed JSON dicts, one per successfully loaded file.
    """
    out: List[Dict[str, Any]] = []
    for path in paths or []:
        obj = load_json(path)
        if isinstance(obj, dict) and obj:
            out.append(obj)
    return out

def _looks_like_processed_text_record(obj: Dict[str, Any]) -> bool:
    """Return True if *obj* has the minimum required fields of a processed_text_record.

    Validates that ``record_id``, ``doc_id``, and ``doc_type`` are strings and
    that ``metadata`` and ``provenance`` are dicts.  Does not perform full
    JSON Schema validation.

    Args:
        obj: Candidate dict to inspect.

    Returns:
        ``True`` if the dict structurally resembles a valid processed_text_record.
    """
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("record_id"), str)
        and isinstance(obj.get("doc_id"), str)
        and isinstance(obj.get("doc_type"), str)
        and isinstance(obj.get("metadata"), dict)
        and isinstance(obj.get("provenance"), dict)
    )

def _partition_processed_text_records(records: Optional[Sequence[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a sequence of records into valid and malformed processed_text_records.

    Args:
        records: Sequence of candidate record dicts, or ``None``.

    Returns:
        A two-tuple ``(good, bad)`` where *good* contains records that pass
        the :func:`_looks_like_processed_text_record` check and *bad* contains
        those that do not.
    """
    good: List[Dict[str, Any]] = []
    bad: List[Dict[str, Any]] = []
    for rec in records or []:
        if _looks_like_processed_text_record(rec):
            good.append(rec)
        else:
            bad.append(rec)
    return good, bad

def ingest_workflow_case_to_neo4j(
    client: Py2Neo,
    schema_paths: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    event: JsonLike = None,
    kg_context: JsonLike = None,
    telemetry_summary: JsonLike = None,
    evidence_bundle: JsonLike = None,
    causality_candidates: JsonLike = None,
    rca_card: JsonLike = None,
    operational_context: JsonLike = None,
    pm_compliance: JsonLike = None,
    documents: Optional[Sequence[Dict[str, Any]]] = None,
    processed_text_records: Optional[Sequence[Dict[str, Any]]] = None,
    database: Optional[str] = None,
    create_constraints: bool = True,
) -> Tuple[int, int]:
    """Build and ingest a full RCA workflow case graph into Neo4j.

    Orchestrates schema constraint application, graph construction from all
    supplied artifacts, and bulk ingestion.  Malformed processed_text_record
    entries are filtered out with a warning before the graph is built.

    Args:
        client: Active :class:`Py2Neo` connection.
        schema_paths: One or more paths to TOML schema files.
        event: Parsed ``event`` artifact dict.
        kg_context: Parsed ``kg_context`` artifact dict.
        telemetry_summary: Parsed ``telemetry_summary`` artifact dict.
        evidence_bundle: Parsed ``evidence_bundle`` artifact dict.
        causality_candidates: Parsed ``causality_candidates`` artifact dict.
        rca_card: Parsed ``rca_card`` artifact dict.
        operational_context: Parsed ``operational_context`` artifact dict.
        pm_compliance: Parsed ``pm_compliance`` artifact dict.
        documents: List of document descriptor dicts.
        processed_text_records: List of ``processed_text_record`` dicts;
            malformed entries are skipped with a warning.
        database: Target Neo4j database name; uses the driver default when ``None``.
        create_constraints: When ``True`` (default), DDL constraints and indexes
            are applied before ingestion.

    Returns:
        A two-tuple ``(node_count, edge_count)`` reflecting the number of
        nodes and edges written to the database.
    """
    good_ptrs, bad_ptrs = _partition_processed_text_records(processed_text_records)
    if bad_ptrs:
        LOGGER.warning("Skipping %d malformed processed_text_record objects before graph build.", len(bad_ptrs))

    if create_constraints:
        apply_schema_constraints(client, schema_paths, database=database)

    nodes, edges = build_graph_from_workflow_artifacts(
        schema_paths,
        event=event,
        kg_context=kg_context,
        telemetry_summary=telemetry_summary,
        evidence_bundle=evidence_bundle,
        causality_candidates=causality_candidates,
        rca_card=rca_card,
        operational_context=operational_context,
        pm_compliance=pm_compliance,
        documents=documents,
        processed_text_records=good_ptrs,
    )
    LOGGER.info(
        "Built workflow graph with %d nodes and %d edges (processed_text_records accepted=%d rejected=%d)",
        len(nodes),
        len(edges),
        len(good_ptrs),
        len(bad_ptrs),
    )
    ingest_graph_toml(client, nodes, edges, database=database)
    return len(nodes), len(edges)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Neo4j ingestion script.

    Returns:
        Populated :class:`argparse.Namespace` with connection settings and
        optional paths to each artifact type.
    """
    parser = argparse.ArgumentParser(description="Ingest RCA workflow artifacts into Neo4j")
    parser.add_argument("--schema", action="append", required=True, dest="schemas", help="Path to TOML schema file")
    parser.add_argument("--neo4j-uri", required=True)
    parser.add_argument("--neo4j-user", required=True)
    parser.add_argument("--neo4j-pass", required=True)
    parser.add_argument("--database", default=None)
    parser.add_argument("--event")
    parser.add_argument("--kg-context")
    parser.add_argument("--telemetry-summary")
    parser.add_argument("--evidence-bundle")
    parser.add_argument("--candidates")
    parser.add_argument("--rca-card")
    parser.add_argument("--operational-context")
    parser.add_argument("--pm-compliance")
    parser.add_argument("--document", action="append", dest="documents")
    parser.add_argument("--processed-text-record", action="append", dest="processed_text_records")
    parser.add_argument("--no-constraints", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Entry point for the CLI ingestion script.

    Parses command-line arguments, opens a Neo4j connection, runs
    :func:`ingest_workflow_case_to_neo4j` with the provided artifact paths,
    and logs a completion summary.  The Neo4j driver is always closed on exit.
    """
    args = _parse_args()
    client = Py2Neo(args.neo4j_uri, args.neo4j_user, args.neo4j_pass)
    try:
        nodes, edges = ingest_workflow_case_to_neo4j(
            client,
            schema_paths=args.schemas,
            event=load_json(args.event),
            kg_context=load_json(args.kg_context),
            telemetry_summary=load_json(args.telemetry_summary),
            evidence_bundle=load_json(args.evidence_bundle),
            causality_candidates=load_json(args.candidates),
            rca_card=load_json(args.rca_card),
            operational_context=load_json(args.operational_context),
            pm_compliance=load_json(args.pm_compliance),
            documents=load_json_list(args.documents),
            processed_text_records=load_json_list(args.processed_text_records),
            database=args.database,
            create_constraints=not args.no_constraints,
        )
        LOGGER.info("Ingestion complete: %d nodes, %d edges", nodes, edges)
    finally:
        client.close()


if __name__ == "__main__":
    main()
