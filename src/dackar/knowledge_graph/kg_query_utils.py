from __future__ import annotations

from typing import Any, Optional


def is_upstream(
    component_a: Optional[str],
    component_b: Optional[str],
    neo4j_client: Optional[Any],
    database: Optional[str] = None,
) -> bool:
    """Return True if *component_a* is upstream of *component_b* in the graph.

    Walks up to six containment/connectivity hops (``has_part_usage``,
    ``owns_port_usage``, ``connects_port``) from ``component_a`` toward
    ``component_b`` over ``element_usage`` nodes.

    Args:
        component_a: Source component id (``element_usage.id``). A falsy value,
            or ``component_a == component_b``, yields ``False``.
        component_b: Target component id (``element_usage.id``).
        neo4j_client: Object exposing ``query(cypher, params, db=...)``; when
            ``None`` the function returns ``False`` without querying.
        database: Optional Neo4j database name; ``None`` uses the client default.

    Returns:
        ``True`` if a directed path ``a -> ... -> b`` exists, otherwise
        ``False``.  Any query error is swallowed and reported as ``False``.
    """
    if not component_a or not component_b or component_a == component_b or neo4j_client is None:
        return False
    query = """
    MATCH path = (a:element_usage {id: $cid_a})
                 -[:has_part_usage|owns_port_usage|connects_port*1..6]->
                 (b:element_usage {id: $cid_b})
    RETURN count(path) > 0 AS reachable
    LIMIT 1
    """
    try:
        rows = [
            dict(r)
            for r in neo4j_client.query(
                query,
                {"cid_a": component_a, "cid_b": component_b},
                db=database,
            )
        ]
        return bool(rows and rows[0].get("reachable"))
    except Exception:
        return False


def resolve_edge_type(
    component_a: Optional[str],
    component_b: Optional[str],
    neo4j_client: Optional[Any],
    database: Optional[str] = None,
) -> str:
    """Classify the relationship path between two components.

    Inspects the relationship types along the (up to six-hop) path from
    ``component_a`` to ``component_b`` and buckets them into a single category.

    Args:
        component_a: Source component id (``element_usage.id``). A falsy value
            yields ``"mixed"``.
        component_b: Target component id (``element_usage.id``).
        neo4j_client: Object exposing ``query(cypher, params, db=...)``; when
            ``None`` the function returns ``"mixed"`` without querying.
        database: Optional Neo4j database name; ``None`` uses the client default.

    Returns:
        ``"containment"`` if the path uses only ``has_part_usage`` edges,
        ``"connectivity"`` if it uses only ``owns_port_usage`` /
        ``connects_port`` edges, otherwise ``"mixed"`` (also returned when no
        path is found or a query error occurs).
    """
    if not component_a or not component_b or neo4j_client is None:
        return "mixed"
    query = """
    MATCH path = (a:element_usage {id: $cid_a})
                 -[:has_part_usage|owns_port_usage|connects_port*1..6]->
                 (b:element_usage {id: $cid_b})
    UNWIND relationships(path) AS rel
    RETURN collect(DISTINCT type(rel)) AS rel_types
    LIMIT 1
    """
    try:
        rows = [
            dict(r)
            for r in neo4j_client.query(
                query,
                {"cid_a": component_a, "cid_b": component_b},
                db=database,
            )
        ]
        rel_types = set(rows[0].get("rel_types") or []) if rows else set()
    except Exception:
        rel_types = set()
    if not rel_types:
        return "mixed"
    contains_containment = "has_part_usage" in rel_types
    contains_connectivity = bool({"owns_port_usage", "connects_port"} & rel_types)
    if contains_containment and not contains_connectivity:
        return "containment"
    if contains_connectivity and not contains_containment:
        return "connectivity"
    return "mixed"
