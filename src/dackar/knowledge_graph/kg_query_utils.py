from __future__ import annotations

from typing import Any, Optional


def is_upstream(
    component_a: Optional[str],
    component_b: Optional[str],
    neo4j_client: Optional[Any],
    database: Optional[str] = None,
) -> bool:
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
