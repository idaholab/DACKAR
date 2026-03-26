from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from neo4j import GraphDatabase

LOGGER = logging.getLogger(__name__)
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_token(value: str, kind: str) -> str:
    if not isinstance(value, str) or not _SAFE_TOKEN_RE.match(value):
        raise ValueError(f"Invalid Neo4j {kind}: {value!r}")
    return value


class Py2Neo:
    """Thin Neo4j wrapper tailored for schema-governed KG ingestion."""

    def __init__(self, uri: str, user: str, pwd: str):
        self._uri = uri
        self._user = user
        self._pwd = pwd
        self._driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def restart(self) -> None:
        self.close()
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._pwd))

    def query(self, query: str, parameters: Optional[Dict[str, Any]] = None, db: Optional[str] = None):
        with self._driver.session(database=db) if db else self._driver.session() as session:
            return list(session.run(query, parameters or {}))

    def write(self, query: str, parameters: Optional[Dict[str, Any]] = None, db: Optional[str] = None) -> None:
        with self._driver.session(database=db) if db else self._driver.session() as session:
            session.execute_write(lambda tx: tx.run(query, parameters or {}))

    def reset(self, db: Optional[str] = None) -> None:
        self.write("MATCH (n) DETACH DELETE n", db=db)

    def create_node(self, label: str, properties: Dict[str, Any], db: Optional[str] = None) -> None:
        if "id" not in properties:
            raise ValueError("Node properties must include 'id'")
        label = _safe_token(label, "label")
        query = f"MERGE (n:`{label}` {{id: $id}}) SET n += $props"
        self.write(query, {"id": properties["id"], "props": properties}, db=db)

    def create_relation(
        self,
        source_label: str,
        source_key: Dict[str, Any],
        target_label: str,
        target_key: Dict[str, Any],
        rel_type: str,
        rel_props: Optional[Dict[str, Any]] = None,
        db: Optional[str] = None,
    ) -> None:
        source_label = _safe_token(source_label, "label")
        target_label = _safe_token(target_label, "label")
        rel_type = _safe_token(rel_type, "relationship type")
        rel_props = rel_props or {}

        def _match(alias: str, props: Dict[str, Any], prefix: str) -> tuple[str, Dict[str, Any]]:
            clauses = []
            params = {}
            for key, value in props.items():
                pname = f"{prefix}_{key}"
                clauses.append(f"{key}: ${pname}")
                params[pname] = value
            return ", ".join(clauses), params

        src_match, src_params = _match("a", source_key, "src")
        dst_match, dst_params = _match("b", target_key, "dst")
        query = (
            f"MATCH (a:`{source_label}` {{{src_match}}}) "
            f"MATCH (b:`{target_label}` {{{dst_match}}}) "
            f"MERGE (a)-[r:`{rel_type}`]->(b) "
            f"SET r += $rel_props"
        )
        params = {**src_params, **dst_params, "rel_props": rel_props}
        self.write(query, params, db=db)

    def upsert_nodes_batch(self, nodes: Sequence[Dict[str, Any]], db: Optional[str] = None) -> None:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for node in nodes:
            label = _safe_token(node["label"], "label")
            grouped.setdefault(label, []).append(node["attrs"])

        for label, rows in grouped.items():
            query = (
                f"UNWIND $rows AS row "
                f"MERGE (n:`{label}` {{id: row.id}}) "
                f"SET n += row"
            )
            self.write(query, {"rows": rows}, db=db)

    def upsert_edges_batch(self, edges: Sequence[Dict[str, Any]], db: Optional[str] = None) -> None:
        grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = {}
        for edge in edges:
            key = (
                _safe_token(edge["from_label"], "label"),
                _safe_token(edge["to_label"], "label"),
                _safe_token(edge["type"], "relationship type"),
            )
            grouped.setdefault(key, []).append(edge)

        for (src_label, dst_label, rel_type), rows in grouped.items():
            query = (
                "UNWIND $rows AS row "
                f"MATCH (a:`{src_label}` {{id: row.from_id}}) "
                f"MATCH (b:`{dst_label}` {{id: row.to_id}}) "
                f"MERGE (a)-[r:`{rel_type}`]->(b) "
                "SET r += row.attrs"
            )
            payload = [
                {"from_id": r["from"], "to_id": r["to"], "attrs": r.get("attrs", {})}
                for r in rows
            ]
            self.write(query, {"rows": payload}, db=db)

    def get_all(self, db: Optional[str] = None):
        return self.query("MATCH (n) RETURN n", db=db)
